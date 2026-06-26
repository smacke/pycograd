# -*- coding: utf-8 -*-
"""The reverse-mode tape node.

``Var`` wraps a numpy array; arithmetic operators are overloaded so that running a
program builds a computation graph, and ``Var.backward()`` walks it in reverse to
accumulate gradients. The convenience methods (``sum``/``mean``/``__pow__``/``T``/
...) delegate to the differentiable primitives in :mod:`pycograd.ops`; those
imports are deferred to function bodies so this module has no import-time
dependency on ``ops`` (which imports ``Var`` from here).

----------------------------------------------------------------------------------
Three reverse-mode mechanisms (and why they coexist)
----------------------------------------------------------------------------------
pycograd computes a gradient three different ways. They are *not* three copies of the
calculus -- the local derivatives live once (forward's ``_JVP_FOR`` + the shared
``ops._UNARY_DERIV`` / ``_pow_base_deriv`` / ``_gated_act_coeffs`` tables, which the
reverse rules also build from). They are three *execution strategies*, each the right
trade for a different job:

1. **Base eager tape** -- ``Var.backward()`` raw pass (this file). First-order ``grad``
   / ``value_and_grad`` with nothing enclosing them. Each op attaches a per-node
   ``_backward`` closure when it runs (e.g. ``d_tanh`` stores ``lambda a, g: g*(1-v*v)``
   reusing the cached ``v``); the reverse pass calls those closures, mutating numpy
   ``.grad`` directly. Raw numpy on cached values -- no dispatch, no tape allocation.
   *Why a separate strategy:* it is the hot path (all training), so it must be fast.

2. **Differentiable backward** -- ``_backward_differentiable`` (this file). Higher-order
   eager: ``grad(grad(f))``, ``jvp(grad(f))``, Hessians/HVPs -- i.e. a ``grad`` nested
   inside another ``grad``/``jvp`` (triggered when ``num_transform_levels() > 0``). It
   walks the same tape but accumulates each cotangent as a *bind-riding* ``Var``/tracer
   via ``ops._VJP_FOR``, so the cotangent graph is itself differentiable and the
   enclosing transform can differentiate the gradient.
   *Why not just reuse #1:* the base closures emit raw, non-differentiable arrays, so
   they cannot feed an outer transform. *Why not just reuse #2 for #1 too:* pointing the
   first-order pass at ``_VJP_FOR`` was benchmarked ~2x slower -- every cotangent op
   redispatches through the trace stack, allocates a throwaway tape node, and recomputes
   cached primals. So the split is a deliberate speed-vs-composability trade, not an
   oversight (see the ``backward`` docstring for the exact trigger).

3. **Graph mode** -- ``grad`` / ``value_and_grad`` of a captured graph (the
   ``ad_graph._grad_graph`` branch) / ``transpose.vjp_graph`` (opt-in via ``capture`` /
   ``jit``). Reverse mode on the capture IR, so forward *and* backward live in one graph
   the optimization passes can work across (e.g. CSE merging a recomputed ``sigmoid``).
   ``_grad_graph`` applies the VJP rules to a captured forward; ``vjp_graph``
   instead *derives* reverse from forward as ``transpose ∘ linearize`` (linearize reuses
   the JVP rules; transpose flips only the linear ops). Neither is on the eager hot path.
   *Why a third strategy:* eager reverse (#1/#2) is a tape, not a graph, so it cannot be
   optimized across the forward/backward boundary; the IR can.

The unifying idea: one set of derivatives, executed as a raw closure (#1, speed), a
bind-riding tracer (#2, composability), or graph nodes (#3, optimization).
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
from types import ModuleType
from typing import Any, Callable, Iterator, cast

import numpy as np

from pycograd._typing import Array, ArrayLike, Boxed, DTypeLike, Index, Operand, Prim
from pycograd.backends import current_backend
from pycograd.dtypes import conj_if_complex, current_dtype, resolve_dtype


# ---------------------------------------------------------------------------
# The array-module seam: data is computed with the *active* backend's array
# library (numpy by default, cupy under ``device("cupy")``), so the same tape and
# the same VJP rules run on whatever device that backend lives on.
# ---------------------------------------------------------------------------
def _xp() -> ModuleType:
    """The active backend's array module -- ``numpy`` unless a device is activated.

    A ``ModuleType`` chosen at runtime (numpy / cupy / ...); attribute access stays
    ``Any``, and the surface pycograd uses is the shared array-API subset both provide.
    """
    # An array backend (the only kind with a live Var tape) always sets array_module.
    return cast(ModuleType, current_backend().array_module)


# ---------------------------------------------------------------------------
# Per-example backward axes.
#
# During a *batched-cotangent* backward (``Var.backward(cotangent, keep_batch_axis=0)``,
# driven by ``vmap(grad(f))``), the reverse pass carries a leading per-example axis. The
# VJP closures all reduce a shared operand's gradient with ``_unbroadcast(g, shape)`` and
# do not know about this axis, so it is threaded out-of-band via this contextvar: when set
# to ``{0}``, ``_unbroadcast`` keeps a size-1 batch axis instead of summing it, so a
# shared parameter's gradient comes back ``(B, *param.shape)``. Empty by default, so every
# ordinary backward is byte-for-byte unchanged.
# ---------------------------------------------------------------------------
_KEEP_BATCH_AXES: contextvars.ContextVar[tuple[int, ...]] = contextvars.ContextVar(
    "pycograd_keep_batch_axes", default=()
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _unbroadcast(
    grad: Array, shape: tuple[int, ...], keep_axes: tuple[int, ...] = ()
) -> Array:
    """Sum ``grad`` over axes that were broadcast, so it matches ``shape``.

    ``keep_axes`` names size-1 axes of ``shape`` that should NOT be summed even though
    ``grad`` is larger there: the gradient keeps a leading *per-example* axis instead of
    collapsing it. This is how ``vmap(grad(f))`` returns a per-sample gradient of a
    *shared* parameter -- the parameter enters the batched forward with a size-1 batch
    axis (so it broadcasts over the batch), and on the way back that axis is kept,
    yielding shape ``(B, *param.shape)`` rather than the batch-summed ``param.shape``.

    The default (``keep_axes=()``) is exactly the historical behavior: every broadcast
    axis is summed away, so nothing else regresses.
    """
    # Preserve the cotangent's natural dtype -- it follows the forward computation's data
    # dtype (the working dtype is a creation default, not a propagation cast). In-place
    # accumulation into each ``.grad`` buffer keeps that buffer's dtype regardless.
    grad = _xp().asarray(grad)
    # Extra leading axes (rank grew under broadcasting) are summed unless kept: a kept
    # leading axis is reshaped into ``shape`` below, preserving the per-example stack.
    # An empty explicit ``keep_axes`` falls back to the backward-pass-global set (the
    # per-example axis of a batched-cotangent backward); both empty is the historical
    # path.
    keep = set(keep_axes) or set(_KEEP_BATCH_AXES.get())
    while grad.ndim > len(shape):
        if 0 in keep and grad.shape[0] != 1:
            break
        grad = grad.sum(axis=0)
    out_shape = list(shape)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            if axis in keep:
                out_shape[axis] = grad.shape[axis]  # keep the per-example axis
                continue
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(out_shape)


def _accumulate(acc: Array, delta: Array) -> Array:
    """Accumulate a cotangent contribution into a ``.grad`` buffer.

    In place when the shapes match -- the overwhelmingly common case -- so the reverse
    pass reuses each buffer instead of allocating a fresh sum array per contribution.
    Falls back to an out-of-place add when ``delta`` is *larger*: the ``vmap(grad)``
    per-sample path has ``_unbroadcast`` keep a leading per-example axis (``keep_axes``),
    so the contribution outgrows the unbatched accumulator and must broadcast-grow it.
    The right-hand side is always derived from a *child*'s ``.grad`` (never the
    accumulator itself), so the in-place add can never alias its own input.

    A complex cotangent flowing into a *real-valued* primal contributes only its real
    part: perturbing a real variable moves only along the real axis, so ``dL/dx`` is real.
    Dropping the imaginary part keeps a real node's gradient real (and lets the in-place
    add stay same-dtype)."""
    if acc.dtype.kind != "c" and np.iscomplexobj(delta):
        delta = delta.real
    if acc.shape == delta.shape:
        acc += delta
        return acc
    return acc + delta


def _logical_shape(x: Boxed) -> tuple[int, ...]:
    """The shape of a cotangent that may be a ``Var`` (``.value.shape``) or a level
    tracer (``.shape``)."""
    if isinstance(x, Var):
        return x.value.shape
    return tuple(getattr(x, "shape"))  # a Tracer exposes a logical shape


def _d_unbroadcast(grad: Boxed, shape: tuple[int, ...]) -> Boxed:
    """A *differentiable* ``_unbroadcast``: sum a cotangent ``Var`` over broadcast axes so
    it matches ``shape``, built from ``d_sum`` + ``d_reshape`` (which themselves ride
    ``bind``), so the cotangent graph stays differentiable.

    Mirrors the raw :func:`_unbroadcast` for the historical (base-level) path, but operates
    on the tape: extra leading axes are summed away, then each axis broadcast from size 1
    is summed with ``keepdims`` and the result reshaped to ``shape``. The per-example
    ``keep_axes`` logic of the raw variant is intentionally absent -- ``vmap``-composed
    higher-order AD is out of scope.
    """
    from pycograd import ops
    from pycograd.trace import bind

    g = grad
    gshape = _logical_shape(g)
    # Drop extra leading axes that broadcasting added (rank grew).
    while len(gshape) > len(shape):
        g = bind(ops.d_sum, g, axis=0)
        gshape = _logical_shape(g)
    sum_axes = tuple(
        axis for axis, size in enumerate(shape) if size == 1 and gshape[axis] != 1
    )
    if sum_axes:
        g = bind(ops.d_sum, g, axis=sum_axes, keepdims=True)
    if _logical_shape(g) != tuple(shape):
        g = bind(ops.d_reshape, g, tuple(shape))
    return g


def _unwrap_weight(x: Operand) -> Var | ArrayLike:
    """Resolve a ``Weight`` proxy to its current value; leave anything else."""
    # Deferred import: params imports Var/_lift/_unbroadcast from this module.
    from pycograd.params import Weight

    return x._live() if isinstance(x, Weight) else x


# Reverse-mode "grad in progress" depth. A reverse differentiation (``value_and_grad`` /
# ``grad`` / ``ParamDict.grad``) brackets the forward pass it records with
# ``grad_recording()``. ``vmap`` reads this: when it runs *inside* a live grad pass -- e.g.
# the objective of ``weights.grad`` calls ``vmap(forward)(X)`` with the weights captured
# ambiently rather than passed as mapped arguments -- ``vmap`` cannot see those weight
# ``Var``s among its args, so it would materialize its output to a plain array and sever
# the weight gradient. The flag tells it to keep results on the tape instead (the same
# behavior as nesting under an outer transform). It does *not* fire for ``vmap(grad(g))``,
# where each inner ``grad`` has already returned before ``vmap`` finishes.
# Scoped like every other ambient flag in the project (a ``ContextVar``, cf.
# ``_KEEP_BATCH_AXES`` above) so nested/concurrent grad passes don't clobber one global.
_GRAD_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "pycograd_grad_depth", default=0
)


@contextlib.contextmanager
def grad_recording() -> "Iterator[None]":
    """Mark a reverse-mode grad pass as in progress for its dynamic extent."""
    token = _GRAD_DEPTH.set(_GRAD_DEPTH.get() + 1)
    try:
        yield
    finally:
        _GRAD_DEPTH.reset(token)


def grad_is_recording() -> bool:
    """True while a reverse-mode grad pass is recording (see :func:`grad_recording`)."""
    return _GRAD_DEPTH.get() > 0


def _value(x: Operand) -> ArrayLike:
    x = _unwrap_weight(x)
    return x.value if isinstance(x, Var) else x


def _is_array(x: object) -> bool:
    """True for a numpy array or the active backend's array type (e.g. a cupy array)."""
    if isinstance(x, np.ndarray):
        return True
    ndarray = getattr(_xp(), "ndarray", None)
    return ndarray is not None and isinstance(x, ndarray)


def _is_numeric(x: object) -> bool:
    return (
        isinstance(x, (int, float, complex)) and not isinstance(x, bool)
    ) or _is_array(x)


# ---------------------------------------------------------------------------
# The reverse-mode tape node.
# ---------------------------------------------------------------------------
class Var:
    # Tell numpy to defer operators (``ndarray + Var``, ``ndarray @ Var``) to our
    # reflected methods, and to fail loudly on un-intercepted ufuncs.
    __array_ufunc__ = None

    def __init__(
        self,
        value: ArrayLike,
        _parents: tuple[Var, ...] = (),
        *,
        dtype: DTypeLike | None = None,
    ) -> None:
        xp = _xp()
        if dtype is not None:
            self.value: Array = xp.asarray(value, dtype=resolve_dtype(dtype))
        else:
            arr = xp.asarray(value)
            wd = current_dtype()
            if arr.dtype.kind == "c":
                # A complex value is preserved unconditionally -- casting it to a float
                # working dtype would silently drop the imaginary part. So complex tensors
                # flow through the tape whether or not a ``dtype("complex128")`` block is
                # active (the suite passes complex arrays straight to ``grad``).
                self.value = arr
            elif wd.kind == "c":
                # Complex working dtype: a complex computation mixes real- and complex-valued
                # tensors (``abs``/``real`` are real, the rest complex), so intermediates
                # follow numpy's *natural* promotion. Only a non-floating value (a python
                # scalar / int leaf) is lifted to the complex working dtype.
                self.value = (
                    arr if arr.dtype.kind == "f" else xp.asarray(value, dtype=wd)
                )
            elif hasattr(value, "dtype") and (
                arr.dtype.kind == "f" or arr.dtype.name == "bfloat16"
            ):
                # An *existing* float array keeps its own dtype: the data dtype flows through
                # the tape (so a float32 input yields a float32 gradient, like numpy/autograd).
                # The working dtype is a *default for new values* (the ``else`` below, plus
                # ``Param``/buffer creation in params.py), not a cast forced onto inputs.
                # (``bfloat16`` is the floating type we resolve "bf16" to, but reports kind "V".)
                self.value = arr
            else:
                # A raw python scalar/list, or a non-float (int/bool) array: no data dtype to
                # follow, so create it at the working dtype.
                self.value = xp.asarray(value, dtype=wd)
        self.grad: Array = xp.zeros_like(self.value)
        self._parents = _parents
        self._backward: Callable[[], None] = lambda: None
        # Optional differentiable-VJP record, set by primitives that want the higher-order
        # reverse path (``backward`` under a live higher trace level). ``_vjp_prim`` is the
        # producing primitive; ``_vjp_operands`` are the *primal* operand ``Var``s aligned
        # with ``_parents``; ``_vjp_params`` are its static keyword args. When present, the
        # differentiable backward looks up ``ops._VJP_FOR[_vjp_prim]`` to build each
        # operand's cotangent as a tape-connected ``Var`` (rather than mutating ``.grad``).
        self._vjp_prim: Prim | None = None
        self._vjp_operands: tuple[Var, ...] = ()
        self._vjp_params: dict[str, Any] = {}

    # -- construction helpers ------------------------------------------------
    def _unary(
        self,
        value: ArrayLike,
        grad_fn: Callable[[Array, Array], Array],
        prim: Prim | None = None,
    ) -> Var:
        out = Var(value, _parents=(self,))

        def _backward() -> None:
            # Accumulate in place (base/raw path) via ``_accumulate``: ``self.grad`` is
            # a private accumulator the driver pre-zeros for every topo node, summed in
            # reverse order before any parent reads it. The differentiable path
            # (``_backward_differentiable``) is separate and untouched.
            #
            # Hermitian-adjoint wrap (complex): a holomorphic op's grad_fn is C-linear in
            # the cotangent (returns ``g * f'(z)``), so ``conj(grad_fn(z, conj(g)))`` =
            # ``g * conj(f'(z))`` -- the adjoint under the real inner product. ``conj_if_
            # complex`` is the identity on real dtypes, so the real path is unchanged. The
            # non-holomorphic prims (conj/real/imag/angle) set ``_backward`` directly and
            # never reach this generic path.
            contrib = conj_if_complex(grad_fn(self.value, conj_if_complex(out.grad)))
            self.grad = _accumulate(self.grad, _unbroadcast(contrib, self.value.shape))

        out._backward = _backward
        if prim is not None:
            _record_vjp(out, prim, (self,))
        return out

    def _binary(
        self,
        other: Operand,
        value_fn: Callable[[Array, Array], Array],
        grad_fn: Callable[[Array, Array, Array], tuple[Array, Array]],
        prim: Prim | None = None,
    ) -> Var:
        other = _lift(other)
        out = Var(value_fn(self.value, other.value), _parents=(self, other))

        def _backward() -> None:
            # Hermitian-adjoint wrap for complex (see ``_unary``): conjugate the cotangent
            # in and each contribution out so a C-linear grad_fn yields ``g * conj(D)``.
            # Identity on real dtypes.
            cg = conj_if_complex(out.grad)
            ga, gb = grad_fn(self.value, other.value, cg)
            ga, gb = conj_if_complex(ga), conj_if_complex(gb)
            # Accumulate in place on the base path (see ``_unary``). ``self`` and
            # ``other`` may be the same node (``x + x``); accumulating twice still sums
            # correctly, and neither rhs aliases the accumulator.
            self.grad = _accumulate(self.grad, _unbroadcast(ga, self.value.shape))
            other.grad = _accumulate(other.grad, _unbroadcast(gb, other.value.shape))

        out._backward = _backward
        if prim is not None:
            _record_vjp(out, prim, (self, other))
        return out

    # -- arithmetic ----------------------------------------------------------
    # The operator dunders pass their differentiable primitive to ``_binary``/``_unary`` so
    # the higher-order reverse path (``backward`` under a live higher level) can look up the
    # matching ``_VJP_FOR`` rule. ``prim`` is read only on that path; the base path is
    # unaffected. ``ops`` is imported lazily (it imports ``Var`` from here).
    def __add__(self, o: Operand) -> Var:
        from pycograd import ops

        return self._binary(o, lambda a, b: a + b, lambda a, b, g: (g, g), ops.d_add)

    __radd__ = __add__  # addition is commutative

    def __mul__(self, o: Operand) -> Var:
        from pycograd import ops

        return self._binary(
            o, lambda a, b: a * b, lambda a, b, g: (g * b, g * a), ops.d_mul
        )

    __rmul__ = __mul__  # multiplication is commutative

    def __sub__(self, o: Operand) -> Var:
        from pycograd import ops

        return self._binary(o, lambda a, b: a - b, lambda a, b, g: (g, -g), ops.d_sub)

    def __rsub__(self, o: Operand) -> Var:
        # ``o - self``; record as the canonical ``d_sub(o, self)`` (operands in the order
        # ``_VJP_FOR[d_sub]`` expects: ``value = operands[0] - operands[1]``).
        from pycograd import ops

        return _lift(o)._binary(
            self, lambda a, b: a - b, lambda a, b, g: (g, -g), ops.d_sub
        )

    def __truediv__(self, o: Operand) -> Var:
        from pycograd import ops

        return self._binary(
            o, lambda a, b: a / b, lambda a, b, g: (g / b, -g * a / (b * b)), ops.d_div
        )

    def __rtruediv__(self, o: Operand) -> Var:
        # ``o / self``; record as the canonical ``d_div(o, self)`` so the operand order
        # matches ``_VJP_FOR[d_div]`` (``value = operands[0] / operands[1]``).
        from pycograd import ops

        return _lift(o)._binary(
            self,
            lambda a, b: a / b,
            lambda a, b, g: (g / b, -g * a / (b * b)),
            ops.d_div,
        )

    def __mod__(self, o: Operand) -> Var:
        # a % b == a - b*floor(a/b); d/da = 1, d/db = -floor(a/b) (floor is stop-gradient).
        from pycograd import ops

        return self._binary(
            o,
            lambda a, b: a % b,
            lambda a, b, g: (g, -_xp().floor(a / b) * g),
            ops.d_mod,
        )

    def __rmod__(self, o: Operand) -> Var:
        from pycograd import ops

        return _lift(o)._binary(
            self,
            lambda a, b: a % b,
            lambda a, b, g: (g, -_xp().floor(a / b) * g),
            ops.d_mod,
        )

    def __neg__(self) -> Var:
        from pycograd import ops

        return self._unary(-self.value, lambda a, g: -g, ops.d_neg)

    def __pow__(self, p: Operand) -> Var:
        from pycograd import ops

        if isinstance(p, Var):
            # general x ** y == exp(y * log x)
            return ops.d_exp(p * ops.d_log(self))
        # Record the *constant* exponent as a second operand so ``_VJP_FOR[d_pow]`` can read
        # it; ``_lift(p)`` makes it a leaf ``Var`` aligned with ``_parents`` (its gradient is
        # unused -- the exponent is constant on this path).
        return self._binary(
            p, lambda a, b: a**b, lambda a, b, g: (g * b * a ** (b - 1), g), ops.d_pow
        )

    def __abs__(self) -> Var:
        from pycograd import ops

        return ops.d_abs(self)

    def __matmul__(self, o: Operand) -> Var:
        from pycograd import ops

        return ops._matmul(self, o)

    def __rmatmul__(self, o: Operand) -> Var:
        from pycograd import ops

        return ops._matmul(o, self)

    # -- comparisons return plain (non-differentiable) booleans --------------
    def __lt__(self, o: Operand) -> Array:
        return self.value < _value(o)

    def __le__(self, o: Operand) -> Array:
        return self.value <= _value(o)

    def __gt__(self, o: Operand) -> Array:
        return self.value > _value(o)

    def __ge__(self, o: Operand) -> Array:
        return self.value >= _value(o)

    def __eq__(self, o: Operand) -> Array:  # type: ignore[override]
        return self.value == _value(o)

    def __ne__(self, o: Operand) -> Array:  # type: ignore[override]
        return self.value != _value(o)

    __hash__ = None  # type: ignore[assignment]

    # -- indexing (gather forward, scatter-add backward) ---------------------
    def __getitem__(self, key: Index) -> Var:
        out = Var(self.value[key], _parents=(self,))

        def _backward() -> None:
            grad = _xp().zeros_like(self.value)
            # scatter-add (np.add.at / cupyx.scatter_add) handles repeated indices
            current_backend().scatter_add(grad, key, out.grad)
            self.grad = _accumulate(self.grad, grad)

        out._backward = _backward
        from pycograd import ops

        _record_vjp(out, ops.d_getitem, (self,), {"key": key})
        return out

    # -- numpy-style methods -------------------------------------------------
    def __getattr__(self, name: str) -> Callable[..., Var]:
        # Provide numpy's array-method surface (``x.sum(...)``, ``x.mean(...)``,
        # ``x.reshape(...)``, ``x.clip(...)``, ...) generically rather than enumerating
        # each: route a numpy function name we have a VJP rule for to its
        # differentiable primitive. Only reached for names not resolved normally, so
        # ``value`` / ``grad`` / ``T`` / dunders are unaffected; an unknown name (or
        # one with no rule) raises ``AttributeError``.
        if name.startswith("__"):
            raise AttributeError(name)
        from pycograd import ops

        np_fn = getattr(np, name, None)
        prim = ops._INTERCEPT.get(np_fn) if callable(np_fn) else None
        if (
            prim is None and name == "flatten"
        ):  # no ``np.flatten``; it's ``ravel`` (a copy)
            prim = ops.d_ravel
        if prim is None:
            raise AttributeError(name)
        return functools.partial(prim, self)

    @property
    def T(self) -> Var:
        from pycograd import ops

        return ops.d_transpose(self)

    # -- non-differentiable metadata (read-only views of the underlying array) -
    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def ndim(self) -> int:
        return self.value.ndim

    @property
    def size(self) -> int:
        return self.value.size

    @property
    def dtype(self) -> np.dtype:
        return self.value.dtype

    def __len__(self) -> int:
        # Mirror numpy: the length of the leading axis (a 0-d array has no len).
        return len(self.value)

    def __array__(self, *args: object, **kwargs: object) -> Array:
        # No concrete array view: a numpy call reaching here was NOT intercepted,
        # so fail loudly instead of silently building an object array. Defining it
        # also makes Var statically array-like, so numpy-typed code checks.
        raise TypeError(
            "Var has no array conversion; this numpy call should have been "
            "intercepted -- is the calling function instrumented?"
        )

    def __repr__(self) -> str:
        return f"Var({self.value!r})"

    # -- reverse pass --------------------------------------------------------
    def backward(
        self,
        cotangent: Array | None = None,
        keep_batch_axis: int | None = None,
        differentiable: bool | None = None,
    ) -> None:
        """Accumulate gradients into every ancestor leaf's ``.grad``.

        With no arguments this is the ordinary reverse pass: seed ``self.grad`` with ones
        and walk the tape. ``cotangent`` instead seeds ``self.grad`` with an explicit
        (possibly *batched*) cotangent -- e.g. ``vmap(grad(f))`` seeds a per-example
        cotangent over a batched output so each example's gradient flows independently.

        ``keep_batch_axis`` marks an axis of that batched cotangent as the per-example
        axis: it is threaded through every VJP closure's ``_unbroadcast`` (via the
        ``_KEEP_BATCH_AXES`` contextvar) so a *shared* parameter -- one entering the
        batched forward with a size-1 batch axis -- keeps that axis on the way back,
        yielding a per-sample gradient of shape ``(B, *param.shape)`` instead of the
        batch-summed ``param.shape``.

        ``differentiable`` selects the *differentiable cotangent* pass (cotangents
        accumulated as tape/level-connected ``Var``s, see :meth:`_backward_differentiable`)
        rather than the raw-numpy ``.grad`` pass. ``value_and_grad`` passes ``True`` exactly
        when an enclosing differentiation context is live -- an outer ``jvp``
        (forward-over-reverse) or an outer ``grad`` (reverse-over-reverse, ``grad(grad)``)
        -- so the enclosing transform can differentiate this gradient. ``None`` (a direct
        ``.backward()`` call) falls back to "a real ``jvp``/``vmap`` transform level is
        live", keeping the default safe. The raw pass is re-entrant: its toposort/visited
        and zero/seed state are entirely local to this call, so an *outer* raw pass can
        walk the cotangent ``Var`` graph an *inner* differentiable pass produced
        (``grad(grad(f))`` materializes at the top via the outer raw pass).
        """
        topo: list[Var] = []
        visited = set()

        def build(v: Var) -> None:
            if id(v) in visited:
                return
            visited.add(id(v))
            for parent in v._parents:
                build(parent)
            topo.append(v)

        build(self)
        # Differentiable path: an enclosing differentiation context wants the gradient
        # computation *recorded* (on a ``jvp`` level for forward-over-reverse, or on the
        # outer ``grad``'s ``Var`` tape for reverse-over-reverse). The raw-numpy path below
        # is otherwise byte-for-byte unchanged. (``keep_batch_axis`` -- the ``vmap(grad)``
        # per-sample path -- shares state with the differentiable ``_unbroadcast`` and is
        # explicitly out of scope, so it always takes the raw path.)
        if differentiable is None:
            from pycograd.trace import num_transform_levels

            differentiable = num_transform_levels() > 0
        if keep_batch_axis is None and differentiable:
            self._backward_differentiable(topo, cotangent)
            return
        xp = _xp()
        for v in topo:
            v.grad = xp.zeros_like(v.value)
        self.grad = xp.ones_like(self.value) if cotangent is None else cotangent
        if keep_batch_axis is None:
            for v in reversed(topo):
                v._backward()
            return
        token = _KEEP_BATCH_AXES.set((keep_batch_axis,))
        try:
            for v in reversed(topo):
                v._backward()
        finally:
            _KEEP_BATCH_AXES.reset(token)

    def _backward_differentiable(
        self, topo: "list[Var]", cotangent: Array | None
    ) -> None:
        """Reverse pass that accumulates each node's cotangent as a tape/level-connected
        ``Var`` in a local ``dict[id(node) -> Var]`` instead of mutating numpy ``.grad``.

        Walks the same ``_parents`` toposort in reverse; each node's VJP is computed via
        ``ops._VJP_FOR`` with ``bind``-riding ops, so the cotangent graph differentiates
        under an enclosing ``jvp``/``grad``. The leaves' cotangents are written back to
        ``.grad`` (as ``Var``s the higher level can read) so the existing
        ``value_and_grad`` plumbing -- which reads ``leaf_var.grad`` -- carries them out.
        """
        from pycograd import ops
        from pycograd.forward import _hof_tracer_for
        from pycograd.trace import bind

        # Map a primal ``Var`` (a forward-tape node) to the *level tracer* that wraps it
        # (e.g. a ``JVPTracer`` carrying its tangent), so each VJP rides the enclosing
        # level and the cotangent graph carries second-order information.
        tracer_for = _hof_tracer_for()

        def _connected(node: Var) -> Boxed:
            return tracer_for.get(id(node), node)

        cot: dict[int, Boxed] = {}
        # The seed cotangent is a constant (zero tangent at the enclosing level), so a
        # plain ``Var`` suffices; ``bind``-riding VJP ops lift it as needed.
        seed: Boxed = _lift(
            _xp().ones_like(self.value) if cotangent is None else cotangent
        )
        cot[id(self)] = seed
        for v in reversed(topo):
            g = cot.get(id(v))
            if g is None or not v._parents:
                continue
            operands = tuple(_connected(o) for o in v._vjp_operands)
            contribs = ops._vjp_contributions(v, g, operands)
            for parent, contrib in zip(v._parents, contribs):
                if contrib is None:
                    continue
                if _logical_shape(contrib) != parent.value.shape:
                    contrib = _d_unbroadcast(contrib, parent.value.shape)
                prev = cot.get(id(parent))
                cot[id(parent)] = (
                    contrib if prev is None else bind(ops.d_add, prev, contrib)
                )
        # Expose each node's cotangent (a level-connected ``Var``/``Tracer``) on ``.grad``
        # so the higher transform reads it out of each leaf. On this path ``.grad`` carries
        # a tape/level value rather than a raw array (cast for the type-checker).
        for v in topo:
            g = cot.get(id(v))
            v.grad = cast(
                Array, g if g is not None else _lift(_xp().zeros_like(v.value))
            )


def _record_vjp(
    out: Var,
    prim: Prim,
    operands: tuple[Var, ...],
    params: dict[str, Any] | None = None,
) -> Var:
    """Tag ``out`` with the data the differentiable backward needs: the producing
    primitive, its primal operand ``Var``s (aligned with ``out._parents``), and its static
    keyword params. Returns ``out`` for call-site chaining. A no-op on the fast path
    (read only when a higher trace level is live)."""
    out._vjp_prim = prim
    out._vjp_operands = operands
    out._vjp_params = params or {}
    return out


def _lift(x: Operand) -> Var:
    x = _unwrap_weight(x)
    if isinstance(x, Var):
        return x
    if isinstance(x, (list, tuple)) and _seq_has_box(x):
        # A python list/tuple holding tape values -- e.g. ``np.mean([a, b])`` or a list arg
        # numpy can't stack because the leaves have no array conversion. Stack it onto a new
        # leading axis (a differentiable constructor) so the reduction/op sees one array.
        from pycograd import ops

        return cast(Var, ops.d_stack(list(x)))
    return Var(x)


def _seq_has_box(seq: "list | tuple") -> bool:
    from pycograd.trace import Tracer

    return any(isinstance(e, (Var, Tracer)) for e in seq)


def detach(x: Operand) -> Var:
    """A fresh leaf with the same value but no gradient history (stop-gradient)."""
    x = _unwrap_weight(x)
    return Var(x.value if isinstance(x, Var) else x)

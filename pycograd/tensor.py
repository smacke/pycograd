# -*- coding: utf-8 -*-
"""The reverse-mode tape node.

``Var`` wraps a numpy array; arithmetic operators are overloaded so that running a
program builds a computation graph, and ``Var.backward()`` walks it in reverse to
accumulate gradients. The convenience methods (``sum``/``mean``/``__pow__``/``T``/
...) delegate to the differentiable primitives in :mod:`pycograd.ops`; those
imports are deferred to function bodies so this module has no import-time
dependency on ``ops`` (which imports ``Var`` from here).
"""
from __future__ import annotations

import contextvars
import functools
from typing import Any, Callable

import numpy as np

from pycograd._typing import Array, ArrayLike, Index, Operand
from pycograd.backends import current_backend
from pycograd.dtypes import current_dtype, resolve_dtype


# ---------------------------------------------------------------------------
# The array-module seam: data is computed with the *active* backend's array
# library (numpy by default, cupy under ``device("cupy")``), so the same tape and
# the same VJP rules run on whatever device that backend lives on.
# ---------------------------------------------------------------------------
def _xp() -> Any:
    """The active backend's array module -- ``numpy`` unless a device is activated.

    Typed ``Any`` because the module is chosen at runtime (numpy / cupy / ...); the
    surface pycograd uses is the shared array-API subset both provide.
    """
    return current_backend().array_module


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
    grad = _xp().asarray(grad, dtype=current_dtype())
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


def _unwrap_weight(x: Operand) -> Var | ArrayLike:
    """Resolve a ``Weight`` proxy to its current value; leave anything else."""
    # Deferred import: params imports Var/_lift/_unbroadcast from this module.
    from pycograd.params import Weight

    return x._live() if isinstance(x, Weight) else x


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
    return (isinstance(x, (int, float)) and not isinstance(x, bool)) or _is_array(x)


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
        dtype: object = None,
    ) -> None:
        xp = _xp()
        dt = current_dtype() if dtype is None else resolve_dtype(dtype)
        self.value: Array = xp.asarray(value, dtype=dt)
        self.grad: Array = xp.zeros_like(self.value)
        self._parents = _parents
        self._backward: Callable[[], None] = lambda: None

    # -- construction helpers ------------------------------------------------
    def _unary(self, value: ArrayLike, grad_fn: Callable[[Array, Array], Array]) -> Var:
        out = Var(value, _parents=(self,))

        def _backward() -> None:
            self.grad = self.grad + _unbroadcast(
                grad_fn(self.value, out.grad), self.value.shape
            )

        out._backward = _backward
        return out

    def _binary(
        self,
        other: Operand,
        value_fn: Callable[[Array, Array], Array],
        grad_fn: Callable[[Array, Array, Array], tuple[Array, Array]],
    ) -> Var:
        other = _lift(other)
        out = Var(value_fn(self.value, other.value), _parents=(self, other))

        def _backward() -> None:
            ga, gb = grad_fn(self.value, other.value, out.grad)
            self.grad = self.grad + _unbroadcast(ga, self.value.shape)
            other.grad = other.grad + _unbroadcast(gb, other.value.shape)

        out._backward = _backward
        return out

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, o: Operand) -> Var:
        return self._binary(o, lambda a, b: a + b, lambda a, b, g: (g, g))

    __radd__ = __add__  # addition is commutative

    def __mul__(self, o: Operand) -> Var:
        return self._binary(o, lambda a, b: a * b, lambda a, b, g: (g * b, g * a))

    __rmul__ = __mul__  # multiplication is commutative

    def __sub__(self, o: Operand) -> Var:
        return self._binary(o, lambda a, b: a - b, lambda a, b, g: (g, -g))

    def __rsub__(self, o: Operand) -> Var:
        return self._binary(o, lambda a, b: b - a, lambda a, b, g: (-g, g))

    def __truediv__(self, o: Operand) -> Var:
        return self._binary(
            o, lambda a, b: a / b, lambda a, b, g: (g / b, -g * a / (b * b))
        )

    def __rtruediv__(self, o: Operand) -> Var:
        return self._binary(
            o, lambda a, b: b / a, lambda a, b, g: (-g * b / (a * a), g / a)
        )

    def __neg__(self) -> Var:
        return self._unary(-self.value, lambda a, g: -g)

    def __pow__(self, p: Operand) -> Var:
        from pycograd import ops

        if isinstance(p, Var):
            # general x ** y == exp(y * log x)
            return ops.d_exp(p * ops.d_log(self))
        return self._unary(self.value**p, lambda a, g: g * p * a ** (p - 1))

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
            self.grad = self.grad + grad

        out._backward = _backward
        return out

    # -- numpy-style methods -------------------------------------------------
    def __getattr__(self, name: str) -> object:
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


def _lift(x: Operand) -> Var:
    x = _unwrap_weight(x)
    return x if isinstance(x, Var) else Var(x)


def detach(x: Operand) -> Var:
    """A fresh leaf with the same value but no gradient history (stop-gradient)."""
    x = _unwrap_weight(x)
    return Var(x.value if isinstance(x, Var) else x)

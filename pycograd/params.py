# -*- coding: utf-8 -*-
"""Parameters and the ambient-weights proxy.

A bare array/number is trainable -- the default -- so existing code is unchanged.
Wrap a value in a ``Param`` only when you need to *freeze* it (``frozen``) or *tie*
it to another leaf (``tied``). ``Weight`` is a late-bound proxy injected by
``with weights:`` so a forward pass can reference parameters unqualified and serve
both inference and training; ``ParamDict`` is the named-parameter container.

``ParamDict.grad``/``step`` and ``param_values`` bridge to
:mod:`pycograd.transforms`, :mod:`pycograd.tracer`, and :mod:`pycograd.tree` via
deferred imports, so this module has no import-time dependency on them.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, cast

import numpy as np

from pycograd._typing import (
    Array,
    ArrayLike,
    BackendArray,
    DTypeLike,
    Index,
    Operand,
    Prim,
)
from pycograd.backends import active_backend_or_none
from pycograd.dtypes import current_dtype
from pycograd.ops import _INTERCEPT, _warn_wrapper
from pycograd.tensor import Var, _is_numeric, _value, grad_recording

if TYPE_CHECKING:
    from pycograd.tree import PyTree


# ---------------------------------------------------------------------------
# Parameters (optional, opt-in).
# ---------------------------------------------------------------------------
@dataclass
class Param:
    """A model-parameter leaf carrying differentiation metadata.

    ``trainable=False`` holds the value fixed: its gradient comes back ``None``
    and ``sgd_update`` leaves it untouched. ``mutable=True`` (with
    ``trainable=False``) marks a *buffer* -- a non-gradient leaf the forward pass
    may still advance (a batch-norm running mean/variance), updated out-of-band via
    :meth:`ParamDict.update_buffers` rather than by the optimizer. ``tie`` groups
    leaves that are the *same* underlying weight -- every ``Param`` sharing a
    ``tie`` key is backed by one tape node, so the gradient accumulates once and is
    reported identically at each position (tied params must be initialized equal;
    the shared node adopts the first occurrence's value).
    """

    value: Array
    trainable: bool = True
    # A non-trainable-but-mutable buffer (running stats etc.): no gradient, skipped
    # by the optimizer, but writable via ``ParamDict.update_buffers``.
    mutable: bool = False
    tie: Hashable | None = None
    # Stamped by the ``params{...}`` DSL with the block that declared this param,
    # so ``value_and_grad`` can reject a weight also passed in by hand.
    origin: object = None

    def __post_init__(self) -> None:
        self.value = np.asarray(self.value, dtype=current_dtype())


class _Frozen:
    """A non-trainable parameter, as a call or a subscript: ``frozen(v)`` /
    ``frozen[v]`` (the bracket form reads naturally inside a ``params{...}`` block)."""

    def __call__(self, value: ArrayLike) -> Param:
        return Param(np.asarray(value, dtype=current_dtype()), trainable=False)

    def __getitem__(self, value: ArrayLike) -> Param:
        return self(value)


frozen = _Frozen()


class _Buffer:
    """A non-trainable but mutable *buffer*, as a call or a subscript:
    ``buffer(v)`` / ``buffer[v]`` (the bracket form reads naturally inside a
    ``params{...}`` block). Carries no gradient and is skipped by the optimizer,
    but the forward pass advances it via :meth:`ParamDict.update_buffers` (e.g. a
    batch-norm running mean/variance)."""

    def __call__(self, value: ArrayLike) -> Param:
        return Param(
            np.asarray(value, dtype=current_dtype()), trainable=False, mutable=True
        )

    def __getitem__(self, value: ArrayLike) -> Param:
        return self(value)


buffer = _Buffer()


class _TieRef:
    """Marker produced by ``tied[w]``: tie this slot to the sibling parameter whose
    value is ``target`` (the same array), reusing its init -- resolved by ``params``."""

    __slots__ = ("target",)

    def __init__(self, target: Operand | Param) -> None:
        self.target = target


class _Tied:
    """Weight tying. ``tied(key, value)`` ties every parameter sharing ``key``;
    ``tied[w]`` (inside a ``params(...)`` block) ties to the sibling parameter ``w``
    by reference -- just the name, with no restatement of its initializer."""

    def __call__(self, key: Hashable, value: ArrayLike) -> Param:
        return Param(np.asarray(value, dtype=current_dtype()), tie=key)

    def __getitem__(self, ref: Operand | Param) -> _TieRef:
        return _TieRef(ref)


tied = _Tied()


# ---------------------------------------------------------------------------
# Weight: a late-bound proxy for an ambient parameter, plus its unwrap helpers.
# ---------------------------------------------------------------------------
def _unwrap(x: object) -> object:
    """Resolve a ``Weight`` proxy to its current value; leave anything else."""
    return x._live() if isinstance(x, Weight) else x


def _as_arr(x: object) -> Array:
    """``_unwrap`` an operand and view it as an ndarray for typing (a ``Var``
    operand still dispatches correctly at runtime)."""
    return cast(Array, _unwrap(x))


def _deep_unwrap(x: object) -> object:
    """``_unwrap`` through lists/tuples (e.g. ``np.concatenate([w1, w2])``)."""
    if isinstance(x, Weight):
        return x._live()
    if isinstance(x, list):
        return [_deep_unwrap(e) for e in x]
    if isinstance(x, tuple):
        return tuple(_deep_unwrap(e) for e in x)
    return x


# Sentinel: "no delegate backend is active, fall through to the normal dispatch".
_NO_DELEGATE = object()


def _delegate_dispatch(
    func: Prim, args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    """Route a numpy ufunc / array-function over a bare weight onto the active *delegate*
    backend, mirroring the recording branch's ``_INTERCEPT`` routing -- but with operands
    coerced onto the backend. Eager forwards don't need this (their binops/calls already
    route through the tracer seam), but *graph capture* does: ``make_fx``'s proxy mode
    dispatches ``proxy @ weight`` through ``__array_ufunc__``, where the default branch
    would hand a grad tensor to a raw numpy op. Returns :data:`_NO_DELEGATE` when no
    delegate backend is active, so the ``Var``-tape / plain-numpy dispatch is unchanged.
    """
    be = active_backend_or_none()
    if be is None or not be.is_delegate:
        return _NO_DELEGATE
    operands = [be.coerce_operand(_deep_unwrap(a)) for a in args]
    repl = be.intercept.get(func)
    fn = repl if repl is not None else be.on_unmapped(func)
    return fn(*operands, **kwargs)


def _any_recording(obj: object) -> bool:
    """True if a numpy call over ``obj`` should record -- i.e. some operand
    resolves to a ``Var`` (training), vs. plain arrays (inference)."""
    if isinstance(obj, Var):
        return True
    if isinstance(obj, Weight):
        return isinstance(obj._live(), Var)
    if isinstance(obj, (list, tuple)):
        return any(_any_recording(e) for e in obj)
    return False


class Weight:
    """A late-bound proxy for an ambient parameter (injected by ``with weights:``).

    It forwards every operation to the *current* value of its weight: the plain
    array during inference (so nothing is taped) and the live ``Var`` during a
    ``grad`` pass (so the op records). The proxy is mode-agnostic -- the mode is
    whatever its owner is currently bound to -- which is what lets a single model
    definition serve both inference and training without binding a ``Var`` into it.
    It speaks numpy's dispatch protocols, so ``np.exp(w)`` / ``np.sum(w)`` on a bare
    weight route to the differentiable primitive when recording and to plain numpy
    otherwise.
    """

    __slots__ = ("_owner", "_key")

    def __init__(self, owner: ParamDict, key: str) -> None:
        self._owner = owner
        self._key = key

    def _live(self) -> Var | ArrayLike:
        return self._owner._resolve_weight(self._key)

    # The live value is a Var or ndarray; both support the operators below, but
    # mypy can't express "whichever it is, it has @/+/...", so we view it as an
    # ndarray for typing -- the runtime op dispatches correctly to Var either way.
    def _arr(self) -> Array:
        live = self._live()
        # Under a delegate backend with no grad binding (a forward-only ``compile_to`` of an
        # ambient net), the weight resolves to its raw numpy value; lift it onto the backend
        # so ``backend_input @ weight`` works. No-op on the numpy tape (no backend active)
        # and on the grad path (``_live`` already holds a backend tensor).
        be = active_backend_or_none()
        if be is not None and be.is_delegate:
            return cast(Array, be.coerce_operand(live))
        return cast(Array, live)

    # -- operators: forward to the live value, unwrapping proxy operands ----------
    def __matmul__(self, o: Operand) -> Operand:
        return self._arr() @ _as_arr(o)

    def __rmatmul__(self, o: Operand) -> Operand:
        return _as_arr(o) @ self._arr()

    def __add__(self, o: Operand) -> Operand:
        return self._arr() + _as_arr(o)

    __radd__ = __add__

    def __sub__(self, o: Operand) -> Operand:
        return self._arr() - _as_arr(o)

    def __rsub__(self, o: Operand) -> Operand:
        return _as_arr(o) - self._arr()

    def __mul__(self, o: Operand) -> Operand:
        return self._arr() * _as_arr(o)

    __rmul__ = __mul__

    def __truediv__(self, o: Operand) -> Operand:
        return self._arr() / _as_arr(o)

    def __rtruediv__(self, o: Operand) -> Operand:
        return _as_arr(o) / self._arr()

    def __pow__(self, o: Operand) -> Operand:
        return self._arr() ** _as_arr(o)

    def __neg__(self) -> Operand:
        return -self._arr()

    def __getitem__(self, key: Index) -> Operand:
        return cast(Operand, self._arr()[key])

    @property
    def T(self) -> Operand:
        return self._arr().T

    @property
    def shape(self) -> tuple[int, ...]:
        return np.shape(_value(self))

    @property
    def ndim(self) -> int:
        return np.ndim(_value(self))

    @property
    def size(self) -> int:
        return int(np.size(_value(self)))

    # -- numpy dispatch: route a ufunc / array-function over a bare weight --------
    def __array_ufunc__(
        self, ufunc: Prim, method: str, *inputs: object, **kwargs: Any
    ) -> object:
        if method != "__call__":
            return NotImplemented
        delegate = _delegate_dispatch(ufunc, inputs, kwargs)
        if delegate is not _NO_DELEGATE:
            return delegate
        if _any_recording(inputs):
            prim = _INTERCEPT.get(ufunc)
            if prim is not None:
                return prim(*[_unwrap(i) for i in inputs], **kwargs)
            return _warn_wrapper(ufunc)(*[_deep_unwrap(i) for i in inputs], **kwargs)
        return ufunc(*[_deep_unwrap(i) for i in inputs], **kwargs)

    def __array_function__(
        self,
        func: Prim,
        types: Iterable[type],
        args: tuple[object, ...],
        kwargs: dict[str, Any],
    ) -> object:
        delegate = _delegate_dispatch(func, args, kwargs)
        if delegate is not _NO_DELEGATE:
            return delegate
        if _any_recording(args):
            prim = _INTERCEPT.get(func)
            if prim is not None:
                return prim(*args, **kwargs)
            return _warn_wrapper(func)(*[_deep_unwrap(a) for a in args], **kwargs)
        return func(*[_deep_unwrap(a) for a in args], **kwargs)

    def __repr__(self) -> str:
        return f"Weight({self._key!r})"


class ParamDict(dict):
    """A parameter dict that also allows attribute access: ``model.w`` reads
    ``model["w"]`` (and ``model.w = v`` sets it). ``params(...)`` returns one at
    every dict level. It is an ordinary ``dict`` otherwise, so the pytree
    machinery, ``value_and_grad`` and ``sgd_update`` treat it as a dict; the
    flatten/unflatten round trip preserves the type, so attribute access survives
    an optimizer step (and gradient pytrees come back attribute-accessible too). A
    weight named like a dict method (``items``) is still reachable via
    ``model["items"]``.

    Used as a context manager, ``with weights:`` injects a ``Weight`` proxy for each
    parameter into the caller's module/cell globals (collision-checked, removed on
    exit), so a forward pass can reference the weights *unqualified* -- ``w`` rather
    than ``weights.w`` -- and still serve both clean inference and training. Drive
    training with :meth:`grad` (binds the weights to ``Var``s, backprops, returns a
    gradient ``ParamDict``) and :meth:`step` (in-place SGD).
    """

    # Live binding during a grad pass (name -> Var/array), and the injected-globals
    # bookkeeping for the context manager. Both are real instance attributes, set
    # via the leading-underscore path in ``__setattr__``.
    _live: dict[str, object] | None = None
    _scope: tuple[dict[str, object], list[str]] | None = None

    # Leading-underscore attributes are real instance state (e.g. the live-binding
    # used during a grad pass); every other name maps to a dict item.
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            object.__delattr__(self, name)
            return
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    # -- ambient-weights context manager + tape-style training --------------------
    def _resolve_weight(self, key: str) -> Var | ArrayLike:
        """The current value of ``key``: the live ``Var`` during a grad pass, else
        the parameter's plain array (inference)."""
        live = getattr(self, "_live", None)
        if live is not None and key in live:
            return cast("Var | ArrayLike", live[key])
        leaf = self[key]
        return leaf.value if isinstance(leaf, Param) else cast("Var | ArrayLike", leaf)

    def __enter__(self) -> ParamDict:
        g = sys._getframe(1).f_globals
        injected: list[str] = []
        for key in self:
            if not isinstance(self[key], Param):
                continue
            existing = g.get(key)
            if key in g and not (
                isinstance(existing, Weight) and existing._owner is self
            ):
                for name in injected:
                    del g[name]
                raise ValueError(
                    f"with-weights: name {key!r} already exists in the enclosing "
                    "scope; rename the parameter or the existing variable"
                )
            g[key] = Weight(self, key)
            injected.append(key)
        self._scope = (g, injected)
        return self

    def __exit__(self, *exc: object) -> None:
        scope = self._scope
        if scope is not None:
            g, injected = scope
            for key in injected:
                got = g.get(key)
                if isinstance(got, Weight) and got._owner is self:
                    del g[key]
            self._scope = None

    def grad(
        self,
        objective: Callable[[], PyTree],
        *,
        backend: "str | None" = None,
        dtype: "DTypeLike | None" = None,
        jit: bool = False,
    ) -> tuple[Array, ParamDict]:
        """Run ``objective`` (a no-arg callable returning a scalar, built from this
        model) with the weights bound to ``Var``s, backprop, and return
        ``(value, grads)`` where ``grads`` is a ``ParamDict`` of gradients (``None``
        at frozen weights). Tied weights share one gradient. Trainable leaves only;
        nest by calling per sub-tree.

        With ``backend=`` (``"torch"`` / ``"jax"`` / ``"tf"``) the same ambient objective
        is instead run on that framework and differentiated by *its* autodiff -- the
        compile twin of the numpy-tape path, with identical ``frozen`` / ``tied`` / ``None``
        semantics. ``dtype`` selects the working precision the leaves are lifted in.
        No framework is imported until a non-numpy ``backend`` is named.

        ``jit=True`` (jax only; ignored elsewhere) compiles the gradient **once** and
        reuses it across calls, so a training loop traces the net a single time instead of
        every step. Cached per ``(objective, leaf structure)`` on this model; valid only
        while the objective's non-weight inputs (e.g. data captured by closure) stay fixed
        -- so it fits full-batch loops, not minibatching that swaps the data each step.
        """
        if backend is not None and backend != "numpy":
            return self._grad_via_backend(objective, backend, dtype, jit)
        from pycograd.tracer import _INSTRUMENTED, _make_runner
        from pycograd.transforms import _wrap_leaf

        tie_vars: dict[Hashable, Var] = {}
        live: dict[str, object] = {}
        grad_vars: dict[str, Var | None] = {}
        for key in self:
            leaf = self[key]
            if isinstance(leaf, Param) or _is_numeric(leaf):
                var, call_value = _wrap_leaf(leaf, tie_vars)
                live[key] = call_value
                grad_vars[key] = var
            else:
                live[key] = leaf
                grad_vars[key] = None
        runner = _INSTRUMENTED.get(objective)
        if runner is None:
            runner = _make_runner(objective)  # cache: avoid recompiling per call
            _INSTRUMENTED[objective] = runner
        prev = getattr(self, "_live", None)
        self._live = live
        # ``grad_recording`` lets an ambient ``vmap(forward)(X)`` inside ``objective`` keep
        # its output on the tape even though the weights arrive by closure, not as args.
        try:
            with grad_recording():
                out = runner()
        finally:
            self._live = prev
        if isinstance(out, Var):
            out.backward()
            value: Array = out.value
        else:
            value = np.asarray(out, dtype=current_dtype())
        grads = ParamDict(
            {
                key: (gv.grad if gv is not None else None)
                for key, gv in grad_vars.items()
            }
        )
        return value, grads

    def _grad_via_backend(
        self,
        objective: Callable[[], PyTree],
        backend: str,
        dtype: DTypeLike | None,
        jit: bool,
    ) -> tuple[Array, ParamDict]:
        """``grad`` on a delegate backend: bind each weight to a *backend tensor* in the
        ambient ``_live`` slot (instead of a ``Var``), run the instrumented objective under
        that backend, and read gradients from the framework's own autodiff. Leaf planning
        (frozen -> constant, tied -> shared, per-leaf gradient) reuses the same helpers as
        :func:`pycograd.compile.value_and_grad`, so the two paths agree leaf-for-leaf.

        With ``jit=True`` the backend's compiled gradient (``Backend.compile_grad``) is
        cached on this model per ``(objective, leaf structure)`` and reused across calls, so
        jax traces the net once rather than every step.
        """
        from pycograd.backends import activate, get_backend
        from pycograd.compile import _fill, _grad_for, _plan_leaf
        from pycograd.dtypes import _maybe_dtype
        from pycograd.tracer import _INSTRUMENTED, _make_runner

        be = get_backend(backend)
        with _maybe_dtype(dtype):
            trainable: list[Array] = []
            tie_slot: dict[object, int] = {}
            slots = {key: _plan_leaf(self[key], trainable, tie_slot) for key in self}

            runner = _INSTRUMENTED.get(objective)
            if runner is None:
                runner = _make_runner(objective)
                _INSTRUMENTED[objective] = runner

            # ``slots`` is rebuilt each call but is structurally identical step to step, so a
            # cached compiled gradient (closed over the first call's slots) stays valid.
            def make_scalar_fn(
                plan: dict[str, tuple],
            ) -> Callable[[list[object]], object]:
                def scalar_fn(tensors: list[object]) -> object:
                    with activate(be):
                        prev = getattr(self, "_live", None)
                        self._live = {k: _fill(plan[k], tensors, be) for k in self}
                        try:
                            return runner()
                        finally:
                            self._live = prev

                return scalar_fn

            if jit:
                cache = getattr(self, "_compiled_grad", None)
                if cache is None:
                    cache = {}
                    self._compiled_grad = cache
                sig = (
                    (id(objective), be.name)
                    + tuple((key, slots[key][0]) for key in self)
                    + tuple((np.shape(t), str(np.asarray(t).dtype)) for t in trainable)
                )
                compiled = cache.get(sig)
                if compiled is None:
                    compiled = be.compile_grad(make_scalar_fn(slots))
                    cache[sig] = compiled
                value, grad_leaves = compiled(trainable)
            else:
                value, grad_leaves = be.grad_and_value(make_scalar_fn(slots), trainable)
            grads = ParamDict(
                {key: _grad_for(slots[key], grad_leaves, be) for key in self}
            )
            return np.asarray(value), grads

    def step(self, grads: ParamDict, lr: float) -> None:
        """One in-place SGD step: ``p.value -= lr * grad`` for each trainable leaf
        (frozen / buffer leaves, whose gradient is ``None``, are left untouched). The
        model's proxies read the updated values on the next call."""
        for key in self:
            leaf = self[key]
            g = grads.get(key)
            if isinstance(leaf, Param) and leaf.trainable and g is not None:
                leaf.value = cast(Array, leaf.value) - lr * cast(Array, g)

    def update_buffers(self, updates: dict[str, ArrayLike]) -> None:
        """Write new values into mutable ``buffer`` leaves in place (running stats
        and the like). The controlled, explicit mutation point for non-gradient
        state: thread the advanced buffers a forward returns (e.g. ``batch_norm``'s
        ``new_mean`` / ``new_var``) back here. Refuses any key that is not a
        ``buffer`` (a trainable weight or a frozen constant)."""
        for key, value in updates.items():
            leaf = self.get(key)
            if not (isinstance(leaf, Param) and leaf.mutable):
                raise ValueError(
                    f"update_buffers: {key!r} is not a mutable buffer "
                    "(declare it with buffer[...] in params(...))"
                )
            leaf.value = np.asarray(value, dtype=current_dtype())

    def to_torch_module(
        self, forward: Callable[..., object], *, dtype: DTypeLike | None = None
    ) -> object:
        """Wrap an ambient ``forward`` (built against this model) as a ``torch.nn.Module``.

        The DSL twin of :func:`pycograd.export.to_torch_module`: this model's trainable
        leaves become ``Parameter``s (frozen leaves become buffers), and ``module(*inputs)``
        binds those live tensors into the ambient slot and runs the compiled-to-torch
        ``forward``. So the module trains with any torch optimizer (``loss.backward()``) and
        traces to TorchScript / ONNX via :func:`pycograd.export.export_torchscript` /
        :func:`~pycograd.export.export_onnx`. torch is imported only here.

        ``forward`` reads the weights by the names ``with weights:`` injects, so call the
        module (and trace it for export) **inside that block** -- the exported TorchScript /
        ONNX artifact then carries the captured graph and runs with no pycograd dependency.
        """
        import torch

        from pycograd.backends.torch_backend import _as_torch
        from pycograd.compile import compile_to
        from pycograd.dtypes import _maybe_dtype

        run = compile_to(forward, "torch", dtype=dtype)
        owner = self
        keys = list(self)

        class AmbientModule(torch.nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self._slots: list[tuple[str, str, object]] = []
                self._weights = torch.nn.ParameterList()
                for key in keys:
                    leaf = owner[key]
                    value = leaf.value if isinstance(leaf, Param) else leaf
                    with _maybe_dtype(dtype):
                        tensor = _as_torch(torch, value)
                    if isinstance(leaf, Param) and not leaf.trainable:
                        name = f"_frozen_{len(self._slots)}"
                        self.register_buffer(name, tensor)
                        self._slots.append((key, "buffer", name))
                    else:
                        self._slots.append((key, "weight", len(self._weights)))
                        self._weights.append(torch.nn.Parameter(tensor))

            def _live_tensors(self) -> dict[str, object]:
                return {
                    key: (
                        self._weights[cast(int, ref)]
                        if kind == "weight"
                        else getattr(self, cast(str, ref))
                    )
                    for key, kind, ref in self._slots
                }

            def forward(self, *inputs: BackendArray) -> BackendArray:
                prev = getattr(owner, "_live", None)
                owner._live = self._live_tensors()
                try:
                    return run(*inputs)
                finally:
                    owner._live = prev

        return AmbientModule()


# ---------------------------------------------------------------------------
# Declarative parameter blocks (optional). ``params(...)`` builds a parameter
# pytree: name each weight as a keyword, write plain numbers/arrays for trainable
# weights, and use ``frozen`` / ``tied`` for the rest.
# ---------------------------------------------------------------------------
def params(spec: dict[str, PyTree] | None = None, **named: PyTree) -> ParamDict:
    """Build a parameter pytree as a ``ParamDict``: ``{name: Param}``, with nested
    dicts/lists allowed and attribute access (``model.w`` as well as ``model["w"]``).

    ``params(w=0.1 * rng.standard_normal((2, 3)), b=np.zeros(3))`` makes both
    trainable; wrap a value in ``frozen(...)``/``frozen[...]`` to hold it fixed, or
    ``tied(key, ...)`` to share it by key. Inside the block, ``tied[w]`` ties a slot
    to the sibling parameter ``w`` by reference, reusing its initializer. Pass a
    mapping positionally instead of keywords if your names are not valid
    identifiers. The result flows through ``value_and_grad`` and ``sgd_update``
    unchanged; use ``param_values`` to recover the raw arrays for inference.
    """
    if spec is not None and named:
        raise TypeError("params() takes either a mapping or keyword args, not both")
    items = dict(spec) if spec is not None else dict(named)
    token = object()  # this block's identity, for the single-owner check

    # Map each top-level value's array identity to its declaring name, so a
    # ``tied[w]`` reference can find the sibling parameter it shares a weight with.
    declarer: dict[int, str] = {}
    for name, value in items.items():
        arr = value.value if isinstance(value, Param) else value
        if isinstance(arr, np.ndarray):
            declarer.setdefault(id(arr), name)
    tie_keys: dict[str, Hashable] = {}  # target name -> shared tie sentinel

    def wrap(v: PyTree, key: str | None = None) -> PyTree:
        if isinstance(v, _TieRef):
            arr = v.target.value if isinstance(v.target, Param) else v.target
            target = declarer.get(id(arr)) if isinstance(arr, np.ndarray) else None
            if target is None or target == key:
                raise ValueError(
                    "autodiff: tied[...] must reference another parameter declared "
                    "in the same params(...) call (by name)"
                )
            p = Param(np.asarray(cast(ArrayLike, arr), dtype=current_dtype()))
            p.tie = tie_keys.setdefault(target, object())
            p.origin = token
            return p
        if isinstance(v, Param):
            if v.origin is None:
                v.origin = token  # adopt leaves from frozen()/tied() and nesting
            return v
        if isinstance(v, dict):
            return ParamDict({k: wrap(child) for k, child in v.items()})
        if isinstance(v, list):
            return [wrap(child) for child in v]
        if isinstance(v, tuple):
            return tuple(wrap(child) for child in v)
        if _is_numeric(v):
            p = Param(np.asarray(cast(ArrayLike, v), dtype=current_dtype()))
            p.origin = token
            return p
        return v  # non-numeric leaf (e.g. a label): left alone, no gradient

    result = ParamDict({key: wrap(value, key) for key, value in items.items()})
    # Each referenced target must carry the same tie key as the slots tied to it.
    for target, sentinel in tie_keys.items():
        leaf = result.get(target)
        if isinstance(leaf, Param):
            leaf.tie = sentinel
    return result


def param_values(tree: PyTree) -> PyTree:
    """Strip ``Param`` wrappers back to raw values, preserving structure -- the
    inverse of what ``params`` adds, for running a model at inference time."""
    from pycograd.tree import tree_map

    return tree_map(lambda p: p.value if isinstance(p, Param) else p, tree)


def register_pipescript_params_macro() -> None:
    """Wire the literal ``params{ w = ...; b = ... }`` brace surface into pipescript.

    pipescript's namespace-block mechanism runs the brace block, harvests its
    top-level assignments (``_``-prefixed names are local temporaries, excluded),
    and hands them to a builder; we point that builder at ``params(**names)``, so::

        model = params{
            w = 0.1 * rng.standard_normal((2, 16))
            b = np.zeros(16)
            g = frozen[np.ones(16)]
            out = tied[w]               # ties to w by name, reusing its init
        }

    builds the same ``{name: Param}`` pytree as the call form. ``frozen`` / ``tied``
    must be in scope inside the block. Requires pipescript; call once after loading
    it (e.g. in a notebook after ``%load_ext pipescript``)."""
    from pipescript.tracers.macro_tracer import register_namespace_macro

    register_namespace_macro(
        "params", lambda namespace: params(**namespace), call_form=params
    )

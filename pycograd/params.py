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
from typing import TYPE_CHECKING, Callable, cast

import numpy as np

from pycograd._typing import Array, ArrayLike, Index, Operand
from pycograd.dtypes import current_dtype
from pycograd.ops import _INTERCEPT, _warn_wrapper
from pycograd.tensor import Var, _is_numeric, _value

if TYPE_CHECKING:
    from pycograd.tree import PyTree


# ---------------------------------------------------------------------------
# Parameters (optional, opt-in).
# ---------------------------------------------------------------------------
@dataclass
class Param:
    """A model-parameter leaf carrying differentiation metadata.

    ``trainable=False`` holds the value fixed: its gradient comes back ``None``
    and ``sgd_update`` leaves it untouched. ``tie`` groups leaves that are the
    *same* underlying weight -- every ``Param`` sharing a ``tie`` key is backed
    by one tape node, so the gradient accumulates once and is reported
    identically at each position (tied params must be initialized equal; the
    shared node adopts the first occurrence's value).
    """

    value: Array
    trainable: bool = True
    tie: object = None
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


class _TieRef:
    """Marker produced by ``tied[w]``: tie this slot to the sibling parameter whose
    value is ``target`` (the same array), reusing its init -- resolved by ``params``."""

    __slots__ = ("target",)

    def __init__(self, target: object) -> None:
        self.target = target


class _Tied:
    """Weight tying. ``tied(key, value)`` ties every parameter sharing ``key``;
    ``tied[w]`` (inside a ``params(...)`` block) ties to the sibling parameter ``w``
    by reference -- just the name, with no restatement of its initializer."""

    def __call__(self, key: object, value: ArrayLike) -> Param:
        return Param(np.asarray(value, dtype=current_dtype()), tie=key)

    def __getitem__(self, ref: object) -> _TieRef:
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
        return cast(Array, self._live())

    # -- operators: forward to the live value, unwrapping proxy operands ----------
    def __matmul__(self, o: object) -> Operand:
        return self._arr() @ _as_arr(o)

    def __rmatmul__(self, o: object) -> Operand:
        return _as_arr(o) @ self._arr()

    def __add__(self, o: object) -> Operand:
        return self._arr() + _as_arr(o)

    __radd__ = __add__

    def __sub__(self, o: object) -> Operand:
        return self._arr() - _as_arr(o)

    def __rsub__(self, o: object) -> Operand:
        return _as_arr(o) - self._arr()

    def __mul__(self, o: object) -> Operand:
        return self._arr() * _as_arr(o)

    __rmul__ = __mul__

    def __truediv__(self, o: object) -> Operand:
        return self._arr() / _as_arr(o)

    def __rtruediv__(self, o: object) -> Operand:
        return _as_arr(o) / self._arr()

    def __pow__(self, o: object) -> Operand:
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
        self, ufunc: object, method: str, *inputs: object, **kwargs: object
    ) -> object:
        if method != "__call__":
            return NotImplemented
        if _any_recording(inputs):
            prim = _INTERCEPT.get(ufunc)
            if prim is not None:
                return prim(*[_unwrap(i) for i in inputs], **kwargs)
            return _warn_wrapper(cast("Callable[..., object]", ufunc))(
                *[_deep_unwrap(i) for i in inputs], **kwargs
            )
        return cast("Callable[..., object]", ufunc)(
            *[_deep_unwrap(i) for i in inputs], **kwargs
        )

    def __array_function__(
        self,
        func: Callable[..., object],
        types: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
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

    def grad(self, objective: Callable[[], object]) -> tuple[Array, ParamDict]:
        """Run ``objective`` (a no-arg callable returning a scalar, built from this
        model) with the weights bound to ``Var``s, backprop, and return
        ``(value, grads)`` where ``grads`` is a ``ParamDict`` of gradients (``None``
        at frozen weights). Tied weights share one gradient. Trainable leaves only;
        nest by calling per sub-tree."""
        from pycograd.tracer import _INSTRUMENTED, _make_runner
        from pycograd.transforms import _wrap_leaf

        tie_vars: dict[object, Var] = {}
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
        try:
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

    def step(self, grads: ParamDict, lr: float) -> None:
        """One in-place SGD step: ``p.value -= lr * grad`` for each trainable leaf
        (frozen leaves, whose gradient is ``None``, are left untouched). The model's
        proxies read the updated values on the next call."""
        for key in self:
            leaf = self[key]
            g = grads.get(key)
            if isinstance(leaf, Param) and leaf.trainable and g is not None:
                leaf.value = cast(Array, leaf.value) - lr * cast(Array, g)


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
    tie_keys: dict[str, object] = {}  # target name -> shared tie sentinel

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

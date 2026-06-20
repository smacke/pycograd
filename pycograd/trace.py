# -*- coding: utf-8 -*-
"""The trace-level interpreter stack (dispatch core).

This is the spine a JAX-"autodidax"-style composable-transform engine hangs off:
every operation flows through :func:`bind`, which finds the *top* (highest-level)
interpreter among its operands and lets that interpreter process the primitive. The
bottom of the stack is :class:`EvalTrace` -- the base level, level 0 -- whose
``process_primitive`` just runs the primitive on the active backend's values. So for
ordinary arrays / :class:`~pycograd.tensor.Var` operands, ``bind(d_add, a, b)`` is
exactly ``a + b``: the operator primitive (:mod:`pycograd.ops`) reuses ``Var``'s tape.

**Levels in play.** The base level is always :class:`EvalTrace`. ``vmap`` pushes a
:class:`~pycograd.batching.BatchTrace` level per call (via :func:`new_main`), so
``vmap(vmap(f))`` runs two batch levels at once: :func:`find_top_trace` selects the
outer one, its rules ``bind`` one level down, and the recursion bottoms out at
``EvalTrace``. ``eval_shape``'s :class:`~pycograd.shapes.ShapedArray` is *also* a
:class:`Tracer` now, at an :class:`~pycograd.shapes.AbstractTrace` level pushed by
``eval_shape``: an operand routed through ``bind`` selects that level, whose
``process_primitive`` runs the per-primitive shape rule. (A ``ShapedArray`` built
outside a live ``eval_shape`` carries a level-0 sentinel trace, so it never out-ranks a
real level and its standalone operator dunders -- which call the same rules -- work.)

The pyccolo front-door (:mod:`pycograd.tracer`) routes intercepted operators here via
:data:`_OP_PRIM` (ast op type -> primitive); intercepted ``np.*`` *calls* route here too
whenever a non-base level is live (so ``vmap`` vectorizes them), and run on the active
backend directly otherwise.
"""
from __future__ import annotations

import ast
import contextlib
import contextvars
import operator
from typing import Any, Callable, Iterator, Sequence

from pycograd import ops


# ---------------------------------------------------------------------------
# Trace / Tracer / MainTrace.
# ---------------------------------------------------------------------------
class MainTrace:
    """One level of the interpreter stack: a trace *type* plus its global data.

    ``level`` orders the stack (0 is the base :class:`EvalTrace`); ``trace_type`` is the
    :class:`Trace` subclass to instantiate for this level; ``global_data`` is whatever
    that trace needs that is shared across all its tracers (e.g. a batch size). The
    instance is created lazily so a level can be pushed before its trace is used.
    """

    __slots__ = ("level", "trace_type", "global_data")

    def __init__(
        self, level: int, trace_type: type["Trace"], global_data: object = None
    ) -> None:
        self.level = level
        self.trace_type = trace_type
        self.global_data = global_data

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"MainTrace(level={self.level}, {self.trace_type.__name__})"


class Trace:
    """An interpreter for one stack level.

    Subclasses define how a value enters this level (``pure`` for a constant, ``lift``
    for a value from the level below) and how a primitive is processed at this level
    (``process_primitive``). The base level is :class:`EvalTrace`.
    """

    def __init__(self, main: MainTrace) -> None:
        self.main = main

    def pure(self, val: object) -> object:
        raise NotImplementedError

    def lift(self, val: object) -> object:
        raise NotImplementedError

    def process_primitive(
        self, prim: Callable[..., object], args: Sequence[object], params: dict
    ) -> object:
        """Process ``prim`` over its *raw* call arguments at this level.

        ``args`` are the operands exactly as ``bind`` received them (not pre-raised);
        each trace lifts the ones it cares about into its own level via
        :func:`full_raise`. ``params`` are the static keyword arguments.
        """
        raise NotImplementedError


class Tracer:
    """A value tagged with the stack level that is currently tracing it.

    ``EvalTrace`` operates on raw arrays / ``Var``s (not ``Tracer``s); the concrete
    subclasses are :class:`~pycograd.batching.BatchTracer` (``vmap``),
    :class:`~pycograd.forward.JVPTracer` (``jvp``), and
    :class:`~pycograd.shapes.ShapedArray` (``eval_shape``'s abstract level). The base
    class is the shared seam they plug into: :func:`find_top_trace` ranks operands by
    ``_trace.main.level`` and ``aval`` exposes the value's shape/dtype.
    """

    _trace: Trace

    @property
    def aval(self) -> object:
        raise NotImplementedError

    @property
    def shape(self) -> object:
        return self.aval.shape  # type: ignore[attr-defined]

    @property
    def ndim(self) -> int:
        return len(self.aval.shape)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The base level: EvalTrace.
# ---------------------------------------------------------------------------
class EvalTrace(Trace):
    """The bottom of the stack (level 0): just evaluate.

    A primitive is run directly on its operands; for the differentiable ``d_*``
    primitives that means computing on the *active backend* (numpy by default, cupy
    under ``device(...)``, the foreign framework under a compile backend) and -- for a
    numpy/cupy tape backend -- building the ``Var`` tape, exactly as a bare operator or
    intercepted ``np.*`` call would. So nothing about ``grad`` / the tape changes; the
    stack just adds a uniform entry point above it.
    """

    def pure(self, val: object) -> object:
        return val

    lift = pure

    def process_primitive(
        self, prim: Callable[..., object], args: Sequence[object], params: dict
    ) -> object:
        return prim(*args, **params)


# ---------------------------------------------------------------------------
# Reverse-mode marker level.
#
# ``value_and_grad`` pushes one of these around its forward+backward so that a *nested*
# ``grad``/``value_and_grad`` can detect it is running inside an enclosing differentiation
# context (``grad(grad(f))``): the inner call sees the stack is deeper than base and takes
# the differentiable backward, building a cotangent graph the outer ``grad`` then walks.
#
# It is a pure depth marker: it carries no :class:`Tracer`, so :func:`find_top_trace`
# never selects it and ``bind`` dispatch is completely unaffected (every operation still
# lands on the base ``EvalTrace`` or whatever real transform level is live). The
# fast/slow discriminators that must stay byte-for-byte for a single top-level ``grad``
# (the tracer's ``np.*``-routing gate and ``jvp``'s top-level coercion) therefore count
# *transform* levels via :func:`num_transform_levels`, which skips these markers.
# ---------------------------------------------------------------------------
class ReverseTrace(Trace):
    """A do-nothing marker level pushed by ``value_and_grad`` (see module note above).

    It exists only to raise ``len(_get_stack())`` for the duration of a reverse pass so a
    nested ``grad`` knows it is enclosed. It never processes a primitive (no operand ever
    carries it, so :func:`find_top_trace` skips it); the methods are present only to honor
    the :class:`Trace` interface.
    """

    def pure(self, val: object) -> object:  # pragma: no cover - never selected
        return val

    lift = pure

    def process_primitive(  # pragma: no cover - never selected
        self, prim: Callable[..., object], args: Sequence[object], params: dict
    ) -> object:
        return prim(*args, **params)


# ---------------------------------------------------------------------------
# The stack.
# ---------------------------------------------------------------------------
_BASE_MAIN = MainTrace(0, EvalTrace)

trace_stack: contextvars.ContextVar[list[MainTrace]] = contextvars.ContextVar(
    "pycograd_trace_stack", default=[_BASE_MAIN]
)


def _get_stack() -> list[MainTrace]:
    return trace_stack.get()


def num_transform_levels() -> int:
    """How many *dispatch-affecting* transform levels are live above the base.

    Counts every pushed level except the base ``EvalTrace`` and the
    :class:`ReverseTrace` markers ``value_and_grad`` pushes. This is the discriminator for
    behavior that must stay byte-for-byte under a single top-level ``grad`` (whose only
    extra level is its own marker): a real ``vmap``/``jvp`` level is present iff this is
    nonzero, while a bare reverse pass leaves it at zero.
    """
    return sum(
        1 for m in _get_stack() if m.level > 0 and m.trace_type is not ReverseTrace
    )


@contextlib.contextmanager
def new_main(
    trace_type: type["Trace"], global_data: object = None
) -> "Iterator[MainTrace]":
    """Push a fresh interpreter level onto the stack for the duration of the block.

    The new level sits one above the current top, so its trace becomes the one
    :func:`find_top_trace` selects for any operand tagged with it; nesting two
    ``new_main(BatchTrace)`` blocks gives two batch levels, which is exactly how
    ``vmap(vmap(f))`` composes. The contextvar is reset on exit, so the push is
    scoped to the ``with`` block and threads cleanly across async/contextvar copies.
    """
    stack = _get_stack()
    main = MainTrace(len(stack), trace_type, global_data)
    token = trace_stack.set(stack + [main])
    try:
        yield main
    finally:
        trace_stack.reset(token)


def _iter_tracers(args: Sequence[object]) -> "list[Tracer]":
    """The :class:`Tracer` operands among ``args``, recursing one level into lists/tuples
    (so a batched operand inside a ``concatenate``/``stack`` sequence, or a batched index
    key, still raises the trace level)."""
    out: list[Tracer] = []
    for a in args:
        if isinstance(a, Tracer):
            out.append(a)
        elif isinstance(a, (list, tuple)):
            out.extend(x for x in a if isinstance(x, Tracer))
    return out


def find_top_trace(args: Sequence[object]) -> Trace:
    """The highest-level interpreter among ``args`` (the base ``EvalTrace`` if none).

    Only :class:`Tracer` operands carry a level; raw arrays / ``Var``s are level 0, so
    they leave the base trace on top and ``bind`` evaluates. A live ``ShapedArray`` (under
    ``eval_shape``) is a ``Tracer`` at its :class:`~pycograd.shapes.AbstractTrace` level
    and so selects that level here. Sequences are searched one level deep so a batched
    operand inside a ``concatenate``/``stack`` list still selects its level.
    """
    top = _get_stack()[0]
    for a in _iter_tracers(args):
        if a._trace.main.level > top.level:
            top = a._trace.main
    return top.trace_type(top)


def full_raise(trace: Trace, val: object) -> object:
    """Lift ``val`` into ``trace``'s level.

    A value already at this level passes through; a :class:`Tracer` from a lower level
    is ``lift``ed; a constant is made ``pure``. (At the base level both are identities,
    so this is a no-op in Phase 1; it is the seam higher levels use to insert an
    unbatched ``bdim=None`` operand.)
    """
    if not isinstance(val, Tracer):
        return trace.pure(val)
    level = trace.main.level
    if val._trace.main.level == level:
        return val
    if val._trace.main.level < level:
        return trace.lift(val)
    raise ValueError(  # pragma: no cover - a lower trace never lifts a higher value
        "can't lift a value from a higher trace level into a lower one"
    )


# ---------------------------------------------------------------------------
# Operator primitives: ast op -> primitive, plus the raw Python operator used to
# fall through for values the stack does not (yet) manage.
# ---------------------------------------------------------------------------
# Each entry is (differentiable primitive, raw Python operator). The primitive routes
# base-level values through the tape; the raw operator is invoked verbatim when no value
# the stack manages is present (plain numbers/arrays, the abstract ``ShapedArray``), so
# their own dunders run and their behavior is unchanged.
_BINOP_PRIM: dict[type, tuple[Callable[..., object], Callable[..., object]]] = {
    ast.Add: (ops.d_add, operator.add),
    ast.Sub: (ops.d_sub, operator.sub),
    ast.Mult: (ops.d_mul, operator.mul),
    ast.Div: (ops.d_div, operator.truediv),
    ast.Pow: (ops.d_pow, operator.pow),
    ast.MatMult: (ops._matmul, operator.matmul),
}

_UNARYOP_PRIM: dict[type, tuple[Callable[..., object], Callable[..., object]]] = {
    ast.USub: (ops.d_neg, operator.neg),
}

_COMPARE_PRIM: dict[type, tuple[Callable[..., object], Callable[..., object]]] = {
    ast.Lt: (ops.d_lt, operator.lt),
    ast.LtE: (ops.d_le, operator.le),
    ast.Gt: (ops.d_gt, operator.gt),
    ast.GtE: (ops.d_ge, operator.ge),
    ast.Eq: (ops.d_eq, operator.eq),
    ast.NotEq: (ops.d_ne, operator.ne),
}

# ast op type -> differentiable primitive (the ``_OP_PRIM`` table the plan names).
_OP_PRIM: dict[type, Callable[..., object]] = {
    **{op: prim for op, (prim, _) in _BINOP_PRIM.items()},
    **{op: prim for op, (prim, _) in _UNARYOP_PRIM.items()},
    **{op: prim for op, (prim, _) in _COMPARE_PRIM.items()},
    # ``ast.Subscript`` has no ``.op``; the subscript handler keys on it directly so the
    # name->primitive table is complete.
    ast.Subscript: ops.d_getitem,
}

# The reverse map a managed-or-not fall-through needs: primitive -> raw operator.
_RAW_OPERATOR: dict[Callable[..., object], Callable[..., object]] = {
    prim: raw
    for table in (_BINOP_PRIM, _UNARYOP_PRIM, _COMPARE_PRIM)
    for prim, raw in table.values()
}
_RAW_OPERATOR[ops.d_getitem] = operator.getitem


class _SubscriptProxy:
    """A one-shot stand-in returned by the ``before_subscript_load`` handler so that
    ``var[key]`` routes through :func:`bind`.

    pyccolo's subscript event replaces the *object being subscripted* (not the op with a
    callable, the way binop/compare work), then performs ``replacement[key]``. So to
    intercept ``x[key]`` we return this proxy in place of ``x``; its ``__getitem__`` may
    bind ``d_getitem`` over the original object and the key. The handler wraps a ``Var``
    (the base level pycograd manages) and a raw ``np.ndarray`` (so a shared/unbatched
    table gathered with a batched index reaches ``bind``); ``dict``/``list``/``ParamDict``
    are never wrapped, so those subscripts are untouched.

    For a wrapped object the proxy only routes to ``bind`` when the object is a ``Var`` or
    the *key* involves a :class:`Tracer` (e.g. a ``vmap`` ``BatchTracer`` index over a
    shared table). Otherwise it falls through to plain ``self._obj[key]``, so ordinary
    array indexing by non-tracer keys is byte-for-byte unchanged.
    """

    __slots__ = ("_obj",)

    def __init__(self, obj: object) -> None:
        self._obj = obj

    def __getitem__(self, key: object) -> object:
        from pycograd.tensor import Var

        if isinstance(self._obj, Var) or _iter_tracers(
            key if isinstance(key, tuple) else (key,)
        ):
            return bind(ops.d_getitem, self._obj, key)
        return self._obj[key]  # type: ignore[index]


def _is_managed(val: object) -> bool:
    """True for a value the base level (``EvalTrace``) drives through a ``d_*`` operator
    primitive: a :class:`~pycograd.tensor.Var`.

    Everything else with no :class:`Tracer` present -- plain numbers / arrays (plain
    arithmetic) and a compile backend's foreign tensors -- must keep its *original*
    operator behavior, so for an operator primitive ``bind`` falls through to the raw
    Python operator unless a ``Var`` is present. (A live ``ShapedArray`` *is* a ``Tracer``,
    so it is handled by its level, not here.) Imported lazily to keep ``trace``
    import-light and dodge an import cycle.
    """
    from pycograd.tensor import Var

    return isinstance(val, Var)


# ---------------------------------------------------------------------------
# bind: the single entry point every (intercepted) operation flows through.
# ---------------------------------------------------------------------------
def bind(prim: Callable[..., object], *args: object, **params: Any) -> object:
    """Dispatch ``prim`` over ``args`` through the trace-level stack.

    Find the top trace among the operands (searching one level into sequences, so a
    batched operand inside a ``concatenate``/``stack`` list still selects its level) and
    let it ``process_primitive`` over the *raw* arguments. For the base level this is
    just ``prim(*args, **params)``.

    **Operator fall-through.** An *operator* primitive only builds the ``Var`` tape when
    a ``Var`` is actually present; with no :class:`Tracer` and no ``Var`` operand, routing
    through a ``d_*`` primitive would be wrong -- two plain numbers want plain arithmetic,
    and a compile backend's foreign tensors want their *own* dunders. So for an operator
    primitive (recognized by its entry in :data:`_RAW_OPERATOR`) with no tracer and no
    managed operand, ``bind`` applies the raw Python operator. A :class:`Tracer` operand
    (a :class:`~pycograd.batching.BatchTracer`, a :class:`~pycograd.forward.JVPTracer`, or
    an ``eval_shape`` :class:`~pycograd.shapes.ShapedArray`) instead selects its level,
    which processes the primitive.
    """
    if (
        prim in _RAW_OPERATOR
        and not _iter_tracers(args)
        and not any(_is_managed(a) for a in args)
    ):
        return _RAW_OPERATOR[prim](*args)
    trace = find_top_trace(args)
    return trace.process_primitive(prim, args, params)

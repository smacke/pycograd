# -*- coding: utf-8 -*-
"""Forward-mode automatic differentiation (``jvp``): a *trace level* that carries a
tangent alongside each primal.

``jvp`` is the forward-mode dual of reverse-mode ``grad``, realized -- like
:mod:`pycograd.batching`'s ``vmap`` -- as one level of the trace-level interpreter
stack (:mod:`pycograd.trace`). Each value flowing through the level is a
:class:`JVPTracer` pairing a *primal* (the ordinary value) with a *tangent* (its
directional derivative). When a primitive reaches ``bind``, the top :class:`JVPTrace`
handles it: it raises every operand into the level (a constant / lower value lifts
with a **zero tangent**), computes ``primal_out = bind(prim, *primals)`` (recursing one
level down, exactly as ``EvalTrace`` would), and computes ``tangent_out`` via the
per-primitive **jvp rule** -- the forward derivative ``f'(x) . dx``.

Because every jvp rule builds its tangent with ordinary ``ops`` / operators (which all
flow through ``bind``), forward mode *composes*: nesting a ``JVPTrace`` under a
:class:`~pycograd.batching.BatchTrace` (``vmap(jvp(...))``) just sees the tangent
computation recurse into the batch level, and nesting a ``JVPTrace`` under ``grad``
(reverse-over-forward) keeps the tangent arithmetic on the ``Var`` tape.
"""
from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any, Callable, NoReturn, Sequence, cast

import numpy as np

from pycograd import ops
from pycograd._typing import Array, Axis, BindArg, Boxed, DTypeLike, Index, Prim, Rule
from pycograd.tensor import Var, _value, _xp
from pycograd.trace import Trace, Tracer, bind, full_raise

if TYPE_CHECKING:
    from pycograd.shapes import ShapeDtypeStruct

# ---------------------------------------------------------------------------
# Higher-order bridge: map a *primal* ``Var`` (a forward-tape node) to the ``JVPTracer``
# that wraps it, so the differentiable reverse pass (``Var.backward`` under a live ``jvp``)
# can route each VJP through the tangent-carrying tracer rather than the bare primal --
# i.e. forward-over-reverse (Hessians / HVPs). Populated as the ``jvp`` forward runs and
# read by ``tensor._backward_differentiable``; an empty map (no live ``jvp``) means the
# reverse pass simply uses the primal ``Var``s, which is the plain ``grad`` behavior.
# ---------------------------------------------------------------------------
_HOF_TRACER_FOR: contextvars.ContextVar[dict[int, Boxed]] = contextvars.ContextVar(
    "pycograd_hof_tracer_for", default={}
)


def _hof_tracer_for() -> dict[int, Boxed]:
    return _HOF_TRACER_FOR.get()


def _hof_register(tracer: "JVPTracer") -> None:
    """Record ``id(tracer.primal) -> tracer`` so the reverse pass can recover the
    tangent-carrying tracer for a primal ``Var``. Only the primal is keyed; the tracer it
    maps to carries the tangent the higher-order reverse pass needs."""
    p = tracer.primal
    if isinstance(p, Var):
        _HOF_TRACER_FOR.get()[id(p)] = tracer


# ---------------------------------------------------------------------------
# JVPTracer / JVPTrace: a (primal, tangent) pair, and the level that processes
# primitives over such pairs.
# ---------------------------------------------------------------------------
class JVPTracer(Tracer):
    """A value carrying its directional derivative (``tangent``) beside its ``primal``.

    Both ``primal`` and ``tangent`` are level-down values (a :class:`~pycograd.tensor.Var`
    / array for a single ``jvp``, a lower-level :class:`Tracer` when nested). The
    ``tangent`` has the same shape/dtype as the ``primal``; a constant entering the level
    gets a zero tangent (see :meth:`JVPTrace.pure`).
    """

    __slots__ = ("_trace", "primal", "tangent")
    __array_ufunc__ = None

    def __init__(self, trace: "JVPTrace", primal: Boxed, tangent: Boxed) -> None:
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    @property
    def aval(self) -> "ShapeDtypeStruct":
        from pycograd.shapes import ShapeDtypeStruct

        p = self.primal
        if isinstance(p, Tracer):
            # Nested under another transform (e.g. a BatchTracer): defer to its own
            # logical shape/dtype rather than materializing it.
            return ShapeDtypeStruct(tuple(cast(Any, p.shape)), np.dtype(p.dtype))  # type: ignore[attr-defined]
        arr = p.value if isinstance(p, Var) else p
        shp = tuple(np.shape(cast(Any, _value(cast(Any, arr)))))
        dt = np.asarray(cast(Any, _value(cast(Any, arr)))).dtype
        return ShapeDtypeStruct(shp, np.dtype(dt))

    @property
    def dtype(self) -> np.dtype:
        return cast(Any, self.aval).dtype

    @property
    def size(self) -> int:
        return int(np.prod(cast(Any, self.shape), dtype=np.int64))

    @property
    def T(self) -> Boxed:
        return bind(ops.d_transpose, self)

    def __getitem__(self, key: Index) -> Boxed:
        return bind(ops.d_getitem, self, key)

    def __getattr__(self, name: str) -> "Callable[..., Boxed]":
        # Route a numpy method name we have a primitive for through ``bind`` so the
        # method call (``x.sum(axis=0)``) differentiates exactly like the free function
        # ``np.sum(x, axis=0)`` -- mirrors ``BatchTracer.__getattr__``.
        if name.startswith("__"):
            raise AttributeError(name)
        np_fn = getattr(np, name, None)
        prim = ops._INTERCEPT.get(np_fn) if callable(np_fn) else None
        if prim is None:
            raise AttributeError(name)

        def _method(*a: BindArg, **k: Any) -> Boxed:
            return bind(prim, self, *a, **k)

        return _method

    def __repr__(self) -> str:
        return (
            f"JVPTracer(level={self._trace.main.level}, "
            f"shape={self.shape}, dtype={self.dtype})"
        )


class JVPTrace(Trace):
    """One forward-mode level: process each primitive by computing the primal one level
    down and the tangent via the per-primitive jvp rule."""

    def pure(self, val: Boxed) -> JVPTracer:
        return JVPTracer(self, val, _zeros_like(val))

    def lift(self, val: Boxed) -> JVPTracer:
        # A value from a lower level enters this level with a zero tangent (it does not
        # depend on the differentiated input at this level).
        return JVPTracer(self, val, _zeros_like(val))

    def process_primitive(
        self, prim: Prim, args: Sequence[BindArg], params: dict[str, Any]
    ) -> Boxed:
        rule = _JVP_FOR.get(prim)
        if rule is None:
            _process_unknown(prim)
        out = rule(self, *args, **params)
        # Bridge for higher-order reverse: remember the (primal ``Var`` -> this tracer)
        # link, plus the same link for each raised operand, so a later differentiable
        # ``backward`` routes its VJPs through the tangent-carrying tracers.
        if isinstance(out, JVPTracer):
            _hof_register(out)
        for a in args:
            if isinstance(a, JVPTracer):
                _hof_register(a)
            elif isinstance(a, (list, tuple)):
                for el in a:
                    if isinstance(el, JVPTracer):
                        _hof_register(el)
        return out

    def _raise(self, val: Boxed) -> "JVPTracer":
        return cast(JVPTracer, full_raise(self, val))


# ---------------------------------------------------------------------------
# Zero-tangent construction.
# ---------------------------------------------------------------------------
def _zeros_like(val: Boxed) -> Boxed:
    """A zero tangent shaped like ``val`` (a constant entering the level, or a value from
    a lower level)."""
    arr = _value(cast(Any, val)) if isinstance(val, (Var,)) else val
    arr = np.asarray(cast(Any, _value(cast(Any, arr))))
    return _xp().zeros(arr.shape, dtype=arr.dtype)


# ---------------------------------------------------------------------------
# Rule infrastructure.
#
# Each rule has the *natural call signature of its primitive* plus a leading ``trace``.
# It raises every operand into this level (constants / lower-level values -> zero
# tangent), computes ``primal_out`` by ``bind``-ing the primitive over the primals (one
# level down), and computes ``tangent_out`` with ordinary ``ops`` over the primals and
# input tangents. Because those ops flow through ``bind`` too, forward mode composes and
# nests automatically. The result is re-wrapped as a ``JVPTracer``.
# ---------------------------------------------------------------------------
def _result(trace: JVPTrace, primal: Boxed, tangent: Boxed) -> JVPTracer:
    return JVPTracer(trace, primal, tangent)


def _primals(tracers: Sequence[JVPTracer]) -> list[Boxed]:
    return [t.primal for t in tracers]


def _tangents(tracers: Sequence[JVPTracer]) -> list[Boxed]:
    return [t.tangent for t in tracers]


# -- linear ops: the tangent flows through the SAME op as the primal ---------
def _linear_for(prim: Prim) -> Rule:
    """For a linear primitive ``L`` (add, sub, neg, sum, mean, reshape, transpose,
    concatenate, expand_dims, stack, ...) the jvp is itself: ``d L(x) = L(dx)``."""

    def rule(trace: JVPTrace, *operands: Boxed, **kwargs: Any) -> JVPTracer:
        tracers = [trace._raise(o) for o in operands]
        primal_out = bind(prim, *_primals(tracers), **kwargs)
        tangent_out = bind(prim, *_tangents(tracers), **kwargs)
        return _result(trace, primal_out, tangent_out)

    return rule


def _linear_struct_for(prim: Prim) -> Rule:
    """Like :func:`_linear_for`, but the primitive's *trailing* arguments are static
    structural metadata (a reshape target, an ``expand_dims`` axis, a ``broadcast_to``
    shape) -- not differentiable operands. Only the first operand is raised; the rest pass
    through unraised so they are not (wrongly) turned into zero-tangent tracers."""

    def rule(trace: JVPTrace, x: Boxed, *static: BindArg, **kwargs: Any) -> JVPTracer:
        t = trace._raise(x)
        primal_out = bind(prim, t.primal, *static, **kwargs)
        tangent_out = bind(prim, t.tangent, *static, **kwargs)
        return _result(trace, primal_out, tangent_out)

    return rule


def _linear_seq_for(prim: Prim) -> Rule:
    """Linear over a *sequence* operand (concatenate / stack): the tangent of the join is
    the join of the per-element tangents."""

    def rule(trace: JVPTrace, seq: Sequence[Boxed], **kwargs: Any) -> JVPTracer:
        tracers = [trace._raise(s) for s in seq]
        primal_out = bind(prim, _primals(tracers), **kwargs)
        tangent_out = bind(prim, _tangents(tracers), **kwargs)
        return _result(trace, primal_out, tangent_out)

    return rule


# -- elementwise unary: tangent_out = f'(primal) * tangent ------------------
def _unary_for(deriv: Callable[[Boxed], Boxed]) -> Callable[[Prim], Rule]:
    """``deriv(primal)`` is the local derivative ``f'(x)``; the jvp is ``f'(x) * dx``.
    ``deriv`` is written with ordinary ``ops`` so the tangent rides ``bind``."""

    def make(prim: Prim) -> Rule:
        def rule(trace: JVPTrace, x: Boxed, **kwargs: Any) -> JVPTracer:
            t = trace._raise(x)
            primal_out = bind(prim, t.primal, **kwargs)
            tangent_out = bind(ops.d_mul, deriv(t.primal), t.tangent)
            return _result(trace, primal_out, tangent_out)

        return rule

    return make


# -- binary / selection -----------------------------------------------------
def _mul_rule(trace: JVPTrace, a: Boxed, b: Boxed) -> JVPTracer:
    ta, tb = trace._raise(a), trace._raise(b)
    primal_out = bind(ops.d_mul, ta.primal, tb.primal)
    # d(a*b) = a*db + b*da
    tangent_out = bind(
        ops.d_add,
        bind(ops.d_mul, ta.primal, tb.tangent),
        bind(ops.d_mul, tb.primal, ta.tangent),
    )
    return _result(trace, primal_out, tangent_out)


def _gated_act_rule(trace: JVPTrace, f: Boxed, s: Boxed) -> JVPTracer:
    # gate = tanh(f) * sigmoid(s); jvp = (∂/∂f)·df + (∂/∂s)·ds.
    tf, ts = trace._raise(f), trace._raise(s)
    primal_out = bind(ops.d_gated_act, tf.primal, ts.primal)
    # The gate coefficients (d/df, d/ds) are the *shared* ``ops._gated_act_coeffs`` -- the
    # reverse rule builds from the same helper, so they are written once.
    df_coeff, ds_coeff = ops._gated_act_coeffs(tf.primal, ts.primal)
    tangent_out = bind(
        ops.d_add,
        bind(ops.d_mul, df_coeff, tf.tangent),
        bind(ops.d_mul, ds_coeff, ts.tangent),
    )
    return _result(trace, primal_out, tangent_out)


def _softmax_rule(trace: JVPTrace, x: Boxed, axis: Axis = -1) -> JVPTracer:
    # y = softmax(x); dy = y * (dx - sum(y*dx, axis, keepdims)).
    t = trace._raise(x)
    y = bind(ops.d_softmax, t.primal, axis=axis)
    ydx_sum = bind(ops.d_sum, bind(ops.d_mul, y, t.tangent), axis=axis, keepdims=True)
    tangent_out = bind(ops.d_mul, y, bind(ops.d_sub, t.tangent, ydx_sum))
    return _result(trace, y, tangent_out)


def _logsumexp_rule(
    trace: JVPTrace, x: Boxed, axis: Axis = None, keepdims: bool = False
) -> JVPTracer:
    # d(logsumexp(x)) = sum(softmax(x) * dx, axis) -- softmax is the gradient of logsumexp.
    t = trace._raise(x)
    primal_out = bind(ops.d_logsumexp, t.primal, axis=axis, keepdims=keepdims)
    sm = bind(ops.d_softmax, t.primal, axis=axis)
    tangent_out = bind(
        ops.d_sum, bind(ops.d_mul, sm, t.tangent), axis=axis, keepdims=keepdims
    )
    return _result(trace, primal_out, tangent_out)


def _div_rule(trace: JVPTrace, a: Boxed, b: Boxed) -> JVPTracer:
    ta, tb = trace._raise(a), trace._raise(b)
    primal_out = bind(ops.d_div, ta.primal, tb.primal)
    # d(a/b) = (da*b - a*db) / b**2
    num = bind(
        ops.d_sub,
        bind(ops.d_mul, ta.tangent, tb.primal),
        bind(ops.d_mul, ta.primal, tb.tangent),
    )
    tangent_out = bind(ops.d_div, num, bind(ops.d_mul, tb.primal, tb.primal))
    return _result(trace, primal_out, tangent_out)


def _mod_rule(trace: JVPTrace, a: Boxed, b: Boxed) -> JVPTracer:
    ta, tb = trace._raise(a), trace._raise(b)
    primal_out = bind(ops.d_mod, ta.primal, tb.primal)
    # d(a % b) = da - floor(a/b)*db (floor is piecewise constant: stop-gradient).
    q = bind(ops.d_floor, bind(ops.d_div, ta.primal, tb.primal))
    tangent_out = bind(ops.d_sub, ta.tangent, bind(ops.d_mul, q, tb.tangent))
    return _result(trace, primal_out, tangent_out)


def _pow_rule(trace: JVPTrace, a: Boxed, b: Boxed) -> JVPTracer:
    ta, tb = trace._raise(a), trace._raise(b)
    primal_out = bind(ops.d_pow, ta.primal, tb.primal)
    # d(a**b) = b*a**(b-1)*da + a**b*log(a)*db. Drop the exponent-tangent term when the
    # exponent is constant (its tangent is zero), which is the common case (``x**2``) and
    # avoids ``log`` of a non-positive base.
    # ``b*a**(b-1)`` is the *shared* ``ops._pow_base_deriv`` (the reverse rule builds from
    # the same helper); only the exponent-tangent term below is forward-mode-specific.
    da_term = bind(ops.d_mul, ops._pow_base_deriv(ta.primal, tb.primal), ta.tangent)
    if _is_zero(tb.tangent):
        tangent_out: Boxed = da_term
    else:
        db_term = bind(
            ops.d_mul,
            bind(ops.d_mul, primal_out, bind(ops.d_log, ta.primal)),
            tb.tangent,
        )
        tangent_out = bind(ops.d_add, da_term, db_term)
    return _result(trace, primal_out, tangent_out)


def _matmul_rule(trace: JVPTrace, a: Boxed, b: Boxed) -> JVPTracer:
    ta, tb = trace._raise(a), trace._raise(b)
    primal_out = bind(ops._matmul, ta.primal, tb.primal)
    # d(a@b) = da@b + a@db
    tangent_out = bind(
        ops.d_add,
        bind(ops._matmul, ta.tangent, tb.primal),
        bind(ops._matmul, ta.primal, tb.tangent),
    )
    return _result(trace, primal_out, tangent_out)


def _einsum_rule(trace: JVPTrace, subscripts: str, *operands: Boxed) -> JVPTracer:
    # einsum is multilinear in its operands, so by the product rule the tangent is the
    # sum over operands of the einsum with that operand replaced by its tangent.
    # ``subscripts`` is static (not raised).
    tracers = [trace._raise(o) for o in operands]
    primals = [t.primal for t in tracers]
    primal_out = bind(ops.d_einsum, subscripts, *primals)
    tangent_out: Boxed = None
    for i, t in enumerate(tracers):
        args = list(primals)
        args[i] = t.tangent
        term = bind(ops.d_einsum, subscripts, *args)
        tangent_out = (
            term if tangent_out is None else bind(ops.d_add, tangent_out, term)
        )
    return _result(trace, primal_out, tangent_out)


def _roll_rule(trace: JVPTrace, x: Boxed, shift: Any, axis: Any = None) -> JVPTracer:
    """``roll`` is a linear permutation: roll the tangent the same way."""
    t = trace._raise(x)
    primal_out = bind(ops.d_roll, t.primal, shift, axis=axis)
    tangent_out = bind(ops.d_roll, t.tangent, shift, axis=axis)
    return _result(trace, primal_out, tangent_out)


def _pad_rule(
    trace: JVPTrace, x: Boxed, pad_width: Any, mode: str = "constant", **kw: Any
) -> JVPTracer:
    """``pad`` (constant mode) is linear: pad the tangent with *zeros* (the pad constant
    does not depend on ``x``)."""
    t = trace._raise(x)
    primal_out = bind(ops.d_pad, t.primal, pad_width, mode=mode, **kw)
    tangent_out = bind(ops.d_pad, t.tangent, pad_width, mode="constant")
    return _result(trace, primal_out, tangent_out)


def _repeat_rule(
    trace: JVPTrace, x: Boxed, repeats: Any, axis: Any = None
) -> JVPTracer:
    t = trace._raise(x)
    primal_out = bind(ops.d_repeat, t.primal, repeats, axis=axis)
    tangent_out = bind(ops.d_repeat, t.tangent, repeats, axis=axis)
    return _result(trace, primal_out, tangent_out)


def _tile_rule(trace: JVPTrace, x: Boxed, reps: Any) -> JVPTracer:
    t = trace._raise(x)
    return _result(
        trace, bind(ops.d_tile, t.primal, reps), bind(ops.d_tile, t.tangent, reps)
    )


def _select_for(prim: Prim) -> Rule:
    """max / min of two operands: route each operand's tangent through wherever that
    operand was selected (``mask = primal_out == operand``)."""

    def rule(trace: JVPTrace, a: Boxed, b: Boxed) -> JVPTracer:
        ta, tb = trace._raise(a), trace._raise(b)
        primal_out = bind(prim, ta.primal, tb.primal)
        mask = bind(ops.d_eq, ta.primal, primal_out)  # boolean: where a was picked
        tangent_out = bind(ops.d_where, mask, ta.tangent, tb.tangent)
        return _result(trace, primal_out, tangent_out)

    return rule


def _clip_rule(
    trace: JVPTrace,
    x: Boxed,
    a_min: Boxed = None,
    a_max: Boxed = None,
) -> JVPTracer:
    tx = trace._raise(x)
    primal_out = bind(ops.d_clip, tx.primal, a_min, a_max)
    # Tangent flows only where the value is strictly inside the clip window; where it is
    # pinned to a bound the derivative is zero.
    tangent_out = tx.tangent
    if a_min is not None:
        amin = trace._raise(a_min).primal
        inside = bind(ops.d_gt, tx.primal, amin)
        tangent_out = bind(ops.d_where, inside, tangent_out, _zeros_like(tx.tangent))
    if a_max is not None:
        amax = trace._raise(a_max).primal
        inside = bind(ops.d_lt, tx.primal, amax)
        tangent_out = bind(ops.d_where, inside, tangent_out, _zeros_like(tx.tangent))
    return _result(trace, primal_out, tangent_out)


def _where_rule(trace: JVPTrace, cond: Boxed, a: Boxed, b: Boxed) -> JVPTracer:
    ta, tb = trace._raise(a), trace._raise(b)
    cmask = (
        _value(cast(Any, trace._raise(cond).primal))
        if isinstance(cond, Tracer)
        else cond
    )
    primal_out = bind(ops.d_where, cmask, ta.primal, tb.primal)
    tangent_out = bind(ops.d_where, cmask, ta.tangent, tb.tangent)
    return _result(trace, primal_out, tangent_out)


def _getitem_rule(trace: JVPTrace, x: Boxed, key: Index) -> JVPTracer:
    tx = trace._raise(x)
    k = _unwrap_key(key)
    primal_out = bind(ops.d_getitem, tx.primal, k)
    tangent_out = bind(ops.d_getitem, tx.tangent, k)
    return _result(trace, primal_out, tangent_out)


def _unwrap_key(key: Index) -> Index:
    """Strip any JVPTracer wrapper off an index, using its primal (an index is not
    differentiated)."""

    def _one(k: Index) -> Index:
        if isinstance(k, JVPTracer):
            return _value(cast(Any, k.primal))
        return k

    if isinstance(key, tuple):
        return tuple(_one(k) for k in key)
    return _one(key)


def _reduce_for(prim: Prim) -> Rule:
    """sum / mean are linear: ``d reduce(x) = reduce(dx)`` (same axis/keepdims)."""

    def rule(
        trace: JVPTrace,
        x: Boxed,
        axis: Axis = None,
        keepdims: bool = False,
        **kw: Any,
    ) -> JVPTracer:
        t = trace._raise(x)
        primal_out = bind(prim, t.primal, axis=axis, keepdims=keepdims, **kw)
        tangent_out = bind(prim, t.tangent, axis=axis, keepdims=keepdims, **kw)
        return _result(trace, primal_out, tangent_out)

    return rule


def _prod_rule(
    trace: JVPTrace,
    x: Boxed,
    axis: Axis = None,
    keepdims: bool = False,
    **kw: Any,
) -> JVPTracer:
    """``prod`` is non-linear: d prod(x) = sum_axis( (prod(x)/x_i) * dx_i )."""
    t = trace._raise(x)
    primal_out = bind(ops.d_prod, t.primal, axis=axis, keepdims=keepdims)
    pk = bind(ops.d_prod, t.primal, axis=axis, keepdims=True)
    contrib = bind(ops.d_div, bind(ops.d_mul, pk, t.tangent), t.primal)
    tangent_out = bind(ops.d_sum, contrib, axis=axis, keepdims=keepdims)
    return _result(trace, primal_out, tangent_out)


def _reduced_count(x: Boxed, axis: Axis) -> int:
    shp = np.asarray(cast(Any, _value(cast(Any, x)))).shape
    if axis is None:
        return int(np.prod(shp, dtype=np.int64))
    axes = axis if isinstance(axis, tuple) else (axis,)
    return int(np.prod([shp[a] for a in axes], dtype=np.int64))


def _var_rule(
    trace: JVPTrace,
    x: Boxed,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Array | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    **_: Any,
) -> JVPTracer:
    """``var`` is *not* linear, so it cannot reuse ``_reduce_for``. Re-express it from
    primitives -- ``var(x) = sum((x - mean(x))**2) / (n - ddof)`` -- entirely via ``bind``
    over the incoming :class:`JVPTracer`, so each sub-primitive (sub / mean / square /
    sum / div) picks up its own jvp rule and the tangent is computed correctly (and
    composes/nests)."""
    t = trace._raise(x)
    n = _reduced_count(t.primal, axis)
    centered = bind(ops.d_sub, t, bind(ops.d_mean, t, axis=axis, keepdims=True))
    sq = bind(ops.d_mul, centered, centered)
    out_t = bind(
        ops.d_div, bind(ops.d_sum, sq, axis=axis, keepdims=keepdims), float(n - ddof)
    )
    return cast(JVPTracer, out_t)


def _std_rule(
    trace: JVPTrace,
    x: Boxed,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Array | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    **_: Any,
) -> JVPTracer:
    """``std = sqrt(var)``; re-expressed via ``bind`` so the chain rule applies."""
    v = _var_rule(trace, x, axis=axis, ddof=ddof, keepdims=keepdims)
    return cast(JVPTracer, bind(ops.d_sqrt, v))


def _reduce_select_for(prim: Prim) -> Rule:
    """max / min reduction: route the tangent through the selected element(s) (split on
    ties, matching the reverse-mode rule), via a normalized argmax mask."""

    def rule(
        trace: JVPTrace,
        x: Boxed,
        axis: Axis = None,
        keepdims: bool = False,
        **kw: Any,
    ) -> JVPTracer:
        t = trace._raise(x)
        primal_out = bind(prim, t.primal, axis=axis, keepdims=keepdims, **kw)
        kept = bind(prim, t.primal, axis=axis, keepdims=True, **kw)
        mask = bind(ops.d_eq, t.primal, kept)  # boolean: the selected positions
        weighted = bind(ops.d_where, mask, t.tangent, _zeros_like(t.tangent))
        count = bind(ops.d_sum, _as_float(mask), axis=axis, keepdims=True)
        # keepdims=True sum of the masked tangents, normalized by the tie count: the
        # average input tangent over the selected position(s).
        sel = bind(
            ops.d_div, bind(ops.d_sum, weighted, axis=axis, keepdims=True), count
        )
        if keepdims:
            tangent_out: Boxed = sel
        else:
            # Drop the kept (size-1) reduced axes so the tangent matches the
            # non-keepdims primal output shape. The input's logical shape comes from the
            # tracer's ``aval`` (works at any nesting level); ``d_reshape`` rides
            # ``bind`` so it recurses correctly.
            in_shape = cast("tuple[int, ...]", t.shape)
            if axis is None:
                out_shape: tuple[int, ...] = ()
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = tuple(a % len(in_shape) for a in axes)
                out_shape = tuple(s for i, s in enumerate(in_shape) if i not in axes)
            tangent_out = bind(ops.d_reshape, sel, out_shape)
        return _result(trace, primal_out, tangent_out)

    return rule


def _as_float(mask: Boxed) -> Boxed:
    """Cast a boolean mask to the working dtype so it can be summed / multiplied."""
    return bind(ops.d_add, mask, 0.0)


# -- comparisons: a boolean mask, not differentiable; tangent is zero -------
def _compare_for(prim: Prim) -> Rule:
    """A comparison yields a non-differentiable boolean array. The result carries no
    tangent, so return the raw primal (it leaves the level as a plain mask)."""

    def rule(trace: JVPTrace, a: Boxed, b: Boxed) -> Boxed:
        ta, tb = trace._raise(a), trace._raise(b)
        return bind(prim, ta.primal, tb.primal)

    return rule


# ---------------------------------------------------------------------------
# The smooth-unary local derivatives now live once, in ``ops._UNARY_DERIV`` (both the
# forward jvp and the reverse vjp build their rules from it). Only ``abs`` stays here:
# its derivative is a value-dependent ``sign`` mask, not a ``bind``-expression.
# ---------------------------------------------------------------------------
def _d_abs(x: Boxed) -> Boxed:
    # ``np.sign`` has no differentiable primitive (it is piecewise constant), so the
    # local derivative of ``abs`` is computed on the primal value directly. ``x`` here is
    # the primal (a Var/array or a lower-level tracer); pull its value to host.
    return _xp().sign(np.asarray(cast(Any, _value(cast(Any, x)))))


# ---------------------------------------------------------------------------
# Small predicates.
# ---------------------------------------------------------------------------
def _is_zero(t: Boxed) -> bool:
    """True when a tangent is statically the zero array we lift constants with (so the
    exponent-tangent term of ``pow`` can be skipped for a constant exponent)."""
    if isinstance(t, (Var, Tracer)):
        return False
    try:
        return bool(np.all(np.asarray(cast(Any, t)) == 0))
    except Exception:  # pragma: no cover - defensive
        return False


def _process_unknown(prim: Prim) -> "NoReturn":
    raise NotImplementedError(
        f"jvp: no forward-mode rule for {getattr(prim, '__name__', prim)!r}; "
        "cannot differentiate it in forward mode."
    )


# ---------------------------------------------------------------------------
# Rule registry, keyed by primitive.
# ---------------------------------------------------------------------------
def _build_jvp_for() -> dict[Prim, Rule]:
    # The smooth-unary local derivatives are the *shared* ``ops._UNARY_DERIV`` table (the
    # reverse rules use the same one), so ``1 - tanh²`` etc. is written once. ``abs`` is
    # special (its derivative is a value-dependent ``sign`` mask, not a bind-expression).
    unary_deriv = {**ops._UNARY_DERIV, ops.d_abs: _d_abs}
    jvp_for: dict[Prim, Rule] = {
        prim: _unary_for(deriv)(prim) for prim, deriv in unary_deriv.items()
    }
    jvp_for.update(
        {
            # linear operator primitives: tangent flows through the same op.
            ops.d_add: _linear_for(ops.d_add),
            ops.d_sub: _linear_for(ops.d_sub),
            ops.d_neg: _linear_for(ops.d_neg),
            # nonlinear binary.
            ops.d_mul: _mul_rule,
            ops.d_gated_act: _gated_act_rule,
            ops.d_softmax: _softmax_rule,
            ops.d_logsumexp: _logsumexp_rule,
            ops.d_div: _div_rule,
            ops.d_mod: _mod_rule,
            ops.d_pow: _pow_rule,
            ops._matmul: _matmul_rule,
            ops.d_dot: ops.contraction_transform_rule(ops.d_dot),
            ops.d_inner: ops.contraction_transform_rule(ops.d_inner),
            ops.d_tensordot: ops.contraction_transform_rule(ops.d_tensordot),
            ops.d_moveaxis: ops._transpose_lowering_transform(ops.moveaxis_perm),
            ops.d_swapaxes: ops._transpose_lowering_transform(ops.swapaxes_perm),
            ops.d_rollaxis: ops._transpose_lowering_transform(ops.rollaxis_perm),
            ops.d_tril: ops._tri_lowering_transform(np.tril),
            ops.d_triu: ops._tri_lowering_transform(np.triu),
            ops.d_roll: _roll_rule,
            ops.d_pad: _pad_rule,
            ops.d_repeat: _repeat_rule,
            ops.d_tile: _tile_rule,
            ops.d_ravel: ops._reshape_lowering_transform(ops.ravel_shape),
            ops.d_squeeze: ops._reshape_lowering_transform(ops.squeeze_shape),
            ops.d_atleast_1d: ops._reshape_lowering_transform(ops.atleast_1d_shape),
            ops.d_atleast_2d: ops._reshape_lowering_transform(ops.atleast_2d_shape),
            ops.d_atleast_3d: ops._reshape_lowering_transform(ops.atleast_3d_shape),
            ops.d_einsum: _einsum_rule,
            # cumsum is linear: the tangent is the cumsum of the tangent.
            ops.d_cumsum: _linear_for(ops.d_cumsum),
            # comparisons: zero tangent (a plain boolean mask leaves the level).
            ops.d_lt: _compare_for(ops.d_lt),
            ops.d_le: _compare_for(ops.d_le),
            ops.d_gt: _compare_for(ops.d_gt),
            ops.d_ge: _compare_for(ops.d_ge),
            ops.d_eq: _compare_for(ops.d_eq),
            ops.d_ne: _compare_for(ops.d_ne),
            # gather.
            ops.d_getitem: _getitem_rule,
            # selection.
            ops.d_maximum: _select_for(ops.d_maximum),
            ops.d_minimum: _select_for(ops.d_minimum),
            ops.d_where: _where_rule,
            ops.d_clip: _clip_rule,
            # reductions: sum/mean/var/std are (composed-)linear; max/min route the
            # tangent through the selected element.
            ops.d_sum: _reduce_for(ops.d_sum),
            ops.d_prod: _prod_rule,
            ops.d_mean: _reduce_for(ops.d_mean),
            ops.d_var: _var_rule,
            ops.d_std: _std_rule,
            ops.d_max: _reduce_select_for(ops.d_max),
            ops.d_min: _reduce_select_for(ops.d_min),
            # shape / structure: all linear.
            ops.d_transpose: _linear_struct_for(ops.d_transpose),
            ops.d_reshape: _linear_struct_for(ops.d_reshape),
            ops.d_broadcast_to: _linear_struct_for(ops.d_broadcast_to),
            ops.d_expand_dims: _linear_struct_for(ops.d_expand_dims),
            ops.d_concatenate: _linear_seq_for(ops.d_concatenate),
            ops.d_stack: _linear_seq_for(ops.d_stack),
            ops.d_vstack: _linear_seq_for(ops.d_vstack),
            ops.d_hstack: _linear_seq_for(ops.d_hstack),
            ops.d_column_stack: _linear_seq_for(ops.d_column_stack),
            ops.d_dstack: _linear_seq_for(ops.d_dstack),
        }
    )
    return jvp_for


# ``_JVP_FOR`` is keyed by *primitive* (incl. the operator primitives), consulted by
# ``JVPTrace.process_primitive``. ``_JVP`` denormalizes it to numpy-callable keys so the
# coverage-parity test (``set(_JVP) == set(ops._INTERCEPT)``) holds, exactly like
# ``batching._BATCH``.
_JVP_FOR: dict[Prim, Rule] = _build_jvp_for()
# The internal reverse-pass scatter primitive (``ops._scatter``) is not in ``_RULES`` (no
# numpy callable maps to it), so it adds no ``_JVP`` coverage key; it gets a jvp rule here
# directly so the differentiable gather VJP composes with a live ``jvp``.
_JVP_FOR[ops._scatter] = ops._jvp_scatter
_JVP: dict[Prim, Rule] = {
    fn: _JVP_FOR[prim] for prim, fns in ops._RULES.items() for fn in fns
}

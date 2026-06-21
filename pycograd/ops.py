# -*- coding: utf-8 -*-
"""Differentiable primitives (vector-Jacobian-product rules).

Each ``d_*`` computes the primal on ``.value`` with the *active* array module
(``xp = _xp()`` -- numpy by default, cupy under ``device("cupy")``) and wires the local
derivative; only host-side metadata (shape counts, split sizes, axis ``argsort``) stays
on numpy. ``_RULES`` maps each primitive to the numpy/math callables that should
route to it; ``_INTERCEPT`` is the flat lookup the tracer (and the ``Weight``
proxy's numpy-protocol handlers) use to swap a numpy/math function for its
differentiable version.

Also here: the pure-Python warn machinery (``AutodiffWarning`` / ``_warn_wrapper``)
for when a ``Var`` reaches a function we have no rule for. None of this depends on
pyccolo; the pyccolo seam lives in :mod:`pycograd.tracer`.
"""
from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, cast

import numpy as np

from pycograd._typing import (
    Array,
    Axis,
    BindArg,
    Boxed,
    DTypeLike,
    Index,
    Operand,
    Prim,
    Shape,
)
from pycograd.backends import current_backend
from pycograd.tensor import Var, _lift, _record_vjp, _unbroadcast, _xp

if TYPE_CHECKING:
    from pycograd.forward import JVPTrace, JVPTracer


# ---------------------------------------------------------------------------
# Elementwise unary.
# ---------------------------------------------------------------------------
def d_exp(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    v = xp.exp(x.value)
    return x._unary(v, lambda a, g: g * v, d_exp)


def d_log(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.log(x.value), lambda a, g: g / a, d_log)


def d_sin(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.sin(x.value), lambda a, g: g * xp.cos(a), d_sin)


def d_cos(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.cos(x.value), lambda a, g: -g * xp.sin(a), d_cos)


def d_tanh(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    v = xp.tanh(x.value)
    return x._unary(v, lambda a, g: g * (1 - v * v), d_tanh)


def d_sqrt(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    v = xp.sqrt(x.value)
    return x._unary(v, lambda a, g: g / (2 * v), d_sqrt)


def d_abs(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.abs(x.value), lambda a, g: g * xp.sign(a), d_abs)


def d_square(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.square(x.value), lambda a, g: g * 2 * a, d_square)


def d_sinh(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.sinh(x.value), lambda a, g: g * xp.cosh(a), d_sinh)


def d_cosh(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.cosh(x.value), lambda a, g: g * xp.sinh(a), d_cosh)


def d_arctan(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.arctan(x.value), lambda a, g: g / (1 + a * a), d_arctan)


def d_log1p(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.log1p(x.value), lambda a, g: g / (1 + a), d_log1p)


def d_expm1(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.expm1(x.value), lambda a, g: g * xp.exp(a), d_expm1)


def d_reciprocal(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.reciprocal(x.value), lambda a, g: -g / (a * a), d_reciprocal)


# ---------------------------------------------------------------------------
# Elementwise binary / selection.
# ---------------------------------------------------------------------------
def _elementwise_max(
    a: Operand, b: Operand, pick_a: Callable[..., Array], prim: Prim
) -> Var:
    a, b = _lift(a), _lift(b)
    out = Var(pick_a(a.value, b.value), _parents=(a, b))

    def _backward() -> None:
        mask = (a.value == out.value).astype(float)
        a.grad = a.grad + _unbroadcast(out.grad * mask, a.value.shape)
        b.grad = b.grad + _unbroadcast(out.grad * (1 - mask), b.value.shape)

    out._backward = _backward
    _record_vjp(out, prim, (a, b), {"out_value": out.value})
    return out


def d_maximum(a: Operand, b: Operand) -> Var:
    return _elementwise_max(a, b, _xp().maximum, d_maximum)


def d_minimum(a: Operand, b: Operand) -> Var:
    return _elementwise_max(a, b, _xp().minimum, d_minimum)


def d_clip(
    x: Operand, a_min: Operand | None = None, a_max: Operand | None = None
) -> Var:
    out = _lift(x)
    if a_min is not None:
        out = d_maximum(out, a_min)
    if a_max is not None:
        out = d_minimum(out, a_max)
    return out


def d_where(cond: Array, a: Operand, b: Operand) -> Var:
    a, b, xp = _lift(a), _lift(b), _xp()
    cond = xp.asarray(cond)
    out = Var(xp.where(cond, a.value, b.value), _parents=(a, b))

    def _backward() -> None:
        a.grad = a.grad + _unbroadcast(xp.where(cond, out.grad, 0.0), a.value.shape)
        b.grad = b.grad + _unbroadcast(xp.where(cond, 0.0, out.grad), b.value.shape)

    out._backward = _backward
    _record_vjp(out, d_where, (a, b), {"cond": cond})
    return out


# ---------------------------------------------------------------------------
# Operator primitives.
#
# The Python operators (`+`, `-`, `*`, `/`, unary `-`, `**`, and the comparisons)
# already build the tape via ``Var``'s dunders. These thin ``d_*`` wrappers give the
# trace-level dispatcher (:mod:`pycograd.trace`) a *named primitive* per operator, so
# ``bind(d_add, a, b)`` on base-level values is identical to ``a + b`` -- they simply
# call the operator on the (lifted) operands, reusing ``Var``'s existing backward.
# Registering them in ``_RULES`` with an empty tuple of numpy callables keeps them out
# of ``_INTERCEPT`` (operators are not numpy functions to swap) while still listing them
# as primitives, so ``_BATCH`` / ``_ABSTRACT`` can carry matching rules.
# ---------------------------------------------------------------------------
def d_add(a: Operand, b: Operand) -> Var:
    return _lift(a) + b


def d_sub(a: Operand, b: Operand) -> Var:
    return _lift(a) - b


def d_mul(a: Operand, b: Operand) -> Var:
    return _lift(a) * b


def d_div(a: Operand, b: Operand) -> Var:
    return _lift(a) / b


def d_neg(a: Operand) -> Var:
    return -_lift(a)


def d_pow(a: Operand, b: Operand) -> Var:
    return _lift(a) ** b


def d_lt(a: Operand, b: Operand) -> Array:
    return _lift(a) < b


def d_le(a: Operand, b: Operand) -> Array:
    return _lift(a) <= b


def d_gt(a: Operand, b: Operand) -> Array:
    return _lift(a) > b


def d_ge(a: Operand, b: Operand) -> Array:
    return _lift(a) >= b


def d_eq(a: Operand, b: Operand) -> Array:
    return _lift(a) == b


def d_ne(a: Operand, b: Operand) -> Array:
    return _lift(a) != b


def d_getitem(x: Operand, key: Index) -> Var:
    return _lift(x)[key]


# ---------------------------------------------------------------------------
# Linear algebra.
# ---------------------------------------------------------------------------
def _matmul_grads(a: Array, b: Array, g: Array) -> tuple[Array, Array]:
    xp = _xp()
    if a.ndim == 1 and b.ndim == 1:
        return g * b, g * a
    if a.ndim == 2 and b.ndim == 1:
        return xp.outer(g, b), a.T @ g
    if a.ndim == 1 and b.ndim == 2:
        return b @ g, xp.outer(a, g)
    return g @ b.swapaxes(-1, -2), a.swapaxes(-1, -2) @ g


def _matmul(a: Operand, b: Operand) -> Var:
    a, b = _lift(a), _lift(b)
    try:
        primal = a.value @ b.value
    except ValueError as e:
        from pycograd.shapes import ShapeError, _shape_context

        raise ShapeError(_shape_context("matmul", a.value.shape, b.value.shape)) from e
    out = Var(primal, _parents=(a, b))

    def _backward() -> None:
        da, db = _matmul_grads(a.value, b.value, out.grad)
        a.grad = a.grad + _unbroadcast(da, a.value.shape)
        b.grad = b.grad + _unbroadcast(db, b.value.shape)

    out._backward = _backward
    _record_vjp(out, _matmul, (a, b))
    return out


# ---------------------------------------------------------------------------
# Reductions.
# ---------------------------------------------------------------------------
def d_sum(x: Operand, axis: Axis = None, keepdims: bool = False) -> Var:
    x, xp = _lift(x), _xp()
    out = Var(x.value.sum(axis=axis, keepdims=keepdims), _parents=(x,))

    def _backward() -> None:
        g = out.grad
        if axis is not None and not keepdims:
            g = xp.expand_dims(g, axis)
        x.grad = x.grad + xp.broadcast_to(g, x.value.shape)

    out._backward = _backward
    _record_vjp(out, d_sum, (x,), {"axis": axis, "keepdims": keepdims})
    return out


def _reduced_count(x: Var, axis: Axis) -> int:
    if axis is None:
        return int(x.value.size)
    axes = axis if isinstance(axis, tuple) else (axis,)
    return int(np.prod([x.value.shape[a] for a in axes]))


def d_mean(x: Operand, axis: Axis = None, keepdims: bool = False) -> Var:
    x = _lift(x)
    return d_sum(x, axis=axis, keepdims=keepdims) / _reduced_count(x, axis)


def d_var(
    x: Operand,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Array | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    **_: Any,
) -> Var:
    # Composed from mean/centering/square -- gradient flows for free; the numpy
    # signature order (a, axis, dtype, out, ddof, keepdims) is mirrored so
    # positional intercepted calls line up. dtype/out are ignored.
    x = _lift(x)
    centered = x - d_mean(x, axis=axis, keepdims=True)
    n = _reduced_count(x, axis)
    return d_sum(centered * centered, axis=axis, keepdims=keepdims) / (n - ddof)


def d_std(
    x: Operand,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Array | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    **_: Any,
) -> Var:
    return d_var(x, axis=axis, ddof=ddof, keepdims=keepdims) ** 0.5


def _reduce_select(
    x: Operand,
    axis: Axis,
    keepdims: bool,
    reducer: Callable[..., Array],
    prim: Prim,
) -> Var:
    # max/min: the gradient flows only to the selected element(s), split on ties.
    x, xp = _lift(x), _xp()
    kept = reducer(x.value, axis=axis, keepdims=True)
    out = Var(reducer(x.value, axis=axis, keepdims=keepdims), _parents=(x,))

    def _backward() -> None:
        g = out.grad
        if axis is not None and not keepdims:
            g = xp.expand_dims(g, axis)
        mask = (x.value == kept).astype(float)
        mask /= mask.sum(axis=axis, keepdims=True)
        x.grad = x.grad + mask * g

    out._backward = _backward
    _record_vjp(
        out,
        prim,
        (x,),
        {"axis": axis, "keepdims": keepdims, "reducer": reducer},
    )
    return out


def d_max(x: Operand, axis: Axis = None, keepdims: bool = False) -> Var:
    return _reduce_select(x, axis, keepdims, _xp().max, d_max)


def d_min(x: Operand, axis: Axis = None, keepdims: bool = False) -> Var:
    return _reduce_select(x, axis, keepdims, _xp().min, d_min)


# ---------------------------------------------------------------------------
# Shape / structure.
# ---------------------------------------------------------------------------
def d_concatenate(seq: Sequence[Operand], axis: int = 0) -> Var:
    parts, xp = [_lift(s) for s in seq], _xp()
    try:
        primal = xp.concatenate([p.value for p in parts], axis=axis)
    except ValueError as e:
        from pycograd.shapes import ShapeError, _shape_context

        raise ShapeError(
            _shape_context("concatenate", *(p.value.shape for p in parts))
            + f" along axis {axis}"
        ) from e
    out = Var(primal, _parents=tuple(parts))

    def _backward() -> None:
        # split sizes are host-side ints (shapes), so np.cumsum stays on numpy; the
        # split itself slices the device-resident gradient and uses the active module.
        splits = np.cumsum([p.value.shape[axis] for p in parts])[:-1]
        for part, gpart in zip(parts, xp.split(out.grad, splits.tolist(), axis=axis)):
            part.grad = part.grad + gpart

    out._backward = _backward
    _record_vjp(out, d_concatenate, tuple(parts), {"axis": axis})
    return out


def d_transpose(x: Operand, axes: tuple[int, ...] | None = None) -> Var:
    x, xp = _lift(x), _xp()
    out = Var(xp.transpose(x.value, axes), _parents=(x,))

    def _backward() -> None:
        if axes is None:
            x.grad = x.grad + xp.transpose(out.grad)
        else:
            # np.argsort over the (host-side) axes tuple; the transpose runs on device.
            x.grad = x.grad + xp.transpose(out.grad, tuple(np.argsort(axes)))

    out._backward = _backward
    _record_vjp(out, d_transpose, (x,), {"axes": axes})
    return out


def d_reshape(x: Operand, *shape: Shape) -> Var:
    x, xp = _lift(x), _xp()
    newshape = shape[0] if len(shape) == 1 else shape
    try:
        primal = xp.reshape(x.value, newshape)
    except ValueError as e:
        from pycograd.shapes import ShapeError

        raise ShapeError(
            f"reshape: cannot reshape array of shape {x.value.shape} "
            f"(size {x.value.size}) into {newshape}"
        ) from e
    out = Var(primal, _parents=(x,))

    def _backward() -> None:
        x.grad = x.grad + out.grad.reshape(x.value.shape)

    out._backward = _backward
    _record_vjp(out, d_reshape, (x,))
    return out


def d_broadcast_to(x: Operand, shape: Shape) -> Var:
    """Broadcast ``x`` to ``shape``; the VJP sums the cotangent back over the broadcast
    axes (``_unbroadcast``).

    Introduced as a named primitive so the differentiable reverse pass can express
    ``d_sum``'s gradient (``broadcast_to``) and a differentiable ``_unbroadcast`` with
    ops that themselves ride ``bind`` (and so compose with an enclosing ``jvp``/``grad``).
    The ``shape`` is host-side metadata, constant under differentiation.
    """
    x, xp = _lift(x), _xp()
    target = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
    out = Var(xp.broadcast_to(x.value, target), _parents=(x,))

    def _backward() -> None:
        x.grad = x.grad + _unbroadcast(out.grad, x.value.shape)

    out._backward = _backward
    _record_vjp(out, d_broadcast_to, (x,))
    return out


def d_expand_dims(x: Operand, axis: int) -> Var:
    # A reshape under the hood, so it records ``d_reshape`` (whose VJP reshapes back).
    x = _lift(x)
    pos = axis if axis >= 0 else axis + x.value.ndim + 1
    shape = list(x.value.shape)
    shape.insert(pos, 1)
    return d_reshape(x, tuple(shape))


# The ``*stack`` family is just ``concatenate`` after a shape fix-up, so composing
# the existing differentiable primitives gives correct gradients for free.
def d_stack(seq: Sequence[Operand], axis: int = 0) -> Var:
    # join along a NEW axis: expand each input at ``axis``, then concatenate there.
    return d_concatenate([d_expand_dims(s, axis) for s in seq], axis=axis)


def _atleast_2d_row(x: Operand) -> Var:
    x = _lift(x)
    if x.value.ndim == 0:
        return d_reshape(x, (1, 1))
    if x.value.ndim == 1:
        return d_reshape(x, (1, x.value.shape[0]))
    return x


def d_vstack(seq: Sequence[Operand]) -> Var:
    # row-wise: 1-D inputs become single rows, then concatenate along axis 0.
    return d_concatenate([_atleast_2d_row(s) for s in seq], axis=0)


def d_hstack(seq: Sequence[Operand]) -> Var:
    # column-wise: concatenate along axis 1, except 1-D inputs join along axis 0.
    parts = [_lift(s) for s in seq]
    axis = 0 if all(p.value.ndim <= 1 for p in parts) else 1
    return d_concatenate(parts, axis=axis)


def d_column_stack(seq: Sequence[Operand]) -> Var:
    # 1-D inputs become columns ((n,) -> (n, 1)); then concatenate along axis 1.
    parts = []
    for s in seq:
        p = _lift(s)
        parts.append(d_reshape(p, (p.value.shape[0], 1)) if p.value.ndim == 1 else p)
    return d_concatenate(parts, axis=1)


def _atleast_3d_depth(x: Operand) -> Var:
    x = _lift(x)
    if x.value.ndim == 0:
        return d_reshape(x, (1, 1, 1))
    if x.value.ndim == 1:
        return d_reshape(x, (1, x.value.shape[0], 1))
    if x.value.ndim == 2:
        return d_reshape(x, x.value.shape + (1,))
    return x


def d_dstack(seq: Sequence[Operand]) -> Var:
    # depth-wise: stack along a third axis (after promoting inputs to 3-D).
    return d_concatenate([_atleast_3d_depth(s) for s in seq], axis=2)


# ---------------------------------------------------------------------------
# Interception tables: which numpy/math callables route to which primitive.
# ---------------------------------------------------------------------------
# Only numpy ufuncs / math C-functions need entries here: they bypass our Python
# operator overloads (computing in C, or failing the disabled __array_ufunc__),
# so we must supply an explicit primitive + backward. Builtins like abs / sum /
# min / max are deliberately absent -- they dispatch through dunders (__abs__,
# repeated __add__, __lt__) onto ops that already have backward passes, so the
# tape builds itself and a rule would be redundant.
#
# Each differentiable primitive is listed once alongside every numpy/math
# callable that should route to it (e.g. np.exp and math.exp share d_exp); the
# flat lookup the tracer uses is denormalized from this below.
_RULES: dict[Prim, tuple[Prim, ...]] = {
    # operator primitives -- no numpy callable swaps to these (operators are not
    # numpy functions), so their tuples are empty: they contribute no ``_INTERCEPT``
    # key (coverage parity with the numpy-keyed tables is preserved) but are still
    # listed as primitives so ``_BATCH`` / ``_ABSTRACT`` can register rules for them.
    d_add: (),
    d_sub: (),
    d_mul: (),
    d_div: (),
    d_neg: (),
    d_pow: (),
    d_lt: (),
    d_le: (),
    d_gt: (),
    d_ge: (),
    d_eq: (),
    d_ne: (),
    d_getitem: (),
    # broadcast_to: an internal primitive used by the differentiable reverse pass
    # (``d_sum``'s VJP / a differentiable ``_unbroadcast``). No numpy callable swaps to
    # it, so its tuple is empty -- it adds no ``_INTERCEPT`` key (coverage parity holds)
    # but is still listed so every rule table (``_JVP_FOR`` / ``_RULE_FOR`` / ``_ABSTRACT``
    # / ``_VJP_FOR``) can register a matching rule.
    d_broadcast_to: (),
    # elementwise unary
    d_exp: (np.exp, math.exp),
    d_log: (np.log, math.log),
    d_sin: (np.sin, math.sin),
    d_cos: (np.cos, math.cos),
    d_tanh: (np.tanh, math.tanh),
    d_sqrt: (np.sqrt, math.sqrt),
    d_sinh: (np.sinh, math.sinh),
    d_cosh: (np.cosh, math.cosh),
    d_arctan: (np.arctan, math.atan),
    d_log1p: (np.log1p, math.log1p),
    d_expm1: (np.expm1, math.expm1),
    d_abs: (np.abs,),
    d_square: (np.square,),
    d_reciprocal: (np.reciprocal,),
    # elementwise binary
    d_maximum: (np.maximum,),
    d_minimum: (np.minimum,),
    # selection
    d_where: (np.where,),
    d_clip: (np.clip,),
    # reductions
    d_sum: (np.sum,),
    d_mean: (np.mean,),
    d_var: (np.var,),
    d_std: (np.std,),
    d_max: (np.max, np.amax),
    d_min: (np.min, np.amin),
    # linear algebra / shape / structure
    _matmul: (np.dot, np.matmul),
    d_transpose: (np.transpose,),
    d_reshape: (np.reshape,),
    d_expand_dims: (np.expand_dims,),
    d_concatenate: (np.concatenate,),
    d_stack: (np.stack,),
    d_vstack: (np.vstack,),
    d_hstack: (np.hstack,),
    d_column_stack: (np.column_stack,),
    d_dstack: (np.dstack,),
}

_INTERCEPT: dict[Prim, Prim] = {fn: impl for impl, fns in _RULES.items() for fn in fns}


# ---------------------------------------------------------------------------
# Differentiable VJP rules (the higher-order reverse path).
#
# Each rule maps ``(primals, operands, params, upstream_cotangent) -> per-operand
# cotangents`` (aligned with the producing node's ``_parents``):
#   * ``primals`` are the recorded primal operand ``Var``s -- used for host-side shapes and
#     for masks (which are stop-gradient constants computed from primal values).
#   * ``operands`` are the *level-connected* operands (the same ``Var``s, or the
#     ``JVPTracer``s wrapping them when a ``jvp`` is live) -- the values the tape arithmetic
#     must ride so the cotangent graph carries second-order information.
# Every rule builds its cotangents with ``bind``-riding ``d_*`` ops, so the cotangent graph
# is itself differentiable and composes with an enclosing ``jvp``/``grad`` (Phase 1:
# forward-over-reverse Hessians). ``Var.backward`` takes this path only when a higher trace
# level is live; the base level keeps its raw ``.grad`` closures untouched.
#
# Non-smooth primitives (max/min/abs/where/clip) compute a selection mask from the
# *primals* and treat it as a CONSTANT (the ``forward.py`` convention), giving the correct
# a.e. zero second derivative through the kink.
# ---------------------------------------------------------------------------
def _b(prim: Prim, *args: BindArg, **kw: Any) -> Boxed:
    from pycograd.trace import bind

    return bind(prim, *args, **kw)


def _const_like(arr: Array) -> Var:
    """A constant ``Var`` (no tape history) holding ``arr`` -- used for masks/signs that are
    stop-gradient by construction."""
    return Var(arr)


# Local-derivative helpers for the elementwise-unary VJPs, written with ``_b`` (``bind``)
# so each tangent rides the enclosing level. ``primal`` is the level-connected operand;
# ``f'(primal)`` times the upstream cotangent is the operand cotangent.
def _vjp_unary_derivs() -> dict[Prim, Callable[[Boxed], Boxed]]:
    return {
        d_exp: lambda a: _b(d_exp, a),
        d_log: lambda a: _b(d_reciprocal, a),
        d_sin: lambda a: _b(d_cos, a),
        d_cos: lambda a: _b(d_neg, _b(d_sin, a)),
        d_tanh: lambda a: _b(d_sub, 1.0, _b(d_mul, _b(d_tanh, a), _b(d_tanh, a))),
        d_sqrt: lambda a: _b(d_div, 0.5, _b(d_sqrt, a)),
        d_sinh: lambda a: _b(d_cosh, a),
        d_cosh: lambda a: _b(d_sinh, a),
        d_arctan: lambda a: _b(d_reciprocal, _b(d_add, 1.0, _b(d_mul, a, a))),
        d_log1p: lambda a: _b(d_reciprocal, _b(d_add, 1.0, a)),
        d_expm1: lambda a: _b(d_exp, a),
        d_square: lambda a: _b(d_mul, 2.0, a),
        d_reciprocal: lambda a: _b(d_neg, _b(d_reciprocal, _b(d_mul, a, a))),
    }


def _vjp_unary_for(prim: Callable[..., Var]) -> Callable[..., list[Boxed]]:
    derivs = _VJP_UNARY_DERIVS

    def rule(
        primals: tuple[Var, ...],
        operands: tuple[Boxed, ...],
        params: dict[str, Any],
        g: Boxed,
    ) -> list[Boxed]:
        (a,) = operands
        return [_b(d_mul, g, derivs[prim](a))]

    return rule


def _vjp_abs(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # sign(primal) is a piecewise-constant derivative -> a stop-gradient constant.
    (p,) = primals
    return [_b(d_mul, g, _const_like(_xp().sign(p.value)))]


def _vjp_add(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    return [g, g]


def _vjp_sub(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    return [g, _b(d_neg, g)]


def _vjp_neg(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    return [_b(d_neg, g)]


def _vjp_mul(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    a, b = operands
    return [_b(d_mul, g, b), _b(d_mul, g, a)]


def _vjp_div(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    a, b = operands
    return [_b(d_div, g, b), _b(d_neg, _b(d_div, _b(d_mul, g, a), _b(d_mul, b, b)))]


def _vjp_pow(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # Only the base-operand cotangent: ``Var.__pow__`` rides ``d_pow`` only for a constant
    # exponent (a ``Var`` exponent is lowered to exp/log), so the exponent has no gradient.
    # The exponent stays a *raw scalar/array* (not a ``Var``) so ``d_pow`` keeps the power
    # path -- a ``Var`` exponent would trigger ``exp(b*log(a))`` (nan for a negative base).
    a, _b_operand = operands
    pa, pb = primals
    p = pb.value
    ga = _b(d_mul, g, _b(d_mul, p, _b(d_pow, a, p - 1)))
    return [ga, None]


def _swap_last2(x: Boxed, ndim: int) -> Boxed:
    axes = tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
    return _b(d_transpose, x, axes)


def _vjp_matmul(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    a, b = operands
    pa, pb = primals
    na, nb = pa.value.ndim, pb.value.ndim
    if na == 1 and nb == 1:  # inner product: g is a scalar
        return [_b(d_mul, g, b), _b(d_mul, g, a)]
    if na == 2 and nb == 1:  # da = outer(g, b); db = a.T @ g
        ga = _b(
            _matmul,
            _b(d_reshape, g, (pa.value.shape[0], 1)),
            _b(d_reshape, b, (1, pb.value.shape[0])),
        )
        return [ga, _b(_matmul, _b(d_transpose, a), g)]
    if na == 1 and nb == 2:  # da = b @ g ; db = outer(a, g)
        gb = _b(
            _matmul,
            _b(d_reshape, a, (pa.value.shape[0], 1)),
            _b(d_reshape, g, (1, pb.value.shape[1])),
        )
        return [_b(_matmul, b, g), gb]
    # batched / 2-D @ 2-D: da = g @ bᵀ, db = aᵀ @ g (transposing the last two axes).
    return [_b(_matmul, g, _swap_last2(b, nb)), _b(_matmul, _swap_last2(a, na), g)]


def _vjp_getitem(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # The reverse VJP of a gather is a scatter-add into a zero buffer; that scatter (the
    # internal ``_scatter`` primitive) is linear in ``g`` and rides ``bind``, and its *own*
    # VJP is another gather (no new scatter primitive needed) -- so a second-order pass
    # differentiates it too.
    (p,) = primals
    key = params["key"]
    return [_b(_scatter, g, key, p.value.shape, p.value.dtype)]


def _scatter(g: Operand, key: Index, shape: tuple[int, ...], dtype: DTypeLike) -> Var:
    """Scatter-add ``g`` into a zero buffer of ``shape`` at ``key`` (the reverse VJP of a
    gather). An internal primitive: linear in ``g``, with a JVP rule (``d scatter(g) =
    scatter(dg)``) and a differentiable VJP (a gather at ``key``), so it composes with a
    live ``jvp``/``grad``. Not registered in ``_RULES`` (no numpy callable maps to it).
    """
    g = _lift(g)
    xp = _xp()
    buf = xp.zeros(shape, dtype=dtype)
    current_backend().scatter_add(buf, key, g.value)
    out = Var(buf, _parents=(g,))

    def _backward() -> None:
        g.grad = g.grad + out.grad[cast(Index, key)]  # gather cotangent at scatter key

    out._backward = _backward
    _record_vjp(out, _scatter, (g,), {"key": key})
    return out


def _jvp_scatter(
    trace: "JVPTrace",
    g: Boxed,
    key: Index,
    shape: tuple[int, ...],
    dtype: DTypeLike,
) -> "JVPTracer":
    """JVP of the internal scatter: it is linear, so the tangent scatters the same way."""
    from pycograd.trace import bind

    t = trace._raise(g)
    primal_out = bind(_scatter, t.primal, key, shape, dtype)
    tangent_out = bind(_scatter, t.tangent, key, shape, dtype)
    from pycograd.forward import JVPTracer

    return JVPTracer(trace, primal_out, tangent_out)


def _vjp_scatter(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # VJP of scatter-add is a gather of the cotangent at the same key.
    return [_b(d_getitem, g, params["key"])]


def _expand_dims_multi(g: Boxed, axis: int | tuple[int, ...]) -> Boxed:
    axes = axis if isinstance(axis, tuple) else (axis,)
    for ax in sorted(axes):
        g = _b(d_expand_dims, g, ax)
    return g


def _vjp_sum(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    (p,) = primals
    axis = params.get("axis")
    keepdims = params.get("keepdims", False)
    if axis is not None and not keepdims:
        g = _expand_dims_multi(g, axis)
    return [_b(d_broadcast_to, g, p.value.shape)]


def _vjp_reshape(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    (p,) = primals
    return [_b(d_reshape, g, p.value.shape)]


def _vjp_broadcast_to(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    from pycograd.tensor import _d_unbroadcast

    (p,) = primals
    return [_d_unbroadcast(g, p.value.shape)]


def _vjp_transpose(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    axes = params.get("axes")
    if axes is None:
        return [_b(d_transpose, g)]
    return [_b(d_transpose, g, tuple(int(a) for a in np.argsort(axes)))]


def _vjp_concatenate(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    axis = params.get("axis", 0)
    ndim = primals[0].value.ndim
    ax = axis % ndim
    out: list[Boxed] = []
    start = 0
    for p in primals:
        end = start + p.value.shape[ax]
        key = tuple(slice(start, end) if i == ax else slice(None) for i in range(ndim))
        out.append(_b(d_getitem, g, key))
        start = end
    return out


def _vjp_reduce_select(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    (p,) = primals
    axis = params.get("axis")
    keepdims = params.get("keepdims", False)
    reducer = params["reducer"]
    kept = reducer(p.value, axis=axis, keepdims=True)
    mask = (p.value == kept).astype(p.value.dtype)
    mask = mask / mask.sum(axis=axis, keepdims=True)
    if axis is not None and not keepdims:
        g = _expand_dims_multi(g, axis)
    return [_b(d_mul, g, _const_like(mask))]


def _vjp_select(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # max/min of two operands: mask (from primals) is a stop-gradient constant.
    pa, _pb = primals
    out_val = params["out_value"]
    mask = (pa.value == out_val).astype(pa.value.dtype)
    return [_b(d_mul, g, _const_like(mask)), _b(d_mul, g, _const_like(1.0 - mask))]


def _vjp_where(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    pa, _pb = primals
    cond = params["cond"]
    xp = _xp()
    cmask = _const_like(xp.asarray(cond).astype(pa.value.dtype))
    omask = _const_like((~xp.asarray(cond).astype(bool)).astype(pa.value.dtype))
    return [_b(d_mul, g, cmask), _b(d_mul, g, omask)]


def _build_vjp_for() -> dict[Prim, Callable[..., list[Boxed]]]:
    vjp_for: dict[Prim, Callable[..., list[Boxed]]] = {
        prim: _vjp_unary_for(cast(Callable[..., Var], prim))
        for prim in _VJP_UNARY_DERIVS
    }
    vjp_for.update(
        {
            d_abs: _vjp_abs,
            d_add: _vjp_add,
            d_sub: _vjp_sub,
            d_neg: _vjp_neg,
            d_mul: _vjp_mul,
            d_div: _vjp_div,
            d_pow: _vjp_pow,
            _matmul: _vjp_matmul,
            d_getitem: _vjp_getitem,
            _scatter: _vjp_scatter,
            d_sum: _vjp_sum,
            d_reshape: _vjp_reshape,
            d_broadcast_to: _vjp_broadcast_to,
            d_expand_dims: _vjp_reshape,  # expand_dims is a reshape; the VJP reshapes back
            d_transpose: _vjp_transpose,
            d_concatenate: _vjp_concatenate,
            d_max: _vjp_reduce_select,
            d_min: _vjp_reduce_select,
            d_maximum: _vjp_select,
            d_minimum: _vjp_select,
            d_where: _vjp_where,
        }
    )
    return vjp_for


_VJP_UNARY_DERIVS: dict[Prim, Callable[[Boxed], Boxed]] = _vjp_unary_derivs()
_VJP_FOR: dict[Prim, Callable[..., list[Boxed]]] = _build_vjp_for()


def _vjp_contributions(v: Var, g: Boxed, operands: tuple[Boxed, ...]) -> list[Boxed]:
    """Per-parent cotangent contributions for node ``v`` given its cotangent ``g``.

    ``operands`` are the level-connected operands (a ``JVPTracer`` wrapping each primal
    when a ``jvp`` is live, else the primal ``Var``). Looks up the producing primitive's
    rule in ``_VJP_FOR``; a node with no recorded primitive contributes nothing. Returns a
    list aligned with ``v._parents``."""
    prim = v._vjp_prim
    if prim is None:
        return [None] * len(v._parents)
    rule = _VJP_FOR.get(prim)
    if rule is None:
        raise NotImplementedError(
            f"higher-order reverse: no differentiable VJP rule for "
            f"{getattr(prim, '__name__', prim)!r}"
        )
    return rule(v._vjp_operands, operands, v._vjp_params, g)


# ---------------------------------------------------------------------------
# Warn machinery: a Var reaching a function we have no rule for.
# ---------------------------------------------------------------------------
class AutodiffWarning(UserWarning):
    """Emitted when a ``Var`` flows into a numpy/math function we cannot differentiate."""


def _contains_var(values: Iterable[object]) -> bool:
    for v in values:
        if isinstance(v, Var):
            return True
        if isinstance(v, (list, tuple)) and any(isinstance(x, Var) for x in v):
            return True
    return False


def _is_mathy(func: Prim) -> bool:
    # numpy ufuncs / numpy functions / the math module are the calls that bypass
    # our operator overloading and silently drop gradients; builtins like abs/sum
    # work through dunder dispatch and must NOT be flagged.
    if isinstance(func, np.ufunc):
        return True
    module = getattr(func, "__module__", None) or ""
    return module == "math" or module.startswith("numpy")


_WRAPPERS: dict[Prim, Prim] = {}


def _warn_wrapper(func: Prim) -> Prim:
    wrapper = _WRAPPERS.get(func)
    if wrapper is not None:
        return wrapper
    name = getattr(func, "__name__", repr(func))

    def _wrapped(*args: object, **kwargs: object) -> object:
        if _contains_var(args) or _contains_var(kwargs.values()):
            warnings.warn(
                f"autodiff: no differentiation rule for {name!r}; "
                "the gradient will not flow through this call",
                AutodiffWarning,
                stacklevel=2,
            )
        return func(*args, **kwargs)

    _WRAPPERS[func] = _wrapped
    return _wrapped

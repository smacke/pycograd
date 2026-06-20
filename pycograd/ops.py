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
from typing import Callable, Iterable, Sequence

import numpy as np

from pycograd._typing import Array, Axis, Operand, Shape
from pycograd.tensor import Var, _lift, _unbroadcast, _xp


# ---------------------------------------------------------------------------
# Elementwise unary.
# ---------------------------------------------------------------------------
def d_exp(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    v = xp.exp(x.value)
    return x._unary(v, lambda a, g: g * v)


def d_log(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.log(x.value), lambda a, g: g / a)


def d_sin(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.sin(x.value), lambda a, g: g * xp.cos(a))


def d_cos(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.cos(x.value), lambda a, g: -g * xp.sin(a))


def d_tanh(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    v = xp.tanh(x.value)
    return x._unary(v, lambda a, g: g * (1 - v * v))


def d_sqrt(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    v = xp.sqrt(x.value)
    return x._unary(v, lambda a, g: g / (2 * v))


def d_abs(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.abs(x.value), lambda a, g: g * xp.sign(a))


def d_square(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.square(x.value), lambda a, g: g * 2 * a)


def d_sinh(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.sinh(x.value), lambda a, g: g * xp.cosh(a))


def d_cosh(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.cosh(x.value), lambda a, g: g * xp.sinh(a))


def d_arctan(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.arctan(x.value), lambda a, g: g / (1 + a * a))


def d_log1p(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.log1p(x.value), lambda a, g: g / (1 + a))


def d_expm1(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.expm1(x.value), lambda a, g: g * xp.exp(a))


def d_reciprocal(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.reciprocal(x.value), lambda a, g: -g / (a * a))


# ---------------------------------------------------------------------------
# Elementwise binary / selection.
# ---------------------------------------------------------------------------
def _elementwise_max(a: Operand, b: Operand, pick_a: Callable[..., Array]) -> Var:
    a, b = _lift(a), _lift(b)
    out = Var(pick_a(a.value, b.value), _parents=(a, b))

    def _backward() -> None:
        mask = (a.value == out.value).astype(float)
        a.grad = a.grad + _unbroadcast(out.grad * mask, a.value.shape)
        b.grad = b.grad + _unbroadcast(out.grad * (1 - mask), b.value.shape)

    out._backward = _backward
    return out


def d_maximum(a: Operand, b: Operand) -> Var:
    return _elementwise_max(a, b, _xp().maximum)


def d_minimum(a: Operand, b: Operand) -> Var:
    return _elementwise_max(a, b, _xp().minimum)


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


def d_getitem(x: Operand, key: object) -> Var:
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
    dtype: object = None,
    out: object = None,
    ddof: int = 0,
    keepdims: bool = False,
    **_: object,
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
    dtype: object = None,
    out: object = None,
    ddof: int = 0,
    keepdims: bool = False,
    **_: object,
) -> Var:
    return d_var(x, axis=axis, ddof=ddof, keepdims=keepdims) ** 0.5


def _reduce_select(
    x: Operand, axis: Axis, keepdims: bool, reducer: Callable[..., Array]
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
    return out


def d_max(x: Operand, axis: Axis = None, keepdims: bool = False) -> Var:
    return _reduce_select(x, axis, keepdims, _xp().max)


def d_min(x: Operand, axis: Axis = None, keepdims: bool = False) -> Var:
    return _reduce_select(x, axis, keepdims, _xp().min)


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
    return out


def d_expand_dims(x: Operand, axis: int) -> Var:
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
_RULES: dict[Callable[..., object], tuple[object, ...]] = {
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

_INTERCEPT: dict[object, Callable[..., object]] = {
    fn: impl for impl, fns in _RULES.items() for fn in fns
}


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


def _is_mathy(func: Callable[..., object]) -> bool:
    # numpy ufuncs / numpy functions / the math module are the calls that bypass
    # our operator overloading and silently drop gradients; builtins like abs/sum
    # work through dunder dispatch and must NOT be flagged.
    if isinstance(func, np.ufunc):
        return True
    module = getattr(func, "__module__", None) or ""
    return module == "math" or module.startswith("numpy")


_WRAPPERS: dict[Callable[..., object], Callable[..., object]] = {}


def _warn_wrapper(func: Callable[..., object]) -> Callable[..., object]:
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

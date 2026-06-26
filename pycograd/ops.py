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
import operator
import warnings
from collections import Counter
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
    Rule,
    Shape,
)
from pycograd.backends import current_backend
from pycograd.dtypes import conj_if_complex, resolve_dtype
from pycograd.tensor import (
    Var,
    _accumulate,
    _d_unbroadcast,
    _lift,
    _record_vjp,
    _unbroadcast,
    _xp,
)

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


def d_sigmoid(x: Operand) -> Var:
    # Logistic sigmoid 1 / (1 + exp(-x)). A fused primitive rather than a numpy
    # composition: there is no ``np.sigmoid`` to intercept, so this is tape-only
    # (no ``_RULES`` entry) and the local derivative reuses the cached output --
    # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)).
    #
    # Because it has no numpy name, user code calls ``d_sigmoid`` directly rather than
    # reaching it through call interception. Under an enclosing transform the operand is
    # a higher-level tracer (a jvp/vmap ``Tracer``, not a base ``Var``); dispatch it
    # through ``bind`` so the registered jvp/vmap/abstract rule runs, exactly as an
    # intercepted ``np.*`` call would. A base ``Var``/array falls through to the kernel.
    from pycograd.trace import Tracer, bind

    if isinstance(x, Tracer):
        return cast(Var, bind(d_sigmoid, x))
    x, xp = _lift(x), _xp()
    v = xp.reciprocal(1.0 + xp.exp(-x.value))
    return x._unary(v, lambda a, g: g * v * (1 - v), d_sigmoid)


def _nonholo_unary(
    x: Operand,
    value: Array,
    adjoint: Callable[[Array, Array], Array],
    prim: Prim,
    out_dtype: DTypeLike | None = None,
) -> Var:
    """Build a non-holomorphic unary op (conj/real/imag/angle/abs).

    Unlike ``Var._unary``, the eager backward applies ``adjoint(z, g)`` *directly* -- it is
    already the real-adjoint contribution under the real inner product on complex tensors,
    so it must NOT go through the holomorphic ``conj_if_complex`` wrap. ``out_dtype`` pins
    the result dtype (real/imag/angle/abs are real-valued even inside a complex ``dtype``
    context, where ``Var`` would otherwise re-cast to the complex working dtype).
    """
    from pycograd.tensor import _accumulate, _unbroadcast

    x = _lift(x)
    out = Var(value, _parents=(x,), dtype=out_dtype)

    def _backward() -> None:
        x.grad = _accumulate(
            x.grad, _unbroadcast(adjoint(x.value, out.grad), x.value.shape)
        )

    out._backward = _backward
    _record_vjp(out, prim, (x,))
    return out


def _real_dtype_of(arr: Array) -> "np.dtype":
    """The real-component dtype of ``arr`` (complex128 -> float64; a real dtype unchanged)."""
    return np.asarray(arr).real.dtype


def d_abs(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    # |z| is non-holomorphic for complex: the real-adjoint is ``g * z/|z|`` (which reduces
    # to ``g * sign(x)`` for real x). Output is real-valued in both cases.
    val = xp.abs(x.value)
    if np.iscomplexobj(x.value):
        return _nonholo_unary(
            x, val, lambda z, g: g * (z / xp.abs(z)), d_abs, _real_dtype_of(x.value)
        )
    return x._unary(val, lambda a, g: g * xp.sign(a), d_abs)


def d_conj(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    # conj: z -> conj(z); real-adjoint of conj is conj (an involution).
    return _nonholo_unary(x, xp.conj(x.value), lambda z, g: xp.conj(g), d_conj)


def d_real(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    # real: z -> Re(z); adjoint embeds the real cotangent back as a complex number.
    return _nonholo_unary(
        x,
        xp.real(x.value),
        lambda z, g: xp.asarray(g, dtype=z.dtype),
        d_real,
        _real_dtype_of(x.value),
    )


def d_imag(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    # imag: z -> Im(z); adjoint of Im is multiply-by-i (real g -> imaginary cotangent).
    return _nonholo_unary(
        x,
        xp.imag(x.value),
        lambda z, g: (1j * g).astype(z.dtype) if np.iscomplexobj(z) else g * 0.0,
        d_imag,
        _real_dtype_of(x.value),
    )


def d_angle(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    # angle: z -> atan2(Im, Re); real-adjoint is ``g * i*z/|z|^2``.
    return _nonholo_unary(
        x,
        xp.angle(x.value),
        lambda z, g: (
            (g * (1j * z) / (xp.abs(z) ** 2)).astype(z.dtype)
            if np.iscomplexobj(z)
            else g * 0.0
        ),
        d_angle,
        _real_dtype_of(x.value),
    )


def d_real_if_close(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    # real_if_close: drop a near-zero imaginary part (else pass through). It is identity on
    # the values, so its real-adjoint is identity on the cotangent.
    return _nonholo_unary(x, xp.real_if_close(x.value), lambda z, g: g, d_real_if_close)


def d_nan_to_num(x: Operand, *args: Any, **kwargs: Any) -> Var:
    # Replace nan -> 0 and +/-inf -> large finite. Differentiable where the input is finite
    # (those pass through ~unchanged); a nan/inf input becomes a constant, so its gradient is
    # zero -- the VJP masks by ``isfinite(x)``.
    x, xp = _lift(x), _xp()
    mask = xp.isfinite(x.value).astype(x.value.dtype)
    out = Var(xp.nan_to_num(x.value, *args, **kwargs), _parents=(x,))

    def _backward() -> None:
        x.grad = _accumulate(x.grad, _unbroadcast(out.grad * mask, x.value.shape))

    out._backward = _backward
    _record_vjp(out, d_nan_to_num, (x,), {"mask": mask})
    return out


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


def d_tan(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    c = xp.cos(x.value)
    return x._unary(xp.tan(x.value), lambda a, g: g / (c * c), d_tan)


def d_arcsin(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.arcsin(x.value), lambda a, g: g / xp.sqrt(1 - a * a), d_arcsin)


def d_arccos(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.arccos(x.value), lambda a, g: -g / xp.sqrt(1 - a * a), d_arccos)


def d_arctanh(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.arctanh(x.value), lambda a, g: g / (1 - a * a), d_arctanh)


def d_arcsinh(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.arcsinh(x.value), lambda a, g: g / xp.sqrt(a * a + 1), d_arcsinh)


def d_arccosh(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.arccosh(x.value), lambda a, g: g / xp.sqrt(a * a - 1), d_arccosh)


def d_exp2(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    v = xp.exp2(x.value)
    return x._unary(v, lambda a, g: g * v * math.log(2.0), d_exp2)


def d_log2(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.log2(x.value), lambda a, g: g / (a * math.log(2.0)), d_log2)


def d_log10(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.log10(x.value), lambda a, g: g / (a * math.log(10.0)), d_log10)


def d_deg2rad(x: Operand) -> Var:
    # Also backs ``np.radians`` (a numpy alias for ``deg2rad``); derivative is the constant
    # ``pi/180``.
    x, xp = _lift(x), _xp()
    return x._unary(xp.deg2rad(x.value), lambda a, g: g * (math.pi / 180.0), d_deg2rad)


def d_rad2deg(x: Operand) -> Var:
    # Also backs ``np.degrees``; derivative is the constant ``180/pi``.
    x, xp = _lift(x), _xp()
    return x._unary(xp.rad2deg(x.value), lambda a, g: g * (180.0 / math.pi), d_rad2deg)


def d_sign(x: Operand) -> Var:
    # ``sign`` is piecewise-constant: its derivative is zero a.e. (the kink at 0 is ignored,
    # matching autograd / the ``forward.py`` mask convention).
    x, xp = _lift(x), _xp()
    _reject_complex("sign", x)
    return x._unary(xp.sign(x.value), lambda a, g: g * 0.0, d_sign)


def d_ceil(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.ceil(x.value), lambda a, g: g * 0.0, d_ceil)


def d_floor(x: Operand) -> Var:
    x, xp = _lift(x), _xp()
    return x._unary(xp.floor(x.value), lambda a, g: g * 0.0, d_floor)


# ---------------------------------------------------------------------------
# Elementwise binary / selection.
# ---------------------------------------------------------------------------
def _reject_complex(op_name: str, *operands: Operand) -> None:
    """Raise a clear error if any operand is complex -- order-dependent ops (max/min/clip/
    sort/sign/...) have no meaning on the unordered complex field."""
    for o in operands:
        arr = o.value if isinstance(o, Var) else o
        if np.iscomplexobj(cast(Any, arr)):
            raise TypeError(
                f"{op_name}: complex tensors are unordered, so {op_name} (and its "
                "gradient) is undefined; restrict it to real-valued operands"
            )


def _elementwise_max(
    a: Operand, b: Operand, pick_a: Callable[..., Array], prim: Prim
) -> Var:
    a, b = _lift(a), _lift(b)
    _reject_complex(getattr(prim, "__name__", "maximum/minimum"), a, b)
    out = Var(pick_a(a.value, b.value), _parents=(a, b))

    def _backward() -> None:
        mask = (a.value == out.value).astype(float)
        a.grad = _accumulate(a.grad, _unbroadcast(out.grad * mask, a.value.shape))
        b.grad = _accumulate(b.grad, _unbroadcast(out.grad * (1 - mask), b.value.shape))

    out._backward = _backward
    _record_vjp(out, prim, (a, b), {"out_value": out.value})
    return out


def d_maximum(a: Operand, b: Operand) -> Var:
    return _elementwise_max(a, b, _xp().maximum, d_maximum)


def d_minimum(a: Operand, b: Operand) -> Var:
    return _elementwise_max(a, b, _xp().minimum, d_minimum)


def d_fmax(a: Operand, b: Operand) -> Var:
    # Like maximum (gradient flows to the larger operand); fmax ignores NaN, which the
    # ``a == out`` selection mask handles for the real, non-NaN inputs we differentiate.
    return _elementwise_max(a, b, _xp().fmax, d_fmax)


def d_fmin(a: Operand, b: Operand) -> Var:
    return _elementwise_max(a, b, _xp().fmin, d_fmin)


# logaddexp(a, b) = log(exp a + exp b); logaddexp2(a, b) = log2(2^a + 2^b). Smooth: the
# gradient w.r.t. each operand is its softmax weight ``exp(operand - out)`` (base-2 for
# logaddexp2), computed stably (the exponent is <= 0).
def _logaddexp_like(
    a: Operand,
    b: Operand,
    fn: Callable[..., Array],
    base_exp: Callable[..., Array],
    prim: Prim,
) -> Var:
    a, b = _lift(a), _lift(b)
    out_val = fn(a.value, b.value)
    out = Var(out_val, _parents=(a, b))

    def _backward() -> None:
        wa = base_exp(a.value - out_val)
        wb = base_exp(b.value - out_val)
        a.grad = _accumulate(a.grad, _unbroadcast(out.grad * wa, a.value.shape))
        b.grad = _accumulate(b.grad, _unbroadcast(out.grad * wb, b.value.shape))

    out._backward = _backward
    _record_vjp(out, prim, (a, b))
    return out


def d_logaddexp(a: Operand, b: Operand) -> Var:
    xp = _xp()
    return _logaddexp_like(a, b, xp.logaddexp, xp.exp, d_logaddexp)


def d_logaddexp2(a: Operand, b: Operand) -> Var:
    xp = _xp()
    return _logaddexp_like(a, b, xp.logaddexp2, lambda z: xp.exp2(z), d_logaddexp2)


def _vjp_logaddexp_for(prim: Prim, expp: Prim) -> Callable[..., list[Boxed]]:
    """``ga = g * base^(a - out)``, ``gb = g * base^(b - out)`` -- the softmax weights,
    re-bound (differentiable) via the op's own primitive ``prim`` and exp ``expp``."""

    def rule(
        primals: tuple[Var, ...],
        operands: tuple[Boxed, ...],
        params: dict[str, Any],
        g: Boxed,
    ) -> list[Boxed]:
        a, b = operands
        out = _b(prim, a, b)
        wa = _b(expp, _b(d_sub, a, out))
        wb = _b(expp, _b(d_sub, b, out))
        return [_b(d_mul, g, wa), _b(d_mul, g, wb)]

    return rule


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
        a.grad = _accumulate(
            a.grad, _unbroadcast(xp.where(cond, out.grad, 0.0), a.value.shape)
        )
        b.grad = _accumulate(
            b.grad, _unbroadcast(xp.where(cond, 0.0, out.grad), b.value.shape)
        )

    out._backward = _backward
    _record_vjp(out, d_where, (a, b), {"cond": cond})
    return out


def d_gated_act(f: Operand, s: Operand) -> Var:
    # Fused gated activation ``tanh(f) * sigmoid(s)`` (the WaveNet / GLU gate). A fused
    # primitive: one tape node and one VJP instead of a tanh/sigmoid/multiply chain, and
    # the compile backends lower it to a native ``tanh*sigmoid`` (see their intercepts).
    # Tape-only (no numpy name, so no ``_RULES`` callable) -- user code calls it directly,
    # so under an enclosing transform an operand is a higher-level tracer; dispatch through
    # ``bind`` so the registered jvp/vmap/abstract rule runs (the ``d_sigmoid`` pattern).
    from pycograd.trace import Tracer, bind

    if isinstance(f, Tracer) or isinstance(s, Tracer):
        return cast(Var, bind(d_gated_act, f, s))
    f, s, xp = _lift(f), _lift(s), _xp()
    tanh_f = xp.tanh(f.value)
    sig_s = xp.reciprocal(1.0 + xp.exp(-s.value))
    out = Var(tanh_f * sig_s, _parents=(f, s))

    def _backward() -> None:
        # d/df = g * sigmoid(s) * (1 - tanh(f)^2);  d/ds = g * tanh(f) * sigmoid(s)(1-sigmoid(s))
        df = out.grad * sig_s * (1 - tanh_f * tanh_f)
        ds = out.grad * tanh_f * sig_s * (1 - sig_s)
        f.grad = _accumulate(f.grad, _unbroadcast(df, f.value.shape))
        s.grad = _accumulate(s.grad, _unbroadcast(ds, s.value.shape))

    out._backward = _backward
    _record_vjp(out, d_gated_act, (f, s))
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


def d_mod(a: Operand, b: Operand) -> Var:
    return _lift(a) % b


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
        # Conjugate the operand factors (not the cotangent) for the complex Hermitian
        # adjoint: ``da = g @ conj(b).T``, ``db = conj(a).T @ g``. ``conj_if_complex`` is
        # the identity on real arrays, so the real path is unchanged.
        da, db = _matmul_grads(
            conj_if_complex(a.value), conj_if_complex(b.value), out.grad
        )
        a.grad = _accumulate(a.grad, _unbroadcast(da, a.value.shape))
        b.grad = _accumulate(b.grad, _unbroadcast(db, b.value.shape))

    out._backward = _backward
    _record_vjp(out, _matmul, (a, b))
    return out


# ---------------------------------------------------------------------------
# Einsum -- a fused primitive (general tensor contraction).
#
# A single ``np.einsum`` call cannot be generically decomposed into our other
# primitives, so it carries its own rules. The VJP of an einsum w.r.t. operand ``i``
# is *another* einsum whose inputs are the upstream cotangent (subscripted as the
# output) and the remaining operands, contracted down to operand ``i``'s subscript.
# A label of operand ``i`` that was summed out (appears nowhere else and not in the
# output) is reconstructed by appending a constant ``ones`` operand carrying it, so a
# single reverse einsum covers both contraction and reduction-broadcast.
#
# An ellipsis (``...``) is supported with full numpy broadcasting: it is expanded
# into explicit, right-aligned fresh labels using each operand's rank (so the
# grad/vmap/shape machinery below sees only ordinary labels), and numpy broadcasts
# any shared size-1 label in both the forward and reverse einsum -- the backward
# then sums each gradient back to its operand's shape (``_unbroadcast``).
#
# The one form still rejected (a clear error): a label repeated *within one operand*
# (a diagonal / trace), which a plain reverse einsum cannot express.
# ---------------------------------------------------------------------------
def _fresh_labels(n: int, used: "set[str]") -> str:
    """``n`` distinct letters not in ``used`` (to name expanded ellipsis axes)."""
    import string

    out: list[str] = []
    for c in string.ascii_letters:
        if len(out) == n:
            break
        if c not in used:
            out.append(c)
    if len(out) != n:
        raise ValueError("einsum: ran out of labels to expand the ellipsis")
    return "".join(out)


def _expand_ellipsis(
    ins: list[str], sep: str, rhs: str, ranks: "Sequence[int]"
) -> tuple[list[str], str]:
    """Replace ``...`` in each operand (and the output) with explicit, right-aligned
    fresh labels. The ellipsis width is the max over operands of ``rank - #explicit``;
    an operand with fewer ellipsis axes takes the *trailing* labels, so they align
    (and broadcast) exactly as numpy's ellipsis does."""
    n_implicit = []
    for sub, rank in zip(ins, ranks):
        explicit = sub.replace("...", "")
        if "..." in sub:
            k = rank - len(explicit)
            if k < 0:
                raise ValueError(
                    f"einsum: operand subscript {sub!r} has more labels than the "
                    f"operand has axes ({rank})"
                )
            n_implicit.append(k)
        else:
            if len(explicit) != rank:
                raise ValueError(f"einsum: operand subscript {sub!r} vs rank {rank}")
            n_implicit.append(0)
    width = max(n_implicit)
    used = {c for c in "".join(ins) + rhs if c.isalpha()}
    ell = _fresh_labels(width, used)
    new_ins = [sub.replace("...", ell[width - k :]) for sub, k in zip(ins, n_implicit)]
    if sep:
        new_rhs = rhs.replace("...", ell)
    else:  # implicit output: ellipsis labels first, then explicit singles, sorted
        counts = Counter("".join(new_ins))
        singles = sorted(c for c, n in counts.items() if n == 1 and c not in ell)
        new_rhs = ell + "".join(singles)
    return new_ins, new_rhs


def _normalize_einsum_args(subscripts: Any, operands: tuple) -> tuple[str, tuple]:
    """Accept either einsum form and return ``(subscripts_string, operands)``.

    numpy's *interleaved* form passes ``op0, sublist0, op1, sublist1, ..., [out_sublist]``
    instead of a subscript string -- detected by a non-string first argument. Each integer
    (or ``Ellipsis``) index label is mapped to a stable letter, and a trailing output
    sublist (present iff the argument count is odd) becomes the ``->`` group. A string
    first argument passes straight through.
    """
    import string

    if isinstance(subscripts, str):
        return subscripts, operands
    args = (subscripts, *operands)
    ops_list: list = []
    sublists: list = []
    i = 0
    while i + 1 < len(args):
        ops_list.append(args[i])
        sublists.append(args[i + 1])
        i += 2
    out_sublist = args[i] if i < len(args) else None

    label_map: dict = {}

    def lbl(x: Any) -> str:
        if x is Ellipsis:
            return "..."
        if x not in label_map:
            if len(label_map) >= len(string.ascii_letters):
                raise ValueError("einsum: too many distinct index labels")
            label_map[x] = string.ascii_letters[len(label_map)]
        return label_map[x]

    spec = ",".join("".join(lbl(x) for x in sl) for sl in sublists)
    if out_sublist is not None:
        spec += "->" + "".join(lbl(x) for x in out_sublist)
    return spec, tuple(ops_list)


def _parse_einsum(subscripts: str, ranks: "Sequence[int]") -> tuple[list[str], str]:
    """Split an einsum spec into per-operand input labels and the (possibly implicit)
    output labels, expanding any ellipsis into explicit labels (using ``ranks``, the
    per-operand ndims) and rejecting the forms our reverse rule can't express."""
    if not isinstance(subscripts, str):
        raise NotImplementedError(
            "einsum: only the subscripts-string form is supported"
        )
    lhs, sep, rhs = subscripts.replace(" ", "").partition("->")
    ins = lhs.split(",")
    if len(ins) != len(ranks):
        raise ValueError(
            f"einsum: {len(ins)} subscript group(s) for {len(ranks)} operand(s)"
        )
    if "..." in subscripts:
        ins, rhs = _expand_ellipsis(ins, sep, rhs, ranks)
    elif not sep:  # implicit output: labels appearing exactly once, sorted (numpy's)
        counts = Counter("".join(ins))
        rhs = "".join(sorted(c for c, n in counts.items() if n == 1))
    for sub in ins:
        if len(set(sub)) != len(sub):
            raise NotImplementedError(
                f"einsum: a repeated label within one operand ({sub!r}, a diagonal / "
                "trace) is not supported"
            )
    return ins, rhs


def _einsum_grad_spec(
    ins: list[str], out: str, i: int
) -> "tuple[str, list[int], list[str]]":
    """The reverse-einsum spec for operand ``i``'s cotangent, plus the indices of the
    other operands (in order) and the labels of operand ``i`` that were summed out (and
    so need a ``ones`` operand). The cotangent is subscripted as ``out``."""
    sub_i = ins[i]
    others = [j for j in range(len(ins)) if j != i]
    avail = set("".join(ins[j] for j in others)) | set(out)
    missing = [c for c in sub_i if c not in avail]
    in_subs = [out] + [ins[j] for j in others] + (["".join(missing)] if missing else [])
    return ",".join(in_subs) + "->" + sub_i, others, missing


def d_einsum(subscripts: Any, *operands: Operand) -> Var:
    from pycograd.trace import Tracer, bind

    subscripts, operands = _normalize_einsum_args(subscripts, operands)
    # A higher trace level is live (a vmap/jvp/abstract Tracer flowed in via a direct
    # call rather than ``bind``): route through the stack so the registered rule runs.
    if any(isinstance(o, Tracer) for o in operands):
        return cast(Var, bind(d_einsum, subscripts, *operands))
    xp = _xp()
    lifted = [_lift(o) for o in operands]
    vals = [o.value for o in lifted]
    ins, out = _parse_einsum(subscripts, [v.ndim for v in vals])
    fwd_spec = ",".join(ins) + "->" + out
    try:
        primal = xp.einsum(fwd_spec, *vals)
    except ValueError as e:
        from pycograd.shapes import ShapeError, _shape_context

        raise ShapeError(
            _shape_context("einsum " + subscripts, *(v.shape for v in vals))
        ) from e
    node = Var(primal, _parents=tuple(lifted))

    def _backward() -> None:
        for i, op in enumerate(lifted):
            spec, others, missing = _einsum_grad_spec(ins, out, i)
            # Conjugate the *other* operand factors (not the cotangent) for the complex
            # Hermitian adjoint; identity on real operands.
            arrays = [node.grad] + [conj_if_complex(vals[j]) for j in others]
            if missing:
                mshape = tuple(op.value.shape[ins[i].index(c)] for c in missing)
                arrays.append(xp.ones(mshape, dtype=node.grad.dtype))
            # ``_unbroadcast`` sums any size-1 operand axis that numpy broadcast up.
            op.grad = _accumulate(
                op.grad, _unbroadcast(xp.einsum(spec, *arrays), op.value.shape)
            )

    node._backward = _backward
    _record_vjp(node, d_einsum, tuple(lifted), {"subscripts": subscripts})
    return node


# ---------------------------------------------------------------------------
# Tensor contraction ops (np.dot / np.inner / np.outer / np.tensordot) -- each *lowers to
# einsum*: the eager call and all three transform rules build the appropriate einsum
# subscript from the operand ranks and dispatch ``d_einsum``, which already carries the
# reverse / forward / vmap / shape rules. So these ops appear only in the forward / batch /
# abstract tables (delegating to einsum); they never become tape nodes themselves (the tape
# records the underlying ``d_einsum``), so they need no ``_VJP_FOR`` entry.
# ---------------------------------------------------------------------------
# ``x: object`` -- a duck-typed shape probe over the whole operand zoo (array / scalar /
# ``Var`` / any ``Tracer`` / ``ShapedArray`` / ``Weight``), all of which either expose
# ``.shape`` or are accepted by ``np.ndim``; a narrower type would not cover the union.
def _on_tape(x: object) -> bool:
    """True if ``x`` is a tape ``Var`` or a transform ``Tracer`` (i.e. a value being
    differentiated). A *plain* array/scalar passing through a structural op (e.g. an integer
    index built with ``np.repeat(np.arange(...))``) is a constant and must stay a plain
    array, not be lifted onto the tape -- matching autograd, whose primitives only box when
    an argument is already boxed."""
    from pycograd.trace import Tracer

    return isinstance(x, (Var, Tracer))


def _logical_ndim(x: object) -> int:
    """The rank of an operand that may be a concrete array, a ``Var``, or a higher-level
    tracer / abstract value (all expose ``.shape``)."""
    shp = getattr(x, "shape", None)
    if shp is not None:
        return len(cast(Any, shp))
    return int(np.ndim(cast(Any, x)))  # a raw scalar/array


def _abc(n: int, start: int) -> str:
    import string

    return string.ascii_lowercase[start : start + n]


def dot_subscript(na: int, nb: int) -> str:
    """``np.dot``: last axis of ``a`` contracts with the second-to-last of ``b`` (or the
    single axis of a 1-D ``b``)."""
    if na == 1 and nb == 1:
        return "i,i->"
    la = _abc(na, 0)
    if nb == 1:
        return f"{la},{la[-1]}->{la[:-1]}"
    lb = list(_abc(nb, na))
    lb[nb - 2] = la[-1]  # contraction axis of b shares a's last label
    lbs = "".join(lb)
    out = la[:-1] + "".join(c for k, c in enumerate(lbs) if k != nb - 2)
    return f"{la},{lbs}->{out}"


def inner_subscript(na: int, nb: int) -> str:
    """``np.inner``: the last axes of both operands contract."""
    la = _abc(na, 0)
    lb = list(_abc(nb, na))
    lb[-1] = la[-1]
    lbs = "".join(lb)
    return f"{la},{lbs}->{la[:-1]}{lbs[:-1]}"


def tensordot_subscript(na: int, nb: int, axes: Any) -> str:
    """``np.tensordot``: contract the given axis pairs (``axes`` an int N -> last N of ``a``
    with first N of ``b``; or a pair of axis lists)."""
    if isinstance(axes, int):
        a_ax = list(range(na - axes, na))
        b_ax = list(range(axes))
    else:
        a_raw, b_raw = axes
        a_ax = [
            a % na for a in (a_raw if isinstance(a_raw, (list, tuple)) else [a_raw])
        ]
        b_ax = [
            b % nb for b in (b_raw if isinstance(b_raw, (list, tuple)) else [b_raw])
        ]
    la = list(_abc(na, 0))
    lb = list(_abc(nb, na))
    for ai, bi in zip(a_ax, b_ax):
        lb[bi] = la[ai]  # contracted pair shares a label
    out = [la[k] for k in range(na) if k not in a_ax]
    out += [lb[k] for k in range(nb) if k not in b_ax]
    return f"{''.join(la)},{''.join(lb)}->{''.join(out)}"


def d_dot(a: Operand, b: Operand) -> Var:
    na, nb = _logical_ndim(a), _logical_ndim(b)
    if na == 0 or nb == 0:  # np.dot with a scalar is just multiplication
        return cast(Var, _lift(a) * b)
    return d_einsum(dot_subscript(na, nb), a, b)


def d_inner(a: Operand, b: Operand) -> Var:
    na, nb = _logical_ndim(a), _logical_ndim(b)
    if na == 0 or nb == 0:
        return cast(Var, _lift(a) * b)
    return d_einsum(inner_subscript(na, nb), a, b)


def d_tensordot(a: Operand, b: Operand, axes: Any = 2) -> Var:
    na, nb = _logical_ndim(a), _logical_ndim(b)
    return d_einsum(tensordot_subscript(na, nb, axes), a, b)


def _contraction_subscript(prim: Prim, na: int, nb: int, params: dict[str, Any]) -> str:
    if prim is d_dot:
        return dot_subscript(na, nb)
    if prim is d_inner:
        return inner_subscript(na, nb)
    return tensordot_subscript(na, nb, params.get("axes", 2))


def contraction_transform_rule(prim: Prim) -> Callable[..., Boxed]:
    """The forward (jvp) and batching (vmap) rule for a contraction op: build the einsum
    subscript from the operand ranks and re-bind ``d_einsum`` (whose own rule then fires at
    the live level). A scalar operand degrades to elementwise multiply."""
    from pycograd.trace import bind

    def rule(_trace: Boxed, a: Boxed, b: Boxed, **params: Any) -> Boxed:
        na, nb = _logical_ndim(a), _logical_ndim(b)
        if na == 0 or nb == 0:
            return bind(d_mul, a, b)
        return bind(d_einsum, _contraction_subscript(prim, na, nb, params), a, b)

    return rule


def contraction_abstract_rule(prim: Prim) -> Callable[..., Boxed]:
    """The shape-inference (eval_shape) rule: delegate to ``abstract_einsum`` on the built
    subscript."""

    def rule(a: Boxed, b: Boxed, **params: Any) -> Boxed:
        from pycograd.shapes import abstract_binary, abstract_einsum

        na, nb = _logical_ndim(a), _logical_ndim(b)
        if na == 0 or nb == 0:
            return cast(Boxed, abstract_binary(cast(Any, a), cast(Any, b)))
        sub = _contraction_subscript(prim, na, nb, params)
        return cast(Boxed, abstract_einsum(sub, cast(Any, a), cast(Any, b)))

    return rule


# ---------------------------------------------------------------------------
# Axis-reordering ops (np.moveaxis / np.swapaxes / np.rollaxis) -- each is a *transpose*
# with a permutation computed from the operand rank and the (host-side) axis arguments, so
# they lower to ``d_transpose`` exactly as the contraction ops lower to ``d_einsum``. No
# ``_VJP_FOR`` entry (the tape records ``d_transpose``); only the transform tables delegate.
# ---------------------------------------------------------------------------
def _as_axis_list(a: Any) -> list[int]:
    return list(a) if isinstance(a, (list, tuple)) else [int(a)]


def moveaxis_perm(ndim: int, source: Any, destination: Any) -> tuple[int, ...]:
    src = [s % ndim for s in _as_axis_list(source)]
    dst = [d % ndim for d in _as_axis_list(destination)]
    order = [n for n in range(ndim) if n not in src]
    for d, s in sorted(zip(dst, src)):
        order.insert(d, s)
    return tuple(order)


def swapaxes_perm(ndim: int, axis1: int, axis2: int) -> tuple[int, ...]:
    perm = list(range(ndim))
    a1, a2 = axis1 % ndim, axis2 % ndim
    perm[a1], perm[a2] = perm[a2], perm[a1]
    return tuple(perm)


def rollaxis_perm(ndim: int, axis: int, start: int = 0) -> tuple[int, ...]:
    axis %= ndim
    if start < 0:
        start += ndim
    if start > axis:
        start -= 1
    axes = list(range(ndim))
    axes.remove(axis)
    axes.insert(start, axis)
    return tuple(axes)


def d_moveaxis(x: Operand, source: Any, destination: Any) -> Var:
    return d_transpose(x, moveaxis_perm(_logical_ndim(x), source, destination))


def d_swapaxes(x: Operand, axis1: int, axis2: int) -> Var:
    return d_transpose(x, swapaxes_perm(_logical_ndim(x), axis1, axis2))


def d_rollaxis(x: Operand, axis: int, start: int = 0) -> Var:
    return d_transpose(x, rollaxis_perm(_logical_ndim(x), axis, start))


def _transpose_lowering_transform(
    perm_builder: Callable[..., tuple[int, ...]],
) -> Callable[..., Boxed]:
    from pycograd.trace import bind

    def rule(_trace: Boxed, x: Boxed, *args: Any, **kw: Any) -> Boxed:
        perm = perm_builder(_logical_ndim(x), *args, **kw)
        return bind(d_transpose, x, axes=perm)

    return rule


def _transpose_lowering_abstract(
    perm_builder: Callable[..., tuple[int, ...]],
) -> Callable[..., Boxed]:
    def rule(x: Boxed, *args: Any, **kw: Any) -> Boxed:
        from pycograd.shapes import abstract_transpose

        perm = perm_builder(_logical_ndim(x), *args, **kw)
        return cast(Boxed, abstract_transpose(cast(Any, x), perm))

    return rule


# ---------------------------------------------------------------------------
# Triangular ops (np.tril / np.triu) -- multiply by a constant lower/upper-triangular mask
# (applied to the final two axes, like numpy). A composition of ``d_mul`` with a stop-
# gradient mask, so again only the transform tables delegate; the tape records ``d_mul``.
# ---------------------------------------------------------------------------
def d_tril(x: Operand, k: int = 0) -> Var:
    x2, xp = _lift(x), _xp()
    return cast(Var, x2 * xp.tril(xp.ones_like(x2.value), k))


def d_triu(x: Operand, k: int = 0) -> Var:
    x2, xp = _lift(x), _xp()
    return cast(Var, x2 * xp.triu(xp.ones_like(x2.value), k))


def _tri_lowering_transform(np_tri: Callable[..., Array]) -> Callable[..., Boxed]:
    from pycograd.trace import bind

    def rule(_trace: Boxed, x: Boxed, k: int = 0, **kw: Any) -> Boxed:
        mask = np_tri(_xp().ones(tuple(cast(Any, x).shape)), k)
        return bind(d_mul, x, mask)

    return rule


def _tri_lowering_abstract(x: Boxed, k: int = 0, **kw: Any) -> Boxed:
    from pycograd.shapes import abstract_unary

    return cast(Boxed, abstract_unary(cast(Any, x)))


# ---------------------------------------------------------------------------
# Reshape-only ops (np.ravel / np.squeeze / np.atleast_{1,2,3}d) -- a pure reshape whose
# target shape is a function of the input shape, so they lower to ``d_reshape``.
# ---------------------------------------------------------------------------
def _logical_shape(x: object) -> tuple[int, ...]:
    shp = getattr(x, "shape", None)
    if shp is not None:
        return tuple(cast(Any, shp))
    return tuple(np.shape(cast(Any, x)))


def ravel_shape(shp: tuple[int, ...], *_a: Any, **_kw: Any) -> tuple[int, ...]:
    return (int(np.prod(shp, dtype=np.int64)),)


def squeeze_shape(
    shp: tuple[int, ...], axis: Any = None, **_kw: Any
) -> tuple[int, ...]:
    if axis is None:
        return tuple(d for d in shp if d != 1)
    axes = {a % len(shp) for a in _as_axis_list(axis)}
    return tuple(d for i, d in enumerate(shp) if i not in axes)


def atleast_1d_shape(shp: tuple[int, ...], *_a: Any, **_kw: Any) -> tuple[int, ...]:
    return shp if len(shp) >= 1 else (1,)


def atleast_2d_shape(shp: tuple[int, ...], *_a: Any, **_kw: Any) -> tuple[int, ...]:
    if len(shp) >= 2:
        return shp
    return (1,) * (2 - len(shp)) + shp


def atleast_3d_shape(shp: tuple[int, ...], *_a: Any, **_kw: Any) -> tuple[int, ...]:
    if len(shp) >= 3:
        return shp
    if len(shp) == 2:
        return shp + (1,)
    if len(shp) == 1:
        return (1,) + shp + (1,)
    return (1, 1, 1)


def d_ravel(x: Operand) -> Var:
    return d_reshape(x, ravel_shape(_logical_shape(x)))


def d_squeeze(x: Operand, axis: Any = None) -> Var:
    return d_reshape(x, squeeze_shape(_logical_shape(x), axis))


def d_atleast_1d(x: Operand) -> Var:
    return d_reshape(x, atleast_1d_shape(_logical_shape(x)))


def d_atleast_2d(x: Operand) -> Var:
    return d_reshape(x, atleast_2d_shape(_logical_shape(x)))


def d_atleast_3d(x: Operand) -> Var:
    return d_reshape(x, atleast_3d_shape(_logical_shape(x)))


def _reshape_lowering_transform(
    shape_builder: Callable[..., tuple[int, ...]],
) -> Callable[..., Boxed]:
    from pycograd.trace import bind

    def rule(_trace: Boxed, x: Boxed, *args: Any, **kw: Any) -> Boxed:
        target = shape_builder(_logical_shape(x), *args, **kw)
        return bind(d_reshape, x, target)

    return rule


def _reshape_lowering_abstract(
    shape_builder: Callable[..., tuple[int, ...]],
) -> Callable[..., Boxed]:
    def rule(x: Boxed, *args: Any, **kw: Any) -> Boxed:
        from pycograd.shapes import abstract_reshape

        target = shape_builder(_logical_shape(x), *args, **kw)
        return cast(Boxed, abstract_reshape(cast(Any, x), target))

    return rule


# ---------------------------------------------------------------------------
# np.roll -- a circular shift: a linear, shape-preserving permutation. The VJP rolls the
# cotangent back by the negated shift. A genuine primitive (no composition expresses it).
# ---------------------------------------------------------------------------
def _neg_shift(shift: Any) -> Any:
    if isinstance(shift, (list, tuple)):
        return tuple(-s for s in shift)
    return -shift


def d_roll(x: Operand, shift: Any, axis: Any = None) -> Var:
    from pycograd.trace import Tracer, bind

    if isinstance(x, Tracer):
        return cast(Var, bind(d_roll, x, shift, axis=axis))
    if not isinstance(x, Var):
        return cast(Var, _xp().roll(_xp().asarray(x), shift, axis=axis))
    x, xp = _lift(x), _xp()
    out = Var(xp.roll(x.value, shift, axis=axis), _parents=(x,))

    def _backward() -> None:
        x.grad = _accumulate(x.grad, xp.roll(out.grad, _neg_shift(shift), axis=axis))

    out._backward = _backward
    _record_vjp(out, d_roll, (x,), {"shift": shift, "axis": axis})
    return out


def _vjp_roll(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    (x,) = operands
    return [_b(d_roll, g, _neg_shift(params["shift"]), axis=params["axis"])]


def _roll_abstract(x: Boxed, shift: Any = None, axis: Any = None) -> Boxed:
    from pycograd.shapes import abstract_unary  # roll preserves shape

    return cast(Boxed, abstract_unary(cast(Any, x)))


# ---------------------------------------------------------------------------
# np.pad (constant mode) -- linear: place ``x`` into a larger zero-filled array. The VJP
# slices the padded cotangent back to ``x``'s region (a getitem adjoint); the forward pads
# the tangent with zeros (the pad constant does not depend on ``x``).
# ---------------------------------------------------------------------------
def normalize_pad_width(pad_width: Any, ndim: int) -> tuple[tuple[int, int], ...]:
    """numpy's ``pad_width`` broadcasting -> one explicit ``(before, after)`` per axis."""
    if isinstance(pad_width, (int, np.integer)):
        return tuple((int(pad_width), int(pad_width)) for _ in range(ndim))
    items = list(pad_width)
    if all(isinstance(e, (int, np.integer)) for e in items):
        if len(items) == 1:
            return tuple((int(items[0]), int(items[0])) for _ in range(ndim))
        return tuple((int(items[0]), int(items[1])) for _ in range(ndim))
    if len(items) == 1:  # ((before, after),) broadcast to all axes
        b, a = items[0]
        return tuple((int(b), int(a)) for _ in range(ndim))
    return tuple((int(b), int(a)) for b, a in items)


def d_pad(x: Operand, pad_width: Any, mode: str = "constant", **kw: Any) -> Var:
    if not _on_tape(x):
        return cast(Var, _xp().pad(_xp().asarray(x), pad_width, mode=mode, **kw))
    if mode != "constant":
        raise NotImplementedError(
            f"pad: only mode='constant' is differentiable so far (got {mode!r})"
        )
    x, xp = _lift(x), _xp()
    out = Var(xp.pad(x.value, pad_width, mode=mode, **kw), _parents=(x,))
    pw = normalize_pad_width(pad_width, x.value.ndim)
    sl = tuple(slice(b, b + n) for (b, _a), n in zip(pw, x.value.shape))

    def _backward() -> None:
        x.grad = _accumulate(x.grad, out.grad[sl])

    out._backward = _backward
    _record_vjp(out, d_pad, (x,), {"slices": sl})
    return out


def _vjp_pad(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # The VJP of a constant pad gathers back the interior. Eager records the precomputed
    # ``slices``; a captured graph instead carries ``pad_width`` (the bind arg), so derive the
    # slices from it and the primal's shape when ``slices`` isn't present.
    slices = params.get("slices")
    if slices is None:
        in_shape = _pshape(primals[0])
        pw = normalize_pad_width(params["pad_width"], len(in_shape))
        slices = tuple(slice(b, b + n) for (b, _a), n in zip(pw, in_shape))
    return [_b(d_getitem, g, slices)]


def _pad_abstract(x: Boxed, pad_width: Any, mode: str = "constant", **kw: Any) -> Boxed:
    from pycograd.shapes import ShapedArray, _aval

    a = _aval(cast(Any, x))
    pw = normalize_pad_width(pad_width, len(a.shape))
    out_shape = tuple(d + b + af for d, (b, af) in zip(a.shape, pw))
    return cast(Boxed, ShapedArray(out_shape, a.dtype))


# ---------------------------------------------------------------------------
# np.repeat / np.tile -- linear "copy" ops whose VJP is the matching *sum over copies*.
# ``repeats`` (np.repeat) and ``reps`` (np.tile) are integer constants here (the array-valued
# ``repeats`` form is not supported).
# ---------------------------------------------------------------------------
def _repeat_grad_spec(
    x_shape: tuple[int, ...], repeats: int, axis: Any
) -> tuple[tuple[int, ...], int]:
    """The ``(reshape_shape, sum_axis)`` whose ``reshape(g).sum(sum_axis)`` is the repeat
    adjoint (``axis=None`` ravels first)."""
    if axis is None:
        n = int(np.prod(x_shape, dtype=np.int64))
        return ((n, repeats), 1)
    ax = axis % len(x_shape)
    new = x_shape[:ax] + (x_shape[ax], repeats) + x_shape[ax + 1 :]
    return (new, ax + 1)


def d_repeat(x: Operand, repeats: int, axis: Any = None) -> Var:
    if not _on_tape(x):
        return cast(Var, _xp().repeat(_xp().asarray(x), repeats, axis=axis))
    x, xp = _lift(x), _xp()
    out = Var(xp.repeat(x.value, repeats, axis=axis), _parents=(x,))
    rshape, sax = _repeat_grad_spec(x.value.shape, repeats, axis)
    x_shape = x.value.shape

    def _backward() -> None:
        x.grad = _accumulate(
            x.grad, out.grad.reshape(rshape).sum(axis=sax).reshape(x_shape)
        )

    out._backward = _backward
    _record_vjp(out, d_repeat, (x,), {"repeats": repeats, "axis": axis})
    return out


def _vjp_repeat(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    (x,) = operands
    x_shape = _pshape(primals[0])
    rshape, sax = _repeat_grad_spec(x_shape, params["repeats"], params["axis"])
    summed = _b(d_sum, _b(d_reshape, g, rshape), axis=sax)
    return [_b(d_reshape, summed, x_shape)]


def _repeat_abstract(x: Boxed, repeats: int, axis: Any = None) -> Boxed:
    from pycograd.shapes import ShapedArray, _aval

    a = _aval(cast(Any, x))
    if axis is None:
        return cast(
            Boxed,
            ShapedArray(
                (int(np.prod(cast(Any, a.shape), dtype=np.int64)) * repeats,), a.dtype
            ),
        )
    ax = axis % len(a.shape)
    sh = list(a.shape)
    sh[ax] = sh[ax] * repeats
    return cast(Boxed, ShapedArray(tuple(sh), a.dtype))


def _tile_dims(
    x_shape: tuple[int, ...], reps: Any
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rp = (
        (int(reps),)
        if isinstance(reps, (int, np.integer))
        else tuple(int(r) for r in reps)
    )
    d = max(len(x_shape), len(rp))
    xs = (1,) * (d - len(x_shape)) + tuple(x_shape)
    rp = (1,) * (d - len(rp)) + rp
    return xs, rp


def d_tile(x: Operand, reps: Any) -> Var:
    if not _on_tape(x):
        return cast(Var, _xp().tile(_xp().asarray(x), reps))
    x, xp = _lift(x), _xp()
    out = Var(xp.tile(x.value, reps), _parents=(x,))
    x_shape = x.value.shape
    xs, rp = _tile_dims(x_shape, reps)
    inter = tuple(v for pair in zip(rp, xs) for v in pair)  # (r0, x0, r1, x1, ...)
    rep_axes = tuple(range(0, 2 * len(xs), 2))

    def _backward() -> None:
        x.grad = _accumulate(
            x.grad, out.grad.reshape(inter).sum(axis=rep_axes).reshape(x_shape)
        )

    out._backward = _backward
    _record_vjp(out, d_tile, (x,), {"reps": reps})
    return out


def _vjp_tile(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    (x,) = operands
    x_shape = _pshape(primals[0])
    xs, rp = _tile_dims(x_shape, params["reps"])
    inter = tuple(v for pair in zip(rp, xs) for v in pair)
    rep_axes = tuple(range(0, 2 * len(xs), 2))
    summed = _b(d_sum, _b(d_reshape, g, inter), axis=rep_axes)
    return [_b(d_reshape, summed, x_shape)]


def _tile_abstract(x: Boxed, reps: Any) -> Boxed:
    from pycograd.shapes import ShapedArray, _aval

    a = _aval(cast(Any, x))
    xs, rp = _tile_dims(tuple(cast(Any, a.shape)), reps)
    return cast(Boxed, ShapedArray(tuple(r * s for r, s in zip(rp, xs)), a.dtype))


# ---------------------------------------------------------------------------
# np.split / array_split / vsplit / hsplit / dsplit -- the inverse of concatenate: cut ``x``
# into pieces along an axis. Each piece is a *slice* (``d_getitem``), so split is a
# composition of getitems whose VJPs scatter-add back into ``x`` (the reverse of
# concatenate); the op returns a *list* of values. ``hsplit`` uses axis 1 (axis 0 for 1-D),
# ``vsplit`` axis 0, ``dsplit`` axis 2.
# ---------------------------------------------------------------------------
def split_slices(
    shape: tuple[int, ...], indices_or_sections: Any, axis: int
) -> list[tuple]:
    n = shape[axis]
    if isinstance(indices_or_sections, (int, np.integer)):
        parts = int(indices_or_sections)
        base, rem = divmod(n, parts)
        sizes = [base + 1] * rem + [base] * (parts - rem)  # array_split semantics
        bounds = list(np.cumsum([0] + sizes))
    else:
        idx = [min(max(int(i), 0), n) for i in indices_or_sections]
        bounds = [0] + idx + [n]
    out: list[tuple] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sl: list[Any] = [slice(None)] * len(shape)
        sl[axis] = slice(int(a), int(b))
        out.append(tuple(sl))
    return out


def split_axis(which: str, ndim: int) -> int:
    """The axis each split variant cuts along: ``vsplit`` -> 0, ``dsplit`` -> 2,
    ``hsplit`` -> 1 (0 for 1-D), ``split``/``array_split`` -> the caller's ``axis``."""
    if which == "vsplit":
        return 0
    if which == "dsplit":
        return 2
    return 1 if ndim > 1 else 0  # hsplit


def _resolve_split(which: str, ndim: int, args: tuple, kwargs: dict) -> tuple[Any, int]:
    """``(indices_or_sections, axis)`` for a split-family call, by variant."""
    ind = args[0]
    if which in ("split", "array_split"):
        axis = args[1] if len(args) > 1 else kwargs.get("axis", 0)
        return ind, int(axis)
    return ind, split_axis(which, ndim)


def d_split(x: Operand, indices_or_sections: Any, axis: int = 0) -> list:
    shape = _logical_shape(x)
    return [d_getitem(x, sl) for sl in split_slices(shape, indices_or_sections, axis)]


def d_array_split(x: Operand, indices_or_sections: Any, axis: int = 0) -> list:
    return d_split(x, indices_or_sections, axis)


def d_vsplit(x: Operand, indices_or_sections: Any) -> list:
    return d_split(x, indices_or_sections, 0)


def d_hsplit(x: Operand, indices_or_sections: Any) -> list:
    return d_split(x, indices_or_sections, split_axis("hsplit", _logical_ndim(x)))


def d_dsplit(x: Operand, indices_or_sections: Any) -> list:
    return d_split(x, indices_or_sections, 2)


def _sliced_shape(shape: tuple[int, ...], sl: tuple) -> tuple[int, ...]:
    return tuple(len(range(*s.indices(int(d)))) for s, d in zip(sl, shape))


# ---------------------------------------------------------------------------
# np.diff -- the n-th discrete difference along an axis. One difference is
# ``x[1:] - x[:-1]``: a composition of two getitems and a subtract (all with full rules), so
# like the split family it carries no ``_VJP_FOR`` of its own.
# ---------------------------------------------------------------------------
def _diff_slices(ndim: int, axis: int) -> tuple[tuple, tuple]:
    ax = axis % ndim
    upper = tuple(slice(1, None) if i == ax else slice(None) for i in range(ndim))
    lower = tuple(slice(None, -1) if i == ax else slice(None) for i in range(ndim))
    return upper, lower


def d_diff(x: Operand, n: int = 1, axis: int = -1) -> Var:
    ndim = _logical_ndim(x)
    upper, lower = _diff_slices(ndim, axis)
    cur: Operand = x
    for _ in range(int(n)):
        cur = cast(Var, d_getitem(cur, upper) - d_getitem(cur, lower))
    return cast(Var, cur)


def _resolve_diff(args: tuple, kwargs: dict) -> tuple[int, int]:
    n = int(args[0]) if len(args) > 0 else int(kwargs.get("n", 1))
    axis = int(args[1]) if len(args) > 1 else int(kwargs.get("axis", -1))
    return n, axis


def diff_transform_rule(_trace: Boxed, x: Boxed, *args: Any, **kwargs: Any) -> Boxed:
    from pycograd.trace import bind

    n, axis = _resolve_diff(args, kwargs)
    upper, lower = _diff_slices(len(cast(Any, x).shape), axis)
    cur: Boxed = x
    for _ in range(n):
        cur = bind(d_sub, bind(d_getitem, cur, upper), bind(d_getitem, cur, lower))
    return cur


def diff_abstract_rule(x: Boxed, *args: Any, **kwargs: Any) -> Boxed:
    from pycograd.shapes import ShapedArray, _aval

    a = _aval(cast(Any, x))
    n, axis = _resolve_diff(args, kwargs)
    ax = axis % len(a.shape)
    sh = list(a.shape)
    sh[ax] = sh[ax] - n
    return cast(Boxed, ShapedArray(tuple(sh), a.dtype))


# ---------------------------------------------------------------------------
# np.diag / np.diagonal -- read/write a matrix diagonal. *Extracting* a diagonal is a gather
# at the diagonal index arrays (``d_getitem``); *constructing* one from a 1-D vector is the
# adjoint scatter (``_scatter``). Both carry full rules, so diag is a composition (no
# ``_VJP_FOR`` of its own) that also vmaps / eval-shapes for free.
# ---------------------------------------------------------------------------
def _diag_key(shape: tuple[int, ...], k: int) -> tuple[tuple, tuple[int, ...]]:
    """``(key, out_shape)`` for ``np.diag``: a 1-D ``shape`` *constructs* a square matrix with
    the vector on the ``k``-diagonal; a 2-D ``shape`` *extracts* the ``k``-diagonal."""
    if len(shape) == 1:
        length = int(shape[0])
        m = length + abs(k)
        i = np.arange(length)
        rows, cols = (i, i + k) if k >= 0 else (i - k, i)
        return (rows, cols), (m, m)
    rows_n, cols_n = int(shape[0]), int(shape[1])
    if k >= 0:
        length = max(min(rows_n, cols_n - k), 0)
        i = np.arange(length)
        rows, cols = i, i + k
    else:
        length = max(min(rows_n + k, cols_n), 0)
        i = np.arange(length)
        rows, cols = i - k, i
    return (rows, cols), (length,)


def _operand_dtype(x: object) -> Any:
    if isinstance(x, Var):
        return x.value.dtype
    d = getattr(x, "dtype", None)
    return d if d is not None else np.asarray(cast(Any, x)).dtype


# ---------------------------------------------------------------------------
# np.fliplr / np.flipud / np.rot90 -- reverse an axis (a ``::-1`` slice) and, for rot90, a
# transpose; both are getitem/transpose compositions (full rules), so no own VJP.
# ---------------------------------------------------------------------------
def _flip_key(ndim: int, axis: int) -> tuple:
    return tuple(
        slice(None, None, -1) if i == axis else slice(None) for i in range(ndim)
    )


def d_flipud(m: Operand) -> Var:
    return d_getitem(m, _flip_key(_logical_ndim(m), 0))


def d_fliplr(m: Operand) -> Var:
    return d_getitem(m, _flip_key(_logical_ndim(m), 1))


def _rot90_once(m: Boxed, ndim: int, a0: int, a1: int, use_bind: bool) -> Boxed:
    # One counter-clockwise quarter turn in the (a0, a1) plane: swap the two axes, then
    # reverse the new a0 -- ``flipud(transpose)`` for the 2-D default.
    if use_bind:
        from pycograd.trace import bind

        sw = bind(d_transpose, m, swapaxes_perm(ndim, a0, a1))
        return bind(d_getitem, sw, _flip_key(ndim, a0))
    sw = d_transpose(cast(Operand, m), swapaxes_perm(ndim, a0, a1))
    return d_getitem(sw, _flip_key(ndim, a0))


def d_rot90(m: Operand, k: int = 1, axes: tuple = (0, 1)) -> Var:
    ndim = _logical_ndim(m)
    a0, a1 = axes[0] % ndim, axes[1] % ndim
    cur: Boxed = cast(Boxed, m)
    for _ in range(int(k) % 4):
        cur = _rot90_once(cur, ndim, a0, a1, use_bind=False)
    return cast(Var, cur)


def _flip_transform_rule(axis: int) -> Callable[..., Boxed]:
    def rule(_trace: Boxed, m: Boxed) -> Boxed:
        from pycograd.trace import bind

        return bind(d_getitem, m, _flip_key(len(cast(Any, m).shape), axis))

    return rule


def rot90_transform_rule(
    _trace: Boxed, m: Boxed, k: int = 1, axes: tuple = (0, 1)
) -> Boxed:
    ndim = len(cast(Any, m).shape)
    a0, a1 = axes[0] % ndim, axes[1] % ndim
    cur: Boxed = m
    for _ in range(int(k) % 4):
        cur = _rot90_once(cur, ndim, a0, a1, use_bind=True)
    return cur


def _flip_abstract(m: Boxed) -> Boxed:
    from pycograd.shapes import abstract_unary  # axis-reversal preserves shape

    return cast(Boxed, abstract_unary(cast(Any, m)))


def rot90_abstract_rule(m: Boxed, k: int = 1, axes: tuple = (0, 1)) -> Boxed:
    from pycograd.shapes import ShapedArray, _aval

    a = _aval(cast(Any, m))
    sh = list(a.shape)
    if int(k) % 2 == 1:  # an odd quarter-turn swaps the two plane axes
        a0, a1 = axes[0] % len(sh), axes[1] % len(sh)
        sh[a0], sh[a1] = sh[a1], sh[a0]
    return cast(Boxed, ShapedArray(tuple(sh), a.dtype))


# ---------------------------------------------------------------------------
# np.trace -- sum of a matrix diagonal. Gathers the diagonal indices (``d_getitem`` over the
# leading two axes) and sums that axis; a getitem/sum composition (default axes only).
# ---------------------------------------------------------------------------
def d_trace(a: Operand, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Var:
    shape = _logical_shape(a)
    if {axis1 % len(shape), axis2 % len(shape)} != {0, 1}:
        raise NotImplementedError(
            "trace: only the leading two axes (0, 1) are supported"
        )
    key, _ = _diag_key((shape[0], shape[1]), offset)
    return d_sum(d_getitem(a, key), axis=0)


def trace_transform_rule(
    _trace: Boxed, a: Boxed, offset: int = 0, axis1: int = 0, axis2: int = 1
) -> Boxed:
    from pycograd.trace import bind

    shape = tuple(cast(Any, a).shape)
    key, _ = _diag_key((shape[0], shape[1]), offset)
    return bind(d_sum, bind(d_getitem, a, key), axis=0)


def trace_abstract_rule(
    a: Boxed, offset: int = 0, axis1: int = 0, axis2: int = 1
) -> Boxed:
    from pycograd.shapes import ShapedArray, _aval

    av = _aval(cast(Any, a))
    return cast(Boxed, ShapedArray(tuple(av.shape[2:]), av.dtype))


# ---------------------------------------------------------------------------
# np.outer -- the outer product of two flattened vectors: ``einsum('i,j->ij', ravel(a),
# ravel(b))``; a ravel/einsum composition (full rules), so no own VJP.
# ---------------------------------------------------------------------------
def d_outer(a: Operand, b: Operand) -> Var:
    return d_einsum("i,j->ij", d_ravel(a), d_ravel(b))


def outer_transform_rule(_trace: Boxed, a: Boxed, b: Boxed) -> Boxed:
    from pycograd.trace import bind

    return bind(d_einsum, "i,j->ij", bind(d_ravel, a), bind(d_ravel, b))


def outer_abstract_rule(a: Boxed, b: Boxed) -> Boxed:
    from pycograd.shapes import ShapedArray, _aval

    av, bv = _aval(cast(Any, a)), _aval(cast(Any, b))
    na = int(np.prod(cast(Any, av.shape), dtype=np.int64))
    nb = int(np.prod(cast(Any, bv.shape), dtype=np.int64))
    return cast(Boxed, ShapedArray((na, nb), av.dtype))


# ---------------------------------------------------------------------------
# np.cross -- the 3-vector cross product along an axis. Bilinear: build each output component
# ``c_i = a_{i+1} b_{i+2} - a_{i+2} b_{i+1}`` (cyclic) from getitem/mul/sub and ``stack`` them
# back along ``axisc``. A composition (no own VJP), so it also lowers at graph capture.
# ---------------------------------------------------------------------------
def _cross_build(a: Boxed, b: Boxed, axisa: int, axisb: int, axisc: int) -> Boxed:
    from pycograd.trace import bind

    na, nb = len(_logical_shape(a)), len(_logical_shape(b))

    def comp(x: Boxed, ax: int, nd: int, i: int) -> Boxed:
        axn = ax % nd
        key = tuple(i if d == axn else slice(None) for d in range(nd))
        return bind(d_getitem, x, key)

    a0, a1, a2 = (comp(a, axisa, na, i) for i in range(3))
    b0, b1, b2 = (comp(b, axisb, nb, i) for i in range(3))
    c0 = bind(d_sub, bind(d_mul, a1, b2), bind(d_mul, a2, b1))
    c1 = bind(d_sub, bind(d_mul, a2, b0), bind(d_mul, a0, b2))
    c2 = bind(d_sub, bind(d_mul, a0, b1), bind(d_mul, a1, b0))
    out_nd = max(na, nb)
    return bind(d_stack, [c0, c1, c2], axis=axisc % out_nd)


def _cross_axes(axisa: int, axisb: int, axisc: int, axis: Any) -> tuple[int, int, int]:
    return (axis, axis, axis) if axis is not None else (axisa, axisb, axisc)


def d_cross(
    a: Operand,
    b: Operand,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: Any = None,
) -> Var:
    aa, ab, ac = _cross_axes(axisa, axisb, axisc, axis)
    return cast(Var, _cross_build(cast(Boxed, a), cast(Boxed, b), aa, ab, ac))


def cross_transform_rule(
    _trace: Boxed,
    a: Boxed,
    b: Boxed,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: Any = None,
) -> Boxed:
    aa, ab, ac = _cross_axes(axisa, axisb, axisc, axis)
    return _cross_build(a, b, aa, ab, ac)


def cross_abstract_rule(
    a: Boxed,
    b: Boxed,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: Any = None,
) -> Boxed:
    aa, ab, ac = _cross_axes(axisa, axisb, axisc, axis)
    return _cross_build(a, b, aa, ab, ac)


# ---------------------------------------------------------------------------
# np.kron -- the Kronecker product. Promote both operands to a common ndim (leading 1s), then
# ``einsum`` with interleaved labels (``ab,cd->acbd``) and reshape each axis-pair to one axis. A
# reshape/einsum composition (no own VJP), so it also lowers at graph capture.
# ---------------------------------------------------------------------------
def _kron_build(a: Boxed, b: Boxed) -> Boxed:
    import string

    from pycograd.trace import bind

    sa = tuple(int(d) for d in _logical_shape(a))
    sb = tuple(int(d) for d in _logical_shape(b))
    nd = max(len(sa), len(sb))
    sa_p = (1,) * (nd - len(sa)) + sa
    sb_p = (1,) * (nd - len(sb)) + sb
    A = bind(d_reshape, a, sa_p) if len(sa) < nd else a
    B = bind(d_reshape, b, sb_p) if len(sb) < nd else b
    la, lb = string.ascii_letters[:nd], string.ascii_letters[nd : 2 * nd]
    spec = ",".join([la, lb]) + "->" + "".join(la[i] + lb[i] for i in range(nd))
    e = bind(d_einsum, spec, A, B)
    final = tuple(sa_p[i] * sb_p[i] for i in range(nd))
    return bind(d_reshape, e, final)


def d_kron(a: Operand, b: Operand) -> Var:
    return cast(Var, _kron_build(cast(Boxed, a), cast(Boxed, b)))


def kron_transform_rule(_trace: Boxed, a: Boxed, b: Boxed) -> Boxed:
    return _kron_build(a, b)


def kron_abstract_rule(a: Boxed, b: Boxed) -> Boxed:
    return _kron_build(a, b)


def d_diag(v: Operand, k: int = 0) -> Var:
    shape = _logical_shape(v)
    key, out_shape = _diag_key(shape, k)
    if len(shape) == 1:  # construct a matrix with v on the k-diagonal
        return _scatter(v, key, out_shape, _operand_dtype(v))
    return d_getitem(v, key)  # extract the k-diagonal


def diag_transform_rule(_trace: Boxed, v: Boxed, k: int = 0) -> Boxed:
    from pycograd.trace import bind

    shape = tuple(cast(Any, v).shape)
    key, out_shape = _diag_key(shape, k)
    if len(shape) == 1:
        return bind(_scatter, v, key, out_shape, _operand_dtype(v))
    return bind(d_getitem, v, key)


def diag_abstract_rule(v: Boxed, k: int = 0) -> Boxed:
    from pycograd.shapes import ShapedArray, _aval

    a = _aval(cast(Any, v))
    _key, out_shape = _diag_key(tuple(cast(Any, a.shape)), k)
    return cast(Boxed, ShapedArray(out_shape, a.dtype))


def d_diagonal(v: Operand, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Var:
    shape = _logical_shape(v)
    if len(shape) != 2 or {axis1 % 2, axis2 % 2} != {0, 1}:
        raise NotImplementedError(
            "diagonal: only the 2-D default (axis1=0, axis2=1) is supported"
        )
    key, _ = _diag_key(shape, offset)
    return d_getitem(v, key)


def diagonal_transform_rule(
    _trace: Boxed, v: Boxed, offset: int = 0, axis1: int = 0, axis2: int = 1
) -> Boxed:
    from pycograd.trace import bind

    key, _ = _diag_key(tuple(cast(Any, v).shape), offset)
    return bind(d_getitem, v, key)


def diagonal_abstract_rule(
    v: Boxed, offset: int = 0, axis1: int = 0, axis2: int = 1
) -> Boxed:
    from pycograd.shapes import ShapedArray, _aval

    a = _aval(cast(Any, v))
    _key, out_shape = _diag_key(tuple(cast(Any, a.shape)), offset)
    return cast(Boxed, ShapedArray(out_shape, a.dtype))


# ---------------------------------------------------------------------------
# np.sort / np.partition -- a value-dependent *permutation* along an axis. The permutation
# (``argsort`` / ``argpartition``) is a stop-gradient index array, so both are a gather along
# that axis (``take_along_axis``) whose adjoint scatters the cotangent back (``put_along_axis``;
# a bijection, so no collisions). Shape-preserving.
# ---------------------------------------------------------------------------
def _take_along_var(x: Operand, perm: Array, axis: int) -> Var:
    """Gather ``x`` by the (constant) per-axis index array ``perm``; the VJP scatters the
    cotangent back to the gathered positions."""
    x, xp = _lift(x), _xp()
    out = Var(xp.take_along_axis(x.value, perm, axis), _parents=(x,))

    def _backward() -> None:
        gx = xp.zeros_like(x.value)
        xp.put_along_axis(gx, perm, out.grad, axis)
        x.grad = _accumulate(x.grad, gx)

    out._backward = _backward
    return out


def d_sort(x: Operand, axis: int = -1) -> Var:
    x = _lift(x)
    _reject_complex("sort", x)
    perm = _xp().argsort(x.value, axis=axis)
    return _take_along_var(x, perm, axis)


def d_partition(x: Operand, kth: Any, axis: int = -1) -> Var:
    x = _lift(x)
    _reject_complex("partition", x)
    perm = _xp().argpartition(x.value, kth, axis=axis)
    return _take_along_var(x, perm, axis)


def _sort_like_abstract(x: Boxed, *args: Any, **kwargs: Any) -> Boxed:
    from pycograd.shapes import abstract_unary  # sort/partition preserve shape

    return cast(Boxed, abstract_unary(cast(Any, x)))


# ---------------------------------------------------------------------------
# np.select(condlist, choicelist, default) -- pick from ``choicelist`` by the first true
# condition: a right-fold of ``where`` (``where(c0, ch0, where(c1, ch1, ..., default))``).
# The conditions are boolean (stop-gradient); a composition of ``d_where`` (no own VJP).
# ---------------------------------------------------------------------------
def d_select(condlist: Any, choicelist: Any, default: Any = 0) -> Var:
    acc: Operand = default
    for cond, choice in zip(reversed(list(condlist)), reversed(list(choicelist))):
        acc = d_where(np.asarray(cond), choice, acc)
    return _lift(cast(Operand, acc))


def select_transform_rule(
    _trace: Boxed, condlist: Any, choicelist: Any, default: Any = 0
) -> Boxed:
    from pycograd.trace import Tracer, bind

    acc: Boxed = default
    for cond, choice in zip(reversed(list(condlist)), reversed(list(choicelist))):
        # The condition is stop-gradient; pass a box (Var/Tracer) straight through. Only
        # array-ify a plain condition -- ``np.asarray`` on a tracer makes a 0-d *object*
        # array wrapping it, which then breaks the where mask arithmetic in the graph.
        c = cond if isinstance(cond, (Var, Tracer)) else np.asarray(cond)
        acc = bind(d_where, c, choice, acc)
    return acc


def select_abstract_rule(condlist: Any, choicelist: Any, default: Any = 0) -> Boxed:
    from pycograd.shapes import abstract_where

    acc: Any = default
    for cond, choice in zip(reversed(list(condlist)), reversed(list(choicelist))):
        acc = abstract_where(cast(Any, cond), cast(Any, choice), cast(Any, acc))
    return cast(Boxed, acc)


# ---------------------------------------------------------------------------
# np.gradient -- central-difference numerical gradient (unit spacing, edge_order=1, the numpy
# default). Along one axis: interior ``(f[2:] - f[:-2]) / 2`` with one-sided first-order
# boundaries, assembled by ``concatenate`` of getitem slices (all with full rules). Returns a
# list (one array per axis) for ``axis=None`` / a tuple of axes, else a single array.
# Non-default spacing (``varargs``) is unsupported.
# ---------------------------------------------------------------------------
def _gradient_axes(ndim: int, axis: Any) -> tuple[list[int], bool]:
    if axis is None:
        return list(range(ndim)), True
    if isinstance(axis, (tuple, list)):
        return [a % ndim for a in axis], True
    return [axis % ndim], False


def _gradient_along(f: Boxed, ax: int, ndim: int) -> Boxed:
    from pycograd.trace import bind

    def sl(s: slice) -> tuple:
        return tuple(s if i == ax else slice(None) for i in range(ndim))

    def g(s: slice) -> Boxed:
        return bind(d_getitem, f, sl(s))

    first = bind(d_sub, g(slice(1, 2)), g(slice(0, 1)))
    interior = bind(d_mul, bind(d_sub, g(slice(2, None)), g(slice(0, -2))), 0.5)
    last = bind(d_sub, g(slice(-1, None)), g(slice(-2, -1)))
    return bind(d_concatenate, [first, interior, last], axis=ax)


def _gradient_impl(f: Boxed, ndim: int, axis: Any) -> Boxed:
    axes, as_list = _gradient_axes(ndim, axis)
    results = [_gradient_along(f, ax, ndim) for ax in axes]
    return cast(Boxed, results) if as_list else results[0]


def d_gradient(f: Operand, *varargs: Any, axis: Any = None, edge_order: int = 1) -> Var:
    if varargs:
        raise NotImplementedError("gradient: non-default spacing is not supported")
    return cast(Var, _gradient_impl(cast(Boxed, f), _logical_ndim(f), axis))


def gradient_transform_rule(
    _trace: Boxed, f: Boxed, *varargs: Any, axis: Any = None, edge_order: int = 1
) -> Boxed:
    return _gradient_impl(f, len(cast(Any, f).shape), axis)


def gradient_abstract_rule(
    f: Boxed, *varargs: Any, axis: Any = None, edge_order: int = 1
) -> Boxed:
    from pycograd.shapes import ShapedArray, _aval

    a = _aval(cast(Any, f))
    axes, as_list = _gradient_axes(len(a.shape), axis)
    if as_list:
        return cast(Boxed, [ShapedArray(a.shape, a.dtype) for _ in axes])
    return cast(Boxed, ShapedArray(a.shape, a.dtype))


# ---------------------------------------------------------------------------
# np.append(arr, values, axis) -- a concatenate (``axis=None`` ravels both operands first).
# A composition of ``d_concatenate`` (+ ``d_ravel``), so no own VJP. (A python-list ``arr`` is
# the separate np.array-of-boxes gap; array operands work.)
# ---------------------------------------------------------------------------
def d_append(arr: Operand, values: Operand, axis: Any = None) -> Var:
    if axis is None:
        return d_concatenate([d_ravel(arr), d_ravel(values)], axis=0)
    return d_concatenate([arr, values], axis=axis)


def append_transform_rule(
    _trace: Boxed, arr: Boxed, values: Boxed, axis: Any = None
) -> Boxed:
    from pycograd.trace import bind

    if axis is None:
        return bind(d_concatenate, [bind(d_ravel, arr), bind(d_ravel, values)], axis=0)
    return bind(d_concatenate, [arr, values], axis=axis)


def append_abstract_rule(arr: Boxed, values: Boxed, axis: Any = None) -> Boxed:
    from pycograd.shapes import ShapedArray, _aval, abstract_concatenate

    a, v = _aval(cast(Any, arr)), _aval(cast(Any, values))
    if axis is None:
        n = int(np.prod(cast(Any, a.shape), dtype=np.int64)) + int(
            np.prod(cast(Any, v.shape), dtype=np.int64)
        )
        return cast(Boxed, ShapedArray((n,), a.dtype))
    return cast(Boxed, abstract_concatenate([cast(Any, arr), cast(Any, values)], axis))


def split_abstract_rule(which: str) -> Callable[..., Boxed]:
    def rule(x: Boxed, *args: Any, **kwargs: Any) -> Boxed:
        from pycograd.shapes import ShapedArray, _aval

        a = _aval(cast(Any, x))
        ind, axis = _resolve_split(which, len(a.shape), args, kwargs)
        slices = split_slices(tuple(cast(Any, a.shape)), ind, axis)
        return cast(
            Boxed,
            [
                ShapedArray(_sliced_shape(tuple(cast(Any, a.shape)), sl), a.dtype)
                for sl in slices
            ],
        )

    return rule


def split_transform_rule(which: str) -> Callable[..., Boxed]:
    """Forward (jvp) *and* batching (vmap) rule for a split variant: cut ``x`` into slices via
    ``d_getitem`` (which carries both rules), returning a *list* of pieces."""
    from pycograd.trace import bind

    def rule(_trace: Boxed, x: Boxed, *args: Any, **kwargs: Any) -> Boxed:
        ind, axis = _resolve_split(which, len(cast(Any, x).shape), args, kwargs)
        slices = split_slices(tuple(cast(Any, x).shape), ind, axis)
        return cast(Boxed, [bind(d_getitem, x, sl) for sl in slices])

    return rule


# ---------------------------------------------------------------------------
# Cumulative sum -- a fused primitive (linear; no composition expresses the prefix
# sum). The VJP is a reverse cumulative sum (flip -> cumsum -> flip).
# ---------------------------------------------------------------------------
def d_cumsum(x: Operand, axis: int | None = None) -> Var:
    from pycograd.trace import Tracer, bind

    if axis is None:  # flatten-all: ravel and cumsum along the single axis (1-D result)
        return cast(Var, d_cumsum(d_ravel(x), axis=0))
    if isinstance(x, Tracer):
        return cast(Var, bind(d_cumsum, x, axis=axis))
    x, xp = _lift(x), _xp()
    out = Var(xp.cumsum(x.value, axis=axis), _parents=(x,))

    def _backward() -> None:
        g = out.grad
        x.grad = _accumulate(
            x.grad, xp.flip(xp.cumsum(xp.flip(g, axis=axis), axis=axis), axis=axis)
        )

    out._backward = _backward
    _record_vjp(out, d_cumsum, (x,), {"axis": axis})
    return out


# ---------------------------------------------------------------------------
# Reductions.
# ---------------------------------------------------------------------------
def d_sum(x: Operand, axis: Axis = None, keepdims: bool = False) -> Var:
    x, xp = _lift(x), _xp()
    # numpy's ndarray.sum overloads split on keepdims being a Literal once axis is
    # a union, so a plain bool fails to match any variant (newer stubs only).
    summed = x.value.sum(axis=axis, keepdims=keepdims)  # type: ignore[call-overload]
    out = Var(summed, _parents=(x,))

    def _backward() -> None:
        g = out.grad
        if axis is not None and not keepdims:
            g = xp.expand_dims(g, axis)
        x.grad = _accumulate(x.grad, xp.broadcast_to(g, x.value.shape))

    out._backward = _backward
    _record_vjp(out, d_sum, (x,), {"axis": axis, "keepdims": keepdims})
    return out


def d_prod(x: Operand, axis: Axis = None, keepdims: bool = False) -> Var:
    # Reduction by multiplication. d/dx_i prod(x) = prod(x) / x_i (the product of the other
    # elements); the backward broadcasts the keepdims product over the reduced axis and
    # divides by x. (Like autograd, this assumes the reduced slice has no exact zero.)
    x, xp = _lift(x), _xp()
    pk = xp.prod(x.value, axis=axis, keepdims=True)  # for broadcasting in backward
    out = Var(
        x.value.prod(axis=axis, keepdims=keepdims),  # type: ignore[call-overload]
        _parents=(x,),
    )

    def _backward() -> None:
        g = out.grad
        if axis is not None and not keepdims:
            g = xp.expand_dims(g, axis)
        x.grad = _accumulate(x.grad, xp.broadcast_to(g, x.value.shape) * pk / x.value)

    out._backward = _backward
    _record_vjp(out, d_prod, (x,), {"axis": axis, "keepdims": keepdims})
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
    # Variance is ``mean(|x - mean|^2)``; for complex x the squared magnitude
    # ``|centered|^2`` (real) is required -- ``centered*centered`` would be the complex
    # square. ``d_abs(centered)**2`` reduces to ``centered**2`` for real x.
    sq = d_square(d_abs(centered)) if x.value.dtype.kind == "c" else centered * centered
    return d_sum(sq, axis=axis, keepdims=keepdims) / (n - ddof)


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
    _reject_complex(getattr(prim, "__name__", "max/min"), x)
    kept = reducer(x.value, axis=axis, keepdims=True)
    out = Var(reducer(x.value, axis=axis, keepdims=keepdims), _parents=(x,))

    def _backward() -> None:
        g = out.grad
        if axis is not None and not keepdims:
            g = xp.expand_dims(g, axis)
        mask = (x.value == kept).astype(float)
        mask /= mask.sum(axis=axis, keepdims=True)
        x.grad = _accumulate(x.grad, mask * g)

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
# Fused softmax / logsumexp.
#
# Both are *fused* primitives (the ``d_gated_act`` / ``d_sigmoid`` template): one tape
# node + one closed-form VJP instead of the ~6-node max/sub/exp/sum/log chain the
# composed ``functional.softmax`` / ``logsumexp`` would unroll to. Both compute the
# numerically *stable* max-shifted form internally, so fusing a naive
# ``log(sum(exp(x)))`` into ``d_logsumexp`` is also an overflow fix. Tape-only (no numpy
# name, so no ``_RULES`` entry); user code calls them directly, so under an enclosing
# transform an operand is a higher-level tracer -- dispatch through ``bind`` so the
# registered jvp/vmap/abstract rule runs. The ``axis`` is a bind *param* (not baked) so
# vmap can shift it past the inserted batch axis.
# ---------------------------------------------------------------------------
def d_softmax(x: Operand, axis: Axis = -1) -> Var:
    from pycograd.trace import Tracer, bind

    if isinstance(x, Tracer):
        return cast(Var, bind(d_softmax, x, axis=axis))
    x, xp = _lift(x), _xp()
    e = xp.exp(x.value - xp.max(x.value, axis=axis, keepdims=True))
    y = e / xp.sum(e, axis=axis, keepdims=True)
    out = Var(y, _parents=(x,))

    def _backward() -> None:
        # dx = y * (g - sum(y*g, axis, keepdims)) -- the standard stable softmax backward.
        g = out.grad
        dx = y * (g - xp.sum(y * g, axis=axis, keepdims=True))
        x.grad = _accumulate(x.grad, dx)

    out._backward = _backward
    _record_vjp(out, d_softmax, (x,), {"axis": axis})
    return out


def d_logsumexp(x: Operand, axis: Axis = None, keepdims: bool = False) -> Var:
    from pycograd.trace import Tracer, bind

    if isinstance(x, Tracer):
        return cast(Var, bind(d_logsumexp, x, axis=axis, keepdims=keepdims))
    x, xp = _lift(x), _xp()
    m = xp.max(x.value, axis=axis, keepdims=True)
    e = xp.exp(x.value - m)
    s = xp.sum(e, axis=axis, keepdims=True)
    lse_kd = m + xp.log(s)  # keepdims-shaped result
    if keepdims:
        out_val = lse_kd
    elif axis is None:
        out_val = lse_kd.reshape(())
    else:
        axes = axis if isinstance(axis, tuple) else (axis,)
        out_val = xp.squeeze(lse_kd, axis=axes)
    out = Var(out_val, _parents=(x,))
    sm = e / s  # softmax = d(lse)/dx

    def _backward() -> None:
        # dx = softmax(x) * g, broadcasting g back over the reduced axis.
        g = out.grad
        if axis is not None and not keepdims:
            g = xp.expand_dims(g, axis)
        x.grad = _accumulate(x.grad, sm * g)

    out._backward = _backward
    _record_vjp(out, d_logsumexp, (x,), {"axis": axis, "keepdims": keepdims})
    return out


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
            part.grad = _accumulate(part.grad, gpart)

    out._backward = _backward
    _record_vjp(out, d_concatenate, tuple(parts), {"axis": axis})
    return out


def d_transpose(x: Operand, axes: tuple[int, ...] | None = None) -> Var:
    x, xp = _lift(x), _xp()
    out = Var(xp.transpose(x.value, axes), _parents=(x,))

    def _backward() -> None:
        if axes is None:
            x.grad = _accumulate(x.grad, xp.transpose(out.grad))
        else:
            # np.argsort over the (host-side) axes tuple; the transpose runs on device.
            x.grad = _accumulate(
                x.grad, xp.transpose(out.grad, tuple(np.argsort(axes)))
            )

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
        x.grad = _accumulate(x.grad, out.grad.reshape(x.value.shape))

    out._backward = _backward
    _record_vjp(out, d_reshape, (x,))
    return out


def d_astype(x: Operand, dtype: DTypeLike, **_ignored: Any) -> Var:
    """Cast ``x`` to floating-or-complex dtype ``dtype`` -- the in-graph precision/kind cast
    behind mixed precision (e.g. ``x.astype("float64")`` inside a float32 tape) and real<->
    complex embedding (``x.astype("complex128")``).

    A cast is *linear*, so the VJP casts the cotangent back to the input's dtype (a
    complex->real cast-back drops the imaginary part via ``_accumulate``, which is the
    adjoint of the real->complex embedding). ``resolve_dtype`` rejects ints/bools (a ``Var``
    holds real- or complex-valued tensors), so casting to an integer dtype -- an index/label
    -- is not a differentiable tape op. Extra numpy ``astype`` kwargs (``order``/``casting``/
    ``copy``/``subok``/``device``) are accepted and ignored.
    """
    x, xp = _lift(x), _xp()
    target = resolve_dtype(dtype)
    # Pass ``dtype=target`` so ``Var.__init__`` keeps the cast dtype rather than re-casting
    # the value back to the ambient ``current_dtype()``.
    out = Var(xp.asarray(x.value).astype(target), _parents=(x,), dtype=target)

    def _backward() -> None:
        # ``_accumulate`` reconciles the dtype (complex cotangent -> real input keeps the
        # real part), so no explicit cast here -- avoids a noisy ComplexWarning.
        x.grad = _accumulate(x.grad, out.grad)

    out._backward = _backward
    _record_vjp(out, d_astype, (x,), {"dtype": target})
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
        x.grad = _accumulate(x.grad, _unbroadcast(out.grad, x.value.shape))

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
# numpy's stack family accepts a ``dtype=``/``casting=`` kwarg (and ``row_stack`` forwards
# them to ``vstack``); we compute in the operand dtype, so ``**_`` swallows them.
def d_stack(seq: Sequence[Operand], axis: int = 0, **_: Any) -> Var:
    # join along a NEW axis: expand each input at ``axis``, then concatenate there.
    return d_concatenate([d_expand_dims(s, axis) for s in seq], axis=axis)


def _atleast_1d_part(x: Operand) -> Var:
    # numpy's hstack/column_stack run atleast_1d on each input, so a single 1-D array passed as
    # the sequence (its elements iterate to 0-d scalars) still concatenates.
    x = _lift(x)
    return d_reshape(x, (1,)) if x.value.ndim == 0 else x


def _atleast_2d_row(x: Operand) -> Var:
    x = _lift(x)
    if x.value.ndim == 0:
        return d_reshape(x, (1, 1))
    if x.value.ndim == 1:
        return d_reshape(x, (1, x.value.shape[0]))
    return x


def d_vstack(seq: Sequence[Operand], **_: Any) -> Var:
    # row-wise: 1-D inputs become single rows, then concatenate along axis 0.
    return d_concatenate([_atleast_2d_row(s) for s in seq], axis=0)


def d_hstack(seq: Sequence[Operand], **_: Any) -> Var:
    # column-wise: concatenate along axis 1, except 1-D inputs join along axis 0.
    parts = [_atleast_1d_part(s) for s in seq]
    axis = 0 if all(p.value.ndim == 1 for p in parts) else 1
    return d_concatenate(parts, axis=axis)


def d_column_stack(seq: Sequence[Operand], **_: Any) -> Var:
    # 1-D inputs become columns ((n,) -> (n, 1)); then concatenate along axis 1.
    parts = []
    for s in seq:
        p = _atleast_1d_part(s)
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


def d_dstack(seq: Sequence[Operand], **_: Any) -> Var:
    # depth-wise: stack along a third axis (after promoting inputs to 3-D).
    return d_concatenate([_atleast_3d_depth(s) for s in seq], axis=2)


# numpy's ``np.r_[...]`` / ``np.c_[...]`` index-expression objects: row- and column-wise
# concatenation of the bracketed pieces, where an int ``slice`` (``1:10``) expands to an
# ``arange``. Routed here from the subscript handler (tracer.py); the array/box pieces stay
# differentiable, the ``arange`` pieces are constants. (String directives are not supported.)
def _index_expr_pieces(key: Any) -> list:
    pieces = list(key) if isinstance(key, tuple) else [key]
    out: list = []
    for p in pieces:
        if isinstance(p, slice):
            start = 0 if p.start is None else p.start
            step = 1 if p.step is None else p.step
            out.append(_xp().arange(start, p.stop, step))
        else:
            out.append(p)
    return out


def d_r_(key: Any) -> Var:
    # row-wise: concatenate the pieces along axis 0 (numpy's default for ``r_``). Routed
    # through ``bind`` so tracer pieces (under jvp/vmap/capture) dispatch to their level.
    from pycograd.trace import bind

    return cast(Var, bind(d_concatenate, _index_expr_pieces(key), axis=0))


def d_c_(key: Any) -> Var:
    # column-wise: each piece becomes a column (1-D -> (n, 1)), concatenated along axis 1.
    # Built directly from reshape/concatenate (not ``d_column_stack``) so it lowers to
    # graph-differentiable primitives under capture.
    from pycograd.trace import bind

    cols = []
    for p in _index_expr_pieces(key):
        shp = _logical_shape(p)
        cols.append(bind(d_reshape, p, (shp[0], 1)) if len(shp) == 1 else p)
    return cast(Var, bind(d_concatenate, cols, axis=1))


# ---------------------------------------------------------------------------
# np.array over differentiable leaves -- ``np.array([v0, v1, ...])`` where the (possibly
# nested) list/tuple holds ``Var``/``Tracer`` boxes. numpy can't build such an array (the
# boxes have no array conversion), so we *stack* the leaves: each nesting level becomes a
# ``d_stack`` along a fresh leading axis. A list with no boxes (and ``np.array`` of a single
# box, an identity) passes straight through to numpy -- so intercepting the pervasive
# ``np.array`` is transparent for every ordinary call. A composition of ``d_stack`` (full
# rules), so no own VJP.
# ---------------------------------------------------------------------------
def _contains_box(obj: object) -> bool:
    from pycograd.trace import Tracer

    if isinstance(obj, (Var, Tracer)):
        return True
    if isinstance(obj, (list, tuple)):
        return any(_contains_box(o) for o in obj)
    return False


def _array_build(obj: Any) -> Boxed:
    from pycograd.trace import bind

    if isinstance(obj, (list, tuple)):
        return bind(d_stack, [_array_build(o) for o in obj], axis=0)
    return obj  # a single box -> identity (np.array of an array is a copy)


def d_array(obj: Any, *args: Any, **kwargs: Any) -> Var:
    if not _contains_box(obj):
        return cast(Var, np.array(obj, *args, **kwargs))
    return cast(Var, _array_build(obj))


def array_transform_rule(_trace: Boxed, obj: Any, *args: Any, **kwargs: Any) -> Boxed:
    if not _contains_box(obj):
        return cast(Boxed, np.array(obj, *args, **kwargs))
    return _array_build(obj)


def array_abstract_rule(obj: Any, *args: Any, **kwargs: Any) -> Boxed:
    if not _contains_box(obj):
        return cast(Boxed, np.array(obj, *args, **kwargs))
    return _array_build(obj)


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
    # operator primitives -- the *operators* (``+ - * / **``) reach these through ``Var``'s
    # dunders directly, but the numpy *function* forms (``np.add`` etc.) are intercepted, so
    # those callables are mapped here too (the rule tables are keyed by the primitive, so the
    # function and operator share one set of rules). Comparison operators stay empty (no
    # numpy callable / their gradient is zero).
    # numpy *and* ``operator`` function forms map to the operator primitives. The
    # ``operator.*`` callables matter when the differentiated function is e.g. ``op.mul``
    # itself (intercepted as a call), since the bare ``*`` on two same-level tracers is not
    # otherwise routed.
    d_add: (np.add, operator.add),
    d_sub: (np.subtract, operator.sub),
    d_mul: (np.multiply, operator.mul),
    d_div: (np.divide, np.true_divide, operator.truediv),
    d_neg: (np.negative, operator.neg),
    d_pow: (np.power, operator.pow),
    d_mod: (np.mod, np.remainder, operator.mod),
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
    d_tan: (np.tan, math.tan),
    d_arcsin: (np.arcsin, math.asin),
    d_arccos: (np.arccos, math.acos),
    d_arctanh: (np.arctanh, math.atanh),
    d_arcsinh: (np.arcsinh, math.asinh),
    d_arccosh: (np.arccosh, math.acosh),
    d_exp2: (np.exp2,),
    d_log2: (np.log2, math.log2),
    d_log10: (np.log10, math.log10),
    # ``radians`` / ``degrees`` are numpy aliases for ``deg2rad`` / ``rad2deg``.
    d_deg2rad: (np.deg2rad, np.radians, math.radians),
    d_rad2deg: (np.rad2deg, np.degrees, math.degrees),
    d_sign: (np.sign,),
    d_ceil: (np.ceil,),
    d_floor: (np.floor,),
    d_log1p: (np.log1p, math.log1p),
    d_expm1: (np.expm1, math.expm1),
    # ``np.fabs`` is ``abs`` for real inputs -- reuse ``d_abs``'s kernel/rules.
    d_abs: (np.abs, np.fabs),
    # Complex component ops (non-holomorphic; excluded from the conj wrap, see
    # ``_NONHOLOMORPHIC``).
    d_conj: (np.conj, np.conjugate),
    d_real: (np.real,),
    d_real_if_close: (np.real_if_close,),
    d_nan_to_num: (np.nan_to_num,),
    d_imag: (np.imag,),
    d_angle: (np.angle,),
    d_square: (np.square,),
    d_reciprocal: (np.reciprocal,),
    # elementwise binary
    d_maximum: (np.maximum,),
    d_minimum: (np.minimum,),
    d_fmax: (np.fmax,),
    d_fmin: (np.fmin,),
    d_logaddexp: (np.logaddexp,),
    d_logaddexp2: (np.logaddexp2,),
    # selection
    d_where: (np.where,),
    d_clip: (np.clip,),
    # reductions
    d_sum: (np.sum,),
    d_prod: (np.prod,),
    d_mean: (np.mean,),
    d_var: (np.var,),
    d_std: (np.std,),
    d_max: (np.max, np.amax),
    d_min: (np.min, np.amin),
    # linear algebra / shape / structure
    _matmul: (np.matmul,),
    d_dot: (np.dot,),
    d_inner: (np.inner,),
    d_tensordot: (np.tensordot,),
    d_einsum: (np.einsum,),
    d_cumsum: (np.cumsum,),
    d_transpose: (np.transpose,),
    d_moveaxis: (np.moveaxis,),
    d_swapaxes: (np.swapaxes,),
    d_rollaxis: (np.rollaxis,),
    d_tril: (np.tril,),
    d_triu: (np.triu,),
    d_roll: (np.roll,),
    d_pad: (np.pad,),
    d_repeat: (np.repeat,),
    d_tile: (np.tile,),
    d_split: (np.split,),
    d_array_split: (np.array_split,),
    d_vsplit: (np.vsplit,),
    d_hsplit: (np.hsplit,),
    d_dsplit: (np.dsplit,),
    d_diff: (np.diff,),
    d_diag: (np.diag,),
    d_diagonal: (np.diagonal,),
    d_sort: (np.sort,),
    d_partition: (np.partition,),
    d_select: (np.select,),
    d_gradient: (np.gradient,),
    d_append: (np.append,),
    d_flipud: (np.flipud,),
    d_fliplr: (np.fliplr,),
    d_rot90: (np.rot90,),
    d_trace: (np.trace,),
    d_outer: (np.outer,),
    d_cross: (np.cross,),
    d_kron: (np.kron,),
    d_array: (np.array,),
    d_ravel: (np.ravel,),
    d_squeeze: (np.squeeze,),
    d_atleast_1d: (np.atleast_1d,),
    d_atleast_2d: (np.atleast_2d,),
    d_atleast_3d: (np.atleast_3d,),
    d_reshape: (np.reshape,),
    d_astype: (np.astype,),
    d_expand_dims: (np.expand_dims,),
    d_concatenate: (np.concatenate,),
    d_stack: (np.stack,),
    d_vstack: (np.vstack, np.row_stack),
    d_hstack: (np.hstack,),
    d_column_stack: (np.column_stack,),
    d_dstack: (np.dstack,),
}

_INTERCEPT: dict[Prim, Prim] = {fn: impl for impl, fns in _RULES.items() for fn in fns}


# ---------------------------------------------------------------------------
# Lowering rules: ops that carry no VJP of their own because they *expand* into other
# primitives (a contraction einsum, a transpose, a stack of getitems, ...). Their single
# trace-agnostic rule re-``bind``s those primitives, so it runs identically under every trace
# level -- ``jvp`` (forward._JVP_FOR), ``vmap`` (batching._RULE_FOR), and graph ``capture``
# (which would otherwise record an opaque node the graph reverse pass can't differentiate).
# ``capture`` consumes this table directly; the forward/batch tables list the same rules
# inline, and ``test_lowering_rules_consistent`` guards that they cover exactly these ops and
# that none of them also claims a ``_VJP_FOR`` entry (a lowering op has no VJP of its own).
# (``sort``/``partition`` are NOT here: their permutation is value-dependent, so their forward
# rule needs the concrete primal and they stay genuine primitives.)
# ---------------------------------------------------------------------------
_LOWERING_RULES: dict[Prim, Rule] = {
    d_dot: contraction_transform_rule(d_dot),
    d_inner: contraction_transform_rule(d_inner),
    d_tensordot: contraction_transform_rule(d_tensordot),
    d_moveaxis: _transpose_lowering_transform(moveaxis_perm),
    d_swapaxes: _transpose_lowering_transform(swapaxes_perm),
    d_rollaxis: _transpose_lowering_transform(rollaxis_perm),
    d_tril: _tri_lowering_transform(np.tril),
    d_triu: _tri_lowering_transform(np.triu),
    d_split: split_transform_rule("split"),
    d_array_split: split_transform_rule("array_split"),
    d_vsplit: split_transform_rule("vsplit"),
    d_hsplit: split_transform_rule("hsplit"),
    d_dsplit: split_transform_rule("dsplit"),
    d_diff: diff_transform_rule,
    d_diag: diag_transform_rule,
    d_diagonal: diagonal_transform_rule,
    d_select: select_transform_rule,
    d_gradient: gradient_transform_rule,
    d_append: append_transform_rule,
    d_flipud: _flip_transform_rule(0),
    d_fliplr: _flip_transform_rule(1),
    d_rot90: rot90_transform_rule,
    d_trace: trace_transform_rule,
    d_outer: outer_transform_rule,
    d_cross: cross_transform_rule,
    d_kron: kron_transform_rule,
    d_array: array_transform_rule,
    d_ravel: _reshape_lowering_transform(ravel_shape),
    d_squeeze: _reshape_lowering_transform(squeeze_shape),
    d_atleast_1d: _reshape_lowering_transform(atleast_1d_shape),
    d_atleast_2d: _reshape_lowering_transform(atleast_2d_shape),
    d_atleast_3d: _reshape_lowering_transform(atleast_3d_shape),
}


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
# So each VJP is effectively spelled twice -- the raw ``.grad`` closure on ``_unary`` /
# ``_binary`` (the fast base path) and the ``bind``-riding rule here. This duplication is
# deliberate: collapsing the base path onto this differentiable one (always routing through
# ``bind``) was benchmarked ~2x slower on the backward pass, since it builds tape nodes for
# cotangents even when no second-order pass needs them. The two are kept in parity by the
# finite-diff / higher-order tests rather than by sharing one implementation.
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


# A VJP rule reads its primals' shape/ndim/dtype to size the cotangent. Those primals are
# recorded ``Var``s on the eager higher-order path (read ``.value.*``) but ``GraphTracer``s
# when ``_grad_graph`` differentiates a captured graph (read the aval via ``.shape``/
# ``.dtype``). These accessors are byte-identical for a ``Var``, so the eager path is
# unchanged; they just let the same rules also build a backward *graph*.
def _pshape(p: Boxed) -> "tuple[int, ...]":
    return p.value.shape if isinstance(p, Var) else tuple(cast(Any, p).shape)


def _pndim(p: Boxed) -> int:
    return p.value.ndim if isinstance(p, Var) else len(cast(Any, p).shape)


def _pdtype(p: Boxed) -> "np.dtype":
    return p.value.dtype if isinstance(p, Var) else cast(Any, p).dtype


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
        d_sigmoid: lambda a: _b(
            d_mul, _b(d_sigmoid, a), _b(d_sub, 1.0, _b(d_sigmoid, a))
        ),
        d_sinh: lambda a: _b(d_cosh, a),
        d_cosh: lambda a: _b(d_sinh, a),
        d_arctan: lambda a: _b(d_reciprocal, _b(d_add, 1.0, _b(d_mul, a, a))),
        # tan' = 1/cos^2
        d_tan: lambda a: _b(d_reciprocal, _b(d_mul, _b(d_cos, a), _b(d_cos, a))),
        # arcsin' = 1/sqrt(1-a^2); arccos' = -that
        d_arcsin: lambda a: _b(
            d_reciprocal, _b(d_sqrt, _b(d_sub, 1.0, _b(d_mul, a, a)))
        ),
        d_arccos: lambda a: _b(
            d_neg, _b(d_reciprocal, _b(d_sqrt, _b(d_sub, 1.0, _b(d_mul, a, a))))
        ),
        # arctanh' = 1/(1-a^2)
        d_arctanh: lambda a: _b(d_reciprocal, _b(d_sub, 1.0, _b(d_mul, a, a))),
        # arcsinh' = 1/sqrt(a^2+1); arccosh' = 1/sqrt(a^2-1)
        d_arcsinh: lambda a: _b(
            d_reciprocal, _b(d_sqrt, _b(d_add, _b(d_mul, a, a), 1.0))
        ),
        d_arccosh: lambda a: _b(
            d_reciprocal, _b(d_sqrt, _b(d_sub, _b(d_mul, a, a), 1.0))
        ),
        # exp2' = ln2 * 2^a; log2' = 1/(a ln2); log10' = 1/(a ln10)
        d_exp2: lambda a: _b(d_mul, math.log(2.0), _b(d_exp2, a)),
        d_log2: lambda a: _b(d_reciprocal, _b(d_mul, math.log(2.0), a)),
        d_log10: lambda a: _b(d_reciprocal, _b(d_mul, math.log(10.0), a)),
        # constant-derivative angle conversions; zero-derivative step functions
        d_deg2rad: lambda a: math.pi / 180.0,
        d_rad2deg: lambda a: 180.0 / math.pi,
        d_sign: lambda a: 0.0,
        d_ceil: lambda a: 0.0,
        d_floor: lambda a: 0.0,
        d_log1p: lambda a: _b(d_reciprocal, _b(d_add, 1.0, a)),
        d_expm1: lambda a: _b(d_exp, a),
        d_square: lambda a: _b(d_mul, 2.0, a),
        d_reciprocal: lambda a: _b(d_neg, _b(d_reciprocal, _b(d_mul, a, a))),
    }


def _vjp_unary_for(prim: Callable[..., Var]) -> Callable[..., list[Boxed]]:
    derivs = _UNARY_DERIV

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
    # Real: d|x|/dx = sign(x), a piecewise-constant derivative -> a stop-gradient constant.
    # Complex: |z| is non-holomorphic; the real-adjoint is ``g * z/|z|``. Excluded from the
    # holomorphic conj wrap (it is already the final real-coords gradient).
    (p,) = primals
    if _pdtype(p).kind == "c":
        z = operands[0]
        return [_b(d_mul, g, _b(d_div, z, _b(d_abs, z)))]
    return [_b(d_mul, g, _const_like(_xp().sign(p.value)))]


def _vjp_conj(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # Real-adjoint of conj is conj (an involution); identity on a real cotangent.
    return [_b(d_conj, g)]


def _vjp_real_if_close(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    return [g]  # identity on the values -> identity on the cotangent


def _vjp_nan_to_num(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    return [_b(d_mul, g, params["mask"])]  # zero where the input was nan/inf


def _vjp_real(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # Adjoint of Re embeds the real cotangent back into the complex input's space; a real
    # ``g`` promotes into the complex parent grad on accumulation, so pass it through.
    return [g]


def _vjp_imag(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # Adjoint of Im is multiply-by-i for a complex input (real g -> imaginary cotangent);
    # for a real input, Im is identically zero, so no gradient flows.
    (p,) = primals
    if _pdtype(p).kind == "c":
        return [_b(d_mul, 1j, g)]
    return [_b(d_mul, 0.0, g)]


def _vjp_angle(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # Real-adjoint of angle is ``g * i*z/|z|^2`` (complex input); zero for a real input.
    (p,) = primals
    if not _pdtype(p).kind == "c":
        return [_b(d_mul, 0.0, g)]
    z = operands[0]
    return [_b(d_mul, g, _b(d_div, _b(d_mul, 1j, z), _b(d_square, _b(d_abs, z))))]


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


# Shared local derivatives for the elementwise nonlinear binaries -- like ``_UNARY_DERIV``,
# each is written once and consumed by *both* the reverse rule (below) and the forward jvp
# (``forward.py``), so e.g. ``b*a**(b-1)`` and the gate coefficients are not restated.
def _pow_base_deriv(a: Boxed, b: Boxed) -> Boxed:
    # d(a**b)/da = b * a**(b-1). ``b`` may be a raw constant exponent or a tracer.
    return _b(d_mul, b, _b(d_pow, a, _b(d_sub, b, 1.0)))


def _gated_act_coeffs(f: Boxed, s: Boxed) -> "tuple[Boxed, Boxed]":
    # gate = tanh(f) * sigmoid(s); returns (d/df, d/ds) of the gate w.r.t. its inputs.
    sig = _b(d_sigmoid, s)
    tanh_f = _b(d_tanh, f)
    df_coeff = _b(d_mul, sig, _b(d_sub, 1.0, _b(d_mul, tanh_f, tanh_f)))
    ds_coeff = _b(d_mul, tanh_f, _b(d_mul, sig, _b(d_sub, 1.0, sig)))
    return df_coeff, ds_coeff


def _vjp_gated_act(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # gate = tanh(f) * sigmoid(s); built bind-riding so the cotangent graph differentiates.
    f, s = operands
    df_coeff, ds_coeff = _gated_act_coeffs(f, s)
    return [_b(d_mul, g, df_coeff), _b(d_mul, g, ds_coeff)]


def _vjp_div(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    a, b = operands
    return [_b(d_div, g, b), _b(d_neg, _b(d_div, _b(d_mul, g, a), _b(d_mul, b, b)))]


def _vjp_mod(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # a % b = a - b*floor(a/b): d/da = 1, d/db = -floor(a/b). ``floor`` is piecewise
    # constant (``d_floor`` has zero derivative), so the second-order term through it is
    # correctly zero.
    a, b = operands
    return [g, _b(d_mul, _b(d_neg, _b(d_floor, _b(d_div, a, b))), g)]


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
    ga = _b(d_mul, g, _pow_base_deriv(a, p))
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
    na, nb = _pndim(pa), _pndim(pb)
    if na == 1 and nb == 1:  # inner product: g is a scalar
        return [_b(d_mul, g, b), _b(d_mul, g, a)]
    if na == 2 and nb == 1:  # da = outer(g, b); db = a.T @ g
        ga = _b(
            _matmul,
            _b(d_reshape, g, (_pshape(pa)[0], 1)),
            _b(d_reshape, b, (1, _pshape(pb)[0])),
        )
        return [ga, _b(_matmul, _b(d_transpose, a), g)]
    if na == 1 and nb == 2:  # da = b @ g ; db = outer(a, g)
        gb = _b(
            _matmul,
            _b(d_reshape, a, (_pshape(pa)[0], 1)),
            _b(d_reshape, g, (1, _pshape(pb)[1])),
        )
        return [_b(_matmul, b, g), gb]
    # batched / 2-D @ 2-D: da = g @ bᵀ, db = aᵀ @ g (transposing the last two axes).
    return [_b(_matmul, g, _swap_last2(b, nb)), _b(_matmul, _swap_last2(a, na), g)]


def _vjp_einsum(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # The same reverse-einsum as the eager backward, but built with ``bind``-riding
    # ``d_einsum`` over the level-connected operands so the cotangent graph is itself
    # differentiable (composes with an enclosing jvp/grad).
    ins, out = _parse_einsum(params["subscripts"], [_pndim(p) for p in primals])
    xp = _xp()
    grads: list[Boxed] = []
    for i in range(len(primals)):
        spec, others, missing = _einsum_grad_spec(ins, out, i)
        arrays: list[Boxed] = [g] + [operands[j] for j in others]
        if missing:
            mshape = tuple(_pshape(primals[i])[ins[i].index(c)] for c in missing)
            arrays.append(_const_like(xp.ones(mshape)))
        # ``_d_unbroadcast`` sums any size-1 operand axis numpy broadcast up.
        grads.append(_d_unbroadcast(_b(d_einsum, spec, *arrays), _pshape(primals[i])))
    return grads


def _vjp_cumsum(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # Reverse cumulative sum, built with bind-riding ops (flip = a reversed-slice gather)
    # so it composes with an enclosing jvp/grad.
    axis = params["axis"]
    ndim = _pndim(primals[0])
    ax = axis % ndim
    key = tuple(slice(None, None, -1) if d == ax else slice(None) for d in range(ndim))
    return [_b(d_getitem, _b(d_cumsum, _b(d_getitem, g, key), axis=axis), key)]


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
    return [_b(_scatter, g, key, _pshape(p), _pdtype(p))]


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
        g.grad = _accumulate(
            g.grad, out.grad[cast(Index, key)]
        )  # gather at scatter key

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
    return [_b(d_broadcast_to, g, _pshape(p))]


def _vjp_prod(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # cotangent_i = g * prod(x) / x_i, with the keepdims product broadcast over the reduced
    # axis. ``d_prod`` is re-bound (differentiable) so the rule composes to second order.
    (p,) = primals
    (x,) = operands
    axis = params.get("axis")
    keepdims = params.get("keepdims", False)
    if axis is not None and not keepdims:
        g = _expand_dims_multi(g, axis)
    gx = _b(d_broadcast_to, g, _pshape(p))
    pk = _b(d_prod, x, axis=axis, keepdims=True)
    return [_b(d_div, _b(d_mul, gx, pk), x)]


def _vjp_reshape(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    (p,) = primals
    return [_b(d_reshape, g, _pshape(p))]


def _vjp_astype(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # A cast is linear: pull the cotangent back into the input's dtype. The input is a
    # ``Var``/``GraphTracer`` (always floating), so the cast-back is well defined; skip it
    # when the dtype already matches to avoid an identity node.
    (p,) = primals
    in_dt = _pdtype(p)
    return [g if in_dt == params["dtype"] else _b(d_astype, g, in_dt)]


def _vjp_broadcast_to(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    from pycograd.tensor import _d_unbroadcast

    (p,) = primals
    return [_d_unbroadcast(g, _pshape(p))]


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
    ndim = _pndim(primals[0])
    ax = axis % ndim
    out: list[Boxed] = []
    start = 0
    for p in primals:
        end = start + _pshape(p)[ax]
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


def _vjp_softmax(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # y = softmax(x); dx = y * (g - sum(y*g, axis, keepdims)). ``d_softmax`` is recomputed
    # bind-riding -- ``cse`` dedups it against the forward node (the cross-pass-CSE path).
    (x,) = operands
    axis = params.get("axis", -1)
    y = _b(d_softmax, x, axis=axis)
    yg_sum = _b(d_sum, _b(d_mul, y, g), axis=axis, keepdims=True)
    return [_b(d_mul, y, _b(d_sub, g, yg_sum))]


def _vjp_logsumexp(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    # d(logsumexp(x))/dx = softmax(x); dx = softmax(x) * g (g broadcast over the reduced axis).
    (x,) = operands
    axis = params.get("axis")
    keepdims = params.get("keepdims", False)
    if axis is not None and not keepdims:
        g = _expand_dims_multi(g, axis)
    return [_b(d_mul, _b(d_softmax, x, axis=axis), g)]


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
    pdt = _pdtype(pa)
    cmask = _const_like(xp.asarray(cond).astype(pdt))
    omask = _const_like((~xp.asarray(cond).astype(bool)).astype(pdt))
    return [_b(d_mul, g, cmask), _b(d_mul, g, omask)]


def _remat(*operands: Boxed) -> Boxed:
    """Sentinel primitive marking a ``checkpoint`` boundary node. It is never bound to
    produce a value (the boundary node is constructed directly by ``pycograd.checkpoint``);
    it exists only as the ``_vjp_prim`` key under which the differentiable reverse path
    locates the per-instance rematerialization rule. See :func:`_vjp_remat`."""
    raise RuntimeError(
        "_remat is a checkpoint-boundary marker, not a callable primitive"
    )


def _vjp_remat(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    """Differentiable VJP of a ``checkpoint`` boundary: rematerialize the segment's forward
    on the level-connected ``operands`` and run an inner differentiable backward. The
    instance-specific logic (the runner, saved inputs, weight snapshot, slice layout) rides
    ``params["remat"]`` -- a ``pycograd.checkpoint`` box -- so this stays a thin dispatch.
    """
    return params["remat"].differentiable_vjp(operands, g)


def _spill(x: Operand) -> Var:
    """Identity marker: this value is paged to SSD after its forward uses and reloaded on
    its backward use. Inserted by :mod:`pycograd.remat`'s ``apply_remat_plan``. The actual
    disk I/O lives in the scheduled interpreter (``eval_scheduled``), so here ``_spill`` is
    a pure value-identity -- ``eval_graph`` of a rewritten graph is byte-for-byte unchanged
    -- with an identity VJP so an enclosing ``grad`` still flows straight through."""
    x = _lift(x)
    out = Var(x.value, _parents=(x,))

    def _backward() -> None:
        x.grad = _accumulate(x.grad, out.grad)

    out._backward = _backward
    _record_vjp(out, _spill, (x,), {})
    return out


def _recompute(x: Operand) -> Var:
    """Identity marker: this value is dropped after its forward uses and rematerialized on
    its backward use. The recompute counterpart to :func:`_spill` (see its note) -- a pure
    value-identity with an identity VJP; the scheduled interpreter performs the on-demand
    recomputation of the producing subgraph."""
    x = _lift(x)
    out = Var(x.value, _parents=(x,))

    def _backward() -> None:
        x.grad = _accumulate(x.grad, out.grad)

    out._backward = _backward
    _record_vjp(out, _recompute, (x,), {})
    return out


def _vjp_identity(
    primals: tuple[Var, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    """VJP of the identity markers :func:`_spill` / :func:`_recompute`: the single operand's
    cotangent is the incoming cotangent, passed straight through."""
    return [g]


def _build_vjp_for() -> dict[Prim, Callable[..., list[Boxed]]]:
    vjp_for: dict[Prim, Callable[..., list[Boxed]]] = {
        prim: _vjp_unary_for(cast(Callable[..., Var], prim)) for prim in _UNARY_DERIV
    }
    vjp_for.update(
        {
            d_abs: _vjp_abs,
            d_conj: _vjp_conj,
            d_real: _vjp_real,
            d_real_if_close: _vjp_real_if_close,
            d_nan_to_num: _vjp_nan_to_num,
            d_imag: _vjp_imag,
            d_angle: _vjp_angle,
            d_add: _vjp_add,
            d_sub: _vjp_sub,
            d_neg: _vjp_neg,
            d_mul: _vjp_mul,
            d_gated_act: _vjp_gated_act,
            d_div: _vjp_div,
            d_mod: _vjp_mod,
            d_pow: _vjp_pow,
            _matmul: _vjp_matmul,
            d_einsum: _vjp_einsum,
            d_cumsum: _vjp_cumsum,
            d_getitem: _vjp_getitem,
            _scatter: _vjp_scatter,
            d_sum: _vjp_sum,
            d_prod: _vjp_prod,
            d_roll: _vjp_roll,
            d_pad: _vjp_pad,
            d_repeat: _vjp_repeat,
            d_tile: _vjp_tile,
            d_reshape: _vjp_reshape,
            d_astype: _vjp_astype,
            d_broadcast_to: _vjp_broadcast_to,
            d_expand_dims: _vjp_reshape,  # expand_dims is a reshape; the VJP reshapes back
            d_transpose: _vjp_transpose,
            d_concatenate: _vjp_concatenate,
            d_max: _vjp_reduce_select,
            d_min: _vjp_reduce_select,
            d_softmax: _vjp_softmax,
            d_logsumexp: _vjp_logsumexp,
            d_maximum: _vjp_select,
            d_minimum: _vjp_select,
            d_fmax: _vjp_select,
            d_fmin: _vjp_select,
            d_logaddexp: _vjp_logaddexp_for(d_logaddexp, d_exp),
            d_logaddexp2: _vjp_logaddexp_for(d_logaddexp2, d_exp2),
            d_where: _vjp_where,
            _remat: _vjp_remat,
            _spill: _vjp_identity,
            _recompute: _vjp_identity,
        }
    )
    return vjp_for


# The local derivative ``f'(x)`` of each elementwise-unary primitive, as a ``bind``-
# expression. The **single source** for these derivatives: ``_vjp_unary_for`` builds the
# reverse rule ``g * f'(x)`` from it, and ``forward.py``'s jvp builds the forward rule
# ``f'(x) * dx`` from it -- so e.g. ``1 - tanh²`` is written once, not once per direction.
_UNARY_DERIV: dict[Prim, Callable[[Boxed], Boxed]] = _vjp_unary_derivs()
_VJP_FOR: dict[Prim, Callable[..., list[Boxed]]] = _build_vjp_for()


# The non-holomorphic primitives, whose ``_VJP_FOR`` rules already return the final
# real-adjoint cotangent (in the real (a, b) coordinates packed as a complex number). They
# are EXCLUDED from the holomorphic ``conj`` wrap below; every other primitive's rule is
# C-linear in ``g`` and the wrap turns it into the correct Hermitian adjoint.
_NONHOLOMORPHIC: "frozenset[Prim]" = frozenset({d_abs, d_conj, d_real, d_imag, d_angle})


def _boxed_is_complex(x: Boxed) -> bool:
    """Whether a trace-level cotangent/operand carries a complex dtype."""
    if x is None:
        return False
    if isinstance(x, Var):
        return x.value.dtype.kind == "c"
    aval = getattr(x, "aval", None)
    dt = getattr(aval, "dtype", None)
    if dt is not None:
        return np.dtype(dt).kind == "c"
    try:
        return np.asarray(cast(Any, x)).dtype.kind == "c"
    except Exception:  # pragma: no cover - defensive
        return False


def _conj_boxed(x: Boxed) -> Boxed:
    """Tracer-aware ``conj`` for the Hermitian-adjoint wrap: ``bind(d_conj, x)`` when ``x``
    is complex (so it composes under graph/jvp/vmap and grad-of-grad), else ``x`` itself.
    Identity on real cotangents, so the real path adds no nodes."""
    return _b(d_conj, x) if _boxed_is_complex(x) else x


def _vjp_apply(
    prim: Prim,
    rule: Callable[..., list[Boxed]],
    primals: tuple[Boxed, ...],
    operands: tuple[Boxed, ...],
    params: dict[str, Any],
    g: Boxed,
) -> list[Boxed]:
    """Invoke a VJP ``rule`` with the central Hermitian-adjoint conj wrap.

    For a holomorphic primitive the rule is C-linear in ``g`` (returns ``g * D``), so
    ``conj(rule(conj(g)))`` = ``g * conj(D)`` -- the adjoint under the real inner product on
    complex tensors. The non-holomorphic prims are already real-adjoint, so the wrap is
    skipped for them. Real cotangents short-circuit ``_conj_boxed`` to the identity."""
    if prim in _NONHOLOMORPHIC:
        return rule(primals, operands, params, g)
    contribs = rule(primals, operands, params, _conj_boxed(g))
    return [None if c is None else _conj_boxed(c) for c in contribs]


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
    return _vjp_apply(prim, rule, v._vjp_operands, operands, v._vjp_params, g)


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

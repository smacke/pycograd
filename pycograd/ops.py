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
    Shape,
)
from pycograd.backends import current_backend
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
def _elementwise_max(
    a: Operand, b: Operand, pick_a: Callable[..., Array], prim: Prim
) -> Var:
    a, b = _lift(a), _lift(b)
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
        da, db = _matmul_grads(a.value, b.value, out.grad)
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


def d_einsum(subscripts: str, *operands: Operand) -> Var:
    from pycograd.trace import Tracer, bind

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
            arrays = [node.grad] + [vals[j] for j in others]
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
# Cumulative sum -- a fused primitive (linear; no composition expresses the prefix
# sum). The VJP is a reverse cumulative sum (flip -> cumsum -> flip).
# ---------------------------------------------------------------------------
def d_cumsum(x: Operand, axis: int | None = None) -> Var:
    from pycograd.trace import Tracer, bind

    if isinstance(x, Tracer):
        return cast(Var, bind(d_cumsum, x, axis=axis))
    if axis is None:
        raise NotImplementedError(
            "cumsum: pass an explicit integer axis (the flatten-all default is not "
            "supported)"
        )
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
            d_reshape: _vjp_reshape,
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

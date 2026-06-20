# -*- coding: utf-8 -*-
"""Auto-batching (``vmap``): run a per-example function over a batch axis, vectorized.

``vmap`` is a *forward* program transformation. Each batched value is a
:class:`BatchedArray` wrapping an ordinary :class:`~pycograd.tensor.Var` whose
``.value`` carries the batch axis **materialized at axis 0**; user code sees the
*logical* (per-example) shape. Every op is replaced by a *batching rule* that adjusts
axis arguments to skip the batch dim and calls the underlying differentiable primitive
directly -- so the tape is an ordinary `Var` tape over batched arrays, and
``backward()`` differentiates it with no vmap-specific backward code (a shared,
unbatched operand's gradient is summed over the batch by ``_unbroadcast``).

Invariant: a ``BatchedArray``'s batch axis is always physical axis 0. ``vmap`` moves
inputs' batch axes to 0 on entry and to ``out_axes`` on exit; rules keep it at 0.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, cast

import numpy as np

from pycograd import ops
from pycograd.tensor import Var


class BatchedArray:
    """A value carrying a batch axis at physical axis 0; logical shape is ``[1:]``."""

    __slots__ = ("inner",)
    __array_ufunc__ = None

    def __init__(self, inner: object) -> None:
        # ``inner`` is a Var (differentiable) or a raw array (e.g. a boolean mask).
        self.inner = inner

    # -- logical (per-example) metadata --------------------------------------
    @property
    def shape(self) -> tuple:
        return tuple(_phys_shape(self.inner)[1:])

    @property
    def ndim(self) -> int:
        return len(_phys_shape(self.inner)) - 1

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(_phys(self.inner).dtype)

    @property
    def T(self) -> "BatchedArray":
        return cast("BatchedArray", batch_transpose(self))

    # -- arithmetic (elementwise; batch axis transparent after alignment) -----
    def __add__(self, o: object) -> object:
        return _elementwise(lambda a, b: a + b, self, o)

    __radd__ = __add__

    def __mul__(self, o: object) -> object:
        return _elementwise(lambda a, b: a * b, self, o)

    __rmul__ = __mul__

    def __sub__(self, o: object) -> object:
        return _elementwise(lambda a, b: a - b, self, o)

    def __rsub__(self, o: object) -> object:
        return _elementwise(lambda a, b: b - a, self, o)

    def __truediv__(self, o: object) -> object:
        return _elementwise(lambda a, b: a / b, self, o)

    def __rtruediv__(self, o: object) -> object:
        return _elementwise(lambda a, b: b / a, self, o)

    def __pow__(self, p: object) -> object:
        return _elementwise(lambda a, b: a**b, self, p)

    def __rpow__(self, p: object) -> object:
        return _elementwise(lambda a, b: b**a, self, p)

    def __neg__(self) -> object:
        return _elementwise(lambda a: -a, self)

    def __abs__(self) -> object:
        return _elementwise(abs, self)

    def __matmul__(self, o: object) -> object:
        return batch_matmul(self, o)

    def __rmatmul__(self, o: object) -> object:
        return batch_matmul(o, self)

    # -- comparisons return a batched boolean mask ---------------------------
    def _cmp(self, o: object, op: Callable) -> object:
        return _elementwise(op, self, o)

    def __lt__(self, o: object) -> object:
        return self._cmp(o, lambda a, b: a < b)

    def __le__(self, o: object) -> object:
        return self._cmp(o, lambda a, b: a <= b)

    def __gt__(self, o: object) -> object:
        return self._cmp(o, lambda a, b: a > b)

    def __ge__(self, o: object) -> object:
        return self._cmp(o, lambda a, b: a >= b)

    __hash__ = None  # type: ignore[assignment]

    def __getitem__(self, key: object) -> object:
        return batch_getitem(self, key)

    # -- numpy-method surface (x.sum(...), x.reshape(...), x.mean(...)) -------
    def __getattr__(self, name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        np_fn = getattr(np, name, None)
        rule = _BATCH.get(np_fn) if callable(np_fn) else None
        if rule is None:
            raise AttributeError(name)
        return functools.partial(rule, self)

    def __repr__(self) -> str:
        return f"BatchedArray(logical={self.shape}, dtype={self.dtype})"


# ---------------------------------------------------------------------------
# Unwrap / rewrap helpers (batch axis always at 0).
# ---------------------------------------------------------------------------
def _phys(x: object) -> Any:
    """The physical Var/array under a value (the BatchedArray's inner, else itself)."""
    return x.inner if isinstance(x, BatchedArray) else x


def _phys_shape(inner: object) -> tuple:
    arr = inner.value if isinstance(inner, Var) else inner
    return tuple(np.shape(cast(Any, arr)))


def _is_batched(x: object) -> bool:
    return isinstance(x, BatchedArray)


def _logical_ndim(x: object) -> int:
    if isinstance(x, BatchedArray):
        return x.ndim
    return len(_phys_shape(x))


def _rewrap(inner: object, batched: bool) -> object:
    return BatchedArray(inner) if batched else inner


def _batch_size(args: "tuple") -> int:
    for a in args:
        if isinstance(a, BatchedArray):
            return _phys_shape(a.inner)[0]
    raise ValueError("vmap: no batched argument")


# ---------------------------------------------------------------------------
# Axis bookkeeping (logical -> physical, batch at 0).
# ---------------------------------------------------------------------------
def _shift_axis(axis: object, logical_ndim: int) -> object:
    """Map a logical reduce/transpose axis to the physical axis (batch at 0)."""
    if axis is None:
        return tuple(range(1, logical_ndim + 1))  # all logical axes, not the batch
    axes = axis if isinstance(axis, tuple) else (axis,)
    shifted = tuple(a + 1 if a >= 0 else a for a in axes)  # negatives count from end
    return shifted if isinstance(axis, tuple) else shifted[0]


def _insert_leading_logical(v: Any, pad: int, batched: bool) -> object:
    """Insert ``pad`` size-1 axes at the front of the *logical* shape (after batch)."""
    if pad <= 0:
        return v
    shp = _phys_shape(v)
    pos = 1 if batched else 0
    new = shp[:pos] + (1,) * pad + shp[pos:]
    return ops.d_reshape(v, new)


# ---------------------------------------------------------------------------
# Elementwise rule: align logical ranks, then apply -- batch axis broadcasts.
# ---------------------------------------------------------------------------
def _elementwise(fn: Callable, *args: object, **kwargs: object) -> object:
    batched = any(_is_batched(a) for a in args)
    if not batched:
        return fn(*(_phys(a) for a in args), **kwargs)
    max_l = max(_logical_ndim(a) for a in args)
    aligned = []
    for a in args:
        v = _phys(a)
        pad = max_l - _logical_ndim(a)
        aligned.append(_insert_leading_logical(v, pad, _is_batched(a)))
    return _rewrap(fn(*aligned, **kwargs), True)


def _lift_rule(prim: Callable) -> Callable:
    """A batching rule for an elementwise primitive (np.exp, np.maximum, where, ...)."""

    def rule(*args: object, **kwargs: object) -> object:
        return _elementwise(lambda *xs: prim(*xs, **kwargs), *args)

    return rule


# ---------------------------------------------------------------------------
# Axis-shifting rules.
# ---------------------------------------------------------------------------
def batch_reduce(prim: Callable) -> Callable:
    def rule(x: object, axis: object = None, keepdims: bool = False, **kw: object):
        if not _is_batched(x):
            return prim(_phys(x), axis=axis, keepdims=keepdims, **kw)
        v = _phys(x)
        ax = _shift_axis(axis, _logical_ndim(x))
        return BatchedArray(prim(v, axis=ax, keepdims=keepdims, **kw))

    return rule


def batch_transpose(x: object, axes: object = None) -> object:
    if not _is_batched(x):
        return ops.d_transpose(_phys(x), cast(Any, axes))
    v = _phys(x)
    n = _logical_ndim(x)
    if axes is None:
        perm = (0,) + tuple(range(n, 0, -1))  # reverse logical axes, keep batch at 0
    else:
        perm = (0,) + tuple(
            (a + 1 if a >= 0 else a + n + 1) for a in cast("tuple", axes)
        )
    return BatchedArray(ops.d_transpose(v, perm))


def batch_expand_dims(x: object, axis: int) -> object:
    if not _is_batched(x):
        return ops.d_expand_dims(_phys(x), axis)
    v = _phys(x)
    n = _logical_ndim(x)
    pos = axis + 1 if axis >= 0 else axis + n + 2  # logical position -> physical
    return BatchedArray(ops.d_expand_dims(v, pos))


def batch_reshape(x: object, *shape: object) -> object:
    if not _is_batched(x):
        return ops.d_reshape(_phys(x), *shape)
    v = _phys(x)
    newshape = shape[0] if len(shape) == 1 else shape
    if isinstance(newshape, int):
        newshape = (newshape,)
    b = _phys_shape(v)[0]
    # Prepend the (concrete) batch size; a -1 then infers against the per-example size.
    return BatchedArray(ops.d_reshape(v, (b,) + tuple(cast("tuple", newshape))))


def batch_matmul(a: object, b: object) -> object:
    ba, bb = _is_batched(a), _is_batched(b)
    av, bv = _phys(a), _phys(b)
    if not (ba or bb):
        return ops._matmul(av, bv)
    la, lb = _logical_ndim(a), _logical_ndim(b)
    # Promote 1-D logical operands to matrices so a single batched matmul covers every
    # vector/matrix combination; squeeze the temporary axes off the result afterwards.
    squeeze_m = squeeze_n = False
    if la == 1:  # (k,) -> (1, k): insert a row axis at the logical front
        av = ops.d_expand_dims(av, 1 if ba else 0)
        squeeze_m = True
    if lb == 1:  # (k,) -> (k, 1): insert a column axis at the logical end
        bv = ops.d_expand_dims(bv, -1)
        squeeze_n = True
    out = ops._matmul(av, bv)  # numpy batches over the leading axes
    out_shape = list(_phys_shape(out))
    # The promoted m-axis sits at -2, the n-axis at -1; remove n first so that, if both
    # were added, the m-axis is then last too.
    if squeeze_n:
        del out_shape[-1]
    if squeeze_m:
        del out_shape[-1 if squeeze_n else -2]
    if squeeze_m or squeeze_n:
        out = ops.d_reshape(out, tuple(out_shape))
    return BatchedArray(out)


def batch_getitem(x: object, key: object) -> object:
    if not _is_batched(x):
        return _phys(x)[key]
    if isinstance(key, BatchedArray) or (
        isinstance(key, tuple) and any(isinstance(k, BatchedArray) for k in key)
    ):
        raise NotImplementedError(
            "vmap: per-example (batched) index keys are not supported yet; "
            "use a key that is the same across the batch"
        )
    keys = key if isinstance(key, tuple) else (key,)
    return BatchedArray(_phys(x)[(slice(None),) + keys])  # skip the batch axis


def _concat_like(name: str) -> Callable:
    def rule(seq: "list", axis: int = 0, **kw: object) -> object:
        if not any(_is_batched(s) for s in seq):
            return getattr(ops, name)(seq, axis=axis, **kw)
        raise NotImplementedError(
            f"vmap: {name} over batched operands is not supported yet"
        )

    return rule


def batch_concatenate(seq: "list", axis: int = 0) -> object:
    if not any(_is_batched(s) for s in seq):
        return ops.d_concatenate(seq, axis=axis)
    if not all(_is_batched(s) for s in seq):
        raise NotImplementedError(
            "vmap: concatenate mixing batched and unbatched operands is not supported yet"
        )
    n = _logical_ndim(seq[0])
    ax = axis + 1 if axis >= 0 else axis + n + 1
    return BatchedArray(ops.d_concatenate([_phys(s) for s in seq], axis=ax))


def batch_stack(seq: "list", axis: int = 0) -> object:
    if not any(_is_batched(s) for s in seq):
        return ops.d_stack(seq, axis=axis)
    if not all(_is_batched(s) for s in seq):
        raise NotImplementedError(
            "vmap: stack mixing batched and unbatched operands is not supported yet"
        )
    n = _logical_ndim(seq[0])
    ax = axis + 1 if axis >= 0 else axis + n + 2
    return BatchedArray(ops.d_stack([_phys(s) for s in seq], axis=ax))


# ---------------------------------------------------------------------------
# The batching interception table, keyed identically to ``ops._INTERCEPT``.
# ---------------------------------------------------------------------------
def _build_batch_table() -> "dict[object, Callable]":
    unary = (
        ops.d_exp,
        ops.d_log,
        ops.d_sin,
        ops.d_cos,
        ops.d_tanh,
        ops.d_sqrt,
        ops.d_sinh,
        ops.d_cosh,
        ops.d_arctan,
        ops.d_log1p,
        ops.d_expm1,
        ops.d_abs,
        ops.d_square,
        ops.d_reciprocal,
    )
    rule_for: "dict[object, Callable]" = {p: _lift_rule(p) for p in unary}
    rule_for.update(
        {
            ops.d_maximum: _lift_rule(ops.d_maximum),
            ops.d_minimum: _lift_rule(ops.d_minimum),
            ops.d_where: _lift_rule(ops.d_where),
            ops.d_clip: _lift_rule(ops.d_clip),
            ops.d_sum: batch_reduce(ops.d_sum),
            ops.d_mean: batch_reduce(ops.d_mean),
            ops.d_var: batch_reduce(ops.d_var),
            ops.d_std: batch_reduce(ops.d_std),
            ops.d_max: batch_reduce(ops.d_max),
            ops.d_min: batch_reduce(ops.d_min),
            ops._matmul: batch_matmul,
            ops.d_transpose: batch_transpose,
            ops.d_reshape: batch_reshape,
            ops.d_expand_dims: batch_expand_dims,
            ops.d_concatenate: batch_concatenate,
            ops.d_stack: batch_stack,
            ops.d_vstack: _concat_like("d_vstack"),
            ops.d_hstack: _concat_like("d_hstack"),
            ops.d_column_stack: _concat_like("d_column_stack"),
            ops.d_dstack: _concat_like("d_dstack"),
        }
    )
    return {fn: rule_for[prim] for prim, fns in ops._RULES.items() for fn in fns}


_BATCH: "dict[object, Callable]" = _build_batch_table()

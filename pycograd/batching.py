# -*- coding: utf-8 -*-
"""Auto-batching (``vmap``): a *trace level* that vectorizes over a batch axis.

``vmap`` is a *forward* program transformation realized as one level of the
trace-level interpreter stack (:mod:`pycograd.trace`). Each batched value is a
:class:`BatchTracer` carrying a physical :class:`~pycograd.tensor.Var`/array plus an
explicit ``bdim`` -- *which* physical axis is the batch axis (``None`` means the value
is unbatched/shared at this level). When a primitive flows through ``bind``, the top
:class:`BatchTrace` handles it: it moves every operand's batch axis to the front, runs
the existing per-primitive batch rule (written for "batch at axis 0"), and tags the
result with ``bdim=0``. So a single ``vmap`` is the level-1 trace over level-0
``Var``s; ``vmap(vmap(f))`` is a level-2 trace whose rules ``bind`` *one level down*,
recursing into the level-1 trace, until :class:`~pycograd.trace.EvalTrace` computes.

Because each rule adjusts axis arguments to skip the batch dim and then calls the
underlying differentiable primitive directly, the tape is an ordinary ``Var`` tape over
batched arrays and ``backward()`` differentiates it with no vmap-specific backward code
(a shared, unbatched operand's gradient is summed over the batch by ``_unbroadcast``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, NoReturn, Sequence, cast

import numpy as np

from pycograd import ops
from pycograd._typing import Axis, BindArg, Boxed, Index, Prim, Rule, Shape
from pycograd.tensor import Var, _value
from pycograd.trace import Trace, Tracer, bind, full_raise

if TYPE_CHECKING:
    from pycograd.shapes import ShapeDtypeStruct


# ---------------------------------------------------------------------------
# BatchTracer / BatchTrace: a value tagged with its batch axis, and the level
# that processes primitives over such values.
# ---------------------------------------------------------------------------
class BatchTracer(Tracer):
    """A value carrying an explicit batch axis (``bdim``) at a given trace ``level``.

    ``value`` is the physical :class:`~pycograd.tensor.Var`/array (a level-down value);
    ``bdim`` is the position of the batch axis within ``value``'s physical shape, or
    ``None`` if this value is unbatched/shared at this level. The *logical*
    (per-example) shape drops the batch axis.
    """

    __slots__ = ("_trace", "value", "bdim")
    __array_ufunc__ = None

    def __init__(self, trace: "BatchTrace", value: Boxed, bdim: int | None) -> None:
        self._trace = trace
        self.value = value
        self.bdim = bdim

    @property
    def aval(self) -> "ShapeDtypeStruct":
        # The logical shape/dtype seen by user code: the physical shape minus the batch
        # axis. Reported via a ShapeDtypeStruct so ``.shape``/``.ndim`` work.
        from pycograd.shapes import ShapeDtypeStruct

        shp = list(_phys_shape(self.value))
        dt = _phys_dtype(self.value)
        if self.bdim is not None:
            del shp[self.bdim]
        return ShapeDtypeStruct(tuple(shp), dt)

    @property
    def dtype(self) -> np.dtype:
        return _phys_dtype(self.value)

    @property
    def size(self) -> int:
        return int(np.prod(cast(Any, self.shape), dtype=np.int64))

    @property
    def T(self) -> Boxed:
        return bind(ops.d_transpose, self)

    def __getitem__(self, key: Index) -> Boxed:
        return bind(ops.d_getitem, self, key)

    # -- numpy-method surface (x.sum(...), x.reshape(...), x.mean(...)) --------
    def __getattr__(self, name: str) -> "Callable[..., Boxed]":
        # Route a numpy method name we have a primitive for through ``bind`` so the
        # method call (``x.sum(axis=0)``) vectorizes and nests exactly like the free
        # function ``np.sum(x, axis=0)``. Mirrors ``Var.__getattr__`` but goes through
        # the trace-level stack so the right ``vmap`` level handles it.
        if name.startswith("__"):
            raise AttributeError(name)
        from pycograd import ops as _ops

        np_fn = getattr(np, name, None)
        prim = _ops._INTERCEPT.get(np_fn) if callable(np_fn) else None
        if prim is None and name == "flatten":  # no ``np.flatten``; it's ``ravel``
            prim = _ops.d_ravel
        if prim is None:
            raise AttributeError(name)

        def _method(*a: BindArg, **k: Any) -> Boxed:
            return bind(prim, self, *a, **k)

        return _method

    def __repr__(self) -> str:
        return (
            f"BatchTracer(level={self._trace.main.level}, "
            f"logical={self.shape}, bdim={self.bdim}, dtype={self.dtype})"
        )


class BatchTrace(Trace):
    """One ``vmap`` level: process each primitive by moving batch axes to the front,
    running the existing per-primitive batch rule, and tagging the result ``bdim=0``."""

    def pure(self, val: Boxed) -> BatchTracer:
        return BatchTracer(self, val, None)

    def lift(self, val: Boxed) -> BatchTracer:
        # A value from a lower level enters this level unbatched.
        return BatchTracer(self, val, None)

    def process_primitive(
        self, prim: Prim, args: Sequence[BindArg], params: dict[str, Any]
    ) -> Boxed:
        rule = _RULE_FOR.get(prim)
        if rule is None:
            _process_unmapped(prim)
        return rule(self, *args, **params)

    def _raise(self, val: Boxed) -> "BatchTracer":
        """Lift one operand into this level: a tracer from a lower level (or a constant)
        becomes an unbatched ``BatchTracer``; one already at this level passes through.
        """
        return cast(BatchTracer, full_raise(self, val))


# ---------------------------------------------------------------------------
# Physical-array helpers.
# ---------------------------------------------------------------------------
def _phys_shape(inner: Boxed) -> tuple[int, ...]:
    arr = inner.value if isinstance(inner, Var) else inner
    return tuple(np.shape(cast(Any, arr)))


def _phys_dtype(inner: Boxed) -> np.dtype:
    arr = inner.value if isinstance(inner, Var) else inner
    dt = (
        cast(Any, arr).dtype
        if hasattr(arr, "dtype")
        else np.asarray(cast(Any, arr)).dtype
    )
    return np.dtype(dt)


def _move_bdim_to_front(t: BatchTracer) -> Boxed:
    """The physical value with its batch axis moved to position 0 (``None`` -> unchanged
    physical value).

    Routed through :func:`~pycograd.trace.bind` so it works at every level: for a
    single ``vmap`` the value is a level-0 ``Var`` and ``bind`` transposes it directly
    (staying on the tape so gradients flow); for nested ``vmap`` the value is a
    lower-level ``BatchTracer`` and ``bind`` recurses into that level's transpose rule.
    A plain numpy array (e.g. an index) is moved with ``np.moveaxis``.
    """
    if t.bdim is None or t.bdim == 0:
        return t.value
    v = t.value
    if isinstance(v, (Var, BatchTracer)):
        ndim = len(_phys_shape(v))
        perm = list(range(ndim))
        perm.insert(0, perm.pop(t.bdim))
        return bind(ops.d_transpose, v, tuple(perm))
    return np.moveaxis(np.asarray(cast(Any, v)), t.bdim, 0)


def _logical_ndim(t: BatchTracer) -> int:
    n = len(_phys_shape(t.value))
    return n - 1 if t.bdim is not None else n


# ---------------------------------------------------------------------------
# Rule infrastructure.
#
# Each rule has the *natural call signature of its primitive* (so positional/keyword
# arguments bind exactly as the user wrote them) plus a leading ``trace``. It raises
# every operand into this level (constants / lower-level tracers -> ``bdim=None``),
# moves each batched operand's batch axis to the front (the operand then looks like the
# old "batch at 0" layout), runs the original rule body on the physical values, and
# returns a ``BatchTracer`` tagged ``bdim=0`` (or ``bdim=None`` if nothing was batched).
# Because the body calls the underlying ``d_*`` primitive (which itself flows through
# ``bind``, one level down), nesting composes automatically.
# ---------------------------------------------------------------------------
def _result(trace: BatchTrace, value: Boxed, bdim: int | None) -> BatchTracer:
    return BatchTracer(trace, value, bdim)


def _insert_leading_logical(v: Boxed, pad: int, batched: bool) -> Boxed:
    """Insert ``pad`` size-1 axes at the front of the *logical* shape (after batch)."""
    if pad <= 0:
        return v
    shp = _phys_shape(v)
    pos = 1 if batched else 0
    new = shp[:pos] + (1,) * pad + shp[pos:]
    return bind(ops.d_reshape, v, new)


# -- elementwise ------------------------------------------------------------
def _elementwise_rule(
    trace: BatchTrace, prim: Prim, *operands: Boxed, **kwargs: Any
) -> BatchTracer:
    tracers = [trace._raise(o) for o in operands]
    if not any(t.bdim is not None for t in tracers):
        return _result(trace, bind(prim, *(t.value for t in tracers), **kwargs), None)
    max_l = max(_logical_ndim(t) for t in tracers)
    aligned = []
    for t in tracers:
        v = _move_bdim_to_front(t)
        pad = max_l - _logical_ndim(t)
        aligned.append(_insert_leading_logical(v, pad, t.bdim is not None))
    return _result(trace, bind(prim, *aligned, **kwargs), 0)


def _pow_rule(trace: BatchTrace, a: Boxed, b: Boxed, **kwargs: Any) -> BatchTracer:
    # ``d_pow``'s exponent is a *constant* by convention: ``Var.__pow__`` routes only a
    # constant exponent through ``d_pow`` (a ``Var`` exponent is lowered to ``exp(b*log a)``
    # first), and ``_vjp_pow`` reads it as a raw value. The generic elementwise rule would
    # ``_raise`` the exponent to a tracer too, so the base-level ``d_pow`` would see a ``Var``
    # exponent and take the ``exp(b*log a)`` branch -- nan for a negative base. Keep the
    # exponent raw and batch only the base, so the safe power path is preserved.
    ta = trace._raise(a)
    if ta.bdim is None:
        return _result(trace, bind(ops.d_pow, ta.value, b, **kwargs), None)
    return _result(trace, bind(ops.d_pow, _move_bdim_to_front(ta), b, **kwargs), 0)


def _elementwise_for(prim: Prim) -> Rule:
    def rule(trace: BatchTrace, *operands: Boxed, **kwargs: Any) -> BatchTracer:
        return _elementwise_rule(trace, prim, *operands, **kwargs)

    return rule


# -- reductions -------------------------------------------------------------
def _shift_axis(axis: Axis, logical_ndim: int) -> Axis:
    """Map a logical reduce axis to the physical axis (batch at 0)."""
    if axis is None:
        return tuple(range(1, logical_ndim + 1))  # all logical axes, not the batch
    axes = axis if isinstance(axis, tuple) else (axis,)
    shifted = tuple(a + 1 if a >= 0 else a for a in axes)  # negatives count from end
    return shifted if isinstance(axis, tuple) else shifted[0]


def _reduce_for(prim: Prim) -> Rule:
    def rule(
        trace: BatchTrace,
        x: Boxed,
        axis: Axis = None,
        keepdims: bool = False,
        **kw: Any,
    ) -> BatchTracer:
        t = trace._raise(x)
        if t.bdim is None:
            return _result(
                trace, bind(prim, t.value, axis=axis, keepdims=keepdims, **kw), None
            )
        v = _move_bdim_to_front(t)
        ax = _shift_axis(axis, _logical_ndim(t))
        return _result(trace, bind(prim, v, axis=ax, keepdims=keepdims, **kw), 0)

    return rule


def _axis_preserving_for(prim: Prim) -> Rule:
    """Like :func:`_reduce_for` but for a shape-preserving ``axis``-bearing op (softmax):
    the batch axis moves to the front and the logical ``axis`` shifts past it, but the
    result keeps the batch axis (``bdim`` 0) rather than reducing it away."""

    def rule(trace: BatchTrace, x: Boxed, axis: Axis = -1, **kw: Any) -> BatchTracer:
        t = trace._raise(x)
        if t.bdim is None:
            return _result(trace, bind(prim, t.value, axis=axis, **kw), None)
        v = _move_bdim_to_front(t)
        ax = _shift_axis(axis, _logical_ndim(t))
        return _result(trace, bind(prim, v, axis=ax, **kw), 0)

    return rule


# -- transpose --------------------------------------------------------------
def _transpose_rule(
    trace: BatchTrace, x: Boxed, axes: tuple[int, ...] | None = None
) -> BatchTracer:
    t = trace._raise(x)
    if t.bdim is None:
        return _result(trace, bind(ops.d_transpose, t.value, cast(Any, axes)), None)
    v = _move_bdim_to_front(t)
    n = _logical_ndim(t)
    if axes is None:
        perm = (0,) + tuple(range(n, 0, -1))  # reverse logical axes, keep batch at 0
    else:
        perm = (0,) + tuple(
            (a + 1 if a >= 0 else a + n + 1) for a in cast("tuple", axes)
        )
    return _result(trace, bind(ops.d_transpose, v, perm), 0)


# -- expand_dims ------------------------------------------------------------
def _expand_dims_rule(trace: BatchTrace, x: Boxed, axis: int) -> BatchTracer:
    t = trace._raise(x)
    if t.bdim is None:
        return _result(trace, bind(ops.d_expand_dims, t.value, axis), None)
    v = _move_bdim_to_front(t)
    n = _logical_ndim(t)
    pos = axis + 1 if axis >= 0 else axis + n + 2  # logical position -> physical
    return _result(trace, bind(ops.d_expand_dims, v, pos), 0)


# -- reshape ----------------------------------------------------------------
def _reshape_rule(trace: BatchTrace, x: Boxed, *shape: Shape) -> BatchTracer:
    t = trace._raise(x)
    if t.bdim is None:
        return _result(trace, bind(ops.d_reshape, t.value, *shape), None)
    v = _move_bdim_to_front(t)
    newshape = shape[0] if len(shape) == 1 else shape
    if isinstance(newshape, int):
        newshape = (newshape,)
    b = _phys_shape(v)[0]
    # Prepend the (concrete) batch size; a -1 then infers against the per-example size.
    return _result(
        trace, bind(ops.d_reshape, v, (b,) + tuple(cast("tuple", newshape))), 0
    )


# -- astype -----------------------------------------------------------------
def _astype_rule(trace: BatchTrace, x: Boxed, dtype: Any, **kwargs: Any) -> BatchTracer:
    # A cast is elementwise: shape is unchanged, so the batch axis stays exactly where it
    # is (no move-to-front needed). ``dtype`` is static metadata, threaded through.
    t = trace._raise(x)
    return _result(trace, bind(ops.d_astype, t.value, dtype, **kwargs), t.bdim)


# -- broadcast_to -----------------------------------------------------------
def _broadcast_to_rule(trace: BatchTrace, x: Boxed, shape: Shape) -> BatchTracer:
    t = trace._raise(x)
    target = tuple(shape) if isinstance(shape, (tuple, list)) else (cast(int, shape),)
    if t.bdim is None:
        return _result(trace, bind(ops.d_broadcast_to, t.value, target), None)
    v = _move_bdim_to_front(t)
    b = _phys_shape(v)[0]
    return _result(trace, bind(ops.d_broadcast_to, v, (b,) + tuple(target)), 0)


# -- matmul -----------------------------------------------------------------
def _matmul_rule(trace: BatchTrace, a: Boxed, b: Boxed) -> BatchTracer:
    ta, tb = trace._raise(a), trace._raise(b)
    ba, bb = ta.bdim is not None, tb.bdim is not None
    if not (ba or bb):
        return _result(trace, bind(ops._matmul, ta.value, tb.value), None)
    av, bv = _move_bdim_to_front(ta), _move_bdim_to_front(tb)
    la, lb = _logical_ndim(ta), _logical_ndim(tb)
    # Promote 1-D logical operands to matrices so a single batched matmul covers every
    # vector/matrix combination; squeeze the temporary axes off the result afterwards.
    squeeze_m = squeeze_n = False
    if la == 1:  # (k,) -> (1, k): insert a row axis at the logical front
        av = bind(ops.d_expand_dims, av, 1 if ba else 0)
        squeeze_m = True
    if lb == 1:  # (k,) -> (k, 1): insert a column axis at the logical end
        bv = bind(ops.d_expand_dims, bv, -1)
        squeeze_n = True
    # One operand unbatched: give it a size-1 leading axis so numpy's batched matmul
    # broadcasts it across the batch (matching the shared-weight loop oracle).
    if ba and not bb:
        bv = _lead1(bv)
    elif bb and not ba:
        av = _lead1(av)
    out = bind(ops._matmul, av, bv)  # numpy batches over the leading axis
    out_shape = list(_phys_shape(out))
    # The promoted m-axis sits at -2, the n-axis at -1; remove n first so that, if both
    # were added, the m-axis is then last too.
    if squeeze_n:
        del out_shape[-1]
    if squeeze_m:
        del out_shape[-1 if squeeze_n else -2]
    if squeeze_m or squeeze_n:
        out = bind(ops.d_reshape, out, tuple(out_shape))
    return _result(trace, out, 0)


def _lead1(v: Boxed) -> Boxed:
    """Prepend a size-1 axis (numpy then broadcasts it over the batch in matmul); kept on
    the tape (via ``bind``) so the shared-operand gradient is summed over the batch and
    nesting recurses correctly."""
    if isinstance(v, (Var, BatchTracer)):
        return bind(ops.d_expand_dims, v, 0)
    return np.asarray(cast(Any, v))[np.newaxis, ...]


# -- einsum -----------------------------------------------------------------
def _fresh_label(used: "set[str]") -> str:
    """A single einsum label not already in ``used`` (for the new batch axis)."""
    import string

    for c in string.ascii_letters:
        if c not in used:
            return c
    raise ValueError("einsum: ran out of labels to name the vmap batch axis")


def _einsum_rule(trace: BatchTrace, subscripts: Any, *operands: Boxed) -> BatchTracer:
    # vmap of einsum: give each batched operand (and the output) a fresh leading label
    # for the batch axis. An unbatched operand keeps its labels and is reused across the
    # batch -- exactly einsum's semantics for an index it lacks.
    subscripts, operands = ops._normalize_einsum_args(subscripts, operands)
    tracers = [trace._raise(o) for o in operands]
    if not any(t.bdim is not None for t in tracers):
        vals = tuple(t.value for t in tracers)
        return _result(trace, bind(ops.d_einsum, subscripts, *vals), None)
    ins, out = ops._parse_einsum(subscripts, [len(t.shape) for t in tracers])
    bc = _fresh_label(set("".join(ins)) | set(out))
    new_ins, aligned = [], []
    for t, sub in zip(tracers, ins):
        if t.bdim is not None:
            aligned.append(_move_bdim_to_front(t))
            new_ins.append(bc + sub)
        else:
            aligned.append(t.value)
            new_ins.append(sub)
    new_spec = ",".join(new_ins) + "->" + bc + out
    return _result(trace, bind(ops.d_einsum, new_spec, *aligned), 0)


def _cumsum_rule(trace: BatchTrace, x: Boxed, axis: int = -1) -> BatchTracer:
    t = trace._raise(x)
    if t.bdim is None:
        return _result(trace, bind(ops.d_cumsum, t.value, axis=axis), None)
    # The logical ``axis`` shifts past the leading batch axis once it's at the front.
    ax = axis % _logical_ndim(t)
    return _result(trace, bind(ops.d_cumsum, _move_bdim_to_front(t), axis=ax + 1), 0)


def _sort_rule(trace: BatchTrace, x: Boxed, axis: int = -1) -> BatchTracer:
    t = trace._raise(x)
    if t.bdim is None:
        return _result(trace, bind(ops.d_sort, t.value, axis=axis), None)
    ax = axis % _logical_ndim(t)
    return _result(trace, bind(ops.d_sort, _move_bdim_to_front(t), axis=ax + 1), 0)


def _partition_rule(
    trace: BatchTrace, x: Boxed, kth: Any, axis: int = -1
) -> BatchTracer:
    t = trace._raise(x)
    if t.bdim is None:
        return _result(trace, bind(ops.d_partition, t.value, kth, axis=axis), None)
    ax = axis % _logical_ndim(t)
    return _result(
        trace, bind(ops.d_partition, _move_bdim_to_front(t), kth, axis=ax + 1), 0
    )


def _roll_rule(
    trace: BatchTrace, x: Boxed, shift: Any, axis: Any = None
) -> BatchTracer:
    t = trace._raise(x)
    if t.bdim is None:
        return _result(trace, bind(ops.d_roll, t.value, shift, axis=axis), None)
    if axis is None:
        raise NotImplementedError(
            "vmap(np.roll) with axis=None is not supported (it would roll across the "
            "batch); pass an explicit axis"
        )
    ax = axis % _logical_ndim(t)
    return _result(
        trace, bind(ops.d_roll, _move_bdim_to_front(t), shift, axis=ax + 1), 0
    )


def _pad_rule(
    trace: BatchTrace, x: Boxed, pad_width: Any, mode: str = "constant", **kw: Any
) -> BatchTracer:
    t = trace._raise(x)
    if t.bdim is None:
        return _result(
            trace, bind(ops.d_pad, t.value, pad_width, mode=mode, **kw), None
        )
    # Batch axis at the front: pad it by (0, 0) so only the logical axes are padded.
    pw = ops.normalize_pad_width(pad_width, _logical_ndim(t))
    full = ((0, 0),) + pw
    return _result(
        trace, bind(ops.d_pad, _move_bdim_to_front(t), full, mode=mode, **kw), 0
    )


def _repeat_rule(
    trace: BatchTrace, x: Boxed, repeats: Any, axis: Any = None
) -> BatchTracer:
    t = trace._raise(x)
    if t.bdim is None:
        return _result(trace, bind(ops.d_repeat, t.value, repeats, axis=axis), None)
    if axis is None:
        raise NotImplementedError("vmap(np.repeat) with axis=None is not supported")
    ax = axis % _logical_ndim(t)
    return _result(
        trace, bind(ops.d_repeat, _move_bdim_to_front(t), repeats, axis=ax + 1), 0
    )


def _tile_rule(trace: BatchTrace, x: Boxed, reps: Any) -> BatchTracer:
    t = trace._raise(x)
    if t.bdim is None:
        return _result(trace, bind(ops.d_tile, t.value, reps), None)
    # numpy right-aligns ``reps`` to the array shape, so the leading batch axis (at the
    # front) gets an implicit ``reps=1`` and is left untiled.
    return _result(trace, bind(ops.d_tile, _move_bdim_to_front(t), reps), 0)


# -- getitem (incl. batched gather) -----------------------------------------
def _getitem_rule(trace: BatchTrace, x: Boxed, key: Index) -> BatchTracer:
    tx = trace._raise(x)
    keys = key if isinstance(key, tuple) else (key,)
    if any(isinstance(k, BatchTracer) and k.bdim is not None for k in keys):
        return _batched_gather(trace, tx, keys)
    # Static key (same across the batch).
    static = tuple(_unwrap(k) for k in keys)
    if tx.bdim is None:
        unwrapped = static if isinstance(key, tuple) else static[0]
        return _result(trace, bind(ops.d_getitem, tx.value, unwrapped), None)
    v = _move_bdim_to_front(tx)
    skip = (slice(None),) + static  # skip the batch axis
    return _result(trace, bind(ops.d_getitem, v, skip), 0)


def _peel_x(v: Boxed) -> "tuple[Boxed, list[BatchTrace]]":
    """Peel a (possibly nested) batched ``x`` down to its bottom ``Var``/array, moving
    each level's batch axis to the front as it goes, and record the trace at each level
    (outermost first). The returned bottom value has all ``nb`` batch axes leading, in
    stack order, so a single advanced-index gather (with one ``arange`` per batch axis)
    selects each example's own row."""
    traces: list[BatchTrace] = []
    while isinstance(v, BatchTracer):
        if v.bdim is not None:
            traces.append(cast(BatchTrace, v._trace))
            v = _move_bdim_to_front(v)
        else:
            v = v.value
    return v, traces


def _peel_to_array(v: Boxed) -> "np.ndarray":
    """Materialize a (non-differentiable) index value to a concrete numpy array with its
    batch axes leading (outermost first)."""
    bottom, _ = _peel_x(v)
    return np.asarray(cast(Any, _value(cast(Any, bottom))))


def _batched_gather(
    trace: BatchTrace, tx: BatchTracer, keys: tuple[Index, ...]
) -> BatchTracer:
    """``x[idx]`` with a per-example integer-array index. Indexes the first logical axis
    of ``x`` with each example's own indices, via advanced indexing paired with an
    ``arange`` over every batch axis; the gather rides ``ops.d_getitem`` so its
    scatter-add backward (hence grad) comes for free. Works at any nesting depth: ``x``
    is peeled to its bottom ``Var`` with every batch axis leading, the
    (non-differentiable) index is materialized the same way, and one ``arange`` per *x*
    batch axis pairs example ``(i, j, ...)`` of the index with row ``(i, j, ...)`` of
    ``x``; the gathered result is then re-wrapped one ``bdim=0`` level per peeled level.
    """
    if len(keys) != 1:
        raise NotImplementedError(
            "vmap: a batched index is only supported as a single key (x[idx]), "
            "not combined with other indices in a tuple"
        )
    _, key_traces = _peel_x(keys[0])
    idx_arr = _peel_to_array(keys[0]).astype(np.intp)
    xbottom, traces = _peel_x(tx)
    nb = len(traces)  # number of leading batch axes on the bottom value
    shp = _phys_shape(xbottom)
    extra = idx_arr.ndim - nb  # trailing (non-batch) index axes
    aranges: list = []
    for k in range(nb):
        a = np.arange(shp[k])
        reshape = [1] * (nb + extra)
        reshape[k] = shp[k]
        aranges.append(a.reshape(reshape))
    gathered = ops.d_getitem(cast(Any, xbottom), tuple(aranges) + (idx_arr,))
    if nb == 0:
        # Shared/unbatched table: ``x`` carries no batch axis, so the result's batch axes
        # come entirely from the per-example index. ``idx_arr`` has its batch axes leading
        # (outermost first), so ``x[idx]`` already has bdim=0 per key level; re-wrap one
        # ``bdim=0`` level per key batch level (innermost first).
        out: Boxed = gathered
        for tr in reversed(key_traces):
            out = BatchTracer(tr, out, 0)
        return cast(BatchTracer, out)
    # Re-wrap: the gather put all batch axes at the front (bdim=0 per level), innermost
    # level wrapping the bottom value first.
    out = gathered
    for tr in reversed(traces):
        out = BatchTracer(tr, out, 0)
    return cast(BatchTracer, out)


# -- concatenate / stack ----------------------------------------------------
def _unwrap(s: Boxed) -> Boxed:
    return s.value if isinstance(s, BatchTracer) else s


def _join_for(op: Prim, new_axis: bool) -> Rule:
    def rule(trace: BatchTrace, seq: Sequence[Boxed], axis: int = 0) -> BatchTracer:
        tracers = [trace._raise(s) for s in seq]
        if not any(t.bdim is not None for t in tracers):
            return _result(trace, bind(op, [t.value for t in tracers], axis=axis), None)
        if not all(t.bdim is not None for t in tracers):
            raise NotImplementedError(
                "vmap: concatenate/stack mixing batched and unbatched operands is "
                "not supported yet"
            )
        n = _logical_ndim(tracers[0])
        if new_axis:
            ax = axis + 1 if axis >= 0 else axis + n + 2
        else:
            ax = axis + 1 if axis >= 0 else axis + n + 1
        fronted = [_move_bdim_to_front(t) for t in tracers]
        return _result(trace, bind(op, fronted, axis=ax), 0)

    return rule


def _unmapped_join_for(name: str) -> Rule:
    def rule(trace: BatchTrace, seq: Sequence[Boxed], **kw: Any) -> BatchTracer:
        tracers = [trace._raise(s) for s in seq]
        if not any(t.bdim is not None for t in tracers):
            return _result(
                trace, bind(getattr(ops, name), [t.value for t in tracers], **kw), None
            )
        raise NotImplementedError(
            f"vmap: {name} over batched operands is not supported yet"
        )

    return rule


def _process_unmapped(prim: Prim) -> "NoReturn":
    raise NotImplementedError(
        f"vmap: no batching rule for {getattr(prim, '__name__', prim)!r}; "
        "cannot vectorize it. Rewrite the net using ops pycograd has a rule for."
    )


# ---------------------------------------------------------------------------
# Rule registry, keyed by primitive.
# ---------------------------------------------------------------------------
def _build_rule_for() -> dict[Prim, Rule]:
    unary = (
        ops.d_exp,
        ops.d_log,
        ops.d_sin,
        ops.d_cos,
        ops.d_tanh,
        ops.d_sqrt,
        ops.d_sigmoid,
        ops.d_sinh,
        ops.d_cosh,
        ops.d_arctan,
        ops.d_tan,
        ops.d_arcsin,
        ops.d_arccos,
        ops.d_arctanh,
        ops.d_arcsinh,
        ops.d_arccosh,
        ops.d_exp2,
        ops.d_log2,
        ops.d_log10,
        ops.d_deg2rad,
        ops.d_rad2deg,
        ops.d_sign,
        ops.d_ceil,
        ops.d_floor,
        ops.d_log1p,
        ops.d_expm1,
        ops.d_abs,
        ops.d_square,
        ops.d_reciprocal,
    )
    rule_for: dict[Prim, Rule] = {p: _elementwise_for(p) for p in unary}
    rule_for.update(
        {
            ops.d_add: _elementwise_for(ops.d_add),
            ops.d_sub: _elementwise_for(ops.d_sub),
            ops.d_mul: _elementwise_for(ops.d_mul),
            ops.d_gated_act: _elementwise_for(ops.d_gated_act),
            ops.d_div: _elementwise_for(ops.d_div),
            ops.d_mod: _elementwise_for(ops.d_mod),
            ops.d_neg: _elementwise_for(ops.d_neg),
            ops.d_pow: _pow_rule,  # exponent stays constant; see _pow_rule
            ops.d_lt: _elementwise_for(ops.d_lt),
            ops.d_le: _elementwise_for(ops.d_le),
            ops.d_gt: _elementwise_for(ops.d_gt),
            ops.d_ge: _elementwise_for(ops.d_ge),
            ops.d_eq: _elementwise_for(ops.d_eq),
            ops.d_ne: _elementwise_for(ops.d_ne),
            ops.d_getitem: _getitem_rule,
            ops.d_conj: _elementwise_for(ops.d_conj),
            ops.d_real: _elementwise_for(ops.d_real),
            ops.d_imag: _elementwise_for(ops.d_imag),
            ops.d_angle: _elementwise_for(ops.d_angle),
            ops.d_maximum: _elementwise_for(ops.d_maximum),
            ops.d_fmax: _elementwise_for(ops.d_fmax),
            ops.d_fmin: _elementwise_for(ops.d_fmin),
            ops.d_logaddexp: _elementwise_for(ops.d_logaddexp),
            ops.d_logaddexp2: _elementwise_for(ops.d_logaddexp2),
            ops.d_minimum: _elementwise_for(ops.d_minimum),
            ops.d_where: _elementwise_for(ops.d_where),
            ops.d_clip: _elementwise_for(ops.d_clip),
            ops.d_sum: _reduce_for(ops.d_sum),
            ops.d_prod: _reduce_for(ops.d_prod),
            ops.d_mean: _reduce_for(ops.d_mean),
            ops.d_var: _reduce_for(ops.d_var),
            ops.d_std: _reduce_for(ops.d_std),
            ops.d_max: _reduce_for(ops.d_max),
            ops.d_min: _reduce_for(ops.d_min),
            ops.d_softmax: _axis_preserving_for(ops.d_softmax),
            ops.d_logsumexp: _reduce_for(ops.d_logsumexp),
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
            ops.d_split: ops.split_transform_rule("split"),
            ops.d_array_split: ops.split_transform_rule("array_split"),
            ops.d_vsplit: ops.split_transform_rule("vsplit"),
            ops.d_hsplit: ops.split_transform_rule("hsplit"),
            ops.d_dsplit: ops.split_transform_rule("dsplit"),
            ops.d_diff: ops.diff_transform_rule,
            ops.d_diag: ops.diag_transform_rule,
            ops.d_diagonal: ops.diagonal_transform_rule,
            ops.d_sort: _sort_rule,
            ops.d_partition: _partition_rule,
            ops.d_select: ops.select_transform_rule,
            ops.d_gradient: ops.gradient_transform_rule,
            ops.d_append: ops.append_transform_rule,
            ops.d_flipud: ops._flip_transform_rule(0),
            ops.d_fliplr: ops._flip_transform_rule(1),
            ops.d_rot90: ops.rot90_transform_rule,
            ops.d_trace: ops.trace_transform_rule,
            ops.d_outer: ops.outer_transform_rule,
            ops.d_cross: ops.cross_transform_rule,
            ops.d_kron: ops.kron_transform_rule,
            ops.d_array: ops.array_transform_rule,
            ops.d_ravel: ops._reshape_lowering_transform(ops.ravel_shape),
            ops.d_squeeze: ops._reshape_lowering_transform(ops.squeeze_shape),
            ops.d_atleast_1d: ops._reshape_lowering_transform(ops.atleast_1d_shape),
            ops.d_atleast_2d: ops._reshape_lowering_transform(ops.atleast_2d_shape),
            ops.d_atleast_3d: ops._reshape_lowering_transform(ops.atleast_3d_shape),
            ops.d_einsum: _einsum_rule,
            ops.d_cumsum: _cumsum_rule,
            ops.d_transpose: _transpose_rule,
            ops.d_reshape: _reshape_rule,
            ops.d_astype: _astype_rule,
            ops.d_broadcast_to: _broadcast_to_rule,
            ops.d_expand_dims: _expand_dims_rule,
            ops.d_concatenate: _join_for(ops.d_concatenate, new_axis=False),
            ops.d_stack: _join_for(ops.d_stack, new_axis=True),
            ops.d_vstack: _unmapped_join_for("d_vstack"),
            ops.d_hstack: _unmapped_join_for("d_hstack"),
            ops.d_column_stack: _unmapped_join_for("d_column_stack"),
            ops.d_dstack: _unmapped_join_for("d_dstack"),
        }
    )
    return rule_for


# ``_RULE_FOR`` is keyed by *primitive* (incl. the operator primitives), consulted by
# ``BatchTrace.process_primitive``. ``_BATCH`` denormalizes it to numpy-callable keys so
# the coverage-parity test (``set(_BATCH) == set(ops._INTERCEPT)``) still holds.
_RULE_FOR: dict[Prim, Rule] = _build_rule_for()
_BATCH: dict[Prim, Rule] = {
    fn: _RULE_FOR[prim] for prim, fns in ops._RULES.items() for fn in fns
}

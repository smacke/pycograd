# -*- coding: utf-8 -*-
"""Shape inference: what shapes a net produces, without training it.

pycograd is *eager* -- every :class:`~pycograd.tensor.Var` holds a concrete array, so
a shape is only ever known *after* an op runs. This module recovers the shapes ahead
of time, for two everyday uses: a Keras-style :func:`summary` of a model's parameters
and output, and friendlier shape-mismatch errors (:class:`ShapeError`).

:func:`eval_shape` runs ``fn`` for its shapes one of two ways:

* ``method="abstract"`` (and ``"auto"``, the default) -- carry only ``(shape, dtype)``
  through :class:`ShapedArray` values on a dedicated :class:`AbstractTrace` level of the
  trace-level stack, with one shape rule per primitive (dispatched via
  :func:`~pycograd.trace.bind`). Allocates nothing (so it sizes a 100000x100000 matmul
  instantly). A
  *data-dependent* dimension (e.g. the length of a boolean-mask index, which depends on
  values) flows through as a symbolic :class:`~pycograd._dims.Dim` rather than a guess
  or an error -- so the inferred shape might read ``f64[n0]``.
* ``method="dummy"`` -- run plain numpy on zero-filled **dummy** arrays and read the
  result's shape. Needs no tape and no tracer, but allocates and silently mis-sizes
  data-dependent shapes. It is the conformance oracle the abstract path is tested
  against for data-*independent* nets (it cannot model symbolic dims).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterator, Sequence, Union, cast

import numpy as np

from pycograd import _constraints, _dims
from pycograd._dims import Dim
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
from pycograd.params import Param, Weight, _TieRef
from pycograd.tensor import Var, _is_numeric, _value
from pycograd.trace import MainTrace, Trace, Tracer, new_main
from pycograd.tree import Leaf, PyTree, tree_flatten, tree_unflatten

# A value an abstract shape rule can consume: an abstract ``ShapedArray`` (the live
# ``eval_shape`` case), a ``ShapeDtypeStruct`` spec, a ``Param``, or any ``Operand``
# (a tape ``Var``, a raw scalar/array constant, or a ``Weight`` proxy). The ``object``
# annotations these rules and their coercion helpers (``_aval`` / ``_spec_of``) used to
# carry all denote this set -- it is the abstract-level superset of a pytree ``Leaf``.
AbstractVal = Union["ShapedArray", "ShapeDtypeStruct", Param, Operand, None]

# A short numpy-dtype tag for ``__repr__`` ("f64", "i32", ...).
_DTYPE_TAG = {
    "float64": "f64",
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
    "int64": "i64",
    "int32": "i32",
}


# ---------------------------------------------------------------------------
# Shape errors.
# ---------------------------------------------------------------------------
class ShapeError(ValueError):
    """A shape mismatch, annotated with the operation and the operand shapes.

    Raised in place of numpy's opaque message (e.g. matmul's "shapes (3,) and (4,2)
    not aligned") so the report names the op and what flowed into it. Subclasses
    ``ValueError`` so existing ``except ValueError`` handlers still catch it.
    """


def _norm_dim(d: int | str | Dim) -> int | Dim:
    """Normalize one dimension: a symbolic :class:`Dim` passes through; a ``str`` names
    a symbolic input dimension (``"B"`` -> the symbol ``B``, so same-named dims across
    inputs are the same symbol); everything else coerces to a plain ``int``."""
    if isinstance(d, Dim):
        return d
    if isinstance(d, str):
        return _dims.symbol(d, name=d)
    return int(cast(Any, d))


# Provenance: a hashable fingerprint of how an abstract value was produced, used only
# to dedupe data-dependent symbols (so identical ``x > 0`` masks share one symbol).
_PROV_CTR = itertools.count()


def _fresh_prov() -> tuple:
    """A unique token -- the default provenance, so distinct values never merge."""
    return ("fresh", next(_PROV_CTR))


def _prov_of(x: AbstractVal) -> Hashable:
    """The provenance of an operand: a value's own ``prov``; a structural tag for a
    concrete constant; else a fresh token."""
    if isinstance(x, ShapedArray):
        return x.prov
    if _is_numeric(x) and np.ndim(cast(Any, x)) == 0:
        item = cast(Any, x).item() if hasattr(x, "item") else x
        return ("const", type(x).__name__, item)
    return _fresh_prov()


def _shape_context(op_name: str, *shapes: tuple[int | Dim, ...]) -> str:
    joined = " and ".join(str(tuple(s)) for s in shapes)
    return f"{op_name}: incompatible shapes {joined}"


# ---------------------------------------------------------------------------
# Abstract shape/dtype spec.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShapeDtypeStruct:
    """A shape + dtype with no data -- pass it to :func:`eval_shape`/:func:`summary`
    in place of a real array, the way ``jax.eval_shape`` takes ``ShapeDtypeStruct``.

    ``shape`` is a tuple of dims (ints, or a symbolic :class:`~pycograd._dims.Dim` for
    a data-dependent size); ``dtype`` defaults to float64 (pycograd's working dtype).
    ``ndim``/``size`` mirror the numpy attributes so it reads like an array.
    """

    shape: tuple[int | Dim, ...]
    dtype: np.dtype = np.dtype(np.float64)

    def __post_init__(self) -> None:
        # Normalize so callers may pass a list, a bare int, or a numpy dtype string.
        shape = self.shape
        if isinstance(shape, int):
            shape = (shape,)
        object.__setattr__(self, "shape", tuple(_norm_dim(d) for d in shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int | Dim:
        return _dims.prod_dims(self.shape)

    def __repr__(self) -> str:
        tag = _DTYPE_TAG.get(self.dtype.name, self.dtype.name)
        return f"{tag}[{','.join(map(str, self.shape))}]"


def _spec_of(x: AbstractVal) -> ShapeDtypeStruct:
    """The shape/dtype of an array, ``Var``, ``Param``, or ``ShapeDtypeStruct``."""
    if isinstance(x, ShapeDtypeStruct):
        return x
    if isinstance(x, ShapedArray):
        return ShapeDtypeStruct(x.shape, x.dtype)
    if isinstance(x, Var):
        return ShapeDtypeStruct(x.value.shape, x.value.dtype)
    if isinstance(x, Param):
        arr = np.asarray(x.value)
        return ShapeDtypeStruct(arr.shape, arr.dtype)
    arr = np.asarray(x)
    return ShapeDtypeStruct(arr.shape, arr.dtype)


# ---------------------------------------------------------------------------
# Dummy-array inference (Tier A).
# ---------------------------------------------------------------------------
def _dummy(spec: ShapeDtypeStruct) -> Array:
    """A zero-filled stand-in of the given shape/dtype.

    Only the *shape* matters downstream, so zeros are as good as anything and avoid
    NaNs from ``log(0)`` etc. derailing a forward we only run to read shapes from.
    """
    return np.zeros(cast("tuple[int, ...]", spec.shape), dtype=spec.dtype)


def _dummy_leaf(leaf: Leaf) -> Leaf:
    """Replace one argument leaf with a same-shaped dummy; pass non-arrays through.

    A ``ShapeDtypeStruct``/``Var``/``Param``/number/array becomes a dummy array of its
    shape; anything else (a ``bool`` flag like ``training=``, ``None``, a string) is
    handed through untouched, exactly as it would reach the function normally.
    """
    if isinstance(leaf, _TieRef):
        raise ValueError(
            "eval_shape: tied[...] is only meaningful inside params(...); it reached "
            "shape inference unresolved"
        )
    if isinstance(leaf, ShapeDtypeStruct):
        return _dummy(leaf)
    if isinstance(leaf, (Var, Param)) or _is_numeric(leaf):
        return _dummy(_spec_of(leaf))
    return leaf


def _to_specs(out: PyTree) -> PyTree:
    """Map every array/``Var`` leaf of a result pytree to its ``ShapeDtypeStruct``.

    A ``ShapeDtypeStruct`` is not a tape ``Leaf``, but it rides through
    ``tree_unflatten`` as one (the treedef only cares about container structure)."""
    leaves, treedef = tree_flatten(out)
    return tree_unflatten(
        treedef,
        cast(
            "list[Leaf]",
            [None if leaf is None else _spec_of(leaf) for leaf in leaves],
        ),
    )


def _subs_spec(spec: ShapeDtypeStruct, mapping: dict) -> ShapeDtypeStruct:
    shape = tuple(
        d.subs(mapping) if isinstance(d, Dim) else d for d in spec.shape  # type: ignore[union-attr]
    )
    return ShapeDtypeStruct(shape, spec.dtype)


def _substitute_specs(specs: PyTree, mapping: dict) -> PyTree:
    """Apply a symbol-key -> value ``mapping`` to every ``ShapeDtypeStruct`` leaf."""
    leaves, treedef = tree_flatten(specs)
    return tree_unflatten(
        treedef,
        cast(
            "list[Leaf]",
            [
                None if s is None else _subs_spec(cast(ShapeDtypeStruct, s), mapping)
                for s in leaves
            ],
        ),
    )


def _eval_shape_dummy(fn: Callable[..., PyTree], args: tuple[PyTree, ...]) -> PyTree:
    call_args = [
        tree_unflatten(treedef, [_dummy_leaf(leaf) for leaf in leaves])
        for leaves, treedef in (tree_flatten(a) for a in args)
    ]
    return _to_specs(cast(PyTree, fn(*call_args)))


# ---------------------------------------------------------------------------
# Abstract values (Tier B): shape + dtype, no data.
# ---------------------------------------------------------------------------
class ShapedArray(Tracer):
    """An abstract array -- a shape and dtype with no data behind them -- as a
    :class:`~pycograd.trace.Tracer`.

    It is the value an :class:`AbstractTrace` level computes on: operators and numpy
    calls on a ``ShapedArray`` produce another ``ShapedArray`` via the shape rules below,
    never touching real numbers. Unlike the dummy-array path it allocates nothing, and a
    *data-dependent* dimension (e.g. the length of ``x[x > 0]``) flows through as a
    symbolic :class:`~pycograd._dims.Dim` rather than a wrong guess.

    Like :class:`~pycograd.batching.BatchTracer` / :class:`~pycograd.forward.JVPTracer`,
    a ``ShapedArray`` is the :class:`Tracer` of its level: under :func:`eval_shape`'s
    :class:`AbstractTrace` an operand routed through :func:`~pycograd.trace.bind` selects
    that level, whose ``process_primitive`` runs the shape rule. A ``ShapedArray``
    constructed *outside* a live level (e.g. ``ShapedArray((3, 4))`` directly, or one a
    rule mints) is tagged with a sentinel level-0 trace, so its own operator dunders
    (which call the same rules) still work standalone -- exactly the surface
    ``test_shapedarray_metadata_is_concrete`` exercises.

    The read-only surface a model performs on a tensor is mirrored from
    :class:`~pycograd.tensor.Var`: ``.shape``/``.ndim``/``.size``/``.dtype`` are concrete
    (so ``q.shape[-1] ** -0.5`` and ``rng.random(x.shape)`` work) -- though a dim may be
    a symbolic ``Dim`` -- only the *element data* is abstract. ``__array_ufunc__ = None``
    makes numpy defer ufuncs/operators to our reflected methods, exactly as ``Var`` does.

    ``prov`` is a hashable *provenance* fingerprint used only to recognize when two
    data-dependent values are structurally the same (so two ``x > 0`` masks intern the
    same symbol); a freshly-minted token by default, so distinct values never merge.
    """

    __slots__ = ("shape", "dtype", "prov", "_trace")
    __array_ufunc__ = None

    def __init__(
        self,
        shape: tuple[int | Dim, ...] | int,
        dtype: DTypeLike = np.float64,
        prov: Hashable = None,
        trace: "Trace | None" = None,
    ) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        self.shape: tuple[int | Dim, ...] = tuple(_norm_dim(d) for d in shape)
        self.dtype: np.dtype = np.dtype(cast(Any, dtype))
        self.prov: Hashable = _fresh_prov() if prov is None else prov
        # A value not built under a live ``AbstractTrace`` (a bare ``ShapedArray(...)`` or
        # one a rule mints) rides the sentinel level-0 trace, so it never out-ranks a real
        # level in ``find_top_trace`` and its standalone dunders still work.
        self._trace: Trace = _SENTINEL_ABSTRACT_TRACE if trace is None else trace

    # -- Tracer interface ----------------------------------------------------
    @property
    def aval(self) -> "ShapeDtypeStruct":
        return ShapeDtypeStruct(self.shape, self.dtype)

    # -- concrete metadata (mirrors Var) -------------------------------------
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int | Dim:
        return _dims.prod_dims(self.shape)

    def _keep(self, result: ShapedArray) -> ShapedArray:
        """Tag a rule's result with *this* value's trace so a ``ShapedArray`` minted by a
        direct dunder / method call (``x.T``, ``x.sum()``, ``x[key]``) keeps flowing at the
        live ``AbstractTrace`` level rather than dropping to the sentinel. For a standalone
        ``ShapedArray`` this re-tags with the (level-0) sentinel -- a no-op."""
        result._trace = self._trace
        return result

    @property
    def T(self) -> ShapedArray:
        return self._keep(abstract_transpose(self))

    # -- arithmetic: every elementwise op is a broadcast of operand shapes ----
    def __add__(self, o: AbstractVal) -> ShapedArray:
        return self._keep(_ew_binary(self, o))

    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __add__
    __truediv__ = __rtruediv__ = __add__

    def __pow__(self, p: AbstractVal) -> ShapedArray:
        return self._keep(_ew_binary(self, p))

    __rpow__ = __pow__

    def __neg__(self) -> ShapedArray:
        return self._keep(ShapedArray(self.shape, self.dtype))

    def __abs__(self) -> ShapedArray:
        return self._keep(ShapedArray(self.shape, self.dtype))

    def __matmul__(self, o: AbstractVal) -> ShapedArray:
        return self._keep(abstract_matmul(self, o))

    def __rmatmul__(self, o: AbstractVal) -> ShapedArray:
        return self._keep(abstract_matmul(o, self))

    # -- comparisons return a boolean abstract mask (broadcast shape) ---------
    # The op tag enters ``prov`` so ``x > 0`` and ``x < 0`` get distinct fingerprints
    # (a collision would unsoundly claim their masked lengths are equal).
    def _cmp(self, o: AbstractVal, op: str) -> ShapedArray:
        prov = ("cmp", op, _prov_of(self), _prov_of(o))
        return self._keep(
            ShapedArray(_broadcast_shape("compare", self, o), np.dtype(bool), prov)
        )

    def __lt__(self, o: AbstractVal) -> ShapedArray:
        return self._cmp(o, "lt")

    def __le__(self, o: AbstractVal) -> ShapedArray:
        return self._cmp(o, "le")

    def __gt__(self, o: AbstractVal) -> ShapedArray:
        return self._cmp(o, "gt")

    def __ge__(self, o: AbstractVal) -> ShapedArray:
        return self._cmp(o, "ge")

    def __getitem__(self, key: Index) -> ShapedArray:
        return self._keep(abstract_getitem(self, key))

    # -- numpy-method surface (x.sum(...), x.reshape(...), ...) ---------------
    def __getattr__(self, name: str) -> "Callable[..., Boxed]":
        if name.startswith("__"):
            raise AttributeError(name)
        np_fn = getattr(np, name, None)
        prim = _ABSTRACT.get(np_fn) if callable(np_fn) else None
        if prim is None:
            raise AttributeError(name)

        def _method(*a: BindArg, **k: Any) -> Boxed:
            out = prim(self, *a, **k)
            return self._keep(out) if isinstance(out, ShapedArray) else out

        return _method

    def __repr__(self) -> str:
        return f"ShapedArray({_spec_of(self)!r})"


def _aval(x: AbstractVal) -> ShapedArray:
    """Coerce an array / number / ``ShapeDtypeStruct`` / ``ShapedArray`` to an aval.

    A ``Var``/``Param`` reaches here when a pure-constant subexpression (no abstract
    operand) is evaluated eagerly mid-``eval_shape`` -- e.g. ``np.reshape`` of a
    closure array routes through the base ``EvalTrace`` and yields a concrete
    ``Var``. Its value carries a real shape/dtype, so read those directly rather
    than ``np.asarray`` (which a ``Var`` rejects)."""
    if isinstance(x, ShapedArray):
        return x
    if isinstance(x, ShapeDtypeStruct):
        return ShapedArray(x.shape, x.dtype)
    if isinstance(x, Weight):
        x = _value(
            x
        )  # an ambient-parameter proxy -> its live array/Var (sizes like Param)
    if isinstance(x, Var):
        return ShapedArray(x.value.shape, x.value.dtype)
    if isinstance(x, Param):
        x = x.value
    arr = np.asarray(x)
    return ShapedArray(arr.shape, arr.dtype)


def _broadcast_shape(op: str, *operands: AbstractVal) -> tuple[int | Dim, ...]:
    shapes = [_aval(o).shape for o in operands]
    try:
        return _dims.broadcast_shapes(*shapes)
    except ValueError as e:
        raise ShapeError(_shape_context(op, *shapes)) from e


def _result_dtype(*operands: AbstractVal) -> np.dtype:
    return np.result_type(*[_aval(o).dtype for o in operands])


# ---------------------------------------------------------------------------
# Shape rules -- one per differentiable primitive in ``ops._RULES``.
# ---------------------------------------------------------------------------
def abstract_unary(x: AbstractVal) -> ShapedArray:
    a = _aval(x)
    return ShapedArray(a.shape, a.dtype)


def abstract_axis_preserving(
    x: AbstractVal, axis: Axis = None, **_: Any
) -> ShapedArray:
    # Shape-preserving but ``axis``-bearing (e.g. ``softmax``): same shape/dtype as the
    # input, but it accepts the ``axis`` bind-param ``abstract_unary`` would reject.
    a = _aval(x)
    return ShapedArray(a.shape, a.dtype)


def _ew_binary(a: AbstractVal, b: AbstractVal) -> ShapedArray:
    return ShapedArray(_broadcast_shape("elementwise", a, b), _result_dtype(a, b))


def abstract_binary(a: AbstractVal, b: AbstractVal) -> ShapedArray:
    return _ew_binary(a, b)


def _abstract_compare_for(op: str) -> Callable[..., ShapedArray]:
    """A comparison shape rule for one op (``"lt"``/``"gt"``/...). Like
    :meth:`ShapedArray._cmp`, it tags the boolean result with a dedup *provenance* built
    from the op and the operands, so two structurally identical masks (``x > 0`` twice)
    intern one symbolic length while distinct predicates (``x > 0`` vs ``x < 0``) do not.
    Per-op so the op tag enters ``prov`` -- the same fingerprint the dunder produces."""

    def rule(a: AbstractVal, b: AbstractVal) -> ShapedArray:
        prov = ("cmp", op, _prov_of(a), _prov_of(b))
        return ShapedArray(_broadcast_shape("compare", a, b), np.dtype(bool), prov)

    return rule


# Back-compat alias: a single bare comparison rule (no dedup provenance). Unused by the
# trace level (which uses the per-op rules above) but kept for any direct caller.
def abstract_compare(a: AbstractVal, b: AbstractVal) -> ShapedArray:
    return ShapedArray(_broadcast_shape("compare", a, b), np.dtype(bool))


def abstract_where(cond: AbstractVal, a: AbstractVal, b: AbstractVal) -> ShapedArray:
    return ShapedArray(_broadcast_shape("where", cond, a, b), _result_dtype(a, b))


def abstract_clip(
    x: AbstractVal, a_min: AbstractVal = None, a_max: AbstractVal = None
) -> ShapedArray:
    bounds = [b for b in (x, a_min, a_max) if b is not None]
    return ShapedArray(_broadcast_shape("clip", *bounds), _aval(x).dtype)


def abstract_matmul(a: AbstractVal, b: AbstractVal) -> ShapedArray:
    av, bv = _aval(a), _aval(b)
    sa, sb = av.shape, bv.shape
    dtype = _result_dtype(a, b)
    if len(sa) == 0 or len(sb) == 0:
        raise ShapeError(_shape_context("matmul (needs >=1-D operands)", sa, sb))
    if len(sa) == 1 and len(sb) == 1:
        if not _constraints.register_eq(sa[0], sb[0]):
            raise ShapeError(_shape_context("matmul", sa, sb))
        return ShapedArray((), dtype)
    if len(sa) == 1:  # (k,) @ (..., k, n) -> (..., n)
        if not _constraints.register_eq(sa[0], sb[-2]):
            raise ShapeError(_shape_context("matmul", sa, sb))
        return ShapedArray(tuple(sb[:-2]) + (sb[-1],), dtype)
    if len(sb) == 1:  # (..., m, k) @ (k,) -> (..., m)
        if not _constraints.register_eq(sa[-1], sb[0]):
            raise ShapeError(_shape_context("matmul", sa, sb))
        return ShapedArray(tuple(sa[:-1]), dtype)
    if not _constraints.register_eq(sa[-1], sb[-2]):  # (..., m, k) @ (..., k, n)
        raise ShapeError(_shape_context("matmul", sa, sb))
    try:
        batch = _dims.broadcast_shapes(sa[:-2], sb[:-2])
    except ValueError as e:
        raise ShapeError(_shape_context("matmul (batch dims)", sa, sb)) from e
    return ShapedArray(batch + (sa[-2], sb[-1]), dtype)


def abstract_einsum(subscripts: str, *operands: AbstractVal) -> ShapedArray:
    from pycograd import ops

    avals = [_aval(o) for o in operands]
    ins, out = ops._parse_einsum(subscripts, [len(av.shape) for av in avals])
    label_dim: dict[str, "int | Dim"] = {}
    for sub, av in zip(ins, avals):
        if len(sub) != len(av.shape):
            raise ShapeError(f"einsum: subscript {sub!r} vs operand shape {av.shape}")
        for label, d in zip(sub, av.shape):
            prev = label_dim.get(label)
            if prev is None:
                label_dim[label] = d
            else:
                label_dim[label] = _einsum_label_dim(prev, d, label)
    return ShapedArray(tuple(label_dim[c] for c in out), _result_dtype(*avals))


def _einsum_label_dim(a: "int | Dim", b: "int | Dim", label: str) -> "int | Dim":
    """Resolve a shared einsum label's size with numpy broadcasting: a concrete ``1``
    yields the other; otherwise the two extents must match (``register_eq``)."""
    if not isinstance(a, Dim) and int(a) == 1:
        return b
    if not isinstance(b, Dim) and int(b) == 1:
        return a
    if not _constraints.register_eq(a, b):
        raise ShapeError(f"einsum: conflicting sizes for index {label!r}")
    return a


def abstract_cumsum(x: AbstractVal, axis: "int | None" = None) -> ShapedArray:
    a = _aval(x)
    return ShapedArray(a.shape, a.dtype)  # cumsum is shape-preserving for an int axis


def abstract_reduce(
    x: AbstractVal, axis: Axis = None, keepdims: bool = False, **_: Any
) -> ShapedArray:
    a = _aval(x)
    shp = a.shape
    new: tuple[int | Dim, ...]
    if axis is None:
        new = tuple(1 for _ in shp) if keepdims else ()
    else:
        axes = axis if isinstance(axis, tuple) else (axis,)
        axes = tuple(ax % len(shp) for ax in axes)
        if keepdims:
            new = tuple(1 if i in axes else d for i, d in enumerate(shp))
        else:
            new = tuple(d for i, d in enumerate(shp) if i not in axes)
    return ShapedArray(new, a.dtype)


def abstract_transpose(
    x: AbstractVal, axes: tuple[int, ...] | None = None
) -> ShapedArray:
    a = _aval(x)
    if axes is None:
        return ShapedArray(tuple(reversed(a.shape)), a.dtype)
    return ShapedArray(tuple(a.shape[ax] for ax in axes), a.dtype)


def abstract_reshape(x: AbstractVal, *shape: Shape) -> ShapedArray:
    a = _aval(x)
    newshape = shape[0] if len(shape) == 1 else shape
    if isinstance(newshape, int):
        newshape = (newshape,)
    newshape = tuple(int(d) for d in cast(Sequence, newshape))
    size = a.size  # int, or a Dim when the input has a symbolic dim
    if -1 in newshape:
        known = 1
        for d in newshape:
            if d != -1:
                known *= d
        if isinstance(size, int):  # concrete: validate divisibility as before
            if known == 0 or size % known != 0:
                raise ShapeError(f"reshape: cannot reshape size {size} into {newshape}")
        newshape = tuple(size // known if d == -1 else d for d in newshape)
    # The total-size check only holds when nothing symbolic is involved; a symbolic
    # size is trusted (it flowed in from a data-dependent dim).
    if isinstance(size, int) and not _dims.has_symbol(newshape):
        if int(np.prod(cast("tuple[int, ...]", newshape))) != size:
            raise ShapeError(f"reshape: cannot reshape size {size} into {newshape}")
    return ShapedArray(newshape, a.dtype)


def abstract_broadcast_to(x: AbstractVal, shape: Shape) -> ShapedArray:
    a = _aval(x)
    target = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
    return ShapedArray(tuple(int(d) for d in target), a.dtype)


def abstract_scatter(
    g: AbstractVal, key: object, shape: Shape, dtype: DTypeLike
) -> ShapedArray:
    # The internal scatter (a getitem VJP) produces a buffer of the given shape/dtype.
    # Reached only when ``_grad_graph`` records a backward pass; harmless elsewhere.
    target = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
    return ShapedArray(tuple(int(d) for d in target), np.dtype(cast(Any, dtype)))


def abstract_expand_dims(x: AbstractVal, axis: int) -> ShapedArray:
    a = _aval(x)
    pos = axis if axis >= 0 else axis + a.ndim + 1
    shp = list(a.shape)
    shp.insert(pos, 1)
    return ShapedArray(tuple(shp), a.dtype)


def abstract_concatenate(seq: Sequence[AbstractVal], axis: int = 0) -> ShapedArray:
    avals = [_aval(s) for s in seq]
    ndim = avals[0].ndim
    ax = axis % ndim
    out = list(avals[0].shape)
    for a in avals[1:]:
        if a.ndim != ndim or not all(
            _constraints.register_eq(d, out[i])
            for i, d in enumerate(a.shape)
            if i != ax
        ):
            raise ShapeError(
                _shape_context("concatenate", *(a.shape for a in avals))
                + f" along axis {axis}"
            )
        out[ax] += a.shape[ax]
    return ShapedArray(tuple(out), np.result_type(*[a.dtype for a in avals]))


def abstract_stack(seq: Sequence[AbstractVal], axis: int = 0) -> ShapedArray:
    return abstract_concatenate([abstract_expand_dims(s, axis) for s in seq], axis=axis)


def _atleast_2d_row(x: AbstractVal) -> ShapedArray:
    a = _aval(x)
    if a.ndim == 0:
        return abstract_reshape(a, (1, 1))
    if a.ndim == 1:
        return abstract_reshape(a, (1, a.shape[0]))
    return a


def abstract_vstack(seq: Sequence[AbstractVal]) -> ShapedArray:
    return abstract_concatenate([_atleast_2d_row(s) for s in seq], axis=0)


def abstract_hstack(seq: Sequence[AbstractVal]) -> ShapedArray:
    avals = [_aval(s) for s in seq]
    axis = 0 if all(a.ndim <= 1 for a in avals) else 1
    return abstract_concatenate(avals, axis=axis)


def abstract_column_stack(seq: Sequence[AbstractVal]) -> ShapedArray:
    parts: list[AbstractVal] = []
    for s in seq:
        a = _aval(s)
        parts.append(abstract_reshape(a, (a.shape[0], 1)) if a.ndim == 1 else a)
    return abstract_concatenate(parts, axis=1)


def _atleast_3d_depth(x: AbstractVal) -> ShapedArray:
    a = _aval(x)
    if a.ndim == 0:
        return abstract_reshape(a, (1, 1, 1))
    if a.ndim == 1:
        return abstract_reshape(a, (1, a.shape[0], 1))
    if a.ndim == 2:
        return abstract_reshape(a, a.shape + (1,))
    return a


def abstract_dstack(seq: Sequence[AbstractVal]) -> ShapedArray:
    return abstract_concatenate([_atleast_3d_depth(s) for s in seq], axis=2)


def abstract_getitem(x: AbstractVal, key: Index) -> ShapedArray:
    """Shape of ``x[key]`` for basic *and* array indexing.

    Basic keys (int/slice/ellipsis/newaxis) over a concrete shape are resolved on a
    zero-stride view (no allocation); over a symbolic shape they are resolved per-axis
    via :func:`~pycograd._dims.slice_dim`. *Advanced* (array) keys split two ways:

    * an **integer/array index** has a result shape determined by the *key's* shape
      (not its values), so it is computed exactly -- e.g. ``x[idx]`` with ``idx`` of
      shape ``(4,)`` and ``x`` of shape ``(10, 3)`` gives ``(4, 3)``;
    * a **boolean mask** is genuinely *data-dependent* (its length is the count of
      ``True``), so it contributes a symbolic :class:`~pycograd._dims.Dim`. Two
      structurally identical masks (same provenance) share one symbol.
    """
    a = _aval(x)
    keys = key if isinstance(key, tuple) else (key,)
    if not any(_is_advanced(k) for k in keys):
        if not _dims.has_symbol(a.shape):
            view = np.broadcast_to(
                np.zeros((), a.dtype), cast("tuple[int, ...]", a.shape)
            )
            return ShapedArray(view[cast(Any, key)].shape, a.dtype)
        return _basic_getitem(a, _expand_ellipsis(keys, a.ndim))
    return _advanced_getitem(a, _expand_ellipsis(keys, a.ndim))


def _is_advanced(k: Index) -> bool:
    """An array-valued key (boolean mask or integer index array / list)."""
    return isinstance(k, (ShapedArray, np.ndarray, list))


def _is_bool_key(k: Index) -> bool:
    return isinstance(k, (np.ndarray, ShapedArray)) and k.dtype == np.dtype(bool)


def _consumed(k: Index) -> int:
    """How many source axes a key element consumes (``newaxis`` adds, consumes none;
    a k-D boolean mask consumes k axes; everything else consumes one)."""
    if k is None:
        return 0
    if _is_bool_key(k):
        return _aval(k).ndim
    return 1


def _expand_ellipsis(keys: tuple[Index, ...], ndim: int) -> tuple[Index, ...]:
    """Replace a single ``...`` with the full slices it stands for."""
    if not any(k is Ellipsis for k in keys):
        return keys
    fill = ndim - sum(_consumed(k) for k in keys if k is not Ellipsis)
    out: list = []
    for k in keys:
        if k is Ellipsis:
            out.extend([slice(None)] * max(fill, 0))
        else:
            out.append(k)
    return tuple(out)


def _basic_getitem(a: ShapedArray, keys: tuple[Index, ...]) -> ShapedArray:
    """Per-axis basic indexing over a (possibly symbolic) shape: ``int`` drops an axis,
    ``slice`` maps a dim via :func:`slice_dim`, ``newaxis`` inserts ``1``."""
    out: list = []
    axis = 0
    for k in keys:
        if k is None:
            out.append(1)
        elif isinstance(k, slice):
            out.append(_dims.slice_dim(a.shape[axis], k))
            axis += 1
        else:  # an int drops its axis
            axis += 1
    out.extend(a.shape[axis:])  # trailing axes with no key are kept whole
    return ShapedArray(tuple(out), a.dtype)


def _advanced_getitem(a: ShapedArray, keys: tuple[Index, ...]) -> ShapedArray:
    """Array indexing. Integer/array keys broadcast to a single advanced block (placed
    in-position when contiguous, else at the front, mirroring numpy); a boolean mask
    contributes one symbolic count dimension."""
    pieces: list = []  # ("basic", dim) for kept axes, or ("adv",) block markers
    adv_shapes: list = []
    axis = 0
    for k in keys:
        if k is None:
            pieces.append(("basic", 1))
        elif _is_bool_key(k):
            kav = _aval(k)
            adv_shapes.append((_dims.symbol(("nonzero", kav.prov)),))
            pieces.append(("adv",))
            axis += kav.ndim
        elif _is_advanced(k):
            adv_shapes.append(_aval(k).shape)
            pieces.append(("adv",))
            axis += 1
        elif isinstance(k, slice):
            pieces.append(("basic", _dims.slice_dim(a.shape[axis], k)))
            axis += 1
        else:  # int drops its axis
            axis += 1
    for rem in range(axis, a.ndim):
        pieces.append(("basic", a.shape[rem]))

    adv_shape = _dims.broadcast_shapes(*adv_shapes) if adv_shapes else ()
    adv_at = [i for i, p in enumerate(pieces) if p[0] == "adv"]
    contiguous = adv_at == list(range(adv_at[0], adv_at[0] + len(adv_at)))

    out: list = []
    if contiguous:
        inserted = False
        for p in pieces:
            if p[0] == "adv":
                if not inserted:
                    out.extend(adv_shape)
                    inserted = True
            else:
                out.append(p[1])
    else:  # separated advanced indices -> their broadcast block leads the result
        out.extend(adv_shape)
        out.extend(p[1] for p in pieces if p[0] == "basic")
    return ShapedArray(tuple(out), a.dtype)


# The shape rules above form the ``abstract_eval(primitive, *avals) -> aval`` registry,
# keyed by primitive in ``_ABS_FOR`` and consulted by ``AbstractTrace.process_primitive``
# (so abstract evaluation rides the trace-level stack, exactly like ``vmap``/``jvp``). A
# future graph-capture tracer (cf. the ROADMAP's trace-and-compile) would reuse this same
# registry to size each node without data: ``ShapedArray`` already *is* the level's tracer,
# so the remaining increment is graph recording on top. Data-dependent dims already carry
# a symbolic :class:`~pycograd._dims.Dim` (see ``abstract_getitem``).


# ---------------------------------------------------------------------------
# The abstract interception table, keyed identically to ``ops._INTERCEPT``.
# ---------------------------------------------------------------------------
def _build_abstract_table() -> "tuple[dict[Prim, Prim], dict[Prim, Prim]]":
    """Map every numpy/math callable pycograd differentiates to its shape rule.

    Derived from ``ops._RULES`` so coverage is *identical* to ``ops._INTERCEPT`` by
    construction (asserted by a test); a newly added primitive without an entry in
    ``_ABS_FOR`` fails loudly here rather than silently lacking a shape rule.
    """
    from pycograd import ops

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
    abs_for: dict[Prim, Prim] = {p: abstract_unary for p in unary}
    abs_for.update(
        {
            # operator primitives: elementwise broadcasts / comparisons. No numpy
            # callable maps to them, so they only widen ``_ABS_FOR`` for a future
            # by-primitive ``bind``; they add no key to ``_ABSTRACT``.
            ops.d_add: abstract_binary,
            ops.d_sub: abstract_binary,
            ops.d_mul: abstract_binary,
            ops.d_gated_act: abstract_binary,
            ops.d_div: abstract_binary,
            ops.d_mod: abstract_binary,
            ops.d_neg: abstract_unary,
            ops.d_pow: abstract_binary,
            ops.d_lt: _abstract_compare_for("lt"),
            ops.d_le: _abstract_compare_for("le"),
            ops.d_gt: _abstract_compare_for("gt"),
            ops.d_ge: _abstract_compare_for("ge"),
            ops.d_eq: _abstract_compare_for("eq"),
            ops.d_ne: _abstract_compare_for("ne"),
            ops.d_getitem: abstract_getitem,
            ops.d_maximum: abstract_binary,
            ops.d_minimum: abstract_binary,
            ops.d_where: abstract_where,
            ops.d_clip: abstract_clip,
            ops.d_sum: abstract_reduce,
            ops.d_prod: abstract_reduce,
            ops.d_mean: abstract_reduce,
            ops.d_var: abstract_reduce,
            ops.d_std: abstract_reduce,
            ops.d_max: abstract_reduce,
            ops.d_min: abstract_reduce,
            ops.d_softmax: abstract_axis_preserving,
            ops.d_logsumexp: abstract_reduce,
            ops._matmul: abstract_matmul,
            ops.d_dot: ops.contraction_abstract_rule(ops.d_dot),
            ops.d_inner: ops.contraction_abstract_rule(ops.d_inner),
            ops.d_tensordot: ops.contraction_abstract_rule(ops.d_tensordot),
            ops.d_moveaxis: ops._transpose_lowering_abstract(ops.moveaxis_perm),
            ops.d_swapaxes: ops._transpose_lowering_abstract(ops.swapaxes_perm),
            ops.d_rollaxis: ops._transpose_lowering_abstract(ops.rollaxis_perm),
            ops.d_tril: ops._tri_lowering_abstract,
            ops.d_triu: ops._tri_lowering_abstract,
            ops.d_roll: ops._roll_abstract,
            ops.d_pad: ops._pad_abstract,
            ops.d_repeat: ops._repeat_abstract,
            ops.d_tile: ops._tile_abstract,
            ops.d_ravel: ops._reshape_lowering_abstract(ops.ravel_shape),
            ops.d_squeeze: ops._reshape_lowering_abstract(ops.squeeze_shape),
            ops.d_atleast_1d: ops._reshape_lowering_abstract(ops.atleast_1d_shape),
            ops.d_atleast_2d: ops._reshape_lowering_abstract(ops.atleast_2d_shape),
            ops.d_atleast_3d: ops._reshape_lowering_abstract(ops.atleast_3d_shape),
            ops.d_einsum: abstract_einsum,
            ops.d_cumsum: abstract_cumsum,
            ops.d_transpose: abstract_transpose,
            ops.d_reshape: abstract_reshape,
            ops.d_broadcast_to: abstract_broadcast_to,
            ops._scatter: abstract_scatter,
            # identity remat/spill markers (pycograd.remat): shape/dtype pass through
            ops._spill: abstract_unary,
            ops._recompute: abstract_unary,
            ops.d_expand_dims: abstract_expand_dims,
            ops.d_concatenate: abstract_concatenate,
            ops.d_stack: abstract_stack,
            ops.d_vstack: abstract_vstack,
            ops.d_hstack: abstract_hstack,
            ops.d_column_stack: abstract_column_stack,
            ops.d_dstack: abstract_dstack,
        }
    )
    table = {fn: abs_for[prim] for prim, fns in ops._RULES.items() for fn in fns}
    return table, abs_for


# ``_ABSTRACT`` is keyed by numpy callable (parity with ``ops._INTERCEPT``); ``_ABS_FOR``
# is keyed by *primitive* (incl. the operator primitives, which have no numpy callable),
# which is what ``AbstractTrace.process_primitive`` consults via ``bind``.
_ABSTRACT: dict[Prim, Prim]
_ABS_FOR: dict[Prim, Prim]
_ABSTRACT, _ABS_FOR = _build_abstract_table()


# ---------------------------------------------------------------------------
# AbstractTrace: the abstract-eval level on the trace-level stack.
#
# Reframes shape inference as one level of the interpreter stack
# (:mod:`pycograd.trace`), exactly like ``vmap``'s :class:`~pycograd.batching.BatchTrace`
# and ``jvp``'s :class:`~pycograd.forward.JVPTrace`. ``eval_shape`` pushes one of these
# via ``new_main`` and seeds the function's array/Var inputs as :class:`ShapedArray`
# tracers; every operator / ``np.*`` call routes through ``bind``, which selects this
# level and lets ``process_primitive`` run the per-primitive shape rule in ``_ABS_FOR``.
# Nothing about the rules changes -- only how they are dispatched.
# ---------------------------------------------------------------------------
class AbstractTrace(Trace):
    """One abstract-eval level: process each primitive by its shape rule, carrying only
    ``(shape, dtype)`` through :class:`ShapedArray` values (no data)."""

    def pure(self, val: Boxed) -> ShapedArray:
        return self._tag(_aval(cast(AbstractVal, val)))

    # A value from a lower level enters this level as its abstract shape/dtype.
    lift = pure

    def process_primitive(
        self, prim: Prim, args: Sequence[BindArg], params: dict[str, Any]
    ) -> Boxed:
        rule = _ABS_FOR.get(prim)
        if rule is None:  # pragma: no cover - every primitive in ``ops._RULES`` has one
            raise NotImplementedError(
                f"eval_shape: no shape rule for {getattr(prim, '__name__', prim)!r}"
            )
        return self._tag(rule(*args, **params))

    def _tag(self, val: Boxed) -> ShapedArray:
        """Re-tag a rule's result (a sentinel-level ``ShapedArray``) with this level, so it
        out-ranks lower levels in ``find_top_trace`` and keeps flowing through ``bind``.
        The rule's symbolic dims / ``prov`` are preserved unchanged."""
        if isinstance(val, ShapedArray):
            val._trace = self
            return val
        a = _aval(cast(AbstractVal, val))
        a._trace = self
        return a


# A sentinel level-0 ``AbstractTrace`` carried by any ``ShapedArray`` built outside a
# live ``eval_shape`` (a bare ``ShapedArray(...)`` or one a rule mints). Level 0 means it
# never out-ranks a real pushed level, so it leaves dispatch unaffected; it only exists so
# a standalone ``ShapedArray``'s operator dunders (which call the rules directly) work.
_SENTINEL_ABSTRACT_TRACE: Trace = AbstractTrace(MainTrace(0, AbstractTrace))


# ---------------------------------------------------------------------------
# Abstract inference driver.
# ---------------------------------------------------------------------------
def _abstract_leaf(leaf: Leaf, trace: Trace) -> "ShapedArray | Leaf":
    """Seed one argument leaf as a :class:`ShapedArray` tracer at ``trace``'s level.

    A ``ShapeDtypeStruct``/``Var``/``Param``/number becomes a same-shaped abstract value
    tagged with the live ``AbstractTrace`` (so it selects that level through ``bind``);
    anything else (a ``bool`` flag, ``None``, a string) passes through untouched."""
    if isinstance(leaf, _TieRef):
        raise ValueError(
            "eval_shape: tied[...] is only meaningful inside params(...); it reached "
            "shape inference unresolved"
        )
    if isinstance(leaf, ShapeDtypeStruct):
        return ShapedArray(leaf.shape, leaf.dtype, trace=trace)
    if isinstance(leaf, (Var, Param)) or _is_numeric(leaf):
        spec = _spec_of(leaf)
        return ShapedArray(spec.shape, spec.dtype, trace=trace)
    return leaf


def _eval_shape_abstract(fn: Callable[..., PyTree], args: tuple[PyTree, ...]) -> PyTree:
    # Deferred so importing this module (and ``pycograd``) stays light: the tracer
    # pulls in pyccolo, only needed when shape inference actually runs.
    from pycograd.tracer import _INSTRUMENTED, _make_runner

    runner = _INSTRUMENTED.get(fn)
    if runner is None:
        runner = _make_runner(fn)
        _INSTRUMENTED[fn] = runner

    # ``naming_scope`` restarts symbolic-dim names at ``n0`` per call (deterministic
    # reprs); ``constraint_scope`` records dim equalities so contractions can refine a
    # pinned symbol or report a contradiction across operands; ``new_main`` pushes the
    # abstract-eval level so the function's ops route through ``bind`` to its shape rules.
    with (
        _dims.naming_scope(),
        _constraints.constraint_scope() as env,
        new_main(AbstractTrace) as main,
    ):
        trace = AbstractTrace(main)
        call_args = [
            tree_unflatten(
                treedef,
                cast("list[Leaf]", [_abstract_leaf(leaf, trace) for leaf in leaves]),
            )
            for leaves, treedef in (tree_flatten(a) for a in args)
        ]
        out = runner(*call_args)
        specs = _to_specs(cast(PyTree, out))
    mapping = env.mapping()
    return _substitute_specs(specs, mapping) if mapping else specs


def eval_shape(
    fn: Callable[..., PyTree], *example_args: PyTree, method: str = "auto"
) -> PyTree:
    """The shape/dtype pytree ``fn`` would return, without computing real values.

    Each positional argument may be a real array, a ``Var``, a :class:`ShapeDtypeStruct`,
    or a pytree of those; numeric leaves are stood in with abstract values of the same
    shape and ``fn`` is run for its shapes alone. The result mirrors ``fn``'s output
    pytree with every leaf replaced by a :class:`ShapeDtypeStruct`.

    ``method``:

    * ``"abstract"`` (and ``"auto"``, the default) -- carry only ``(shape, dtype)``
      through the data-free :class:`ShapedArray` backend; allocates nothing and raises
      a clear error on data-dependent shapes.
    * ``"dummy"`` -- run plain numpy on zero-filled arrays. Simpler and the conformance
      oracle for the abstract path, but it allocates and silently mis-sizes
      data-dependent shapes.
    """
    if method in ("auto", "abstract"):
        return _eval_shape_abstract(fn, example_args)
    if method == "dummy":
        return _eval_shape_dummy(fn, example_args)
    raise ValueError(
        f"eval_shape: unknown method {method!r}; expected 'auto', 'abstract', or 'dummy'"
    )


def infer_shapes(fn: Callable[..., PyTree], *example_args: PyTree) -> PyTree:
    """Like :func:`eval_shape` but with plain shape tuples as leaves (no dtype)."""
    specs = eval_shape(fn, *example_args)
    leaves, treedef = tree_flatten(specs)
    return tree_unflatten(
        treedef,
        cast(
            "list[Leaf]",
            [None if s is None else cast(ShapeDtypeStruct, s).shape for s in leaves],
        ),
    )


def substitute(specs: PyTree, assignment: "dict[str, int]") -> PyTree:
    """Plug concrete sizes into a *polymorphic* shape result.

    ``specs`` is an :func:`eval_shape` output whose dims may be symbolic (declared via
    string input dims, e.g. ``ShapeDtypeStruct(("B", 768))``); ``assignment`` maps a
    symbol's name to an int. Every symbolic dim is re-evaluated, so e.g. ``f64[B,N]``
    with ``{"B": 8, "N": 256}`` becomes ``f64[8,256]`` and ``f64[2*B]`` becomes
    ``f64[16]``. Names not in ``assignment`` are left symbolic.
    """
    return _substitute_specs(specs, dict(assignment))


def bind(
    fn: Callable[..., PyTree],
    *example_args: PyTree,
    sizes: "dict[str, int] | None" = None,
) -> PyTree:
    """Solve a polymorphic signature for a concrete call: infer ``fn``'s (possibly
    symbolic) output shapes, then :func:`substitute` ``sizes`` into them. A size that
    contradicts a constraint ``fn`` imposes (e.g. a matmul forcing two dims equal)
    raises :class:`ShapeError`. With no symbolic inputs this is just :func:`eval_shape`.
    """
    specs = eval_shape(fn, *example_args)
    return substitute(specs, sizes) if sizes else specs


# ---------------------------------------------------------------------------
# Model summary.
# ---------------------------------------------------------------------------
def _named_leaves(tree: PyTree, prefix: str = "") -> Iterator[tuple[str, Leaf]]:
    """Walk a param pytree yielding ``(dotted_path, leaf)`` for each leaf, dict keys
    sorted so the order matches :func:`tree_flatten`."""
    if isinstance(tree, dict):
        for key in sorted(tree):
            yield from _named_leaves(tree[key], f"{prefix}{key}.")
    elif isinstance(tree, (list, tuple)):
        for i, child in enumerate(tree):
            yield from _named_leaves(child, f"{prefix}{i}.")
    else:
        yield (prefix.rstrip(".") or "<root>", tree)


@dataclass
class _ParamRow:
    name: str
    spec: ShapeDtypeStruct
    trainable: bool

    @property
    def count(self) -> int:
        return cast(int, self.spec.size)  # a parameter's shape is always concrete


@dataclass
class Summary:
    """The result of :func:`summary`: a per-parameter table plus the output shape.

    Prints as an aligned table; the fields (``rows``, ``output``, ``total`` /
    ``trainable``) are exposed so the summary can be asserted on in tests.
    """

    rows: list[_ParamRow]
    output: PyTree

    @property
    def total(self) -> int:
        return sum(r.count for r in self.rows)

    @property
    def trainable(self) -> int:
        return sum(r.count for r in self.rows if r.trainable)

    def __str__(self) -> str:
        name_w = max([len("parameter")] + [len(r.name) for r in self.rows])
        shape_w = max([len("shape")] + [len(repr(r.spec)) for r in self.rows])
        head = f"{'parameter':<{name_w}}  {'shape':<{shape_w}}  {'count':>10}"
        sep = "-" * len(head)
        lines = [head, sep]
        for r in self.rows:
            flag = "" if r.trainable else "  (frozen)"
            lines.append(
                f"{r.name:<{name_w}}  {repr(r.spec):<{shape_w}}  {r.count:>10,}{flag}"
            )
        lines.append(sep)
        lines.append(f"output: {self.output}")
        lines.append(f"trainable params: {self.trainable:,}  (total: {self.total:,})")
        return "\n".join(lines)


def summary(
    fn: Callable[..., PyTree],
    params: PyTree,
    *example_input_shapes: "ShapeDtypeStruct | tuple[int, ...]",
    print_fn: Callable[[str], None] | None = print,
) -> Summary:
    """Tabulate ``params``' per-tensor shapes and counts, plus the shape ``fn`` returns.

    ``params`` is the model's parameter pytree (the first argument ``fn`` takes).
    ``example_input_shapes`` are the *remaining* positional inputs to ``fn``, given as
    :class:`ShapeDtypeStruct`s or plain shape tuples -- omit them when ``fn`` closes
    over its data. The output shape is obtained via :func:`eval_shape`. Returns a
    :class:`Summary`; unless ``print_fn`` is ``None``, it is also printed.
    """
    rows = [
        _ParamRow(
            name,
            _spec_of(leaf),
            trainable=not (isinstance(leaf, Param) and not leaf.trainable),
        )
        for name, leaf in _named_leaves(params)
        if isinstance(leaf, (Var, Param)) or _is_numeric(leaf)
    ]
    specs = [
        s if isinstance(s, ShapeDtypeStruct) else ShapeDtypeStruct(s)  # type: ignore[arg-type]
        for s in example_input_shapes
    ]
    output = eval_shape(fn, params, *cast("list[PyTree]", specs))
    result = Summary(rows, output)
    if print_fn is not None:
        print_fn(str(result))
    return result

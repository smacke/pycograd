# -*- coding: utf-8 -*-
"""Shape inference: what shapes a net produces, without training it.

pycograd is *eager* -- every :class:`~pycograd.tensor.Var` holds a concrete array, so
a shape is only ever known *after* an op runs. This module recovers the shapes ahead
of time, for two everyday uses: a Keras-style :func:`summary` of a model's parameters
and output, and friendlier shape-mismatch errors (:class:`ShapeError`).

:func:`eval_shape` runs ``fn`` for its shapes one of two ways:

* ``method="abstract"`` (and ``"auto"``, the default) -- carry only ``(shape, dtype)``
  through :class:`ShapedArray` values via the abstract backend, with one shape rule per
  primitive. Allocates nothing (so it sizes a 100000x100000 matmul instantly) and
  raises a clear :class:`ShapeError` on *data-dependent* shapes (e.g. boolean-mask
  indexing, whose size depends on values) rather than guessing.
* ``method="dummy"`` -- run plain numpy on zero-filled **dummy** arrays and read the
  result's shape. Needs no tape and no tracer, but allocates and silently mis-sizes
  data-dependent shapes. It is the conformance oracle the abstract path is tested
  against.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence, cast

import numpy as np

from pycograd._typing import Array
from pycograd.params import Param, _TieRef
from pycograd.tensor import Var, _is_numeric
from pycograd.tree import Leaf, PyTree, tree_flatten, tree_unflatten

# A short numpy-dtype tag for ``__repr__`` ("f64", "i32", ...).
_DTYPE_TAG = {"float64": "f64", "float32": "f32", "int64": "i64", "int32": "i32"}


# ---------------------------------------------------------------------------
# Shape errors.
# ---------------------------------------------------------------------------
class ShapeError(ValueError):
    """A shape mismatch, annotated with the operation and the operand shapes.

    Raised in place of numpy's opaque message (e.g. matmul's "shapes (3,) and (4,2)
    not aligned") so the report names the op and what flowed into it. Subclasses
    ``ValueError`` so existing ``except ValueError`` handlers still catch it.
    """


def _shape_context(op_name: str, *shapes: "tuple[int, ...]") -> str:
    joined = " and ".join(str(tuple(s)) for s in shapes)
    return f"{op_name}: incompatible shapes {joined}"


# ---------------------------------------------------------------------------
# Abstract shape/dtype spec.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShapeDtypeStruct:
    """A shape + dtype with no data -- pass it to :func:`eval_shape`/:func:`summary`
    in place of a real array, the way ``jax.eval_shape`` takes ``ShapeDtypeStruct``.

    ``shape`` is a tuple of ints; ``dtype`` defaults to float64 (pycograd's working
    dtype). ``ndim``/``size`` mirror the numpy attributes so it reads like an array.
    """

    shape: tuple[int, ...]
    dtype: np.dtype = np.dtype(np.float64)

    def __post_init__(self) -> None:
        # Normalize so callers may pass a list, a bare int, or a numpy dtype string.
        shape = self.shape
        if isinstance(shape, int):
            shape = (shape,)
        object.__setattr__(self, "shape", tuple(int(d) for d in shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def __repr__(self) -> str:
        tag = _DTYPE_TAG.get(self.dtype.name, self.dtype.name)
        return f"{tag}[{','.join(map(str, self.shape))}]"


def _spec_of(x: object) -> ShapeDtypeStruct:
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
    return np.zeros(spec.shape, dtype=spec.dtype)


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


def _eval_shape_dummy(fn: Callable[..., object], args: tuple[PyTree, ...]) -> PyTree:
    call_args = [
        tree_unflatten(treedef, [_dummy_leaf(leaf) for leaf in leaves])
        for leaves, treedef in (tree_flatten(a) for a in args)
    ]
    return _to_specs(cast(PyTree, fn(*call_args)))


# ---------------------------------------------------------------------------
# Abstract values (Tier B): shape + dtype, no data.
# ---------------------------------------------------------------------------
class ShapedArray:
    """An abstract array -- a shape and dtype with no data behind them.

    It is the value the :class:`~pycograd.backends.abstract_backend.AbstractBackend`
    computes on: operators and numpy calls on a ``ShapedArray`` produce another
    ``ShapedArray`` via the shape rules below, never touching real numbers. Unlike the
    dummy-array path it allocates nothing, and it makes *data-dependent* shapes an
    explicit error rather than a wrong guess.

    The read-only surface a model performs on a tensor is mirrored from
    :class:`~pycograd.tensor.Var`: ``.shape``/``.ndim``/``.size``/``.dtype`` are concrete
    (so ``q.shape[-1] ** -0.5`` and ``rng.random(x.shape)`` work), only the *element
    data* is abstract. ``__array_ufunc__ = None`` makes numpy defer ufuncs/operators to
    our reflected methods, exactly as ``Var`` does.
    """

    __slots__ = ("shape", "dtype")
    __array_ufunc__ = None

    def __init__(self, shape: "tuple[int, ...]", dtype: object = np.float64) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        self.shape: tuple[int, ...] = tuple(int(d) for d in shape)
        self.dtype: np.dtype = np.dtype(cast(Any, dtype))

    # -- concrete metadata (mirrors Var) -------------------------------------
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def T(self) -> "ShapedArray":
        return abstract_transpose(self)

    # -- arithmetic: every elementwise op is a broadcast of operand shapes ----
    def __add__(self, o: object) -> "ShapedArray":
        return _ew_binary(self, o)

    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __add__
    __truediv__ = __rtruediv__ = __add__

    def __pow__(self, p: object) -> "ShapedArray":
        return _ew_binary(self, p)

    __rpow__ = __pow__

    def __neg__(self) -> "ShapedArray":
        return ShapedArray(self.shape, self.dtype)

    def __abs__(self) -> "ShapedArray":
        return ShapedArray(self.shape, self.dtype)

    def __matmul__(self, o: object) -> "ShapedArray":
        return abstract_matmul(self, o)

    def __rmatmul__(self, o: object) -> "ShapedArray":
        return abstract_matmul(o, self)

    # -- comparisons return a boolean abstract mask (broadcast shape) ---------
    def _cmp(self, o: object) -> "ShapedArray":
        return ShapedArray(_broadcast_shape("compare", self, o), np.dtype(bool))

    __lt__ = __le__ = __gt__ = __ge__ = _cmp

    def __getitem__(self, key: object) -> "ShapedArray":
        return abstract_getitem(self, key)

    # -- numpy-method surface (x.sum(...), x.reshape(...), ...) ---------------
    def __getattr__(self, name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        np_fn = getattr(np, name, None)
        prim = _ABSTRACT.get(np_fn) if callable(np_fn) else None
        if prim is None:
            raise AttributeError(name)
        return functools.partial(prim, self)

    def __repr__(self) -> str:
        return f"ShapedArray({_spec_of(self)!r})"


def _aval(x: object) -> ShapedArray:
    """Coerce an array / number / ``ShapeDtypeStruct`` / ``ShapedArray`` to an aval."""
    if isinstance(x, ShapedArray):
        return x
    if isinstance(x, ShapeDtypeStruct):
        return ShapedArray(x.shape, x.dtype)
    arr = np.asarray(x)
    return ShapedArray(arr.shape, arr.dtype)


def _broadcast_shape(op: str, *operands: object) -> "tuple[int, ...]":
    shapes = [_aval(o).shape for o in operands]
    try:
        return tuple(np.broadcast_shapes(*shapes))
    except ValueError as e:
        raise ShapeError(_shape_context(op, *shapes)) from e


def _result_dtype(*operands: object) -> np.dtype:
    return np.result_type(*[_aval(o).dtype for o in operands])


# ---------------------------------------------------------------------------
# Shape rules -- one per differentiable primitive in ``ops._RULES``.
# ---------------------------------------------------------------------------
def abstract_unary(x: object) -> ShapedArray:
    a = _aval(x)
    return ShapedArray(a.shape, a.dtype)


def _ew_binary(a: object, b: object) -> ShapedArray:
    return ShapedArray(_broadcast_shape("elementwise", a, b), _result_dtype(a, b))


def abstract_binary(a: object, b: object) -> ShapedArray:
    return _ew_binary(a, b)


def abstract_where(cond: object, a: object, b: object) -> ShapedArray:
    return ShapedArray(_broadcast_shape("where", cond, a, b), _result_dtype(a, b))


def abstract_clip(x: object, a_min: object = None, a_max: object = None) -> ShapedArray:
    bounds = [b for b in (x, a_min, a_max) if b is not None]
    return ShapedArray(_broadcast_shape("clip", *bounds), _aval(x).dtype)


def abstract_matmul(a: object, b: object) -> ShapedArray:
    av, bv = _aval(a), _aval(b)
    sa, sb = av.shape, bv.shape
    dtype = _result_dtype(a, b)
    if len(sa) == 0 or len(sb) == 0:
        raise ShapeError(_shape_context("matmul (needs >=1-D operands)", sa, sb))
    if len(sa) == 1 and len(sb) == 1:
        if sa[0] != sb[0]:
            raise ShapeError(_shape_context("matmul", sa, sb))
        return ShapedArray((), dtype)
    if len(sa) == 1:  # (k,) @ (..., k, n) -> (..., n)
        if sa[0] != sb[-2]:
            raise ShapeError(_shape_context("matmul", sa, sb))
        return ShapedArray(tuple(sb[:-2]) + (sb[-1],), dtype)
    if len(sb) == 1:  # (..., m, k) @ (k,) -> (..., m)
        if sa[-1] != sb[0]:
            raise ShapeError(_shape_context("matmul", sa, sb))
        return ShapedArray(tuple(sa[:-1]), dtype)
    if sa[-1] != sb[-2]:  # (..., m, k) @ (..., k, n) -> (..., m, n)
        raise ShapeError(_shape_context("matmul", sa, sb))
    try:
        batch = tuple(np.broadcast_shapes(sa[:-2], sb[:-2]))
    except ValueError as e:
        raise ShapeError(_shape_context("matmul (batch dims)", sa, sb)) from e
    return ShapedArray(batch + (sa[-2], sb[-1]), dtype)


def abstract_reduce(
    x: object, axis: object = None, keepdims: bool = False, **_: object
) -> ShapedArray:
    a = _aval(x)
    shp = a.shape
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


def abstract_transpose(x: object, axes: "tuple[int, ...] | None" = None) -> ShapedArray:
    a = _aval(x)
    if axes is None:
        return ShapedArray(tuple(reversed(a.shape)), a.dtype)
    return ShapedArray(tuple(a.shape[ax] for ax in axes), a.dtype)


def abstract_reshape(x: object, *shape: object) -> ShapedArray:
    a = _aval(x)
    newshape = shape[0] if len(shape) == 1 else shape
    if isinstance(newshape, int):
        newshape = (newshape,)
    newshape = tuple(int(d) for d in cast(Sequence, newshape))
    size = a.size
    if -1 in newshape:
        known = 1
        for d in newshape:
            if d != -1:
                known *= d
        if known == 0 or size % known != 0:
            raise ShapeError(f"reshape: cannot reshape size {size} into {newshape}")
        newshape = tuple(size // known if d == -1 else d for d in newshape)
    if int(np.prod(newshape)) != size:
        raise ShapeError(f"reshape: cannot reshape size {size} into {newshape}")
    return ShapedArray(newshape, a.dtype)


def abstract_expand_dims(x: object, axis: int) -> ShapedArray:
    a = _aval(x)
    pos = axis if axis >= 0 else axis + a.ndim + 1
    shp = list(a.shape)
    shp.insert(pos, 1)
    return ShapedArray(tuple(shp), a.dtype)


def abstract_concatenate(seq: "Sequence[object]", axis: int = 0) -> ShapedArray:
    avals = [_aval(s) for s in seq]
    ndim = avals[0].ndim
    ax = axis % ndim
    out = list(avals[0].shape)
    for a in avals[1:]:
        if a.ndim != ndim or any(d != out[i] for i, d in enumerate(a.shape) if i != ax):
            raise ShapeError(
                _shape_context("concatenate", *(a.shape for a in avals))
                + f" along axis {axis}"
            )
        out[ax] += a.shape[ax]
    return ShapedArray(tuple(out), np.result_type(*[a.dtype for a in avals]))


def abstract_stack(seq: "Sequence[object]", axis: int = 0) -> ShapedArray:
    return abstract_concatenate([abstract_expand_dims(s, axis) for s in seq], axis=axis)


def _atleast_2d_row(x: object) -> ShapedArray:
    a = _aval(x)
    if a.ndim == 0:
        return abstract_reshape(a, (1, 1))
    if a.ndim == 1:
        return abstract_reshape(a, (1, a.shape[0]))
    return a


def abstract_vstack(seq: "Sequence[object]") -> ShapedArray:
    return abstract_concatenate([_atleast_2d_row(s) for s in seq], axis=0)


def abstract_hstack(seq: "Sequence[object]") -> ShapedArray:
    avals = [_aval(s) for s in seq]
    axis = 0 if all(a.ndim <= 1 for a in avals) else 1
    return abstract_concatenate(avals, axis=axis)


def abstract_column_stack(seq: "Sequence[object]") -> ShapedArray:
    parts: list[object] = []
    for s in seq:
        a = _aval(s)
        parts.append(abstract_reshape(a, (a.shape[0], 1)) if a.ndim == 1 else a)
    return abstract_concatenate(parts, axis=1)


def _atleast_3d_depth(x: object) -> ShapedArray:
    a = _aval(x)
    if a.ndim == 0:
        return abstract_reshape(a, (1, 1, 1))
    if a.ndim == 1:
        return abstract_reshape(a, (1, a.shape[0], 1))
    if a.ndim == 2:
        return abstract_reshape(a, a.shape + (1,))
    return a


def abstract_dstack(seq: "Sequence[object]") -> ShapedArray:
    return abstract_concatenate([_atleast_3d_depth(s) for s in seq], axis=2)


def abstract_getitem(x: object, key: object) -> ShapedArray:
    """Shape of ``x[key]`` for *static* indexing (int/slice/ellipsis/newaxis).

    A boolean mask or integer-array key makes the result size depend on data values,
    which shape inference cannot determine -- raise a clear error pointing at a hint
    (handled fully in a later phase). Static keys are resolved on a zero-stride view,
    which costs no allocation.
    """
    a = _aval(x)
    keys = key if isinstance(key, tuple) else (key,)
    for k in keys:
        # An array-valued key (a boolean mask or an integer index array) makes the
        # output size depend on the data, not just the shape -- inference can't know it.
        if isinstance(k, (ShapedArray, np.ndarray, list)):
            kind = "boolean mask" if _is_bool_key(k) else "array index"
            raise ShapeError(
                f"indexing: result shape depends on data values ({kind}); "
                "shape inference cannot determine it -- use a static int/slice index"
            )
    view = np.broadcast_to(np.zeros((), a.dtype), a.shape)
    return ShapedArray(view[cast(Any, key)].shape, a.dtype)


def _is_bool_key(k: object) -> bool:
    return isinstance(k, (np.ndarray, ShapedArray)) and k.dtype == np.dtype(bool)


# The shape rules above form the ``abstract_eval(primitive, *avals) -> aval`` registry
# that a future graph-capture tracer (cf. the ROADMAP's Phase 3/4 trace-and-compile and
# ``vmap``) would call to size each node without data. The seam to get there: a tracer
# value subclasses ``ShapedArray`` with an extra graph-node field and reuses these exact
# rules unchanged -- shape inference lives on the abstract value, and tracing adds only
# graph recording on top. Data-dependent dims (today an error in ``abstract_getitem``)
# would carry a symbolic placeholder instead; that symbolic-dimension algebra is the
# natural next increment and deliberately not built yet (no model here exercises it).


# ---------------------------------------------------------------------------
# The abstract interception table, keyed identically to ``ops._INTERCEPT``.
# ---------------------------------------------------------------------------
def _build_abstract_table() -> "dict[object, Callable[..., object]]":
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
        ops.d_sinh,
        ops.d_cosh,
        ops.d_arctan,
        ops.d_log1p,
        ops.d_expm1,
        ops.d_abs,
        ops.d_square,
        ops.d_reciprocal,
    )
    abs_for: dict[object, Callable[..., object]] = {p: abstract_unary for p in unary}
    abs_for.update(
        {
            ops.d_maximum: abstract_binary,
            ops.d_minimum: abstract_binary,
            ops.d_where: abstract_where,
            ops.d_clip: abstract_clip,
            ops.d_sum: abstract_reduce,
            ops.d_mean: abstract_reduce,
            ops.d_var: abstract_reduce,
            ops.d_std: abstract_reduce,
            ops.d_max: abstract_reduce,
            ops.d_min: abstract_reduce,
            ops._matmul: abstract_matmul,
            ops.d_transpose: abstract_transpose,
            ops.d_reshape: abstract_reshape,
            ops.d_expand_dims: abstract_expand_dims,
            ops.d_concatenate: abstract_concatenate,
            ops.d_stack: abstract_stack,
            ops.d_vstack: abstract_vstack,
            ops.d_hstack: abstract_hstack,
            ops.d_column_stack: abstract_column_stack,
            ops.d_dstack: abstract_dstack,
        }
    )
    return {fn: abs_for[prim] for prim, fns in ops._RULES.items() for fn in fns}


_ABSTRACT: "dict[object, Callable[..., object]]" = _build_abstract_table()


# ---------------------------------------------------------------------------
# Abstract inference driver.
# ---------------------------------------------------------------------------
def _abstract_leaf(leaf: Leaf) -> object:
    if isinstance(leaf, _TieRef):
        raise ValueError(
            "eval_shape: tied[...] is only meaningful inside params(...); it reached "
            "shape inference unresolved"
        )
    if isinstance(leaf, ShapeDtypeStruct):
        return ShapedArray(leaf.shape, leaf.dtype)
    if isinstance(leaf, (Var, Param)) or _is_numeric(leaf):
        spec = _spec_of(leaf)
        return ShapedArray(spec.shape, spec.dtype)
    return leaf


def _eval_shape_abstract(fn: Callable[..., object], args: tuple[PyTree, ...]) -> PyTree:
    # Deferred so importing this module (and ``pycograd``) stays light: the tracer
    # pulls in pyccolo, and the abstract backend is only needed when actually used.
    from pycograd.backends import activate, get_backend
    from pycograd.tracer import _INSTRUMENTED, _make_runner

    runner = _INSTRUMENTED.get(fn)
    if runner is None:
        runner = _make_runner(fn)
        _INSTRUMENTED[fn] = runner

    call_args = [
        tree_unflatten(
            treedef, cast("list[Leaf]", [_abstract_leaf(leaf) for leaf in leaves])
        )
        for leaves, treedef in (tree_flatten(a) for a in args)
    ]
    with activate(get_backend("abstract")):
        out = runner(*call_args)
    return _to_specs(cast(PyTree, out))


def eval_shape(
    fn: Callable[..., object], *example_args: PyTree, method: str = "auto"
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


def infer_shapes(fn: Callable[..., object], *example_args: PyTree) -> PyTree:
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
        return self.spec.size


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
    fn: Callable[..., object],
    params: PyTree,
    *example_input_shapes: object,
    print_fn: "Callable[[str], None] | None" = print,
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

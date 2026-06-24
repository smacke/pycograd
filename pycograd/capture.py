# -*- coding: utf-8 -*-
"""Graph-capture IR: record the primitives a function executes into a flat SSA
graph, so optimization passes (see :mod:`pycograd.passes`) can rewrite it.

This is a :class:`~pycograd.trace.Trace` level exactly like ``eval_shape``'s
``AbstractTrace`` (``shapes.py``) -- in fact it *reuses* the abstract shape rules
(``_ABS_FOR``) to size each node without data, as the ``shapes.py`` roadmap note
anticipated ("``ShapedArray`` already is the level's tracer; the remaining
increment is graph recording on top"). Every op flows through ``bind`` ->
``find_top_trace`` -> :meth:`GraphTrace.process_primitive`, which records a node and
returns a :class:`GraphTracer` carrying the output's abstract value.

``capture(f, *args)`` returns a :class:`Graph`; ``eval_graph(graph, *inputs)``
replays it through ``bind`` (so it computes -- and, under ``value_and_grad``,
differentiates -- exactly as the original).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, cast

import numpy as np

from pycograd._typing import BindArg, Boxed, Prim
from pycograd.params import Param, Weight
from pycograd.shapes import _ABS_FOR, ShapedArray, ShapeDtypeStruct, _aval
from pycograd.tensor import Var, _value
from pycograd.trace import Trace, Tracer, bind, new_main
from pycograd.tree import Leaf, PyTree, tree_flatten, tree_unflatten

if TYPE_CHECKING:
    import graphviz

    from pycograd.cost import CostModel, GraphCost
    from pycograd.remat import RematPlan


# ---------------------------------------------------------------------------
# IR value references and nodes.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Ref:
    """An edge: this operand is the output of node ``id``."""

    id: int


@dataclass(frozen=True)
class Const:
    """A constant operand inlined on a node (a host-side array / scalar / string /
    slice / index tuple -- anything that was not a traced value). ``value`` is
    genuinely arbitrary (an op's static non-tracer argument), hence ``Any``."""

    value: Any


# An ``arg_spec`` element: a ``Ref``/``Const`` leaf, or a list/tuple of them (the
# structural operands -- ``concatenate``'s sequence, etc.). Kept as plain nested
# Python so the interpreter can rebuild the exact call.
ArgSpec = Any

# Sentinel "primitives" marking the two non-op node kinds.
_INPUT = cast(Prim, "input")
_CONST = cast(Prim, "const")


@dataclass
class Node:
    """One SSA node: ``prim`` applied to ``args`` (a tuple of :data:`ArgSpec`) with
    static ``params``, producing a value of abstract type ``aval``. ``prim`` is
    :data:`_INPUT` for a graph input and :data:`_CONST` for a captured constant
    (whose value is in ``params['value']``)."""

    id: int
    prim: Prim
    args: tuple
    params: dict
    aval: ShapeDtypeStruct


@dataclass
class Graph:
    """A captured computation: ``nodes`` in SSA order, ``inputs``/``outputs`` as node
    ids, and ``out_treedef`` to rebuild the original output pytree."""

    nodes: list[Node]
    inputs: list[int]
    outputs: list[int]
    out_treedef: Any
    in_avals: list[ShapeDtypeStruct] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        n = sum(1 for nd in self.nodes if nd.prim not in (_INPUT, _CONST))
        return f"Graph({len(self.inputs)} in, {n} ops, {len(self.outputs)} out)"

    def __str__(self) -> str:  # pragma: no cover - debug aid
        return pretty(self)

    def __call__(self, *inputs: PyTree) -> PyTree:
        """Run the captured computation on concrete ``inputs`` -- ``graph(x)`` is the
        inference path. Delegates to :func:`eval_graph`, so the replay dispatches through
        ``bind`` (computing on the active backend) and stays differentiable under
        ``value_and_grad``/``vmap``/``jvp`` just like the original function."""
        return eval_graph(self, *inputs)

    def pretty(self) -> str:
        """A jaxpr-style text listing of the graph (nodes in SSA order), e.g.::

            graph(%0:f64[4,3], %1:f64[3,2]) {
              %2 = matmul %0 %1 -> f64[4,2]
              %3 = tanh %2 -> f64[4,2]
              ...
              outputs: %7
            }

        ``print(graph)`` shows this; ``repr(graph)`` stays the one-line summary."""
        return pretty(self)

    def to_dot(self) -> str:
        """Graphviz DOT source for the graph. Render it with e.g.
        ``open("g.dot","w").write(graph.to_dot())`` then ``dot -Tpng g.dot -o g.png``,
        or :meth:`render` for inline display when the ``graphviz`` package is installed.
        """
        return to_dot(self)

    def cost(self, model: "CostModel | None" = None) -> "GraphCost":
        """A static CPU / memory / disk :class:`~pycograd.cost.GraphCost` estimate for
        this graph (no execution). ``model`` is an optional
        :class:`~pycograd.cost.CostModel`; the default uses conservative NVMe-class
        hardware constants. ``print(graph.cost())`` shows a per-node listing + totals.
        """
        from pycograd.cost import DEFAULT_COST_MODEL, cost_report

        return cost_report(self, DEFAULT_COST_MODEL if model is None else model)

    def plan_remat(
        self, budget: int, model: "CostModel | None" = None, **kwargs: Any
    ) -> "RematPlan":
        """Plan keep/spill/recompute for this (forward+backward) graph so its peak resident
        memory fits ``budget`` bytes -- a :class:`~pycograd.remat.RematPlan`. ``model`` is an
        optional :class:`~pycograd.cost.CostModel`; ``**kwargs`` (``exact``/``iters``) pass
        through to :func:`~pycograd.remat.plan_remat`."""
        from pycograd.cost import DEFAULT_COST_MODEL
        from pycograd.remat import plan_remat

        return plan_remat(
            self, budget, DEFAULT_COST_MODEL if model is None else model, **kwargs
        )

    def eval_scheduled(
        self, *inputs: PyTree, store_dir: "str | None" = None
    ) -> "tuple[PyTree, int]":
        """Evaluate this graph (typically after :func:`~pycograd.remat.apply_remat_plan`)
        with the memory-managed interpreter, returning ``(outputs, peak_resident_bytes)``.
        See :func:`~pycograd.remat.eval_scheduled`."""
        from pycograd.remat import eval_scheduled

        return eval_scheduled(self, *inputs, store_dir=store_dir)

    def render(self) -> "graphviz.Source":  # pragma: no cover - optional dependency
        """A ``graphviz.Source`` (renders inline in Jupyter). Requires the optional
        ``graphviz`` Python package; otherwise use :meth:`to_dot`."""
        try:
            import graphviz
        except ImportError as e:  # keep graphviz an optional, not a hard, dependency
            raise ImportError(
                "Graph.render() needs the 'graphviz' package (pip install graphviz); "
                "or use Graph.to_dot() to get DOT source and render it yourself."
            ) from e
        return graphviz.Source(self.to_dot())

    def _repr_mimebundle_(
        self, include: Any = None, exclude: Any = None
    ) -> "dict[str, Any]":  # pragma: no cover - notebook display
        """Rich display for Jupyter: a bare ``graph`` in a cell auto-renders to SVG when
        ``graphviz`` (the package *and* the ``dot`` binary) is available, and always
        offers the pretty text listing -- so it degrades gracefully to the listing when
        graphviz is not installed. (``repr`` stays the one-line summary regardless.)"""
        bundle: dict[str, Any] = {"text/plain": self.pretty()}
        try:
            import graphviz

            svg = graphviz.Source(self.to_dot()).pipe(format="svg").decode("utf-8")
            bundle["image/svg+xml"] = svg
        except Exception:
            pass  # no graphviz package / no ``dot`` on PATH -> text listing only
        return bundle


# ---------------------------------------------------------------------------
# Human-readable rendering of a captured graph (text listing + Graphviz DOT). Pure
# string formatting, no dependencies -- the heavy bits stay optional (see Graph.render).
# ---------------------------------------------------------------------------
def _dtype_str(dt: Any) -> str:
    dt = np.dtype(dt)
    return "bool" if dt.kind == "b" else f"{dt.kind}{dt.itemsize * 8}"


def _aval_str(av: ShapeDtypeStruct) -> str:
    return f"{_dtype_str(av.dtype)}[{','.join(str(d) for d in av.shape)}]"


def _prim_name(prim: Prim) -> str:
    if prim is _INPUT:
        return "input"
    if prim is _CONST:
        return "const"
    name = getattr(prim, "__name__", str(prim))
    return name[2:] if name.startswith("d_") else name.lstrip("_")


def _truncate(s: str, n: int = 32) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _const_str(v: Any) -> str:
    try:
        arr = np.asarray(v)
    except Exception:  # pragma: no cover - genuinely opaque operand
        return _truncate(repr(v))
    if arr.ndim == 0 and arr.dtype.kind in "fiub":
        return _truncate(repr(arr.item()))  # a scalar -> its value
    if arr.dtype != object and arr.ndim > 0:
        return f"{_dtype_str(arr.dtype)}[{','.join(map(str, arr.shape))}]"  # array -> shape
    return _truncate(repr(v))  # string (einsum subscripts), slice, index tuple, ...


def _arg_str(spec: ArgSpec) -> str:
    if isinstance(spec, Ref):
        return f"%{spec.id}"
    if isinstance(spec, Const):
        return _const_str(spec.value)
    if isinstance(spec, (list, tuple)):
        return "[" + ", ".join(_arg_str(s) for s in spec) + "]"
    return _truncate(repr(spec))


def _params_str(params: dict) -> str:
    items = [(k, v) for k, v in params.items() if k != "value"]
    if not items:
        return ""
    return " {" + ", ".join(f"{k}={_truncate(repr(v))}" for k, v in items) + "}"


def _iter_refs(spec: ArgSpec) -> "Iterator[int]":
    if isinstance(spec, Ref):
        yield spec.id
    elif isinstance(spec, (list, tuple)):
        for s in spec:
            yield from _iter_refs(s)


def pretty(graph: "Graph") -> str:
    """A jaxpr-style text listing of ``graph`` (see :meth:`Graph.pretty`)."""
    header = ", ".join(f"%{i}:{_aval_str(graph.nodes[i].aval)}" for i in graph.inputs)
    lines = [f"graph({header}) {{"]
    for nd in graph.nodes:
        if nd.prim is _INPUT:
            continue  # shown in the header
        if nd.prim is _CONST:
            rhs = f"const {_const_str(nd.params['value'])}"
        else:
            args = " ".join(_arg_str(s) for s in nd.args)
            rhs = f"{_prim_name(nd.prim)} {args}{_params_str(nd.params)}".rstrip()
        lines.append(f"  %{nd.id} = {rhs} -> {_aval_str(nd.aval)}")
    lines.append("  outputs: " + ", ".join(f"%{o}" for o in graph.outputs))
    lines.append("}")
    return "\n".join(lines)


def to_dot(graph: "Graph") -> str:
    """Graphviz DOT source for ``graph`` (see :meth:`Graph.to_dot`)."""
    outs = set(graph.outputs)
    lines = [
        "digraph G {",
        "  rankdir=TB;",
        '  node [shape=box, fontname="monospace"];',
    ]
    for nd in graph.nodes:
        if nd.prim is _INPUT:
            attrs = f'label="%{nd.id} in\\n{_aval_str(nd.aval)}", shape=ellipse, style=filled, fillcolor=lightblue'
        elif nd.prim is _CONST:
            attrs = f'label="%{nd.id} const\\n{_const_str(nd.params["value"])}", shape=ellipse, style=filled, fillcolor=lightyellow'
        else:
            attrs = f'label="%{nd.id} {_prim_name(nd.prim)}\\n{_aval_str(nd.aval)}"'
        if nd.id in outs:
            attrs += ", peripheries=2"  # double border marks a graph output
        lines.append(f"  {nd.id} [{attrs}];")
        for src in {r for s in nd.args for r in _iter_refs(s)}:
            lines.append(f"  {src} -> {nd.id};")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The capture trace level.
# ---------------------------------------------------------------------------
class _Builder:
    """Shared recording state, carried on the level's ``MainTrace.global_data`` so the
    per-call :class:`GraphTrace` instances ``find_top_trace`` mints all append here."""

    def __init__(self) -> None:
        self.nodes: list[Node] = []

    def add(self, prim: Prim, args: tuple, params: dict, aval: ShapedArray) -> int:
        nid = len(self.nodes)
        self.nodes.append(
            Node(nid, prim, args, params, ShapeDtypeStruct(aval.shape, aval.dtype))
        )
        return nid


class GraphTracer(Tracer):
    """A value flowing through capture: its abstract value (``ShapedArray``, so
    ``.shape``/``.dtype`` queries inside the traced fn work) plus the id of the node
    that produced it."""

    __slots__ = ("_trace", "id", "_av")
    __array_ufunc__ = None  # keep numpy from treating this as an array in a ufunc

    def __init__(self, trace: "GraphTrace", node_id: int, aval: ShapedArray) -> None:
        self._trace = trace
        self.id = node_id
        self._av = aval

    @property
    def aval(self) -> ShapeDtypeStruct:
        return ShapeDtypeStruct(self._av.shape, self._av.dtype)

    @property
    def dtype(self) -> np.dtype:
        return self._av.dtype

    @property
    def size(self) -> int:
        return int(np.prod(cast(Any, self.shape), dtype=np.int64))

    # The surface instrumentation does *not* rewrite to ``bind`` (subscript, ``.T``,
    # and ``x.sum(...)``-style numpy methods) must route through ``bind`` itself, so
    # capture records them -- mirroring ``BatchTracer`` / ``Var``.
    @property
    def T(self) -> Boxed:
        from pycograd import ops

        return bind(ops.d_transpose, self)

    def __getitem__(self, key: object) -> Boxed:
        from pycograd import ops

        return bind(ops.d_getitem, self, key)

    def __getattr__(self, name: str) -> "Callable[..., Boxed]":
        if name.startswith("__"):
            raise AttributeError(name)
        from pycograd import ops

        np_fn = getattr(np, name, None)
        prim = ops._INTERCEPT.get(np_fn) if callable(np_fn) else None
        if prim is None:
            raise AttributeError(name)

        def _method(*a: BindArg, **k: Any) -> Boxed:
            return bind(prim, self, *a, **k)

        return _method


def _arg_aval(a: BindArg) -> object:
    """The abstract value an operand contributes to a shape rule: a tracer's stored
    ``ShapedArray``, the same map over a structural list, else the raw constant (the
    rule's own ``_aval`` sizes it)."""
    if isinstance(a, GraphTracer):
        return a._av
    if isinstance(a, (list, tuple)):
        return type(a)(_arg_aval(e) for e in a)
    return a


def _snapshot(a: BindArg) -> Any:
    """The concrete value to store for a captured constant. Resolves an ambient-parameter
    proxy (:class:`~pycograd.params.Weight`) or :class:`~pycograd.params.Param` to its live
    array, and a base ``Var`` to its value -- so the graph is a *snapshot* decoupled from
    any live ``with weights:`` binding (it stays valid, and renders, after the block exits).
    """
    if isinstance(a, Param):
        a = a.value
    elif isinstance(a, Weight):
        a = _value(a)
    return a.value if isinstance(a, Var) else a


class GraphTrace(Trace):
    """Records each primitive as a :class:`Node` and sizes it with the abstract rules,
    mirroring ``AbstractTrace`` (the recording is the only addition)."""

    @property
    def _builder(self) -> _Builder:
        return cast(_Builder, self.main.global_data)

    def _spec(self, a: BindArg) -> ArgSpec:
        if isinstance(a, GraphTracer):
            return Ref(a.id)
        if isinstance(a, (list, tuple)):
            return type(a)(self._spec(e) for e in a)
        return Const(_snapshot(a))

    def pure(self, val: Boxed) -> GraphTracer:
        # A constant raised into the level -- record a const node. (Rarely hit: most
        # constants reach ``process_primitive`` as raw args and inline as ``Const``.)
        value = _snapshot(val)
        av = _aval(cast(Any, value))
        nid = self._builder.add(_CONST, (), {"value": value}, av)
        return GraphTracer(self, nid, av)

    lift = pure

    def process_primitive(
        self, prim: Prim, args: Sequence[BindArg], params: dict[str, Any]
    ) -> Boxed:
        rule = _ABS_FOR.get(prim)
        if rule is None:  # pragma: no cover - capture covers what eval_shape covers
            raise NotImplementedError(
                f"capture: no shape rule for {getattr(prim, '__name__', prim)!r}"
            )
        out_aval = cast(ShapedArray, rule(*[_arg_aval(a) for a in args], **params))
        nid = self._builder.add(
            prim, tuple(self._spec(a) for a in args), params, out_aval
        )
        return GraphTracer(self, nid, out_aval)

    def add_input(self, aval: ShapedArray) -> GraphTracer:
        nid = self._builder.add(_INPUT, (), {}, aval)
        return GraphTracer(self, nid, aval)

    def output_id(self, leaf: Boxed) -> int:
        """The node id a returned leaf maps to: a tracer's own id, else a fresh const
        node (a model that returns a constant / a base-level ``Var``)."""
        if isinstance(leaf, GraphTracer):
            return leaf.id
        value = _snapshot(leaf)
        return self._builder.add(_CONST, (), {"value": value}, _aval(cast(Any, value)))


# ---------------------------------------------------------------------------
# Entry point + interpreter.
# ---------------------------------------------------------------------------
def _input_leaf(leaf: object, trace: GraphTrace) -> object:
    """Seed an input leaf as a :class:`GraphTracer` input node; pass through a
    non-numeric leaf (a bool flag / None / string), as ``eval_shape`` does."""
    if isinstance(leaf, (Var,)) or _is_numeric(leaf):
        return trace.add_input(_aval(cast(Any, leaf)))
    return leaf


def _is_numeric(x: object) -> bool:
    return isinstance(
        x, (int, float, complex, np.number, np.ndarray)
    ) and not isinstance(x, bool)


def capture(f: Callable[..., PyTree], *args: PyTree) -> Graph:
    """Trace ``f(*args)`` and return the recorded :class:`Graph`. Numeric input leaves
    become graph inputs; everything else is recorded as it executes. Has the same
    limitations as ``eval_shape``: no data-dependent Python control flow, and apply
    at the outermost level (not inside a live ``vmap``/``jvp``).

    Captures the *forward* only. ``capture(value_and_grad(f))`` does not compose:
    pycograd's reverse pass is not ``bind``-expressed at the base level (the
    base-vs-higher-order split in ``ops.py``), so a trace cannot record it. To get a
    fused *backward*, optimize the captured forward (fusing e.g. ``tanh*sigmoid`` to
    ``d_gated_act``) and differentiate the result through :func:`eval_graph` -- the
    fused primitive's VJP then runs (one backward op, not three)."""
    from pycograd.tracer import _INSTRUMENTED, _make_runner

    runner = _INSTRUMENTED.get(f)
    if runner is None:
        runner = _make_runner(f)
        _INSTRUMENTED[f] = runner

    builder = _Builder()
    with new_main(GraphTrace, builder) as main:
        trace = GraphTrace(main)
        call_args = []
        for a in args:
            leaves, treedef = tree_flatten(a)
            wrapped = cast("list[Leaf]", [_input_leaf(leaf, trace) for leaf in leaves])
            call_args.append(tree_unflatten(treedef, wrapped))
        out = runner(*call_args)
        out_leaves, out_treedef = tree_flatten(cast(PyTree, out))
        outputs = [trace.output_id(cast(Boxed, leaf)) for leaf in out_leaves]

    inputs = [nd.id for nd in builder.nodes if nd.prim is _INPUT]
    in_avals = [builder.nodes[i].aval for i in inputs]
    return Graph(builder.nodes, inputs, outputs, out_treedef, in_avals)


def _rebuild(spec: ArgSpec, env: dict[int, object]) -> object:
    if isinstance(spec, Ref):
        return env[spec.id]
    if isinstance(spec, Const):
        return spec.value
    if isinstance(spec, (list, tuple)):
        return type(spec)(_rebuild(e, env) for e in spec)
    return spec  # pragma: no cover - arg_spec leaves are only Ref/Const/sequences


def eval_graph(graph: Graph, *inputs: PyTree) -> PyTree:
    """Replay ``graph`` on concrete ``inputs`` (flattened to match ``graph.inputs``),
    rebuilding each node's call and dispatching through ``bind`` -- so it computes on
    the active backend and differentiates under ``value_and_grad`` just like the
    original function."""
    in_leaves = [leaf for a in inputs for leaf in tree_flatten(a)[0]]
    if len(in_leaves) != len(graph.inputs):
        raise ValueError(
            f"eval_graph: expected {len(graph.inputs)} input leaves, got {len(in_leaves)}"
        )
    env: dict[int, object] = dict(zip(graph.inputs, in_leaves))
    for node in graph.nodes:
        if node.prim is _INPUT:
            continue
        if node.prim is _CONST:
            env[node.id] = node.params["value"]
            continue
        args = tuple(_rebuild(s, env) for s in node.args)
        env[node.id] = bind(node.prim, *args, **node.params)
    out_leaves = cast("list[Leaf]", [env[o] for o in graph.outputs])
    return tree_unflatten(graph.out_treedef, out_leaves)

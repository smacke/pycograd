# -*- coding: utf-8 -*-
"""Optimization passes over the capture IR (:mod:`pycograd.capture`).

Each pass is ``Graph -> Graph`` and preserves semantics (``eval_graph`` of the
result equals ``eval_graph`` of the input). ``optimize`` runs a list to a fixpoint.
Node ids are stable identifiers (an arg ``Ref`` names a producer by id, not by list
position), so a pass can drop or merge nodes by editing the list and remapping
``Ref``s without renumbering.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Iterator, cast

import numpy as np

from pycograd._typing import Operand
from pycograd.capture import _CONST, _INPUT, Const, Graph, Node, Ref
from pycograd.tensor import Var, _value
from pycograd.trace import bind

Pass = Callable[[Graph], Graph]


# ---------------------------------------------------------------------------
# arg_spec helpers (an arg_spec is a Ref/Const leaf or a nested list/tuple).
# ---------------------------------------------------------------------------
def _spec_refs(spec: Any) -> Iterator[int]:
    if isinstance(spec, Ref):
        yield spec.id
    elif isinstance(spec, (list, tuple)):
        for e in spec:
            yield from _spec_refs(e)


def _map_spec(spec: Any, remap: dict[int, int]) -> Any:
    if isinstance(spec, Ref):
        return Ref(remap.get(spec.id, spec.id))
    if isinstance(spec, (list, tuple)):
        return type(spec)(_map_spec(e, remap) for e in spec)
    return spec


def _freeze(x: Any) -> Any:
    """A hashable canonical form, or raise ``TypeError`` (e.g. an array / slice in a
    ``getitem`` key) so the caller can skip that node conservatively."""
    if isinstance(x, Ref):
        return ("ref", x.id)
    if isinstance(x, Const):
        return ("const", _freeze(x.value))
    if isinstance(x, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_freeze(e) for e in x)
    if isinstance(x, np.ndarray):
        raise TypeError("array operand is not a CSE key")
    hash(x)  # slices, etc. raise here -> the node is skipped
    return x


# ---------------------------------------------------------------------------
# Passes.
# ---------------------------------------------------------------------------
def dce(graph: Graph) -> Graph:
    """Dead-code elimination: keep only nodes reachable from the outputs."""
    by_id = {nd.id: nd for nd in graph.nodes}
    reachable: set[int] = set()
    stack = list(graph.outputs)
    while stack:
        i = stack.pop()
        if i in reachable:
            continue
        reachable.add(i)
        nd = by_id.get(i)
        if nd is not None:
            for s in nd.args:
                stack.extend(_spec_refs(s))
    nodes = [nd for nd in graph.nodes if nd.id in reachable]
    return replace(graph, nodes=nodes)


def cse(graph: Graph) -> Graph:
    """Common-subexpression elimination: merge op nodes with identical
    ``(prim, args, params)``. Nodes whose params aren't hashable (array ``getitem``
    keys, slices) are left alone -- conservative, never wrong."""
    remap: dict[int, int] = {}
    seen: dict[Any, int] = {}
    nodes: list[Node] = []
    for nd in graph.nodes:
        args = tuple(_map_spec(s, remap) for s in nd.args)
        if nd.prim is _INPUT or nd.prim is _CONST:
            nodes.append(replace(nd, args=args))
            continue
        try:
            key: Any = (nd.prim, _freeze(args), _freeze(nd.params))
        except TypeError:
            key = None
        if key is not None and key in seen:
            remap[nd.id] = seen[key]  # this node duplicates an earlier one
            continue
        if key is not None:
            seen[key] = nd.id
        nodes.append(replace(nd, args=args))
    outputs = [remap.get(o, o) for o in graph.outputs]
    return replace(graph, nodes=nodes, outputs=outputs)


def _rebuild_const(spec: Any, vals: dict[int, object]) -> object:
    if isinstance(spec, Ref):
        return vals[spec.id]
    if isinstance(spec, Const):
        return spec.value
    if isinstance(spec, (list, tuple)):
        return type(spec)(_rebuild_const(e, vals) for e in spec)
    return spec


def constant_fold(graph: Graph) -> Graph:
    """Evaluate any op node whose data operands are all constants and replace it with
    a single ``Const`` node (computed once, at optimize time)."""
    vals: dict[int, object] = {
        nd.id: nd.params["value"] for nd in graph.nodes if nd.prim is _CONST
    }
    nodes: list[Node] = []
    for nd in graph.nodes:
        if nd.prim is _INPUT or nd.prim is _CONST:
            nodes.append(nd)
            continue
        refs = list(_spec_refs(tuple(nd.args)))
        if refs and all(r in vals for r in refs):
            args = tuple(_rebuild_const(s, vals) for s in nd.args)
            # At base level (no transform live) ``bind`` returns a ``Var``/value,
            # never a ``Tracer``/``None`` -- cast so ``_value`` accepts it.
            val = _value(cast(Operand, bind(nd.prim, *args, **nd.params)))
            vals[nd.id] = val
            nodes.append(Node(nd.id, _CONST, (), {"value": val}, nd.aval))
        else:
            nodes.append(nd)
    return replace(graph, nodes=nodes)


def _scalar_const(spec: Any, const_of: dict[int, object]) -> "float | None":
    """The python scalar a 0-d constant operand holds (inline ``Const`` or ``Ref`` to a
    scalar ``Const`` node), else ``None``. Used for shape-safe identity rewrites: a
    *scalar* identity broadcasts into the other operand without changing its shape.
    Guards by type -- a non-numeric constant (a ``slice`` from a ``getitem`` key, a
    string subscript) is not a scalar identity."""
    if isinstance(spec, Const):
        v: object = spec.value
    elif isinstance(spec, Ref) and spec.id in const_of:
        v = const_of[spec.id]
    else:
        return None
    if isinstance(v, Var):
        v = _value(v)
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float, np.number)):
        return float(v)
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return float(cast(Any, v))
    return None


def _result_spec(spec: Any) -> "tuple[str, Any] | None":
    if isinstance(spec, Ref):
        return ("ref", spec.id)
    if isinstance(spec, Const):
        return ("const", spec.value)
    return None


def _simplify(nd: Node, const_of: dict[int, object]) -> "tuple[str, Any] | None":
    from pycograd import ops

    if len(nd.args) != 2:
        return None
    a, b = nd.args
    sa, sb = _scalar_const(a, const_of), _scalar_const(b, const_of)
    if nd.prim is ops.d_add:  # x + 0, 0 + x
        if sb == 0.0:
            return _result_spec(a)
        if sa == 0.0:
            return _result_spec(b)
    elif nd.prim is ops.d_sub:  # x - 0
        if sb == 0.0:
            return _result_spec(a)
    elif nd.prim is ops.d_mul:  # x * 1, 1 * x, x * 0, 0 * x
        if sb == 1.0:
            return _result_spec(a)
        if sa == 1.0:
            return _result_spec(b)
        if sb == 0.0 or sa == 0.0:
            return ("const", np.zeros(nd.aval.shape, dtype=nd.aval.dtype))
    elif nd.prim is ops.d_div:  # x / 1
        if sb == 1.0:
            return _result_spec(a)
    return None


def algebraic(graph: Graph) -> Graph:
    """Peephole algebraic identities with a *scalar* identity operand (so the result
    shape provably equals the kept operand's): ``x+0``, ``0+x``, ``x-0``, ``x*1``,
    ``1*x``, ``x/1`` collapse to ``x``; ``x*0`` / ``0*x`` collapse to a zero const."""
    const_of: dict[int, object] = {
        nd.id: nd.params["value"] for nd in graph.nodes if nd.prim is _CONST
    }
    remap: dict[int, int] = {}
    nodes: list[Node] = []
    for nd in graph.nodes:
        nd = replace(nd, args=tuple(_map_spec(s, remap) for s in nd.args))
        res = _simplify(nd, const_of) if nd.prim not in (_INPUT, _CONST) else None
        if res is None:
            nodes.append(nd)
        elif res[0] == "ref":
            remap[nd.id] = res[1]  # node is identity to an operand -> drop, remap
        else:  # ("const", value)
            const_of[nd.id] = res[1]
            nodes.append(Node(nd.id, _CONST, (), {"value": res[1]}, nd.aval))
    outputs = [remap.get(o, o) for o in graph.outputs]
    return replace(graph, nodes=nodes, outputs=outputs)


def _use_counts(graph: Graph) -> dict[int, int]:
    counts: dict[int, int] = {}
    for nd in graph.nodes:
        for s in nd.args:
            for r in _spec_refs(s):
                counts[r] = counts.get(r, 0) + 1
    for o in graph.outputs:
        counts[o] = counts.get(o, 0) + 1
    return counts


def fuse_gated_act(graph: Graph) -> Graph:
    """Fuse ``tanh(f) * sigmoid(s)`` into a single ``d_gated_act`` node when the
    ``d_tanh`` and ``d_sigmoid`` feed only this multiply (use-count 1), then DCE the
    now-dead pair. The first automatic fusion -- the pattern set is extensible."""
    from pycograd import ops

    by_id = {nd.id: nd for nd in graph.nodes}
    uses = _use_counts(graph)
    nodes: list[Node] = []
    for nd in graph.nodes:
        fused = None
        if nd.prim is ops.d_mul and len(nd.args) == 2:
            a, b = nd.args
            if isinstance(a, Ref) and isinstance(b, Ref):
                na, nb = by_id.get(a.id), by_id.get(b.id)
                pair = _gate_pair(na, nb, uses, ops)
                if pair is not None:
                    f_spec, s_spec = pair
                    fused = Node(nd.id, ops.d_gated_act, (f_spec, s_spec), {}, nd.aval)
        nodes.append(fused if fused is not None else nd)
    return dce(replace(graph, nodes=nodes))


def _gate_pair(
    na: "Node | None", nb: "Node | None", uses: dict[int, int], ops: Any
) -> "tuple[Any, Any] | None":
    """``(tanh_operand, sigmoid_operand)`` if ``na``/``nb`` are a tanh+sigmoid pair each
    used exactly once, in either order; else ``None``."""
    if na is None or nb is None:
        return None
    if uses.get(na.id) != 1 or uses.get(nb.id) != 1:
        return None
    if na.prim is ops.d_tanh and nb.prim is ops.d_sigmoid:
        return na.args[0], nb.args[0]
    if na.prim is ops.d_sigmoid and nb.prim is ops.d_tanh:
        return nb.args[0], na.args[0]
    return None


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
DEFAULT_PASSES: list[Pass] = [algebraic, constant_fold, cse, fuse_gated_act, dce]


def _measure(graph: Graph) -> tuple[int, int]:
    ops = sum(
        1 for nd in graph.nodes if nd.prim is not _INPUT and nd.prim is not _CONST
    )
    return (len(graph.nodes), ops)


def optimize(graph: Graph, passes: "list[Pass] | None" = None) -> Graph:
    """Apply ``passes`` (default DCE / CSE / constant-folding) to a fixpoint. Passes are
    monotonic (node and op counts never grow), so this converges; the loop is capped
    as a backstop."""
    passes = DEFAULT_PASSES if passes is None else passes
    for _ in range(100):
        before = _measure(graph)
        for p in passes:
            graph = p(graph)
        if _measure(graph) == before:
            break
    return graph

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
from pycograd.capture import _CONST, _INPUT, _WEIGHT, Const, Graph, Node, Ref
from pycograd.cost import DEFAULT_COST_MODEL, matmul_flops, node_flops
from pycograd.shapes import ShapeDtypeStruct
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
    weight_inputs = {k: v for k, v in graph.weight_inputs.items() if v in reachable}
    return replace(graph, nodes=nodes, weight_inputs=weight_inputs)


def cse(graph: Graph) -> Graph:
    """Common-subexpression elimination: merge op nodes with identical
    ``(prim, args, params)``. Nodes whose params aren't hashable (array ``getitem``
    keys, slices) are left alone -- conservative, never wrong."""
    remap: dict[int, int] = {}
    seen: dict[Any, int] = {}
    nodes: list[Node] = []
    for nd in graph.nodes:
        args = tuple(_map_spec(s, remap) for s in nd.args)
        if nd.prim is _INPUT or nd.prim is _CONST or nd.prim is _WEIGHT:
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
        if nd.prim is _INPUT or nd.prim is _CONST or nd.prim is _WEIGHT:
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


def _const_value(spec: Any, const_of: dict[int, object]) -> object:
    """The constant a spec holds (inline ``Const`` or ``Ref`` to a ``Const`` node, a
    ``Var`` stripped to its value), else a sentinel ``_NOT_CONST``."""
    if isinstance(spec, Const):
        v: object = spec.value
    elif isinstance(spec, Ref) and spec.id in const_of:
        v = const_of[spec.id]
    else:
        return _NOT_CONST
    return _value(v) if isinstance(v, Var) else v


_NOT_CONST = object()


def _is_const_all(spec: Any, const_of: dict[int, object], target: float) -> bool:
    """True if ``spec`` is a numeric constant whose every element equals ``target`` (a
    scalar *or* array zero/one). Non-numeric constants -- a ``slice`` getitem key, a
    string subscript -- are not identities."""
    v = _const_value(spec, const_of)
    if v is _NOT_CONST or isinstance(v, (bool, slice, str)) or v is None:
        return False
    try:
        arr = np.asarray(v)
    except (TypeError, ValueError):
        return False
    if arr.dtype == object or arr.size == 0:
        return False
    return bool(np.all(arr == target))


def _shape_of(
    spec: Any, shape_by_id: dict[int, Any], const_of: dict[int, object]
) -> Any:
    """The (host-side) shape an operand will have, from the producing node's aval or a
    constant's array shape; ``None`` if unknown."""
    if isinstance(spec, Ref):
        return shape_by_id.get(spec.id)
    v = _const_value(spec, const_of)
    if v is _NOT_CONST:
        return None
    try:
        return np.shape(cast(Any, v))
    except (TypeError, ValueError):
        return None


def _result_spec(spec: Any) -> "tuple[str, Any] | None":
    if isinstance(spec, Ref):
        return ("ref", spec.id)
    if isinstance(spec, Const):
        return ("const", spec.value)
    return None


def _simplify(
    nd: Node, const_of: dict[int, object], shape_by_id: dict[int, Any]
) -> "tuple[str, Any] | None":
    """An algebraic rewrite for ``nd`` -- ``("ref", id)`` (node is an operand, drop it)
    or ``("const", value)`` (node becomes a constant) -- else ``None``. The
    shape guard ``operand.shape == out.shape`` is what makes an *array* (not just
    scalar) identity safe: it proves the identity operand did not broadcast the kept
    operand up to a larger shape."""
    from pycograd import ops

    out_shape = nd.aval.shape
    # Shape-only ops: a reshape / broadcast to a shape the input already has is a no-op.
    if nd.prim is ops.d_reshape or nd.prim is ops.d_broadcast_to:
        x = nd.args[0]
        if _shape_of(x, shape_by_id, const_of) == out_shape:
            return _result_spec(x)
        return None
    if len(nd.args) != 2:
        return None
    a, b = nd.args
    keep_a = _shape_of(a, shape_by_id, const_of) == out_shape
    keep_b = _shape_of(b, shape_by_id, const_of) == out_shape
    if nd.prim is ops.d_add:  # x + 0, 0 + x
        if _is_const_all(b, const_of, 0) and keep_a:
            return _result_spec(a)
        if _is_const_all(a, const_of, 0) and keep_b:
            return _result_spec(b)
    elif nd.prim is ops.d_sub:  # x - 0
        if _is_const_all(b, const_of, 0) and keep_a:
            return _result_spec(a)
    elif nd.prim is ops.d_mul:  # x * 1, 1 * x, x * 0, 0 * x
        if _is_const_all(b, const_of, 1) and keep_a:
            return _result_spec(a)
        if _is_const_all(a, const_of, 1) and keep_b:
            return _result_spec(b)
        if _is_const_all(b, const_of, 0) or _is_const_all(a, const_of, 0):
            return ("const", np.zeros(out_shape, dtype=nd.aval.dtype))
    elif nd.prim is ops.d_div:  # x / 1
        if _is_const_all(b, const_of, 1) and keep_a:
            return _result_spec(a)
    return None


def algebraic(graph: Graph) -> Graph:
    """Shape-aware peephole algebraic identities. Using each node's inferred aval, a
    zero/one identity operand -- *scalar or array* -- collapses ``x+0``, ``0+x``,
    ``x-0``, ``x*1``, ``1*x``, ``x/1`` to ``x`` and ``x*0`` / ``0*x`` to a zero const,
    and a ``reshape`` / ``broadcast_to`` to the input's own shape drops out. The
    ``operand.shape == out.shape`` guard keeps the array cases shape-safe."""
    const_of: dict[int, object] = {
        nd.id: nd.params["value"] for nd in graph.nodes if nd.prim is _CONST
    }
    shape_by_id: dict[int, Any] = {nd.id: nd.aval.shape for nd in graph.nodes}
    remap: dict[int, int] = {}
    nodes: list[Node] = []
    for nd in graph.nodes:
        nd = replace(nd, args=tuple(_map_spec(s, remap) for s in nd.args))
        res = (
            _simplify(nd, const_of, shape_by_id)
            if nd.prim not in (_INPUT, _CONST, _WEIGHT)
            else None
        )
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


def _producer(spec: Any, by_id: dict[int, Node]) -> "Node | None":
    """The node ``spec`` refers to, if it is a ``Ref`` to one; else ``None``."""
    return by_id.get(spec.id) if isinstance(spec, Ref) else None


def fuse_logsumexp(graph: Graph) -> Graph:
    """Fuse ``log(sum(exp(Z), axis, keepdims))`` into a single ``d_logsumexp(Z)`` node when
    the ``exp`` and ``sum`` feed only this chain (use-count 1), then DCE the dead pair.
    Always semantics-preserving: ``logsumexp(Z) == log(sum(exp(Z)))`` exactly (the fused
    primitive just computes it stably) -- so a naive inline ``log(sum(exp(x)))`` becomes
    both fused *and* overflow-safe, and a stable ``m + log(sum(exp(x-m)))`` fuses its
    expensive ``exp``/``sum``/``log`` cluster (the cheap ``m``/subtract residual is
    value-equivalent: ``logsumexp(x-m)`` differs from the whole only by the explicit ``+m``).
    """
    from pycograd import ops

    by_id = {nd.id: nd for nd in graph.nodes}
    uses = _use_counts(graph)
    nodes: list[Node] = []
    for nd in graph.nodes:
        fused = None
        if nd.prim is ops.d_log:
            sn = _producer(nd.args[0], by_id)
            if sn is not None and sn.prim is ops.d_sum and uses.get(sn.id) == 1:
                en = _producer(sn.args[0], by_id)
                if en is not None and en.prim is ops.d_exp and uses.get(en.id) == 1:
                    params = {
                        "axis": sn.params.get("axis"),
                        "keepdims": sn.params.get("keepdims", False),
                    }
                    fused = Node(nd.id, ops.d_logsumexp, (en.args[0],), params, nd.aval)
        nodes.append(fused if fused is not None else nd)
    return dce(replace(graph, nodes=nodes))


def fuse_softmax(graph: Graph) -> Graph:
    """Fuse ``exp(Z) / sum(exp(Z), axis, keepdims=True)`` into a single ``d_softmax(Z)``
    node, then DCE the dead ``exp``/``sum``. Always semantics-preserving: softmax is
    shift-invariant, so ``d_softmax(Z) == exp(Z)/sum(exp(Z))`` exactly (computed stably) --
    a naive inline softmax becomes fused *and* overflow-safe, and a stable
    ``exp(x-m)/sum(exp(x-m))`` fuses to the value-equivalent ``d_softmax(x-m) == softmax(x)``.
    Handles the numerator and the in-sum ``exp`` being either one shared node (post-``cse``,
    use-count 2) or two structurally-equal nodes (pre-``cse``)."""
    from pycograd import ops

    by_id = {nd.id: nd for nd in graph.nodes}
    uses = _use_counts(graph)
    nodes: list[Node] = []
    for nd in graph.nodes:
        fused = None
        if nd.prim is ops.d_div and len(nd.args) == 2:
            num, den = _producer(nd.args[0], by_id), _producer(nd.args[1], by_id)
            if (
                num is not None
                and num.prim is ops.d_exp
                and den is not None
                and den.prim is ops.d_sum
                and den.params.get("keepdims") is True
                and uses.get(den.id) == 1
            ):
                inner = _producer(den.args[0], by_id)
                if inner is not None and inner.prim is ops.d_exp:
                    same = inner.id == num.id
                    # The two exps must be exp of the same value, and become dead: either
                    # one shared node used exactly by (div, sum), or two single-use nodes.
                    dead = (same and uses.get(num.id) == 2) or (
                        not same
                        and uses.get(num.id) == 1
                        and uses.get(inner.id) == 1
                        and num.args[0] == inner.args[0]
                    )
                    if dead:
                        params = {"axis": den.params.get("axis")}
                        fused = Node(
                            nd.id, ops.d_softmax, (num.args[0],), params, nd.aval
                        )
        nodes.append(fused if fused is not None else nd)
    return dce(replace(graph, nodes=nodes))


# ---------------------------------------------------------------------------
# Matmul-chain reordering (classic matrix-chain DP over the captured avals).
#
# A chain ``A1 @ A2 @ ... @ Ak`` is captured as some binary tree of ``_matmul`` nodes;
# matmul is associative, so reassociating to minimise FLOPs is value- and gradient-
# preserving (``eval_graph`` replays through ``bind``, so the reverse pass differentiates
# the reordered chain identically -- the only change is ~ULP float reordering). The pass
# keeps the same ``k-1`` matmul *count* (so ``optimize``'s monotonic node/op-count
# invariant holds) and reuses the chain's node ids; it is idempotent at the DP optimum, so
# it cannot oscillate.
# ---------------------------------------------------------------------------
def _matmul_chain_factors(
    node: Node, by_id: dict[int, Node], uses: dict[int, int], matmul: Any
) -> "tuple[list[Any], list[Node]]":
    """Flatten the linear matmul chain rooted at ``node`` into ``(factor_specs, internal)``:
    its ordered operand factors and the absorbed (single-use) ``_matmul`` nodes feeding it.
    A multi-use or non-matmul operand stays an opaque leaf factor."""
    factors: list[Any] = []
    internal: list[Node] = []
    for op in node.args:
        p = by_id.get(op.id) if isinstance(op, Ref) else None
        if p is not None and p.prim is matmul and uses.get(p.id) == 1:
            sub_f, sub_i = _matmul_chain_factors(p, by_id, uses, matmul)
            factors.extend(sub_f)
            internal.append(p)
            internal.extend(sub_i)
        else:
            factors.append(op)
    return factors, internal


def reorder_matmul_chain(graph: Graph) -> Graph:
    """Reassociate each maximal linear ``_matmul`` chain to its minimum-FLOP parenthesization
    (matrix-chain DP). Handles 2-D and batched (equal leading batch dims) chains; skips a
    chain with unknown/symbolic dims, mismatched batch dims, or non-chaining shapes."""
    from pycograd import ops

    matmul = ops._matmul
    by_id = {nd.id: nd for nd in graph.nodes}
    uses = _use_counts(graph)
    shape_by_id = {nd.id: nd.aval.shape for nd in graph.nodes}
    const_of: dict[int, object] = {
        nd.id: nd.params["value"] for nd in graph.nodes if nd.prim is _CONST
    }
    matmuls = [nd for nd in graph.nodes if nd.prim is matmul]
    mm_operand_ids = {r for m in matmuls for op in m.args for r in _spec_refs(op)}

    block_at: dict[int, list[Node]] = (
        {}
    )  # tail id -> reordered chain nodes (topo order)
    removed: set[int] = (
        set()
    )  # all chain matmul ids (tail + internal) of reordered chains
    for tail in matmuls:
        # process each chain once, from its tail (a matmul not absorbed into a parent matmul)
        if uses.get(tail.id) == 1 and tail.id in mm_operand_ids:
            continue
        factors, internal = _matmul_chain_factors(tail, by_id, uses, matmul)
        k = len(factors)
        if k < 3:
            continue  # a single matmul: nothing to reassociate
        shapes = [_shape_of(f, shape_by_id, const_of) for f in factors]
        if any(s is None or len(s) < 2 for s in shapes):
            continue
        rank = len(shapes[0])
        if any(len(s) != rank for s in shapes):
            continue  # mixed rank: skip conservatively
        batch = shapes[0][:-2]
        if any(s[:-2] != batch for s in shapes):
            continue  # require identical batch dims (no broadcasting batch)
        rows = [s[-2] for s in shapes]
        cols = [s[-1] for s in shapes]
        if any(cols[i] != rows[i + 1] for i in range(k - 1)):
            continue  # not a valid contraction chain
        dims = rows + [cols[-1]]  # p[0..k]: factor i is (dims[i], dims[i+1])
        if any(not isinstance(d, int) for d in dims) or any(
            not isinstance(b, int) for b in batch
        ):
            continue  # symbolic dims: the FLOP model needs concrete sizes
        batch_size = 1
        for b in batch:
            batch_size *= b

        # matrix-chain DP: m[i][j] = min FLOPs for factors i..j (inclusive). Candidate
        # products are costed through the shared ``cost.matmul_flops`` (a product i..j has
        # ``batch_size * dims[i] * dims[j+1]`` outputs and contracted dim ``dims[s+1]``).
        inf = float("inf")
        m: list[list[float]] = [[0.0] * k for _ in range(k)]
        split = [[0] * k for _ in range(k)]
        for length in range(2, k + 1):
            for i in range(k - length + 1):
                j = i + length - 1
                m[i][j] = inf
                for s in range(i, j):
                    c = (
                        m[i][s]
                        + m[s + 1][j]
                        + matmul_flops(batch_size * dims[i] * dims[j + 1], dims[s + 1])
                    )
                    if c < m[i][j]:
                        m[i][j], split[i][j] = c, s
        opt_cost = m[0][k - 1]

        # current cost: the existing chain matmuls' FLOPs via the same cost model.
        cur_cost = sum(
            node_flops(mm, by_id, DEFAULT_COST_MODEL) for mm in [tail, *internal]
        )
        if opt_cost >= cur_cost:
            continue  # strictly cheaper only -> idempotent at the optimum

        # Rebuild: reuse the chain's node ids (root keeps the tail id so external refs hold).
        pool = [n.id for n in internal]  # exactly k-2 ids for the k-2 non-root products
        new_nodes: list[Node] = []
        dtype = tail.aval.dtype

        def build(i: int, j: int) -> Any:
            if i == j:
                return factors[i]
            s = split[i][j]
            left, right = build(i, s), build(s + 1, j)
            nid = tail.id if (i == 0 and j == k - 1) else pool.pop()
            shape = batch + (dims[i], dims[j + 1])
            new_nodes.append(
                Node(nid, matmul, (left, right), {}, ShapeDtypeStruct(shape, dtype))
            )
            return Ref(nid)

        build(0, k - 1)
        block_at[tail.id] = new_nodes
        removed.add(tail.id)
        removed.update(n.id for n in internal)

    if not block_at:
        return graph
    nodes: list[Node] = []
    for nd in graph.nodes:
        if nd.id in removed:
            if nd.id in block_at:  # the tail: splice the reordered block in its place
                nodes.extend(block_at[nd.id])
            # internal chain matmuls: dropped (their ids are reused inside the block)
        else:
            nodes.append(nd)
    return replace(graph, nodes=nodes)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
DEFAULT_PASSES: list[Pass] = [
    algebraic,
    constant_fold,
    cse,
    reorder_matmul_chain,
    fuse_gated_act,
    fuse_logsumexp,
    fuse_softmax,
    dce,
]


def _measure(graph: Graph) -> tuple[int, int]:
    ops = sum(
        1
        for nd in graph.nodes
        if nd.prim is not _INPUT and nd.prim is not _CONST and nd.prim is not _WEIGHT
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

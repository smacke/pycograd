# -*- coding: utf-8 -*-
"""Rematerialization / spill planning over the capture IR.

Given a captured forward+backward :class:`~pycograd.capture.Graph` (from
:func:`~pycograd.value_and_grad` on a captured graph) and a hard RAM budget, decide for
each
forward *activation* whether to **keep** it resident, **spill** it to SSD and reload
it on its backward use, or **recompute** (rematerialize) it -- then act on the plan.

The decision is **staged** (see the design discussion that motivated this module):

* **Stage 1 -- where to checkpoint.** Choose the resident set so peak-live-memory <=
  budget. This is the binding, NP-hard part (a hard *temporal packing* constraint, not
  expressible as a min-cut), so we use a greedy density heuristic
  (``bytes_held / recompute_cost``, the Chen/Checkmate rule), with an optional exact
  branch-and-bound for small graphs to bound the gap.
* **Stage 2 -- recompute vs spill.** For each evicted activation, choosing reload-from-SSD
  vs rematerialize *is* exactly Helix's (Xin et al., VLDB'19) project-selection min-cut:
  a genuine cost tradeoff with precedence. PTIME via max-flow. It degenerates to
  all-recompute when reload is never cheaper than recompute (the common small-tensor
  case the cost model flags).

The two couple through the budget (spilling also frees RAM), so we iterate.

All per-node costs come from :mod:`pycograd.cost`. :func:`plan_remat` returns a
:class:`RematPlan` (decisions + projected peak / added time / spilled bytes);
:func:`apply_remat_plan` rewrites the graph with :func:`~pycograd.ops._spill` /
:func:`~pycograd.ops._recompute` markers; :func:`eval_scheduled` runs the rewritten
graph with a memory-managed interpreter so the result is byte-identical to plain
:func:`~pycograd.capture.eval_graph` while the resident high-water stays under budget.
"""
from __future__ import annotations

import enum
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Optional, cast

import numpy as np

from pycograd import ops
from pycograd._typing import Array
from pycograd.capture import _CONST, _INPUT, _WEIGHT, Const, Graph, Node, Ref
from pycograd.cost import (
    DEFAULT_COST_MODEL,
    CostModel,
    cost_report,
    live_intervals,
    peak_of,
)
from pycograd.passes import _map_spec, _spec_refs
from pycograd.tensor import Var, _value
from pycograd.trace import bind
from pycograd.tree import PyTree, tree_flatten, tree_unflatten


class Decision(enum.Enum):
    """What to do with one forward activation for its backward use."""

    KEEP = "keep"  # stay resident in RAM across its whole live range
    SPILL = "spill"  # page to SSD after forward use, reload on backward use
    RECOMPUTE = "recompute"  # drop after forward use, rematerialize on backward use


# ---------------------------------------------------------------------------
# Graph topology helpers.
# ---------------------------------------------------------------------------
def _refs(args: tuple) -> "set[int]":
    out: set[int] = set()
    for s in args:
        out.update(_spec_refs(s))
    return out


def _weight_value(graph: Graph, nd: Node) -> "Array":
    """The current array for a live ``_WEIGHT`` leaf -- read from the graph's source
    ``ParamDict`` (so a stepped weight is picked up), mirroring ``eval_graph``."""
    return np.asarray(_value(graph.weight_owner._resolve_weight(nd.params["key"])))


def _consumers(graph: Graph) -> "dict[int, list[int]]":
    """Producer node id -> list of node ids that reference it as an operand."""
    cons: dict[int, list[int]] = defaultdict(list)
    for nd in graph.nodes:
        for r in _refs(nd.args):
            cons[r].append(nd.id)
    return cons


def _forward_cone(graph: Graph) -> "set[int]":
    """Node ids reachable as ancestors of the primal value output ``outputs[0]`` -- i.e.
    the forward pass. The rest of a value-and-grad graph is the backward pass."""
    if not graph.outputs:
        return {nd.id for nd in graph.nodes}
    by_id = {nd.id: nd for nd in graph.nodes}
    seen: set[int] = set()
    stack = [graph.outputs[0]]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        nd = by_id.get(i)
        if nd is not None:
            stack.extend(_refs(nd.args))
    return seen


@dataclass(frozen=True)
class _Activation:
    """A checkpointable forward activation: a forward-cone op node consumed by the backward
    pass. ``fwd_last`` is the SSA step of its last forward consumer (it can be freed after
    that); ``bwd_steps`` are the steps of its backward consumers (where it must reappear).
    """

    id: int
    produce: int
    fwd_last: int
    bwd_steps: tuple[int, ...]
    nbytes: int


def _activations(
    graph: Graph, model: CostModel
) -> "tuple[dict[int, _Activation], dict[int, tuple[int, int, int]]]":
    """The checkpointable activations and the baseline live intervals."""
    intervals = live_intervals(graph, model)
    step_of = {nd.id: i for i, nd in enumerate(graph.nodes)}
    cone = _forward_cone(graph)
    cons = _consumers(graph)
    outputs = set(graph.outputs)
    acts: dict[int, _Activation] = {}
    for nd in graph.nodes:
        if nd.prim in (_INPUT, _CONST, _WEIGHT) or nd.id in outputs:
            continue
        if nd.id not in cone:
            continue
        fwd = [step_of[c] for c in cons.get(nd.id, ()) if c in cone]
        bwd = [step_of[c] for c in cons.get(nd.id, ()) if c not in cone]
        if not bwd:
            continue  # purely-forward value; backward never needs it
        produce, _last, nbytes = intervals[nd.id]
        acts[nd.id] = _Activation(
            id=nd.id,
            produce=produce,
            fwd_last=max(fwd) if fwd else produce,
            bwd_steps=tuple(sorted(bwd)),
            nbytes=nbytes,
        )
    return acts, intervals


# ---------------------------------------------------------------------------
# Peak-memory model under a keep/evict assignment.
# ---------------------------------------------------------------------------
def _peak_under(
    graph: Graph,
    intervals: "dict[int, tuple[int, int, int]]",
    acts: "dict[int, _Activation]",
    evicted: "set[int]",
) -> "tuple[int, int]":
    """Projected ``(peak_bytes, step)`` when ``evicted`` activations are not kept resident.
    A kept value occupies RAM across its full ``[produce, last_use]`` range; an evicted one
    only across ``[produce, fwd_last]`` plus a one-step blip at each backward use (the
    transient reload/recompute). Inputs and constants are held for the whole schedule (the
    recompute base case ``eval_scheduled`` never frees), so they span ``[produce, n-1]``.
    """
    n = len(graph.nodes)
    by_id = {nd.id: nd for nd in graph.nodes}
    timeline = [0] * n
    # kept nodes: full interval; inputs/consts span the whole run
    for nid, (produce, last, nbytes) in intervals.items():
        if nid in evicted:
            continue
        if by_id[nid].prim in (_INPUT, _CONST, _WEIGHT):
            last = n - 1
        for s in range(produce, last + 1):
            timeline[s] += nbytes
    # evicted activations: short forward span + per-backward-use blips
    for nid in evicted:
        a = acts[nid]
        for s in range(a.produce, a.fwd_last + 1):
            timeline[s] += a.nbytes
        for s in a.bwd_steps:
            timeline[s] += a.nbytes
    return peak_of(timeline)


# ---------------------------------------------------------------------------
# Stage 1: choose the resident set under the budget.
# ---------------------------------------------------------------------------
def _segment_recompute_cost(
    graph: Graph, acts: "dict[int, _Activation]", c_own: "dict[int, float]"
) -> "dict[int, float]":
    """For each activation, the cost to recompute it from the nearest checkpoint boundary
    -- i.e. summing the own roofline times of the transient ops between it and the *other*
    activations / inputs (which act as boundaries). This is the cascade an eviction really
    pays (re-running the layer's matmul to rebuild a cheap relu), not the activation's own
    elementwise cost -- so Stage 1 prioritizes evictions honestly, consistent with Stage 2.
    """
    by_id = {nd.id: nd for nd in graph.nodes}
    boundaries = set(acts)
    out: dict[int, float] = {}
    for aid in acts:
        seen: set[int] = set()
        stack = [aid]
        total = 0.0
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            nd = by_id.get(i)
            if nd is None or nd.prim in (_INPUT, _CONST, _WEIGHT):
                continue
            if i != aid and i in boundaries:
                continue  # stop the cascade at another checkpoint boundary
            total += c_own.get(i, 0.0)
            stack.extend(_refs(nd.args))
        out[aid] = total
    return out


def _stage1_greedy(
    graph: Graph,
    intervals: "dict[int, tuple[int, int, int]]",
    acts: "dict[int, _Activation]",
    costs: "dict[int, float]",
    budget: int,
) -> "tuple[set[int], int]":
    """Greedily evict the densest activation crossing the current peak until peak <= budget
    or no crossing activation remains. ``costs[id]`` is the activation's *regeneration*
    cost (the cheaper of segment-recompute or spill+reload). Returns ``(evicted, peak)``.
    """
    evicted: set[int] = set()
    peak, step = _peak_under(graph, intervals, acts, evicted)
    while peak > budget:
        # candidates: activations whose long tail is live at the peak step
        crossing = [
            a
            for a in acts.values()
            if a.id not in evicted and a.fwd_last < step <= max(a.bwd_steps)
        ]
        if not crossing:
            break  # peak is set by un-evictable values (inputs / kept working set)
        # density: bytes freed per unit of extra recompute cost (Chen/Checkmate rule)
        victim = max(crossing, key=lambda a: a.nbytes / max(costs[a.id], 1e-12))
        evicted.add(victim.id)
        peak, step = _peak_under(graph, intervals, acts, evicted)
    return evicted, peak


def _stage1_exact(
    graph: Graph,
    intervals: "dict[int, tuple[int, int, int]]",
    acts: "dict[int, _Activation]",
    costs: "dict[int, float]",
    budget: int,
    limit: int = 18,
) -> "Optional[tuple[set[int], int]]":
    """Exact min-added-cost resident set with peak <= budget, by branch-and-bound over the
    evictable activations. Returns ``None`` (caller falls back to greedy) when the graph is
    too large to enumerate."""
    ids = list(acts)
    if len(ids) > limit:
        return None
    best: dict[str, object] = {"cost": float("inf"), "evicted": None}

    def rec(i: int, evicted: set[int], added: float) -> None:
        if added >= best["cost"]:  # type: ignore[operator]
            return
        if i == len(ids):
            peak, _ = _peak_under(graph, intervals, acts, evicted)
            if peak <= budget and added < best["cost"]:  # type: ignore[operator]
                best["cost"], best["evicted"] = added, set(evicted)
            return
        # try keeping ids[i]
        rec(i + 1, evicted, added)
        # try evicting ids[i]
        evicted.add(ids[i])
        rec(i + 1, evicted, added + costs[ids[i]])
        evicted.discard(ids[i])

    rec(0, set(), 0.0)
    if best["evicted"] is None:
        return None
    evicted = best["evicted"]  # type: ignore[assignment]
    peak, _ = _peak_under(graph, intervals, acts, evicted)  # type: ignore[arg-type]
    return evicted, peak  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Stage 2: spill vs recompute for the evicted set (Helix project-selection min-cut).
# ---------------------------------------------------------------------------
@dataclass
class _Edge:
    """A residual-graph arc: ``cap`` is the remaining capacity, ``rev`` indexes the paired
    reverse arc in ``g[to]``."""

    to: int
    cap: float
    rev: int


class _MaxFlow:
    """A minimal Dinic max-flow over integer-or-float capacities (graphs here are tiny)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.g: list[list[_Edge]] = [[] for _ in range(n)]

    def add(self, u: int, v: int, cap: float) -> None:
        self.g[u].append(_Edge(v, cap, len(self.g[v])))
        self.g[v].append(_Edge(u, 0.0, len(self.g[u]) - 1))

    def _bfs(self, s: int, t: int) -> "Optional[list[int]]":
        level = [-1] * self.n
        level[s] = 0
        q = [s]
        while q:
            nxt: list[int] = []
            for u in q:
                for e in self.g[u]:
                    if e.cap > 1e-12 and level[e.to] < 0:
                        level[e.to] = level[u] + 1
                        nxt.append(e.to)
            q = nxt
        return level if level[t] >= 0 else None

    def _dfs(
        self, u: int, t: int, f: float, level: "list[int]", it: "list[int]"
    ) -> float:
        if u == t:
            return f
        while it[u] < len(self.g[u]):
            e = self.g[u][it[u]]
            if e.cap > 1e-12 and level[e.to] == level[u] + 1:
                d = self._dfs(e.to, t, min(f, e.cap), level, it)
                if d > 1e-12:
                    e.cap -= d
                    self.g[e.to][e.rev].cap += d
                    return d
            it[u] += 1
        return 0.0

    def max_flow(self, s: int, t: int) -> float:
        flow = 0.0
        while True:
            level = self._bfs(s, t)
            if level is None:
                return flow
            it = [0] * self.n
            while True:
                f = self._dfs(s, t, float("inf"), level, it)
                if f <= 1e-12:
                    break
                flow += f

    def min_cut_source_side(self, s: int) -> "set[int]":
        """Nodes reachable from ``s`` in the residual graph (the source side of the cut)."""
        seen = {s}
        stack = [s]
        while stack:
            u = stack.pop()
            for e in self.g[u]:
                if e.cap > 1e-12 and e.to not in seen:
                    seen.add(e.to)
                    stack.append(e.to)
        return seen


def _stage2_spill_vs_recompute(
    graph: Graph,
    demands: "set[int]",
    resident: "set[int]",
    l_cost: "dict[int, float]",
    c_cost: "dict[int, float]",
) -> "tuple[dict[int, Decision], set[int]]":
    """Helix project-selection min-cut over the *not-resident* forward subgraph feeding the
    ``demands`` (the evicted activations). Faithful to Helix: ``c_i`` is each node's *own*
    recompute time and the cascade lives in the precedence -- to compute a node its parents
    must be available (loaded or themselves computed), so a chain's recompute cost is the
    sum of per-node ``c`` over the forced-computed set, charged once for shared subgraphs.

    The ``demands`` (checkpointable activations) are *spillable* (an ``a``/``b`` pair: load
    at ``l_i`` or compute at ``c_i``); the transient intermediate ancestors are
    *compute-only* (one project at ``c_j``; ``l = inf`` -- we only ever spill an activation,
    never a mid-layer temporary). Inputs/consts and resident (kept) nodes are free sources.

    Returns ``(decision per demand, set of nodes the plan recomputes)`` -- the second is the
    forced-computed closure, for honest added-compute-time accounting."""
    if not demands:
        return {}, set()
    by_id = {nd.id: nd for nd in graph.nodes}

    # N = the demands plus their forward ancestors that are not resident / inputs / consts
    inN: set[int] = set()
    stack = list(demands)
    while stack:
        i = stack.pop()
        if i in inN or i in resident:
            continue
        nd = by_id.get(i)
        if nd is None or nd.prim in (_INPUT, _CONST, _WEIGHT):
            continue
        inN.add(i)
        stack.extend(_refs(nd.args))

    # slot layout: spillable demand -> (a, b); compute-only intermediate -> (q)
    a_slot: dict[int, int] = {}
    b_slot: dict[int, int] = {}
    q_slot: dict[int, int] = {}
    n = 0
    for nid in sorted(inN):
        if nid in demands:
            a_slot[nid], b_slot[nid], n = n, n + 1, n + 2
        else:
            q_slot[nid], n = n, n + 1
    S, T = n, n + 1
    flow = _MaxFlow(n + 2)
    INF = float("inf")

    def avail(nid: int) -> int:
        return a_slot[nid] if nid in demands else q_slot[nid]

    def require_parents(slot: int, nid: int) -> None:
        for r in _refs(by_id[nid].args):
            if r in inN:  # resident / input / const parents are free
                flow.add(slot, avail(r), INF)

    for nid in inN:
        if nid in demands:
            ai, bi = a_slot[nid], b_slot[nid]
            flow.add(
                ai, T, l_cost[nid]
            )  # a_i: pay l_i to make available (spill+reload)
            p = l_cost[nid] - c_cost[nid]  # b_i: upgrade to compute (net cost c_i)
            if p > 0:
                flow.add(S, bi, p)
            elif p < 0:
                flow.add(bi, T, -p)
            flow.add(bi, ai, INF)  # compute requires available
            require_parents(bi, nid)  # compute requires parents available
            flow.add(S, ai, INF)  # demanded by backward -> forced available
        else:
            qj = q_slot[nid]
            flow.add(qj, T, c_cost[nid])  # compute-only: pay c_j if computed
            require_parents(qj, nid)

    flow.max_flow(S, T)
    src = flow.min_cut_source_side(S)
    out: dict[int, Decision] = {}
    for nid in demands:
        out[nid] = Decision.RECOMPUTE if b_slot[nid] in src else Decision.SPILL
    recomputed = {
        nid
        for nid in inN
        if avail(nid) in src and (nid not in demands or b_slot[nid] in src)
    }
    return out, recomputed


# ---------------------------------------------------------------------------
# The plan.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RematPlan:
    """A keep/spill/recompute assignment for one graph under one budget."""

    decisions: "dict[int, Decision]"
    budget: int
    baseline_peak: int
    planned_peak: int
    added_compute_time: float
    added_io_time: float
    spilled_bytes: int
    feasible: bool

    def kept(self) -> "list[int]":
        return [i for i, d in self.decisions.items() if d is Decision.KEEP]

    def spilled(self) -> "list[int]":
        return [i for i, d in self.decisions.items() if d is Decision.SPILL]

    def recomputed(self) -> "list[int]":
        return [i for i, d in self.decisions.items() if d is Decision.RECOMPUTE]

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"RematPlan(peak {self.baseline_peak / 1e6:.3g}->"
            f"{self.planned_peak / 1e6:.3g}MB / budget {self.budget / 1e6:.3g}MB, "
            f"{len(self.spilled())} spill, {len(self.recomputed())} remat, "
            f"{'ok' if self.feasible else 'INFEASIBLE'})"
        )

    def pretty(self, max_ids: int = 16) -> str:
        def _ids(label: str, ids: "list[int]") -> str:
            shown = " ".join(f"%{i}" for i in ids[:max_ids])
            more = f" ... (+{len(ids) - max_ids} more)" if len(ids) > max_ids else ""
            return f"  {label} ({len(ids)}): {shown}{more}"

        lines = [f"remat plan (budget {self.budget / 1e6:.3g} MB) {{"]
        if self.spilled():
            lines.append(_ids("spill", self.spilled()))
        if self.recomputed():
            lines.append(_ids("recompute", self.recomputed()))
        verdict = "feasible" if self.feasible else "INFEASIBLE (peak > budget)"
        lines.append(
            f"  peak {self.baseline_peak / 1e6:.3g} -> {self.planned_peak / 1e6:.3g} MB; "
            f"+{self.added_compute_time * 1e3:.3g} ms compute, "
            f"+{self.added_io_time * 1e3:.3g} ms io, "
            f"{self.spilled_bytes / 1e6:.3g} MB spilled; {verdict}"
        )
        lines.append("}")
        return "\n".join(lines)


def plan_remat(
    graph: Graph,
    budget: int,
    model: CostModel = DEFAULT_COST_MODEL,
    *,
    exact: bool = False,
    iters: int = 2,
) -> RematPlan:
    """Plan keep/spill/recompute for ``graph`` so its peak resident memory fits ``budget``
    (bytes). Stage 1 picks the resident set (greedy, or exact small-graph branch-and-bound
    when ``exact``); Stage 2 splits the evicted set into spill vs recompute via the Helix
    min-cut; the two iterate up to ``iters`` times (spilling also frees RAM)."""
    acts, intervals = _activations(graph, model)
    report = cost_report(graph, model)
    nc = report.by_id()
    # own recompute time c_i for *every* forward op node (the Stage-2 cascade walks these),
    # reload time l_i for the spillable activations.
    c_cost = {i: nc[i].recompute_time for i in nc}
    l_cost = {
        i: nc[i].out_bytes / model.ssd_read_bandwidth + model.ssd_latency for i in acts
    }
    inputs_consts = {
        nd.id for nd in graph.nodes if nd.prim in (_INPUT, _CONST, _WEIGHT)
    }
    # Stage-1 eviction cost = the cheaper of recomputing the segment (cascade through the
    # layer's matmul) or spilling and reloading -- what Stage 2 will actually pay.
    seg_cost = _segment_recompute_cost(graph, acts, c_cost)
    evict_cost = {i: min(seg_cost[i], l_cost.get(i, float("inf"))) for i in acts}
    baseline_peak, _ = _peak_under(graph, intervals, acts, set())

    evicted: set[int] = set()
    planned_peak = baseline_peak
    for _ in range(max(1, iters)):
        if exact:
            res = _stage1_exact(graph, intervals, acts, evict_cost, budget)
            res = (
                res
                if res is not None
                else _stage1_greedy(graph, intervals, acts, evict_cost, budget)
            )
        else:
            res = _stage1_greedy(graph, intervals, acts, evict_cost, budget)
        new_evicted, planned_peak = res
        if new_evicted == evicted:
            break
        evicted = new_evicted

    # Stage 2 runs the Helix min-cut over the not-resident forward subgraph: kept
    # activations (acts we did not evict) and inputs/consts are free sources.
    resident = (set(acts) - evicted) | inputs_consts
    stage2, recomputed = _stage2_spill_vs_recompute(
        graph, evicted, resident, l_cost, c_cost
    )

    decisions = {i: Decision.KEEP for i in acts}
    decisions.update(stage2)
    # added compute = the whole forced-recomputed closure (cascade), each node charged once
    added_compute = sum(c_cost[i] for i in recomputed)
    spilled = [i for i in evicted if stage2[i] is Decision.SPILL]
    spilled_bytes = sum(nc[i].out_bytes for i in spilled)
    added_io = sum(nc[i].spill_time for i in spilled)  # write + read round trip
    return RematPlan(
        decisions=decisions,
        budget=budget,
        baseline_peak=baseline_peak,
        planned_peak=planned_peak,
        added_compute_time=added_compute,
        added_io_time=added_io,
        spilled_bytes=spilled_bytes,
        feasible=planned_peak <= budget,
    )


# ---------------------------------------------------------------------------
# Rewrite: insert _spill / _recompute markers per the plan.
# ---------------------------------------------------------------------------
def apply_remat_plan(graph: Graph, plan: RematPlan) -> Graph:
    """Rewrite ``graph`` so each spilled/recomputed activation is wrapped in an identity
    :func:`~pycograd.ops._spill` / :func:`~pycograd.ops._recompute` marker, with its
    backward consumers remapped to read the marker. The marker is placed right after the
    activation's last forward use, so under :func:`eval_scheduled` the activation is freed
    there. Plain :func:`~pycograd.capture.eval_graph` of the result is unchanged (markers
    are value-identity)."""
    acts, _ = _activations(
        graph, DEFAULT_COST_MODEL
    )  # topology only; nbytes unused here
    cone = _forward_cone(graph)
    step_of = {nd.id: i for i, nd in enumerate(graph.nodes)}
    next_id = max((nd.id for nd in graph.nodes), default=-1) + 1

    # decide marker primitive per rewritten activation
    marked = {
        nid: (ops._spill if d is Decision.SPILL else ops._recompute)
        for nid, d in plan.decisions.items()
        if d in (Decision.SPILL, Decision.RECOMPUTE)
    }
    # build marker nodes and a per-activation remap (backward consumers: old ref -> marker)
    marker_after: dict[int, list[Node]] = defaultdict(list)
    remap: dict[int, int] = {}
    for nid, prim in marked.items():
        a = acts.get(nid)
        if a is None:
            continue
        marker = Node(next_id, prim, (Ref(nid),), {}, graph.nodes[step_of[nid]].aval)
        remap[nid] = next_id
        next_id += 1
        anchor = graph.nodes[
            a.fwd_last
        ].id  # emit the marker right after the last fwd use
        marker_after[anchor].append(marker)

    def remap_backward(spec: object, consumer_in_cone: bool) -> object:
        # only backward consumers (outside the forward cone) read the marker
        if consumer_in_cone:
            return spec
        return _map_spec(spec, remap)

    new_nodes: list[Node] = []
    for nd in graph.nodes:
        in_cone = nd.id in cone
        new_args = tuple(remap_backward(s, in_cone) for s in nd.args)
        new_nodes.append(replace(nd, args=new_args) if new_args != nd.args else nd)
        new_nodes.extend(marker_after.get(nd.id, ()))

    # graph outputs past the primal value are backward -> remap them too
    new_outputs = [remap.get(o, o) if i > 0 else o for i, o in enumerate(graph.outputs)]
    return replace(graph, nodes=new_nodes, outputs=new_outputs)


# ---------------------------------------------------------------------------
# On-disk spill store + memory-managed interpreter.
# ---------------------------------------------------------------------------
class SpillStore:
    """A scratch directory holding spilled arrays. ``put`` writes; ``get`` reads back;
    ``close`` removes the directory. Defaults to a fresh temp dir (override ``root`` to
    pin spills to a specific SSD)."""

    def __init__(self, root: Optional[str] = None) -> None:
        self._own = root is None
        self.root = root or tempfile.mkdtemp(prefix="pycograd-spill-")

    def put(self, key: int, array: Array) -> str:
        path = os.path.join(self.root, f"{key}.npy")
        np.save(path, array)
        return path

    def get(self, path: str) -> Array:
        return np.load(path)

    def close(self) -> None:
        if not self._own:
            return
        for name in os.listdir(self.root):
            try:
                os.unlink(os.path.join(self.root, name))
            except OSError:  # pragma: no cover
                pass
        try:
            os.rmdir(self.root)
        except OSError:  # pragma: no cover
            pass


def eval_scheduled(
    graph: Graph, *inputs: PyTree, store_dir: Optional[str] = None
) -> "tuple[PyTree, int]":
    """Evaluate ``graph`` with a memory-managed interpreter that honours
    :func:`~pycograd.ops._spill` / :func:`~pycograd.ops._recompute` markers, returning
    ``(outputs, peak_resident_bytes)``. Values are freed after their last use; a spilled
    value is paged to ``store_dir`` and reloaded on demand; a recomputed value is dropped
    and its producing subgraph re-evaluated on demand. The returned value is byte-identical
    to plain :func:`~pycograd.capture.eval_graph`; ``peak_resident_bytes`` is the high-water
    of the resident working set (the quantity :func:`plan_remat` budgets)."""
    in_leaves = [leaf for a in inputs for leaf in tree_flatten(a)[0]]
    if len(in_leaves) != len(graph.inputs):
        raise ValueError(
            f"eval_scheduled: expected {len(graph.inputs)} input leaves, "
            f"got {len(in_leaves)}"
        )
    by_id = {nd.id: nd for nd in graph.nodes}
    last_use: dict[int, int] = {}
    for i, nd in enumerate(graph.nodes):
        for r in _refs(nd.args):
            last_use[r] = i
    for o in graph.outputs:
        last_use[o] = len(graph.nodes) - 1

    env: dict[int, Array] = {}  # resident raw arrays (kept working set + inputs/consts)
    disk: dict[int, str] = {}  # spilled: id -> file path
    recompute_src: dict[int, int] = {}  # recompute marker id -> wrapped producer id
    store = SpillStore(store_dir)
    resident = 0
    hwm = 0

    def _nbytes(v: Array) -> int:
        return int(np.asarray(v).nbytes)

    def materialize(nid: int, memo: "dict[int, Array]") -> Array:
        # resident / spilled / already-recomputed-this-demand are cheap hits
        if nid in env:
            return env[nid]
        if nid in disk:
            return store.get(disk[nid])  # transient reload from SSD
        if nid in memo:
            return memo[nid]  # within-demand cache: avoids exponential re-eval
        nd = by_id[nid]
        if (
            nd.prim is _INPUT
        ):  # inputs/consts/weights stay resident -> recompute base case
            return env[nid]
        if nd.prim is _CONST:
            return np.asarray(nd.params["value"])
        if nd.prim is _WEIGHT:
            return _weight_value(graph, nd)
        if nid in recompute_src:  # identity marker: value of its wrapped producer
            v = materialize(recompute_src[nid], memo)
        else:  # freed forward node: re-evaluate from its (materialized) operands
            v = _eval(nd, memo)
        memo[nid] = v
        return v

    def _arg(spec: object, memo: "dict[int, Array]") -> object:
        if isinstance(spec, Ref):
            return materialize(spec.id, memo)
        if isinstance(spec, Const):
            return spec.value
        if isinstance(spec, (list, tuple)):
            return type(spec)(_arg(s, memo) for s in spec)
        return spec

    def _eval(nd: Node, memo: "dict[int, Array]") -> Array:
        args = tuple(_arg(s, memo) for s in nd.args)
        return np.asarray(_value(cast("Var", bind(nd.prim, *args, **nd.params))))

    try:
        for i, nd in enumerate(graph.nodes):
            if nd.prim is _INPUT:
                env[nd.id] = np.asarray(in_leaves[graph.inputs.index(nd.id)])
                resident += _nbytes(env[nd.id])
            elif nd.prim is _CONST:
                env[nd.id] = np.asarray(nd.params["value"])
                resident += _nbytes(env[nd.id])
            elif nd.prim is _WEIGHT:
                env[nd.id] = _weight_value(graph, nd)
                resident += _nbytes(env[nd.id])
            elif nd.prim is ops._spill:
                (src,) = _refs(nd.args)
                disk[nd.id] = store.put(nd.id, materialize(src, {}))  # write to SSD
            elif nd.prim is ops._recompute:
                (src,) = _refs(nd.args)
                recompute_src[nd.id] = src  # drop; rematerialize on demand
            else:
                env[nd.id] = _eval(nd, {})
                resident += _nbytes(env[nd.id])
            hwm = max(hwm, resident)
            # free everything whose last use was this step (never inputs/consts, which
            # stay resident as the recompute base case)
            for u, lu in list(last_use.items()):
                if lu != i:
                    continue
                if by_id[u].prim in (_INPUT, _CONST, _WEIGHT):
                    continue
                if u in env:
                    resident -= _nbytes(env.pop(u))
                if u in disk:
                    try:
                        os.unlink(disk.pop(u))
                    except OSError:  # pragma: no cover
                        disk.pop(u, None)
                recompute_src.pop(u, None)
        out_leaves = [materialize(o, {}) for o in graph.outputs]
    finally:
        store.close()
    return tree_unflatten(graph.out_treedef, out_leaves), hwm

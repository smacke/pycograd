# -*- coding: utf-8 -*-
"""Tests for the rematerialization / spill planner (:mod:`pycograd.remat`).

Two things must hold exactly: the Stage-2 spill-vs-recompute min-cut equals brute force,
and -- the headline -- evaluating a rewritten forward+backward graph with the
memory-managed interpreter reproduces the value *and* gradient of plain ``eval_graph``
while the resident high-water stays within the planner's projection (and the budget when
the plan is feasible). Peak numbers themselves are model estimates, so they are asserted
as invariants (``hwm <= planned_peak``), not wall-clock figures.
"""
import pytest

np = pytest.importorskip("numpy")

from pycograd.ad_graph import grad_graph  # noqa: E402
from pycograd.capture import _INPUT, Graph, Node, Ref, capture, eval_graph  # noqa: E402
from pycograd.cost import CostModel, cost_report  # noqa: E402
from pycograd.examples import models as M  # noqa: E402
from pycograd.passes import optimize  # noqa: E402
from pycograd.remat import (  # noqa: E402
    Decision,
    _MaxFlow,
    _stage2_spill_vs_recompute,
    apply_remat_plan,
    eval_scheduled,
    plan_remat,
)
from pycograd.shapes import ShapeDtypeStruct as SDS  # noqa: E402
from pycograd.tensor import _value  # noqa: E402
from pycograd.tree import tree_leaves  # noqa: E402


def _rng(seed):
    return np.random.default_rng(seed)


def _leaves(out):
    return [np.asarray(_value(x)) for x in tree_leaves(out)]


_MODELS = [
    ("mlp", M.mlp_tree_loss, lambda: (M._init_mlp_tree(_rng(1)),)),
    ("rnn", M.rnn_loss, lambda: (M._init_rnn(_rng(3), vocab=len(M._CHAR_VOCAB)),)),
    ("lstm", M.lstm_loss, lambda: (M._init_lstm(_rng(3), vocab=len(M._CHAR_VOCAB)),)),
]
_IDS = [c[0] for c in _MODELS]


def _grad_graph(loss, args):
    return optimize(grad_graph(capture(loss, *args)))


# ---------------------------------------------------------------------------
# Stage 2: the Helix project-selection min-cut.
# ---------------------------------------------------------------------------
def test_maxflow_classic_small():
    # 4-node network with a known max-flow of 5 (CLRS-style).
    f = _MaxFlow(4)
    f.add(0, 1, 3)
    f.add(0, 2, 2)
    f.add(1, 2, 1)
    f.add(1, 3, 2)
    f.add(2, 3, 3)
    assert f.max_flow(0, 3) == pytest.approx(5)


def _chain_graph(n):
    """A linear chain of ``n`` unary nodes off one input (ids 0..n)."""
    from pycograd import ops

    nodes = [Node(0, _INPUT, (), {}, SDS((4,)))]
    for i in range(1, n + 1):
        nodes.append(Node(i, ops.d_exp, (Ref(i - 1),), {}, SDS((4,))))
    return Graph(nodes, [0], [n], None)


def test_stage2_all_spill_when_reload_cheaper():
    g = _chain_graph(4)
    demands = {1, 2, 3}
    resident = {0}  # only the input is free
    dec, _ = _stage2_spill_vs_recompute(
        g,
        demands,
        resident,
        l_cost={i: 1.0 for i in demands},
        c_cost={i: 5.0 for i in [nd.id for nd in g.nodes]},
    )
    assert all(d is Decision.SPILL for d in dec.values())


def test_stage2_recompute_when_each_parent_resident():
    # When every demand's parent is resident there is no cascade, so cheap own-recompute
    # wins node-by-node.
    g = _chain_graph(4)
    demands = {3}
    resident = {0, 1, 2}
    dec, _ = _stage2_spill_vs_recompute(
        g,
        demands,
        resident,
        l_cost={3: 5.0},
        c_cost={i: 1.0 for i in [nd.id for nd in g.nodes]},
    )
    assert dec[3] is Decision.RECOMPUTE


def test_stage2_cascade_flips_recompute_to_spill():
    # Demand node 3's parents (1,2) are NOT resident, so recomputing 3 cascades through
    # them: true cost 3*c=3 > reload 2 -> SPILL, even though own c (1) < reload (2).
    g = _chain_graph(3)
    demands = {3}
    resident = {0}
    dec, recomputed = _stage2_spill_vs_recompute(
        g,
        demands,
        resident,
        l_cost={3: 2.0},
        c_cost={i: 1.0 for i in [nd.id for nd in g.nodes]},
    )
    assert dec[3] is Decision.SPILL
    assert recomputed == set()  # spilling 3 means 1,2 need not be recomputed


def test_stage2_matches_bruteforce_with_cascade():
    g = _chain_graph(6)
    demands = {2, 4, 6}  # backward needs these; 1,3,5 are compute-only intermediates
    resident = {0}
    by_id = {nd.id: nd for nd in g.nodes}
    for seed in range(30):
        rng = _rng(seed)
        l_cost = {i: float(rng.integers(1, 12)) for i in demands}
        c_cost = {i: float(rng.integers(1, 12)) for i in [nd.id for nd in g.nodes]}
        dec, _ = _stage2_spill_vs_recompute(g, demands, resident, l_cost, c_cost)
        got = _cost_of(g, demands, resident, l_cost, c_cost, dec, by_id)
        best = _bruteforce_cost(g, demands, resident, l_cost, c_cost, by_id)
        assert got == pytest.approx(best)


def _cost_of(g, demands, resident, l_cost, c_cost, dec, by_id):
    """True cost of an assignment: reload for spilled demands + own-recompute for every
    recomputed demand and every compute-only intermediate forced by the cascade -- where
    the cascade stops at any *available* value (resident, an input, or a spilled demand,
    which can be loaded back). Shared intermediates are charged once."""
    spilled = {d for d in demands if dec[d] is Decision.SPILL}
    available = set(resident) | spilled
    need: set = set()
    stack = [d for d in demands if dec[d] is Decision.RECOMPUTE]
    while stack:
        i = stack.pop()
        for s in by_id[i].args:
            if isinstance(s, Ref):
                p = s.id
                if p in available or p in demands or by_id[p].prim is _INPUT:
                    continue  # available, or a demand whose own decision stands
                if p not in need:
                    need.add(p)
                    stack.append(p)
    return (
        sum(l_cost[d] for d in spilled)
        + sum(c_cost[d] for d in demands if dec[d] is Decision.RECOMPUTE)
        + sum(c_cost[j] for j in need)
    )


def _bruteforce_cost(g, demands, resident, l_cost, c_cost, by_id):
    import itertools

    dl = sorted(demands)
    best = float("inf")
    for bits in itertools.product([Decision.SPILL, Decision.RECOMPUTE], repeat=len(dl)):
        dec = dict(zip(dl, bits))
        best = min(best, _cost_of(g, demands, resident, l_cost, c_cost, dec, by_id))
    return best


# ---------------------------------------------------------------------------
# Plan + rewrite + scheduled execution: the headline round-trip.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cid,loss,argf", _MODELS, ids=_IDS)
def test_rewrite_preserves_plain_eval(cid, loss, argf):
    args = argf()
    g = _grad_graph(loss, args)
    base = eval_scheduled(g, *args)[1]
    plan = plan_remat(g, int(base * 0.7))
    g2 = apply_remat_plan(g, plan)
    # markers are value-identity: plain eval_graph is unchanged
    a = _leaves(eval_graph(g, *args))
    b = _leaves(eval_graph(g2, *args))
    assert all(np.allclose(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("cid,loss,argf", _MODELS, ids=_IDS)
def test_scheduled_eval_matches_value_and_grad(cid, loss, argf):
    args = argf()
    g = _grad_graph(loss, args)
    base = eval_scheduled(g, *args)[1]
    plan = plan_remat(g, int(base * 0.7))
    g2 = apply_remat_plan(g, plan)
    ref = _leaves(eval_graph(g, *args))  # (value, grads) the combined graph encodes
    out, hwm = eval_scheduled(g2, *args)
    got = _leaves(out)
    assert len(got) == len(ref)
    assert all(np.allclose(x, y) for x, y in zip(got, ref))
    # the achieved resident high-water never exceeds the planner's (conservative) estimate
    assert hwm <= plan.planned_peak
    if plan.feasible:
        assert hwm <= plan.budget


def test_scheduled_eval_no_plan_is_identity():
    args = (M._init_lstm(_rng(3), vocab=len(M._CHAR_VOCAB)),)
    g = _grad_graph(M.lstm_loss, args)
    ref = _leaves(eval_graph(g, *args))
    out, hwm = eval_scheduled(g, *args)  # no rewrite: pure free-after-last-use
    assert all(np.allclose(x, y) for x, y in zip(_leaves(out), ref))
    assert hwm > 0


def test_plan_reduces_peak_and_flags_feasibility():
    args = (M._init_lstm(_rng(3), vocab=len(M._CHAR_VOCAB)),)
    g = _grad_graph(M.lstm_loss, args)
    # baseline_peak is the interpreter's all-keep high-water (inputs/consts stay resident),
    # which is >= cost_report's peak (that models a plain executor freeing everything).
    base = eval_scheduled(g, *args)[1]
    assert base >= cost_report(g).peak_memory_bytes
    plan = plan_remat(g, int(base * 0.7))
    assert plan.baseline_peak == base
    assert plan.planned_peak < plan.baseline_peak  # eviction helped
    assert plan.recomputed() or plan.spilled()
    # an impossibly tight budget is reported infeasible, not silently violated
    tiny = plan_remat(g, 1)
    assert tiny.feasible is False


def test_spill_path_round_trips_through_disk(tmp_path):
    # A cost model with fast/cheap disk and slow CPU makes reload beat recompute, so the
    # planner chooses SPILL -- exercising SpillStore.put/get on real files.
    args = (M._init_lstm(_rng(3), vocab=len(M._CHAR_VOCAB)),)
    g = _grad_graph(M.lstm_loss, args)
    cheap_disk = CostModel(
        flops_per_sec=1e5,  # slow CPU -> recompute is expensive
        ssd_read_bandwidth=1e12,
        ssd_write_bandwidth=1e12,
        ssd_latency=0.0,  # reload is nearly free
    )
    base = eval_scheduled(g, *args)[1]
    plan = plan_remat(g, int(base * 0.7), cheap_disk)
    assert plan.spilled()  # the model preferred spilling
    g2 = apply_remat_plan(g, plan)
    out, _ = eval_scheduled(g2, *args, store_dir=str(tmp_path))
    ref = _leaves(eval_graph(g, *args))
    assert all(np.allclose(x, y) for x, y in zip(_leaves(out), ref))


def test_exact_no_worse_than_greedy():
    # On a small graph the exact resident-set search must spend no more than the greedy.
    args = (M._init_mlp_tree(_rng(1)),)
    g = _grad_graph(M.mlp_tree_loss, args)
    base = cost_report(g).peak_memory_bytes
    budget = int(base * 0.85)
    greedy = plan_remat(g, budget, exact=False)
    exact = plan_remat(g, budget, exact=True)
    if greedy.feasible and exact.feasible:
        assert exact.added_compute_time <= greedy.added_compute_time + 1e-12


def test_decisions_cover_activations_and_baseline():
    args = (M._init_rnn(_rng(3), vocab=len(M._CHAR_VOCAB)),)
    g = _grad_graph(M.rnn_loss, args)
    plan = plan_remat(g, 10**9)  # unbounded budget: nothing evicted
    assert plan.baseline_peak == eval_scheduled(g, *args)[1]
    assert plan.feasible
    assert not plan.spilled() and not plan.recomputed()
    assert all(d is Decision.KEEP for d in plan.decisions.values())

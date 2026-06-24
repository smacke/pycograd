# -*- coding: utf-8 -*-
"""Tests for the static cost model (:mod:`pycograd.cost`).

The cost model is an *estimate*, so the tests pin the things that must be exact --
the FLOP formulas (matmul ``2MNK``, einsum over the iteration space, movement ops
free), peak-memory liveness, and the disk/budget arithmetic -- and otherwise assert
structural invariants (totals are the sum of parts; every example model costs to a
finite, positive number) rather than wall-clock numbers.
"""
import pytest

np = pytest.importorskip("numpy")

from pycograd import cost_report  # noqa: E402
from pycograd.capture import (  # noqa: E402
    _INPUT,
    Const,
    Graph,
    Node,
    Ref,
    capture,
)
from pycograd.cost import (  # noqa: E402
    CostModel,
    calibrate,
    node_flops,
)
from pycograd.examples import models as M  # noqa: E402
from pycograd.shapes import ShapeDtypeStruct as SDS  # noqa: E402


def _rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Hand-built single-node graphs for exact FLOP assertions.
# ---------------------------------------------------------------------------
def _one_op(prim, in_shapes, out_shape, args=None, params=None):
    """A graph: one input per ``in_shapes`` feeding a single ``prim`` op node."""
    nodes = [Node(i, _INPUT, (), {}, SDS(s)) for i, s in enumerate(in_shapes)]
    oid = len(nodes)
    a = args if args is not None else tuple(Ref(i) for i in range(len(in_shapes)))
    nodes.append(Node(oid, prim, a, params or {}, SDS(out_shape)))
    return Graph(nodes, list(range(len(in_shapes))), [oid], None), nodes[oid]


def test_matmul_flops_2mnk():
    from pycograd.ops import _matmul

    g, node = _one_op(_matmul, [(90, 16), (16, 3)], (90, 3))
    by_id = {nd.id: nd for nd in g.nodes}
    # 2 * M*N * K = 2 * (90*3) * 16
    assert node_flops(node, by_id, CostModel()) == 2 * 90 * 3 * 16


def test_einsum_flops_iteration_space():
    from pycograd.ops import d_einsum

    # ij,jk->ik : the full iteration space is i*j*k, two flops each.
    g, node = _one_op(
        d_einsum,
        [(4, 5), (5, 6)],
        (4, 6),
        args=(Const("ij,jk->ik"), Ref(0), Ref(1)),
    )
    by_id = {nd.id: nd for nd in g.nodes}
    assert node_flops(node, by_id, CostModel()) == 2 * 4 * 5 * 6


def test_movement_ops_are_free():
    from pycograd.ops import d_reshape, d_transpose

    for prim, out in ((d_reshape, (6, 4)), (d_transpose, (4, 6))):
        g, node = _one_op(prim, [(4, 6)], out)
        by_id = {nd.id: nd for nd in g.nodes}
        assert node_flops(node, by_id, CostModel()) == 0


def test_transcendental_costs_more_than_cheap():
    from pycograd.ops import d_add, d_exp

    ge, ne = _one_op(d_exp, [(100,)], (100,))
    ga, na = _one_op(d_add, [(100,), (100,)], (100,))
    fe = node_flops(ne, {nd.id: nd for nd in ge.nodes}, CostModel())
    fa = node_flops(na, {nd.id: nd for nd in ga.nodes}, CostModel())
    assert fe == 100 * 8  # transcendental weight
    assert fa == 100  # cheap weight
    assert fe > fa


def test_reduction_scales_with_input_not_output():
    from pycograd.ops import d_sum

    g, node = _one_op(d_sum, [(90, 16)], (16,), params={"axis": 0})
    by_id = {nd.id: nd for nd in g.nodes}
    assert node_flops(node, by_id, CostModel()) == 90 * 16  # swept the input


def test_symbolic_dim_substituted():
    from pycograd._dims import symbol
    from pycograd.ops import d_exp

    n = symbol("n")
    g, node = _one_op(d_exp, [(n,)], (n,))
    by_id = {nd.id: nd for nd in g.nodes}
    model = CostModel(symbolic_dim_size=32)
    assert node_flops(node, by_id, model) == 32 * 8


# ---------------------------------------------------------------------------
# Peak-memory liveness.
# ---------------------------------------------------------------------------
def test_peak_memory_chain_frees_intermediates():
    # A linear chain x -> a -> b -> out: at most two f64[100] live at once
    # (the producer plus the consumer it feeds), never all three.
    from pycograd.ops import d_exp

    nodes = [
        Node(0, _INPUT, (), {}, SDS((100,))),
        Node(1, d_exp, (Ref(0),), {}, SDS((100,))),
        Node(2, d_exp, (Ref(1),), {}, SDS((100,))),
    ]
    g = Graph(nodes, [0], [2], None)
    rep = cost_report(g, CostModel())
    elem = 100 * 8
    assert rep.peak_memory_bytes == 2 * elem


def test_peak_memory_fanout_keeps_source_live():
    # x feeds two consumers a, b, and out = a + b. x must stay live until the add,
    # so at the add three buffers (x, a, b) overlap before out is allocated.
    from pycograd.ops import d_add, d_exp, d_log

    nodes = [
        Node(0, _INPUT, (), {}, SDS((100,))),
        Node(1, d_exp, (Ref(0),), {}, SDS((100,))),
        Node(2, d_log, (Ref(0),), {}, SDS((100,))),
        Node(3, d_add, (Ref(1), Ref(2)), {}, SDS((100,))),
    ]
    g = Graph(nodes, [0], [3], None)
    rep = cost_report(g, CostModel())
    elem = 100 * 8
    # x still live (last use is the add at step 2) while a and b coexist.
    assert rep.peak_memory_bytes == 3 * elem
    assert rep.peak_memory_node == 2


# ---------------------------------------------------------------------------
# Disk / budget arithmetic.
# ---------------------------------------------------------------------------
def test_spill_time_round_trip():
    m = CostModel(ssd_write_bandwidth=1e9, ssd_read_bandwidth=2e9, ssd_latency=1e-4)
    nbytes = 1_000_000
    expected = (nbytes / 1e9 + 1e-4) + (nbytes / 2e9 + 1e-4)
    assert m.spill_time(nbytes) == pytest.approx(expected)


def test_over_budget_flag():
    from pycograd.ops import d_exp

    nodes = [
        Node(0, _INPUT, (), {}, SDS((1000,))),
        Node(1, d_exp, (Ref(0),), {}, SDS((1000,))),
    ]
    g = Graph(nodes, [0], [1], None)
    assert cost_report(g, CostModel(ram_capacity=None)).over_budget is False
    assert cost_report(g, CostModel(ram_capacity=100)).over_budget is True
    assert cost_report(g, CostModel(ram_capacity=10**9)).over_budget is False


def test_spill_candidates_prefers_big_cheap_to_recompute():
    # A huge transpose (free to recompute? no -- movement is 0 flops but the round-trip
    # spill of a large buffer can still lose to recompute) vs a costed op. Use a model
    # with cheap disk so spilling a large intermediate beats recomputing a pricey one.
    big = 1 << 20  # 1M elems
    nodes = [
        Node(0, _INPUT, (), {}, SDS((big,))),
        Node(1, _from_exp(), (Ref(0),), {}, SDS((big,))),  # pricey to recompute
        Node(2, _from_sum(), (Ref(1),), {"axis": None}, SDS(())),
    ]
    g = Graph(nodes, [0], [2], None)
    # Fast disk, slow CPU -> spilling the big exp output beats recomputing it.
    m = CostModel(
        flops_per_sec=1e6,
        ssd_write_bandwidth=1e12,
        ssd_read_bandwidth=1e12,
        ssd_latency=0.0,
    )
    rep = cost_report(g, m)
    cands = rep.spill_candidates(5)
    assert any(nc.id == 1 for nc in cands)


def _from_exp():
    from pycograd.ops import d_exp

    return d_exp


def _from_sum():
    from pycograd.ops import d_sum

    return d_sum


# ---------------------------------------------------------------------------
# Whole-graph invariants + every example model.
# ---------------------------------------------------------------------------
_CASES = [
    ("mlp_tree", M.mlp_tree_loss, lambda: (M._init_mlp_tree(_rng(1)),)),
    ("rnn", M.rnn_loss, lambda: (M._init_rnn(_rng(3), vocab=len(M._CHAR_VOCAB)),)),
    ("lstm", M.lstm_loss, lambda: (M._init_lstm(_rng(3), vocab=len(M._CHAR_VOCAB)),)),
]


@pytest.mark.parametrize("cid,loss,argf", _CASES, ids=[c[0] for c in _CASES])
def test_example_models_cost_to_positive_finite(cid, loss, argf):
    g = capture(loss, *argf())
    rep = g.cost()
    assert rep.total_flops > 0
    assert rep.total_compute_time > 0
    assert rep.peak_memory_bytes > 0
    # the reported total time is the sum of the per-node roofline times
    assert rep.total_compute_time == pytest.approx(sum(nc.time for nc in rep.nodes))
    # the total flops is the sum of the per-node flops
    assert rep.total_flops == sum(nc.flops for nc in rep.nodes)
    # input / const nodes carry no compute
    for nc in rep.nodes:
        if nc.prim in ("input", "const"):
            assert nc.flops == 0


def test_pretty_cost_lists_ops_and_totals():
    g = capture(M.mlp_tree_loss, M._init_mlp_tree(_rng(1)))
    text = str(g.cost())
    assert text.startswith("cost {")
    assert "total:" in text
    assert "peak mem" in text


def test_calibrate_returns_positive_constants():
    m = calibrate()  # CPU + memory only (no disk_path)
    assert m.flops_per_sec > 0
    assert m.mem_bandwidth > 0
    # disk fields fall back to the base defaults when not probed
    assert m.ssd_read_bandwidth == CostModel().ssd_read_bandwidth

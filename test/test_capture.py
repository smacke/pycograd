# -*- coding: utf-8 -*-
"""Tests for the graph-capture IR (:mod:`pycograd.capture`).

The contract: ``eval_graph(capture(f, *args), *args)`` reproduces ``f(*args)`` --
value *and*, differentiated through, gradient -- on the finite-diff-checked example
models (the "don't regress" bar). Later phases add optimization passes; each must
preserve this round-trip.
"""
import pytest

np = pytest.importorskip("numpy")

from pycograd import d_sigmoid, ops, value_and_grad  # noqa: E402
from pycograd.capture import (  # noqa: E402
    _CONST,
    _INPUT,
    Graph,
    Node,
    Ref,
    capture,
    eval_graph,
)
from pycograd.examples import models as M  # noqa: E402
from pycograd.passes import (  # noqa: E402
    algebraic,
    constant_fold,
    cse,
    fuse_gated_act,
    optimize,
)
from pycograd.shapes import ShapeDtypeStruct  # noqa: E402
from pycograd.tensor import _value  # noqa: E402
from pycograd.tree import tree_flatten, tree_leaves  # noqa: E402


def _rng(seed):
    return np.random.default_rng(seed)


# Deterministic example losses (no dropout, so a fresh call equals the replay).
# (id, loss_fn, args_factory)
_CASES = [
    ("mlp_tree", M.mlp_tree_loss, lambda: (M._init_mlp_tree(_rng(1)),)),
    ("rnn", M.rnn_loss, lambda: (M._init_rnn(_rng(3), vocab=len(M._CHAR_VOCAB)),)),
    ("gru", M.gru_loss, lambda: (M._init_gru(_rng(3), vocab=len(M._CHAR_VOCAB)),)),
    ("lstm", M.lstm_loss, lambda: (M._init_lstm(_rng(3), vocab=len(M._CHAR_VOCAB)),)),
    (
        "rwkv",
        M.rwkv_loss,
        lambda: M._init_rwkv(_rng(2), vocab=12, d_model=8, n_blocks=2),
    ),
]
_IDS = [c[0] for c in _CASES]


@pytest.mark.parametrize("cid,loss,argf", _CASES, ids=_IDS)
def test_capture_value_roundtrip(cid, loss, argf):
    args = argf()
    graph = capture(loss, *args)
    got = eval_graph(graph, *args)
    ref = loss(*args)
    assert np.allclose(float(_value(got)), float(_value(ref)), atol=1e-10)


@pytest.mark.parametrize("cid,loss,argf", _CASES, ids=_IDS)
def test_capture_grad_roundtrip(cid, loss, argf):
    args = argf()
    graph = capture(loss, *args)

    def replay(*a):
        return eval_graph(graph, *a)

    # Run the replay directly (not instrumented) so the captured-graph closure
    # survives and eval_graph drives its own bind dispatch.
    replay._pycograd_run_directly = True

    _, grads_replay = value_and_grad(replay)(*args)
    _, grads_ref = value_and_grad(loss)(*args)
    lr = [np.asarray(g) for arg in grads_replay for g in tree_leaves(arg)]
    lf = [np.asarray(g) for arg in grads_ref for g in tree_leaves(arg)]
    assert lr and len(lr) == len(lf)
    for a, b in zip(lr, lf):
        assert np.allclose(a, b, atol=1e-9)


def test_capture_records_a_graph():
    # Sanity on the IR structure: inputs are _INPUT nodes, ops are recorded, the
    # output references a real node, and constants inline as Const arg specs.
    args = (M._init_mlp_tree(_rng(1)),)
    graph = capture(M.mlp_tree_loss, *args)
    assert len(graph.inputs) == 4  # the 4 mlp leaves
    assert all(graph.nodes[i].prim is _INPUT for i in graph.inputs)
    n_ops = sum(1 for nd in graph.nodes if nd.prim not in (_INPUT, _CONST))
    assert n_ops > 0
    assert len(graph.outputs) == 1


# --- D2: passes -------------------------------------------------------------
def _n_ops(graph):
    return sum(1 for nd in graph.nodes if nd.prim not in (_INPUT, _CONST))


def _redundant_fn(x):
    a = np.tanh(x)
    b = np.tanh(x)  # identical to a -> CSE merges
    dead = np.exp(x)  # noqa: F841  -- unused -> DCE removes
    return np.sum(a * b)


def test_cse_merges_and_dce_drops():
    x = _rng(0).standard_normal((3, 4))
    g = capture(_redundant_fn, x)
    base = _n_ops(g)  # tanh, tanh, exp, mul, sum
    assert _n_ops(cse(g)) == base - 1  # the duplicate tanh merged
    g_opt = optimize(g)
    assert _n_ops(g_opt) == base - 2  # duplicate tanh merged + dead exp removed
    assert np.allclose(
        float(_value(eval_graph(g_opt, x))), float(_value(_redundant_fn(x)))
    )


def test_constant_fold_collapses_const_subgraph():
    # Hand-built graph: add(const 2, const 3) folds to a single const 5.
    sds = ShapeDtypeStruct((), np.dtype("float64"))
    _, td = tree_flatten(np.array(0.0))
    nodes = [
        Node(0, _CONST, (), {"value": np.array(2.0)}, sds),
        Node(1, _CONST, (), {"value": np.array(3.0)}, sds),
        Node(2, ops.d_add, (Ref(0), Ref(1)), {}, sds),
    ]
    g = Graph(nodes, inputs=[], outputs=[2], out_treedef=td)
    folded = constant_fold(g)
    assert folded.nodes[2].prim is _CONST
    assert np.allclose(float(_value(eval_graph(g))), 5.0)
    assert np.allclose(float(_value(eval_graph(folded))), 5.0)


@pytest.mark.parametrize("cid,loss,argf", _CASES, ids=_IDS)
def test_optimize_preserves_semantics(cid, loss, argf):
    args = argf()
    g = optimize(capture(loss, *args))
    assert np.allclose(
        float(_value(eval_graph(g, *args))), float(_value(loss(*args))), atol=1e-9
    )

    def replay(*a):
        return eval_graph(g, *a)

    replay._pycograd_run_directly = True
    _, grads_replay = value_and_grad(replay)(*args)
    _, grads_ref = value_and_grad(loss)(*args)
    lr = [np.asarray(x) for arg in grads_replay for x in tree_leaves(arg)]
    lf = [np.asarray(x) for arg in grads_ref for x in tree_leaves(arg)]
    for a, b in zip(lr, lf):
        assert np.allclose(a, b, atol=1e-9)


# --- D3: algebraic simplification + fusion ----------------------------------
def _alg_fn(x):
    return np.sum(x * 1.0 + x * 0.0)  # x*1 -> x, x*0 -> zeros


def test_algebraic_simplifies_identities():
    x = _rng(0).standard_normal((3, 4))
    g = capture(_alg_fn, x)
    g2 = algebraic(g)
    assert _n_ops(g2) < _n_ops(g)
    assert np.allclose(
        float(_value(eval_graph(g2, x))), float(_value(_alg_fn(x))), atol=1e-12
    )


def _explicit_gate_fn(x):
    # Uses the d_sigmoid primitive (no np.sigmoid exists) so the graph carries a
    # genuine d_sigmoid node for the fusion pass to match against d_tanh.
    return np.sum(np.tanh(x) * d_sigmoid(x))


def test_fuse_gated_act_rewrites_tanh_times_sigmoid():
    x = _rng(1).standard_normal((3, 4))
    g = capture(_explicit_gate_fn, x)
    g2 = fuse_gated_act(g)
    prims = [nd.prim for nd in g2.nodes]
    assert ops.d_gated_act in prims
    assert ops.d_tanh not in prims  # fused away
    assert ops.d_sigmoid not in prims
    assert ops.d_mul not in prims
    # still evaluates equal to a composed-sigmoid reference (gate == tanh * sigmoid)
    ref = float(np.sum(np.tanh(x) * (1.0 / (1.0 + np.exp(-x)))))
    assert np.allclose(float(_value(eval_graph(g2, x))), ref, atol=1e-12)


def test_fusion_skips_shared_subexpr():
    # If the sigmoid feeds something else too, the fuse must not fire (use-count > 1).
    def shared(x):
        s = d_sigmoid(x)
        return np.sum(np.tanh(x) * s + s)

    x = _rng(2).standard_normal((2, 3))
    g2 = fuse_gated_act(capture(shared, x))
    assert ops.d_gated_act not in [nd.prim for nd in g2.nodes]

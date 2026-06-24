# -*- coding: utf-8 -*-
"""Tests for the graph-capture IR (:mod:`pycograd.capture`).

The contract: ``eval_graph(capture(f, *args), *args)`` reproduces ``f(*args)`` --
value *and*, differentiated through, gradient -- on the finite-diff-checked example
models (the "don't regress" bar). Later phases add optimization passes; each must
preserve this round-trip.
"""
import pytest

np = pytest.importorskip("numpy")

from pycograd import (  # noqa: E402
    d_logsumexp,
    d_sigmoid,
    d_softmax,
    ops,
    value_and_grad,
)
from pycograd.ad_graph import grad_graph  # noqa: E402
from pycograd.capture import (  # noqa: E402
    _CONST,
    _INPUT,
    Const,
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
    fuse_logsumexp,
    fuse_softmax,
    optimize,
    reorder_matmul_chain,
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


# An ambient-DSL forward reading weights injected by `with weights:` -- module-level so the
# instrumented capture runner (recompiled from source) sees them as globals.
def _dsl_weight_forward(x):
    return np.sum(np.tanh(x @ cew + ceb))  # noqa: F821  (ambient weights)


def test_capture_composes_with_ambient_weights():
    from pycograd.params import Weight, params

    weights = params(cew=_rng(2).standard_normal((3, 2)), ceb=np.zeros(2))
    X = _rng(1).standard_normal((4, 3))
    with weights:
        g = capture(_dsl_weight_forward, X)
        ref = _dsl_weight_forward(
            X
        )  # eager reference (ambient weights resolve to numpy)

    # The weights are sized by their real shapes, not () -- so x @ cew is (4, 3) @ (3, 2).
    assert any(nd.prim is ops._matmul and nd.aval.shape == (4, 2) for nd in g.nodes)
    # ...and captured as concrete-array constants (a snapshot), not live Weight proxies.
    const_vals = [a.value for nd in g.nodes for a in nd.args if isinstance(a, Const)]
    assert any(
        getattr(c, "shape", None) == (3, 2) for c in const_vals
    )  # cew, snapshotted
    assert not any(isinstance(c, Weight) for c in const_vals)
    # Because it's a snapshot, the graph evaluates OUTSIDE the `with` block and matches eager.
    assert np.allclose(float(_value(eval_graph(g, X))), float(_value(ref)), atol=1e-9)


def test_graph_pretty_listing():
    args = (M._init_mlp_tree(_rng(1)),)
    g = capture(M.mlp_tree_loss, *args)
    s = g.pretty()
    assert s.startswith("graph(") and s.rstrip().endswith("}")
    assert "outputs:" in s
    assert "matmul" in s and "maximum" in s  # relu is np.maximum(x, 0)
    assert "-> f64[]" in s  # the scalar loss output
    assert "softmax" in s  # the fused stable-softmax node (functional.softmax)
    assert "{axis=1}" in s  # params shown on a node (softmax / the loss reduction)
    # every op node appears as "%id = ..."
    for nd in g.nodes:
        if nd.prim not in (_INPUT, _CONST):
            assert f"%{nd.id} =" in s
    assert str(g) == s  # print(graph) shows the listing...
    assert repr(g).startswith("Graph(") and "ops" in repr(g)  # ...repr stays terse


def test_graph_to_dot():
    args = (M._init_mlp_tree(_rng(1)),)
    g = capture(M.mlp_tree_loss, *args)
    dot = g.to_dot()
    assert dot.startswith("digraph G {") and dot.rstrip().endswith("}")
    assert "->" in dot  # has edges
    assert dot.count("->") >= _n_ops(g)  # at least one edge feeding each op
    assert "peripheries=2" in dot  # the output node is marked
    assert "fillcolor=lightblue" in dot  # inputs are drawn distinctly
    # every node id is declared exactly once
    for nd in g.nodes:
        assert f"  {nd.id} [" in dot


def test_grad_graph_pretty_lists_value_and_grads():
    gg = grad_graph(capture(M.mlp_tree_loss, *(M._init_mlp_tree(_rng(1)),)))
    s = gg.pretty()
    assert s.startswith("graph(") and "outputs:" in s
    out_line = next(ln for ln in s.splitlines() if ln.strip().startswith("outputs:"))
    assert all(f"%{o}" in out_line for o in gg.outputs)  # value + every grad listed


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


def _alg_shape_fn(x):
    a = np.reshape(x, x.shape)  # reshape to the same shape -> x
    b = x + (x * 0.0)  # x*0 -> zeros array; x + zeros (matching shape) -> x
    return np.sum(a + b)


def test_algebraic_shape_aware_identities():
    x = _rng(7).standard_normal((3, 4))
    g = capture(_alg_shape_fn, x)
    g2 = algebraic(g)
    assert _n_ops(g2) < _n_ops(g)  # reshape-to-same and the array identity both drop
    assert np.allclose(
        float(_value(eval_graph(g2, x))), float(_value(_alg_shape_fn(x))), atol=1e-12
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


# --- D4: the backward benefits from forward fusion --------------------------
# Capturing value_and_grad(f) into one graph does NOT compose -- pycograd's reverse
# pass is not bind-expressed at the base level (the base-vs-higher-order split), so a
# trace cannot record it. But fusing the *forward* graph still improves the backward:
# differentiating the optimized graph through eval_graph runs the *fused* primitive's
# VJP (one backward op for d_gated_act instead of three for tanh/sigmoid/mul).
def _gate_loss(x):
    return np.sum(np.tanh(x) * d_sigmoid(x))


def test_optimized_forward_gives_a_fused_backward():
    x = _rng(5).standard_normal((3, 4))
    g = optimize(capture(_gate_loss, x))  # fuses tanh*sigmoid -> d_gated_act
    assert ops.d_gated_act in [nd.prim for nd in g.nodes]

    def replay(a):
        return eval_graph(g, a)

    replay._pycograd_run_directly = True
    # Gradient through the fused graph uses d_gated_act's VJP; check vs the analytic
    # derivative of sum(tanh(x) * sigmoid(x)).
    _, (grad_fused,) = value_and_grad(replay)(x)
    s = 1.0 / (1.0 + np.exp(-x))
    t = np.tanh(x)
    ref = s * (1 - t * t) + t * s * (1 - s)
    assert np.allclose(np.asarray(grad_fused), ref, atol=1e-9)


# --- stable softmax / logsumexp fusion --------------------------------------
def _grads_match_after(g, fn, args, atol=1e-9):
    """Replay-vs-reference value and gradient check for an optimized/rewritten graph."""

    def replay(*a):
        return eval_graph(g, *a)

    replay._pycograd_run_directly = True
    assert np.allclose(
        float(_value(eval_graph(g, *args))), float(_value(fn(*args))), atol=atol
    )
    _, gr = value_and_grad(replay)(*args)
    _, gf = value_and_grad(fn)(*args)
    lr = [np.asarray(x) for arg in gr for x in tree_leaves(arg)]
    lf = [np.asarray(x) for arg in gf for x in tree_leaves(arg)]
    assert lr and len(lr) == len(lf)
    for a, b in zip(lr, lf):
        assert np.allclose(a, b, atol=atol)


def _naive_softmax_fn(x):
    e = np.exp(x)
    sm = e / np.sum(e, axis=-1, keepdims=True)  # naive softmax (inline)
    return np.sum(sm * sm)


def _naive_logsumexp_fn(x):
    return np.sum(np.log(np.sum(np.exp(x), axis=-1)))  # naive log-sum-exp (inline)


def _stable_softmax_fn(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)  # stable, max-shifted softmax (inline)
    sm = e / np.sum(e, axis=-1, keepdims=True)
    return np.sum(sm * sm)


def test_fuse_softmax_rewrites_exp_over_sum():
    x = _rng(1).standard_normal((4, 5))
    g2 = fuse_softmax(cse(capture(_naive_softmax_fn, x)))
    prims = [nd.prim for nd in g2.nodes]
    assert d_softmax in prims
    assert ops.d_div not in prims  # the exp/sum/div cluster fused away
    assert ops.d_exp not in prims
    _grads_match_after(g2, _naive_softmax_fn, (x,))


def test_fuse_softmax_handles_stable_max_shifted_form():
    # Shift-invariance: exp(x-m)/sum(exp(x-m)) fuses to d_softmax(x-m) == softmax(x).
    x = _rng(3).standard_normal((4, 5))
    g2 = fuse_softmax(cse(capture(_stable_softmax_fn, x)))
    assert d_softmax in [nd.prim for nd in g2.nodes]
    _grads_match_after(g2, _stable_softmax_fn, (x,))


def test_fuse_logsumexp_rewrites_log_sum_exp():
    x = _rng(2).standard_normal((4, 5))
    g2 = fuse_logsumexp(capture(_naive_logsumexp_fn, x))
    prims = [nd.prim for nd in g2.nodes]
    assert d_logsumexp in prims
    assert ops.d_log not in prims  # log/sum/exp triple fused away
    assert ops.d_exp not in prims
    _grads_match_after(g2, _naive_logsumexp_fn, (x,))


# --- matmul-chain reordering ------------------------------------------------
def _n_matmul(graph):
    return sum(1 for nd in graph.nodes if nd.prim is ops._matmul)


def _chain2d_fn(x, w1, w2, w3):
    return np.sum(((x @ w1) @ w2) @ w3)  # left-assoc; reorder cuts FLOPs


def _chain_batched_fn(q, k, v):
    return np.sum((q @ k) @ v)  # batched chain (leading batch dim)


def test_reorder_matmul_chain_preserves_count_value_and_grad():
    # Shapes chosen so right-leaning association is far cheaper than left-assoc.
    args = (
        _rng(0).standard_normal((100, 5)),
        _rng(1).standard_normal((5, 80)),
        _rng(2).standard_normal((80, 4)),
        _rng(3).standard_normal((4, 60)),
    )
    g = capture(_chain2d_fn, *args)
    gr = reorder_matmul_chain(g)
    assert _n_matmul(gr) == _n_matmul(g) == 3  # reassociation keeps the matmul count
    # the reordered chain materializes a smaller intermediate (w1@w2 is 5x4)
    assert any(tuple(nd.aval.shape) == (5, 4) for nd in gr.nodes)
    _grads_match_after(gr, _chain2d_fn, args, atol=1e-7)


def test_reorder_matmul_chain_is_idempotent_at_optimum():
    args = (
        _rng(0).standard_normal((100, 5)),
        _rng(1).standard_normal((5, 80)),
        _rng(2).standard_normal((80, 4)),
        _rng(3).standard_normal((4, 60)),
    )
    gr = reorder_matmul_chain(capture(_chain2d_fn, *args))
    gr2 = reorder_matmul_chain(gr)
    assert [nd.id for nd in gr2.nodes] == [nd.id for nd in gr.nodes]


def test_reorder_matmul_chain_batched():
    args = (
        _rng(0).standard_normal((8, 3, 5)),
        _rng(1).standard_normal((8, 5, 7)),
        _rng(2).standard_normal((8, 7, 2)),
    )
    g = capture(_chain_batched_fn, *args)
    gr = reorder_matmul_chain(g)
    assert _n_matmul(gr) == _n_matmul(g) == 2
    _grads_match_after(gr, _chain_batched_fn, args, atol=1e-7)

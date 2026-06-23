# -*- coding: utf-8 -*-
"""Tests for the graph-capture IR (:mod:`pycograd.capture`).

The contract: ``eval_graph(capture(f, *args), *args)`` reproduces ``f(*args)`` --
value *and*, differentiated through, gradient -- on the finite-diff-checked example
models (the "don't regress" bar). Later phases add optimization passes; each must
preserve this round-trip.
"""
import pytest

np = pytest.importorskip("numpy")

from pycograd import value_and_grad  # noqa: E402
from pycograd.capture import _CONST, _INPUT, capture, eval_graph  # noqa: E402
from pycograd.examples import models as M  # noqa: E402
from pycograd.tensor import _value  # noqa: E402
from pycograd.tree import tree_leaves  # noqa: E402


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

# -*- coding: utf-8 -*-
"""Spike + tests for linearize/transpose. The linchpin: does ``jvp`` compose *under*
``capture`` (jvp inner, capture outer), so the tangent computation gets recorded as a
linear graph?"""
import pytest

np = pytest.importorskip("numpy")

from pycograd import jvp, value_and_grad  # noqa: E402
from pycograd.capture import capture, eval_graph  # noqa: E402
from pycograd.examples import models as M  # noqa: E402
from pycograd.tensor import _value  # noqa: E402
from pycograd.transpose import linearize, vjp_graph  # noqa: E402
from pycograd.tree import tree_leaves  # noqa: E402


def _smooth(x):
    return np.sum(np.tanh(x) * np.tanh(x))


def _tangent_of(x, t):
    # the tangent output of jvp -- a function of (primal x, tangent t), LINEAR in t.
    _, tangent_out = jvp(_smooth, (x,), (t,))
    return tangent_out


def test_jvp_under_capture_spike():
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((3, 4))
    t0 = rng.standard_normal((3, 4))

    # linearize: capture the tangent computation as a graph in (x, t).
    lin = capture(_tangent_of, x0, t0)

    # at fixed x0, evaluating the graph at t0 matches jvp's tangent.
    got = np.asarray(_value(eval_graph(lin, x0, t0)))
    _, ref = jvp(_smooth, (x0,), (t0,))
    assert np.allclose(got, np.asarray(_value(ref))), "graph tangent != jvp tangent"

    # ...and it is LINEAR in t (the property transpose relies on).
    got2 = np.asarray(_value(eval_graph(lin, x0, 2.0 * t0)))
    assert np.allclose(got2, 2.0 * got), "tangent graph is not linear in t"


def test_linearize_matches_jvp():
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((3, 4))
    t0 = rng.standard_normal((3, 4))
    graph, n_primal = linearize(_smooth, x0)
    assert n_primal == 1  # one primal leaf

    po, to = eval_graph(graph, (x0,), (t0,))
    rpo, rto = jvp(_smooth, (x0,), (t0,))
    assert np.allclose(float(_value(po)), float(_value(rpo)))  # primal value
    assert np.allclose(float(_value(to)), float(_value(rto)))  # tangent at t0

    _, to2 = eval_graph(graph, (x0,), (2.0 * t0,))
    assert np.allclose(float(_value(to2)), 2.0 * float(_value(to)))  # linear in t


def test_vjp_graph_matches_value_and_grad():
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal((3, 4))
    g = vjp_graph(_smooth, x0)
    val, grads = eval_graph(g, x0)
    rval, (rg,) = value_and_grad(_smooth)(x0)
    assert np.allclose(float(_value(val)), float(_value(rval)))
    assert np.allclose(np.asarray(_value(grads[0])), np.asarray(rg), atol=1e-9)


def _rng(seed):
    return np.random.default_rng(seed)


_MODELS = [
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


@pytest.mark.parametrize("cid,loss,argf", _MODELS, ids=[c[0] for c in _MODELS])
def test_vjp_graph_models_match_value_and_grad(cid, loss, argf):
    args = argf()
    g = vjp_graph(loss, *args)
    val, grads = eval_graph(g, *args)
    rval, rgrads = value_and_grad(loss)(*args)
    assert np.allclose(float(_value(val)), float(_value(rval)), atol=1e-9)
    got = [np.asarray(_value(x)) for x in grads]
    ref = [np.asarray(x) for arg in rgrads for x in tree_leaves(arg)]
    assert len(got) == len(ref) and got
    for a, b in zip(got, ref):
        assert np.allclose(a, b, atol=1e-9), (cid, a, b)

# -*- coding: utf-8 -*-
"""Tests for ``grad_graph`` (autodiff on the capture IR): differentiating a captured
forward graph yields one graph computing value + gradients, matching ``value_and_grad``.
G1 covers the smooth/linear ops; G2 the mask ops (relu/softmax); G3 the cross-pass CSE.
"""
import pytest

np = pytest.importorskip("numpy")

from pycograd import d_sigmoid, jit, ops, value_and_grad  # noqa: E402
from pycograd.ad_graph import grad_graph  # noqa: E402
from pycograd.capture import capture, eval_graph  # noqa: E402
from pycograd.examples import models as M  # noqa: E402
from pycograd.passes import optimize  # noqa: E402
from pycograd.tensor import _value  # noqa: E402
from pycograd.tree import tree_leaves  # noqa: E402


def _rng(seed):
    return np.random.default_rng(seed)


def _grads_match(gg, args, loss_fn, atol=1e-9):
    val, grads = eval_graph(gg, *args)
    ref_val, ref_grads = value_and_grad(loss_fn)(*args)
    assert np.allclose(float(_value(val)), float(_value(ref_val)), atol=atol)
    got = [np.asarray(_value(x)) for x in grads]
    ref = [np.asarray(x) for arg in ref_grads for x in tree_leaves(arg)]
    assert len(got) == len(ref) and got, (len(got), len(ref))
    for a, b in zip(got, ref):
        assert np.allclose(a, b, atol=atol), (a, b)


# --- G1: smooth / linear ops ------------------------------------------------
def _smooth_loss(x, w):
    h = np.tanh(x @ w)  # matmul + tanh
    return np.sum(h * h)  # mul + sum


def test_grad_graph_smooth_roundtrip():
    x, w = _rng(0).standard_normal((4, 3)), _rng(1).standard_normal((3, 2))
    _grads_match(grad_graph(capture(_smooth_loss, x, w)), (x, w), _smooth_loss)


def _reshape_einsum_loss(x, w):
    y = np.einsum("ij,jk->ik", x, w)  # einsum
    z = np.reshape(y, (y.shape[0] * y.shape[1],))  # reshape
    return np.sum(np.exp(z))  # exp + sum


def test_grad_graph_reshape_einsum_roundtrip():
    x, w = _rng(2).standard_normal((3, 4)), _rng(3).standard_normal((4, 2))
    _grads_match(
        grad_graph(capture(_reshape_einsum_loss, x, w)), (x, w), _reshape_einsum_loss
    )


# --- G2: mask ops (relu / max-reduce / abs / pow) ---------------------------
def _relu_softmax_loss(x, w):
    h = np.maximum(x @ w, 0.0)  # relu -> d_maximum (select mask)
    m = np.max(h, axis=1, keepdims=True)  # d_max (reduce-select mask)
    e = np.exp(h - m)
    p = e / np.sum(e, axis=1, keepdims=True)
    return np.sum(p * p)


def test_grad_graph_relu_softmax_roundtrip():
    x, w = _rng(4).standard_normal((4, 5)), _rng(5).standard_normal((5, 3))
    _grads_match(
        grad_graph(capture(_relu_softmax_loss, x, w)), (x, w), _relu_softmax_loss
    )


def _abs_pow_loss(x):
    return np.sum(np.abs(x) + x**2 * 0.5)  # abs (sign mask) + pow (const exponent)


def test_grad_graph_abs_pow_roundtrip():
    x = _rng(6).standard_normal((3, 4))
    _grads_match(grad_graph(capture(_abs_pow_loss, x)), (x,), _abs_pow_loss)


# --- G2: the full example models (the finite-diff-checked "don't regress" bar) ---
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
def test_grad_graph_models_roundtrip(cid, loss, argf):
    args = argf()
    _grads_match(grad_graph(capture(loss, *args)), args, loss)


# --- G3: cross-pass optimization (CSE across the forward/backward boundary) --
def _n_prim(graph, prim):
    return sum(1 for nd in graph.nodes if nd.prim is prim)


def _sigmoid_loss(x):
    # sigmoid's VJP is g * sigmoid(x) * (1 - sigmoid(x)) -- it *recomputes* sigmoid(x),
    # so the combined graph has the forward's sigmoid(x) plus two in the backward.
    return np.sum(d_sigmoid(x))


def test_cross_pass_cse_merges_recomputed_sigmoid():
    x = _rng(7).standard_normal((3, 4))
    combined = grad_graph(capture(_sigmoid_loss, x))
    before = _n_prim(combined, ops.d_sigmoid)
    opt = optimize(combined)
    after = _n_prim(opt, ops.d_sigmoid)
    assert before >= 3  # 1 forward + 2 recomputed in the VJP
    assert after == 1  # CSE merged them across the forward/backward boundary
    # ...and it still computes the right gradient.
    _, (gx,) = eval_graph(opt, x)
    s = 1.0 / (1.0 + np.exp(-x))
    assert np.allclose(np.asarray(_value(gx)), s * (1 - s), atol=1e-9)


def test_optimize_preserves_grad_graph_semantics():
    # optimize() over the combined forward+backward graph must keep value AND grads.
    args = (M._init_mlp_tree(_rng(1)),)
    opt = optimize(grad_graph(capture(M.mlp_tree_loss, *args)))
    _grads_match(opt, args, M.mlp_tree_loss)


# --- the jit entrypoint -----------------------------------------------------
def test_jit_forward_matches_eager():
    args = (M._init_mlp_tree(_rng(1)),)
    fast = jit(M.mlp_tree_loss)
    ref = float(_value(M.mlp_tree_loss(*args)))
    assert np.allclose(float(_value(fast(*args))), ref)
    assert np.allclose(float(_value(fast(*args))), ref)  # second call (cache reuse)


def test_jit_grad_matches_value_and_grad():
    args = (M._init_mlp_tree(_rng(1)),)
    val, grads = jit(M.mlp_tree_loss, grad=True)(*args)
    rval, rgrads = value_and_grad(M.mlp_tree_loss)(*args)
    assert np.allclose(float(_value(val)), float(_value(rval)))
    gl = [np.asarray(_value(x)) for arg in grads for x in tree_leaves(arg)]
    rl = [np.asarray(x) for arg in rgrads for x in tree_leaves(arg)]
    assert len(gl) == len(rl) and gl
    for a, b in zip(gl, rl):
        assert np.allclose(a, b, atol=1e-9)


def _dynamic_fn(x):
    if float(np.sum(x)) > 0.0:  # data-dependent branch -> can't be captured
        return np.sum(x * x)
    return np.sum(x)


def test_jit_falls_back_to_eager_on_dynamic_control_flow():
    x = _rng(0).standard_normal((4,))
    got = jit(_dynamic_fn)(x)
    assert np.allclose(float(_value(got)), float(_value(_dynamic_fn(x))))

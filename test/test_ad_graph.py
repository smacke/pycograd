# -*- coding: utf-8 -*-
"""Tests for autodiff on the capture IR (``value_and_grad`` / ``grad`` of a captured
:class:`Graph`): differentiating a captured forward graph yields one graph computing
value + gradients (or grads alone), matching the callable ``value_and_grad`` / ``grad``.
G1 covers the smooth/linear ops; G2 the mask ops (relu/softmax); G3 the cross-pass CSE.
"""
import pytest

np = pytest.importorskip("numpy")

from pycograd import d_sigmoid, grad, jit, ops, value_and_grad  # noqa: E402
from pycograd.capture import capture, eval_graph  # noqa: E402
from pycograd.examples import models as M  # noqa: E402
from pycograd.params import ParamDict, frozen, params, tied  # noqa: E402
from pycograd.passes import optimize  # noqa: E402
from pycograd.tensor import _value  # noqa: E402
from pycograd.tree import tree_leaves  # noqa: E402


def _rng(seed):
    return np.random.default_rng(seed)


def _grads_match(gg, args, loss_fn, atol=1e-9):
    val, grads = eval_graph(gg, *args)
    ref_val, ref_grads = value_and_grad(loss_fn)(*args)
    assert np.allclose(float(_value(val)), float(_value(ref_val)), atol=atol)
    # ``grads`` is a tuple of per-argument pytrees (matching ``value_and_grad``); flatten
    # both to compare leaf-for-leaf.
    got = [np.asarray(_value(x)) for arg in grads for x in tree_leaves(arg)]
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
    _grads_match(value_and_grad(capture(_smooth_loss, x, w)), (x, w), _smooth_loss)


def _reshape_einsum_loss(x, w):
    y = np.einsum("ij,jk->ik", x, w)  # einsum
    z = np.reshape(y, (y.shape[0] * y.shape[1],))  # reshape
    return np.sum(np.exp(z))  # exp + sum


def test_grad_graph_reshape_einsum_roundtrip():
    x, w = _rng(2).standard_normal((3, 4)), _rng(3).standard_normal((4, 2))
    _grads_match(
        value_and_grad(capture(_reshape_einsum_loss, x, w)),
        (x, w),
        _reshape_einsum_loss,
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
        value_and_grad(capture(_relu_softmax_loss, x, w)), (x, w), _relu_softmax_loss
    )


def _abs_pow_loss(x):
    return np.sum(np.abs(x) + x**2 * 0.5)  # abs (sign mask) + pow (const exponent)


def test_grad_graph_abs_pow_roundtrip():
    x = _rng(6).standard_normal((3, 4))
    _grads_match(value_and_grad(capture(_abs_pow_loss, x)), (x,), _abs_pow_loss)


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
    _grads_match(value_and_grad(capture(loss, *args)), args, loss)


# --- G3: cross-pass optimization (CSE across the forward/backward boundary) --
def _n_prim(graph, prim):
    return sum(1 for nd in graph.nodes if nd.prim is prim)


def _sigmoid_loss(x):
    # sigmoid's VJP is g * sigmoid(x) * (1 - sigmoid(x)) -- it *recomputes* sigmoid(x),
    # so the combined graph has the forward's sigmoid(x) plus two in the backward.
    return np.sum(d_sigmoid(x))


def test_cross_pass_cse_merges_recomputed_sigmoid():
    x = _rng(7).standard_normal((3, 4))
    combined = value_and_grad(capture(_sigmoid_loss, x))
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
    opt = optimize(value_and_grad(capture(M.mlp_tree_loss, *args)))
    _grads_match(opt, args, M.mlp_tree_loss)


def test_grad_of_graph_drops_value():
    # grad(graph) mirrors grad(callable): a graph whose output is grads alone (no value),
    # one cotangent per input leaf, matching value_and_grad(graph)'s grad outputs.
    x, w = _rng(0).standard_normal((4, 3)), _rng(1).standard_normal((3, 2))
    fwd = capture(_smooth_loss, x, w)
    gonly = grad(fwd)
    vg = value_and_grad(fwd)
    assert len(gonly.outputs) == len(vg.outputs) - 1  # value output dropped
    grads_only = eval_graph(gonly, x, w)
    _, grads = eval_graph(vg, x, w)
    got = [np.asarray(_value(g)) for g in grads_only]
    ref = [np.asarray(_value(g)) for g in grads]
    assert len(got) == len(ref) and got
    for a, b in zip(got, ref):
        assert np.allclose(a, b, atol=1e-9)


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


# --- G4: ambient weights -> a ParamDict of weight gradients -----------------
# Module-level so the instrumented capture runner (recompiled from source) sees the
# `with weights:`-injected names as globals.
def _ambient_loss(x):
    return np.sum(np.tanh(x @ aw + ab))  # noqa: F821  (ambient weights aw, ab)


def test_grad_graph_returns_paramdict_of_weight_grads():
    weights = params(aw=_rng(2).standard_normal((3, 2)), ab=np.zeros(2))
    X = _rng(1).standard_normal((4, 3))
    with weights:
        gg = value_and_grad(capture(_ambient_loss, X))
        val, grads = eval_graph(gg, X)
        ref_val, ref_grads = weights.grad(_ambient_loss, X)
    # grads come back as a ParamDict keyed by trainable weight name -- what `weights.step`
    # consumes -- not a flat tuple.
    assert isinstance(grads, ParamDict)
    assert set(grads) == {"aw", "ab"}
    assert np.allclose(float(_value(val)), float(_value(ref_val)), atol=1e-9)
    for key in ("aw", "ab"):
        assert np.allclose(
            np.asarray(_value(grads[key])), np.asarray(ref_grads[key]), atol=1e-9
        )


def test_grad_graph_weights_step_trains():
    weights = params(aw=0.3 * _rng(5).standard_normal((3, 2)), ab=np.zeros(2))
    X = _rng(6).standard_normal((4, 3))
    with weights:
        optimized = optimize(value_and_grad(capture(_ambient_loss, X)))
        first = float(_value(optimized(X)[0]))
        for _ in range(25):
            _v, grads = optimized(X)
            weights.step(grads, 0.1)  # minimize: step along -grad
        last = float(_value(optimized(X)[0]))
    # The optimized graph re-reads the (stepped) weights each call, so the loss moves.
    assert last < first - 1e-6


def test_grad_graph_frozen_weight_has_no_grad():
    weights = params(aw=_rng(2).standard_normal((3, 2)), ab=frozen(np.zeros(2)))
    X = _rng(1).standard_normal((4, 3))
    with weights:
        g = capture(_ambient_loss, X)
        val, grads = eval_graph(value_and_grad(g), X)
    # Only the trainable weight is a live `_WEIGHT` leaf; the frozen one stays a constant.
    assert set(g.weight_inputs) == {"aw"}
    assert isinstance(grads, ParamDict) and set(grads) == {"aw"}
    # `weights.step` simply skips the absent frozen key.
    before = np.asarray(weights["ab"].value).copy()
    with weights:
        weights.step(grads, 0.1)
    assert np.allclose(np.asarray(weights["ab"].value), before)


def _tied_ambient_loss(x):
    return np.sum(np.tanh(x @ ta)) + np.sum(np.tanh(x @ tb))  # noqa: F821


def test_grad_graph_tied_weights_share_one_leaf_and_grad():
    w0 = _rng(3).standard_normal((3, 3))
    weights = params(ta=tied("k", w0), tb=tied("k", w0))
    X = _rng(1).standard_normal((4, 3))
    with weights:
        g = capture(_tied_ambient_loss, X)
        _val, grads = eval_graph(value_and_grad(g), X)
        _rv, ref_grads = weights.grad(_tied_ambient_loss, X)
    # Both tied names map to a single `_WEIGHT` node and come back with equal gradients.
    assert g.weight_inputs["ta"] == g.weight_inputs["tb"]
    assert np.allclose(
        np.asarray(_value(grads["ta"])), np.asarray(_value(grads["tb"])), atol=1e-9
    )
    for key in ("ta", "tb"):
        assert np.allclose(
            np.asarray(_value(grads[key])), np.asarray(ref_grads[key]), atol=1e-9
        )


def test_grad_graph_without_weights_is_per_arg_tuple():
    # No ambient block: grads come back as a tuple with one entry per positional argument
    # (each a plain array here), matching eager value_and_grad.
    x, w = _rng(0).standard_normal((4, 3)), _rng(1).standard_normal((3, 2))
    _v, grads = eval_graph(value_and_grad(capture(_smooth_loss, x, w)), x, w)
    assert isinstance(grads, tuple) and len(grads) == 2


def _dict_arg_loss(x, p):
    # p is a dict argument; its gradient must come back as a matching dict.
    return np.sum(np.tanh(x @ p["w"] + p["b"]))


def test_grad_graph_dict_arg_yields_dict_grad():
    from pycograd.tree import tree_leaves

    x = _rng(0).standard_normal((4, 3))
    p = {"w": _rng(1).standard_normal((3, 2)), "b": _rng(2).standard_normal((2,))}
    val, grads = eval_graph(value_and_grad(capture(_dict_arg_loss, x, p)), x, p)
    ref_val, ref_grads = value_and_grad(_dict_arg_loss)(x, p)
    # grads is (dx, dp) with dp a dict matching p -- "dict in, dict out".
    assert isinstance(grads, tuple) and len(grads) == 2
    dx, dp = grads
    assert isinstance(dp, dict) and set(dp) == {"w", "b"}
    assert np.allclose(float(_value(val)), float(_value(ref_val)), atol=1e-9)
    assert np.allclose(np.asarray(_value(dx)), np.asarray(ref_grads[0]), atol=1e-9)
    for k in ("w", "b"):
        assert np.allclose(
            np.asarray(_value(dp[k])), np.asarray(ref_grads[1][k]), atol=1e-9
        )
    # nothing dropped: leaf counts line up with the reference
    assert len(tree_leaves(grads)) == len(tree_leaves(ref_grads))


def test_grad_graph_inference_call_returns_concrete_array():
    # `graph(x)` at top level is the inference path: it must return a concrete ndarray
    # (not a tape `Var`), so plain numpy can consume it.
    weights = params(aw=_rng(2).standard_normal((3, 2)), ab=np.zeros(2))
    X = _rng(1).standard_normal((4, 3))
    with weights:
        forward = optimize(capture(_ambient_loss, X))
    out = forward(X)
    assert isinstance(out, np.ndarray)
    assert np.isfinite(float(out))

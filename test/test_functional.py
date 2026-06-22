# -*- coding: utf-8 -*-
"""Tests for ``pycograd.functional``: the stable softmax family, cross-entropy and
activations. Gradients are finite-difference checked through ``value_and_grad``;
values are checked against direct numpy references; and each op is exercised under
``vmap`` / ``jvp`` / ``eval_shape`` to confirm it composes (it is a pure composition
of primitives, so this should hold for free)."""
import pytest

np = pytest.importorskip("numpy")

from pycograd import ShapeDtypeStruct as S  # noqa: E402
from pycograd import (  # noqa: E402
    cross_entropy,
    dropout,
    elu,
    embedding,
    eval_shape,
    gelu,
    grad,
    hardsigmoid,
    hardswish,
    jvp,
    layer_norm,
    leaky_relu,
    linear,
    log_softmax,
    logsumexp,
    mish,
    relu,
    rms_norm,
    scaled_dot_product_attention,
    selu,
    silu,
    softmax,
    softplus,
    softsign,
    tanh,
    value_and_grad,
    vmap,
)


# --- finite-difference oracle (matches test_autodiff's convention) ----------
def finite_diff(f, args, h=1e-6):
    def s(*a):
        return float(np.sum(f(*a)))

    base = [np.array(a, dtype=float) for a in args]
    grads = []
    for i, a in enumerate(base):
        g = np.zeros_like(a)
        for idx in np.ndindex(a.shape):
            up = [x.copy() for x in base]
            dn = [x.copy() for x in base]
            up[i][idx] += h
            dn[i][idx] -= h
            g[idx] = (s(*up) - s(*dn)) / (2 * h)
        grads.append(g)
    return tuple(grads)


def _assert_grads_match(f, args, atol=1e-5):
    _, ad = value_and_grad(f)(*args)
    fd = finite_diff(f, args)
    assert len(ad) == len(fd)
    for g_ad, g_fd in zip(ad, fd):
        assert np.allclose(g_ad, g_fd, atol=atol), (g_ad, g_fd)


# --- numpy references -------------------------------------------------------
def _np_log_softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


# --- value correctness ------------------------------------------------------
def test_softmax_values_sum_to_one():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
    out = np.asarray(softmax(x, axis=-1))
    assert np.allclose(out.sum(axis=-1), 1.0)
    assert np.allclose(out, np.exp(_np_log_softmax(x)))


def test_log_softmax_matches_reference():
    x = np.array([[-1.0, 0.5, 2.0], [3.0, 3.0, 3.0]])
    assert np.allclose(np.asarray(log_softmax(x)), _np_log_softmax(x))


def test_logsumexp_matches_reference_and_keepdims():
    x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]])
    ref = np.log(np.sum(np.exp(x), axis=-1))
    assert np.allclose(np.asarray(logsumexp(x, axis=-1)), ref)
    kept = np.asarray(logsumexp(x, axis=-1, keepdims=True))
    assert kept.shape == (2, 1)
    assert np.allclose(kept[:, 0], ref)


def test_softmax_is_numerically_stable():
    # Without the max-shift these overflow to inf/nan.
    x = np.array([1000.0, 1000.0, 1000.0])
    out = np.asarray(softmax(x))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 1.0 / 3.0)
    assert np.isfinite(np.asarray(logsumexp(x)))


# --- gradient checks --------------------------------------------------------
def test_softmax_grad_matches_finite_diff():
    x = np.array([[0.3, -1.2, 0.7], [2.0, 0.1, -0.5]])
    _assert_grads_match(lambda a: softmax(a, axis=-1), (x,))


def test_log_softmax_grad_matches_finite_diff():
    x = np.array([[0.3, -1.2, 0.7], [2.0, 0.1, -0.5]])
    _assert_grads_match(lambda a: log_softmax(a, axis=-1), (x,))


def test_logsumexp_grad_matches_finite_diff():
    x = np.array([[0.5, 1.5, -0.5], [1.0, 0.0, 2.0]])
    _assert_grads_match(lambda a: logsumexp(a, axis=-1), (x,))


def test_cross_entropy_grad_matches_finite_diff():
    # Passed positionally so grads w.r.t. both logits and targets are checked.
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 3))
    targets = np.eye(3)[rng.integers(0, 3, size=4)]
    _assert_grads_match(cross_entropy, (logits, targets))


def test_activation_grads_match_finite_diff():
    # No 0.0 sample: relu's kink there has a subgradient the central difference
    # can't match. silu/gelu are smooth everywhere.
    x = np.array([-2.0, -0.5, 0.5, 1.5, 2.0])
    _assert_grads_match(relu, (x,))
    _assert_grads_match(silu, (x,))
    _assert_grads_match(gelu, (x,))


# --- composition with the transform stack -----------------------------------
def test_softmax_composes_with_vmap_grad():
    rng = np.random.default_rng(1)
    batch = rng.standard_normal((5, 3))

    def row_loss(row):
        return np.sum(softmax(row) ** 2)

    (per_sample,) = vmap(grad(row_loss))(batch)
    assert per_sample.shape == (5, 3)
    for i in range(5):
        (g_i,) = grad(row_loss)(batch[i])
        assert np.allclose(per_sample[i], g_i)


def test_cross_entropy_jvp_runs():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((4, 3))
    targets = np.eye(3)[rng.integers(0, 3, size=4)]
    v = rng.standard_normal((4, 3))
    primal, tangent = jvp(cross_entropy, (logits, targets), (v, np.zeros_like(targets)))
    assert np.isfinite(float(primal))
    assert np.isfinite(float(tangent))


def test_softmax_eval_shape():
    out = eval_shape(lambda x: softmax(x, axis=-1), S((8, 10)))
    assert tuple(out.shape) == (8, 10)


# --- normalization ----------------------------------------------------------
def test_layer_norm_values_and_grad():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 5))
    g, b = np.ones(5), np.zeros(5)
    out = np.asarray(layer_norm(x, g, b))
    # Identity affine (gamma=1, beta=0) => zero mean, unit (biased) variance per row.
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(out.var(axis=-1), 1.0, atol=1e-4)
    _assert_grads_match(layer_norm, (x, g, b))


def test_rms_norm_values_and_grad():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3, 5))
    g = np.ones(5)
    out = np.asarray(rms_norm(x, g))
    ref = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    assert np.allclose(out, ref)
    _assert_grads_match(rms_norm, (x, g))


# --- extra activations ------------------------------------------------------
def test_extra_activation_values_match_reference():
    x = np.array([-2.0, -0.5, 0.5, 1.5, 2.0])
    assert np.allclose(np.asarray(tanh(x)), np.tanh(x))
    assert np.allclose(np.asarray(leaky_relu(x, 0.1)), np.where(x > 0, x, 0.1 * x))
    assert np.allclose(np.asarray(elu(x)), np.where(x > 0, x, np.expm1(x)))
    assert np.allclose(np.asarray(softplus(x)), np.log1p(np.exp(x)))
    assert np.allclose(np.asarray(mish(x)), x * np.tanh(np.log1p(np.exp(x))))
    assert np.allclose(np.asarray(hardsigmoid(x)), np.clip(x + 3, 0, 6) / 6)
    assert np.allclose(np.asarray(hardswish(x)), x * np.clip(x + 3, 0, 6) / 6)
    assert np.allclose(np.asarray(softsign(x)), x / (1 + np.abs(x)))


def test_softplus_is_stable_for_large_inputs():
    x = np.array([-1000.0, 1000.0])
    out = np.asarray(softplus(x))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, np.maximum(x, 0.0), atol=1e-6)


def test_extra_activation_grads_match_finite_diff():
    # Sample away from the kinks (0 for leaky_relu/selu; +-3 for the hard* ops)
    # so the central difference is valid.
    x = np.array([-2.0, -0.5, 0.5, 1.5, 2.0])
    for act in (
        tanh,
        leaky_relu,
        elu,
        softplus,
        mish,
        hardswish,
        hardsigmoid,
        softsign,
        selu,
    ):
        _assert_grads_match(act, (x,))


# --- linear -----------------------------------------------------------------
def test_linear_values_and_grad():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((4, 3))
    w = rng.standard_normal((3, 5))
    b = rng.standard_normal(5)
    assert np.allclose(np.asarray(linear(x, w, b)), x @ w + b)
    assert np.allclose(np.asarray(linear(x, w)), x @ w)  # bias optional
    _assert_grads_match(linear, (x, w, b))


# --- attention --------------------------------------------------------------
def test_attention_matches_manual_reference():
    rng = np.random.default_rng(7)
    q = rng.standard_normal((5, 4))
    k = rng.standard_normal((6, 4))
    v = rng.standard_normal((6, 3))
    scores = (q @ k.T) * (4**-0.5)
    ref = np.asarray(softmax(scores, axis=-1)) @ v
    assert np.allclose(np.asarray(scaled_dot_product_attention(q, k, v)), ref)


def test_attention_mask_zeros_out_positions():
    rng = np.random.default_rng(8)
    q = rng.standard_normal((2, 4))
    k = rng.standard_normal((3, 4))
    v = rng.standard_normal((3, 3))
    mask = np.array([[True, True, False], [True, False, False]])
    out = np.asarray(scaled_dot_product_attention(q, k, v, mask))
    # Masked attention only mixes the allowed value rows.
    ref0 = np.asarray(softmax((q @ k.T * 4**-0.5)[0, :2], axis=-1)) @ v[:2]
    assert np.allclose(out[0], ref0)
    assert np.allclose(out[1], v[0])  # row 1 attends to a single position


def test_attention_grad_matches_finite_diff():
    rng = np.random.default_rng(9)
    q = rng.standard_normal((3, 4))
    k = rng.standard_normal((3, 4))
    v = rng.standard_normal((3, 2))
    _assert_grads_match(scaled_dot_product_attention, (q, k, v))


def test_attention_batches_under_vmap():
    rng = np.random.default_rng(10)
    q = rng.standard_normal((5, 3, 4))
    k = rng.standard_normal((5, 3, 4))
    v = rng.standard_normal((5, 3, 2))
    batched = vmap(scaled_dot_product_attention)(q, k, v)
    assert batched.shape == (5, 3, 2)
    for i in range(5):
        assert np.allclose(batched[i], scaled_dot_product_attention(q[i], k[i], v[i]))


# --- embedding --------------------------------------------------------------
def test_embedding_gathers_rows_and_grad():
    rng = np.random.default_rng(11)
    table = rng.standard_normal((10, 4))
    idx = np.array([[1, 3], [3, 7]])
    out = np.asarray(embedding(table, idx))
    assert out.shape == (2, 2, 4)
    assert np.allclose(out, table[idx])
    # Gradient w.r.t. the table scatter-adds: row 3 is looked up twice.
    _, (g,) = value_and_grad(lambda t: embedding(t, idx))(table)
    expected = np.zeros_like(table)
    for i in idx.ravel():
        expected[i] += 1.0
    assert np.allclose(g, expected)


# --- dropout ----------------------------------------------------------------
def test_dropout_eval_is_identity():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((4, 5))
    assert np.allclose(np.asarray(dropout(x, 0.5, training=False)), x)
    assert np.allclose(np.asarray(dropout(x, 0.0, training=True)), x)


def test_dropout_train_scales_survivors_and_grad_routes_through_mask():
    x = np.ones((100, 100))
    out = np.asarray(dropout(x, 0.5, training=True, rng=np.random.default_rng(99)))
    # Inverted dropout: survivors are scaled to 1/keep, dropped are 0.
    assert set(np.unique(out)).issubset({0.0, 2.0})
    assert abs(out.mean() - 1.0) < 0.05  # expectation preserved
    # The mask is a plain constant, so d/dx sum(dropout(x)) == mask.
    _, (g,) = value_and_grad(
        lambda a: dropout(a, 0.5, training=True, rng=np.random.default_rng(99))
    )(x)
    mask = (np.random.default_rng(99).random(x.shape) < 0.5) / 0.5
    assert np.allclose(g, mask)


# --- composition with the transform stack -----------------------------------
def test_layer_norm_jvp_and_eval_shape():
    rng = np.random.default_rng(14)
    x = rng.standard_normal((3, 5))
    g, b = np.ones(5), np.zeros(5)
    v = rng.standard_normal((3, 5))
    primal, tangent = jvp(lambda a: layer_norm(a, g, b), (x,), (v,))
    assert primal.shape == (3, 5) and tangent.shape == (3, 5)
    out = eval_shape(lambda a: layer_norm(a, g, b), S((3, 5)))
    assert tuple(out.shape) == (3, 5)

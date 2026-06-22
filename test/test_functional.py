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
    batch_norm,
    batch_norm_init,
    conv2d,
    conv_transpose2d,
    cross_entropy,
    dropout,
    elu,
    embedding,
    eval_shape,
    gelu,
    grad,
    group_norm,
    hardsigmoid,
    hardswish,
    instance_norm,
    jvp,
    layer_norm,
    leaky_relu,
    linear,
    log_softmax,
    logsumexp,
    mish,
    multi_head_attention,
    relu,
    rms_norm,
    scaled_dot_product_attention,
    selu,
    silu,
    softmax,
    softplus,
    softsign,
    tanh,
    upsample_nearest2d,
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


# Running stats are module globals (not closure free vars) so the instrumented
# grad helper can read them -- instrumented bodies drop closure captures.
_BN_RM, _BN_RV = batch_norm_init(4)


def _bn_train_y(x, gamma, beta):  # return only y so finite-diff can score it
    return batch_norm(x, gamma, beta, _BN_RM, _BN_RV, training=True)[0]


def test_batch_norm_train_values_running_stats_and_grad():
    rng = np.random.default_rng(15)
    x = rng.standard_normal((8, 4, 3, 3))
    gamma, beta = np.ones(4), np.zeros(4)
    y, new_mean, new_var = batch_norm(x, gamma, beta, _BN_RM, _BN_RV, training=True)
    y = np.asarray(y)
    # Identity affine => per-channel zero mean / unit variance over (N, H, W).
    assert np.allclose(y.mean(axis=(0, 2, 3)), 0.0, atol=1e-6)
    assert np.allclose(y.var(axis=(0, 2, 3)), 1.0, atol=1e-4)
    # Running buffers are advanced (EMA) and come back as plain arrays.
    assert isinstance(new_mean, np.ndarray) and new_mean.shape == (4,)
    assert not np.allclose(new_mean, _BN_RM)
    _assert_grads_match(_bn_train_y, (x, gamma, beta), atol=1e-4)


def test_batch_norm_eval_uses_running_stats_unchanged():
    rng = np.random.default_rng(16)
    x = rng.standard_normal((5, 3, 2, 2))
    gamma, beta = np.ones(3), np.zeros(3)
    rm, rv = np.array([1.0, -1.0, 0.5]), np.array([4.0, 1.0, 9.0])
    y, new_mean, new_var = batch_norm(x, gamma, beta, rm, rv, training=False)
    ref = (x - rm.reshape(1, 3, 1, 1)) / np.sqrt(rv.reshape(1, 3, 1, 1) + 1e-5)
    assert np.allclose(np.asarray(y), ref)
    assert np.allclose(new_mean, rm) and np.allclose(new_var, rv)  # eval: untouched


def test_batch_norm_eval_shape():
    # Plain closure params (gamma/beta/running stats reshaped to the channel axis):
    # these are eagerly-evaluated constants under the shape trace, which now works.
    g, b = np.ones(4), np.zeros(4)
    rm, rv = batch_norm_init(4)
    out = eval_shape(
        lambda x: batch_norm(x, g, b, rm, rv, training=False)[0], S((8, 4, 5, 5))
    )
    assert tuple(out.shape) == (8, 4, 5, 5)


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


# --- group / instance norm --------------------------------------------------
# Module-level wrappers (with the int arg baked in) so value_and_grad can
# getsource them -- a black-wrapped multi-line lambda breaks source extraction.
def _group_norm_3(x, gamma, beta):
    return group_norm(x, gamma, beta, 3)


def test_group_norm_values_and_grad():
    rng = np.random.default_rng(17)
    x = rng.standard_normal((2, 6, 4, 4))
    g, b = np.ones(6), np.zeros(6)
    y = np.asarray(group_norm(x, g, b, num_groups=3)).reshape(2, 3, 2, 4, 4)
    assert np.allclose(y.mean(axis=(2, 3, 4)), 0.0, atol=1e-6)  # per-group zero mean
    assert np.allclose(y.var(axis=(2, 3, 4)), 1.0, atol=1e-4)  # per-group unit var
    _assert_grads_match(_group_norm_3, (x, g, b), atol=1e-4)


def test_instance_norm_normalizes_each_channel_per_sample():
    rng = np.random.default_rng(18)
    x = rng.standard_normal((2, 3, 4, 4))
    g, b = np.ones(3), np.zeros(3)
    y = np.asarray(instance_norm(x, g, b)).reshape(2, 3, 16)
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-4)


# --- multi-head attention ---------------------------------------------------
def test_multi_head_attention_matches_per_head_and_single_head():
    rng = np.random.default_rng(19)
    q, k, v = (rng.standard_normal((5, 8)) for _ in range(3))
    mh = np.asarray(multi_head_attention(q, k, v, num_heads=2))
    heads = [
        np.asarray(scaled_dot_product_attention(q[:, s], k[:, s], v[:, s]))
        for s in (slice(0, 4), slice(4, 8))
    ]
    assert np.allclose(mh, np.concatenate(heads, axis=-1))
    # one head is plain scaled dot-product attention
    assert np.allclose(
        np.asarray(multi_head_attention(q, k, v, 1)),
        np.asarray(scaled_dot_product_attention(q, k, v)),
    )


def _mha_2(q, k, v):
    return multi_head_attention(q, k, v, 2)


def test_multi_head_attention_grad_and_vmap():
    rng = np.random.default_rng(20)
    q, k, v = (rng.standard_normal((4, 6)) for _ in range(3))
    _assert_grads_match(_mha_2, (q, k, v), atol=1e-4)
    bq, bk, bv = (rng.standard_normal((3, 4, 6)) for _ in range(3))
    out = vmap(_mha_2)(bq, bk, bv)
    assert out.shape == (3, 4, 6)
    for i in range(3):
        assert np.allclose(out[i], multi_head_attention(bq[i], bk[i], bv[i], 2))


# --- transposed conv / upsample ---------------------------------------------
def test_conv_transpose2d_is_the_adjoint_of_conv2d():
    rng = np.random.default_rng(21)
    x = rng.standard_normal((2, 3, 5, 5))  # (N, C_in, H, W)
    w = rng.standard_normal((4, 3, 3, 3))  # (C_out, C_in, kH, kW)
    y = np.asarray(conv2d(x, w, stride=2, pad=0))
    g = rng.standard_normal(y.shape)
    xt = np.asarray(conv_transpose2d(g, w, stride=2, pad=0))
    assert xt.shape == x.shape  # inverts conv2d's size map
    # <conv2d(x, w), g> == <x, conv_transpose2d(g, w)>: the transpose identity.
    assert np.allclose(np.sum(y * g), np.sum(x * xt))


def _conv_transpose_s2(x, w):  # module-level so value_and_grad can getsource it
    return conv_transpose2d(x, w, stride=2)


def test_conv_transpose2d_grad_matches_finite_diff():
    rng = np.random.default_rng(22)
    x = rng.standard_normal((1, 3, 4, 4))  # input channels == w's C_out (3)
    w = rng.standard_normal((3, 2, 3, 3))  # (C_out, C_in, kH, kW)
    _assert_grads_match(_conv_transpose_s2, (x, w), atol=1e-4)


def test_upsample_nearest2d_repeats_and_grad_sum_pools():
    x = np.arange(4.0).reshape(1, 1, 2, 2)
    up = np.asarray(upsample_nearest2d(x, 2))
    assert up.shape == (1, 1, 4, 4)
    assert np.allclose(up[0, 0, :2, :2], x[0, 0, 0, 0])  # each source repeated 2x2
    # gradient of a sum sends 1 back to each source scale*scale times.
    _, (g,) = value_and_grad(lambda a: np.sum(upsample_nearest2d(a, 2)))(x)
    assert np.allclose(g, 4.0)


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

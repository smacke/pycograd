# -*- coding: utf-8 -*-
"""Tests for the convolution / pooling / one-hot composed ops and the ``cumsum``
primitive. Forward values are checked against explicit references; gradients against
finite differences; and ``cumsum`` is exercised under vmap / jvp / eval_shape."""
import pytest

np = pytest.importorskip("numpy")

from pycograd import ShapeDtypeStruct as S  # noqa: E402
from pycograd import (  # noqa: E402
    avg_pool2d,
    causal_conv1d,
    conv1d,
    conv2d,
    conv_transpose1d,
    cumsum,
    eval_shape,
    jvp,
    max_pool2d,
    one_hot,
    streaming_conv1d,
    streaming_conv1d_init,
    streaming_conv_transpose1d,
    streaming_conv_transpose1d_init,
    value_and_grad,
    vmap,
)


def finite_diff(f, args, h=1e-5):
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


def _assert_grads_match(f, args, atol=1e-4):
    _, ad = value_and_grad(f)(*args)
    fd = finite_diff(f, args)
    for g_ad, g_fd in zip(ad, fd):
        assert np.allclose(g_ad, g_fd, atol=atol), (g_ad, g_fd)


# --- reference implementations ----------------------------------------------
def _ref_conv2d(x, w, b, stride, pad):
    n, c, h, ww = x.shape
    c_out, _, kh, kw = w.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    h2, w2 = x.shape[2], x.shape[3]
    h_out, w_out = (h2 - kh) // stride + 1, (w2 - kw) // stride + 1
    out = np.zeros((n, c_out, h_out, w_out))
    for ni in range(n):
        for co in range(c_out):
            for oh in range(h_out):
                for ow in range(w_out):
                    hs, ws = oh * stride, ow * stride
                    patch = x[ni, :, hs : hs + kh, ws : ws + kw]
                    out[ni, co, oh, ow] = np.sum(patch * w[co])
    return out + b.reshape(1, c_out, 1, 1)


# --- conv forward & grad ----------------------------------------------------
def test_conv2d_forward_matches_reference():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 6, 5))
    w = rng.standard_normal((4, 3, 3, 3))
    b = rng.standard_normal(4)
    got = np.asarray(conv2d(x, w, b, stride=2, pad=1))
    assert np.allclose(got, _ref_conv2d(x, w, b, stride=2, pad=1))


def _conv_loss(x, w, b):
    return np.sum(conv2d(x, w, b, stride=1, pad=1) ** 2)


def test_conv2d_grad_matches_finite_diff():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 2, 5, 5))
    w = rng.standard_normal((3, 2, 3, 3))
    b = rng.standard_normal(3)
    _assert_grads_match(_conv_loss, (x, w, b))


def test_conv1d_forward_matches_conv2d_collapse():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 7))
    w = rng.standard_normal((4, 3, 3))
    b = rng.standard_normal(4)
    got = np.asarray(conv1d(x, w, b, stride=1, pad=1))
    # reference: pad the length axis, then a height-1 4-D conv
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1)))
    ref = _ref_conv2d(xp[:, :, None, :], w[:, :, None, :], b, stride=1, pad=0)
    assert np.allclose(got, ref[:, :, 0, :])


# --- dilation ---------------------------------------------------------------
def _ref_dilated_conv1d(x, w, b, stride, dilation):
    n, c, ll = x.shape
    c_out, _, k = w.shape
    span = (k - 1) * dilation
    l_out = (ll - span - 1) // stride + 1
    out = np.zeros((n, c_out, l_out))
    for ni in range(n):
        for co in range(c_out):
            for t in range(l_out):
                s = t * stride
                patch = x[ni, :, s : s + span + 1 : dilation]
                out[ni, co, t] = np.sum(patch * w[co])
    return out + b.reshape(1, c_out, 1)


def test_conv1d_dilation_matches_reference():
    rng = np.random.default_rng(10)
    x = rng.standard_normal((2, 3, 11))
    w = rng.standard_normal((4, 3, 3))
    b = rng.standard_normal(4)
    got = np.asarray(conv1d(x, w, b, stride=2, dilation=2))
    assert np.allclose(got, _ref_dilated_conv1d(x, w, b, stride=2, dilation=2))


def test_conv2d_dilation_matches_reference():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((1, 2, 8, 8))
    w = rng.standard_normal((3, 2, 3, 3))
    b = rng.standard_normal(3)
    # height-1 collapse: a 2-D dilated conv matches the 1-D reference row-by-row
    got = np.asarray(conv2d(x[:, :, :1, :], w[:, :, :1, :], b, dilation=2))
    ref = _ref_dilated_conv1d(x[:, :, 0, :], w[:, :, 0, :], b, stride=1, dilation=2)
    assert np.allclose(got[:, :, 0, :], ref)


# --- causal conv1d ----------------------------------------------------------
def test_causal_conv1d_is_left_padded_conv1d():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((2, 3, 9))
    w = rng.standard_normal((4, 3, 3))
    b = rng.standard_normal(4)
    got = np.asarray(causal_conv1d(x, w, b, dilation=2))
    assert got.shape == x.shape[:1] + (4,) + x.shape[2:]  # length preserved
    xp = np.pad(x, ((0, 0), (0, 0), ((3 - 1) * 2, 0)))  # left pad (k-1)*dilation
    ref = np.asarray(conv1d(xp, w, b, dilation=2))
    assert np.allclose(got, ref)


def _causal_conv1d_loss(x, w, b):
    return np.sum(causal_conv1d(x, w, b, stride=1, dilation=2) ** 2)


def test_causal_conv1d_grad_matches_finite_diff():
    rng = np.random.default_rng(13)
    x = rng.standard_normal((1, 2, 6))
    w = rng.standard_normal((3, 2, 3))
    b = rng.standard_normal(3)
    _assert_grads_match(_causal_conv1d_loss, (x, w, b))


# --- conv_transpose1d -------------------------------------------------------
def test_conv_transpose1d_is_the_adjoint_of_conv1d():
    rng = np.random.default_rng(14)
    x = rng.standard_normal((2, 3, 9))  # (N, C_in, L)
    w = rng.standard_normal((4, 3, 3))  # (C_out, C_in, k)
    y = np.asarray(conv1d(x, w, stride=2, pad=0))
    g = rng.standard_normal(y.shape)
    xt = np.asarray(conv_transpose1d(g, w, stride=2, pad=0))
    assert xt.shape == x.shape  # inverts conv1d's size map
    assert np.allclose(np.sum(y * g), np.sum(x * xt))  # the transpose identity


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_conv_transpose1d_dilated_adjoint(stride, dilation):
    rng = np.random.default_rng(140 + stride * 10 + dilation)
    # L=9 (odd) keeps the stride-2 size map exactly invertible for k=3, any dilation.
    x = rng.standard_normal((2, 3, 9))  # (N, C_in, L)
    w = rng.standard_normal((4, 3, 3))  # (C_out, C_in, k)
    y = np.asarray(conv1d(x, w, stride=stride, pad=0, dilation=dilation))
    g = rng.standard_normal(y.shape)
    xt = np.asarray(conv_transpose1d(g, w, stride=stride, pad=0, dilation=dilation))
    assert xt.shape == x.shape  # dilated size map inverts
    assert np.allclose(np.sum(y * g), np.sum(x * xt))  # the transpose identity


def _conv_transpose1d_loss(x, w):
    return np.sum(conv_transpose1d(x, w, stride=2) ** 2)


def test_conv_transpose1d_grad_matches_finite_diff():
    rng = np.random.default_rng(15)
    x = rng.standard_normal((1, 3, 4))  # input channels == w's C_out (3)
    w = rng.standard_normal((3, 2, 3))  # (C_out, C_in, k)
    _assert_grads_match(_conv_transpose1d_loss, (x, w))


def _conv_transpose1d_dilated_loss(x, w):
    return np.sum(conv_transpose1d(x, w, stride=2, dilation=2) ** 2)


def test_conv_transpose1d_dilated_grad_matches_finite_diff():
    rng = np.random.default_rng(25)
    x = rng.standard_normal((1, 3, 4))
    w = rng.standard_normal((3, 2, 3))
    _assert_grads_match(_conv_transpose1d_dilated_loss, (x, w))


# --- streaming convolutions -------------------------------------------------
# The incremental step, fed a sequence chunk-by-chunk, must reproduce the
# parallel op bit-for-bit (the ``rwkv_step`` equivalence pattern).
_CHUNKS = [2, 1, 3, 2, 4, 1]  # deliberately uneven chunk sizes


def _stream_conv1d(x, w, b, stride, dilation):
    state = streaming_conv1d_init(x.shape[1], w.shape[2], dilation, x.shape[0])
    outs, i, j = [], 0, 0
    while i < x.shape[2]:
        c = _CHUNKS[j % len(_CHUNKS)]
        y, state = streaming_conv1d(
            x[:, :, i : i + c], w, b, state, stride=stride, dilation=dilation
        )
        outs.append(np.asarray(y))
        i, j = i + c, j + 1
    return np.concatenate(outs, axis=2)


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
def test_streaming_conv1d_matches_causal(stride, dilation):
    rng = np.random.default_rng(16)
    x = rng.standard_normal((2, 3, 13))
    w = rng.standard_normal((4, 3, 3))
    b = rng.standard_normal(4)
    streamed = _stream_conv1d(x, w, b, stride, dilation)
    parallel = np.asarray(causal_conv1d(x, w, b, stride=stride, dilation=dilation))
    assert streamed.shape == parallel.shape
    assert np.allclose(streamed, parallel, atol=1e-9)


def _stream_conv_transpose1d(x, w, b, stride, dilation=1):
    state = streaming_conv_transpose1d_init()
    outs, i, j = [], 0, 0
    while i < x.shape[2]:
        c = _CHUNKS[j % len(_CHUNKS)]
        y, state = streaming_conv_transpose1d(
            x[:, :, i : i + c], w, b, state, stride=stride, dilation=dilation
        )
        outs.append(np.asarray(y))
        i, j = i + c, j + 1
    outs.append(np.asarray(state[0]))  # end-of-stream flush of the cached tail
    return np.concatenate(outs, axis=2)


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("use_bias", [False, True])
def test_streaming_conv_transpose1d_matches_parallel(stride, use_bias):
    rng = np.random.default_rng(17)
    x = rng.standard_normal((2, 4, 11))  # (N, C_out, L)
    w = rng.standard_normal((4, 3, 3))  # (C_out, C_in, k)
    b = rng.standard_normal(3) if use_bias else None
    streamed = _stream_conv_transpose1d(x, w, b, stride)
    parallel = np.asarray(conv_transpose1d(x, w, b, stride=stride))
    assert streamed.shape == parallel.shape
    assert np.allclose(streamed, parallel, atol=1e-9)


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [2, 3])
def test_streaming_conv_transpose1d_dilation_matches_parallel(stride, dilation):
    rng = np.random.default_rng(19)
    x = rng.standard_normal((2, 4, 11))  # (N, C_out, L)
    w = rng.standard_normal((4, 3, 3))  # (C_out, C_in, k)
    b = rng.standard_normal(3)
    streamed = _stream_conv_transpose1d(x, w, b, stride, dilation)
    parallel = np.asarray(conv_transpose1d(x, w, b, stride=stride, dilation=dilation))
    assert streamed.shape == parallel.shape
    assert np.allclose(streamed, parallel, atol=1e-9)


# --- streaming building blocks still compose with the transforms ------------
def test_causal_conv1d_vmap_and_eval_shape():
    rng = np.random.default_rng(18)
    xb = rng.standard_normal((5, 1, 2, 7))  # batch of (1, C_in, L) sequences
    w = rng.standard_normal((3, 2, 3))
    got = np.asarray(vmap(causal_conv1d, in_axes=(0, None))(xb, w))
    ref = np.stack([np.asarray(causal_conv1d(xb[i], w)) for i in range(len(xb))])
    assert np.allclose(got, ref)
    sh = eval_shape(lambda a, b: causal_conv1d(a, b), S((1, 2, 7)), S((3, 2, 3)))
    assert sh.shape == (1, 3, 7)


# --- grouped & depthwise convolutions ---------------------------------------
def _ref_grouped(conv, x, w, b, groups, **kw):
    # Reference: slice the channels into ``groups`` blocks and run the plain
    # (ungrouped) conv on each, then concatenate -- the definition of a grouped conv.
    cin_g, cout_g = x.shape[1] // groups, w.shape[0] // groups
    parts = []
    for g in range(groups):
        xg = x[:, g * cin_g : (g + 1) * cin_g]
        wg = w[g * cout_g : (g + 1) * cout_g]
        bg = None if b is None else b[g * cout_g : (g + 1) * cout_g]
        parts.append(np.asarray(conv(xg, wg, bg, **kw)))
    return np.concatenate(parts, axis=1)


def test_grouped_conv2d_matches_per_group_loop():
    rng = np.random.default_rng(20)
    x = rng.standard_normal((2, 6, 5, 5))
    w = rng.standard_normal((9, 2, 3, 3))  # (C_out, C_in/groups, kH, kW), groups=3
    b = rng.standard_normal(9)
    got = np.asarray(conv2d(x, w, b, stride=1, pad=1, groups=3))
    ref = _ref_grouped(conv2d, x, w, b, 3, stride=1, pad=1)
    assert got.shape == (2, 9, 5, 5)
    assert np.allclose(got, ref)


def test_grouped_conv1d_matches_per_group_loop():
    rng = np.random.default_rng(21)
    x = rng.standard_normal((2, 6, 11))
    w = rng.standard_normal((4, 3, 3))  # (C_out, C_in/groups, k), groups=2
    b = rng.standard_normal(4)
    got = np.asarray(conv1d(x, w, b, stride=2, pad=1, dilation=2, groups=2))
    ref = _ref_grouped(conv1d, x, w, b, 2, stride=2, pad=1, dilation=2)
    assert np.allclose(got, ref)


def test_depthwise_conv1d_matches_per_channel():
    rng = np.random.default_rng(22)
    x = rng.standard_normal((2, 4, 10))
    w = rng.standard_normal((4, 1, 3))  # depthwise: groups == C_in, one filter each
    got = np.asarray(conv1d(x, w, groups=4, pad=1))
    ref = _ref_grouped(conv1d, x, w, None, 4, pad=1)
    assert np.allclose(got, ref)


def _depthwise_conv1d_loss(x, w, b):
    return np.sum(conv1d(x, w, b, groups=4, pad=1) ** 2)


def test_depthwise_conv1d_grad_matches_finite_diff():
    rng = np.random.default_rng(23)
    x = rng.standard_normal((2, 4, 7))
    w = rng.standard_normal((4, 1, 3))
    b = rng.standard_normal(4)
    _assert_grads_match(_depthwise_conv1d_loss, (x, w, b))


def _stream_grouped_conv1d(x, w, b, stride, dilation, groups):
    state = streaming_conv1d_init(x.shape[1], w.shape[2], dilation, x.shape[0])
    outs, i, j = [], 0, 0
    while i < x.shape[2]:
        c = _CHUNKS[j % len(_CHUNKS)]
        y, state = streaming_conv1d(
            x[:, :, i : i + c],
            w,
            b,
            state,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        outs.append(np.asarray(y))
        i, j = i + c, j + 1
    return np.concatenate(outs, axis=2)


def test_streaming_grouped_conv1d_matches_causal():
    rng = np.random.default_rng(24)
    x = rng.standard_normal((2, 6, 13))
    w = rng.standard_normal((9, 2, 3))  # groups=3
    b = rng.standard_normal(9)
    streamed = _stream_grouped_conv1d(x, w, b, stride=1, dilation=2, groups=3)
    parallel = np.asarray(causal_conv1d(x, w, b, stride=1, dilation=2, groups=3))
    assert streamed.shape == parallel.shape
    assert np.allclose(streamed, parallel, atol=1e-9)


# --- pooling ----------------------------------------------------------------
def test_max_pool2d_forward_and_grad():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 4, 4))
    got = np.asarray(max_pool2d(x, 2))
    assert got.shape == (2, 3, 2, 2)
    # reference: max over each non-overlapping 2x2 window
    ref = x.reshape(2, 3, 2, 2, 2, 2).max(axis=(3, 5))
    assert np.allclose(got, ref)
    _assert_grads_match(lambda a: np.sum(max_pool2d(a, 2) ** 2), (x,))


def test_avg_pool2d_forward_and_grad():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 3, 4, 4))
    got = np.asarray(avg_pool2d(x, 2))
    ref = x.reshape(2, 3, 2, 2, 2, 2).mean(axis=(3, 5))
    assert np.allclose(got, ref)
    _assert_grads_match(lambda a: np.sum(avg_pool2d(a, 2) ** 2), (x,))


# --- one_hot ----------------------------------------------------------------
def test_one_hot():
    idx = np.array([0, 2, 1, 2])
    oh = one_hot(idx, 3)
    assert oh.shape == (4, 3)
    assert np.array_equal(oh, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]]))


# --- cumsum -----------------------------------------------------------------
def test_cumsum_forward_matches_numpy():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3, 4))
    assert np.allclose(cumsum(x, axis=1).value, np.cumsum(x, axis=1))
    assert np.allclose(cumsum(x, axis=0).value, np.cumsum(x, axis=0))


def _cumsum_loss(x):
    return np.sum(np.cumsum(x, axis=1) ** 2)


def test_cumsum_grad_matches_finite_diff():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((3, 5))
    _assert_grads_match(_cumsum_loss, (x,))


def test_cumsum_composes_with_vmap():
    rng = np.random.default_rng(7)
    batch = rng.standard_normal((4, 6))
    out = vmap(lambda r: np.cumsum(r, axis=0))(batch)
    assert out.shape == (4, 6)
    for i in range(4):
        assert np.allclose(np.asarray(out[i]), np.cumsum(batch[i]))


def test_cumsum_jvp_is_linear():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((3, 4))
    v = rng.standard_normal((3, 4))
    primal, tangent = jvp(lambda a: np.cumsum(a, axis=1), (x,), (v,))
    assert np.allclose(np.asarray(primal), np.cumsum(x, axis=1))
    assert np.allclose(np.asarray(tangent), np.cumsum(v, axis=1))  # linear


def test_cumsum_eval_shape_and_rejects_flatten():
    out = eval_shape(lambda a: np.cumsum(a, axis=1), S((3, 4)))
    assert tuple(out.shape) == (3, 4)
    with pytest.raises(NotImplementedError):
        cumsum(np.arange(5.0))  # axis=None flatten is unsupported

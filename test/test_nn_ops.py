# -*- coding: utf-8 -*-
"""Tests for the convolution / pooling / one-hot composed ops and the ``cumsum``
primitive. Forward values are checked against explicit references; gradients against
finite differences; and ``cumsum`` is exercised under vmap / jvp / eval_shape."""
import pytest

np = pytest.importorskip("numpy")

from pycograd import ShapeDtypeStruct as S  # noqa: E402
from pycograd import (  # noqa: E402
    avg_pool2d,
    conv1d,
    conv2d,
    cumsum,
    eval_shape,
    jvp,
    max_pool2d,
    one_hot,
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

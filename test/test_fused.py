# -*- coding: utf-8 -*-
"""Tests for the fused ``gated_act`` primitive (``tanh(f) * sigmoid(s)``, the WaveNet /
GLU gate): forward value against the composed reference, the VJP against finite
differences, and parity under vmap / jvp / eval_shape / higher-order grad. The fused
primitive must agree with its closed-form derivative, exactly as the fused ``sigmoid``
and ``einsum`` primitives are checked."""
import pytest

np = pytest.importorskip("numpy")

from pycograd import ShapeDtypeStruct as S  # noqa: E402
from pycograd import (  # noqa: E402
    d_logsumexp,
    d_softmax,
    eval_shape,
    gated_act,
    grad,
    jvp,
    value_and_grad,
    vmap,
)


def _ref(f, s):
    return np.tanh(f) * (1.0 / (1.0 + np.exp(-s)))


def finite_diff(fn, args, h=1e-5):
    base = [np.array(a, dtype=float) for a in args]
    grads = []
    for i, a in enumerate(base):
        g = np.zeros_like(a)
        for idx in np.ndindex(a.shape):
            up = [x.copy() for x in base]
            dn = [x.copy() for x in base]
            up[i][idx] += h
            dn[i][idx] -= h
            g[idx] = (float(np.sum(fn(*up))) - float(np.sum(fn(*dn)))) / (2 * h)
        grads.append(g)
    return tuple(grads)


def test_gated_act_forward_matches_composed():
    rng = np.random.default_rng(0)
    f, s = rng.standard_normal((3, 4)), rng.standard_normal((3, 4))
    assert np.allclose(gated_act(f, s).value, _ref(f, s))


def _gated_loss(f, s):
    return np.sum(gated_act(f, s) ** 2)


def test_gated_act_grad_matches_finite_diff():
    rng = np.random.default_rng(1)
    f, s = rng.standard_normal((2, 5)), rng.standard_normal((2, 5))
    _, ad = value_and_grad(_gated_loss)(f, s)
    fd = finite_diff(lambda a, b: _ref(a, b) ** 2, (f, s))
    for g_ad, g_fd in zip(ad, fd):
        assert np.allclose(g_ad, g_fd, atol=1e-5)


def test_gated_act_vmap_matches_reference():
    rng = np.random.default_rng(2)
    f, s = rng.standard_normal((6, 4)), rng.standard_normal((6, 4))  # batch over axis 0
    got = np.asarray(vmap(gated_act)(f, s))
    assert np.allclose(got, _ref(f, s))


def test_gated_act_jvp_matches_directional_derivative():
    rng = np.random.default_rng(3)
    f, s = rng.standard_normal((3, 4)), rng.standard_normal((3, 4))
    df, ds = rng.standard_normal((3, 4)), rng.standard_normal((3, 4))
    _, tangent = jvp(gated_act, (f, s), (df, ds))
    sig = 1.0 / (1.0 + np.exp(-s))
    expected = sig * (1 - np.tanh(f) ** 2) * df + np.tanh(f) * sig * (1 - sig) * ds
    assert np.allclose(np.asarray(tangent), expected, atol=1e-9)


def test_gated_act_eval_shape():
    sh = eval_shape(gated_act, S((3, 4)), S((5, 1, 4)))  # broadcasting
    assert sh.shape == (5, 3, 4)


def _gated_sum(f, s):
    return np.sum(gated_act(f, s))


def test_gated_act_higher_order_grad():
    # grad(grad) must work: the fused VJP rides ``bind``, so the cotangent graph
    # itself differentiates. Compare d^2/df^2 sum(gate) against finite diff of grad.
    rng = np.random.default_rng(4)
    f, s = rng.standard_normal((2, 3)), rng.standard_normal((2, 3))

    def df_f(a, b):
        return grad(_gated_sum)(a, b)[0]

    _, (hess_col,) = value_and_grad(lambda a: np.sum(df_f(a, s)))(f)
    # finite-diff the first derivative wrt f
    h = 1e-5
    fd = np.zeros_like(f)
    for idx in np.ndindex(f.shape):
        up, dn = f.copy(), f.copy()
        up[idx] += h
        dn[idx] -= h
        fd[idx] = (np.sum(df_f(up, s)) - np.sum(df_f(dn, s))) / (2 * h)
    assert np.allclose(np.asarray(hess_col), fd, atol=1e-4)


# --- fused stable softmax / logsumexp primitives ----------------------------
def _softmax_ref(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _logsumexp_ref(x, axis=-1, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    lse = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return lse if keepdims else np.squeeze(lse, axis=axis)


def test_softmax_forward_matches_reference():
    x = np.random.default_rng(0).standard_normal((3, 4))
    assert np.allclose(d_softmax(x, axis=-1).value, _softmax_ref(x, axis=-1))
    assert np.allclose(d_softmax(x, axis=0).value, _softmax_ref(x, axis=0))


def test_logsumexp_forward_matches_reference_axes_and_keepdims():
    x = np.random.default_rng(1).standard_normal((3, 4))
    assert np.allclose(d_logsumexp(x, axis=-1).value, _logsumexp_ref(x, axis=-1))
    assert np.allclose(d_logsumexp(x).value, _logsumexp_ref(x, axis=None))
    assert np.allclose(
        d_logsumexp(x, axis=1, keepdims=True).value,
        _logsumexp_ref(x, axis=1, keepdims=True),
    )


def test_softmax_is_numerically_stable():
    x = np.array([[1000.0, 1001.0, 1002.0]])  # naive exp(x) would overflow
    out = d_softmax(x, axis=-1).value
    assert np.all(np.isfinite(out)) and np.allclose(np.sum(out), 1.0)
    assert np.isfinite(d_logsumexp(x, axis=-1).value)


def _softmax_loss(x):
    return np.sum(d_softmax(x, axis=-1) ** 2)


def _logsumexp_loss(x):
    return np.sum(d_logsumexp(x, axis=-1))


def test_softmax_grad_matches_finite_diff():
    x = np.random.default_rng(2).standard_normal((3, 4))
    _, (ad,) = value_and_grad(_softmax_loss)(x)
    (fd,) = finite_diff(lambda a: _softmax_ref(a, axis=-1) ** 2, (x,))
    assert np.allclose(np.asarray(ad), fd, atol=1e-5)


def test_logsumexp_grad_matches_finite_diff():
    x = np.random.default_rng(3).standard_normal((3, 4))
    _, (ad,) = value_and_grad(_logsumexp_loss)(x)
    (fd,) = finite_diff(lambda a: _logsumexp_ref(a, axis=-1), (x,))
    assert np.allclose(np.asarray(ad), fd, atol=1e-5)


def test_softmax_vmap_matches_reference():
    xb = np.random.default_rng(4).standard_normal((6, 3, 4))  # batch over axis 0
    got = np.asarray(vmap(lambda a: d_softmax(a, axis=-1))(xb))
    assert np.allclose(got, _softmax_ref(xb, axis=-1))


def test_logsumexp_vmap_matches_reference():
    xb = np.random.default_rng(5).standard_normal((6, 3, 4))
    got = np.asarray(vmap(lambda a: d_logsumexp(a, axis=-1))(xb))
    assert np.allclose(got, _logsumexp_ref(xb, axis=-1))


def test_softmax_jvp_matches_directional_derivative():
    rng = np.random.default_rng(6)
    x, dx = rng.standard_normal((3, 4)), rng.standard_normal((3, 4))
    _, t = jvp(lambda a: d_softmax(a, axis=-1), (x,), (dx,))
    y = _softmax_ref(x, axis=-1)
    expected = y * (dx - np.sum(y * dx, axis=-1, keepdims=True))
    assert np.allclose(np.asarray(t), expected, atol=1e-9)


def test_logsumexp_jvp_matches_directional_derivative():
    rng = np.random.default_rng(7)
    x, dx = rng.standard_normal((3, 4)), rng.standard_normal((3, 4))
    _, t = jvp(lambda a: d_logsumexp(a, axis=-1), (x,), (dx,))
    expected = np.sum(_softmax_ref(x, axis=-1) * dx, axis=-1)
    assert np.allclose(np.asarray(t), expected, atol=1e-9)


def test_softmax_logsumexp_eval_shape():
    assert eval_shape(lambda x: d_softmax(x, axis=-1), S((3, 4))).shape == (3, 4)
    assert eval_shape(lambda x: d_logsumexp(x, axis=-1), S((3, 4))).shape == (3,)

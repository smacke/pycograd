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
    eval_shape,
    gelu,
    grad,
    jvp,
    log_softmax,
    logsumexp,
    relu,
    silu,
    softmax,
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

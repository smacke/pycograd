# -*- coding: utf-8 -*-
"""Tensor-contraction ops (np.dot general, np.inner, np.tensordot) -- each lowers to einsum.
Gradients vs finite differences, plus forward-mode (jvp), batching (vmap), and shape
inference (eval_shape).

Module-level functions so pyccolo can re-instrument them from source."""
import numpy as np
import pytest

from pycograd import eval_shape, grad, jvp, vmap

_rng = np.random.default_rng(0)


def _fd_grad(f, args, k):
    base = [np.asarray(a, dtype=float) for a in args]
    out = np.zeros(base[k].shape)
    eps = 1e-6
    for i in range(base[k].size):
        ap = [a.copy() for a in base]
        am = [a.copy() for a in base]
        ap[k].flat[i] += eps
        am[k].flat[i] -= eps
        out.flat[i] = (f(*ap) - f(*am)) / (2 * eps)
    return out


def s_dot(a, b):
    return np.sum(np.dot(a, b))


def s_inner(a, b):
    return np.sum(np.inner(a, b))


def s_tensordot(a, b):
    return np.sum(np.tensordot(a, b, axes=[(1,), (0,)]))


CASES = [
    (s_dot, [_rng.standard_normal((2, 3)), _rng.standard_normal((3, 4))]),  # 2D@2D
    (s_dot, [_rng.standard_normal(3), _rng.standard_normal(3)]),  # 1D.1D
    (s_dot, [_rng.standard_normal((2, 2, 3)), _rng.standard_normal(3)]),  # ND@1D
    (s_inner, [_rng.standard_normal((2, 3)), _rng.standard_normal((4, 3))]),
    (s_tensordot, [_rng.standard_normal((2, 3)), _rng.standard_normal((3, 4))]),
]


@pytest.mark.parametrize("fn, args", CASES)
def test_contraction_grad_vs_fd(fn, args):
    g = grad(fn, [0, 1])(*args)
    for k in range(2):
        assert np.allclose(np.asarray(g[k]), _fd_grad(fn, args, k), atol=1e-5)


def test_dot_jvp_matches_fd():
    a = _rng.standard_normal((2, 3))
    b = _rng.standard_normal((3, 4))
    va = _rng.standard_normal((2, 3))
    _, t = jvp(s_dot, (a, b), (va, np.zeros_like(b)))
    fd = (s_dot(a + 1e-6 * va, b) - s_dot(a - 1e-6 * va, b)) / 2e-6
    assert np.isclose(float(np.asarray(t)), fd, atol=1e-5)


def test_dot_vmap_and_eval_shape():
    a = _rng.standard_normal((2, 3))
    b = _rng.standard_normal((3, 4))
    out = np.asarray(vmap(lambda x: np.dot(x, b))(np.stack([a, a + 0.1])))
    assert out.shape == (2, 2, 4)
    assert np.allclose(out[0], a @ b)
    assert eval_shape(lambda x, y: np.dot(x, y), a, b).shape == (2, 4)
    assert eval_shape(lambda x, y: np.inner(x, y), a, np.zeros((4, 3))).shape == (2, 4)


# --- np.outer (ravel/einsum composition) ------------------------------------
def s_outer(a, b):
    return np.sum(np.outer(a, b) ** 2)


def test_outer_grad_and_shape():
    a = _rng.standard_normal(7)
    b = _rng.standard_normal(5)
    ga, gb = grad(s_outer, [0, 1])(a, b)
    assert np.allclose(np.asarray(ga), _fd_grad(s_outer, [a, b], 0), atol=1e-5)
    assert np.allclose(np.asarray(gb), _fd_grad(s_outer, [a, b], 1), atol=1e-5)
    # outer flattens both operands first
    M = _rng.standard_normal((2, 3))
    assert eval_shape(lambda x, y: np.outer(x, y), M, b).shape == (6, 5)
    out = np.asarray(vmap(lambda x: np.outer(x, b))(np.stack([a, a + 0.1])))
    assert out.shape == (2, 7, 5) and np.allclose(out[0], np.outer(a, b))


# --- np.cross / np.kron (bilinear compositions; lower in graph capture too) --------------
def s_cross(a, b):
    return np.sum(np.cross(a, b) ** 2)


def s_kron(a, b):
    return np.sum(np.kron(a, b) ** 2)


def test_cross_grad_jvp_vmap():
    a = _rng.standard_normal((3, 3))
    b = _rng.standard_normal((3, 3))
    ga, gb = grad(s_cross, [0, 1])(a, b)
    assert np.allclose(np.asarray(ga), _fd_grad(s_cross, [a, b], 0), atol=1e-5)
    assert np.allclose(np.asarray(gb), _fd_grad(s_cross, [a, b], 1), atol=1e-5)
    # axis= argument and shape
    assert eval_shape(lambda x, y: np.cross(x, y, axis=0), a, b).shape == (3, 3)
    out = np.asarray(vmap(lambda x: np.cross(x, b))(np.stack([a, a + 0.1])))
    assert out.shape == (2, 3, 3) and np.allclose(out[0], np.cross(a, b))


def test_kron_grad_and_shapes():
    A = _rng.standard_normal((3, 2))
    B = _rng.standard_normal((2, 4))
    ga, gb = grad(s_kron, [0, 1])(A, B)
    assert np.allclose(np.asarray(ga), _fd_grad(s_kron, [A, B], 0), atol=1e-5)
    assert np.allclose(np.asarray(gb), _fd_grad(s_kron, [A, B], 1), atol=1e-5)
    assert eval_shape(lambda x, y: np.kron(x, y), A, B).shape == (6, 8)


def _g_cross(x):
    return np.sum(np.cross(x, np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 1]])) ** 2)


def _g_kron(x):
    return np.sum(np.kron(x, np.array([[1.0, 2], [3, 4]])) ** 2)


@pytest.mark.parametrize("fn", [_g_cross, _g_kron])
def test_cross_kron_capture_grad(fn):
    # cross/kron are compositions, so they lower at graph capture and value_and_grad(capture(f))
    # matches the eager gradient.
    from pycograd import capture, value_and_grad

    X = _rng.standard_normal((3, 3))
    _, gg = value_and_grad(capture(fn, X))(X)
    _, ge = value_and_grad(fn)(X)
    assert np.allclose(np.asarray(gg[0]), np.asarray(ge[0]), atol=1e-7)

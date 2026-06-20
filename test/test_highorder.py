# -*- coding: utf-8 -*-
"""Higher-order reverse-mode AD (Phase 1: forward-over-reverse).

``jvp(grad(f))`` is a Hessian-vector product and ``jacfwd(grad(f))`` is the full
Hessian: the differentiable backward records the gradient computation on the enclosing
``jvp`` level, so the surrounding forward transform differentiates it.

Functions/constants are at MODULE level so pyccolo can re-instrument them from source
(an enclosing-function closure would not be preserved).
"""
import numpy as np

from pycograd import grad, jacfwd, jvp


def _rng(seed=0):
    return np.random.default_rng(seed)


def _hessian_fd(f, x, eps=1e-5):
    """Central finite-difference Hessian of scalar ``f`` (via differences of ``grad``)."""
    n = x.size

    def g(z):
        return np.asarray(grad(f)(z)[0])

    H = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        H[:, j] = (g(x + eps * e) - g(x - eps * e)) / (2 * eps)
    return H


# ---------------------------------------------------------------------------
# A symmetric quadratic 0.5 xᵀ A x: gradient A x, Hessian A.
# ---------------------------------------------------------------------------
_A = _rng(30).standard_normal((4, 4))
_A = _A + _A.T


def quad(x):
    return 0.5 * np.sum((x @ _A) * x)


def test_hessian_of_quadratic_is_A():
    x = _rng(1).standard_normal(4)
    H = np.asarray(jacfwd(grad(quad))(x)).reshape(4, 4)
    assert np.allclose(H, _A, atol=1e-6)


def test_hvp_of_quadratic_is_Av():
    x = _rng(2).standard_normal(4)
    v = _rng(3).standard_normal(4)
    hvp = np.asarray(jvp(grad(quad), (x,), (v,))[1][0])
    assert np.allclose(hvp, _A @ v, atol=1e-8)


# ---------------------------------------------------------------------------
# HVP vs a central finite-difference of grad.
# ---------------------------------------------------------------------------
def f_mixed(x):
    return np.sum(np.tanh(x) * x + np.exp(-x) + np.sin(x) * x**2)


def test_hvp_vs_finite_difference():
    x = _rng(4).standard_normal(5)
    v = _rng(5).standard_normal(5)
    hvp = np.asarray(jvp(grad(f_mixed), (x,), (v,))[1][0])

    def g(z):
        return np.asarray(grad(f_mixed)(z)[0])

    eps = 1e-5
    fd = (g(x + eps * v) - g(x - eps * v)) / (2 * eps)
    assert np.allclose(hvp, fd, atol=1e-5)


# ---------------------------------------------------------------------------
# Second derivative through forward-over-reverse: d/dx grad(sum sin) = -sin(x).
# ---------------------------------------------------------------------------
def sin_sum(x):
    return np.sum(np.sin(x))


def test_second_derivative_of_sin():
    x = _rng(6).standard_normal(5)
    v = _rng(7).standard_normal(5)
    hvp = np.asarray(jvp(grad(sin_sum), (x,), (v,))[1][0])
    assert np.allclose(hvp, -np.sin(x) * v, atol=1e-8)


# ---------------------------------------------------------------------------
# A small MLP scalar loss: Hessian is symmetric and matches finite differences.
# ---------------------------------------------------------------------------
_W1 = _rng(40).standard_normal((3, 5))
_B1 = _rng(41).standard_normal(5)
_W2 = _rng(42).standard_normal((5, 1))


def mlp_loss(x):
    h = np.tanh(x @ _W1 + _B1)
    return np.sum(h @ _W2)


def test_mlp_hessian_symmetric_and_matches_fd():
    x = _rng(8).standard_normal(3)
    H = np.asarray(jacfwd(grad(mlp_loss))(x)).reshape(3, 3)
    assert np.allclose(H, H.T, atol=1e-6)
    assert np.allclose(H, _hessian_fd(mlp_loss, x), atol=1e-4)


# ---------------------------------------------------------------------------
# Gather: the second derivative flows through x[slice] ** 2 too.
# ---------------------------------------------------------------------------
def gather_quad(x):
    return np.sum(x[1:4] ** 2) + x[0]


def test_hvp_through_gather():
    x = _rng(9).standard_normal(5)
    v = _rng(10).standard_normal(5)
    hvp = np.asarray(jvp(grad(gather_quad), (x,), (v,))[1][0])
    expected = np.zeros(5)
    expected[1:4] = 2.0 * v[1:4]
    assert np.allclose(hvp, expected, atol=1e-8)

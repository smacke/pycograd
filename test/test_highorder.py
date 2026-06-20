# -*- coding: utf-8 -*-
"""Higher-order reverse-mode AD.

Phase 1 (forward-over-reverse): ``jvp(grad(f))`` is a Hessian-vector product and
``jacfwd(grad(f))`` is the full Hessian -- the differentiable backward records the
gradient computation on the enclosing ``jvp`` level, so the surrounding forward
transform differentiates it.

Phase 2 (reverse-over-reverse): a literal ``grad(grad(f))`` -- the inner ``grad`` runs
its differentiable backward (because the outer ``grad`` pushes a reverse marker level),
producing a cotangent graph the outer ``grad`` then walks. ``jacrev(grad(f))`` is the
full reverse-over-reverse Hessian, which must agree with the Phase-1
``jacfwd(grad(f))`` Hessian and with finite differences.

Functions/constants are at MODULE level so pyccolo can re-instrument them from source
(an enclosing-function closure would not be preserved).
"""
import numpy as np

from pycograd import grad, jacfwd, jacrev, jvp


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


# ===========================================================================
# Phase 2: reverse-over-reverse (literal ``grad(grad(f))``).
# ===========================================================================


# ---------------------------------------------------------------------------
# grad(grad(sum sin)) == -sin: the inner grad's differentiable backward builds a
# cotangent graph (cos(x)) the outer grad differentiates (d/dx cos = -sin). Scalarized
# with a sum so the outer grad sees a scalar (its gradient is the diagonal of the
# Hessian of sum(sin), which is -sin(x)).
# ---------------------------------------------------------------------------
def grad_sin_sum_total(x):
    return np.sum(grad(sin_sum)(x)[0])


def test_grad_of_grad_sin_is_minus_sin():
    x = _rng(6).standard_normal(5)
    g2 = np.asarray(grad(grad_sin_sum_total)(x)[0])
    assert np.allclose(g2, -np.sin(x), atol=1e-8)


# ---------------------------------------------------------------------------
# A reverse-mode gradient *vector* of a scalar function -- the function jacrev
# differentiates again to form the reverse-over-reverse Hessian.
# ---------------------------------------------------------------------------
def quad_grad(x):
    return grad(quad)(x)[0]


def mlp_grad(x):
    return grad(mlp_loss)(x)[0]


def test_reverse_hessian_of_quadratic_matches_forward_and_fd():
    x = _rng(1).standard_normal(4)
    Hrev = np.asarray(jacrev(quad_grad)(x)).reshape(4, 4)
    Hfwd = np.asarray(jacfwd(grad(quad))(x)).reshape(4, 4)
    assert np.allclose(Hrev, _A, atol=1e-6)  # exact: Hessian of 0.5 xᵀAx is A
    assert np.allclose(
        Hrev, Hfwd, atol=1e-6
    )  # reverse-over-reverse == forward-over-rev
    assert np.allclose(Hrev, _hessian_fd(quad, x), atol=1e-4)


def test_reverse_hessian_of_mlp_matches_forward_and_fd():
    x = _rng(8).standard_normal(3)
    Hrev = np.asarray(jacrev(mlp_grad)(x)).reshape(3, 3)
    Hfwd = np.asarray(jacfwd(grad(mlp_loss))(x)).reshape(3, 3)
    assert np.allclose(Hrev, Hrev.T, atol=1e-6)  # Hessians are symmetric
    assert np.allclose(Hrev, Hfwd, atol=1e-6)
    assert np.allclose(Hrev, _hessian_fd(mlp_loss, x), atol=1e-4)


# ---------------------------------------------------------------------------
# Reverse Hessian built by literal grad(grad): grad of (grad(f) . e_i) is column i of
# the Hessian -- a literal ``grad`` differentiating an inner ``grad``. The direction e_i
# is a *second* (non-differentiated) argument so no closure is captured.
# ---------------------------------------------------------------------------
def quad_grad_dot(x, v):
    return np.sum(grad(quad)(x)[0] * v)


def test_reverse_hessian_via_literal_grad_of_grad():
    x = _rng(2).standard_normal(4)
    cols = [np.asarray(grad(quad_grad_dot)(x, np.eye(4)[j])[0]) for j in range(4)]
    H = np.stack(cols, axis=1)
    assert np.allclose(H, _A, atol=1e-6)


# ---------------------------------------------------------------------------
# A gradient-penalty pattern: a loss that itself contains a grad term. The outer grad
# must differentiate through the inner grad (reverse-over-reverse). Checked vs finite
# differences of the (scalar) penalized loss.
# ---------------------------------------------------------------------------
_GP_LAMBDA = 0.3


def grad_penalty_loss(x):
    base = mlp_loss(x)
    g = grad(mlp_loss)(x)[0]
    return base + _GP_LAMBDA * np.sum(g * g)


def _grad_fd(f, x, eps=1e-5):
    n = x.size
    fd = np.zeros(n)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        fd[j] = (f(x + eps * e) - f(x - eps * e)) / (2 * eps)
    return fd


def test_gradient_penalty_matches_finite_difference():
    x = _rng(11).standard_normal(3)
    g = np.asarray(grad(grad_penalty_loss)(x)[0])
    fd = _grad_fd(grad_penalty_loss, x)
    assert np.allclose(g, fd, atol=1e-4)


# ---------------------------------------------------------------------------
# A single top-level grad must be numerically unchanged by the Phase-2 plumbing
# (it must NOT start taking the differentiable backward when not nested).
# ---------------------------------------------------------------------------
def test_top_level_grad_unchanged():
    x = _rng(6).standard_normal(5)
    g = np.asarray(grad(sin_sum)(x)[0])
    assert np.allclose(g, np.cos(x), atol=1e-12)

    xq = _rng(1).standard_normal(4)
    gq = np.asarray(grad(quad)(xq)[0])
    assert np.allclose(gq, xq @ _A, atol=1e-12)

    xm = _rng(8).standard_normal(3)
    gm = np.asarray(grad(mlp_loss)(xm)[0])
    assert np.allclose(gm, np.asarray(jacfwd(mlp_loss)(xm)), atol=1e-10)


# ---------------------------------------------------------------------------
# grad requires a scalar output: nesting grad without scalarizing the inner
# gradient is ill-posed (as in JAX) and must report a clear error, not an obscure
# "Var has no array conversion".
# ---------------------------------------------------------------------------
def test_grad_of_nonscalar_raises_clear_error():
    import pytest

    with pytest.raises(TypeError, match="returning a single scalar"):
        grad(grad(sin_sum))(_rng(0).standard_normal(3))

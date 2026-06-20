# -*- coding: utf-8 -*-
"""Forward-mode AD (``jvp``): tangents must match central finite differences, the primal
output must equal plain ``f``, and the forward Jacobian (``jacfwd``) must agree with the
reverse-mode ``grad``.

Functions are defined at MODULE level so pyccolo can re-instrument them from source
(an enclosing-function closure would not be preserved).
"""
import numpy as np

from pycograd import grad, jacfwd, jvp, vmap
from pycograd.forward import _JVP


def _rng(seed=0):
    return np.random.default_rng(seed)


def _fd(f, x, v, eps=1e-6):
    """Central finite-difference directional derivative of ``f`` at ``x`` along ``v``."""
    return (np.asarray(f(x + eps * v)) - np.asarray(f(x - eps * v))) / (2 * eps)


# ---------------------------------------------------------------------------
# Module-level functions under test.
# ---------------------------------------------------------------------------
def f_elementwise(x):
    return np.sum(np.exp(x) * np.tanh(x) + np.sin(x) * x)


def f_pow(x):
    return np.sum(x**3 + np.sqrt(np.abs(x) + 0.1))


def f_reduction(x):
    return np.mean(x, axis=0).sum() + np.var(x) + np.std(x)


def f_max(x):
    return np.sum(np.max(x, axis=0)) + np.min(x)


def f_matmul(x, W):
    return np.sum(np.tanh(x @ W))


def f_getitem(x):
    return np.sum(x[1:4] * 2.0) + x[0]


def f_where_clip(x):
    return np.sum(np.where(x > 0, x * x, x) + np.clip(x, -0.5, 0.5))


def f_concat(x):
    return np.sum(np.concatenate([x, x * 2.0]))


def mlp(x, W1, b1, W2):
    h = np.tanh(x @ W1 + b1)
    return np.sum(h @ W2)


def f_scalar(x):
    return np.sum(np.sin(x) * x**2 + np.exp(-x))


def vec_out(x):
    return np.stack([np.sum(x * x), np.sum(np.sin(x)), np.sum(np.exp(x))])


def double_tanh(x):
    return np.tanh(x) * 2.0


# A fixed tangent direction (module-level so pyccolo can re-instrument the closure-free
# function below from source).
VROW = np.array([1.0, 0.5, -0.3])


def jvp_tangent_row(xrow):
    return jvp(double_tanh, (xrow,), (VROW,))[1]


# ---------------------------------------------------------------------------
# jvp tangents vs finite differences.
# ---------------------------------------------------------------------------
def _check_fd(f, x, v):
    primal, tangent = jvp(f, (x,), (v,))
    assert np.allclose(
        np.asarray(primal), np.asarray(f(x)), atol=1e-8
    ), "primal mismatch"
    assert np.allclose(np.asarray(tangent), _fd(f, x, v), atol=1e-4), "tangent mismatch"


def test_jvp_elementwise():
    rng = _rng(0)
    x, v = rng.standard_normal(6), rng.standard_normal(6)
    _check_fd(f_elementwise, x, v)


def test_jvp_pow():
    rng = _rng(1)
    x, v = rng.standard_normal(5), rng.standard_normal(5)
    _check_fd(f_pow, x, v)


def test_jvp_reduction():
    rng = _rng(2)
    x, v = rng.standard_normal((4, 3)), rng.standard_normal((4, 3))
    _check_fd(f_reduction, x, v)


def test_jvp_max_min():
    rng = _rng(3)
    x, v = rng.standard_normal((5, 4)), rng.standard_normal((5, 4))
    _check_fd(f_max, x, v)


def test_jvp_getitem():
    rng = _rng(4)
    x, v = rng.standard_normal(6), rng.standard_normal(6)
    _check_fd(f_getitem, x, v)


def test_jvp_where_clip():
    rng = _rng(5)
    x, v = rng.standard_normal(7), rng.standard_normal(7)
    _check_fd(f_where_clip, x, v)


def test_jvp_concatenate():
    rng = _rng(6)
    x, v = rng.standard_normal(5), rng.standard_normal(5)
    _check_fd(f_concat, x, v)


def test_jvp_matmul():
    rng = _rng(7)
    x, W = rng.standard_normal((3, 4)), rng.standard_normal((4, 2))
    vx, vW = rng.standard_normal((3, 4)), rng.standard_normal((4, 2))
    primal, tangent = jvp(f_matmul, (x, W), (vx, vW))
    assert np.allclose(np.asarray(primal), np.asarray(f_matmul(x, W)), atol=1e-8)
    eps = 1e-6
    fd = (
        f_matmul(x + eps * vx, W + eps * vW) - f_matmul(x - eps * vx, W - eps * vW)
    ) / (2 * eps)
    assert np.allclose(np.asarray(tangent), fd, atol=1e-4)


def test_jvp_mlp():
    rng = _rng(8)
    x = rng.standard_normal(4)
    W1, b1, W2 = (
        rng.standard_normal((4, 5)),
        rng.standard_normal(5),
        rng.standard_normal((5, 2)),
    )
    vx = rng.standard_normal(4)
    primals = (x, W1, b1, W2)
    tangents = (vx, np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2))
    primal, tangent = jvp(mlp, primals, tangents)
    assert np.allclose(np.asarray(primal), np.asarray(mlp(*primals)), atol=1e-8)
    eps = 1e-6
    fd = (mlp(x + eps * vx, W1, b1, W2) - mlp(x - eps * vx, W1, b1, W2)) / (2 * eps)
    assert np.allclose(np.asarray(tangent), fd, atol=1e-4)


def test_jvp_primal_equals_f():
    rng = _rng(9)
    x = rng.standard_normal(5)
    primal, _ = jvp(vec_out, (x,), (np.ones_like(x),))
    assert np.allclose(np.asarray(primal), np.asarray(vec_out(x)))


# ---------------------------------------------------------------------------
# jacfwd vs reverse-mode grad (forward and reverse Jacobians agree).
# ---------------------------------------------------------------------------
def test_jacfwd_matches_grad_scalar():
    rng = _rng(10)
    x = rng.standard_normal(5)
    J = jacfwd(f_scalar)(x)
    g = grad(f_scalar)(x)[0]
    assert np.allclose(J, g, atol=1e-6)


def test_jacfwd_vector_output():
    rng = _rng(11)
    x = rng.standard_normal(4)
    J = jacfwd(vec_out)(x)
    assert J.shape == (3, 4)
    eps = 1e-6
    fdJ = np.zeros((3, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = 1.0
        fdJ[:, j] = (vec_out(x + eps * e) - vec_out(x - eps * e)) / (2 * eps)
    assert np.allclose(J, fdJ, atol=1e-4)


# ---------------------------------------------------------------------------
# Composition: vmap over jvp.
# ---------------------------------------------------------------------------
def test_vmap_of_jvp():
    rng = _rng(12)
    xs = rng.standard_normal((6, 3))
    got = np.asarray(vmap(jvp_tangent_row)(xs))
    ref = np.stack([np.asarray(jvp_tangent_row(xs[i])) for i in range(6)])
    assert got.shape == (6, 3)
    assert np.allclose(got, ref, atol=1e-8)


# ---------------------------------------------------------------------------
# Coverage parity: every intercepted primitive has a jvp rule.
# ---------------------------------------------------------------------------
def test_coverage_matches_intercept():
    from pycograd.ops import _INTERCEPT

    assert set(_JVP) == set(_INTERCEPT)


# ---------------------------------------------------------------------------
# Composition: vmap over the *tangent* (a batch of directions at a fixed point).
# Regression for _jvp_inputs coercing a tracer tangent to a concrete array.
# ---------------------------------------------------------------------------
_DIR_X0 = _rng(20).standard_normal(3)


def _dir_scalar(x):
    return np.sum(np.tanh(x) * x)


def jvp_dir(direction):
    return jvp(_dir_scalar, (_DIR_X0,), (direction,))[1]


def test_vmap_over_tangent():
    dirs = _rng(21).standard_normal((5, 3))
    got = np.asarray(vmap(jvp_dir)(dirs))
    ref = np.array([float(jvp_dir(dirs[k])) for k in range(5)])
    assert got.shape == (5,)
    assert np.allclose(got, ref, atol=1e-8)


# ---------------------------------------------------------------------------
# Second-order forward: jvp of jvp. For a quadratic 0.5 xᵀ A x the second
# directional derivative along (u, v) is exactly uᵀ A v.
# ---------------------------------------------------------------------------
_A = _rng(30).standard_normal((3, 3))
_A = _A + _A.T


def _quad(x):
    return 0.5 * np.sum((x @ _A) * x)


_DDX = _rng(31).standard_normal(3)
_DDV = _rng(32).standard_normal(3)


def _dir_quad(z):
    return jvp(_quad, (z,), (_DDV,))[1]


def test_jvp_of_jvp_second_order():
    u = _rng(33).standard_normal(3)
    second = float(jvp(_dir_quad, (_DDX,), (u,))[1])
    assert np.isclose(second, float(u @ _A @ _DDV), atol=1e-8)

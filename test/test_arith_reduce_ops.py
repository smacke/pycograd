# -*- coding: utf-8 -*-
"""The numpy *function* forms of the arithmetic operators (np.add/subtract/multiply/divide/
negative/power), np.mod (and the ``%`` operator), and the np.prod reduction -- gradients vs
finite differences, plus forward-mode (jvp), batching (vmap), and shape inference (eval_shape).

Module-level functions so pyccolo can re-instrument them from source."""
import numpy as np
import pytest

from pycograd import eval_shape, grad, jvp, vmap

_x = np.array([1.3, 2.7, 3.1])
_y = np.array([0.7, 1.1, 0.9])


def _fd_dir(f, x, v, eps=1e-6):
    return (f(x + eps * v) - f(x - eps * v)) / (2 * eps)


# --- numpy function-form arithmetic ----------------------------------------
def f_add(x, y):
    return np.sum(np.add(x, y))


def f_multiply(x, y):
    return np.sum(np.multiply(x, y))


def f_divide(x, y):
    return np.sum(np.divide(x, y))


def f_subtract(x, y):
    return np.sum(np.subtract(x, y))


def f_negative(x):
    return np.sum(np.negative(x))


def f_power(x, y):
    return np.sum(np.power(x, y))


@pytest.mark.parametrize(
    "fn, expect_gx",
    [
        (f_add, np.ones(3)),
        (f_multiply, _y),
        (f_divide, 1.0 / _y),
        (f_subtract, np.ones(3)),
    ],
)
def test_function_form_binary_grads(fn, expect_gx):
    gx = np.asarray(grad(fn, 0)(_x, _y))
    assert np.allclose(gx, expect_gx)


def test_function_form_negative_and_power():
    assert np.allclose(np.asarray(grad(f_negative)(_x)[0]), -np.ones(3))
    gx = np.asarray(grad(f_power, 0)(_x, _y))
    assert np.allclose(gx, _y * _x ** (_y - 1))


# --- mod --------------------------------------------------------------------
def f_mod_sum(x, y):
    return np.sum(np.mod(x, y))


def f_modop(x, y):
    return np.sum(x % y)


def test_mod_grad():
    for f in (f_mod_sum, f_modop):
        gx, gy = grad(f, [0, 1])(_x, _y)
        assert np.allclose(gx, 1.0)
        assert np.allclose(gy, -np.floor(_x / _y))


def test_mod_jvp_and_vmap_and_shape():
    _, t = jvp(f_mod_sum, (_x, _y), (np.ones_like(_x), np.zeros_like(_y)))
    assert np.isclose(float(np.asarray(t)), 3.0)
    out = np.asarray(vmap(lambda a: np.mod(a, _y))(np.stack([_x, _x + 0.1])))
    assert out.shape == (2, 3) and np.allclose(out[0], np.mod(_x, _y))
    assert eval_shape(f_mod_sum, _x, _y).shape == ()


# --- prod -------------------------------------------------------------------
def f_prod(x):
    return np.prod(x)


def f_prod_axis(x):
    return np.sum(np.prod(x, axis=1))


def test_prod_grad_vs_fd():
    x = np.array([1.5, 2.0, 0.5, 3.0])
    g = np.asarray(grad(f_prod)(x)[0])
    v = np.eye(4)
    fd = np.array([_fd_dir(f_prod, x, v[i]) for i in range(4)])
    assert np.allclose(g, fd, atol=1e-5)


def test_prod_axis_jvp_vmap_shape():
    X = np.array([[1.5, 2.0, 0.5], [3.0, 1.0, 2.0]])
    assert np.asarray(grad(f_prod_axis)(X)[0]).shape == (2, 3)
    _, t = jvp(f_prod, (X[0],), (np.ones(3),))
    assert np.isfinite(float(np.asarray(t)))
    out = np.asarray(vmap(lambda a: np.prod(a))(X))
    assert np.allclose(out, np.prod(X, axis=1))
    assert eval_shape(f_prod, X[0]).shape == ()

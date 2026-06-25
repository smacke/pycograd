# -*- coding: utf-8 -*-
"""The autograd/JAX-style differential-operator surface: ``argnum`` + ``**kwargs`` on
``grad``/``value_and_grad`` and the ``jacobian``/``hessian``/``elementwise_grad``/
``make_jvp``/``make_vjp`` operators.

Functions are at MODULE level so pyccolo can re-instrument them from source.
"""
import numpy as np

import pycograd as pg


def f(x, y):
    return np.sum(x * x + np.sin(y))


def h(x, scale=2.0):
    return np.sum(scale * x * x)


def vecfun(x):
    # vector output via array-native ops (np.array of Vars is a separate, unsupported path)
    return np.concatenate([x**3, np.sin(x)])


def quad(x):
    return 0.5 * np.sum(x * x)


A = np.array([1.0, 2.0, 3.0])
B = np.array([0.5, 1.0, 1.5])


def test_grad_argnum_int_is_bare():
    assert np.allclose(pg.grad(f, 0)(A, B), 2 * A)
    assert np.allclose(pg.grad(f, 1)(A, B), np.cos(B))


def test_grad_argnum_sequence_is_tuple_in_order():
    g = pg.grad(f, [1, 0])(A, B)
    assert len(g) == 2
    assert np.allclose(g[0], np.cos(B)) and np.allclose(g[1], 2 * A)


def test_grad_default_returns_tuple_over_all_args():
    g = pg.grad(f)(A, B)
    assert isinstance(g, tuple) and len(g) == 2
    assert np.allclose(g[0], 2 * A) and np.allclose(g[1], np.cos(B))


def test_kwargs_held_fixed():
    assert np.allclose(pg.grad(h, 0)(A, scale=3.0), 2 * 3.0 * A)


def test_value_and_grad_argnum():
    v, g = pg.value_and_grad(f, 0)(A, B)
    assert np.isclose(v, f(A, B)) and np.allclose(g, 2 * A)


def test_jacobian_vector_output():
    J = np.asarray(pg.jacobian(vecfun)(A))
    assert J.shape == (6, 3)
    assert np.allclose(J[:3], np.diag(3 * A**2))
    assert np.allclose(J[3:], np.diag(np.cos(A)))


def test_jacobian_of_scalar_is_gradient():
    J = np.asarray(pg.jacobian(quad)(A))
    assert np.allclose(J, A)


def test_hessian_of_quadratic_is_identity():
    H = np.asarray(pg.hessian(quad)(A)).reshape(3, 3)
    assert np.allclose(H, np.eye(3))


def test_elementwise_grad():
    eg = pg.elementwise_grad(np.sin)(A)
    assert np.allclose(eg, np.cos(A))


def test_make_jvp_matches_jacobian_column():
    v = np.array([1.0, 0.0, 0.0])
    ans, jv = pg.make_jvp(vecfun)(A)(v)
    assert np.allclose(np.asarray(ans), vecfun(A))
    expect = np.concatenate([3 * A**2, np.cos(A)]) * np.concatenate([v, v])
    assert np.allclose(np.asarray(jv), expect)


def test_make_vjp_pulls_back_cotangent():
    vjp_fn, ans = pg.make_vjp(vecfun)(A)
    assert np.allclose(np.asarray(ans), vecfun(A))
    g = np.zeros(6)
    g[0] = 1.0
    assert np.allclose(np.asarray(vjp_fn(g)), [3 * A[0] ** 2, 0, 0])


def test_make_vjp_reusable_across_cotangents():
    vjp_fn, _ = pg.make_vjp(vecfun)(A)
    g0 = np.zeros(6)
    g0[0] = 1.0
    g3 = np.zeros(6)
    g3[3] = 1.0
    assert np.allclose(np.asarray(vjp_fn(g0)), [3 * A[0] ** 2, 0, 0])
    assert np.allclose(np.asarray(vjp_fn(g3)), [np.cos(A[0]), 0, 0])

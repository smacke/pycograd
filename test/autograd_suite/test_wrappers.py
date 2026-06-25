# -*- coding: utf-8 -*-
"""Ported from autograd ``tests/test_wrappers.py`` (MIT). The differential-operator surface:
grad / value_and_grad / hessian / jacobian / elementwise_grad / make_hvp /
hessian_tensor_product / tensor_jacobian_product / make_jvp / make_ggnvp / grad_and_aux.

Skipped: autograd's deprecated ``defgrad``/``defvjp``/``quick_grad_check`` (custom-primitive
API pycograd lacks), dtype-preservation (pycograd works in float64), and the few cases that
differentiate *through* ``value_and_grad`` (higher-order, out of scope for the first-order
checker)."""
from functools import partial

import numpy as np
import numpy.random as npr
import pytest

from pycograd.tensor import Var

from ._compat import (
    elementwise_grad,
    grad,
    grad_and_aux,
    hessian,
    hessian_vector_product,
    jacobian,
    make_ggnvp,
    make_hvp,
    make_jvp,
    tensor_jacobian_product,
    value_and_grad,
)
from ._test_util import check_equivalent, check_grads

npr.seed(1)


def test_return_both():
    fun = lambda x: 3.0 * x**3.2
    d_fun = grad(fun)
    f_and_d_fun = value_and_grad(fun)

    test_x = 1.7
    f, d = f_and_d_fun(test_x)
    assert f == fun(test_x)
    assert d == d_fun(test_x)


def test_value_and_grad():
    fun = lambda x: np.sum(np.sin(x) ** 2)
    dfun = grad(fun)
    dfun_both = value_and_grad(fun)
    x = npr.randn(5)
    assert not isinstance(dfun_both(x)[0], Var)
    check_equivalent(fun(x), dfun_both(x)[0])
    check_equivalent(dfun(x), dfun_both(x)[1])


def test_hessian():
    # Check Hessian of a quadratic function.
    D = 5
    H = npr.randn(D, D)

    def fun(x):
        return np.dot(np.dot(x, H), x)

    hess = hessian(fun)
    x = npr.randn(D)
    check_equivalent(np.asarray(hess(x)).reshape(D, D), H + H.T)


def test_multigrad():
    def complicated_fun(a, b, c, d, e, f=1.1, g=9.0):
        return a + np.sin(b) + np.cosh(c) + np.cos(d) + np.tan(e) + f + g

    def complicated_fun_3_1(d_b):
        d, b = d_b
        return complicated_fun(A, b, C, d, E, f=F, g=G)

    A = 0.5
    B = -0.3
    C = 0.2
    D = -1.1
    E = 0.7
    F = 0.6
    G = -0.1

    wrapped = grad(complicated_fun, argnum=[3, 1])(A, B, C, D, E, f=F, g=G)
    explicit = grad(complicated_fun_3_1)((D, B))
    check_equivalent(wrapped, explicit)


def test_value_and_multigrad():
    def complicated_fun(a, b, c, d, e, f=1.1, g=9.0):
        return a + np.sin(b) + np.cosh(c) + np.cos(d) + np.tan(e) + f + g

    A, B, C, D, E, F, G = 0.5, -0.3, 0.2, -1.1, 0.7, 0.6, -0.1

    dfun = grad(complicated_fun, argnum=[3, 1])
    dfun_both = value_and_grad(complicated_fun, argnum=[3, 1])

    check_equivalent(
        complicated_fun(A, B, C, D, E, f=F, g=G),
        dfun_both(A, B, C, D, E, f=F, g=G)[0],
    )
    check_equivalent(
        dfun(A, B, C, D, E, f=F, g=G), dfun_both(A, B, C, D, E, f=F, g=G)[1]
    )


def test_multigrad_onearg():
    fun = lambda x, y: np.sum(x + np.sin(y))
    packed_fun = lambda xy: np.sum(xy[0] + np.sin(xy[1]))
    A, B = npr.randn(3), npr.randn(3)
    check_equivalent(grad(fun, argnum=[0])(A, B), (grad(packed_fun)((A, B))[0],))


def test_elementwise_grad():
    def simple_fun(a):
        return a + np.sin(a) + np.cosh(a)

    A = npr.randn(10)
    wrapped = elementwise_grad(simple_fun)(A)
    explicit = np.array([grad(simple_fun)(A[i]) for i in range(len(A))])
    check_equivalent(wrapped, explicit)


def test_elementwise_grad_multiple_args():
    def simple_fun(a, b):
        return a + np.sin(a) + np.cosh(b)

    A = 0.9
    B = npr.randn(10)
    argnum = 1
    wrapped = elementwise_grad(simple_fun, argnum)(A, B)
    explicit = np.array([grad(simple_fun, argnum)(A, B[i]) for i in range(len(B))])
    check_equivalent(wrapped, explicit)


def test_hessian_tensor_product():
    fun = lambda a: np.sum(np.sin(a))
    a = npr.randn(5)
    v = npr.randn(5)
    H = hessian(fun)(a)
    check_equivalent(np.dot(np.asarray(H), v), hessian_vector_product(fun)(a, v))


def test_hvp():
    fun = lambda a: np.sum(np.sin(a))
    a = npr.randn(5)
    v = npr.randn(5)
    H = hessian(fun)(a)
    hvp = make_hvp(fun)(a)[0]
    check_equivalent(np.dot(np.asarray(H), v), np.asarray(hvp(v)))


def test_hessian_matrix_product():
    fun = lambda a: np.sum(np.sin(a))
    a = npr.randn(5, 4)
    V = npr.randn(5, 4)
    H = hessian(fun)(a)
    check_equivalent(np.tensordot(np.asarray(H), V), hessian_vector_product(fun)(a, V))


def test_hessian_tensor_product_3d():
    fun = lambda a: np.sum(np.sin(a))
    a = npr.randn(5, 4, 3)
    V = npr.randn(5, 4, 3)
    H = hessian(fun)(a)
    check_equivalent(
        np.tensordot(np.asarray(H), V, axes=np.ndim(V)),
        hessian_vector_product(fun)(a, V),
    )


@pytest.mark.skip(reason="pycograd-gap: np.roll has no VJP rule")
def test_tensor_jacobian_product():
    fun = lambda a: np.roll(np.sin(a), 1)
    a = npr.randn(5)
    V = npr.randn(5)
    J = jacobian(fun)(a)
    check_equivalent(np.dot(V.T, np.asarray(J)), tensor_jacobian_product(fun)(a, V))


@pytest.mark.skip(reason="pycograd-gap: np.roll has no VJP rule")
def test_matrix_jacobian_product():
    fun = lambda a: np.roll(np.sin(a), 1)
    a = npr.randn(5, 4)
    V = npr.randn(5, 4)
    J = jacobian(fun)(a)
    check_equivalent(np.tensordot(V, np.asarray(J)), tensor_jacobian_product(fun)(a, V))


def test_make_jvp():
    A = npr.randn(3, 5)
    x = npr.randn(5)
    v = npr.randn(5)
    fun = lambda x: np.tanh(np.dot(A, x))

    jvp_explicit = lambda x: lambda v: np.dot(np.asarray(jacobian(fun)(x)), v)
    jvp = make_jvp(fun)

    check_equivalent(jvp_explicit(x)(v), jvp(x)(v)[1])


def _make_explicit_ggnvp(f, g=lambda x: 1.0 / 2 * np.dot(x, x)):
    def ggnvp_maker(x):
        J = np.asarray(jacobian(f)(x))
        H = np.asarray(hessian(g)(f(x)))

        def ggnvp(v):
            return np.dot(J.T, np.dot(H, np.dot(J, v)))

        return ggnvp

    return ggnvp_maker


@pytest.mark.skip(
    reason="pycograd higher-order gap: the default GGN g=0.5*dot(x,x) needs "
    "jvp(grad(.)) through np.dot(x, x) (JVPTracer * JVPTracer), a second-order "
    "contraction pycograd's forward-over-reverse does not support"
)
def test_make_ggnvp():
    A = npr.randn(5, 4)
    x = npr.randn(4)
    v = npr.randn(4)

    fun = lambda x: np.dot(A, x)
    check_equivalent(np.asarray(make_ggnvp(fun)(x)(v)), _make_explicit_ggnvp(fun)(x)(v))

    fun2 = lambda x: np.tanh(np.dot(A, x))
    check_equivalent(
        np.asarray(make_ggnvp(fun2)(x)(v)), _make_explicit_ggnvp(fun2)(x)(v)
    )


def test_make_ggnvp_nondefault_g():
    A = npr.randn(5, 4)
    x = npr.randn(4)
    v = npr.randn(4)

    g = lambda y: np.sum(2.0 * y**2 + y**4)

    fun = lambda x: np.dot(A, x)
    check_equivalent(
        np.asarray(make_ggnvp(fun, g)(x)(v)), _make_explicit_ggnvp(fun, g)(x)(v)
    )

    fun2 = lambda x: np.tanh(np.dot(A, x))
    check_equivalent(
        np.asarray(make_ggnvp(fun2, g)(x)(v)), _make_explicit_ggnvp(fun2, g)(x)(v)
    )


def test_grad_and_aux():
    A = npr.randn(5, 4)
    x = npr.randn(4)

    f = lambda x: (np.sum(np.dot(A, x)), x**2)
    g = lambda x: np.sum(np.dot(A, x))

    assert len(grad_and_aux(f)(x)) == 2
    check_equivalent(grad_and_aux(f)(x)[0], grad(g)(x))
    check_equivalent(grad_and_aux(f)(x)[1], x**2)


def test_partial():
    def f(x, y):
        return x

    grad(partial(f, y=1))


def test_custom_where():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    condition = [True, False, True]
    expected = np.array([1.0, 5.0, 3.0])
    result = np.where(condition, x, y)
    check_equivalent(result, expected)


@pytest.mark.skip(
    reason="autograd-internal: deprecated @primitive/defgrad custom-VJP API"
)
def test_deprecated_defgrad_wrapper():
    pass


@pytest.mark.skip(
    reason="autograd-internal: deprecated @primitive/defvjp custom-VJP API"
)
def test_deprecated_defvjp_wrapper():
    pass


@pytest.mark.skip(reason="autograd-internal: deprecated defvjp_is_zero custom-VJP API")
def test_deprecated_defvjp_is_zero_wrapper():
    pass


@pytest.mark.skip(reason="autograd-internal: deprecated quick_grad_check helper")
def test_deprecated_quick_grad_check_wrapper():
    pass


@pytest.mark.skip(
    reason="pycograd-gap: grad does not preserve float32/float16/longdouble/clongdouble "
    "dtypes (works in float64); longdouble/clongdouble unsupported"
)
def test_dtypes():
    pass


@pytest.mark.skip(
    reason="autograd-internal: asserts autograd's grad.__name__/__doc__ naming convention"
)
def test_wrapped_name_and_docs():
    pass

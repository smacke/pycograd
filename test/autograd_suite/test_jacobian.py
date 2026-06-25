# -*- coding: utf-8 -*-
"""Ported from autograd ``tests/test_jacobian.py`` (MIT)."""
import numpy as np
import numpy.random as npr
import pytest

from ._compat import grad, jacobian
from ._test_util import check_grads

npr.seed(1)


def test_jacobian_against_grad():
    fun = lambda x: np.sum(np.sin(x), axis=1, keepdims=True)
    A = npr.randn(1, 3)
    assert np.allclose(grad(fun)(A), jacobian(fun)(A))


@pytest.mark.skip(
    reason="pycograd-gap: np.array([...]) of Var leaves is not a differentiable "
    "constructor (use np.stack/np.concatenate); scalar->vector jacobian needs it"
)
def test_jacobian_scalar_to_vector():
    fun = lambda x: np.array([x, x**2, x**3])
    val = npr.randn()
    assert np.allclose(jacobian(fun)(val), np.array([1.0, 2 * val, 3 * val**2]))


@pytest.mark.skip(
    reason="pycograd-gap: np.array([...]) of Var leaves; vector_fun stacks scalar grads"
)
def test_jacobian_against_stacked_grads():
    scalar_funs = [
        lambda x: np.sum(x**3),
        lambda x: np.prod(np.sin(x) + np.sin(x)),
        lambda x: grad(lambda y: np.exp(y) * np.tanh(x[0]))(x[1]),
    ]

    vector_fun = lambda x: np.array([f(x) for f in scalar_funs])

    x = npr.randn(5)
    jac = jacobian(vector_fun)(x)
    grads = [grad(f)(x) for f in scalar_funs]

    assert np.allclose(jac, np.vstack(grads))


@pytest.mark.skip(
    reason="pycograd-gap: np.outer has no VJP rule; and second-order jacobian(jacobian) "
    "relies on autograd reverse-over-reverse composition pycograd does not support"
)
def test_jacobian_higher_order():
    fun = lambda x: np.sin(np.outer(x, x)) + np.cos(np.dot(x, x))

    assert jacobian(fun)(npr.randn(2)).shape == (2, 2, 2)
    assert jacobian(jacobian(fun))(npr.randn(2)).shape == (2, 2, 2, 2)

    check_grads(lambda x: np.sum(np.sin(jacobian(fun)(x))))(npr.randn(2))
    check_grads(lambda x: np.sum(np.sin(jacobian(jacobian(fun))(x))))(npr.randn(2))

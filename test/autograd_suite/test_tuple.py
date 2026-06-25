# -*- coding: utf-8 -*-
"""Ported from autograd ``tests/test_tuple.py`` (MIT). Gradients through tuple pytrees."""
import numpy as np
import numpy.random as npr
import pytest

from ._compat import ag_isinstance, ag_tuple, grad
from ._test_util import check_grads

npr.seed(1)


def test_getter():
    def fun(input_tuple):
        A = np.sum(input_tuple[0])
        B = np.sum(input_tuple[1])
        C = np.sum(input_tuple[1])
        return A + B + C

    d_fun = grad(fun)
    input_tuple = (npr.randn(5, 6), npr.randn(4, 3), npr.randn(2, 4))

    result = d_fun(input_tuple)
    assert np.allclose(result[0], np.ones((5, 6)))
    assert np.allclose(result[1], 2 * np.ones((4, 3)))
    assert np.allclose(result[2], np.zeros((2, 4)))


def test_grads():
    def fun(input_tuple):
        A = np.sum(np.sin(input_tuple[0]))
        B = np.sum(np.cos(input_tuple[1]))
        return A + B

    input_tuple = (npr.randn(5, 6), npr.randn(4, 3), npr.randn(2, 4))
    check_grads(fun)(input_tuple)


@pytest.mark.skip(
    reason="pycograd-gap: nested higher-order (grad inside the differentiated fun) relies "
    "on autograd reverse-over-reverse composition pycograd does not support"
)
def test_nested_higher_order():
    def outer_fun(x):
        def inner_fun(y):
            return y[0] * y[1]

        return np.sum(np.sin(np.array(grad(inner_fun)(ag_tuple((x, x))))))

    check_grads(outer_fun)(5.0)
    check_grads(grad(outer_fun))(10.0)
    check_grads(grad(grad(outer_fun)))(10.0)


def test_isinstance():
    def fun(x):
        assert ag_isinstance(x, tuple)
        assert ag_isinstance(x, ag_tuple)
        return x[0]

    fun((1.0, 2.0, 3.0))
    grad(fun)((1.0, 2.0, 3.0))

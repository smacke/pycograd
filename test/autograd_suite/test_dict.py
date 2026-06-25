# -*- coding: utf-8 -*-
"""Ported from autograd ``tests/test_dict.py`` (MIT). Gradients through dict pytrees."""
import numpy as np
import numpy.random as npr

from ._compat import ag_dict, ag_isinstance, grad
from ._test_util import check_grads

npr.seed(0)


def test_getter():
    def fun(input_dict):
        A = np.sum(input_dict["item_1"])
        B = np.sum(input_dict["item_2"])
        C = np.sum(input_dict["item_2"])
        return A + B + C

    d_fun = grad(fun)
    input_dict = {
        "item_1": npr.randn(5, 6),
        "item_2": npr.randn(4, 3),
        "item_X": npr.randn(2, 4),
    }

    result = d_fun(input_dict)
    assert np.allclose(result["item_1"], np.ones((5, 6)))
    assert np.allclose(result["item_2"], 2 * np.ones((4, 3)))
    assert np.allclose(result["item_X"], np.zeros((2, 4)))


def test_grads():
    def fun(input_dict):
        A = np.sum(np.sin(input_dict["item_1"]))
        B = np.sum(np.cos(input_dict["item_2"]))
        return A + B

    input_dict = {
        "item_1": npr.randn(5, 6),
        "item_2": npr.randn(4, 3),
        "item_X": npr.randn(2, 4),
    }
    check_grads(fun)(input_dict)


def test_iter():
    def fun(input_dict):
        A = 0.0
        B = 0.0
        for i, k in enumerate(sorted(input_dict)):
            A = A + np.sum(np.sin(input_dict[k])) * (i + 1.0)
            B = B + np.sum(np.cos(input_dict[k]))
        return A + B

    input_dict = {
        "item_1": npr.randn(5, 6),
        "item_2": npr.randn(4, 3),
        "item_X": npr.randn(2, 4),
    }
    check_grads(fun)(input_dict)


def test_items_values_keys():
    def fun(input_dict):
        A = 0.0
        B = 0.0
        for i, (k, v) in enumerate(sorted(input_dict.items())):
            A = A + np.sum(np.sin(v)) * (i + 1.0)
            B = B + np.sum(np.cos(v))
        for v in input_dict.values():
            A = A + np.sum(np.sin(v))
        for k in sorted(input_dict.keys()):
            A = A + np.sum(np.cos(input_dict[k]))
        return A + B

    input_dict = {
        "item_1": npr.randn(5, 6),
        "item_2": npr.randn(4, 3),
        "item_X": npr.randn(2, 4),
    }
    check_grads(fun)(input_dict)


def test_get():
    def fun(d, x):
        return d.get("item_1", x) ** 2

    check_grads(fun, argnum=(0, 1))({"item_1": 3.0}, 2.0)
    check_grads(fun, argnum=(0, 1))({"item_2": 4.0}, 2.0)
    check_grads(fun, argnum=(0, 1))({}, 2.0)


def test_make_dict():
    def fun(x):
        return ag_dict([("a", x)], b=x)

    check_grads(fun, modes=["rev"])(1.0)

    def fun(x):
        return ag_dict({"a": x})

    check_grads(fun, modes=["rev"])(1.0)


def test_isinstance():
    def fun(x):
        assert ag_isinstance(x, dict)
        assert ag_isinstance(x, ag_dict)
        return x["x"]

    fun({"x": 1.0})
    grad(fun)({"x": 1.0})

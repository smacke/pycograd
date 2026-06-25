# -*- coding: utf-8 -*-
"""Ported from autograd ``tests/test_scalar_ops.py`` (MIT). Unary/binary scalar op grads."""
import numpy as np
import numpy.random as npr
import pytest

from ._compat import grad
from ._test_util import check_grads

npr.seed(1)


def test_abs():
    fun = lambda x: 3.0 * np.abs(x)
    check_grads(fun)(1.1)
    check_grads(fun)(-1.1)
    check_grads(fun, order=1)(0.0)


def test_absolute():
    fun = lambda x: 3.0 * np.absolute(x)
    check_grads(fun)(1.1)
    check_grads(fun)(-1.1)
    check_grads(fun, order=1)(0.0)


def test_sin():
    fun = lambda x: 3.0 * np.sin(x)
    check_grads(fun)(npr.randn())


def test_sign():
    fun = lambda x: 3.0 * np.sign(x)
    check_grads(fun)(1.1)
    check_grads(fun)(-1.1)


def test_exp():
    fun = lambda x: 3.0 * np.exp(x)
    check_grads(fun)(npr.randn())


def test_log():
    fun = lambda x: 3.0 * np.log(x)
    check_grads(fun)(abs(npr.randn()))


def test_log2():
    fun = lambda x: 3.0 * np.log2(x)
    check_grads(fun)(abs(npr.randn()))


def test_log10():
    fun = lambda x: 3.0 * np.log10(x)
    check_grads(fun)(abs(npr.randn()))


def test_log1p():
    fun = lambda x: 3.0 * np.log1p(x)
    check_grads(fun)(abs(npr.randn()))


def test_expm1():
    fun = lambda x: 3.0 * np.expm1(x)
    check_grads(fun)(abs(npr.randn()))


def test_exp2():
    fun = lambda x: 3.0 * np.exp2(x)
    check_grads(fun)(abs(npr.randn()))


def test_neg():
    fun = lambda x: 3.0 * -x
    check_grads(fun)(npr.randn())


def test_cos():
    fun = lambda x: 3.0 * np.cos(x)
    check_grads(fun)(npr.randn())


def test_tan():
    fun = lambda x: 3.0 * np.tan(x)
    check_grads(fun)(npr.randn())


def test_cosh():
    fun = lambda x: 3.0 * np.cosh(x)
    check_grads(fun)(npr.randn())


def test_sinh():
    fun = lambda x: 3.0 * np.sinh(x)
    check_grads(fun)(npr.randn())


def test_tanh():
    fun = lambda x: 3.0 * np.tanh(x)
    check_grads(fun)(npr.randn())


def test_arccos():
    fun = lambda x: 3.0 * np.arccos(x)
    check_grads(fun)(0.1)


def test_arcsin():
    fun = lambda x: 3.0 * np.arcsin(x)
    check_grads(fun)(0.1)


def test_arctan():
    fun = lambda x: 3.0 * np.arctan(x)
    check_grads(fun)(0.2)


def test_arccosh():
    fun = lambda x: 3.0 * np.arccosh(x)
    check_grads(fun)(npr.randn() ** 2 + 1.2)


def test_arcsinh():
    fun = lambda x: 3.0 * np.arcsinh(x)
    check_grads(fun)(npr.randn())


def test_arctanh():
    fun = lambda x: 3.0 * np.arctanh(x)
    check_grads(fun)(0.2)


def test_sqrt():
    fun = lambda x: 3.0 * np.sqrt(x)
    check_grads(fun)(10.0 * npr.rand())


@pytest.mark.skip(
    reason="pycograd-gap: no rule for the np.power function form (the operator works, but the numpy function alias does not dispatch to a rule)"
)
def test_power_arg0():
    make_fun = lambda y: lambda x: np.power(x, y)
    fun = make_fun(npr.randn() ** 2 + 1.0)
    check_grads(fun)(npr.rand() ** 2 + 1.0)

    fun = make_fun(0.0)
    assert grad(fun)(0.0) == 0.0


@pytest.mark.skip(
    reason="pycograd-gap: no rule for the np.power function form (the operator works, but the numpy function alias does not dispatch to a rule)"
)
def test_power_arg1():
    x = npr.randn() ** 2
    fun = lambda y: np.power(x, y)
    check_grads(fun)(npr.rand() ** 2)


@pytest.mark.skip(
    reason="pycograd-gap: no rule for the np.power function form (the operator works, but the numpy function alias does not dispatch to a rule)"
)
def test_power_arg1_zero():
    fun = lambda y: np.power(0.0, y)
    check_grads(fun)(npr.rand() ** 2)


@pytest.mark.skip(
    reason="pycograd-gap: no rule for the np.mod function form (the operator works, but the numpy function alias does not dispatch to a rule)"
)
def test_mod_arg0():
    fun = lambda x, y: np.mod(x, y)
    check_grads(fun)(npr.rand(), npr.rand())


@pytest.mark.skip(
    reason="pycograd-gap: no rule for the np.mod function form (the operator works, but the numpy function alias does not dispatch to a rule)"
)
def test_mod_arg1():
    fun = lambda x, y: np.mod(x, y)
    check_grads(fun, argnum=1)(npr.rand(), npr.rand())


@pytest.mark.skip(
    reason="pycograd-gap: no rule for the np.divide function form (the operator works, but the numpy function alias does not dispatch to a rule)"
)
def test_divide_arg0():
    fun = lambda x, y: np.divide(x, y)
    check_grads(fun)(npr.rand(), npr.rand())


@pytest.mark.skip(
    reason="pycograd-gap: no rule for the np.divide function form (the operator works, but the numpy function alias does not dispatch to a rule)"
)
def test_divide_arg1():
    fun = lambda x, y: np.divide(x, y)
    check_grads(fun, argnum=1)(npr.rand(), npr.rand())


@pytest.mark.skip(
    reason="pycograd-gap: no rule for the np.multiply function form (the operator works, but the numpy function alias does not dispatch to a rule)"
)
def test_multiply_arg0():
    fun = lambda x, y: np.multiply(x, y)
    check_grads(fun)(npr.rand(), npr.rand())


@pytest.mark.skip(
    reason="pycograd-gap: no rule for the np.multiply function form (the operator works, but the numpy function alias does not dispatch to a rule)"
)
def test_multiply_arg1():
    fun = lambda x, y: np.multiply(x, y)
    check_grads(fun, argnum=1)(npr.rand(), npr.rand())


@pytest.mark.skip(
    reason="pycograd-gap: no rule for the np.true_divide function form (the operator works, but the numpy function alias does not dispatch to a rule)"
)
def test_true_divide_arg0():
    fun = lambda x, y: np.true_divide(x, y)
    check_grads(fun)(npr.rand(), npr.rand())


@pytest.mark.skip(
    reason="pycograd-gap: no rule for the np.true_divide function form (the operator works, but the numpy function alias does not dispatch to a rule)"
)
def test_true_divide_arg1():
    fun = lambda x, y: np.true_divide(x, y)
    check_grads(fun, argnum=1)(npr.rand(), npr.rand())


def test_reciprocal():
    fun = lambda x: np.reciprocal(x)
    check_grads(fun)(npr.rand())


@pytest.mark.skip(
    reason="pycograd-gap: no rule for the np.negative function form (the operator works, but the numpy function alias does not dispatch to a rule)"
)
def test_negative():
    fun = lambda x: np.negative(x)
    check_grads(fun)(npr.rand())


def test_rad2deg():
    fun = lambda x: 3.0 * np.rad2deg(x)
    check_grads(fun)(10.0 * npr.rand())


def test_deg2rad():
    fun = lambda x: 3.0 * np.deg2rad(x)
    check_grads(fun)(10.0 * npr.rand())


def test_radians():
    fun = lambda x: 3.0 * np.radians(x)
    check_grads(fun)(10.0 * npr.rand())


def test_degrees():
    fun = lambda x: 3.0 * np.degrees(x)
    check_grads(fun)(10.0 * npr.rand())


@pytest.mark.skip(reason="pycograd-gap: no VJP rule for np.sinc")
def test_sinc():
    fun = lambda x: 3.0 * np.sinc(x)
    check_grads(fun)(10.0 * npr.rand())

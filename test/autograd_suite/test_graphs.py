# -*- coding: utf-8 -*-
"""Ported from autograd ``tests/test_graphs.py`` (MIT). Computation-graph shapes: fanout,
constants, identity, mutating outgrads, singleton outputs.

The higher-order cases (``check_grads`` of a ``grad`` function, third derivatives,
Hessian-vector products built by nesting ``grad``) and the complex-number cases are skipped:
the ported ``check_grads`` is first-order, and pycograd's reverse pass detaches, so an
autograd-style reverse-over-reverse numerical check is not supported (pycograd's native
higher-order AD is covered in ``test/test_highorder.py``)."""
import warnings

import numpy as np
import numpy.random as npr
import pytest

from ._compat import grad
from ._test_util import check_grads

npr.seed(1)


def test_grad_fanout():
    fun = lambda x: np.sin(np.sin(x) + np.sin(x))
    check_grads(fun)(npr.randn())


def test_grad_const():
    fun = lambda x: 1.0
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        df = grad(fun)
        assert np.allclose(df(2.0), 0.0)


def test_grad_identity():
    fun = lambda x: x
    df = grad(fun)
    assert np.allclose(df(2.0), 1.0)


def test_hess_vector_prod_first_order():
    npr.seed(1)
    randv = npr.randn(10)

    def fun(x):
        return np.sin(np.dot(x, randv))

    A = npr.randn(10)
    check_grads(fun)(A)


def test_mutating_outgrad():
    def fun(a):
        b = a + 1.0
        c = b + 1.5
        d = a + b
        e = d + c
        return np.sum(e)

    A = npr.randn(5)
    check_grads(fun)(A)


def test_mutating_outgrad_from_indexing():
    def fun(a):
        b = a + 1.0
        c = b[0] + 1.5
        d = a + b
        e = d + c
        return np.sum(e)

    A = npr.randn(5)
    check_grads(fun)(A)


def test_singleton_array_output():
    fun = lambda x: np.sum(np.sin(x), keepdims=True)
    check_grads(fun)(npr.randn(3, 3))


def test_singleton_array_output_axis0():
    fun = lambda x: np.sum(np.sin(x), axis=0, keepdims=False)
    check_grads(fun)(npr.randn(3, 1))


def test_singleton_array_output_axis1():
    fun = lambda x: np.sum(np.sin(x), axis=1, keepdims=False)
    check_grads(fun)(npr.randn(1, 3))


def test_singleton_array_output_axis0_keepdims():
    fun = lambda x: np.sum(np.sin(x), axis=0, keepdims=True)
    check_grads(fun)(npr.randn(3, 1))


def test_singleton_array_output_axis1_keepdims():
    fun = lambda x: np.sum(np.sin(x), axis=1, keepdims=True)
    check_grads(fun)(npr.randn(1, 3))


@pytest.mark.skip(
    reason="pycograd-gap: in-place item assignment (A[1] = b) is not supported, so this "
    "does not raise the autograd TypeError"
)
def test_assignment_raises_error():
    pass


@pytest.mark.skip(
    reason="pycograd higher-order via the first-order check_grads: see module docstring "
    "(covered natively in test/test_highorder.py)"
)
def test_enclosing_scope_ref():
    pass


@pytest.mark.skip(reason="pycograd higher-order: see module docstring")
def test_enclosing_scope_ref_2():
    pass


@pytest.mark.skip(reason="pycograd higher-order: check_grads of a grad function")
def test_third_derivative():
    pass


@pytest.mark.skip(reason="pycograd higher-order: check_grads of a grad function")
def test_third_derivative_other_args():
    pass


@pytest.mark.skip(reason="pycograd higher-order: check_grads of a grad function")
def test_third_derivative_other_args2():
    pass


@pytest.mark.skip(reason="pycograd-gap: complex numbers")
def test_complex_mutating_outgrad_from_indexing():
    pass


@pytest.mark.skip(reason="pycograd-gap: complex numbers")
def test_complex_separate_real_and_imaginary():
    pass

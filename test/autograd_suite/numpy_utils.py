# -*- coding: utf-8 -*-
"""Ported from autograd ``tests/numpy_utils.py`` (MIT). Systematic shape sweeps for the
op-coverage tests. The complex branches are dropped (``test_complex`` defaults to False);
pycograd is real-only."""
import numpy.random as npr

from ._test_util import combo_check


def stat_check(fun, test_complex=False, **kwargs):
    x = 3.5
    A = npr.randn()
    B = npr.randn(3)
    C = npr.randn(2, 3)
    D = npr.randn(1, 3)
    check = combo_check(fun, (0,), **kwargs)
    check([x, A])
    check([B, C, D], axis=[None, 0], keepdims=[True, False])
    check([C, D], axis=[None, 0, 1], keepdims=[True, False])


def unary_ufunc_check(fun, lims=[-2, 2], test_complex=False, **kwargs):
    scalar = transform(lims, 0.4)
    vector = transform(lims, npr.rand(2))
    mat = transform(lims, npr.rand(3, 2))
    mat2 = transform(lims, npr.rand(1, 2))
    check = combo_check(fun, (0,), **kwargs)
    check([scalar, vector, mat, mat2])


def binary_ufunc_check(
    fun, lims_A=[-2, 2], lims_B=[-2, 2], test_complex=False, **kwargs
):
    T_A = lambda x: transform(lims_A, x)
    T_B = lambda x: transform(lims_B, x)
    scalar = 0.6
    vector = npr.rand(2)
    mat = npr.rand(3, 2)
    mat2 = npr.rand(1, 2)
    check = combo_check(fun, (0, 1), **kwargs)
    check(
        [T_A(scalar), T_A(vector), T_A(mat), T_A(mat2)],
        [T_B(scalar), T_B(vector), T_B(mat), T_B(mat2)],
    )


def binary_ufunc_check_no_same_args(
    fun, lims_A=[-2, 2], lims_B=[-2, 2], test_complex=False, **kwargs
):
    T_A = lambda x: transform(lims_A, x)
    T_B = lambda x: transform(lims_B, x)
    scalar1, scalar2 = 0.6, 0.7
    vector1, vector2 = npr.rand(2), npr.rand(2)
    mat11, mat12 = npr.rand(3, 2), npr.rand(3, 2)
    mat21, mat22 = npr.rand(1, 2), npr.rand(1, 2)
    check = combo_check(fun, (0, 1), **kwargs)
    check(
        [T_A(scalar1), T_A(vector1), T_A(mat11), T_A(mat21)],
        [T_B(scalar2), T_B(vector2), T_B(mat12), T_B(mat22)],
    )


def transform(lims, x):
    return x * (lims[1] - lims[0]) + lims[0]

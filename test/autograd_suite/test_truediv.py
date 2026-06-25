# -*- coding: utf-8 -*-
"""Ported from autograd ``tests/test_truediv.py`` (MIT)."""
import itertools as it

import numpy as np
import numpy.random as npr

from ._test_util import check_grads

rs = npr.RandomState(0)


def test_div():
    fun = lambda x, y: x / y
    make_gap_from_zero = lambda x: np.sqrt(x**2 + 0.5)
    scalar = 2.0
    vector = rs.randn(4)
    mat = rs.randn(3, 4)
    mat2 = rs.randn(1, 4)
    allargs = [scalar, vector, mat, mat2]
    for arg1, arg2 in it.product(allargs, allargs):
        arg1 = make_gap_from_zero(arg1)
        arg2 = make_gap_from_zero(arg2)
        check_grads(fun)(arg1, arg2)

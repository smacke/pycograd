# -*- coding: utf-8 -*-
"""Reductions over a python list/tuple of tape values (``np.mean([a, b])``), ``len`` on a box,
and python builtins (``sum``) over boxes -- the operand has no array conversion, so ``_lift``
stacks a list-of-boxes onto a new axis, and the tracers carry the ``len``/arithmetic dunders the
builtins reach in C (bypassing instrumentation).

Module-level functions so pyccolo can re-instrument them from source."""
import numpy as np
import pytest

from pycograd import grad, jvp, vmap


def f_mean_list(x):
    return np.mean([x, x + 2])


def f_mean_tuple(x):
    return np.mean((x, x + 2))


def f_std_list(x):
    return np.std([x, x + 2])


def f_var_list(x):
    return np.var([x, x + 2])


def f_sum_axis_list(x):
    return np.sum(np.sum([x, x * 2.0], axis=0) ** 2)


def test_list_reduction_grads():
    assert np.isclose(float(np.asarray(grad(f_mean_list)(0.0)[0])), 1.0)
    assert np.isclose(float(np.asarray(grad(f_mean_tuple)(0.0)[0])), 1.0)
    assert np.isclose(float(np.asarray(grad(f_std_list)(0.0)[0])), 0.0)
    assert np.isclose(float(np.asarray(grad(f_var_list)(0.0)[0])), 0.0)
    A = np.random.default_rng(0).standard_normal((3, 2))
    # mean([a, a*2]) stacks then reduces; grad flows to the stacked leaves
    g = np.asarray(grad(f_sum_axis_list)(A)[0])
    assert g.shape == (3, 2)


def f_py_sum(x, y):
    return np.sum(sum([x, y]))  # python builtin sum -> 0 + x + y at C level


def f_len(x):
    return np.sum(x) * len(x)  # len bypasses instrumentation -> tracer.__len__


def test_builtins_over_boxes():
    A = np.ones((3, 2))
    B = np.ones((3, 2))
    ga, gb = grad(f_py_sum, [0, 1])(A, B)
    assert np.allclose(np.asarray(ga), np.ones((3, 2)))
    assert np.allclose(np.asarray(gb), np.ones((3, 2)))
    # len(x) == 3 used as a factor
    assert np.allclose(np.asarray(grad(f_len)(A)[0]), 3 * np.ones((3, 2)))


def test_builtins_under_transforms():
    # sum/len bypass instrumentation, so the jvp/vmap tracers must carry the dunders too.
    A = np.ones((3, 2))
    _, t = jvp(lambda a: np.sum(sum([a, a])), (A,), (A,))
    assert np.isfinite(float(np.asarray(t)))
    out = np.asarray(vmap(lambda a: np.sum(sum([a, a])))(np.stack([A, A])))
    assert out.shape == (2,)

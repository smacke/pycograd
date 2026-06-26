# -*- coding: utf-8 -*-
"""np.array over differentiable leaves -- ``np.array([v0, v1, ...])`` where the (possibly
nested) list holds Var/Tracer boxes. numpy can't build such an array, so each nesting level
becomes a stack along a fresh leading axis. A list with no boxes passes straight through, so
intercepting the pervasive np.array stays transparent. Plus Var.dtype.

Module-level functions so pyccolo can re-instrument them from source."""
import numpy as np
import pytest

from pycograd import eval_shape, grad, jvp, vmap


def f_fanout_1d(x):
    A = np.array([x, x * 1.0, x + 2.5])
    return np.sum((A + A) ** 2)


def f_fanout_2d(x):
    A = np.array([[x, x * 1.0, x + 2.5], [x**2, x, x / 2.0]])
    return np.sum((A + A) ** 2)


def f_scalar(x):
    return np.array(x) ** 2


def f_from_arrays(x):  # x is an array; stack two copies along a new leading axis
    return np.sum(np.array([x, x]) ** 2)


def f_max_tie(x):  # gradient splits equally across the two equal maxima
    return np.max(np.array([x, x]))


@pytest.mark.parametrize(
    "fn, x", [(f_fanout_1d, 3.0), (f_fanout_2d, 3.0), (f_scalar, 3.0)]
)
def test_scalar_fanout_grad(fn, x):
    g = float(np.asarray(grad(fn)(x)[0]))
    fd = (fn(x + 1e-6) - fn(x - 1e-6)) / 2e-6
    assert np.isclose(g, float(fd), atol=1e-4)


def test_array_from_arrays_and_shapes():
    X = np.random.default_rng(0).standard_normal((3, 2))
    # sum([x,x]**2) = 2*sum(x**2), so d/dx = 4x at each entry (both copies contribute)
    assert np.allclose(np.asarray(grad(f_from_arrays)(X)[0]), 4 * X)
    assert eval_shape(lambda z: np.array([z, z]), X).shape == (2, 3, 2)
    v = np.asarray(vmap(lambda z: np.array([z, z * 2.0]))(np.arange(4.0)))
    assert v.shape == (4, 2)


def test_max_tie_splits_gradient():
    # both copies are the max, so each gets half the unit cotangent -> total 1.0
    assert np.isclose(float(np.asarray(grad(f_max_tie)(2.0)[0])), 1.0)
    _, t = jvp(f_fanout_1d, (3.0,), (1.0,))
    assert np.isfinite(float(np.asarray(t)))


def test_plain_np_array_passes_through():
    # np.array of a plain list (no boxes) stays an ordinary constant array inside a trace.
    def fn(x):
        idx = np.array([0, 2])  # plain -> must not become a tape Var
        return np.sum(x[idx] ** 2)

    A = np.arange(5.0)  # sum(A[[0,2]]**2); grad is 2*A only at 0 and 2
    assert np.allclose(np.asarray(grad(fn)(A)[0]), np.array([0.0, 0, 4, 0, 0]))


def test_var_dtype_attribute():
    seen = {}

    def fn(x):
        seen["dtype"] = x.dtype
        seen["shape"] = x.shape
        return x**2

    grad(fn)(3.0)
    assert seen["dtype"] == np.float64 and seen["shape"] == ()

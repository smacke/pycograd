# -*- coding: utf-8 -*-
"""Gradients of the newly added unary numpy ufuncs (tan, inverse-trig, exp2/log2/log10,
deg/rad conversions, and the zero-gradient step ufuncs sign/ceil/floor + fabs->abs), checked
against central finite differences, plus their forward-mode (jvp) and batched (vmap) rules.

Module-level functions so pyccolo can re-instrument them from source."""
import numpy as np
import pytest

from pycograd import grad, jvp, vmap


def _fd(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def f_tan(x):
    return np.tan(x)


def f_arcsin(x):
    return np.arcsin(x)


def f_arccos(x):
    return np.arccos(x)


def f_arctanh(x):
    return np.arctanh(x)


def f_arcsinh(x):
    return np.arcsinh(x)


def f_arccosh(x):
    return np.arccosh(x)


def f_exp2(x):
    return np.exp2(x)


def f_log2(x):
    return np.log2(x)


def f_log10(x):
    return np.log10(x)


def f_deg2rad(x):
    return np.deg2rad(x)


def f_radians(x):
    return np.radians(x)


def f_rad2deg(x):
    return np.rad2deg(x)


def f_fabs(x):
    return np.fabs(x)


@pytest.mark.parametrize(
    "fn, x",
    [
        (f_tan, 0.3),
        (f_arcsin, 0.3),
        (f_arccos, 0.3),
        (f_arctanh, 0.3),
        (f_arcsinh, 0.7),
        (f_arccosh, 1.7),
        (f_exp2, 0.4),
        (f_log2, 1.3),
        (f_log10, 1.3),
        (f_deg2rad, 30.0),
        (f_radians, 30.0),
        (f_rad2deg, 0.5),
        (f_fabs, -0.7),
    ],
)
def test_grad_vs_finite_difference(fn, x):
    g = float(np.asarray(grad(fn)(x)[0]))
    assert np.isclose(g, _fd(fn, x), atol=1e-5), (fn.__name__, g, _fd(fn, x))


def f_sign(x):
    return np.sign(x)


def f_ceil(x):
    return np.ceil(x)


def f_floor(x):
    return np.floor(x)


@pytest.mark.parametrize("fn", [f_sign, f_ceil, f_floor])
def test_step_ufuncs_have_zero_gradient(fn):
    assert float(np.asarray(grad(fn)(0.4)[0])) == 0.0


def test_jvp_tan_matches_finite_difference():
    # exercises the forward-mode rule auto-derived from _UNARY_DERIV
    x, v = 0.3, 1.0
    _, t = jvp(f_tan, (x,), (v,))
    assert np.isclose(float(np.asarray(t)), _fd(f_tan, x) * v, atol=1e-5)


def test_vmap_arcsin_is_elementwise():
    # exercises the batching rule registered for the new unary prims
    xs = np.array([0.1, 0.2, 0.3])
    out = np.asarray(vmap(f_arcsin)(xs))
    assert np.allclose(out, np.arcsin(xs))

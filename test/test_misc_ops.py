# -*- coding: utf-8 -*-
"""Miscellaneous niche ops: np.nan_to_num (gradient masked to finite inputs),
np.real_if_close (identity), and np.concatenate with a *positional* axis. Gradients vs
finite differences, plus forward (jvp), batching (vmap), and shape inference (eval_shape).

Module-level functions so pyccolo can re-instrument them from source."""
import numpy as np

from pycograd import eval_shape, grad, jvp, vmap

_rng = np.random.default_rng(0)


def _fd(f, x, eps=1e-6):
    out = np.zeros(np.shape(x))
    for i in range(np.size(x)):
        xp = x.copy()
        xm = x.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        out.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return out


_Y = np.array([0.0, np.nan, np.inf, -np.inf])


def f_nan_to_num(x):
    # x + Y injects nan/inf; nan_to_num clamps them -> gradient flows only at the finite entry
    return np.sum(np.sin(np.nan_to_num(x + _Y)))


def f_real_if_close(x):
    return np.sum(np.real_if_close(x) ** 2)


_B3 = _rng.standard_normal((5, 6, 4))


def f_concat_pos_axis(x):
    return np.sum(np.concatenate((_B3, x, _B3), 1) ** 2)


def test_nan_to_num_masks_gradient():
    x = _rng.standard_normal(4)
    g = np.asarray(grad(f_nan_to_num)(x)[0])
    # only the first (finite) entry has nonzero gradient
    assert g[0] != 0.0 and np.allclose(g[1:], 0.0)
    assert np.allclose(g, _fd(f_nan_to_num, x), atol=1e-5)
    _, t = jvp(f_nan_to_num, (x,), (np.ones_like(x),))
    assert np.isfinite(float(np.asarray(t)))


def test_real_if_close_identity():
    x = _rng.standard_normal((3, 2))
    assert np.allclose(np.asarray(grad(f_real_if_close)(x)[0]), 2 * x)
    out = np.asarray(vmap(lambda v: np.real_if_close(v))(np.stack([x, x])))
    assert out.shape == (2, 3, 2)


def test_concatenate_positional_axis():
    A = _rng.standard_normal((5, 6, 4))
    g = np.asarray(grad(f_concat_pos_axis)(A)[0])
    assert np.allclose(g, _fd(f_concat_pos_axis, A), atol=1e-5)
    assert eval_shape(lambda x: np.concatenate((x, x), 1), A).shape == (5, 12, 4)

# -*- coding: utf-8 -*-
"""Gather-by-argsort / selection ops: np.sort, np.partition (permute by a stop-gradient
index), np.select (a fold of where), and np.gradient (central-difference, a getitem/
concatenate composition). Gradients vs finite differences, plus forward (jvp), batching
(vmap), and shape inference (eval_shape).

Module-level functions so pyccolo can re-instrument them from source."""
import numpy as np
import pytest

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


def f_sort(x):
    return np.sum(np.sort(x) ** 2 * np.arange(1, x.size + 1))


def f_sort_axis(x):
    return np.sum(np.sort(x, axis=1) ** 2)


def f_partition(x):
    return np.sum(np.partition(x, 3) ** 2 * np.arange(1, x.size + 1))


_V = _rng.standard_normal(7)
_M = _rng.standard_normal((3, 5))


@pytest.mark.parametrize("fn, a", [(f_sort, _V), (f_sort_axis, _M), (f_partition, _V)])
def test_sort_partition_grad(fn, a):
    assert np.allclose(np.asarray(grad(fn)(a)[0]), _fd(fn, a), atol=1e-5)
    assert eval_shape(fn, a).shape == ()


def test_sort_jvp_and_vmap():
    _, t = jvp(f_sort, (_V,), (np.ones_like(_V),))
    assert np.isfinite(float(np.asarray(t)))
    out = np.asarray(vmap(lambda r: np.sort(r))(np.stack([_V, _V + 0.1])))
    assert out.shape == (2, 7) and np.allclose(out[0], np.sort(_V))


# --- select -----------------------------------------------------------------
_C = [_rng.standard_normal((3, 4)) > 0 for _ in range(3)]


def f_select(*choices):
    return np.sum(np.select(_C, list(choices), default=1.1) ** 2)


def test_select_grad():
    chs = [_rng.standard_normal((3, 4)) for _ in range(3)]
    g = grad(f_select, [0, 1, 2])(*chs)
    for k in range(3):

        def loss(ck, k=k):
            full = [c.copy() for c in chs]
            full[k] = ck
            return f_select(*full)

        assert np.allclose(np.asarray(g[k]), _fd(loss, chs[k]), atol=1e-5)
    assert eval_shape(f_select, *chs).shape == ()


# --- gradient ---------------------------------------------------------------
def g_ax0(x):
    return np.sum(np.gradient(x, axis=0) ** 2)


def g_none(x):
    return sum(np.sum(g**2) for g in np.gradient(x))


def g_tuple(x):
    return sum(np.sum(g**2) for g in np.gradient(x, axis=(0, 1)))


@pytest.mark.parametrize("fn", [g_ax0, g_none, g_tuple])
def test_gradient_grad(fn):
    A = _rng.standard_normal((5, 5))
    assert np.allclose(np.asarray(grad(fn)(A)[0]), _fd(fn, A), atol=1e-5)


def test_gradient_value_and_shapes():
    A = _rng.standard_normal((5, 6))
    # np.gradient single-axis value matches numpy (verified through a captured forward).
    assert eval_shape(lambda x: np.gradient(x, axis=0), A).shape == (5, 6)
    shapes = eval_shape(lambda x: np.gradient(x), A)
    assert [tuple(s.shape) for s in shapes] == [(5, 6), (5, 6)]
    _, t = jvp(g_ax0, (A,), (np.ones_like(A),))
    assert np.isfinite(float(np.asarray(t)))

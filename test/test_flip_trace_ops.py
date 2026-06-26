# -*- coding: utf-8 -*-
"""Axis-reversal and diagonal-sum ops: np.flipud / np.fliplr / np.rot90 (a ``::-1`` slice
plus, for rot90, a transpose) and np.trace (gather the diagonal indices, then sum). All are
getitem/transpose compositions. Gradients vs finite differences, plus forward (jvp), batching
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


def f_flipud(x):
    return np.sum(np.flipud(x) ** 2 * np.arange(x.size).reshape(x.shape))


def f_fliplr(x):
    return np.sum(np.fliplr(x) * 3.0)


def f_rot90_k1(x):
    return np.sum(np.rot90(x) ** 2)


def f_rot90_k3(x):
    return np.sum(np.rot90(x, 3) * 2.0)


_M = _rng.standard_normal((4, 5))


@pytest.mark.parametrize("fn", [f_flipud, f_fliplr, f_rot90_k1, f_rot90_k3])
def test_flip_grad_vs_fd(fn):
    assert np.allclose(np.asarray(grad(fn)(_M)[0]), _fd(fn, _M), atol=1e-5)
    assert eval_shape(fn, _M).shape == ()


def test_flip_jvp_and_vmap():
    _, t = jvp(f_flipud, (_M,), (np.ones_like(_M),))
    assert np.isfinite(float(np.asarray(t)))
    v = np.asarray(vmap(lambda m: np.fliplr(m))(np.stack([_M, _M + 0.1])))
    assert v.shape == (2, 4, 5) and np.allclose(v[0], np.fliplr(_M))
    # rot90 of a non-square matrix swaps the two axes
    assert eval_shape(lambda x: np.rot90(x), _M).shape == (5, 4)


# --- trace ------------------------------------------------------------------
def f_trace(x):
    return np.trace(x) ** 2


def f_trace_offset(x):
    return np.sum(np.trace(x, offset=1))


def f_trace_3d(x):
    return np.sum(np.trace(x) ** 2)


def test_trace_grad_vs_fd():
    Sq = _rng.standard_normal((5, 5))
    T = _rng.standard_normal((4, 4, 3))
    assert np.allclose(np.asarray(grad(f_trace)(Sq)[0]), _fd(f_trace, Sq), atol=1e-5)
    assert np.allclose(
        np.asarray(grad(f_trace_offset)(_M)[0]), _fd(f_trace_offset, _M), atol=1e-5
    )
    assert np.allclose(
        np.asarray(grad(f_trace_3d)(T)[0]), _fd(f_trace_3d, T), atol=1e-5
    )
    # trace over the leading two axes of a 3-D array leaves the trailing axis
    assert eval_shape(lambda x: np.trace(x), T).shape == (3,)
    tv = np.asarray(vmap(lambda m: np.trace(m))(np.stack([Sq, Sq])))
    assert tv.shape == (2,) and np.allclose(tv[0], np.trace(Sq))

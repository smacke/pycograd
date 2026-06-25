# -*- coding: utf-8 -*-
"""Array-manipulation ops that lower to existing primitives: axis reordering
(np.moveaxis / np.swapaxes / np.rollaxis -> transpose) and triangular masking
(np.tril / np.triu -> multiply by a constant mask). Gradients vs finite differences, plus
forward-mode (jvp), batching (vmap), and shape inference (eval_shape).

Module-level functions so pyccolo can re-instrument them from source."""
import numpy as np
import pytest

from pycograd import eval_shape, grad, jvp, vmap

_rng = np.random.default_rng(0)


def _fd(f, x, eps=1e-6):
    out = np.zeros(x.shape)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        out.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return out


def f_moveaxis(x):
    return np.sum(np.moveaxis(x, 0, 2) * 2.0)


def f_swapaxes(x):
    return np.sum(np.swapaxes(x, 0, 1) ** 2)


def f_rollaxis(x):
    return np.sum(np.rollaxis(x, 2, 0) * 3.0)


def f_tril(x):
    return np.sum(np.tril(x, -1) ** 2)


def f_triu(x):
    return np.sum(np.triu(x))


_X = _rng.standard_normal((2, 3, 4))
_M = _rng.standard_normal((5, 5))


@pytest.mark.parametrize(
    "fn, a",
    [
        (f_moveaxis, _X),
        (f_swapaxes, _X),
        (f_rollaxis, _X),
        (f_tril, _M),
        (f_triu, _M),
    ],
)
def test_grad_vs_fd(fn, a):
    g = np.asarray(grad(fn)(a)[0])
    assert np.allclose(g, _fd(fn, a), atol=1e-5)
    assert eval_shape(fn, a).shape == ()


def test_tril_jvp_matches_fd():
    v = _rng.standard_normal((5, 5))
    _, t = jvp(f_tril, (_M,), (v,))
    fd = (f_tril(_M + 1e-6 * v) - f_tril(_M - 1e-6 * v)) / 2e-6
    assert np.isclose(float(np.asarray(t)), fd, atol=1e-4)


def test_vmap_is_per_example():
    out = np.asarray(vmap(lambda m: np.tril(m, -1))(np.stack([_M, _M + 0.1])))
    assert out.shape == (2, 5, 5)
    assert np.allclose(out[0], np.tril(_M, -1))
    sw = np.asarray(vmap(lambda x: np.swapaxes(x, 0, 1))(np.stack([_X, _X])))
    assert sw.shape == (2, 3, 2, 4)
    assert np.allclose(sw[0], np.swapaxes(_X, 0, 1))

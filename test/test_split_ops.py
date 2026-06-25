# -*- coding: utf-8 -*-
"""The split family (np.split / array_split / vsplit / hsplit / dsplit) -- the inverse of
concatenate, lowered to ``d_getitem`` slices (so the op returns a *list* of pieces).
Gradients vs finite differences, plus forward (jvp), batching (vmap), and shape inference
(eval_shape, incl. the list output).

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


def f_split(x):
    a, b, c, d = np.split(x, 4, axis=0)
    return np.sum(a**2) + np.sum(b * 2) + np.sum(np.sin(c)) + np.sum(d)


def f_split_indices(x):
    a, b, c = np.split(x, [2, 5], axis=0)
    return np.sum(a**2) + np.sum(b) + np.sum(c**3)


def f_array_split(x):
    return sum(np.sum(p**2) for p in np.array_split(x, 3, axis=0))


def f_vsplit(x):
    a, b = np.vsplit(x, 2)
    return np.sum(a**2) + np.sum(b * 3)


def f_hsplit(x):
    a, b = np.hsplit(x, 2)
    return np.sum(a**2) + np.sum(b * 3)


_V = _rng.standard_normal(8)
_W = _rng.standard_normal(7)
_M = _rng.standard_normal((4, 6))


@pytest.mark.parametrize(
    "fn, a",
    [
        (f_split, _V),
        (f_split_indices, _V),
        (f_array_split, _W),
        (f_vsplit, _M),
        (f_hsplit, _M),
    ],
)
def test_split_grad_vs_fd(fn, a):
    assert np.allclose(np.asarray(grad(fn)(a)[0]), _fd(fn, a), atol=1e-5)
    assert eval_shape(fn, a).shape == ()


def test_split_jvp_and_list_eval_shape():
    _, t = jvp(f_split, (_V,), (np.ones_like(_V),))
    assert np.isfinite(float(np.asarray(t)))
    shapes = eval_shape(lambda x: np.split(x, 4, axis=0), _V)
    assert [tuple(s.shape) for s in shapes] == [(2,)] * 4


def test_split_vmap():
    def first_piece(m):
        return np.split(m, 2, axis=0)[0]

    out = np.asarray(vmap(first_piece)(np.stack([_V, _V + 0.1])))
    assert out.shape == (2, 4)
    assert np.allclose(out[0], np.split(_V, 2, axis=0)[0])

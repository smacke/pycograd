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


# --- roll + reshape-lowered ops --------------------------------------------
def f_roll(x):
    return np.sum(np.roll(x, 2, axis=1) ** 2)


def f_roll_flat(x):
    return np.sum(np.roll(x, 1) * 3.0)


def f_ravel(x):
    return np.sum(np.ravel(x) * np.arange(x.size))


def f_squeeze(x):
    return np.sum(np.squeeze(x) ** 2)


def f_atleast_2d(x):
    return np.sum(np.atleast_2d(x) * 2.0)


def f_atleast_3d(x):
    return np.sum(np.atleast_3d(x) * 2.0)


_A = _rng.standard_normal((3, 4))
_S = _rng.standard_normal((1, 3, 1))
_V = _rng.standard_normal(4)


@pytest.mark.parametrize(
    "fn, a",
    [
        (f_roll, _A),
        (f_roll_flat, _A),
        (f_ravel, _A),
        (f_squeeze, _S),
        (f_atleast_2d, _V),
        (f_atleast_3d, _V),
    ],
)
def test_roll_reshape_grad_vs_fd(fn, a):
    assert np.allclose(np.asarray(grad(fn)(a)[0]), _fd(fn, a), atol=1e-5)


def test_roll_jvp_and_vmap():
    _, t = jvp(f_roll, (_A,), (np.ones_like(_A),))
    assert np.isfinite(float(np.asarray(t)))
    v = np.asarray(vmap(lambda m: np.roll(m, 1, axis=1))(np.stack([_A, _A + 0.1])))
    assert v.shape == (2, 3, 4)
    assert np.allclose(v[0], np.roll(_A, 1, axis=1))


# --- ndarray methods (.flatten/.ravel/.squeeze) + np.append -----------------
def f_flatten_method(x):
    return np.sum(x.flatten() * np.arange(x.size))


def f_ravel_method(x):
    return np.sum(x.ravel() ** 2)


def f_squeeze_method(x):
    return np.sum(x.squeeze() ** 2)


def f_append(a, b):
    return np.sum(np.append(a, b) ** 2)


def f_append_axis(a, b):
    return np.sum(np.append(a, b, axis=0) ** 2)


def test_method_grads():
    A = _rng.standard_normal((3, 4))
    S = _rng.standard_normal((3, 1, 4))
    assert np.allclose(
        np.asarray(grad(f_flatten_method)(A)[0]), _fd(f_flatten_method, A), atol=1e-5
    )
    assert np.allclose(
        np.asarray(grad(f_ravel_method)(A)[0]), _fd(f_ravel_method, A), atol=1e-5
    )
    assert np.allclose(
        np.asarray(grad(f_squeeze_method)(S)[0]), _fd(f_squeeze_method, S), atol=1e-5
    )
    # flatten under vmap routes through ravel
    v = np.asarray(vmap(lambda m: m.flatten())(np.stack([A, A])))
    assert v.shape == (2, 12) and np.allclose(v[0], A.flatten())


def test_append_grad_and_shape():
    a = _rng.standard_normal(4)
    b = np.array(3.0)
    ga, gb = grad(f_append, [0, 1])(a, b)
    assert np.allclose(np.asarray(ga), 2 * a) and np.isclose(
        float(np.asarray(gb)), 2 * b
    )
    A2 = _rng.standard_normal((2, 3))
    B2 = _rng.standard_normal((1, 3))
    assert np.allclose(
        np.asarray(grad(f_append_axis)(A2, B2)[0]),
        _fd(lambda x: f_append_axis(x, B2), A2),
        atol=1e-5,
    )
    assert eval_shape(lambda x, y: np.append(x, y), a, b).shape == (5,)


# --- pad / repeat / tile (segment/scatter adjoints) ------------------------
def f_pad(x):
    return np.sum(np.pad(x, ((1, 2), (3, 4))) ** 2)


def f_repeat0(x):
    return np.sum(np.repeat(x, 2, axis=0) ** 2)


def f_repeat_none(x):
    return np.sum(np.repeat(x, 3) ** 2)


def f_tile(x):
    return np.sum(np.tile(x, (2, 3)) ** 2)


@pytest.mark.parametrize("fn", [f_pad, f_repeat0, f_repeat_none, f_tile])
def test_segment_grad_vs_fd(fn):
    assert np.allclose(np.asarray(grad(fn)(_A)[0]), _fd(fn, _A), atol=1e-5)
    assert eval_shape(fn, _A).shape == ()


def test_pad_eval_shape_and_vmap():
    assert eval_shape(lambda x: np.pad(x, 2), _A).shape == (7, 8)
    v = np.asarray(vmap(lambda m: np.pad(m, ((1, 1), (2, 2))))(np.stack([_A, _A])))
    assert v.shape == (2, 5, 8)
    assert np.allclose(v[0], np.pad(_A, ((1, 1), (2, 2))))


def f_uses_index_repeat(x):
    # np.repeat/np.tile on a *plain* index array must stay a plain array (not a tape Var),
    # so fancy indexing still works -- regression for the conv im2col index path.
    idx = np.repeat(np.arange(x.shape[0]), 1)
    return np.sum(x[idx] ** 2)


def test_structural_ops_pass_through_plain_indices():
    g = np.asarray(grad(f_uses_index_repeat)(_A)[0])
    assert np.allclose(g, _fd(f_uses_index_repeat, _A), atol=1e-5)


# --- stack family 1-D edges + dtype kwarg ----------------------------------
def f_hstack_single(x):
    # a single 1-D array passed as the sequence: its scalars promote via atleast_1d
    return np.sum(np.hstack(x) ** 2)


def f_column_stack(a, b):
    return np.sum(np.column_stack((a, b)) ** 2)


def f_row_stack(a, b):
    return np.sum(np.row_stack((a, b)) ** 2)


def test_stack_1d_edges():
    v = _rng.standard_normal(4)
    assert np.allclose(
        np.asarray(grad(f_hstack_single)(v)[0]), _fd(f_hstack_single, v), atol=1e-5
    )
    a1, b1 = _rng.standard_normal(3), _rng.standard_normal(3)
    ga, gb = grad(f_column_stack, [0, 1])(a1, b1)
    assert np.allclose(np.asarray(ga), 2 * a1) and np.allclose(np.asarray(gb), 2 * b1)
    # row_stack forwards a dtype kwarg to vstack; it must be swallowed
    A, B = _rng.standard_normal((2, 3)), _rng.standard_normal((1, 3))
    assert eval_shape(lambda x, y: np.row_stack((x, y)), A, B).shape == (3, 3)
    assert np.allclose(
        np.asarray(grad(f_row_stack)(A, B)[0]),
        _fd(lambda x: f_row_stack(x, B), A),
        atol=1e-5,
    )

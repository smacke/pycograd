# -*- coding: utf-8 -*-
"""logaddexp / logaddexp2 / fmax / fmin (binary), np.diff (getitem/sub composition), and
np.diag / np.diagonal (getitem-extract / scatter-construct). Gradients vs finite differences,
plus forward (jvp), batching (vmap), and shape inference (eval_shape).

Module-level functions so pyccolo can re-instrument them from source."""
import numpy as np
import pytest

from pycograd import eval_shape, grad, jvp, vmap

_rng = np.random.default_rng(0)


def _fd1(f, x, eps=1e-6):
    out = np.zeros(np.shape(x))
    for i in range(np.size(x)):
        xp = x.copy()
        xm = x.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        out.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return out


def _fd_arg(f, args, k, eps=1e-6):
    out = np.zeros(np.shape(args[k]))
    for i in range(np.size(args[k])):
        ap = [a.copy() for a in args]
        am = [a.copy() for a in args]
        ap[k].flat[i] += eps
        am[k].flat[i] -= eps
        out.flat[i] = (f(*ap) - f(*am)) / (2 * eps)
    return out


_A = _rng.standard_normal((3, 4))
_B = _rng.standard_normal((3, 4))


def b_logaddexp(a, b):
    return np.sum(np.logaddexp(a, b))


def b_logaddexp2(a, b):
    return np.sum(np.logaddexp2(a, b))


def b_fmax(a, b):
    return np.sum(np.fmax(a, b) ** 2)


def b_fmin(a, b):
    return np.sum(np.fmin(a, b) * 2.0)


@pytest.mark.parametrize("fn", [b_logaddexp, b_logaddexp2, b_fmax, b_fmin])
def test_binary_grad_vs_fd(fn):
    ga, gb = grad(fn, [0, 1])(_A, _B)
    assert np.allclose(np.asarray(ga), _fd_arg(fn, [_A, _B], 0), atol=1e-5)
    assert np.allclose(np.asarray(gb), _fd_arg(fn, [_A, _B], 1), atol=1e-5)
    assert eval_shape(fn, _A, _B).shape == ()


def test_logaddexp_jvp_and_vmap():
    _, t = jvp(b_logaddexp, (_A, _B), (np.ones_like(_A), np.zeros_like(_B)))
    assert np.isfinite(float(np.asarray(t)))
    out = np.asarray(vmap(lambda x: np.logaddexp(x, _B))(np.stack([_A, _A])))
    assert out.shape == (2, 3, 4)


# --- diff -------------------------------------------------------------------
def f_diff1(x):
    return np.sum(np.diff(x, axis=0) ** 2)


def f_diff2(x):
    return np.sum(np.diff(x, n=2, axis=1) * 3.0)


@pytest.mark.parametrize("fn", [f_diff1, f_diff2])
def test_diff_grad_vs_fd(fn):
    M = _rng.standard_normal((5, 5))
    assert np.allclose(np.asarray(grad(fn)(M)[0]), _fd1(fn, M), atol=1e-5)
    assert eval_shape(fn, M).shape == ()


def test_diff_vmap():
    M = _rng.standard_normal((5, 5))
    out = np.asarray(vmap(lambda m: np.diff(m, axis=0))(np.stack([M, M])))
    assert out.shape == (2, 4, 5)
    assert np.allclose(out[0], np.diff(M, axis=0))


# --- diag / diagonal --------------------------------------------------------
def f_diag_construct(v):
    return np.sum(np.diag(v, 1) ** 2)


def f_diag_extract(m):
    return np.sum(np.diag(m, -1) ** 2)


def f_diagonal(m):
    return np.sum(np.diagonal(m, 1) * 3.0)


def test_diag_grad_vs_fd():
    V = _rng.standard_normal(5)
    M = _rng.standard_normal((5, 5))
    R = _rng.standard_normal((4, 5))
    assert np.allclose(
        np.asarray(grad(f_diag_construct)(V)[0]), _fd1(f_diag_construct, V), atol=1e-5
    )
    assert np.allclose(
        np.asarray(grad(f_diag_extract)(M)[0]), _fd1(f_diag_extract, M), atol=1e-5
    )
    assert np.allclose(
        np.asarray(grad(f_diagonal)(R)[0]), _fd1(f_diagonal, R), atol=1e-5
    )


def test_diag_eval_shape_and_extract_vmap():
    V = _rng.standard_normal(5)
    M = _rng.standard_normal((5, 5))
    assert eval_shape(lambda v: np.diag(v, 1), V).shape == (6, 6)
    assert eval_shape(lambda m: np.diag(m), M).shape == (5,)
    out = np.asarray(vmap(lambda m: np.diag(m))(np.stack([M, M])))
    assert out.shape == (2, 5) and np.allclose(out[0], np.diag(M))

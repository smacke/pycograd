# -*- coding: utf-8 -*-
"""numpy's ``np.r_[...]`` / ``np.c_[...]`` index-expression objects -- row/column-wise
concatenation of the bracketed pieces, where an int slice (``1:10``) expands to an arange.
Intercepted via the subscript handler and routed to concatenate-compositions, so tape-value
pieces stay differentiable (eager and in graph capture).

Module-level functions so pyccolo can re-instrument them from source."""
import numpy as np

from pycograd import capture, eval_shape, grad, jvp, value_and_grad, vmap

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


def f_r_single(x):
    return np.sum(np.r_[x] ** 2)


def f_r_double(x):
    return np.sum(np.r_[x, x] ** 2)


_C2 = _rng.standard_normal((3, 2))


def f_r_node_const(x):
    return np.sum(np.r_[x, _C2] ** 2)


def f_r_slice(x):
    c = np.ones(10)
    return np.sum(np.r_[x, c, 1:10] ** 2)


def f_c(x):
    return np.sum(np.c_[x, _C2, x] ** 2)


def test_r_c_grads():
    A2 = _rng.standard_normal((3, 2))
    A1 = _rng.standard_normal(10)
    for fn, a in [(f_r_single, A2), (f_r_double, A2), (f_r_node_const, A2), (f_c, A2)]:
        assert np.allclose(np.asarray(grad(fn)(a)[0]), _fd(fn, a), atol=1e-5)
    assert np.allclose(
        np.asarray(grad(f_r_slice)(A1)[0]), _fd(f_r_slice, A1), atol=1e-5
    )


def test_r_c_shapes_and_transforms():
    A2 = _rng.standard_normal((3, 2))
    assert eval_shape(lambda x: np.r_[x, x], A2).shape == (6, 2)
    assert eval_shape(lambda x: np.c_[x, _C2], A2).shape == (3, 4)
    _, t = jvp(f_r_double, (A2,), (np.ones_like(A2),))
    assert np.isfinite(float(np.asarray(t)))
    out = np.asarray(vmap(lambda x: np.r_[x, x])(np.stack([A2, A2])))
    assert out.shape == (2, 6, 2)


def test_r_c_capture_grad():
    # r_ lowers to concatenate, c_ to reshape+concatenate -- both graph-differentiable.
    A2 = _rng.standard_normal((3, 2))
    for fn in [f_r_double, f_c]:
        _, gg = value_and_grad(capture(fn, A2))(A2)
        _, ge = value_and_grad(fn)(A2)
        assert np.allclose(np.asarray(gg[0]), np.asarray(ge[0]), atol=1e-7)

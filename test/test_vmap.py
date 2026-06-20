# -*- coding: utf-8 -*-
"""vmap (auto-batching): the vectorized transform must match a plain Python loop.

``_loop_vmap`` (map ``f`` over the batch with a list comprehension, then stack) is the
oracle, exactly as the dummy-array path is the oracle for shape inference. Every
vectorized ``vmap(f)`` must equal it; gradients of vmapped functions are checked
against finite differences.
"""
import numpy as np
import pytest

from pycograd import grad, value_and_grad, vmap
from pycograd.transforms import per_example_grad


def _rng(seed=0):
    return np.random.default_rng(seed)


def _loop_vmap(f, in_axes=0):
    def wrapped(*args):
        axes = in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(args)
        batch = next(a.shape[ax] for a, ax in zip(args, axes) if ax is not None)
        outs = []
        for i in range(batch):
            sliced = [
                np.take(a, i, axis=ax) if ax is not None else a
                for a, ax in zip(args, axes)
            ]
            outs.append(np.asarray(f(*sliced)))
        return np.stack(outs)

    return wrapped


def _check(f, args, in_axes=0):
    got = np.asarray(vmap(f, in_axes=in_axes)(*args))
    ref = _loop_vmap(f, in_axes=in_axes)(*args)
    assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
    assert np.allclose(got, ref), f"value mismatch\n{got}\n{ref}"


# ---------------------------------------------------------------------------
# forward conformance vs the loop oracle
# ---------------------------------------------------------------------------
def elementwise(x):
    return np.exp(x) * 2.0 - 1.0


def reduce_all(x):
    return np.sum(x)


def reduce_axis(x):
    return np.sum(x, axis=0)


def matvec(x):
    return x @ np.ones(4)


def per_sample_dot(x):
    return x @ x


def reshape_fn(x):
    return x.reshape(2, 2)


def transpose_fn(x):
    return x.reshape(2, 2).T


def getitem_fn(x):
    return x[1:3]


def mlp(x, w1, b1, w2, b2):
    return np.tanh(x @ w1 + b1) @ w2 + b2


def test_vmap_elementwise():
    _check(elementwise, (_rng().standard_normal((6, 4)),))


def test_vmap_reduce_all():
    _check(reduce_all, (_rng().standard_normal((6, 4)),))


def test_vmap_reduce_axis():
    _check(reduce_axis, (_rng().standard_normal((6, 3, 5)),))


def test_vmap_matvec():
    _check(matvec, (_rng().standard_normal((6, 4)),))


def test_vmap_per_sample_dot():
    _check(per_sample_dot, (_rng().standard_normal((6, 4)),))


def test_vmap_reshape():
    _check(reshape_fn, (_rng().standard_normal((6, 4)),))


def test_vmap_transpose():
    _check(transpose_fn, (_rng().standard_normal((6, 4)),))


def test_vmap_getitem():
    _check(getitem_fn, (_rng().standard_normal((6, 5)),))


def matmul_shared(x, w):
    return x @ w


def test_vmap_matmul_shared_weight():
    r = _rng()
    _check(
        matmul_shared,
        (r.standard_normal((6, 4)), r.standard_normal((4, 3))),
        in_axes=(0, None),
    )


def test_vmap_matmul_both_batched():
    r = _rng()
    _check(
        matmul_shared,
        (r.standard_normal((6, 2, 4)), r.standard_normal((6, 4, 3))),
        in_axes=(0, 0),
    )


def test_vmap_mlp_shared_params():
    r = _rng()
    X = r.standard_normal((6, 3))
    w1, b1 = r.standard_normal((3, 5)), r.standard_normal((5,))
    w2, b2 = r.standard_normal((5, 2)), r.standard_normal((2,))
    _check(mlp, (X, w1, b1, w2, b2), in_axes=(0, None, None, None, None))


def test_vmap_out_axes():
    f = elementwise
    X = _rng().standard_normal((6, 4))
    got = np.asarray(vmap(f, out_axes=1)(X))
    ref = np.moveaxis(_loop_vmap(f)(X), 0, 1)
    assert got.shape == ref.shape and np.allclose(got, ref)


# ---------------------------------------------------------------------------
# gradient composition
# ---------------------------------------------------------------------------
def _fd(f, x, h=1e-5):
    g = np.zeros_like(x)
    flat = x.reshape(-1)
    for i in range(flat.size):
        xp = flat.copy()
        xp[i] += h
        xm = flat.copy()
        xm[i] -= h
        g.reshape(-1)[i] = (f(xp.reshape(x.shape)) - f(xm.reshape(x.shape))) / (2 * h)
    return g


def batch_loss(x):
    # x: (B, d) -> scalar mean of per-sample squared norms
    return np.sum(vmap(lambda r: r @ r)(x)) / x.shape[0]


def test_grad_of_vmap():
    X = _rng().standard_normal((5, 4))
    (g,) = grad(batch_loss)(X)
    expected = _fd(lambda z: float(np.sum(np.sum(z * z, axis=1)) / z.shape[0]), X)
    assert np.allclose(g, expected, atol=1e-5)


def test_value_and_grad_of_vmap_unwraps():
    X = _rng().standard_normal((5, 4))
    val, (g,) = value_and_grad(batch_loss)(X)
    assert np.isscalar(val) or np.ndim(val) == 0
    assert g.shape == X.shape


def per_sample_sq(x):
    return x @ x  # scalar per example


def test_per_example_grad_matches_loop():
    X = _rng().standard_normal((5, 4))
    g = per_example_grad(per_sample_sq)(X)
    # d/dx (x . x) = 2x, per sample
    assert g.shape == X.shape
    assert np.allclose(g, 2 * X, atol=1e-6)


# ---------------------------------------------------------------------------
# documented v1 limits
# ---------------------------------------------------------------------------
def test_nested_vmap_not_supported():
    X = _rng().standard_normal((3, 4))
    with pytest.raises(NotImplementedError):
        vmap(vmap(lambda r: r * 2.0))(X)


def test_coverage_matches_intercept():
    from pycograd.batching import _BATCH
    from pycograd.ops import _INTERCEPT

    assert set(_BATCH) == set(_INTERCEPT)

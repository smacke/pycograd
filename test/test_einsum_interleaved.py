# -*- coding: utf-8 -*-
"""numpy's *interleaved* einsum form -- ``np.einsum(op0, sublist0, op1, sublist1, ..., out)``
with integer index labels instead of a subscript string. It normalizes to the subscript form
and reuses the full einsum machinery, so reverse / forward / vmap / eval_shape all work.

Module-level functions so pyccolo can re-instrument them from source."""
import numpy as np
import pytest

from pycograd import eval_shape, grad, jvp, vmap

_rng = np.random.default_rng(0)
_B = _rng.standard_normal((4, 5))
_C = _rng.standard_normal((5, 6))


def _fd(f, x, eps=1e-6):
    out = np.zeros(np.shape(x))
    for i in range(np.size(x)):
        xp = x.copy()
        xm = x.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        out.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return out


def f_matmul(a):  # ij,jk->ik
    return np.sum(np.einsum(a, [0, 1], _B, [1, 2], [0, 2]) ** 2)


def f_transpose(a):  # implicit output (ji)
    return np.sum(np.einsum(a, [1, 0]) * 3.0)


def f_three(a):  # ij,jk,kl->il
    return np.sum(np.einsum(a, [0, 1], _B, [1, 2], _C, [2, 3], [0, 3]))


def f_covsum(a):  # tij,tik->jk (repeated operand, summed batch label)
    return np.sum(np.einsum(a, [0, 1, 2], a, [0, 1, 3], [2, 3]))


_A = _rng.standard_normal((3, 4))
_A3 = _rng.standard_normal((2, 3, 4))


@pytest.mark.parametrize(
    "fn, a", [(f_matmul, _A), (f_transpose, _A), (f_three, _A), (f_covsum, _A3)]
)
def test_interleaved_grad_vs_fd(fn, a):
    assert np.allclose(np.asarray(grad(fn)(a)[0]), _fd(fn, a), atol=1e-5)
    assert eval_shape(fn, a).shape == ()


def test_interleaved_matches_string_form():
    # The interleaved form is exactly the subscript form.
    assert np.allclose(
        np.asarray(
            eval_shape(lambda x: np.einsum(x, [0, 1], _B, [1, 2], [0, 2]), _A).shape
        ),
        np.asarray(eval_shape(lambda x: np.einsum("ij,jk->ik", x, _B), _A).shape),
    )


def test_interleaved_jvp_and_vmap():
    _, t = jvp(f_matmul, (_A,), (np.ones_like(_A),))
    assert np.isfinite(float(np.asarray(t)))
    v = np.asarray(
        vmap(lambda x: np.einsum(x, [0, 1], _B, [1, 2], [0, 2]))(np.stack([_A, _A]))
    )
    assert v.shape == (2, 3, 5)
    assert np.allclose(v[0], np.einsum("ij,jk->ik", _A, _B))

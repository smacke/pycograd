# -*- coding: utf-8 -*-
"""Tests for the fused ``einsum`` primitive: forward values against ``np.einsum``,
gradients against finite differences across contraction shapes (matmul, batched,
transpose, outer, reduction, 3-operand, ellipsis with broadcasting), and composition
with vmap / jvp / eval_shape. The within-operand diagonal must still raise clearly."""
import pytest

np = pytest.importorskip("numpy")

from pycograd import ShapeDtypeStruct as S  # noqa: E402
from pycograd import einsum, eval_shape, grad, jvp, value_and_grad, vmap  # noqa: E402


def finite_diff(f, args, h=1e-6):
    def s(*a):
        return float(np.sum(f(*a)))

    base = [np.array(a, dtype=float) for a in args]
    grads = []
    for i, a in enumerate(base):
        g = np.zeros_like(a)
        for idx in np.ndindex(a.shape):
            up = [x.copy() for x in base]
            dn = [x.copy() for x in base]
            up[i][idx] += h
            dn[i][idx] -= h
            g[idx] = (s(*up) - s(*dn)) / (2 * h)
        grads.append(g)
    return tuple(grads)


def _assert_grads_match(f, args, atol=1e-5):
    _, ad = value_and_grad(f)(*args)
    fd = finite_diff(f, args)
    assert len(ad) == len(fd)
    for g_ad, g_fd in zip(ad, fd):
        assert np.allclose(g_ad, g_fd, atol=atol), (g_ad, g_fd)


# Each case: (subscripts, operand shapes). Covers matmul, transpose, trace-free batched
# contraction, outer product, axis reduction, sum-all, and a 3-operand chain.
CASES = [
    ("ij,jk->ik", [(3, 4), (4, 5)]),
    ("ij->ji", [(3, 4)]),
    ("ij->i", [(3, 4)]),  # reduction over j (the summed-out-axis path)
    ("ij->", [(3, 4)]),  # full reduction
    ("i,j->ij", [(3,), (4,)]),  # outer product
    ("bij,bjk->bik", [(2, 3, 4), (2, 4, 5)]),  # batched matmul
    ("bhqd,bhkd->bhqk", [(2, 2, 3, 4), (2, 2, 5, 4)]),  # attention scores
    ("ij,jk,kl->il", [(2, 3), (3, 4), (4, 5)]),  # 3-operand chain
    ("ij,ij->ij", [(3, 4), (3, 4)]),  # hadamard
    ("...ij,...jk->...ik", [(2, 3, 4), (2, 4, 5)]),  # ellipsis batched matmul
    ("...ij->...ji", [(2, 3, 4)]),  # ellipsis transpose of the last two axes
    ("...ij,jk->...ik", [(2, 6, 3, 4), (4, 5)]),  # differing ellipsis rank
    ("...qd,...kd->...qk", [(2, 2, 3, 4), (2, 2, 5, 4)]),  # ellipsis attention
    ("...ij", [(2, 3, 4)]),  # implicit-output ellipsis (identity copy)
]


@pytest.mark.parametrize("subscripts,shapes", CASES)
def test_einsum_forward_matches_numpy(subscripts, shapes):
    # ``einsum`` is a primitive: a direct call always returns a Var, read via ``.value``.
    rng = np.random.default_rng(0)
    arrs = [rng.standard_normal(s) for s in shapes]
    got = einsum(subscripts, *arrs).value
    assert np.allclose(got, np.einsum(subscripts, *arrs))


@pytest.mark.parametrize("subscripts,shapes", CASES)
def test_einsum_grad_matches_finite_diff(subscripts, shapes):
    rng = np.random.default_rng(1)
    arrs = [rng.standard_normal(s) for s in shapes]

    # ``f`` closes over ``subscripts``; pyccolo recompiles the instrumented loss but now
    # preserves captured free variables, so this resolves on the real interception path.
    def f(*operands):
        return np.einsum(subscripts, *operands)

    _assert_grads_match(f, tuple(arrs))


def test_einsum_implicit_output():
    # No "->": numpy's implicit output is the labels appearing once, sorted.
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((3, 4)), rng.standard_normal((4, 5))
    assert np.allclose(einsum("ij,jk", a, b).value, np.einsum("ij,jk", a, b))


def test_einsum_composes_with_vmap():
    rng = np.random.default_rng(3)
    a_batch = rng.standard_normal((6, 3, 4))
    w = rng.standard_normal((4, 5))

    def f(a, b):
        return np.einsum("ij,jk->ik", a, b)

    out = vmap(f, in_axes=(0, None))(a_batch, w)
    assert out.shape == (6, 3, 5)
    for i in range(6):
        assert np.allclose(np.asarray(out[i]), a_batch[i] @ w)


def test_einsum_per_sample_grad_via_vmap_grad():
    rng = np.random.default_rng(4)
    a_batch = rng.standard_normal((5, 3, 4))
    w = rng.standard_normal((4, 2))

    def loss(a, b):
        return np.sum(np.einsum("ij,jk->ik", a, b) ** 2)

    ga, gw = vmap(grad(loss), in_axes=(0, None))(a_batch, w)
    assert ga.shape == (5, 3, 4)
    assert gw.shape == (5, 4, 2)
    for i in range(5):
        gi_a, gi_w = grad(loss)(a_batch[i], w)
        assert np.allclose(ga[i], gi_a)
        assert np.allclose(gw[i], gi_w)


def test_einsum_jvp_matches_grad_contraction():
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal((3, 4)), rng.standard_normal((4, 5))
    va, vb = rng.standard_normal((3, 4)), rng.standard_normal((4, 5))

    def f(x, y):
        return np.einsum("ij,jk->ik", x, y)

    primal, tangent = jvp(f, (a, b), (va, vb))
    assert np.allclose(np.asarray(primal), a @ b)
    # d(a@b) = va@b + a@vb
    assert np.allclose(np.asarray(tangent), va @ b + a @ vb)


def test_einsum_eval_shape():
    out = eval_shape(
        lambda a, b: np.einsum("bij,bjk->bik", a, b), S((7, 3, 4)), S((7, 4, 5))
    )
    assert tuple(out.shape) == (7, 3, 5)


def test_einsum_within_operand_diagonal():
    # A label repeated within one operand extracts that diagonal (the operand is gathered via
    # getitem before the contraction, so the gradient scatters back onto the diagonal).
    x = np.arange(9.0).reshape(3, 3)

    def f(a):
        return np.sum(einsum("ii->i", a) ** 2)

    val, grads = value_and_grad(f)(x)
    assert np.isclose(float(np.asarray(val)), float(np.sum(np.einsum("ii->i", x) ** 2)))
    expected = np.zeros((3, 3))
    np.fill_diagonal(expected, 2 * np.diag(x))  # gradient only on the diagonal
    assert np.allclose(np.asarray(grads[0]), expected)


def test_einsum_ellipsis_broadcasts_size_one():
    # A size-1 ellipsis axis broadcasts against size N (numpy parity); the gradient
    # of the broadcast operand must come back at its true (size-1) shape.
    rng = np.random.default_rng(6)
    a = rng.standard_normal((1, 2, 3))
    b = rng.standard_normal((4, 3, 5))
    spec = "...ij,...jk->...ik"
    assert np.allclose(einsum(spec, a, b).value, np.einsum(spec, a, b))

    def f(x, y):
        return np.einsum(spec, x, y)

    _, (ga, gb) = value_and_grad(f)(a, b)
    assert ga.shape == (1, 2, 3) and gb.shape == (4, 3, 5)
    _assert_grads_match(f, (a, b))


def test_einsum_ellipsis_composes_with_vmap():
    rng = np.random.default_rng(7)
    a_batch = rng.standard_normal((6, 2, 3, 4))
    w = rng.standard_normal((4, 5))

    def f(a, b):
        return np.einsum("...ij,jk->...ik", a, b)

    out = vmap(f, in_axes=(0, None))(a_batch, w)
    assert out.shape == (6, 2, 3, 5)
    for i in range(6):
        assert np.allclose(
            np.asarray(out[i]), np.einsum("...ij,jk->...ik", a_batch[i], w)
        )


def test_einsum_ellipsis_eval_shape():
    out = eval_shape(
        lambda a, b: np.einsum("...ij,...jk->...ik", a, b), S((7, 3, 4)), S((7, 4, 5))
    )
    assert tuple(out.shape) == (7, 3, 5)

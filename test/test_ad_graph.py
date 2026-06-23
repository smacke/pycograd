# -*- coding: utf-8 -*-
"""Tests for ``grad_graph`` (autodiff on the capture IR): differentiating a captured
forward graph yields one graph computing value + gradients, matching ``value_and_grad``.
G1 covers the smooth/linear ops; G2 the mask ops (relu/softmax); G3 the cross-pass CSE.
"""
import pytest

np = pytest.importorskip("numpy")

from pycograd import value_and_grad  # noqa: E402
from pycograd.ad_graph import grad_graph  # noqa: E402
from pycograd.capture import capture, eval_graph  # noqa: E402
from pycograd.tensor import _value  # noqa: E402
from pycograd.tree import tree_leaves  # noqa: E402


def _rng(seed):
    return np.random.default_rng(seed)


def _grads_match(gg, args, loss_fn, atol=1e-9):
    val, grads = eval_graph(gg, *args)
    ref_val, ref_grads = value_and_grad(loss_fn)(*args)
    assert np.allclose(float(_value(val)), float(_value(ref_val)), atol=atol)
    got = [np.asarray(_value(x)) for x in grads]
    ref = [np.asarray(x) for arg in ref_grads for x in tree_leaves(arg)]
    assert len(got) == len(ref) and got, (len(got), len(ref))
    for a, b in zip(got, ref):
        assert np.allclose(a, b, atol=atol), (a, b)


# --- G1: smooth / linear ops ------------------------------------------------
def _smooth_loss(x, w):
    h = np.tanh(x @ w)  # matmul + tanh
    return np.sum(h * h)  # mul + sum


def test_grad_graph_smooth_roundtrip():
    x, w = _rng(0).standard_normal((4, 3)), _rng(1).standard_normal((3, 2))
    _grads_match(grad_graph(capture(_smooth_loss, x, w)), (x, w), _smooth_loss)


def _reshape_einsum_loss(x, w):
    y = np.einsum("ij,jk->ik", x, w)  # einsum
    z = np.reshape(y, (y.shape[0] * y.shape[1],))  # reshape
    return np.sum(np.exp(z))  # exp + sum


def test_grad_graph_reshape_einsum_roundtrip():
    x, w = _rng(2).standard_normal((3, 4)), _rng(3).standard_normal((4, 2))
    _grads_match(
        grad_graph(capture(_reshape_einsum_loss, x, w)), (x, w), _reshape_einsum_loss
    )

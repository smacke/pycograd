# -*- coding: utf-8 -*-
"""Targeted coverage of the *differentiable* reverse pass (``Var._backward_differentiable``)
-- mechanism #2: the eager higher-order backward that runs when a ``grad`` is nested inside
another ``grad``/``jvp``. It accumulates each node's cotangent as a tape/level-connected
``Var`` (via ``ops._VJP_FOR``) so the enclosing transform can differentiate the gradient.

``test_highorder.py`` covers the big compositions (Hessians of an MLP, etc.); this file is a
*per-op* sweep of second derivatives, so every primitive's differentiable VJP is exercised
on its own -- including the recently shared ``pow``/``gated_act`` rules and the mask-routed
``maximum``. Both triggers are hit: ``jvp(grad(f))`` (forward-over-reverse) and the literal
``grad(grad(f))`` (reverse-over-reverse).

Functions are MODULE-level so pyccolo can re-instrument them from source.
"""
import numpy as np
import pytest

from pycograd import d_sigmoid, gated_act, grad, jvp


def _rng(seed=0):
    return np.random.default_rng(seed)


# --- one scalar loss per primitive (each isolates that op's local derivative) ----------
def _l_exp(x):
    return np.sum(np.exp(x))


def _l_log(x):
    return np.sum(np.log(x))  # x > 0


def _l_sin(x):
    return np.sum(np.sin(x))


def _l_cos(x):
    return np.sum(np.cos(x))


def _l_tanh(x):
    return np.sum(np.tanh(x))


def _l_sqrt(x):
    return np.sum(np.sqrt(x))  # x > 0


def _l_arctan(x):
    return np.sum(np.arctan(x))


def _l_sigmoid(x):
    return np.sum(d_sigmoid(x))


def _l_square(x):
    return np.sum(np.square(x))


def _l_reciprocal(x):
    return np.sum(np.reciprocal(x))  # x != 0


def _l_pow2(x):
    return np.sum(x**2)  # const exponent -> _pow_base_deriv (shared)


def _l_pow3(x):
    return np.sum(x**3)


def _l_div(x):
    return np.sum(2.0 / x)  # x != 0


def _l_gated(x):
    return np.sum(gated_act(x, 0.5 * x))  # tanh(x)*sigmoid(0.5x) -> _gated_act_coeffs


def _l_relu_sq(x):
    return np.sum(np.maximum(x, 0.0) ** 2)  # mask-routed select under 2nd order


def _normal(rng):
    return 0.5 * rng.standard_normal(5)


def _positive(rng):
    return rng.uniform(0.5, 2.0, size=5)


def _mixed_sign(rng):
    # away from the relu kink, so finite differences stay clean while the mask is nontrivial.
    return rng.choice([-1.0, 1.0], size=5) * rng.uniform(0.5, 2.0, size=5)


# (id, loss, x_factory)
_OPS = [
    ("exp", _l_exp, _normal),
    ("log", _l_log, _positive),
    ("sin", _l_sin, _normal),
    ("cos", _l_cos, _normal),
    ("tanh", _l_tanh, _normal),
    ("sqrt", _l_sqrt, _positive),
    ("arctan", _l_arctan, _normal),
    ("sigmoid", _l_sigmoid, _normal),
    ("square", _l_square, _normal),
    ("reciprocal", _l_reciprocal, _positive),
    ("pow2", _l_pow2, _normal),
    ("pow3", _l_pow3, _normal),
    ("div", _l_div, _positive),
    ("gated_act", _l_gated, _normal),
    ("relu_sq", _l_relu_sq, _mixed_sign),
]


def _grad_vec(loss, x):
    return np.asarray(grad(loss)(x)[0])


def _hvp_fd(loss, x, v, eps=1e-5):
    """Central finite-difference Hessian-vector product: d/deps grad(loss)(x + eps v)."""
    return (_grad_vec(loss, x + eps * v) - _grad_vec(loss, x - eps * v)) / (2 * eps)


@pytest.mark.parametrize("cid,loss,xf", _OPS, ids=[c[0] for c in _OPS])
def test_hvp_matches_finite_difference(cid, loss, xf):
    # jvp(grad(f)) is a Hessian-vector product -- forward-over-reverse, which routes the
    # inner backward through _backward_differentiable. Check it vs finite differences.
    rng = _rng(hash(cid) % 1000)
    x, v = xf(rng), rng.standard_normal(5)
    hvp = np.asarray(jvp(grad(loss), (x,), (v,))[1][0])
    fd = _hvp_fd(loss, x, v)
    assert np.allclose(hvp, fd, atol=1e-5), (cid, hvp, fd)


# --- reverse-over-reverse (the OTHER trigger): literal grad(grad(f)) --------------------
# The inner grad runs its differentiable backward because the outer grad pushes a reverse
# marker level; the outer grad then walks the cotangent Var graph it produced.
def _grad_pow3_total(x):
    (g,) = grad(_l_pow3)(x)  # g = 3 x**2
    return np.sum(g)  # sum(3 x**2)


def test_grad_of_grad_through_pow():
    x = _normal(_rng(1))
    g2 = np.asarray(grad(_grad_pow3_total)(x)[0])  # d/dx sum(3 x**2) = 6 x
    assert np.allclose(g2, 6.0 * x, atol=1e-9)


def _grad_gated_total(x):
    (g,) = grad(_l_gated)(x)
    return np.sum(g)


def test_grad_of_grad_through_gated_act_matches_fd():
    x = _normal(_rng(2))
    g2 = np.asarray(grad(_grad_gated_total)(x)[0])  # reverse-over-reverse
    # finite-difference the inner total-gradient to check the second derivative.
    eps = 1e-5
    e = np.eye(x.size)
    fd = np.array(
        [
            (_grad_gated_total(x + eps * e[i]) - _grad_gated_total(x - eps * e[i]))
            / (2 * eps)
            for i in range(x.size)
        ]
    )
    assert np.allclose(g2, fd, atol=1e-5)

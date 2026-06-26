# -*- coding: utf-8 -*-
"""Complex-number autodiff (real / non-holomorphic convention).

A complex tensor ``z = a + bi`` is differentiated as the real pair ``(a, b)``: for a
real-valued loss ``f: C^n -> R`` the gradient is ``dL/da + i*dL/db`` and satisfies the
adjoint identity ``Re(<grad, v>) = D_v f`` (the criterion the autograd suite checks).
Complex is auto-detected from the input dtype -- no ``dtype("complex128")`` block is
required, though one may be used. ``holomorphic_grad`` gives the analytic ``f'(z)`` for an
analytic ``f: C -> C``. Order-dependent ops (max/min/sort/...) raise on complex.
"""
import numpy as np
import pytest

import pycograd as pg
from pycograd import capture
from pycograd import compile as C
from pycograd import grad, holomorphic_grad, jvp, value_and_grad, vmap
from pycograd.tensor import _value
from pycograd.tree import tree_leaves

Z = np.array([1 + 2j, -0.5 + 0.3j, 2 - 1j])


# Module-level targets (re-sourced by the tracer). All return a real scalar.
def loss_abs2(z):
    return np.sum(np.real(z * np.conj(z)))


def loss_realexp(z):
    return np.sum(np.real(np.exp(z)))


def loss_resin(z):
    return np.sum(np.real(np.sin(z)))


def loss_absnp(z):
    return np.sum(np.abs(z))


def loss_angle(z):
    return np.sum(np.angle(z) ** 2)


def loss_imag(z):
    return np.sum(np.imag(z) ** 2)


def loss_logabs(z):
    return np.sum(np.log(np.abs(z)))


_NP = {
    loss_abs2: lambda z: np.sum((z * np.conj(z)).real),
    loss_realexp: lambda z: np.sum(np.exp(z).real),
    loss_resin: lambda z: np.sum(np.sin(z).real),
    loss_absnp: lambda z: np.sum(np.abs(z)),
    loss_angle: lambda z: np.sum(np.angle(z) ** 2),
    loss_imag: lambda z: np.sum(np.imag(z) ** 2),
    loss_logabs: lambda z: np.sum(np.log(np.abs(z))),
}

_TARGETS = list(_NP)


def _dir_deriv(fnp, z, v, eps=1e-6):
    return (fnp(z + eps * v) - fnp(z - eps * v)) / (2 * eps)


@pytest.mark.parametrize("target", _TARGETS)
def test_grad_satisfies_adjoint_identity(target):
    # Re(<grad, v>) == D_v f for random complex directions v (the autograd-suite criterion).
    (g,) = grad(target)(Z)
    assert g.dtype.kind == "c"
    rng = np.random.default_rng(0)
    for _ in range(4):
        v = rng.standard_normal(Z.shape) + 1j * rng.standard_normal(Z.shape)
        lhs = np.real(np.sum(np.conj(g) * v))
        rhs = _dir_deriv(_NP[target], Z, v)
        assert np.isclose(lhs, rhs, atol=1e-4), (target.__name__, lhs, rhs)


def test_grad_abs2_is_2z():
    # d/dz |z|^2 = 2z in pycograd's convention (dL/da + i dL/db).
    (g,) = grad(loss_abs2)(Z)
    assert np.allclose(g, 2 * Z)


def test_dtype_context_matches_no_context():
    (g_no,) = grad(loss_realexp)(Z)
    with pg.dtype("complex128"):
        (g_ctx,) = grad(loss_realexp)(Z)
    assert np.allclose(g_no, g_ctx)


def test_graph_capture_matches_eager():
    for target in (loss_abs2, loss_realexp, loss_absnp):
        (g_eager,) = grad(target)(Z)
        v, grads = value_and_grad(capture(target, Z))(Z)
        g_graph = np.asarray(_value(tree_leaves(grads)[0]))
        assert np.allclose(g_eager, g_graph), target.__name__


def test_vmap_complex():
    zs = np.stack([Z, 2 * Z])
    out = np.asarray(vmap(loss_abs2)(zs))
    assert np.allclose(out, [_NP[loss_abs2](Z), _NP[loss_abs2](2 * Z)])


def test_jvp_complex_directional():
    v = np.ones_like(Z)
    _primal, tangent = jvp(loss_realexp, (Z,), (v,))
    fd = _dir_deriv(_NP[loss_realexp], Z, v)
    assert np.allclose(float(np.real(np.asarray(tangent))), fd, atol=1e-4)


# -- holomorphic_grad: analytic f'(z) ---------------------------------------
def f_square(z):
    return z * z


def f_exp(z):
    return np.exp(z)


def f_cube(z):
    return z * z * z


def test_holomorphic_grad():
    assert np.allclose(holomorphic_grad(f_square)(Z), 2 * Z)
    assert np.allclose(holomorphic_grad(f_exp)(Z), np.exp(Z))
    assert np.allclose(holomorphic_grad(f_cube)(Z), 3 * Z * Z)


def test_holomorphic_grad_scalar():
    z = 1.5 + 0.7j
    assert np.allclose(holomorphic_grad(f_square)(z), 2 * z)


# -- astype real <-> complex ------------------------------------------------
def cast_then_abs2(x):
    z = x.astype("complex128")
    return np.sum(np.real(z * np.conj(z)))  # == sum(x^2)


def test_astype_real_to_complex_grad():
    x = np.array([1.0, 2.0, 3.0])
    (g,) = grad(cast_then_abs2)(x)
    assert g.dtype.kind == "f"
    assert np.allclose(g, 2 * x)


# -- compile backends agree with eager --------------------------------------
@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_compile_backend_matches_eager(backend):
    pytest.importorskip(backend)
    for target in (loss_abs2, loss_realexp):
        (g_eager,) = grad(target)(Z)
        _v, (g_be,) = C.value_and_grad(target, backend=backend)(Z)
        assert np.allclose(g_eager, np.asarray(g_be)), (backend, target.__name__)


# -- order-dependent ops reject complex -------------------------------------
def f_max(z):
    return np.sum(np.maximum(z, 0.0))


def f_sort(z):
    return np.sum(np.sort(z))


def f_maxreduce(z):
    return np.max(z)


@pytest.mark.parametrize("target", [f_max, f_sort, f_maxreduce])
def test_order_ops_reject_complex(target):
    with pytest.raises(TypeError, match="unordered"):
        grad(target)(Z)

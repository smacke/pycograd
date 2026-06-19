# -*- coding: utf-8 -*-
"""The working-dtype seam: compute the tape (and compile targets) in a chosen precision.

Default behavior is float64 (covered by the rest of the suite); here we assert that
``pg.dtype(...)`` / the ``dtype=`` kwargs actually switch the precision end to end -- the
tape's ``Var``s and gradients, parameters built under the block, the optimizer state that
consumes them, and the compiled-to-torch forward -- while staying numerically faithful to
the float64 result. bfloat16 (via the optional ``ml_dtypes`` package) is exercised where
that package is installed.

Target functions are module-level so the tracer can re-source them (getsource + recompile),
per the suite's convention.
"""
import numpy as np
import pytest

import pycograd as pg
from pycograd import Adam
from pycograd import compile as C
from pycograd.dtypes import current_dtype, resolve_dtype

# Module-level data + targets so the instrumented functions reference globals.
_W = np.array([[0.5, -0.3, 0.2], [0.1, 0.4, -0.6]])  # (2, 3)
_X = np.array([0.7, -0.4])  # (2,)


def quad(x):
    return np.sum(x * x)


def tanh_net(x):
    z = x @ _W
    h = np.tanh(z) + np.exp(z)
    return np.sum(h * h)


def affine_loss(params, x):
    return np.sum((params["w"] * x + params["b"]) ** 2)


# ---------------------------------------------------------------------------
# resolve_dtype / current_dtype.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, np.float64),
        ("float64", np.float64),
        ("f64", np.float64),
        ("double", np.float64),
        ("float32", np.float32),
        ("f32", np.float32),
        ("single", np.float32),
        ("float16", np.float16),
        ("f16", np.float16),
        ("half", np.float16),
        (np.float32, np.float32),
        (np.dtype("float32"), np.float32),
    ],
)
def test_resolve_dtype_aliases(spec, expected):
    assert resolve_dtype(spec) == np.dtype(expected)


def test_resolve_dtype_rejects_non_float():
    # A real numpy int dtype is recognized but rejected as non-floating...
    with pytest.raises(ValueError, match="floating-point"):
        resolve_dtype(np.int32)
    # ...while an unknown friendly string is reported as unknown.
    with pytest.raises(ValueError, match="unknown dtype"):
        resolve_dtype("not-a-dtype")


def test_current_dtype_default_and_context():
    assert current_dtype() == np.dtype(np.float64)
    with pg.dtype("float32"):
        assert current_dtype() == np.dtype(np.float32)
        with pg.dtype("float16"):
            assert current_dtype() == np.dtype(np.float16)
        assert current_dtype() == np.dtype(np.float32)  # restored on exit
    assert current_dtype() == np.dtype(np.float64)  # restored on exit


# ---------------------------------------------------------------------------
# Tape dtype: Var + gradients.
# ---------------------------------------------------------------------------
def test_var_creates_in_working_dtype():
    assert pg.Var(_X).value.dtype == np.float64
    with pg.dtype("float32"):
        assert pg.Var(_X).value.dtype == np.float32
    # an explicit dtype= overrides the ambient default
    assert pg.Var(_X, dtype="float32").value.dtype == np.float32


@pytest.mark.parametrize("name", ["float32", "float16"])
def test_value_and_grad_dtype_context(name):
    dt = np.dtype(name)
    ref_v, (ref_g,) = pg.value_and_grad(tanh_net)(_X)  # float64 reference
    with pg.dtype(name):
        v, (g,) = pg.value_and_grad(tanh_net)(_X)
    assert g.dtype == dt
    assert np.asarray(v).dtype == dt
    # faithful to the float64 result at a tolerance scaled to the precision
    tol = 1e-2 if name == "float16" else 1e-4
    assert np.allclose(np.asarray(g, dtype=np.float64), ref_g, atol=tol, rtol=tol)


# ---------------------------------------------------------------------------
# Params + optimizer keep the parameter's precision across steps.
# ---------------------------------------------------------------------------
def test_params_built_in_working_dtype():
    with pg.dtype("float32"):
        p = pg.params(w=_W, b=np.zeros(3))
    assert p.w.value.dtype == np.float32
    assert p.b.value.dtype == np.float32


def test_optimizer_preserves_param_dtype():
    x = np.array([0.3, 0.1, -0.2])
    opt = Adam(lr=0.05)
    with pg.dtype("float32"):
        p = pg.params(w=np.array([1.0, -2.0, 0.5]), b=np.zeros(3))
        for _ in range(5):
            _loss, (gp, _gx) = pg.value_and_grad(affine_loss)(p, x)
            p = opt.step(p, gp)
    assert p["w"].value.dtype == np.float32
    # Adam moment buffers ([m, v]) track the parameter dtype too.
    assert opt._state[0][0].dtype == np.float32
    assert opt._state[0][1].dtype == np.float32


# ---------------------------------------------------------------------------
# bfloat16, end to end (needs the optional ml_dtypes package).
# ---------------------------------------------------------------------------
def test_bfloat16_end_to_end():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    bf16 = np.dtype(ml_dtypes.bfloat16)

    ref_v, (ref_g,) = pg.value_and_grad(tanh_net)(_X)  # float64 reference
    with pg.dtype("bf16"):
        assert pg.Var(_X).value.dtype == bf16
        v, (g,) = pg.value_and_grad(tanh_net)(_X)
    assert g.dtype == bf16
    # bf16 has ~3 decimal digits; a coarse tolerance still pins down correctness.
    assert np.allclose(g.astype(np.float64), ref_g, atol=0.2, rtol=0.2)
    assert abs(float(v) - float(ref_v)) <= 0.2 * abs(float(ref_v)) + 0.2


# ---------------------------------------------------------------------------
# Compile + export honor the dtype (torch path).
# ---------------------------------------------------------------------------
def test_compile_torch_float32_parity():
    pytest.importorskip("torch")
    # numpy float32 tape vs torch float32 compile -- both at the same precision.
    with pg.dtype("float32"):
        ref_v, (ref_g,) = pg.value_and_grad(tanh_net)(_X)
    v, (g,) = C.value_and_grad(tanh_net, backend="torch", dtype="float32")(_X)
    assert np.asarray(g).dtype == np.float32
    assert np.allclose(np.asarray(g), ref_g, atol=1e-4, rtol=1e-4)
    assert np.allclose(float(np.asarray(v)), float(ref_v), atol=1e-4, rtol=1e-4)


def test_to_torch_module_float32():
    torch = pytest.importorskip("torch")
    module = pg.to_torch_module(quad_params, pg.params(w=_W), dtype="float32")
    weights = list(module.parameters())
    assert weights and all(w.dtype == torch.float32 for w in weights)


def quad_params(params, x):
    return np.sum((params["w"] @ x) ** 2)

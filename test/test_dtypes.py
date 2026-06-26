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


def scalar_net(x):
    # Pure in x with no python-scalar literals (a scalar literal is a *new* value created at the
    # working dtype, which would upcast a low-precision x). So the result dtype follows x's dtype
    # -- the contract the precision tests below exercise.
    h = np.tanh(x) + np.exp(x)
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
def test_var_dtype_policy():
    # An *existing* float array keeps its own dtype -- the data dtype flows through the tape,
    # regardless of the ambient working dtype. The working dtype is only a *default for new
    # values* (raw python scalars/lists) and an explicit ``dtype=``.
    assert pg.Var(_X).value.dtype == np.float64
    with pg.dtype("float32"):
        assert pg.Var(_X).value.dtype == np.float64  # preserved, NOT force-cast
        assert pg.Var(_X.astype("float32")).value.dtype == np.float32  # preserved
        assert pg.Var(3.0).value.dtype == np.float32  # python scalar -> working dtype
        assert (
            pg.Var([1, 2, 3]).value.dtype == np.float32
        )  # python list -> working dtype
    assert pg.Var(_X, dtype="float32").value.dtype == np.float32  # explicit override


@pytest.mark.parametrize("name", ["float32", "float16"])
def test_grad_dtype_follows_data(name):
    # Precision is controlled by the *data* dtype (the working dtype is a creation default, not
    # a propagation cast): a float32/16 input yields a float32/16 value and gradient, like numpy
    # and autograd -- no context needed.
    dt = np.dtype(name)
    ref_v, (ref_g,) = pg.value_and_grad(scalar_net)(_X)  # float64 reference
    v, (g,) = pg.value_and_grad(scalar_net)(_X.astype(name))
    assert g.dtype == dt
    assert np.asarray(v).dtype == dt
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

    ref_v, (ref_g,) = pg.value_and_grad(scalar_net)(_X)  # float64 reference
    # a new value created under the bf16 context follows it; bf16 *data* flows as bf16
    with pg.dtype("bf16"):
        assert pg.Var(3.0).value.dtype == bf16
    v, (g,) = pg.value_and_grad(scalar_net)(_X.astype(bf16))
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


# ---------------------------------------------------------------------------
# astype: the in-graph precision cast (mixed precision). Differentiable across the
# eager, graph, vmap, and jvp surfaces; a cast is linear, so its VJP casts the
# cotangent back to the input dtype.
# ---------------------------------------------------------------------------
def upcast_sin_sum(x):
    # cast up to float64 for the nonlinearity, regardless of the working dtype
    return np.sum(np.sin(x.astype("float64")))


def astype_scaled_sum(x):
    # free-function form (``np.astype``) so the same forward also lowers onto the compile
    # backends, which intercept numpy calls but not bound ``.astype`` methods
    return np.sum(np.astype(x, "float32") * 2.0)


def test_astype_forward_dtype_and_value():
    x = np.arange(3.0)
    with pg.dtype("float32"):
        y = pg.Var(x).astype("float64")
    assert y.value.dtype == np.float64
    assert np.allclose(y.value, x)


def test_astype_grad_casts_back_to_input_dtype():
    # The leaf is float32; the cotangent flows back from the upcast-to-float64 region into
    # float32 (the cast's VJP is a cast-back to the input dtype).
    x32 = _X.astype("float32")
    ref_v, (ref_g,) = pg.value_and_grad(upcast_sin_sum)(_X)  # float64 reference
    v, (g,) = pg.value_and_grad(upcast_sin_sum)(x32)
    assert g.dtype == np.float32
    assert np.allclose(np.asarray(g, dtype=np.float64), ref_g, atol=1e-4, rtol=1e-4)


def test_astype_graph_differentiable():
    # value_and_grad(capture(f)) lowers the astype node and differentiates it on the graph.
    x32 = _X.astype("float32")
    gg = pg.value_and_grad(pg.capture(upcast_sin_sum, x32))
    v, grads = gg(x32)
    leaf = np.asarray(_value_leaf(grads))
    assert leaf.dtype == np.float32
    ref_v, (ref_g,) = pg.value_and_grad(upcast_sin_sum)(_X)
    assert np.allclose(np.asarray(leaf, dtype=np.float64), ref_g, atol=1e-4, rtol=1e-4)


def test_astype_vmaps():
    xs = np.arange(6.0).reshape(2, 3)
    out = pg.vmap(astype_scaled_sum)(xs)
    expected = [np.sum(r.astype("float32") * 2.0) for r in xs]
    assert np.allclose(np.asarray(out, dtype=np.float64), expected)


def test_astype_jvp():
    x = np.arange(3.0)
    v = np.ones(3)
    _primal, tangent = pg.jvp(astype_scaled_sum, (x,), (v,))
    assert np.allclose(float(np.asarray(tangent)), 2.0 * float(np.sum(v)))


def test_astype_rejects_non_float():
    from pycograd.ops import d_astype

    # A Var holds real-valued tensors; casting to a non-floating dtype is not a tape op.
    with pytest.raises(ValueError):
        d_astype(np.ones(3), np.int64)


def test_astype_torch_backend_parity():
    pytest.importorskip("torch")
    with pg.dtype("float32"):
        ref_v, (ref_g,) = pg.value_and_grad(astype_scaled_sum)(_X)
    v, (g,) = C.value_and_grad(astype_scaled_sum, backend="torch", dtype="float32")(_X)
    assert np.allclose(np.asarray(g), ref_g, atol=1e-4, rtol=1e-4)


def _value_leaf(grads):
    from pycograd.tensor import _value
    from pycograd.tree import tree_leaves

    return _value(tree_leaves(grads)[0])


# ---------------------------------------------------------------------------
# Captured-graph dtype: a captured forward+backward must stay in the working
# dtype, not silently promote to float64. The classic leak is a weak Python
# scalar (``np.maximum(x, 0.0)`` in relu, the ``* 1.0`` bool->float in its graph
# VJP, the ``/ N`` in mean's VJP) upcasting a float32 array via numpy's strong,
# dtype-based promotion -- where eager numpy weak-promotes the scalar and keeps
# float32. The fix lives in ``shapes._result_dtype`` (NEP 50 weak scalars,
# anchored to ``current_dtype()``) plus the backward seeds.
# ---------------------------------------------------------------------------
def relu_mlp_loss(x, y, w1, b1, w2, b2):
    # relu (weak 0.0) -> matmul -> softmax cross-entropy (mean's weak / N), so the
    # captured forward and the value_and_grad backward exercise every weak-scalar
    # and bool->float promotion site.
    h = pg.relu(x @ w1 + b1)
    return pg.cross_entropy(h @ w2 + b2, y)


def _float_node_dtypes(graph):
    # The dtypes of every floating-point node aval (skip the int shape-constants and
    # bool comparison masks, which legitimately stay i64/bool).
    return [
        node.aval.dtype for node in graph.nodes if np.dtype(node.aval.dtype).kind == "f"
    ]


@pytest.mark.parametrize("name", ["float32", "float16"])
def test_capture_keeps_working_dtype(name):
    dt = np.dtype(name)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 3)).astype(dt)
    y = np.eye(2, dtype=dt)[rng.integers(0, 2, size=8)]
    w1 = rng.standard_normal((3, 4)).astype(dt)
    b1 = np.zeros(4, dtype=dt)
    w2 = rng.standard_normal((4, 2)).astype(dt)
    b2 = np.zeros(2, dtype=dt)

    with pg.dtype(name):
        vg = pg.value_and_grad(pg.capture(relu_mlp_loss, x, y, w1, b1, w2, b2))
        fwd = pg.capture(relu_mlp_loss, x, y, w1, b1, w2, b2)

    # Every floating node -- forward and backward (the relu VJP's bool->float
    # ``mul(mask, 1.0)``, the mean VJP's ``/ N``, the ones-seed) -- is the working
    # dtype; nothing leaked to float64.
    for graph in (fwd, vg):
        floats = _float_node_dtypes(graph)
        assert floats, "expected some floating-point nodes"
        assert all(d == dt for d in floats), {str(d) for d in floats}


def test_capture_weak_scalar_does_not_upcast():
    # The core regression, dtype-block-free: a float32 input through relu's weak
    # ``0.0`` (and a weak ``* 2.0``) stays float32, matching eager numpy rather
    # than promoting to float64 the way strong ``np.result_type`` would.
    x = np.arange(6.0, dtype=np.float32).reshape(2, 3)
    g = pg.capture(lambda z: np.sum(np.maximum(z, 0.0) * 2.0 - 1.0), x)
    assert all(d == np.float32 for d in _float_node_dtypes(g))


def test_capture_default_context_stays_float64():
    # Default working dtype is float64: a float64 capture is byte-for-byte
    # unchanged (no weak-scalar rule narrows it).
    x = np.arange(6.0).reshape(2, 3)  # float64
    g = pg.capture(lambda z: np.sum(np.maximum(z, 0.0) * 2.0), x)
    assert all(d == np.float64 for d in _float_node_dtypes(g))


def test_bernoulli_and_max_grad_in_working_dtype():
    # Eager-tape leaks: bernoulli's mask and max/min's select-mask VJP used a bare
    # ``.astype(float)`` (float64). Under a float32 tape they must be float32.
    from pycograd import random as pgr

    with pg.dtype("float32"):
        assert pgr.bernoulli(pgr.key(0), 0.5, (4,)).dtype == np.float32

        def f(x):
            return np.sum(np.maximum(x, 0.0) + np.minimum(x, 0.0))

        _v, (gx,) = pg.value_and_grad(f)(np.array([1.0, -2.0, 3.0], dtype=np.float32))
        assert gx.dtype == np.float32

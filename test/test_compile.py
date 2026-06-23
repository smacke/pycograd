# -*- coding: utf-8 -*-
"""Cross-backend conformance: a net compiled to jax/torch/tf must agree with pycograd.

pycograd's own gradients are finite-difference-checked elsewhere (``test_autodiff``), so
"backend gradient == pycograd gradient" is a strong correctness test for the compiler:
it transitively validates the backend against finite differences. We assert *forward*
and *gradient* parity on the finite-diff-checked example models, for every framework
that is installed (others ``importorskip``).

Backend coverage per model reflects real operator limitations, not pycograd gaps:
TensorFlow's ``@`` needs rank>=2 (so the vector-weight logistic model is jax/torch only)
and a ``tf.Tensor`` has no ``.T`` (so the transformer block is jax/torch only).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import pycograd.compile as C
import pycograd.transforms as T
from pycograd import d_sigmoid, frozen, gated_act
from pycograd.examples import models as M
from pycograd.tree import tree_leaves

# Quiet TensorFlow's C++ logging. TF is imported lazily (only when a tf backend is
# constructed, well after this runs), so setting it here is still before any tf import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_FRAMEWORK_MODULE = {"jax": "jax", "torch": "torch", "tf": "tensorflow"}


def _reseed():
    # The example dropout draws from this module global; reseeding before each run
    # makes the random mask identical across backends so gradients are comparable.
    M._dropout_rng = np.random.default_rng(1234)


def _flat_grads(grads):
    return [np.asarray(g) for arg in grads for g in tree_leaves(arg) if g is not None]


def _rng(seed):
    return np.random.default_rng(seed)


def _gated_act_loss(w):
    # Exercise the fused ``gated_act`` primitive: each backend must lower it to its
    # native ``tanh(f) * sigmoid(s)`` and match pycograd's gradient.
    half = w.shape[1] // 2
    return np.sum(gated_act(w[:, :half], w[:, half:]) ** 2)


# (id, loss_fn, args_factory, backends that support it)
_CASES = [
    (
        "gated_act",
        _gated_act_loss,
        lambda: (_rng(5).standard_normal((4, 8)),),
        {"jax", "torch", "tf"},
    ),
    (
        "mlp_tree",
        M.mlp_tree_loss,
        lambda: (M._init_mlp_tree(_rng(1)),),
        {"jax", "torch", "tf"},
    ),
    (
        "mlp_batch",
        M.mlp_batch_loss,
        lambda: (M._init_mlp_tree(_rng(1)), M._Xc, M._Yoh),
        {"jax", "torch", "tf"},
    ),
    (
        "deep_ln_dropout",
        M.deep_loss,
        lambda: M._init_deep(_rng(2)),
        {"jax", "torch", "tf"},
    ),
    (
        "transformer",
        M.transformer_loss,
        lambda: M._init_transformer(_rng(3)),
        {"jax", "torch"},
    ),
    (
        "logistic_vec_w",
        M.logistic_loss,
        lambda: (_rng(0).standard_normal(2), 0.0),
        {"jax", "torch"},
    ),
    (
        # The RWKV recurrence (token-shift + the numerically-stable WKV scan) lowers
        # to all three frameworks: the one-hot embedding keeps it free of the fancy
        # integer indexing tf can't compile.
        "rwkv",
        M.rwkv_loss,
        lambda: M._init_rwkv(_rng(3), vocab=len(M._CHAR_VOCAB), d_model=8, n_blocks=1),
        {"jax", "torch", "tf"},
    ),
    # The gated recurrent cells lower to all three frameworks: the scans carry the
    # per-step state as a ``(1, D)`` row and slice each timestep as ``x[t : t + 1]``,
    # so every matmul is rank-2 (tf's ``@`` needs rank >= 2).
    (
        "rnn",
        M.rnn_loss,
        lambda: (M._init_rnn(_rng(3), vocab=len(M._CHAR_VOCAB), d_model=8),),
        {"jax", "torch", "tf"},
    ),
    (
        "gru",
        M.gru_loss,
        lambda: (M._init_gru(_rng(3), vocab=len(M._CHAR_VOCAB), d_model=8),),
        {"jax", "torch", "tf"},
    ),
    (
        "lstm",
        M.lstm_loss,
        lambda: (M._init_lstm(_rng(3), vocab=len(M._CHAR_VOCAB), d_model=8),),
        {"jax", "torch", "tf"},
    ),
]

_PARITY_PARAMS = [
    (cid, fn, argf, be) for cid, fn, argf, support in _CASES for be in sorted(support)
]


@pytest.mark.parametrize(
    "cid,fn,argf,backend",
    _PARITY_PARAMS,
    ids=[f"{cid}-{be}" for cid, _, _, be in _PARITY_PARAMS],
)
def test_forward_and_grad_parity(cid, fn, argf, backend):
    pytest.importorskip(_FRAMEWORK_MODULE[backend])

    _reseed()
    ref_v, ref_g = T.value_and_grad(fn)(*argf())
    _reseed()
    cmp_v, cmp_g = C.value_and_grad(fn, backend=backend)(*argf())

    assert np.allclose(ref_v, cmp_v, atol=1e-9, rtol=1e-7), f"{cid}/{backend} forward"

    rf, cf = _flat_grads(ref_g), _flat_grads(cmp_g)
    assert rf and len(rf) == len(cf)
    for a, b in zip(rf, cf):
        assert a.shape == b.shape
        assert np.allclose(a, b, atol=1e-8, rtol=1e-6), f"{cid}/{backend} grad"


def _sigmoid_loss(x):
    # Exercises the tape-only ``d_sigmoid`` primitive's compile lowering: it has no
    # numpy callable, so each backend maps the primitive itself to its native sigmoid.
    return np.sum(d_sigmoid(x) ** 2)


@pytest.mark.parametrize("backend", ["jax", "torch", "tf"])
def test_sigmoid_primitive_lowers(backend):
    pytest.importorskip(_FRAMEWORK_MODULE[backend])
    x = np.linspace(-3.0, 3.0, 7)
    ref_v, (ref_g,) = T.value_and_grad(_sigmoid_loss)(x)
    cmp_v, (cmp_g,) = C.value_and_grad(_sigmoid_loss, backend=backend)(x)
    assert np.allclose(ref_v, cmp_v, atol=1e-9)
    assert np.allclose(np.asarray(ref_g), np.asarray(cmp_g), atol=1e-8)


@pytest.mark.parametrize("backend", ["jax", "torch", "tf"])
def test_frozen_leaf_has_no_gradient(backend):
    pytest.importorskip(_FRAMEWORK_MODULE[backend])
    params = {
        "hidden": {"w": 0.1 * _rng(1).standard_normal((2, 16)), "b": np.zeros(16)},
        "out": {"w": frozen(0.1 * _rng(2).standard_normal((16, 3))), "b": np.zeros(3)},
    }
    _, (g,) = C.value_and_grad(M.mlp_tree_loss, backend=backend)(params)
    assert g["out"]["w"] is None  # frozen -> no gradient
    assert g["hidden"]["w"] is not None and g["out"]["b"] is not None


@pytest.mark.parametrize("backend", ["jax", "torch", "tf"])
def test_compile_to_returns_native_tensor(backend):
    mod = pytest.importorskip(_FRAMEWORK_MODULE[backend])
    params = M._init_mlp_tree(_rng(1))
    forward = C.compile_to(M.mlp_tree_loss, backend)
    # pass the params as the backend's own tensors
    be = C.get_backend(backend)
    tensor_params = {
        k: {kk: be.const(vv) for kk, vv in v.items()} for k, v in params.items()
    }
    out = forward(tensor_params)
    assert type(out).__module__.split(".")[0] in {mod.__name__, "jaxlib", "tensorflow"}
    assert np.isclose(
        float(np.asarray(be.to_numpy(out))), float(M.mlp_tree_loss(params))
    )


def test_importing_pycograd_pulls_in_no_framework():
    # Selecting a backend must be what triggers the framework import -- never `import
    # pycograd` itself. Checked in a clean subprocess so other tests' imports don't leak.
    code = (
        "import sys, pycograd, pycograd.compile, pycograd.backends, pycograd.tracer;"
        "print(','.join(m for m in ('torch','jax','tensorflow') if m in sys.modules))"
    )
    out = subprocess.check_output([sys.executable, "-c", code]).decode().strip()
    assert out == "", f"frameworks leaked on import: {out}"


def test_backend_imports_framework_only_on_demand():
    # get_backend('jax') (or torch/tf) is the first thing that may import the framework.
    pytest.importorskip("jax")
    code = (
        "import sys, pycograd;"
        "assert 'jax' not in sys.modules;"
        "pycograd.get_backend('jax');"
        "assert 'jax' in sys.modules;"
        "print('ok')"
    )
    out = subprocess.check_output([sys.executable, "-c", code]).decode().strip()
    assert out == "ok"


# An ambient-DSL forward: references weights injected by `with weights:`. Module-level so
# the pyccolo-instrumented compile runner (recompiled from source) sees them as globals.
def _dsl_mlp_forward(x):
    h = np.maximum(0.0, x @ ew1 + eb1)  # noqa: F821  (ambient weights)
    return h @ ew2 + eb2  # noqa: F821


# Applies an intercepted *unary* numpy function (np.exp) directly to a bare ambient
# weight. Under a compile backend pyccolo swaps np.exp before the proxy's
# __array_ufunc__ can resolve it, so resolve_call must coerce the bare weight itself.
def _dsl_unary_on_weight():
    return np.sum(np.exp(ewd) * (_DSLX @ ew1))  # noqa: F821  (ambient weights)


_DSLX = np.array([[0.5, -1.0], [2.0, 0.3]])


@pytest.mark.parametrize("backend", ["jax", "torch", "tf"])
def test_ambient_unary_on_bare_weight_compiles(backend):
    pytest.importorskip(_FRAMEWORK_MODULE[backend])
    from pycograd import params

    weights = params(
        ewd=_rng(1).standard_normal(16), ew1=_rng(2).standard_normal((2, 16))
    )
    with weights:
        v_np, g_np = weights.grad(_dsl_unary_on_weight)
        v_be, g_be = weights.grad(_dsl_unary_on_weight, backend=backend)
    assert np.allclose(v_np, v_be, atol=1e-9)
    for k in weights:
        assert np.allclose(np.asarray(g_np[k]), np.asarray(g_be[k]), atol=1e-8)


def test_paramdict_to_torch_module_from_dsl_and_exports(tmp_path):
    torch = pytest.importorskip("torch")
    from pycograd import frozen, params
    from pycograd.export import export_torchscript

    r = _rng(1)
    weights = params(
        ew1=0.3 * r.standard_normal((2, 16)),
        eb1=np.zeros(16),
        ew2=0.3 * r.standard_normal((16, 3)),
        eb2=frozen(np.zeros(3)),  # a frozen leaf -> a non-trainable buffer
    )
    X = _rng(0).standard_normal((20, 2))
    xb = torch.as_tensor(X)
    path = str(tmp_path / "dsl_mlp.pt")
    # Build, run, and trace within `with weights:` (forward reads the injected proxies).
    with weights:
        module = weights.to_torch_module(_dsl_mlp_forward)
        ref = _dsl_mlp_forward(X)  # numpy forward (ambient weights resolve to numpy)
        out = module(xb)
        assert np.allclose(out.detach().numpy(), ref, atol=1e-9)
        # frozen leaf is a buffer, not a Parameter: 3 trainable tensors, not 4
        assert sum(1 for _ in module.parameters()) == 3
        out.sum().backward()  # a real, trainable nn.Module
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
        export_torchscript(module, (xb,), path)
    # The exported TorchScript carries the captured graph -- runs with no pycograd in scope.
    assert np.allclose(
        torch.jit.load(path)(xb).detach().numpy(), out.detach().numpy(), atol=1e-6
    )


def test_to_torch_module_forward_and_trains():
    torch = pytest.importorskip("torch")
    from pycograd.export import to_torch_module

    params = M._init_mlp_tree(_rng(1))
    xb = torch.as_tensor(np.asarray(M._Xc, np.float64))
    yb = torch.as_tensor(np.asarray(M._Yoh, np.float64))

    module = to_torch_module(M.mlp_batch_loss, params)
    out = module(xb, yb)
    assert np.isclose(
        float(out.detach()), float(M.mlp_batch_loss(params, M._Xc, M._Yoh))
    )

    out.backward()  # the wrapped net is a real, trainable nn.Module
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
    )


def test_export_torchscript_runs_standalone(tmp_path):
    torch = pytest.importorskip("torch")
    from pycograd.export import export_torchscript, to_torch_module

    params = M._init_mlp_tree(_rng(1))
    xb = torch.as_tensor(np.asarray(M._Xc, np.float64))
    yb = torch.as_tensor(np.asarray(M._Yoh, np.float64))
    module = to_torch_module(M.mlp_batch_loss, params)
    ref = float(module(xb, yb).detach())

    path = str(tmp_path / "mlp_loss.pt")
    export_torchscript(module, (xb, yb), path)
    reloaded = torch.jit.load(path)  # no pycograd needed to run a saved graph
    assert np.isclose(float(reloaded(xb, yb).detach()), ref)


def test_export_onnx_matches_under_onnxruntime(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from pycograd.export import export_onnx, to_torch_module

    params = M._init_mlp_tree(_rng(1))
    xb = torch.as_tensor(np.asarray(M._Xc, np.float64))
    yb = torch.as_tensor(np.asarray(M._Yoh, np.float64))
    module = to_torch_module(M.mlp_batch_loss, params)
    ref = float(module(xb, yb).detach())

    path = str(tmp_path / "mlp_loss.onnx")
    export_onnx(module, (xb, yb), path, input_names=["xb", "yb"], opset_version=17)
    sess = ort.InferenceSession(path)
    out = sess.run(None, {"xb": xb.numpy(), "yb": yb.numpy()})[0]
    assert np.isclose(float(np.asarray(out)), ref, atol=1e-6)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_one_gradient_step_decreases_loss(backend):
    pytest.importorskip(_FRAMEWORK_MODULE[backend])
    params = M._init_mlp_tree(_rng(1))
    vg = C.value_and_grad(M.mlp_batch_loss, backend=backend)
    loss0, (g, _gx, _gy) = vg(params, M._Xc, M._Yoh)
    lr = 0.5

    def step(p, gp):
        return {
            k: {kk: p[k][kk] - lr * np.asarray(gp[k][kk]) for kk in p[k]} for k in p
        }

    params = step(params, g)
    loss1, _ = vg(params, M._Xc, M._Yoh)
    assert loss1 < loss0

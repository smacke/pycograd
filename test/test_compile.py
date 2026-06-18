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
from pycograd import frozen
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


# (id, loss_fn, args_factory, backends that support it)
_CASES = [
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

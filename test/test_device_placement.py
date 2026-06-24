# -*- coding: utf-8 -*-
"""Per-leaf device placement: part-CPU / part-GPU in one autograd pass.

``on_cpu(...)`` / ``on_device(...)`` pin a leaf's *home device* on the torch/mps backend,
so a large tensor (an embedding table) can stay on CPU while the rest of the net runs on
the GPU -- the gather happens on CPU and only the small slice crosses, auto-unified by the
backend. The pure-API tests below run anywhere; the offload tests skip unless an MPS device
is present (mirroring ``test_mps.py``), at a float32 tolerance.
"""
import numpy as np
import pytest

import pycograd.compile as C
from pycograd import buffer, frozen, on_cpu, on_device, params, tied
from pycograd.params import Param

_ATOL, _RTOL = 1e-4, 1e-3


def _rng(seed):
    return np.random.default_rng(seed)


# --- pure-API tests (no GPU needed) ----------------------------------------------------
def test_on_cpu_makes_trainable_param_on_device():
    p = on_cpu(np.ones(3))
    assert isinstance(p, Param) and p.trainable and p.device == "cpu"
    assert on_device("mps", np.ones(3)).device == "mps"
    assert on_device("mps")[np.ones(3)].device == "mps"  # subscript form


def test_on_cpu_composes_with_frozen_buffer_preserving_flags():
    fr = on_cpu(frozen(np.ones(3)))
    assert fr.device == "cpu" and not fr.trainable and not fr.mutable
    bf = on_cpu(buffer(np.ones(3)))
    assert bf.device == "cpu" and not bf.trainable and bf.mutable


def test_on_device_rejects_tied_ref():
    with pytest.raises(ValueError, match="tied"):
        on_cpu(tied[np.ones(3)])  # tied[...] is a _TieRef, not a value


def test_tied_leaves_must_share_a_device():
    w0 = np.ones(3)
    model = {"a": on_cpu(tied("k", w0)), "b": tied("k", w0)}  # cpu vs default

    def loss(p):
        return np.sum(p["a"] + p["b"])

    with pytest.raises(ValueError, match="share a device"):
        C.value_and_grad(loss, backend="numpy")(model)


def test_device_tag_rejected_on_non_torch_backend():
    def loss(p):
        return np.sum(p["w"])

    with pytest.raises(ValueError, match="per-leaf device"):
        C.value_and_grad(loss, backend="numpy")({"w": on_cpu(np.ones(3))})


def test_device_tag_is_a_noop_on_the_numpy_tape():
    # No delegate backend: the home device is ignored, the leaf trains normally.
    weights = params(w=on_cpu(np.ones(3)))

    def objective():
        return np.sum(w * 2.0)  # noqa: F821  (ambient weight)

    with weights:
        _, g = weights.grad(objective)
    assert np.allclose(np.asarray(g["w"]), 2.0)


def test_torch_coerce_operand_preserves_integer_index_dtype():
    """A delegate backend must lift an integer array (an index/label) keeping its integer
    dtype, not cast it to the working float dtype -- otherwise a piped/binop-arg index
    becomes a float tensor and ``table[idx]`` raises. Runs on CPU torch (no MPS needed).
    """
    _torch = pytest.importorskip("torch")
    from pycograd.backends.torch_backend import TorchBackend

    be = TorchBackend()  # CPU torch, still a delegate backend
    idx = np.array([0, 3, 7, 2], dtype=np.int64)
    t = be.coerce_operand(idx)
    assert isinstance(t, _torch.Tensor) and not t.dtype.is_floating_point
    # The lifted index can actually gather a table (the original failure was an IndexError).
    table = be.lift(np.arange(40.0).reshape(10, 4))
    assert table[t].shape == (4, 4)
    # A float operand is still promoted to the working float dtype (unchanged behavior).
    assert be.coerce_operand(np.ones(3)).dtype.is_floating_point


# --- MPS offload tests (skip off-Metal) ------------------------------------------------
torch = pytest.importorskip("torch")
if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
    pytest.skip("MPS device is not available on this host", allow_module_level=True)

import pycograd.backends.torch_backend as _TB  # noqa: E402

_IDX = np.array([3, 7, 7, 1, 20])
_Y = np.eye(4)[[0, 1, 2, 0, 1]]


def _relu(z):
    return np.maximum(0.0, z)


def _softmax_ce(logits, onehot):
    z = logits - np.max(logits, axis=1, keepdims=True)
    logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    return -np.mean(np.sum(onehot * logp, axis=1))


# Module-level ambient forward (globals so the instrumented runner sees the weights).
def _forward(idx):
    rows = embed[idx]  # noqa: F821  gather on CPU (embed is home-CPU)
    return _relu(rows @ ew1 + eb1) @ ew2 + eb2  # noqa: F821  matmuls on the GPU


def _make_weights(*, frozen_table=False):
    table = 0.1 * _rng(0).standard_normal((50, 8))
    return params(
        embed=on_cpu(frozen(table) if frozen_table else table),  # CPU-resident
        ew1=0.3 * _rng(1).standard_normal((8, 16)),
        eb1=np.zeros(16),
        ew2=0.3 * _rng(2).standard_normal((16, 4)),
        eb2=np.zeros(4),
    )


@pytest.mark.parametrize("jit", [False, True])
def test_embedding_offload_parity(jit):
    """A CPU embedding + GPU matmuls: forward and every per-leaf gradient match the numpy
    tape (to float32 tol), validating cross-device autodiff end-to-end (eager and jit).
    """
    weights = _make_weights()

    def objective():
        return _softmax_ce(_forward(_IDX), _Y)

    with weights:
        v_np, g_np = weights.grad(objective)
        v_mps, g_mps = weights.grad(objective, backend="mps", jit=jit)
    assert np.allclose(v_np, v_mps, atol=_ATOL, rtol=_RTOL)
    for k in weights:
        assert np.allclose(
            np.asarray(g_np[k]), np.asarray(g_mps[k]), atol=_ATOL, rtol=_RTOL
        ), k


def test_cpu_table_grad_and_leaf_stay_on_cpu():
    """The CPU-home table is lifted on CPU and its gradient comes back on CPU, while the
    GPU weights' tensors/grads live on mps -- so the big table never crosses to the GPU.
    """
    weights = _make_weights()

    def objective():
        return _softmax_ce(_forward(_IDX), _Y)

    seen: list[tuple] = []
    orig = _TB._torch_to_numpy

    def spy(torch_mod, t):
        if isinstance(t, torch_mod.Tensor):
            seen.append((tuple(t.shape), t.device.type))
        return orig(torch_mod, t)

    _TB._torch_to_numpy = spy
    try:
        with weights:
            weights.grad(objective, backend="mps")
    finally:
        _TB._torch_to_numpy = orig

    table_devs = {dev for shp, dev in seen if shp == (50, 8)}
    w_devs = {dev for shp, dev in seen if shp == (8, 16)}
    assert table_devs == {"cpu"}, table_devs  # embed grad on CPU
    assert w_devs == {"mps"}, w_devs  # ew1 grad on GPU


def test_frozen_table_on_cpu_has_no_gradient():
    weights = _make_weights(frozen_table=True)

    def objective():
        return _softmax_ce(_forward(_IDX), _Y)

    with weights:
        v_mps, g = weights.grad(objective, backend="mps")
        v_np, _ = weights.grad(objective)
    assert g["embed"] is None and g["ew1"] is not None
    assert np.allclose(v_np, v_mps, atol=_ATOL, rtol=_RTOL)


def test_gather_with_a_gpu_index_aligns_to_the_table():
    """A pre-lifted (GPU) integer index gathering a CPU table is auto-aligned onto the
    table's device by ``align_key`` -- exercises that seam directly."""
    weights = _make_weights()
    idx_gpu = torch.tensor(_IDX, device="mps")  # index already on the GPU

    def objective():
        return _softmax_ce(_forward(idx_gpu), _Y)

    def objective_host():
        return _softmax_ce(_forward(_IDX), _Y)

    with weights:
        v_gpu_idx, _ = weights.grad(objective, backend="mps")
        v_host_idx, _ = weights.grad(objective_host, backend="mps")
    assert np.allclose(v_gpu_idx, v_host_idx, atol=_ATOL, rtol=_RTOL)


def test_jit_recompiles_when_a_leaf_changes_device():
    weights = _make_weights()

    def objective():
        return _softmax_ce(_forward(_IDX), _Y)

    with weights:
        weights.grad(objective, backend="mps", jit=True)
        weights["embed"].device = None  # move the table onto the compute device
        weights.grad(objective, backend="mps", jit=True)
    # the per-leaf device is part of the jit cache key, so the two configs are distinct
    assert len(weights._compiled_grad) == 2


def test_offload_training_decreases_loss():
    weights = _make_weights()

    def objective():
        return _softmax_ce(_forward(_IDX), _Y)

    with weights:
        first = float(weights.grad(objective, backend="mps", jit=True)[0])
        for _ in range(100):
            _, g = weights.grad(objective, backend="mps", jit=True)
            weights.step(g, 0.5)
        last = float(weights.grad(objective, backend="mps")[0])
    assert last < first

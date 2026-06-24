# -*- coding: utf-8 -*-
"""The ``mps`` backend: PyTorch autodiff on Apple Silicon's Metal GPU, vs the numpy tape.

A device twin of ``test_compile``: MPS is just a torch *device*, so the same
finite-difference-checked example models that validate the torch backend validate this
one -- only at a float32 tolerance, since MPS has no float64 and this backend computes
the float64 default in float32 (see ``pycograd/backends/mps_backend.py``).

The whole module skips unless torch is installed *and* ``torch.backends.mps`` reports an
available device, so CI on non-Mac hosts stays green.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
    pytest.skip("MPS device is not available on this host", allow_module_level=True)

import pycograd.compile as C  # noqa: E402
import pycograd.transforms as T  # noqa: E402
from pycograd import frozen, params  # noqa: E402
from pycograd.backends import get_backend  # noqa: E402
from pycograd.backends.mps_backend import MpsBackend  # noqa: E402
from pycograd.dtypes import dtype  # noqa: E402
from pycograd.examples import models as M  # noqa: E402
from pycograd.tree import tree_leaves  # noqa: E402

# MPS computes the float64 default in float32, so parity with the numpy tape holds only to
# single precision; these mirror the float32 tolerances used elsewhere in the suite.
_ATOL, _RTOL = 1e-4, 1e-3


def _rng(seed):
    return np.random.default_rng(seed)


def _flat_grads(grads):
    return [np.asarray(g) for arg in grads for g in tree_leaves(arg) if g is not None]


def _reseed():
    M._dropout_rng = np.random.default_rng(1234)


# (id, loss_fn, args_factory) -- fully-instrumented, finite-diff-checked nets reused from
# ``test_compile`` (rank-2 matmul models that lower cleanly onto a delegate backend).
_CASES = [
    ("mlp_tree", M.mlp_tree_loss, lambda: (M._init_mlp_tree(_rng(1)),)),
    ("mlp_batch", M.mlp_batch_loss, lambda: (M._init_mlp_tree(_rng(1)), M._Xc, M._Yoh)),
    ("deep_ln_dropout", M.deep_loss, lambda: M._init_deep(_rng(2))),
    (
        "rnn",
        M.rnn_loss,
        lambda: (M._init_rnn(_rng(3), vocab=len(M._CHAR_VOCAB), d_model=8),),
    ),
]


@pytest.mark.parametrize("cid,fn,argf", _CASES, ids=[c[0] for c in _CASES])
def test_forward_and_grad_parity(cid, fn, argf):
    """Forward + gradient on MPS must match pycograd's numpy tape (to float32 tol).

    Since the tape is finite-difference-checked, this validates the MPS path end-to-end.
    """
    _reseed()
    ref_v, ref_g = T.value_and_grad(fn)(*argf())
    _reseed()
    cmp_v, cmp_g = C.value_and_grad(fn, backend="mps")(*argf())

    assert np.allclose(ref_v, cmp_v, atol=_ATOL, rtol=_RTOL), f"{cid} forward"
    rf, cf = _flat_grads(ref_g), _flat_grads(cmp_g)
    assert rf and len(rf) == len(cf)
    for a, b in zip(rf, cf):
        assert a.shape == b.shape
        assert np.allclose(a, b, atol=_ATOL, rtol=_RTOL), f"{cid} grad"


def test_frozen_leaf_has_no_gradient():
    p = {
        "hidden": {"w": 0.1 * _rng(1).standard_normal((2, 16)), "b": np.zeros(16)},
        "out": {"w": frozen(0.1 * _rng(2).standard_normal((16, 3))), "b": np.zeros(3)},
    }
    _, (g,) = C.value_and_grad(M.mlp_tree_loss, backend="mps")(p)
    assert g["out"]["w"] is None  # frozen -> no gradient
    assert g["hidden"]["w"] is not None and g["out"]["b"] is not None


def test_compile_to_returns_mps_tensor():
    p = M._init_mlp_tree(_rng(1))
    be = get_backend("mps")
    forward = C.compile_to(M.mlp_tree_loss, "mps")
    tensor_params = {
        k: {kk: be.const(vv) for kk, vv in v.items()} for k, v in p.items()
    }
    out = forward(tensor_params)
    assert isinstance(out, torch.Tensor) and out.device.type == "mps"
    assert np.isclose(
        float(np.asarray(be.to_numpy(out))), float(M.mlp_tree_loss(p)), atol=_ATOL
    )


def test_mps_runs_in_float32_under_float64_default():
    """The float64 default must be computed in float32 (MPS has no float64), so the
    leaves the backend lifts -- and the gradients it returns -- are single precision."""
    be = get_backend("mps")
    t = be.const(np.ones(3))  # no explicit dtype block -> float64 default
    assert t.dtype == torch.float32 and t.device.type == "mps"
    # An explicit float16 block is honored as-is (MPS supports it).
    with dtype("float16"):
        assert be.const(np.ones(3)).dtype == torch.float16


def test_backend_names_resolve_to_mps():
    assert isinstance(get_backend("mps"), MpsBackend)
    assert isinstance(get_backend("metal"), MpsBackend)  # friendly alias
    assert get_backend("mps").is_delegate and get_backend("mps").name == "mps"


# An ambient-DSL forward (the notebook's surface): reads weights injected by `with
# weights:`. Module-level so the pyccolo-instrumented runner sees them as globals.
def _ambient_forward(x):
    return _relu(x @ ew1 + eb1) @ ew2 + eb2  # noqa: F821  (ambient weights)


def _relu(z):
    return np.maximum(0.0, z)


def _softmax_ce(logits, onehot):
    z = logits - np.max(logits, axis=1, keepdims=True)
    logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    return -np.mean(np.sum(onehot * logp, axis=1))


_X = np.array([[0.5, -1.0], [2.0, 0.3], [1.0, 1.0], [-0.7, 0.2]])
_Y = np.eye(3)[[0, 1, 2, 0]]


def test_ambient_grad_jit_trains():
    """The demo's path: ``weights.grad(objective, backend='mps', jit=True)`` agrees with
    the numpy tape, frozen leaves report ``None``, and SGD steps drive the loss down --
    exercising the compiled-gradient (``make_fx`` / ``torch.compile``) reuse on MPS."""
    weights = params(
        ew1=0.3 * _rng(2).standard_normal((2, 16)),
        eb1=np.zeros(16),
        ew2=0.3 * _rng(3).standard_normal((16, 3)),
        eb2=frozen(np.zeros(3)),  # held fixed -> gradient None
    )

    def objective():
        return _softmax_ce(_ambient_forward(_X), _Y)

    with weights:
        v_np, g_np = weights.grad(objective)
        v_mps, g_mps = weights.grad(objective, backend="mps", jit=True)
        assert np.allclose(v_np, v_mps, atol=_ATOL, rtol=_RTOL)
        for k in weights:
            if g_np[k] is not None:
                assert np.allclose(
                    np.asarray(g_np[k]), np.asarray(g_mps[k]), atol=_ATOL, rtol=_RTOL
                )
        assert g_mps["eb2"] is None  # frozen leaf

        first = float(v_mps)
        for _ in range(100):
            value, grads = weights.grad(objective, backend="mps", jit=True)
            weights.step(grads, 0.5)
        last = float(weights.grad(objective, backend="mps")[0])
    assert last < first

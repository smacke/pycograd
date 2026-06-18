# -*- coding: utf-8 -*-
"""The array-backend / device seam.

Two halves:

* **CPU, no GPU** -- a spy *tape* backend (a ``NumpyBackend`` whose ``array_module`` and
  ``scatter_add`` are numpy-backed but instrumented) proves the indirection is real: the
  primitives, the tape, and the optimizers fetch their array ops from the *active*
  backend rather than a hardcoded ``np``, and the results are identical to the plain
  numpy path. This is the regression guard that the seam is wired, runnable anywhere.
* **GPU, gated** -- ``pytest.importorskip("cupy")`` runs the real cupy backend and checks
  its gradients against the numpy path / a finite-difference oracle, and that a short
  Adam training loop agrees.
"""
import pytest

np = pytest.importorskip("numpy")

import pycograd as pg  # noqa: E402
from pycograd import Adam, value_and_grad  # noqa: E402
from pycograd.backends.numpy_backend import NumpyBackend  # noqa: E402

# Module-level data so target functions reference globals (not closures), keeping them
# recompilable by the tracer (getsource + recompile), per the suite's convention.
_W = np.array([[0.5, -0.3, 0.2], [0.1, 0.4, -0.6]])  # (2, 3)
_IDX = np.array([0, 2, 2, 1])  # repeated indices -> scatter-add must accumulate
_X = np.array([0.7, -0.4])  # (2,)


def seam_fn(x):
    # matmul + transcendentals (forward & backward use xp) + gather-with-repeats
    # (scatter-add backward) + a reduction -- a broad slice of the primitive set.
    z = x @ _W
    h = np.tanh(z) + np.exp(z)
    g = h[_IDX]
    return np.sum(g * g)


# ---------------------------------------------------------------------------
# A spy tape backend: numpy underneath, but it records every array-module access
# and every scatter-add, so we can assert the seam actually routed through it.
# ---------------------------------------------------------------------------
class _SpyModule:
    """Forwards every attribute to numpy, recording the name first."""

    def __init__(self):
        self.accessed = set()

    def __getattr__(self, name):
        self.accessed.add(name)
        return getattr(np, name)


class SpyBackend(NumpyBackend):
    name = "spy"

    def __init__(self):
        self.array_module = (
            _SpyModule()
        )  # instance attr shadows NumpyBackend.array_module = np
        self.scatter_calls = 0

    def scatter_add(self, out, key, vals):
        self.scatter_calls += 1
        np.add.at(out, key, vals)


def test_seam_dispatches_through_active_backend():
    # Baseline on the default numpy path.
    v0, (g0,) = value_and_grad(seam_fn)(_X)

    spy = SpyBackend()
    with pg.device(spy):
        v1, (g1,) = value_and_grad(seam_fn)(_X)

    # 1. Same answers -- the seam changed *where* ops come from, not *what* they compute.
    assert np.allclose(v0, v1)
    assert np.allclose(g0, g1)

    # 2. The indirection is real: ops/tape fetched their array ops from the spy module
    #    (if anything were still hardcoded ``np.`` these would be missing), ...
    accessed = spy.array_module.accessed
    assert {
        "asarray",
        "zeros_like",
        "ones_like",
    } <= accessed  # tensor.py (Var/backward)
    assert {"tanh", "exp"} <= accessed  # ops.py forward primals
    assert "broadcast_to" in accessed  # ops.py d_sum backward
    # 3. ... and the indexing VJP went through the backend's scatter-add seam.
    assert spy.scatter_calls > 0


def test_optimizer_state_routes_through_seam():
    params = {"w": np.array([1.0, -2.0, 0.5])}
    grads = {"w": np.array([0.1, 0.2, -0.3])}

    base = Adam(lr=0.05).step(params, grads)

    spy = SpyBackend()
    with pg.device(spy):
        stepped = Adam(lr=0.05).step(params, grads)

    assert np.allclose(base["w"], stepped["w"])
    # Adam's moment buffers + bias-corrected update must come from the active module.
    assert {"zeros_like", "sqrt"} <= spy.array_module.accessed


# ---------------------------------------------------------------------------
# Real cupy backend -- only where a CUDA GPU + cupy are present.
# ---------------------------------------------------------------------------
def _finite_diff_grad(f, x, h=1e-5):
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        up, dn = x.copy(), x.copy()
        up[idx] += h
        dn[idx] -= h
        g[idx] = (float(f(up)) - float(f(dn))) / (2 * h)
    return g


def test_cupy_matches_numpy_and_finite_diff():
    cupy = pytest.importorskip("cupy")

    v_np, (g_np,) = value_and_grad(seam_fn)(_X)
    with pg.device("cupy"):
        v_cp, (g_cp,) = value_and_grad(seam_fn)(_X)

    # value_and_grad on cupy returns device arrays; bring them home to compare.
    assert isinstance(g_cp, cupy.ndarray)
    assert np.allclose(cupy.asnumpy(v_cp), v_np)
    assert np.allclose(cupy.asnumpy(g_cp), g_np)
    # and against an independent finite-difference oracle (host).
    assert np.allclose(cupy.asnumpy(g_cp), _finite_diff_grad(seam_fn, _X), atol=1e-4)


def test_cupy_training_loop_matches_numpy():
    cupy = pytest.importorskip("cupy")

    def run(device_name):
        p = {"w": np.array([0.3, -0.2])}
        opt = Adam(lr=0.1)
        with pg.device(device_name):
            for _ in range(20):
                _loss, (g,) = value_and_grad(seam_fn)(p["w"])
                p = opt.step(p, {"w": g})
        w = p["w"]
        return cupy.asnumpy(w) if device_name == "cupy" else w

    assert np.allclose(run("cupy"), run("numpy"), atol=1e-5)

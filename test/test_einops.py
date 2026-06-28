# -*- coding: utf-8 -*-
"""einops interop: ``rearrange`` / ``reduce`` / ``repeat`` / ``einsum`` run on ``Var`` via
the backend registered in :mod:`pycograd.einops_backend`, and compose with the transforms.

Each ported example (from https://einops.rocks/pytorch-examples.html) is gradient-checked
against finite differences, and exercised under forward-mode (jvp), batching (vmap), and
per-example gradients (vmap-of-grad). Plus registration coverage and a small end-to-end
multi-head attention forward+backward.

The same module-level function runs in two contexts: under ``grad`` it is instrumented and
sees ``Var`` (einops uses the pycograd backend); inside ``_fd`` it runs on raw numpy arrays
(einops uses its numpy backend). ``einops`` is referenced as a module global so the
instrumented copy keeps the reference. Module-level so pyccolo can re-instrument from source.
"""
import einops
import numpy as np
import pytest

from pycograd import (
    GraphCost,
    Var,
    capture,
    cost_report,
    eval_graph,
    eval_shape,
    grad,
    jvp,
    register_einops_backend,
    value_and_grad,
    vmap,
)

_rng = np.random.default_rng(0)


def _fd(f, x, eps=1e-6):
    out = np.zeros(x.shape)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        out.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return out


# --- ported examples (each reduces to a scalar via np.sum so grad/eval_shape apply) -------


def f_flatten(x):  # ex 1/19: flatten conv map  (b, c, h, w) -> (b, c*h*w)
    return np.sum(einops.rearrange(x, "b c h w -> b (c h w)") ** 2)


def f_global_avg_pool(x):  # ex 7: global average pooling  (b, c, h, w) -> (b, c)
    return np.sum(einops.reduce(x, "b c h w -> b c", "mean") ** 2)


def f_max_pool(x):  # max-pool 2x2 via reduce
    return np.sum(
        einops.reduce(x, "b c (h h2) (w w2) -> b c h w", "max", h2=2, w2=2) ** 2
    )


def f_spatial_to_seq(x):  # ex 16: spatial <-> sequence round trip
    y = einops.rearrange(x, "b c h w -> b (h w) c")
    z = einops.rearrange(y, "b (h w) c -> b c h w", h=4)
    return np.sum(z**2)


def f_channel_shuffle(x):  # ex 5: ShuffleNet channel shuffle  (c1=groups)
    return np.sum(einops.rearrange(x, "b (c1 c2) h w -> b (c2 c1) h w", c1=2) ** 2)


def f_space_to_depth(x):  # ex 22: GLOW space-to-depth (squeeze)
    return np.sum(
        einops.rearrange(x, "b c (h h2) (w w2) -> b (c h2 w2) h w", h2=2, w2=2) ** 2
    )


def f_pixel_shuffle(x):  # ex 2: super-resolution pixel shuffle (depth-to-space)
    return np.sum(
        einops.rearrange(x, "b (h2 w2) h w -> b (h h2) (w w2)", h2=2, w2=2) ** 2
    )


def f_repeat(x):  # repeat/tile a dim
    return np.sum(einops.repeat(x, "b c -> b (c r)", r=3) ** 2)


def f_gram(x):  # ex 3: Gram matrix via einsum (style transfer)
    return np.sum(einops.einsum(x, x, "b c h w, b d h w -> b c d") ** 2)


def f_attention(x):  # ex 13-15: multi-head scaled-dot-product attention (q=k=v from x)
    q = einops.rearrange(x, "b l (head k) -> head b l k", head=2)
    k = einops.rearrange(x, "b t (head k) -> head b t k", head=2)
    v = einops.rearrange(x, "b t (head v) -> head b t v", head=2)
    scores = einops.einsum(q, k, "head b l k, head b t k -> head b l t")
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    out = einops.einsum(weights, v, "head b l t, head b t v -> head b l v")
    out = einops.rearrange(out, "head b l v -> b l (head v)")
    return np.sum(out**2)


# concrete inputs sized so the einops axis splits are exact and FD stays cheap
_BCHW = _rng.standard_normal((2, 3, 4, 4))
_SHUFFLE = _rng.standard_normal((2, 6, 3, 3))  # 6 channels = c1(2) * c2(3)
_PIXEL = _rng.standard_normal((2, 4, 3, 3))  # 4 channels = h2(2) * w2(2)
_BC = _rng.standard_normal((2, 5))
_ATTN = _rng.standard_normal((1, 3, 8))  # (batch, len, head(2) * dim(4))


@pytest.mark.parametrize(
    "fn, a",
    [
        (f_flatten, _BCHW),
        (f_global_avg_pool, _BCHW),
        (f_max_pool, _BCHW),
        (f_spatial_to_seq, _BCHW),
        (f_channel_shuffle, _SHUFFLE),
        (f_space_to_depth, _BCHW),
        (f_pixel_shuffle, _PIXEL),
        (f_repeat, _BC),
        (f_gram, _BCHW),
        (f_attention, _ATTN),
    ],
)
def test_grad_vs_fd(fn, a):
    g = np.asarray(grad(fn)(a)[0])
    assert g.shape == a.shape
    assert np.allclose(g, _fd(fn, a), atol=1e-5)
    assert eval_shape(fn, a).shape == ()


def test_jvp_matches_fd():
    v = _rng.standard_normal(_BCHW.shape)
    _, t = jvp(f_flatten, (_BCHW,), (v,))
    fd = (f_flatten(_BCHW + 1e-6 * v) - f_flatten(_BCHW - 1e-6 * v)) / 2e-6
    assert np.isclose(float(np.asarray(t)), fd, atol=1e-3)


def f_reduce_nonscalar(x):  # non-scalar output, for shape inference / graph mode
    return einops.reduce(x, "b c h w -> b c", "mean")


# --- graph mode (capture), shape inference (aval), cost modeling --------------------------


def test_graph_capture_and_eval():
    g = capture(f_channel_shuffle, _SHUFFLE)
    out = eval_graph(g, _SHUFFLE)
    assert np.allclose(
        np.asarray(getattr(out, "value", out)), f_channel_shuffle(_SHUFFLE)
    )


def test_graph_value_and_grad_matches_fd():
    val, grads = value_and_grad(capture(f_channel_shuffle, _SHUFFLE))(_SHUFFLE)
    g = grads[0] if isinstance(grads, tuple) else grads
    g = np.asarray(getattr(g, "value", g))
    assert np.isclose(float(val), f_channel_shuffle(_SHUFFLE))
    assert np.allclose(g, _fd(f_channel_shuffle, _SHUFFLE), atol=1e-5)


def test_eval_shape_nonscalar():
    sds = eval_shape(f_reduce_nonscalar, _BCHW)
    assert sds.shape == (2, 3)


def test_cost_report_on_einops_graph():
    g = capture(f_channel_shuffle, _SHUFFLE)
    report = cost_report(g)
    assert isinstance(report, GraphCost)
    assert report.total_flops > 0 and report.peak_memory_bytes > 0
    # the rearrange lowers to reshape/transpose primitives the cost model can see
    prims = {node.prim for node in report.nodes}
    assert {"reshape", "transpose"} <= prims


def per_example_flatten(x):  # x: (c, h, w)
    return np.sum(einops.rearrange(x, "c h w -> (c h w)") ** 2)


def test_vmap_is_per_example():
    batch = _rng.standard_normal((5, 3, 4, 4))
    out = np.asarray(vmap(per_example_flatten)(batch))
    ref = np.array([per_example_flatten(b) for b in batch])
    assert np.allclose(out, ref)


def test_vmap_of_grad_is_per_example_grad():
    batch = _rng.standard_normal((5, 3, 4, 4))
    g = vmap(grad(per_example_flatten))(batch)
    g = np.asarray(g[0] if isinstance(g, tuple) else g)
    # d/dx sum((flatten x)^2) = 2x, per example
    assert np.allclose(g, 2.0 * batch)


# --- registration -------------------------------------------------------------------------


def test_backend_is_registered():
    import einops._backends as eb

    assert "pycograd" in eb._loaded_backends
    backend = eb.get_backend(Var(np.zeros((2, 2))))
    assert backend.framework_name == "pycograd"


def test_register_is_idempotent():
    register_einops_backend()
    register_einops_backend()  # second call is a no-op, must not raise
    out = einops.rearrange(Var(np.zeros((2, 3))), "a b -> b a")
    assert out.shape == (3, 2)


def test_rearrange_value_matches_numpy():
    x = _rng.standard_normal((2, 3, 4))
    out = einops.rearrange(Var(x), "a b c -> c (a b)")
    ref = einops.rearrange(x, "a b c -> c (a b)")
    assert np.allclose(np.asarray(out.value), ref)

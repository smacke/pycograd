# -*- coding: utf-8 -*-
"""Tests for gradient checkpointing (``pycograd.checkpoint``).

``checkpoint(f)`` must be *gradient-transparent*: the gradients it produces match the
un-checkpointed ``f`` (and finite differences) exactly -- only peak memory differs, because
the segment's activations are recomputed in backward rather than retained. We check that
transparency across the entry points (positional ``value_and_grad`` and ambient
``weights.grad``), pytree outputs, nesting, that rematerialization actually re-runs the
segment, and the supported higher-order / ``vmap`` compositions. The unsupported
reverse-over-reverse corner must fail with a clear error, not a crash.
"""
import pytest

np = pytest.importorskip("numpy")

from pycograd import (  # noqa: E402
    checkpoint,
    grad,
    jacfwd,
    jvp,
    params,
    value_and_grad,
    vmap,
)

rng = np.random.default_rng(0)
Wa = rng.standard_normal((4, 5))
Wb = rng.standard_normal((5, 3))


# --- finite-difference oracle (same shape as test_autodiff) -----------------
def finite_diff(f, args, h=1e-6):
    def s(*a):
        return float(np.sum(f(*a)))

    base = [np.array(a, dtype=float) for a in args]
    grads = []
    for i, a in enumerate(base):
        g = np.zeros_like(a)
        for idx in np.ndindex(a.shape):
            up = [x.copy() for x in base]
            dn = [x.copy() for x in base]
            up[i][idx] += h
            dn[i][idx] -= h
            g[idx] = (s(*up) - s(*dn)) / (2 * h)
        grads.append(g)
    return tuple(grads)


# --- module-level segments (so pyccolo can instrument from source) ----------
def block(x):
    h = np.tanh(x @ Wa)
    return np.tanh(h @ Wb)


def block_pytree(x):
    h = np.tanh(x @ Wa)
    return {"a": h, "b": np.exp(h @ Wb) * 0.1}


def loss_plain(x):
    y = block(x)
    return np.sum(y * y)


def loss_ckpt(x):
    y = checkpoint(block)(x)
    return np.sum(y * y)


def loss_pytree_plain(x):
    o = block_pytree(x)
    return np.sum(o["a"] * o["a"]) + np.sum(o["b"])


def loss_pytree_ckpt(x):
    o = checkpoint(block_pytree)(x)
    return np.sum(o["a"] * o["a"]) + np.sum(o["b"])


def loss_nested_ckpt(x):
    y = checkpoint(lambda z: checkpoint(block)(z))(x)
    return np.sum(y * y)


# --- value_and_grad (positional) transparency ------------------------------
def test_checkpoint_matches_plain_and_finite_diff():
    x = rng.standard_normal((2, 4))
    v_p, g_p = value_and_grad(loss_plain)(x)
    v_c, g_c = value_and_grad(loss_ckpt)(x)
    assert np.allclose(v_p, v_c)
    # Byte-for-byte identical to the un-checkpointed gradient.
    assert np.array_equal(g_p[0], g_c[0])
    # And matches finite differences.
    (fd,) = finite_diff(loss_plain, (x,))
    assert np.allclose(g_c[0], fd, atol=1e-4)


def test_checkpoint_pytree_output():
    x = rng.standard_normal((5, 4))
    _, g_p = value_and_grad(loss_pytree_plain)(x)
    _, g_c = value_and_grad(loss_pytree_ckpt)(x)
    # The multi-output slice/concat boundary reorders the backward reduction, so match to
    # floating tolerance rather than bit-for-bit (the single-output path is exact).
    assert np.allclose(g_p[0], g_c[0], atol=1e-12)
    (fd,) = finite_diff(loss_pytree_plain, (x,))
    assert np.allclose(g_c[0], fd, atol=1e-4)


def test_checkpoint_nested():
    x = rng.standard_normal((2, 4))
    _, g_p = value_and_grad(loss_plain)(x)
    _, g_c = value_and_grad(loss_nested_ckpt)(x)
    assert np.allclose(g_p[0], g_c[0], atol=1e-10)


# --- rematerialization actually recomputes the forward ----------------------
_CALLS = {"n": 0}


def counted_block(x):
    _CALLS["n"] += 1
    return np.tanh(x @ Wa) @ Wb


def _loss_counted_ckpt(x):
    return np.sum(checkpoint(counted_block)(x) ** 2)


def _loss_counted_plain(x):
    return np.sum(counted_block(x) ** 2)


def test_checkpoint_rematerializes():
    x = rng.standard_normal((2, 4))
    _CALLS["n"] = 0
    value_and_grad(_loss_counted_ckpt)(x)
    # Once for the forward (capture) + once for the backward remat.
    assert _CALLS["n"] == 2
    _CALLS["n"] = 0
    value_and_grad(_loss_counted_plain)(x)
    assert _CALLS["n"] == 1


# --- ambient weights.grad entry point ---------------------------------------
def amb_block(x):
    return np.tanh(x @ w1) @ w2  # noqa: F821 -- injected by `with model:`


def amb_loss_plain():
    return np.sum(amb_block(X) ** 2)  # noqa: F821


def amb_loss_ckpt():
    return np.sum(checkpoint(amb_block)(X) ** 2)  # noqa: F821


def test_checkpoint_ambient_weights():
    global X
    X = rng.standard_normal((2, 4))
    model = params(w1=rng.standard_normal((4, 5)), w2=rng.standard_normal((5, 3)))
    with model:
        _, g_p = model.grad(amb_loss_plain)
        _, g_c = model.grad(amb_loss_ckpt)
    for k in ("w1", "w2"):
        assert np.any(g_c[k] != 0)  # weight gradients actually flow through the remat
        assert np.array_equal(g_p[k], g_c[k])


# --- transparency under higher-order compositions ---------------------------
# Under a live jvp (HVP / jacfwd-Hessian) the segment's inputs are Tracers, so checkpoint
# passes through (the boundary can't be built at that level) -- gradients must still be
# correct, just without memory savings in that nested case.
def ho_seg(x):
    return np.tanh(x * x + 0.5 * x)


def ho_plain(x):
    return np.sum(ho_seg(x) ** 2)


def ho_ckpt(x):
    return np.sum(checkpoint(ho_seg)(x) ** 2)


def test_checkpoint_hvp_jvp_of_grad():
    x = rng.standard_normal(4)
    v = rng.standard_normal(4)
    _, hvp_p = jvp(grad(ho_plain), (x,), (v,))
    _, hvp_c = jvp(grad(ho_ckpt), (x,), (v,))
    assert np.allclose(np.asarray(hvp_p[0]), np.asarray(hvp_c[0]), atol=1e-10)


def test_checkpoint_hessian_jacfwd_of_grad():
    x = rng.standard_normal(4)
    H_p = np.asarray(jacfwd(grad(ho_plain))(x)[0])
    H_c = np.asarray(jacfwd(grad(ho_ckpt))(x)[0])
    assert np.allclose(H_p, H_c, atol=1e-10)


# --- transparency under vmap ------------------------------------------------
# Same story for vmap: inputs are BatchTracers, checkpoint passes through, gradients
# (full-batch and per-sample) must match the un-checkpointed function.
def vm_seg(x):
    return np.tanh(x @ Wv)  # noqa: F821


def vm_f(x):
    return np.sum(vm_seg(x) ** 2)


def vm_f_ckpt(x):
    return np.sum(checkpoint(vm_seg)(x) ** 2)


def _grad_of_vmap_plain(Xb):
    return np.sum(vmap(vm_f)(Xb))


def _grad_of_vmap_ckpt(Xb):
    return np.sum(vmap(vm_f_ckpt)(Xb))


def test_checkpoint_inside_vmap_then_grad():
    global Wv
    Wv = rng.standard_normal((4, 4))
    X = rng.standard_normal((6, 4))
    _, g_p = value_and_grad(_grad_of_vmap_plain)(X)
    _, g_c = value_and_grad(_grad_of_vmap_ckpt)(X)
    assert np.allclose(g_p[0], g_c[0], atol=1e-10)


def test_checkpoint_per_sample_vmap_of_grad():
    global Wv
    Wv = rng.standard_normal((4, 4))
    X = rng.standard_normal((6, 4))
    G_p = np.asarray(vmap(grad(vm_f))(X)[0])
    G_c = np.asarray(vmap(grad(vm_f_ckpt))(X)[0])
    assert G_c.shape == (6, 4)
    assert np.allclose(G_p, G_c, atol=1e-10)


# --- unsupported corner fails clearly ---------------------------------------
def rr_seg(x):
    return np.tanh(x * x)


def rr_loss(x):
    return np.sum(checkpoint(rr_seg)(x) ** 2)


def rr_gradvec(x):
    return grad(rr_loss)(x)[0]


def test_checkpoint_reverse_over_reverse_raises():
    from pycograd import jacrev

    x = rng.standard_normal(4)
    with pytest.raises(NotImplementedError, match="reverse-over-reverse"):
        jacrev(rr_gradvec)(x)


# --- inference (no grad pass) is a transparent pass-through -----------------
def test_checkpoint_inference_passthrough():
    x = rng.standard_normal((2, 4))
    assert np.allclose(checkpoint(block)(x), block(x))

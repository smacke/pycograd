# -*- coding: utf-8 -*-
"""Composed neural-network ops: numerically-stable softmax family, cross-entropy,
and the common activations.

These are *not* primitives -- each is written purely from ops that already have
autodiff rules (the intercepted ``np.*`` calls), so the reverse pass, ``jvp``,
``vmap`` and ``eval_shape`` all work with no extra rule-table entries, exactly as
``d_mean``/``d_var``/``layer_norm`` already do. They stay array-in/array-out: on a
plain ndarray they compute eagerly; on a ``Var`` (under an instrumented, traced
function) the same ``np.*`` calls route to the differentiable primitives.

Stability: ``log_softmax``/``logsumexp`` subtract ``max(x)`` before ``exp``. The
shift is *not* detached -- log-sum-exp is exactly shift-invariant, so the gradient
of the shift cancels analytically (a ~1e-15 residual), and keeping it un-detached
preserves array purity. ``cross_entropy`` is fused through ``log_softmax`` rather
than ``log(softmax(x) + eps)``, dropping the epsilon fudge.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# NOTE: like ``examples/models.py``, these helpers are recompiled by pyccolo when
# instrumented on demand, which re-evaluates their annotations -- so the ``Axis``
# alias (a value lookup) is fine, but avoid PEP 604 ``X | None`` spellings here.
from pycograd._typing import Axis, Operand, Tensor


def logsumexp(x: Tensor, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """``log(sum(exp(x), axis))`` computed stably via the max-shift identity
    ``m + log(sum(exp(x - m)))`` with ``m = max(x, axis)``."""
    m = np.max(x, axis=axis, keepdims=True)
    lse = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    if keepdims:
        return lse
    # Drop the reduced (size-1) axes without ``np.squeeze`` (not intercepted):
    # summing over a size-1 axis is value-preserving and differentiable.
    return np.sum(lse, axis=axis)


def log_softmax(x: Tensor, axis: Axis = -1) -> Tensor:
    """``log(softmax(x))`` -- stable (no ``exp`` overflow, no ``log(0)``)."""
    m = np.max(x, axis=axis, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def softmax(x: Tensor, axis: Axis = -1) -> Tensor:
    """Stable softmax, ``exp(log_softmax(x))`` -- output sums to 1 along ``axis``."""
    return np.exp(log_softmax(x, axis=axis))


def cross_entropy(logits: Tensor, targets: Tensor, axis: Axis = -1) -> Operand:
    """Mean soft-target cross-entropy *from logits*:
    ``-mean(sum(targets * log_softmax(logits), axis))``.

    ``targets`` is a probability distribution along ``axis`` (e.g. one-hot labels).
    Fusing the log-softmax avoids the ``log(p + eps)`` underflow guard.
    """
    return -np.mean(np.sum(targets * log_softmax(logits, axis=axis), axis=axis))


def relu(x: Tensor) -> Tensor:
    """``max(x, 0)``."""
    return np.maximum(x, 0.0)


def sigmoid(x: Tensor) -> Tensor:
    """Logistic sigmoid ``1 / (1 + exp(-x))`` (array-pure composition)."""
    return 1.0 / (1.0 + np.exp(-x))


def silu(x: Tensor) -> Tensor:
    """SiLU / swish: ``x * sigmoid(x)``."""
    return x * sigmoid(x)


# ``swish`` is the original name for the same activation.
swish = silu


def gelu(x: Tensor) -> Tensor:
    """GELU via the tanh approximation
    ``0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))``."""
    c = 0.7978845608028654  # sqrt(2 / pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def tanh(x: Tensor) -> Tensor:
    """Hyperbolic tangent -- a friendly alias for ``np.tanh`` (the ``d_tanh``
    primitive) so it reads alongside the other activations at a call site."""
    return np.tanh(x)


def leaky_relu(x: Tensor, slope: float = 0.01) -> Tensor:
    """``max(x, 0) + slope * min(x, 0)`` -- ReLU with a small negative slope."""
    return np.maximum(x, 0.0) + slope * np.minimum(x, 0.0)


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Exponential linear unit: ``x`` for ``x > 0`` else ``alpha (exp(x) - 1)``."""
    return np.where(x > 0.0, x, alpha * np.expm1(x))


def softplus(x: Tensor) -> Tensor:
    """``log(1 + exp(x))`` computed stably as ``max(x, 0) + log1p(exp(-|x|))``
    (no ``exp`` overflow for large ``x``)."""
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def mish(x: Tensor) -> Tensor:
    """Mish: ``x * tanh(softplus(x))``."""
    return x * np.tanh(softplus(x))


def hardsigmoid(x: Tensor) -> Tensor:
    """Piecewise-linear sigmoid approximation ``clip(x + 3, 0, 6) / 6``."""
    return np.clip(x + 3.0, 0.0, 6.0) / 6.0


def hardswish(x: Tensor) -> Tensor:
    """``x * hardsigmoid(x)`` -- the piecewise-linear SiLU approximation."""
    return x * hardsigmoid(x)


def softsign(x: Tensor) -> Tensor:
    """``x / (1 + |x|)`` -- a cheaper, polynomially-saturating tanh-like squash."""
    return x / (1.0 + np.abs(x))


def selu(x: Tensor) -> Tensor:
    """Scaled ELU with the self-normalizing constants
    (``alpha = 1.6732632...``, ``scale = 1.0507009...``)."""
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x > 0.0, x, alpha * np.expm1(x))


# ---------------------------------------------------------------------------
# Convolution & pooling -- composed, not fused.
#
# im2col turns a convolution into a gather (the patch extraction, riding
# ``d_getitem``'s scatter-add backward, which correctly accumulates over overlapping
# patches) followed by a single ``einsum``. Because every step is a differentiable
# primitive, the reverse pass / jvp / vmap come for free -- no conv-specific rule.
# Naive (an explicit im2col materialization), per the readability-over-speed goal.
# ---------------------------------------------------------------------------
def _im2col_indices(
    channels: int, kh: int, kw: int, stride: int, h_out: int, w_out: int
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Advanced-index arrays ``(k, i, j)`` selecting, for every (channel, kernel-row,
    kernel-col), the input element feeding each output position. Pure host-side integers
    (a stop-gradient index), so the gather they drive is what carries the gradient."""
    i0 = np.tile(np.repeat(np.arange(kh), kw), channels)
    i1 = stride * np.repeat(np.arange(h_out), w_out)
    j0 = np.tile(np.arange(kw), kh * channels)
    j1 = stride * np.tile(np.arange(w_out), h_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(channels), kh * kw).reshape(-1, 1)
    return k, i, j


def _pad2d(x: Tensor, ph: int, pw: int) -> Tensor:
    """Zero-pad the last two (spatial) axes by ``ph`` / ``pw`` each side, via
    ``concatenate`` (so the pad is differentiable: its gradient is simply dropped)."""
    n, c = x.shape[0], x.shape[1]
    if ph > 0:
        z = np.zeros((n, c, ph, x.shape[3]))
        x = np.concatenate([z, x, z], axis=2)
    if pw > 0:
        z = np.zeros((n, c, x.shape[2], pw))
        x = np.concatenate([z, x, z], axis=3)
    return x


def conv2d(
    x: Tensor, w: Tensor, b: Optional[Tensor] = None, stride: int = 1, pad: int = 0
) -> Tensor:
    """2-D cross-correlation. ``x`` is ``(N, C_in, H, W)``, ``w`` is
    ``(C_out, C_in, kH, kW)``, optional bias ``b`` is ``(C_out,)``; returns
    ``(N, C_out, H_out, W_out)``."""
    c_out, c_in, kh, kw = w.shape[0], w.shape[1], w.shape[2], w.shape[3]
    x = _pad2d(x, pad, pad)
    n, _, h, ww = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    h_out = (h - kh) // stride + 1
    w_out = (ww - kw) // stride + 1
    k, i, j = _im2col_indices(c_in, kh, kw, stride, h_out, w_out)
    cols = x[:, k, i, j]  # (N, C_in*kH*kW, H_out*W_out) -- the im2col gather
    w_col = np.reshape(w, (c_out, c_in * kh * kw))
    out = np.einsum("oc,ncp->nop", w_col, cols)  # (N, C_out, H_out*W_out)
    out = np.reshape(out, (n, c_out, h_out, w_out))
    if b is not None:
        out = out + np.reshape(b, (1, c_out, 1, 1))
    return out


def conv1d(
    x: Tensor, w: Tensor, b: Optional[Tensor] = None, stride: int = 1, pad: int = 0
) -> Tensor:
    """1-D cross-correlation. ``x`` is ``(N, C_in, L)``, ``w`` is ``(C_out, C_in, k)``;
    returns ``(N, C_out, L_out)``. A height-1 ``conv2d`` underneath."""
    c_out, c_in, kk = w.shape[0], w.shape[1], w.shape[2]
    if pad > 0:
        z = np.zeros((x.shape[0], c_in, pad))
        x = np.concatenate([z, x, z], axis=2)
    x4 = np.reshape(x, (x.shape[0], c_in, 1, x.shape[2]))
    w4 = np.reshape(w, (c_out, c_in, 1, kk))
    out = conv2d(x4, w4, b, stride=stride, pad=0)
    return np.reshape(out, (out.shape[0], c_out, out.shape[3]))


def _pool2d(x: Tensor, k: int, stride: Optional[int], reduce_: str) -> Tensor:
    s = k if stride is None else stride
    n, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    h_out = (h - k) // s + 1
    w_out = (w - k) // s + 1
    # Pool each channel independently: fold (N, C) together and im2col a single channel.
    xr = np.reshape(x, (n * c, 1, h, w))
    ki, ii, jj = _im2col_indices(1, k, k, s, h_out, w_out)
    cols = xr[:, ki, ii, jj]  # (N*C, k*k, H_out*W_out)
    pooled = np.max(cols, axis=1) if reduce_ == "max" else np.mean(cols, axis=1)
    return np.reshape(pooled, (n, c, h_out, w_out))


def max_pool2d(x: Tensor, k: int, stride: Optional[int] = None) -> Tensor:
    """Max pooling with a ``k x k`` window (stride defaults to ``k``). The subgradient
    routes through the argmax for free (``np.max``'s rule)."""
    return _pool2d(x, k, stride, "max")


def avg_pool2d(x: Tensor, k: int, stride: Optional[int] = None) -> Tensor:
    """Average pooling with a ``k x k`` window (stride defaults to ``k``)."""
    return _pool2d(x, k, stride, "mean")


def one_hot(indices: "np.ndarray", num_classes: int) -> "np.ndarray":
    """One-hot encode integer ``indices`` along a new last axis. A constant w.r.t. the
    (integer, non-differentiable) indices -- a plain array, not a tape node."""
    return np.eye(num_classes)[np.asarray(indices)]


# ---------------------------------------------------------------------------
# Normalization -- composed from the mean/var reductions, so the reverse pass,
# jvp and vmap come for free (exactly as ``d_mean``/``d_var`` already do).
# ---------------------------------------------------------------------------
def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Normalize over the last axis to zero mean / unit variance, then scale and
    shift: ``(x - mean) / sqrt(var + eps) * gamma + beta``."""
    mu = np.mean(x, axis=-1, keepdims=True)
    centered = x - mu
    var = np.mean(centered * centered, axis=-1, keepdims=True)
    return centered / (var + eps) ** 0.5 * gamma + beta


def rms_norm(x: Tensor, gamma: Tensor, eps: float = 1e-5) -> Tensor:
    """Root-mean-square norm: ``x / sqrt(mean(x**2, -1) + eps) * gamma`` (no mean
    subtraction / bias -- the LLaMA / RWKV-style normalizer)."""
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / (ms + eps) ** 0.5 * gamma


# ---------------------------------------------------------------------------
# Attention & embedding.
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Optional["np.ndarray"] = None
) -> Tensor:
    """Scaled dot-product attention ``softmax(q k^T / sqrt(d)) v`` over the last
    two axes. ``q``/``k``/``v`` are ``(..., L, d)`` with arbitrary leading
    batch/head dims; ``mask`` (optional, broadcastable to the ``(..., Lq, Lk)``
    scores) keeps positions where it is truthy and drives the rest to ~0 weight.

    Written with batched ``matmul`` (which broadcasts over the leading dims) and a
    last-two-axes ``transpose``, so the whole thing batches under ``vmap`` with no
    attention-specific rule."""
    scale = q.shape[-1] ** -0.5
    nd = len(k.shape)
    perm = tuple(range(nd - 2)) + (nd - 1, nd - 2)  # swap the last two axes
    scores = np.matmul(q, np.transpose(k, perm)) * scale
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v)


def embedding(table: Tensor, indices: "np.ndarray") -> Tensor:
    """Look up rows of ``table`` (``(num_embeddings, dim)``) by integer
    ``indices``, returning ``(*indices.shape, dim)``. A plain gather, so the
    gradient scatter-adds back into the looked-up rows (``d_getitem``'s backward).

    Note: for the compile backends (torch / jax / tf), the equivalent
    ``one_hot(indices, num_embeddings) @ table`` avoids fancy indexing, which does
    not bridge cleanly."""
    return table[np.asarray(indices)]


# ---------------------------------------------------------------------------
# Linear & dropout.
# ---------------------------------------------------------------------------
def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """Affine map ``x @ w (+ b)``. ``w`` is ``(in, out)``; optional ``b`` is
    ``(out,)`` and broadcasts over the leading dims."""
    out = x @ w
    return out if b is None else out + b


# Default generator for the ``rng=None`` path. Prefer threading an explicit
# ``rng`` for reproducibility (the splittable-PRNG story is a separate roadmap
# item; this keeps dropout off a hidden global beyond this fallback).
_default_dropout_rng = np.random.default_rng()


def dropout(
    x: Tensor,
    p: float,
    training: bool = True,
    rng: Optional["np.random.Generator"] = None,
) -> Tensor:
    """Inverted dropout. ``p`` is the *drop* probability (torch/jax convention):
    at training time each element is zeroed with probability ``p`` and the
    survivors are scaled by ``1 / (1 - p)`` so the expected value is unchanged.
    A no-op when ``not training`` or ``p == 0``. The mask is a plain-array
    constant, so the gradient simply routes through it."""
    if not training or p == 0.0:
        return x
    keep = 1.0 - p
    gen = _default_dropout_rng if rng is None else rng
    mask = (gen.random(x.shape) < keep) / keep  # x.shape via Var.shape
    return x * mask

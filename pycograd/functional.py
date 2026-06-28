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

from typing import Optional, cast

import numpy as np

# NOTE: like ``examples/models.py``, these helpers are recompiled by pyccolo when
# instrumented on demand, which re-evaluates their annotations -- so the ``Axis``
# alias (a value lookup) is fine, but avoid PEP 604 ``X | None`` spellings here.
from pycograd import ops, random
from pycograd._typing import Axis, Operand, Tensor
from pycograd.dtypes import current_dtype
from pycograd.tensor import Var, _value, grad_is_recording


def _array_out(x: Tensor, y: Tensor) -> Tensor:
    """Preserve the array-in/array-out contract when wrapping a fused primitive: a
    plain (untraced) input yields a plain array (the fused ``Var`` would otherwise break
    eager numpy composition, e.g. ``np.matmul(softmax(...), v)``); a ``Var``/``Tracer``
    input keeps the tape so the enclosing transform differentiates through it."""
    from pycograd.trace import Tracer

    return y if isinstance(x, (Var, Tracer)) else cast(Tensor, _value(y))


def logsumexp(x: Tensor, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """``log(sum(exp(x), axis))`` computed stably via the max-shift identity
    ``m + log(sum(exp(x - m)))`` with ``m = max(x, axis)``. A thin wrapper over the
    fused :func:`pycograd.ops.d_logsumexp` primitive (one tape node + closed-form VJP).
    """
    return _array_out(x, ops.d_logsumexp(x, axis=axis, keepdims=keepdims))


def log_softmax(x: Tensor, axis: Axis = -1) -> Tensor:
    """``log(softmax(x)) = x - logsumexp(x)`` -- stable (no ``exp`` overflow, no
    ``log(0)``), built on the fused :func:`pycograd.ops.d_logsumexp`."""
    return _array_out(x, x - ops.d_logsumexp(x, axis=axis, keepdims=True))


def softmax(x: Tensor, axis: Axis = -1) -> Tensor:
    """Stable softmax (output sums to 1 along ``axis``), the fused
    :func:`pycograd.ops.d_softmax` primitive -- one tape node instead of the
    max/sub/exp/sum/div chain."""
    return _array_out(x, ops.d_softmax(x, axis=axis))


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
    channels: int,
    kh: int,
    kw: int,
    stride: int,
    h_out: int,
    w_out: int,
    dilation: int = 1,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Advanced-index arrays ``(k, i, j)`` selecting, for every (channel, kernel-row,
    kernel-col), the input element feeding each output position. Pure host-side integers
    (a stop-gradient index), so the gather they drive is what carries the gradient.
    ``dilation`` spaces the kernel taps ``dilation`` apart (the atrous/WaveNet trick).
    """
    i0 = np.tile(np.repeat(dilation * np.arange(kh), kw), channels)
    i1 = stride * np.repeat(np.arange(h_out), w_out)
    j0 = np.tile(dilation * np.arange(kw), kh * channels)
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
        z = np.zeros((n, c, ph, x.shape[3]), dtype=current_dtype())
        x = np.concatenate([z, x, z], axis=2)
    if pw > 0:
        z = np.zeros((n, c, x.shape[2], pw), dtype=current_dtype())
        x = np.concatenate([z, x, z], axis=3)
    return x


def conv2d(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    """2-D cross-correlation. ``x`` is ``(N, C_in, H, W)``, ``w`` is
    ``(C_out, C_in/groups, kH, kW)``, optional bias ``b`` is ``(C_out,)``; returns
    ``(N, C_out, H_out, W_out)``. ``dilation`` spaces the kernel taps apart.
    ``groups`` splits the channels into independent groups (the torch weight
    layout): output group ``g`` sees only input group ``g``; ``groups == C_in``
    (with ``C_out`` a multiple of ``C_in``) is a depthwise convolution."""
    c_out, c_in_g, kh, kw = w.shape[0], w.shape[1], w.shape[2], w.shape[3]
    x = _pad2d(x, pad, pad)
    n, c_in, h, ww = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    h_out = (h - (kh - 1) * dilation - 1) // stride + 1
    w_out = (ww - (kw - 1) * dilation - 1) // stride + 1
    k, i, j = _im2col_indices(c_in, kh, kw, stride, h_out, w_out, dilation)
    cols = x[:, k, i, j]  # (N, C_in*kH*kW, H_out*W_out) -- the im2col gather
    if groups == 1:
        w_col = np.reshape(w, (c_out, c_in * kh * kw))
        out = np.einsum("oc,ncp->nop", w_col, cols)  # (N, C_out, H_out*W_out)
    else:
        # Split the channel-major im2col axis (and C_out) into ``groups`` blocks
        # and contract each group independently in one batched einsum.
        p = h_out * w_out
        cols_g = np.reshape(cols, (n, groups, c_in_g * kh * kw, p))
        w_g = np.reshape(w, (groups, c_out // groups, c_in_g * kh * kw))
        out_g = np.einsum("goc,ngcp->ngop", w_g, cols_g)  # (N, G, C_out/G, P)
        out = np.reshape(out_g, (n, c_out, p))
    out = np.reshape(out, (n, c_out, h_out, w_out))
    if b is not None:
        out = out + np.reshape(b, (1, c_out, 1, 1))
    return out


def conv1d(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    """1-D cross-correlation. ``x`` is ``(N, C_in, L)``, ``w`` is
    ``(C_out, C_in/groups, k)``; returns ``(N, C_out, L_out)``. A height-1
    ``conv2d`` underneath. ``dilation`` spaces the kernel taps apart
    (atrous/WaveNet convolutions); ``groups`` gives grouped / depthwise convs."""
    c_out, c_in_g, kk = w.shape[0], w.shape[1], w.shape[2]
    if pad > 0:
        z = np.zeros((x.shape[0], x.shape[1], pad), dtype=current_dtype())
        x = np.concatenate([z, x, z], axis=2)
    x4 = np.reshape(x, (x.shape[0], x.shape[1], 1, x.shape[2]))
    w4 = np.reshape(w, (c_out, c_in_g, 1, kk))
    out = conv2d(x4, w4, b, stride=stride, pad=0, dilation=dilation, groups=groups)
    return np.reshape(out, (out.shape[0], c_out, out.shape[3]))


def causal_conv1d(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    """1-D *causal* convolution: left-pad the length axis by ``(k-1)*dilation``
    zeros (and nothing on the right), so output position ``t`` depends only on
    inputs ``<= t`` -- the parallel/training form whose incremental counterpart is
    :func:`streaming_conv1d`. For ``stride == 1`` the output length equals the
    input length. Built from :func:`conv1d`, so it differentiates / ``vmap``\\ s /
    ``jvp``\\ s for free."""
    pad = (w.shape[2] - 1) * dilation
    # Data-shaped zero pad (``x[..., :1] * 0`` broadcast to width ``pad``), the
    # ``rwkv``/``models`` trick: derived from ``x`` so it stays batched under
    # ``vmap`` (a plain ``np.zeros`` would be an unbatched concatenate operand).
    z = x[:, :, :1] * 0.0 * np.ones((1, 1, pad), dtype=current_dtype())
    x = np.concatenate([z, x], axis=2)  # left pad only
    return conv1d(x, w, b, stride=stride, pad=0, dilation=dilation, groups=groups)


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


def conv_transpose2d(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
) -> Tensor:
    """2-D transposed convolution -- the transpose (adjoint) of :func:`conv2d`,
    sharing its ``(C_out, C_in, kH, kW)`` weight layout. ``x`` is
    ``(N, C_out, H, W)`` (the conv's *output* channels); returns
    ``(N, C_in, H', W')`` with ``H' = (H-1)*stride - 2*pad + (kH-1)*dilation + 1``
    -- the inverse of ``conv2d``'s size map. Used to *upsample* in decoders /
    generators. ``dilation`` matches the dilation of the conv it transposes.

    There is no forward scatter-add primitive yet (``x.at[i].add`` is a separate
    roadmap item), so this is built from existing primitives via the standard
    equivalence: dilate the input by ``stride`` (insert zeros) with a constant
    stop-gradient selection matrix per axis applied through ``einsum``, then a plain
    ``conv2d`` with the channel-transposed, spatially-flipped kernel (itself dilated
    by ``dilation``). Pure, so the reverse pass / jvp / vmap come for free."""
    kh, kw = w.shape[2], w.shape[3]
    h, ww = x.shape[2], x.shape[3]
    h_dil, w_dil = (h - 1) * stride + 1, (ww - 1) * stride + 1
    # Selection matrices placing input element i at dilated position i*stride.
    sh = np.eye(h_dil, dtype=current_dtype())[
        np.arange(h) * stride
    ]  # (h, h_dil), const
    sw = np.eye(w_dil, dtype=current_dtype())[
        np.arange(ww) * stride
    ]  # (ww, w_dil), const
    x_dil = np.einsum("...hw,hH,wW->...HW", x, sh, sw)  # zero-interleaved input
    # Pad by the (dilated) kernel footprint minus 1, less the conv's own padding.
    x_dil = _pad2d(x_dil, (kh - 1) * dilation - pad, (kw - 1) * dilation - pad)
    # Transpose of cross-correlation = convolution: swap C_out/C_in and flip the kernel.
    rev_h, rev_w = np.arange(kh)[::-1], np.arange(kw)[::-1]
    w_flip = np.transpose(w, (1, 0, 2, 3))[:, :, rev_h][:, :, :, rev_w]
    return conv2d(x_dil, w_flip, b, stride=1, pad=0, dilation=dilation)


def conv_transpose1d(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
) -> Tensor:
    """1-D transposed convolution -- the adjoint of :func:`conv1d`, sharing its
    ``(C_out, C_in, k)`` weight layout. ``x`` is ``(N, C_out, L)`` (the conv's
    *output* channels); returns ``(N, C_in, (L-1)*stride - 2*pad + (k-1)*dilation +
    1)``. A height-1 :func:`conv_transpose2d` underneath, exactly as :func:`conv1d`
    wraps :func:`conv2d`. The parallel/training form whose incremental counterpart
    is :func:`streaming_conv_transpose1d`."""
    c_out, c_in, kk = w.shape[0], w.shape[1], w.shape[2]
    x4 = np.reshape(x, (x.shape[0], c_out, 1, x.shape[2]))
    w4 = np.reshape(w, (c_out, c_in, 1, kk))
    out = conv_transpose2d(x4, w4, b, stride=stride, pad=pad, dilation=dilation)
    return np.reshape(out, (out.shape[0], c_in, out.shape[3]))


# ---------------------------------------------------------------------------
# Streaming (incremental) convolutions -- https://ben.bolte.cc/posts/2023-08-24
#
# The *parallel* convs above run over a whole sequence at once; codec models
# (Encodec, HiFi-GAN, WaveNet) instead need to run them one chunk of samples at a
# time during real-time / autoregressive inference, with O(receptive-field)
# memory per step rather than re-scanning the sequence. The trick: carry a small
# ``state`` buffer between steps so each chunk sees exactly the context it needs,
# producing output bit-for-bit identical to the parallel form.
#
# These are *state-in / state-out* inference helpers (the ``batch_norm`` /
# ``rwkv_step`` pattern), not differentiable ops: they run on plain numpy
# (stripping any tracer via ``_value``) and return ``(y, new_state)``. The
# matching trainable op is :func:`causal_conv1d` / :func:`conv_transpose1d`; the
# equivalence is verified in the tests. ``state`` is ``(carry, count)``:
# the cached overlap array plus a small integer (a stride-skip for the conv, a
# bias-free-prepad for the transpose).
# ---------------------------------------------------------------------------
def streaming_conv1d_init(
    num_channels: int, kernel_size: int, dilation: int = 1, batch: int = 1
) -> "tuple[np.ndarray, int]":
    """Causal zero-seed state for :func:`streaming_conv1d`: a buffer of
    ``(k-1)*dilation`` leading zeros, so streaming a sequence chunk-by-chunk from
    this state reproduces :func:`causal_conv1d` over the whole sequence. (Passing
    ``state=None`` instead gives *valid* mode -- matching plain ``conv1d(pad=0)``,
    whose first output waits for a full receptive field.)"""
    pad = (kernel_size - 1) * dilation
    return np.zeros((batch, num_channels, pad), dtype=current_dtype()), 0


def streaming_conv1d(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    state: "Optional[tuple[np.ndarray, int]]" = None,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
) -> "tuple[np.ndarray, tuple[np.ndarray, int]]":
    """One incremental step of a causal :func:`conv1d` over the next chunk ``x``
    ``(N, C_in, L_chunk)``, returning ``(y, new_state)``. Caches the trailing
    ``(k-1)*dilation`` input samples (the receptive-field overlap) so the next
    chunk picks up seamlessly; a chunk too short to fill one receptive field
    yields an empty ``y`` and keeps accumulating. Seed ``state`` from
    :func:`streaming_conv1d_init` for causal behavior; thread the returned state
    into the next call."""
    xa = np.asarray(_value(x))
    cached, leftover = (None, 0) if state is None else state
    if cached is not None:
        xa = np.concatenate([cached, xa], axis=2)
    if leftover > 0:  # discard samples skipped by a stride that outran a chunk
        drop = min(leftover, xa.shape[2])
        xa, leftover = xa[:, :, drop:], leftover - drop
    recep = (w.shape[2] - 1) * dilation + 1
    if xa.shape[2] < recep:  # not enough context yet -- keep buffering
        empty = np.zeros((xa.shape[0], w.shape[0], 0), dtype=current_dtype())
        return empty, (xa, leftover)
    y = np.asarray(
        conv1d(xa, w, b, stride=stride, pad=0, dilation=dilation, groups=groups)
    )
    t = stride * y.shape[2]  # input samples now fully consumed
    return y, (xa[:, :, t:], max(0, t - xa.shape[2]))


def streaming_conv_transpose1d_init() -> None:
    """Initial state for :func:`streaming_conv_transpose1d` -- simply ``None``
    (the transpose caches *output* tails, of which there are none to start). Kept
    for symmetry with :func:`streaming_conv1d_init`."""
    return None


def streaming_conv_transpose1d(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    state: "Optional[tuple[np.ndarray, int]]" = None,
    stride: int = 1,
    dilation: int = 1,
) -> "tuple[np.ndarray, tuple[np.ndarray, int]]":
    """One incremental step of a :func:`conv_transpose1d` over the next chunk ``x``
    ``(N, C_out, L_chunk)``, returning ``(finalized_y, new_state)``. The transpose
    is the dual of :func:`streaming_conv1d`: it caches the *output* overlap (whose
    bias must not be double-counted when summed with the next chunk's output) and
    finalizes ``stride * L_chunk`` samples per step (independent of ``dilation`` --
    which only lengthens the cached overlap). **End-of-stream flush:** the last
    ``state[0]`` holds the final ``(k-1)*dilation + 1 - stride`` output samples that
    no further input can touch -- concatenate it after the loop to recover the full
    sequence. Streaming from ``state=None`` reproduces :func:`conv_transpose1d` over
    the whole input."""
    xa = np.asarray(_value(x))
    bias = None if b is None else np.asarray(_value(b))
    c_out = w.shape[1]  # transpose swaps channels: weight's C_in is the output
    y = np.asarray(
        conv_transpose1d(xa, w, bias, stride=stride, pad=0, dilation=dilation)
    )
    post_y, post_t = (None, 0) if state is None else state
    if post_t > 0:  # bias-only positions a large stride left between footprints
        init_y = np.zeros((xa.shape[0], c_out, post_t), dtype=current_dtype())
        if bias is not None:
            init_y = init_y + np.reshape(bias, (1, c_out, 1))
        y = np.concatenate([init_y, y], axis=2)
    if post_y is not None:  # sum the overlap, undoing the duplicated bias
        n = min(post_y.shape[2], y.shape[2])
        merged = post_y[:, :, :n] + y[:, :, :n]
        if bias is not None:
            merged = merged - np.reshape(bias, (1, c_out, 1))
        y = np.concatenate([merged, post_y[:, :, n:], y[:, :, n:]], axis=2)
    t = stride * xa.shape[2]  # output samples now fully settled
    return y[:, :, :t], (y[:, :, t:], max(0, t - y.shape[2]))


# ---------------------------------------------------------------------------
# 2-D streaming: stream over a single (time / frame) axis -- the last one (W) --
# while the other spatial axis (H) is processed whole each step. This is the 1-D
# state logic above lifted to ``(N, C, H, L_chunk)``, with the cache running along
# W and H carried as a passenger. Streaming over both spatial axes at once is out
# of scope (a separate two-axis bookkeeping problem); the W axis is the streaming
# direction for video frames / spectrogram columns.
# ---------------------------------------------------------------------------
def streaming_conv2d_init(
    num_channels: int,
    height: int,
    kernel_w: int,
    dilation: int = 1,
    batch: int = 1,
) -> "tuple[np.ndarray, int]":
    """Causal zero-seed state for :func:`streaming_conv2d`: a buffer of
    ``(kW-1)*dilation`` leading zero *columns* (shaped ``(batch, C, height,
    pad)``), so streaming a clip frame-by-frame reproduces a W-causal conv2d over
    the whole clip. ``height`` is fixed (only the W axis streams)."""
    pad = (kernel_w - 1) * dilation
    return np.zeros((batch, num_channels, height, pad), dtype=current_dtype()), 0


def streaming_conv2d(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    state: "Optional[tuple[np.ndarray, int]]" = None,
    stride: int = 1,
    dilation: int = 1,
    pad_h: int = 0,
) -> "tuple[np.ndarray, tuple[np.ndarray, int]]":
    """One incremental step of a W-causal :func:`conv2d` over the next frame chunk
    ``x`` ``(N, C, H, L_chunk)``, returning ``(y, new_state)``. Streams along W
    exactly as :func:`streaming_conv1d` does along L (caching the trailing
    ``(kW-1)*dilation`` columns); the H axis is convolved whole each step, padded
    symmetrically by ``pad_h`` (a plain spatial pad, re-applied per chunk -- the
    cache holds raw, un-H-padded columns). Seed ``state`` from
    :func:`streaming_conv2d_init` for causal behavior."""
    xa = np.asarray(_value(x))
    cached, leftover = (None, 0) if state is None else state
    if cached is not None:
        xa = np.concatenate([cached, xa], axis=3)
    if leftover > 0:  # discard W columns skipped by a stride that outran a chunk
        drop = min(leftover, xa.shape[3])
        xa, leftover = xa[:, :, :, drop:], leftover - drop
    recep_w = (w.shape[3] - 1) * dilation + 1
    if xa.shape[3] < recep_w:  # not enough W context yet -- keep buffering
        h_out = (
            xa.shape[2] + 2 * pad_h - (w.shape[2] - 1) * dilation - 1
        ) // stride + 1
        empty = np.zeros((xa.shape[0], w.shape[0], h_out, 0), dtype=current_dtype())
        return empty, (xa, leftover)
    xp = xa
    if pad_h > 0:  # symmetric pad on H only (axis 2); W stays causal via the cache
        z = np.zeros(
            (xa.shape[0], xa.shape[1], pad_h, xa.shape[3]), dtype=current_dtype()
        )
        xp = np.concatenate([z, xa, z], axis=2)
    y = np.asarray(conv2d(xp, w, b, stride=stride, pad=0, dilation=dilation))
    t = stride * y.shape[3]  # W columns now fully consumed
    return y, (xa[:, :, :, t:], max(0, t - xa.shape[3]))


def streaming_conv_transpose2d_init() -> None:
    """Initial state for :func:`streaming_conv_transpose2d` -- ``None`` (it caches
    *output* columns, of which there are none to start), as for the 1-D dual."""
    return None


def streaming_conv_transpose2d(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    state: "Optional[tuple[np.ndarray, int]]" = None,
    stride: int = 1,
    dilation: int = 1,
) -> "tuple[np.ndarray, tuple[np.ndarray, int]]":
    """One incremental step of a :func:`conv_transpose2d` streaming along W over the
    next frame chunk ``x`` ``(N, C_out, H, L_chunk)``, returning ``(finalized_y,
    new_state)``. The W-axis dual of :func:`streaming_conv2d`, mirroring
    :func:`streaming_conv_transpose1d` (cache the output-column overlap, undo the
    duplicated bias, finalize ``stride * L_chunk`` columns); H is upsampled whole
    each step. **End-of-stream flush:** concatenate the final ``state[0]`` after the
    loop. Streaming from ``state=None`` reproduces :func:`conv_transpose2d`."""
    xa = np.asarray(_value(x))
    bias = None if b is None else np.asarray(_value(b))
    c_out = w.shape[1]  # transpose swaps channels: weight's C_in is the output
    y = np.asarray(
        conv_transpose2d(xa, w, bias, stride=stride, pad=0, dilation=dilation)
    )
    post_y, post_t = (None, 0) if state is None else state
    if post_t > 0:  # bias-only columns a large stride left between footprints
        init_y = np.zeros(
            (y.shape[0], c_out, y.shape[2], post_t), dtype=current_dtype()
        )
        if bias is not None:
            init_y = init_y + np.reshape(bias, (1, c_out, 1, 1))
        y = np.concatenate([init_y, y], axis=3)
    if post_y is not None:  # sum the W overlap, undoing the duplicated bias
        n = min(post_y.shape[3], y.shape[3])
        merged = post_y[:, :, :, :n] + y[:, :, :, :n]
        if bias is not None:
            merged = merged - np.reshape(bias, (1, c_out, 1, 1))
        y = np.concatenate([merged, post_y[:, :, :, n:], y[:, :, :, n:]], axis=3)
    t = stride * xa.shape[3]  # W columns now fully settled
    return y[:, :, :, :t], (y[:, :, :, t:], max(0, t - y.shape[3]))


def upsample_nearest2d(x: Tensor, scale: int) -> Tensor:
    """Nearest-neighbor upsample of the last two (spatial) axes by an integer
    ``scale``: ``(N, C, H, W) -> (N, C, H*scale, W*scale)``, each output pixel a
    copy of its source. A gather, so the gradient sum-pools back over each block
    (``d_getitem``'s scatter-add backward)."""
    h, ww = x.shape[2], x.shape[3]
    ih = np.repeat(np.arange(h), scale)  # source row per output row
    iw = np.repeat(np.arange(ww), scale)
    return x[:, :, ih][:, :, :, iw]


def one_hot(indices: "np.ndarray", num_classes: int) -> "np.ndarray":
    """One-hot encode integer ``indices`` along a new last axis. A constant w.r.t. the
    (integer, non-differentiable) indices -- a plain array, not a tape node."""
    return np.eye(num_classes, dtype=current_dtype())[np.asarray(indices)]


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


def batch_norm_init(num_features: int) -> "tuple[np.ndarray, np.ndarray]":
    """Initial ``(running_mean, running_var)`` buffers for :func:`batch_norm` --
    zeros / ones of length ``num_features`` (the channel count)."""
    return np.zeros(num_features, dtype=current_dtype()), np.ones(
        num_features, dtype=current_dtype()
    )


def batch_norm(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> "tuple[Tensor, np.ndarray, np.ndarray]":
    """Batch normalization over the channel axis (axis 1) of an ``(N, C, ...)``
    input. ``gamma``/``beta``/``running_mean``/``running_var`` are length-``C``.

    *State-in/state-out* (the ``rwkv_step`` pattern), since the running statistics
    are mutable buffers, not gradient-trained weights: returns ``(y, new_mean,
    new_var)``. While ``training`` it normalizes with the *batch* mean/variance
    (so the gradient flows through them) and returns the running buffers advanced
    by an EMA of the **detached** batch stats (``momentum`` weight on the new
    batch); at eval it normalizes with the passed running stats and returns them
    unchanged. Thread the returned buffers forward yourself, or -- with the ambient
    DSL -- declare them as ``buffer[...]`` and write them back via
    ``ParamDict.update_buffers``."""
    nd = len(x.shape)
    c = x.shape[1]
    bshape = tuple(c if i == 1 else 1 for i in range(nd))  # broadcast over channels
    g = np.reshape(gamma, bshape)
    b = np.reshape(beta, bshape)
    # Running stats are stop-gradient buffers: pull their raw arrays so the EMA below
    # is plain numpy (no tape), even if the caller passed them as differentiated leaves.
    rm_raw = cast(np.ndarray, _value(running_mean))
    rv_raw = cast(np.ndarray, _value(running_var))
    if training:
        axes = tuple(i for i in range(nd) if i != 1)  # reduce batch + spatial
        mean = np.mean(x, axis=axes, keepdims=True)
        centered = x - mean
        var = np.mean(centered * centered, axis=axes, keepdims=True)
        x_hat = centered / (var + eps) ** 0.5
        # EMA of the *detached* batch stats (``.reshape`` is a plain-array method,
        # never intercepted), kept as length-C buffer arrays.
        batch_mean = cast(np.ndarray, _value(mean)).reshape((c,))
        batch_var = cast(np.ndarray, _value(var)).reshape((c,))
        new_mean = (1.0 - momentum) * rm_raw + momentum * batch_mean
        new_var = (1.0 - momentum) * rv_raw + momentum * batch_var
        return x_hat * g + b, new_mean, new_var
    x_hat = (x - rm_raw.reshape(bshape)) / (rv_raw.reshape(bshape) + eps) ** 0.5
    return x_hat * g + b, rm_raw, rv_raw


def group_norm(
    x: Tensor, gamma: Tensor, beta: Tensor, num_groups: int, eps: float = 1e-5
) -> Tensor:
    """Group normalization over an ``(N, C, ...)`` input: split the ``C`` channels
    into ``num_groups`` groups and normalize each group (its channels + all spatial
    positions) per sample, then scale/shift per channel with length-``C``
    ``gamma``/``beta``. Stateless -- no batch statistics, so it is batch-size
    independent (unlike :func:`batch_norm`)."""
    nd = len(x.shape)
    n, c = x.shape[0], x.shape[1]
    spatial = tuple(x.shape[i] for i in range(2, nd))
    grouped = (n, num_groups, c // num_groups) + spatial
    xg = np.reshape(x, grouped)
    axes = tuple(range(2, len(grouped)))  # channels-in-group + spatial
    mu = np.mean(xg, axis=axes, keepdims=True)
    centered = xg - mu
    var = np.mean(centered * centered, axis=axes, keepdims=True)
    xn = np.reshape(centered / (var + eps) ** 0.5, (n, c) + spatial)
    bshape = tuple(c if i == 1 else 1 for i in range(nd))
    return xn * np.reshape(gamma, bshape) + np.reshape(beta, bshape)


def instance_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Instance normalization: :func:`group_norm` with one group per channel --
    each channel of each sample is normalized over its spatial positions alone."""
    return group_norm(x, gamma, beta, num_groups=x.shape[1], eps=eps)


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


def multi_head_attention(
    q: Tensor, k: Tensor, v: Tensor, num_heads: int, mask: Optional["np.ndarray"] = None
) -> Tensor:
    """Multi-head attention over ``(..., L, d_model)`` inputs: split ``d_model``
    into ``num_heads`` heads of width ``d_model // num_heads``, run
    :func:`scaled_dot_product_attention` independently per head (the head axis just
    rides along as another leading dim), then concatenate the heads back to
    ``(..., L, d_model)``. The Q/K/V/output projections are the caller's
    :func:`linear`s; ``mask`` broadcasts over the head axis."""
    nd = len(q.shape)
    d_model = q.shape[-1]
    d_head = d_model // num_heads

    def split_heads(t: Tensor) -> Tensor:
        lead = tuple(t.shape[i] for i in range(nd - 2))
        seq = t.shape[-2]
        t = np.reshape(t, lead + (seq, num_heads, d_head))  # (..., L, H, d_head)
        # move the head axis (nd-1) ahead of L (nd-2) -> (..., H, L, d_head)
        return np.transpose(t, tuple(range(nd - 2)) + (nd - 1, nd - 2, nd))

    out = scaled_dot_product_attention(
        split_heads(q), split_heads(k), split_heads(v), mask
    )  # (..., H, L, d_head)
    lead = tuple(out.shape[i] for i in range(nd - 2))
    seq = out.shape[-2]
    # (..., H, L, d_head) -> (..., L, H, d_head) -> (..., L, d_model)
    merged = np.transpose(out, tuple(range(nd - 2)) + (nd - 1, nd - 2, nd))
    return np.reshape(merged, lead + (seq, d_model))


def embedding(
    table: Tensor, indices: "np.ndarray", *, padding_idx: Optional[int] = None
) -> Tensor:
    """Look up rows of ``table`` (``(num_embeddings, *feat)``) by integer
    ``indices``, returning ``(*indices.shape, *feat)``.

    A first-class primitive (``ops.d_embedding``): the forward is a plain gather, so
    the gradient scatter-adds back into the looked-up rows. ``padding_idx`` (if given)
    holds that row fixed by zeroing only its gradient (PyTorch semantics). The compile
    backends (torch / jax / tf) intercept this symbol and lower it to their native
    gather (``F.embedding`` / ``take`` / ``gather``)."""
    from pycograd.trace import Tracer, bind

    out = bind(
        ops.d_embedding, table, indices=np.asarray(indices), padding_idx=padding_idx
    )
    # The primitive always lifts ``table`` to a ``Var``; keep that boxed result only when
    # there is gradient/transform context to preserve (a live transform level, a boxed
    # ``table``, or an ambient grad pass holding a ``Weight``). Otherwise -- a plain
    # untraced gather -- hand back a bare array, matching the other functional ops.
    if (
        isinstance(out, Tracer)
        or isinstance(table, (Var, Tracer))
        or grad_is_recording()
    ):
        return cast(Tensor, out)
    return cast(Tensor, _value(cast(Operand, out)))


# ---------------------------------------------------------------------------
# Linear & dropout.
# ---------------------------------------------------------------------------
def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """Affine map ``x @ w (+ b)``. ``w`` is ``(in, out)``; optional ``b`` is
    ``(out,)`` and broadcasts over the leading dims."""
    out = x @ w
    return out if b is None else out + b


def dropout(
    x: Tensor,
    p: float,
    training: bool = True,
    key: Optional["np.ndarray"] = None,
    rng: Optional["np.random.Generator"] = None,
) -> Tensor:
    """Inverted dropout. ``p`` is the *drop* probability (torch/jax convention):
    at training time each element is zeroed with probability ``p`` and the
    survivors are scaled by ``1 / (1 - p)`` so the expected value is unchanged.
    A no-op when ``not training`` or ``p == 0``.

    Randomness is threaded explicitly -- pass a splittable ``key`` (from
    :mod:`pycograd.random`; preferred) or an ``np.random.Generator`` via ``rng``;
    one is required while training (there is no hidden global generator). On a
    ``(B, ...)`` batch this already zeros each element independently, so a single
    key gives per-sample dropout without ``vmap`` -- ``split`` the key per step (or
    ``fold_in`` the step index) to vary masks across training steps. The mask is a
    stop-gradient plain-array constant -- the gradient routes through it, and it
    lowers to the compile backends as a constant (RNG stays host-side).

    (Mapping a *per-key* mask with ``vmap(..., in_axes=(0, 0))`` over split keys is
    not supported: host-side RNG can't consume a batched-tracer key, and vmap's
    single symbolic pass would share one mask regardless -- draw the full-batch
    mask as above instead.)"""
    if not training or p == 0.0:
        return x
    keep = 1.0 - p
    if key is not None:
        mask = random.bernoulli(key, keep, x.shape) / keep  # x.shape via Var.shape
    elif rng is not None:
        mask = (rng.random(x.shape) < keep) / keep
    else:
        raise ValueError(
            "dropout: training=True needs an explicit `key` (pycograd.random) or "
            "`rng` (np.random.Generator) -- there is no global generator"
        )
    return x * mask

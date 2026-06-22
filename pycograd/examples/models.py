# -*- coding: utf-8 -*-
"""Demo models and synthetic data for the autodiff examples.

The helper/loss functions are typed with ``Tensor`` (``Var``-or-ndarray): numpy
calls on a ``Var`` type-check because ``Var.__array__`` makes it statically
array-like; scalar-returning losses are typed ``Operand``. The "fancy" layers
(LayerNorm, Dropout, attention, a Transformer block) are plain helpers with no
autodiff rule -- each is instrumented on demand so gradients flow through them.
"""
from __future__ import annotations

from typing import Optional, cast

import numpy as np

from pycograd._typing import Array, ArrayLike, Operand, Tensor
from pycograd.functional import softmax as _softmax
from pycograd.tree import PyTree

# NOTE: these demo functions are recompiled by pyccolo during instrumentation,
# which re-evaluates their annotations. PEP 585 builtins (``list``/``dict``/
# ``tuple``) are runtime-safe on 3.9, but PEP 604 ``X | None`` is not -- so the
# one optional argument below keeps ``Optional`` rather than ``| None``.


# ---------------------------------------------------------------------------
# Logistic regression.
# ---------------------------------------------------------------------------
def _make_data() -> tuple[Array, Array]:
    rng = np.random.default_rng(0)
    n = 50
    pos = rng.normal(loc=[2.0, 2.0], scale=0.6, size=(n, 2))
    neg = rng.normal(loc=[-2.0, -2.0], scale=0.6, size=(n, 2))
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    return X, y


_X, _y = _make_data()


def sigmoid(z: Tensor) -> Tensor:
    # A plain helper with no autodiff rule -- differentiated by instrumenting it
    # on demand when ``logistic_loss`` calls it. (The fused ``d_sigmoid`` primitive
    # is the traced-path equivalent; this composition stays array-in/array-out so
    # plain-array inference -- accuracy, text generation -- keeps working too.)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(w: Tensor, b: Tensor) -> Operand:
    z = _X @ w + b  # ndarray @ Var defers to Var.__rmatmul__; + b broadcasts
    p = sigmoid(z)  # helper differentiated transparently
    eps = 1e-12
    return -np.mean(_y * np.log(p + eps) + (1 - _y) * np.log(1 - p + eps))


def _accuracy(w: Array, b: ArrayLike) -> float:
    z = _X @ w + b
    return float(np.mean((z > 0).astype(float) == _y))


# Same logistic model, but parameters are a declarative ``params(...)`` block so
# we can freeze the bias: ``model["b"]`` is read by name and held fixed.
def logistic_param_loss(model: dict[str, Tensor]) -> Operand:
    z = _X @ model["w"] + model["b"]
    p = sigmoid(z)
    eps = 1e-12
    return -np.mean(_y * np.log(p + eps) + (1 - _y) * np.log(1 - p + eps))


# ---------------------------------------------------------------------------
# A 2-layer feedforward net (ReLU hidden + softmax head), 3-way classifier.
# ---------------------------------------------------------------------------
def relu(z: Tensor) -> Tensor:
    return np.maximum(z, 0.0)


def softmax(z: Tensor) -> Tensor:
    # Stable softmax over the class axis, from the first-class op in functional.
    return _softmax(z, axis=1)


def cross_entropy(probs: Tensor, y_onehot: Tensor) -> Operand:
    return -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))


def mlp_forward(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) -> Tensor:
    hidden = relu(x @ w1 + b1)
    return softmax(hidden @ w2 + b2)


def _make_blobs() -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(1)
    n = 30
    centers = np.array([[2.0, 2.0], [-2.0, 2.0], [0.0, -2.5]])
    X = np.vstack([rng.normal(loc=c, scale=0.5, size=(n, 2)) for c in centers])
    labels = np.repeat(np.arange(3), n)
    onehot = np.eye(3)[labels]
    return X, labels, onehot


_Xc, _labels, _Yoh = _make_blobs()


def mlp_loss(w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) -> Operand:
    return cross_entropy(mlp_forward(_Xc, w1, b1, w2, b2), _Yoh)


def _init_mlp(
    rng: np.random.Generator, n_in: int = 2, n_hidden: int = 16, n_out: int = 3
) -> tuple[Array, ...]:
    return (
        0.1 * rng.standard_normal((n_in, n_hidden)),
        np.zeros(n_hidden),
        0.1 * rng.standard_normal((n_hidden, n_out)),
        np.zeros(n_out),
    )


def _mlp_accuracy(params: tuple[Array, ...]) -> float:
    w1, b1, w2, b2 = params
    logits = np.maximum(_Xc @ w1 + b1, 0.0) @ w2 + b2
    return float(np.mean(np.argmax(logits, axis=1) == _labels))


# ---------------------------------------------------------------------------
# The same 2-layer MLP, but with parameters as a *pytree* -- one nested dict
# instead of four positional arrays.
# ---------------------------------------------------------------------------
# A param pytree: ``{"hidden": {"w", "b"}, "out": {"w", "b"}}``. Annotated as a
# concrete nested dict (not the open ``PyTree`` union) so the body indexes it; the
# leaves are arrays at init and ``Var``s once ``value_and_grad`` wraps them, both
# of which ``Tensor`` covers.
MLPParams = dict[str, dict[str, Tensor]]


def mlp_tree_loss(params: MLPParams) -> Operand:
    h, o = params["hidden"], params["out"]
    probs = mlp_forward(_Xc, h["w"], h["b"], o["w"], o["b"])
    return cross_entropy(probs, _Yoh)


def _init_mlp_tree(
    rng: np.random.Generator, n_in: int = 2, n_hidden: int = 16, n_out: int = 3
) -> PyTree:
    return {
        "hidden": {
            "w": 0.1 * rng.standard_normal((n_in, n_hidden)),
            "b": np.zeros(n_hidden),
        },
        "out": {
            "w": 0.1 * rng.standard_normal((n_hidden, n_out)),
            "b": np.zeros(n_out),
        },
    }


# A minibatch-friendly loss: instead of closing over the whole training set, it
# takes the batch explicitly, so an optimizer can step on ``batches(_Xc, _Yoh)``.
def mlp_batch_loss(params: MLPParams, xb: Tensor, yb: Tensor) -> Operand:
    h, o = params["hidden"], params["out"]
    probs = mlp_forward(xb, h["w"], h["b"], o["w"], o["b"])
    return cross_entropy(probs, yb)


def _mlp_tree_accuracy(params: PyTree) -> float:
    p = cast(dict[str, dict[str, Array]], params)
    h, o = p["hidden"], p["out"]
    logits = np.maximum(_Xc @ h["w"] + h["b"], 0.0) @ o["w"] + o["b"]
    return float(np.mean(np.argmax(logits, axis=1) == _labels))


# ---------------------------------------------------------------------------
# The same classifier with a LayerNorm and a Dropout layer. LayerNorm is
# stateless (gamma/beta are just differentiated params), and Dropout's mask is a
# sampled constant the gradient routes through.
# ---------------------------------------------------------------------------
_dropout_rng = np.random.default_rng(0)


def linear(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return x @ w + b


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    mu = np.mean(x, axis=-1, keepdims=True)
    centered = x - mu
    var = np.mean(centered * centered, axis=-1, keepdims=True)
    return centered / (var + eps) ** 0.5 * gamma + beta


def dropout(x: Tensor, keep: float, training: bool) -> Tensor:
    if not training:
        return x
    mask = (_dropout_rng.random(x.shape) < keep) / keep  # x.shape via Var.shape
    return x * mask  # mask is a plain-array constant; grad routes through it


def deep_forward(
    x: Tensor,
    w1: Tensor,
    b1: Tensor,
    g: Tensor,
    beta: Tensor,
    w2: Tensor,
    b2: Tensor,
    training: bool,
) -> Tensor:
    hidden = relu(layer_norm(linear(x, w1, b1), g, beta))
    hidden = dropout(hidden, 0.9, training)
    return softmax(linear(hidden, w2, b2))


def deep_loss(
    w1: Tensor, b1: Tensor, g: Tensor, beta: Tensor, w2: Tensor, b2: Tensor
) -> Operand:
    return cross_entropy(deep_forward(_Xc, w1, b1, g, beta, w2, b2, True), _Yoh)


def _init_deep(
    rng: np.random.Generator, n_in: int = 2, n_hidden: int = 16, n_out: int = 3
) -> tuple[Array, ...]:
    return (
        0.3 * rng.standard_normal((n_in, n_hidden)),
        np.zeros(n_hidden),
        np.ones(n_hidden),  # layernorm gamma
        np.zeros(n_hidden),  # layernorm beta
        0.3 * rng.standard_normal((n_hidden, n_out)),
        np.zeros(n_out),
    )


def _deep_accuracy(params: tuple[Array, ...]) -> float:
    w1, b1, g, beta, w2, b2 = params
    probs = deep_forward(_Xc, w1, b1, g, beta, w2, b2, training=False)  # dropout off
    return float(np.mean(np.argmax(probs, axis=1) == _labels))


# ---------------------------------------------------------------------------
# A single-head Transformer encoder block and a tiny sequence classifier.
# ---------------------------------------------------------------------------
def softmax_last(z: Tensor) -> Tensor:
    # Stable softmax over the last axis (attention weights), via functional.
    return _softmax(z, axis=-1)


def attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Optional[np.ndarray] = None
) -> Tensor:
    scores = (q @ k.T) * (q.shape[-1] ** -0.5)  # scaled dot-product
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # masked positions get ~0 weight
    return softmax_last(scores) @ v


def transformer_block(
    x: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    g1: Tensor,
    beta1: Tensor,
    w1: Tensor,
    bff1: Tensor,
    w2: Tensor,
    bff2: Tensor,
    g2: Tensor,
    beta2: Tensor,
) -> Tensor:
    attended = attention(x @ wq, x @ wk, x @ wv) @ wo
    x = layer_norm(x + attended, g1, beta1)  # residual + LayerNorm
    ff = relu(x @ w1 + bff1) @ w2 + bff2
    return layer_norm(x + ff, g2, beta2)  # residual + LayerNorm


def softmax_ce(logits: Tensor, onehot: Tensor) -> Operand:
    return -np.sum(onehot * np.log(softmax_last(logits) + 1e-12))


def _seq_logits(x: Tensor, *params: Tensor) -> Tensor:
    *block, wc, bc = params
    pooled = np.mean(transformer_block(x, *block), axis=0)  # mean-pool over positions
    return pooled @ wc + bc


def _make_sequences() -> tuple[list[Array], Array, Array]:
    rng = np.random.default_rng(3)
    seq_len, d_model, n_per = 3, 4, 8
    seqs: list[Array] = []
    labels: list[int] = []
    for cls, shift in enumerate((-0.8, 0.8)):
        for _ in range(n_per):
            seqs.append(rng.normal(shift, 0.5, size=(seq_len, d_model)))
            labels.append(cls)
    return seqs, np.array(labels), np.eye(2)[labels]


_SEQS, _SEQ_LABELS, _SEQ_OH = _make_sequences()


def transformer_loss(*params: Tensor) -> Operand:
    total: Operand = 0.0
    for i in range(len(_SEQS)):
        total = total + softmax_ce(_seq_logits(_SEQS[i], *params), _SEQ_OH[i])
    return total / len(_SEQS)


def _init_transformer(
    rng: np.random.Generator, d_model: int = 4, d_ff: int = 8, n_classes: int = 2
) -> tuple[Array, ...]:
    s = 0.3
    rn = rng.standard_normal
    return (
        s * rn((d_model, d_model)),  # wq
        s * rn((d_model, d_model)),  # wk
        s * rn((d_model, d_model)),  # wv
        s * rn((d_model, d_model)),  # wo
        np.ones(d_model),  # layernorm-1 gamma
        np.zeros(d_model),  # layernorm-1 beta
        s * rn((d_model, d_ff)),  # ffn in
        np.zeros(d_ff),
        s * rn((d_ff, d_model)),  # ffn out
        np.zeros(d_model),
        np.ones(d_model),  # layernorm-2 gamma
        np.zeros(d_model),  # layernorm-2 beta
        s * rn((d_model, n_classes)),  # classifier head
        np.zeros(n_classes),
    )


def _transformer_accuracy(params: tuple[Array, ...]) -> float:
    preds = [int(np.argmax(_seq_logits(x, *params))) for x in _SEQS]
    return float(np.mean(np.array(preds) == _SEQ_LABELS))


# ---------------------------------------------------------------------------
# RWKV: a recurrent architecture that trains in a parallel (unrolled-over-time)
# form and runs inference in an O(1)-per-token recurrent form. Like the layers
# above, every piece is a plain helper instrumented on demand -- the whole thing
# is built from ``exp`` / ``maximum`` / ``sigmoid`` / matmul / indexing, so it
# differentiates, batches under ``vmap``, and compiles to torch/jax/tf unchanged.
# ---------------------------------------------------------------------------
def token_shift(x: Tensor) -> Tensor:
    # Shift a ``(T, D)`` sequence one step in time: row ``t`` becomes the previous
    # token, with a zero vector standing in for "the token before the first".
    # ``x[:1] * 0.0`` makes that zero row without reading ``x``'s shape, so the
    # whole thing is concatenate + slice -- composable under every transform.
    zero = x[:1] * 0.0
    return cast(Tensor, np.concatenate([zero, x[:-1]], axis=0))


def wkv(w: Tensor, u: Tensor, k: Tensor, v: Tensor) -> Tensor:
    # The WKV kernel: a softmax-weighted running average of the values ``v``,
    # where weights decay by ``exp(-w)`` per step and the current step gets a
    # ``u`` "bonus". Carried in the numerically-stable form used by RWKV (the
    # "max trick"): running numerator/denominator ``alpha``/``beta`` normalized by
    # a running max exponent ``eps`` so ``exp`` never overflows. ``w``/``u`` are
    # ``(D,)``; ``k``/``v`` are ``(T, D)``.
    alpha = u * 0.0  # zeros shaped like one channel vector, dtype from ``u``
    beta = u * 0.0
    eps = u * 0.0
    outs = []
    for t in range(k.shape[0]):
        kt, vt = k[t], v[t]
        # Output at ``t`` folds the current (k, v) in with the bonus ``u``.
        tau = np.maximum(u + kt, eps)
        e1 = np.exp(eps - tau)
        e2 = np.exp(u + kt - tau)
        outs.append((e1 * alpha + e2 * vt) / (e1 * beta + e2))
        # Advance the state by one step: decay ``eps`` by ``w``, fold in (k, v).
        eps2 = np.maximum(eps - w, kt)
        a1 = np.exp(eps - w - eps2)
        a2 = np.exp(kt - eps2)
        alpha = a1 * alpha + a2 * vt
        beta = a1 * beta + a2
        eps = eps2
    return cast(Tensor, np.stack(outs, axis=0))


def rwkv_time_mixing(
    x: Tensor,
    last_x: Tensor,
    time_decay: Tensor,
    time_first: Tensor,
    mix_k: Tensor,
    mix_v: Tensor,
    mix_r: Tensor,
    wk: Tensor,
    wv: Tensor,
    wr: Tensor,
    wo: Tensor,
) -> Tensor:
    # "Attention" block: per-channel token-shift mixing into key/value/receptance,
    # a sigmoid receptance gate over the WKV output, then an output projection.
    xk = x * mix_k + last_x * (1.0 - mix_k)
    xv = x * mix_v + last_x * (1.0 - mix_v)
    xr = x * mix_r + last_x * (1.0 - mix_r)
    k = xk @ wk
    v = xv @ wv
    sr = sigmoid(xr @ wr)
    out = wkv(np.exp(time_decay), time_first, k, v) * sr
    return cast(Tensor, out @ wo)


def rwkv_channel_mixing(
    x: Tensor,
    last_x: Tensor,
    mix_k: Tensor,
    mix_r: Tensor,
    wk: Tensor,
    wr: Tensor,
    wv: Tensor,
) -> Tensor:
    # Feed-forward block: token-shift mixing, a squared-ReLU hidden layer, and a
    # sigmoid receptance gate.
    xk = x * mix_k + last_x * (1.0 - mix_k)
    xr = x * mix_r + last_x * (1.0 - mix_r)
    k = relu(xk @ wk)
    return cast(Tensor, sigmoid(xr @ wr) * ((k * k) @ wv))


# A block's parameters as one flat ``name -> tensor`` dict (LayerNorm gains/biases
# plus ``att_*`` time-mixing and ``ffn_*`` channel-mixing weights).
RWKVBlockParams = dict[str, Tensor]


def rwkv_block(x: Tensor, p: RWKVBlockParams) -> Tensor:
    # Pre-norm residual block: x -> x + time_mixing(LN(x)); x -> x + ffn(LN(x)).
    xn = layer_norm(x, p["ln1_g"], p["ln1_b"])
    x = x + rwkv_time_mixing(
        xn,
        token_shift(xn),
        p["att_time_decay"],
        p["att_time_first"],
        p["att_mix_k"],
        p["att_mix_v"],
        p["att_mix_r"],
        p["att_wk"],
        p["att_wv"],
        p["att_wr"],
        p["att_wo"],
    )
    xn2 = layer_norm(x, p["ln2_g"], p["ln2_b"])
    x = x + rwkv_channel_mixing(
        xn2,
        token_shift(xn2),
        p["ffn_mix_k"],
        p["ffn_mix_r"],
        p["ffn_wk"],
        p["ffn_wr"],
        p["ffn_wv"],
    )
    return x


def rwkv_lm(
    onehot: Tensor,
    top: dict[str, Tensor],
    blocks: list[RWKVBlockParams],
) -> Tensor:
    # Char-level language model. The input is one-hot ``(T, V)`` token rows, so the
    # embedding is a plain ``onehot @ emb`` matmul -- exactly an embedding lookup,
    # but expressed without fancy indexing so the whole model lowers to torch/jax/tf
    # alike. Pre-norm the embeddings, run the RWKV blocks, final LayerNorm, then
    # project to per-token vocab logits.
    x = onehot @ top["emb"]  # (T, D)
    x = layer_norm(x, top["ln0_g"], top["ln0_b"])
    for block in blocks:
        x = rwkv_block(x, block)
    x = layer_norm(x, top["ln_out_g"], top["ln_out_b"])
    return cast(Tensor, x @ top["head_w"] + top["head_b"])


# A tiny char corpus so the model below is unit-testable (next-char prediction).
def _make_char_data() -> tuple[Array, Array, str]:
    text = "pycograd rwkv "
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    onehot = np.eye(len(vocab))[ids]
    return ids, onehot, "".join(vocab)


_CHAR_IDS, _CHAR_OH, _CHAR_VOCAB = _make_char_data()


# The model is a ``(top, blocks)`` pair -- both clean ``name -> tensor`` pytrees,
# so ``value_and_grad`` differentiates it without a mixed-type container.
RWKVParams = tuple[dict[str, Tensor], list[RWKVBlockParams]]


def _init_rwkv_block(rng: np.random.Generator, d_model: int) -> dict[str, Array]:
    s = 0.1
    rn = rng.standard_normal
    z, o = np.zeros(d_model), np.ones(d_model)
    return {
        "ln1_g": o.copy(),
        "ln1_b": z.copy(),
        "ln2_g": o.copy(),
        "ln2_b": z.copy(),
        "att_time_decay": z.copy(),
        "att_time_first": z.copy(),
        "att_mix_k": rng.random(d_model),
        "att_mix_v": rng.random(d_model),
        "att_mix_r": rng.random(d_model),
        "att_wk": s * rn((d_model, d_model)),
        "att_wv": s * rn((d_model, d_model)),
        "att_wr": s * rn((d_model, d_model)),
        "att_wo": s * rn((d_model, d_model)),
        "ffn_mix_k": rng.random(d_model),
        "ffn_mix_r": rng.random(d_model),
        "ffn_wk": s * rn((d_model, 4 * d_model)),
        "ffn_wr": s * rn((d_model, d_model)),
        "ffn_wv": s * rn((4 * d_model, d_model)),
    }


def _init_rwkv(
    rng: np.random.Generator, vocab: int, d_model: int = 8, n_blocks: int = 1
) -> RWKVParams:
    s = 0.1
    rn = rng.standard_normal
    top: dict[str, Array] = {
        "emb": s * rn((vocab, d_model)),
        "ln0_g": np.ones(d_model),
        "ln0_b": np.zeros(d_model),
        "ln_out_g": np.ones(d_model),
        "ln_out_b": np.zeros(d_model),
        "head_w": s * rn((d_model, vocab)),
        "head_b": np.zeros(vocab),
    }
    blocks = [_init_rwkv_block(rng, d_model) for _ in range(n_blocks)]
    return cast(RWKVParams, (top, blocks))


# ``top`` and ``blocks`` are passed as two positional pytrees (not one tuple) so the
# plain ``gradient_descent`` helper -- which steps ``loss_fn(*params)`` argument by
# argument -- trains the model directly.
def rwkv_loss(top: dict[str, Tensor], blocks: list[RWKVBlockParams]) -> Operand:
    # Mean next-char cross-entropy: predict chars 1..T from the prefix 0..T-1.
    logits = rwkv_lm(_CHAR_OH[:-1], top, blocks)
    targets = _CHAR_OH[1:]
    return -np.mean(np.sum(targets * np.log(softmax_last(logits) + 1e-12), axis=-1))


def _rwkv_accuracy(top: dict[str, Tensor], blocks: list[RWKVBlockParams]) -> float:
    logits = rwkv_lm(_CHAR_OH[:-1], top, blocks)
    preds = np.argmax(np.asarray(logits), axis=-1)
    return float(np.mean(preds == _CHAR_IDS[1:]))


# Per-block recurrent state for O(1)-per-token inference: the previous token's
# normed inputs (the token-shift carry) plus the WKV running numerator/denominator
# and max-exponent.
RWKVState = list[dict[str, Array]]


def rwkv_init_state(d_model: int, n_blocks: int) -> RWKVState:
    z = np.zeros(d_model)
    return [
        {
            "att_x": z.copy(),
            "alpha": z.copy(),
            "beta": z.copy(),
            "eps": z.copy(),
            "ffn_x": z.copy(),
        }
        for _ in range(n_blocks)
    ]


def rwkv_step(
    token: int, top: dict[str, Array], blocks: list[RWKVBlockParams], state: RWKVState
) -> tuple[Array, RWKVState]:
    # The recurrent form: process one token given the carried ``state``, returning
    # next-token logits and the advanced state. Equivalent token-for-token to the
    # parallel ``rwkv_lm`` (verified in the tests) but O(1) memory per step, which is
    # what makes RWKV cheap to sample from. Plain-numpy inference -- no tape.
    x = np.asarray(top["emb"])[token]
    x = layer_norm(x, top["ln0_g"], top["ln0_b"])
    new_state: RWKVState = []
    for p, st in zip(blocks, state):
        xn = layer_norm(x, p["ln1_g"], p["ln1_b"])
        lx = st["att_x"]
        k = (xn * p["att_mix_k"] + lx * (1 - p["att_mix_k"])) @ p["att_wk"]
        v = (xn * p["att_mix_v"] + lx * (1 - p["att_mix_v"])) @ p["att_wv"]
        r = sigmoid((xn * p["att_mix_r"] + lx * (1 - p["att_mix_r"])) @ p["att_wr"])
        w, u = np.exp(p["att_time_decay"]), p["att_time_first"]
        alpha, beta, eps = st["alpha"], st["beta"], st["eps"]
        tau = np.maximum(u + k, eps)
        e1, e2 = np.exp(eps - tau), np.exp(u + k - tau)
        wkv_t = (e1 * alpha + e2 * v) / (e1 * beta + e2)
        eps2 = np.maximum(eps - w, k)
        a1, a2 = np.exp(eps - w - eps2), np.exp(k - eps2)
        x = x + (wkv_t * r) @ p["att_wo"]

        xn2 = layer_norm(x, p["ln2_g"], p["ln2_b"])
        fx = st["ffn_x"]
        fk = relu((xn2 * p["ffn_mix_k"] + fx * (1 - p["ffn_mix_k"])) @ p["ffn_wk"])
        fr = sigmoid((xn2 * p["ffn_mix_r"] + fx * (1 - p["ffn_mix_r"])) @ p["ffn_wr"])
        x = x + fr * ((fk * fk) @ p["ffn_wv"])

        new_state.append(
            {
                "att_x": np.asarray(xn),
                "alpha": a1 * alpha + a2 * v,
                "beta": a1 * beta + a2,
                "eps": eps2,
                "ffn_x": np.asarray(xn2),
            }
        )
    x = layer_norm(x, top["ln_out_g"], top["ln_out_b"])
    return np.asarray(x @ top["head_w"] + top["head_b"]), new_state


# ---------------------------------------------------------------------------
# Classic recurrent cells: vanilla RNN, GRU, LSTM. Like every layer above, each
# is a plain helper with no autodiff rule -- built from ``tanh`` / ``sigmoid`` /
# matmul and a Python time-loop, so it differentiates on the numpy tape, batches
# under ``vmap``, and compiles to torch/jax/tf unchanged.
#
# The scans carry the per-step state as a ``(1, D)`` *row* vector and slice each
# input timestep as ``x[t : t + 1]`` (also ``(1, D)``), so every matmul is
# rank-2 -- TensorFlow's ``@`` needs rank >= 2, and the rows lower cleanly to all
# three backends. The state is seeded from a bias param times zero
# (``np.expand_dims(b * 0.0, 0)``) rather than a hardcoded shape -- the
# data-shaped-zero trick ``wkv`` uses -- so the scan still vectorizes over a batch
# of sequences for free (``vmap`` gives each sequence its own *logical* shape).
# The cells themselves are rank-polymorphic (elementwise + matmul + broadcast), so
# they read identically whether handed a ``(D,)`` vector or a ``(1, D)`` row.
# ---------------------------------------------------------------------------
# A cell's parameters as one flat ``name -> tensor`` dict (input/hidden weight
# matrices plus per-gate biases), like ``RWKVBlockParams``.
RNNParams = dict[str, Tensor]
GRUParams = dict[str, Tensor]
LSTMParams = dict[str, Tensor]


def rnn_cell(x: Tensor, h: Tensor, Wx: Tensor, Wh: Tensor, b: Tensor) -> Tensor:
    # Elman cell: a tanh of the affine mix of the input and the previous hidden.
    return cast(Tensor, np.tanh(x @ Wx + h @ Wh + b))


def gru_cell(x: Tensor, h: Tensor, p: GRUParams) -> Tensor:
    # Gated recurrent unit (torch convention): an update gate ``z`` interpolates
    # between the previous hidden and a reset-gated candidate ``n``.
    z = sigmoid(x @ p["Wz"] + h @ p["Uz"] + p["bz"])  # update gate
    r = sigmoid(x @ p["Wr"] + h @ p["Ur"] + p["br"])  # reset gate
    n = np.tanh(x @ p["Wn"] + (r * h) @ p["Un"] + p["bn"])  # candidate
    return cast(Tensor, (1.0 - z) * n + z * h)


def lstm_cell(x: Tensor, h: Tensor, c: Tensor, p: LSTMParams) -> tuple[Tensor, Tensor]:
    # Long short-term memory: input/forget/output gates around a cell state ``c``.
    # Returns the (hidden, cell) pair carried to the next step.
    i = sigmoid(x @ p["Wi"] + h @ p["Ui"] + p["bi"])  # input gate
    f = sigmoid(x @ p["Wf"] + h @ p["Uf"] + p["bf"])  # forget gate
    g = np.tanh(x @ p["Wg"] + h @ p["Ug"] + p["bg"])  # candidate cell
    o = sigmoid(x @ p["Wo"] + h @ p["Uo"] + p["bo"])  # output gate
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    return cast(Tensor, h_new), cast(Tensor, c_new)


def rnn_scan(x: Tensor, Wx: Tensor, Wh: Tensor, b: Tensor) -> Tensor:
    # Unroll the Elman cell over a ``(T, D_in)`` sequence; return the ``(T, D_hid)``
    # hidden states. The state is a ``(1, D_hid)`` row (rank-2 matmuls), seeded
    # from ``b * 0.0`` so its shape comes from a param, not from ``x``.
    h: Tensor = np.expand_dims(b * 0.0, 0)
    outs = []
    for t in range(x.shape[0]):
        h = rnn_cell(x[t : t + 1], h, Wx, Wh, b)
        outs.append(h)
    return cast(Tensor, np.concatenate(outs, axis=0))


def gru_scan(x: Tensor, p: GRUParams) -> Tensor:
    h: Tensor = np.expand_dims(p["bz"] * 0.0, 0)
    outs = []
    for t in range(x.shape[0]):
        h = gru_cell(x[t : t + 1], h, p)
        outs.append(h)
    return cast(Tensor, np.concatenate(outs, axis=0))


def lstm_scan(x: Tensor, p: LSTMParams) -> Tensor:
    h: Tensor = np.expand_dims(p["bi"] * 0.0, 0)
    c: Tensor = np.expand_dims(p["bi"] * 0.0, 0)
    outs = []
    for t in range(x.shape[0]):
        h, c = lstm_cell(x[t : t + 1], h, c, p)
        outs.append(h)
    return cast(Tensor, np.concatenate(outs, axis=0))


# Char-level language models on top of each scan: embed one-hot rows with a matmul
# (an embedding lookup written so it lowers to every backend), run the recurrence,
# and project the hidden states to per-token vocab logits. ``d_in == d_hidden ==
# d_model`` throughout so the matmuls line up.
def rnn_lm(onehot: Tensor, p: RNNParams) -> Tensor:
    x = onehot @ p["emb"]  # (T, D)
    h = rnn_scan(x, p["Wx"], p["Wh"], p["b"])
    return cast(Tensor, h @ p["head_w"] + p["head_b"])


def gru_lm(onehot: Tensor, p: GRUParams) -> Tensor:
    x = onehot @ p["emb"]
    h = gru_scan(x, p)
    return cast(Tensor, h @ p["head_w"] + p["head_b"])


def lstm_lm(onehot: Tensor, p: LSTMParams) -> Tensor:
    x = onehot @ p["emb"]
    h = lstm_scan(x, p)
    return cast(Tensor, h @ p["head_w"] + p["head_b"])


def _init_rnn_cell(rng: np.random.Generator, d_in: int, d_hidden: int) -> RNNParams:
    s = 0.1
    rn = rng.standard_normal
    return {
        "Wx": s * rn((d_in, d_hidden)),
        "Wh": s * rn((d_hidden, d_hidden)),
        "b": np.zeros(d_hidden),
    }


def _init_gru_cell(rng: np.random.Generator, d_in: int, d_hidden: int) -> GRUParams:
    s = 0.1
    rn = rng.standard_normal
    p: GRUParams = {}
    for gate in ("z", "r", "n"):
        p["W" + gate] = s * rn((d_in, d_hidden))
        p["U" + gate] = s * rn((d_hidden, d_hidden))
        p["b" + gate] = np.zeros(d_hidden)
    return p


def _init_lstm_cell(rng: np.random.Generator, d_in: int, d_hidden: int) -> LSTMParams:
    s = 0.1
    rn = rng.standard_normal
    p: LSTMParams = {}
    for gate in ("i", "f", "g", "o"):
        p["W" + gate] = s * rn((d_in, d_hidden))
        p["U" + gate] = s * rn((d_hidden, d_hidden))
        p["b" + gate] = np.zeros(d_hidden)
    return p


def _init_recurrent_lm(
    cell: RNNParams, rng: np.random.Generator, vocab: int, d_model: int
) -> dict[str, Array]:
    # Wrap a cell-weight dict with an embedding and an output head.
    s = 0.1
    rn = rng.standard_normal
    return {
        "emb": s * rn((vocab, d_model)),
        **cast(dict[str, Array], cell),
        "head_w": s * rn((d_model, vocab)),
        "head_b": np.zeros(vocab),
    }


def _init_rnn(
    rng: np.random.Generator, vocab: int, d_model: int = 8
) -> dict[str, Array]:
    return _init_recurrent_lm(
        _init_rnn_cell(rng, d_model, d_model), rng, vocab, d_model
    )


def _init_gru(
    rng: np.random.Generator, vocab: int, d_model: int = 8
) -> dict[str, Array]:
    return _init_recurrent_lm(
        _init_gru_cell(rng, d_model, d_model), rng, vocab, d_model
    )


def _init_lstm(
    rng: np.random.Generator, vocab: int, d_model: int = 8
) -> dict[str, Array]:
    return _init_recurrent_lm(
        _init_lstm_cell(rng, d_model, d_model), rng, vocab, d_model
    )


def _next_char_ce(logits: Tensor) -> Operand:
    # Mean next-char cross-entropy against ``_CHAR_OH[1:]`` (predict 1..T from 0..T-1).
    targets = _CHAR_OH[1:]
    return -np.mean(np.sum(targets * np.log(softmax_last(logits) + 1e-12), axis=-1))


def rnn_loss(p: RNNParams) -> Operand:
    return _next_char_ce(rnn_lm(_CHAR_OH[:-1], p))


def gru_loss(p: GRUParams) -> Operand:
    return _next_char_ce(gru_lm(_CHAR_OH[:-1], p))


def lstm_loss(p: LSTMParams) -> Operand:
    return _next_char_ce(lstm_lm(_CHAR_OH[:-1], p))


def _recurrent_accuracy(logits: Tensor) -> float:
    preds = np.argmax(np.asarray(logits), axis=-1)
    return float(np.mean(preds == _CHAR_IDS[1:]))


def _rnn_accuracy(p: RNNParams) -> float:
    return _recurrent_accuracy(rnn_lm(_CHAR_OH[:-1], p))


def _gru_accuracy(p: GRUParams) -> float:
    return _recurrent_accuracy(gru_lm(_CHAR_OH[:-1], p))


def _lstm_accuracy(p: LSTMParams) -> float:
    return _recurrent_accuracy(lstm_lm(_CHAR_OH[:-1], p))


def sin_sq(x: Tensor) -> Operand:
    return np.sum(np.sin(x * x))

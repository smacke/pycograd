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
    # on demand when ``logistic_loss`` calls it.
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
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


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
    z = z - np.max(z, axis=-1, keepdims=True)  # subtract max for numerical stability
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


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


def sin_sq(x: Tensor) -> Operand:
    return np.sum(np.sin(x * x))

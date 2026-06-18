# -*- coding: utf-8 -*-
"""Train the demo models from scratch by gradient descent.

Run with ``python -m pycograd.examples``: logistic regression, a 2-layer MLP (both
positional and dict-pytree params), an MLP with LayerNorm + Dropout, and a
single-head Transformer encoder block -- each trained on synthetic data, with a
final finite-difference-style sanity check on ``sum(sin(x*x))``.
"""
from __future__ import annotations

import logging
from typing import cast

import numpy as np

from pycograd import (
    Param,
    ParamDict,
    frozen,
    gradient_descent,
    params,
    sgd_update,
    value_and_grad,
)
from pycograd._typing import Array
from pycograd.examples.models import (
    _accuracy,
    _deep_accuracy,
    _init_deep,
    _init_mlp,
    _init_mlp_tree,
    _init_transformer,
    _mlp_accuracy,
    _mlp_tree_accuracy,
    _transformer_accuracy,
    deep_loss,
    logistic_loss,
    logistic_param_loss,
    mlp_loss,
    mlp_tree_loss,
    sin_sq,
    transformer_loss,
)

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    (w, b), history = gradient_descent(
        logistic_loss, (np.zeros(2), 0.0), lr=0.5, steps=200
    )
    logger.info(
        "logistic regression: loss %.4f -> %.4f over %d steps",
        history[0],
        history[-1],
        len(history),
    )
    logger.info("  learned w=%s b=%.3f", np.round(w, 3), b)
    logger.info("  train accuracy: %.3f", _accuracy(w, b))

    # Same model via the declarative params(...) builder, with the bias frozen:
    # SGD trains w and leaves b pinned at its initial value. Params read by
    # attribute (model.w) as well as by key (model["w"]).
    lr_model: ParamDict = params(w=np.zeros(2), b=frozen(0.0))
    lr_vg = value_and_grad(logistic_param_loss)
    for _step in range(200):
        _loss, (g,) = lr_vg(lr_model)
        lr_model = cast(ParamDict, sgd_update(lr_model, g, lr=0.5))
    lr_w = cast(Param, lr_model.w).value
    lr_b = cast(Param, lr_model.b).value
    logger.info(
        "logistic regression via params(...) with frozen bias: acc %.3f, b held at %g",
        _accuracy(lr_w, lr_b),
        float(lr_b),
    )

    mlp_params = _init_mlp(np.random.default_rng(1))
    mlp_params, mlp_hist = gradient_descent(mlp_loss, mlp_params, lr=0.5, steps=300)
    logger.info(
        "2-layer MLP (relu + softmax), 3 classes: loss %.4f -> %.4f over %d steps",
        mlp_hist[0],
        mlp_hist[-1],
        len(mlp_hist),
    )
    logger.info("  train accuracy: %.3f", _mlp_accuracy(mlp_params))

    # Same MLP, parameters as a nested-dict pytree; SGD via tree_map (sgd_update).
    tree_params = _init_mlp_tree(np.random.default_rng(1))
    vg = value_and_grad(mlp_tree_loss)
    tree_hist: list[float] = []
    for _ in range(300):
        loss, (tree_grads,) = vg(tree_params)
        tree_hist.append(float(loss))
        tree_params = sgd_update(tree_params, tree_grads, lr=0.5)
    logger.info(
        "Same MLP, dict-pytree params (tree_map SGD): loss %.4f -> %.4f over %d steps",
        tree_hist[0],
        tree_hist[-1],
        len(tree_hist),
    )
    logger.info("  train accuracy: %.3f", _mlp_tree_accuracy(tree_params))

    deep_params = _init_deep(np.random.default_rng(2))
    deep_params, deep_hist = gradient_descent(deep_loss, deep_params, lr=0.3, steps=400)
    logger.info(
        "MLP + LayerNorm + Dropout, 3 classes: loss %.4f -> %.4f over %d steps",
        deep_hist[0],
        deep_hist[-1],
        len(deep_hist),
    )
    logger.info("  train accuracy: %.3f", _deep_accuracy(deep_params))

    tparams = _init_transformer(np.random.default_rng(4))
    tparams, t_hist = gradient_descent(transformer_loss, tparams, lr=0.2, steps=300)
    logger.info(
        "Transformer encoder block, 2-class sequences: loss %.4f -> %.4f over %d steps",
        t_hist[0],
        t_hist[-1],
        len(t_hist),
    )
    logger.info("  train accuracy: %.3f", _transformer_accuracy(tparams))

    xv = np.array([0.5, 1.0, 1.5])
    val, (g,) = value_and_grad(sin_sq)(xv)
    logger.info("sum(sin(x*x)):")
    logger.info("  autodiff grad = %s", np.round(cast(Array, g), 4))
    logger.info("  analytic grad = %s", np.round(2 * xv * np.cos(xv * xv), 4))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Shared training boilerplate for the notebook demos.

The demos all drive the ambient-weights DSL the same way: build a ``params{...}``
block, write a ``$ |> ...`` forward, then repeatedly ``weights.grad(objective)`` and
step. :func:`train` packages that loop once (in-place SGD for a float ``lr``, or a
stateful :class:`~pycograd.optimizers.Optimizer` such as ``Adam``), and forwards
``backend`` / ``jit`` so the very same loop trains on torch/jax/tf/mps. :func:`accuracy`
is the matching argmax-vs-labels score. Importing these keeps the notebooks focused on
the model, not the plumbing.
"""
from __future__ import annotations

from typing import Callable, Optional, Union, cast

import numpy as np

from pycograd._typing import Array
from pycograd.optimizers import Optimizer
from pycograd.params import Param, ParamDict
from pycograd.tree import PyTree


def train(
    weights: ParamDict,
    objective: Callable[[], PyTree],
    steps: int,
    opt: Union[float, Optimizer],
    *,
    backend: Optional[str] = None,
    jit: bool = False,
) -> tuple[float, float]:
    """Run ``steps`` updates of ``objective`` against ``weights`` and return
    ``(first_loss, last_loss)``.

    ``opt`` is either a float learning rate (plain in-place SGD via
    :meth:`ParamDict.step`) or an :class:`~pycograd.optimizers.Optimizer` instance, whose
    returned values are copied back into the live ``Param`` leaves so the ambient proxies
    see the update on the next forward. ``backend`` / ``jit`` are forwarded to
    :meth:`ParamDict.grad`, so the same loop trains on a framework's own autodiff.
    """
    first: Optional[float] = None
    last = 0.0
    for _ in range(steps):
        value, grads = weights.grad(objective, backend=backend, jit=jit)
        last = float(value)
        if first is None:
            first = last
        if isinstance(opt, Optimizer):
            updated = cast(ParamDict, opt.step(weights, grads))
            for k in weights:
                leaf = weights[k]
                if isinstance(leaf, Param):
                    leaf.value = cast(Param, updated[k]).value
        else:
            weights.step(grads, opt)  # plain in-place SGD on the numpy weights
    return (last if first is None else first), last


def accuracy(logits: Array, labels: Array, axis: int = -1) -> float:
    """Mean ``argmax(logits) == labels`` accuracy, as a float."""
    preds = np.argmax(np.asarray(logits), axis=axis)
    return float(np.mean(preds == labels))

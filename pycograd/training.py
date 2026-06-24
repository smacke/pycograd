# -*- coding: utf-8 -*-
"""Training-loop helpers for the ambient-weights DSL.

The ambient-weights workflow is always the same: build a ``params{...}`` block, write a
``$ |> ...`` forward, then repeatedly ``weights.grad(objective)`` and step. The helpers
here package that loop once -- in-place SGD for a float ``lr``, or a stateful
:class:`~pycograd.optimizers.Optimizer` such as ``Adam`` -- so a model stays focused on the
forward, not the plumbing:

* :func:`train` -- a *full-batch* loop: it runs a no-arg ``objective`` for a fixed number of
  steps, and forwards ``backend`` / ``jit`` so the very same loop trains on torch/jax/tf/mps.
* :func:`fit` -- a *minibatch* loop: it slices arrays into minibatches (via
  :func:`pycograd.data.batches`) and feeds each through a *parameterized* objective (a
  callable of one minibatch), so stochastic gradient descent needs no hand-rolled sampling.
* :func:`accuracy` -- the matching argmax-vs-labels score.

All three are top-level exports (``from pycograd import train, fit, accuracy``).
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Union

import numpy as np

from pycograd._typing import Array
from pycograd.data import Batch, DataLoader
from pycograd.optimizers import Optimizer
from pycograd.params import ParamDict
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
    """Run ``steps`` full-batch updates of ``objective`` against ``weights`` and return
    ``(first_loss, last_loss)``.

    ``objective`` is a no-arg callable that reads the weights (and its data) by closure;
    ``opt`` is either a float learning rate or an :class:`~pycograd.optimizers.Optimizer`
    instance -- both applied in place via :meth:`ParamDict.step`. ``backend`` / ``jit`` are
    forwarded to :meth:`ParamDict.grad`, so the same loop trains on a framework's own
    autodiff. For *minibatch* SGD over a dataset, use :func:`fit` instead.
    """
    first: Optional[float] = None
    last = 0.0
    for _ in range(steps):
        value, grads = weights.grad(objective, backend=backend, jit=jit)
        last = float(value)
        if first is None:
            first = last
        weights.step(grads, opt)
    return (last if first is None else first), last


def fit(
    weights: ParamDict,
    objective: Callable[..., PyTree],
    *arrays: Array,
    epochs: int,
    opt: Union[float, Optimizer],
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    rng: Optional[np.random.Generator] = None,
    backend: Optional[str] = None,
    on_epoch: Optional[Callable[[int, float], None]] = None,
) -> list[float]:
    """Minibatch (stochastic) gradient descent over ``weights`` for ``epochs`` epochs.

    ``objective`` is a *parameterized* loss -- a callable of one minibatch. A minibatch
    of several arrays arrives as a tuple, so a pipe objective splats it with ``*|>``:
    ``loss = $ *|> ($x, $y) *|> cross_entropy(logits($x), $y)`` (a single-array dataset
    arrives as a bare array, so the objective is just a ``$``-seeded pipe). Each step, a
    minibatch is sliced from ``arrays`` (one shared index keeps inputs and labels aligned) and fed
    through :meth:`ParamDict.grad`; the resulting gradients are applied in place via
    :meth:`ParamDict.step` (float ``lr`` or an :class:`~pycograd.optimizers.Optimizer`).
    This replaces hand-rolled in-objective sampling: sampling is :func:`pycograd.data.batches`'
    job, so no ``rng.choice`` / index threading is needed.

    Pass the data either as loose ``arrays`` plus a ``batch_size`` (a fresh shuffled
    :class:`~pycograd.data.DataLoader` is built internally) **or** as a single prebuilt
    :class:`~pycograd.data.DataLoader` (for ``drop_last`` / a custom rng), not both.
    ``shuffle`` / ``rng`` configure the internal loader. ``backend`` is forwarded to
    :meth:`ParamDict.grad` so a minibatch loop can train on a framework's own autodiff;
    ``jit`` is intentionally unavailable here -- it caches the trace (and the minibatch) and
    would silently reuse stale data, so use full-batch :func:`train` for ``jit``.

    Returns the per-epoch mean training loss (size-weighted, so a short final batch counts
    correctly), and calls ``on_epoch(epoch, mean_loss)`` after each epoch if given -- a
    natural hook for logging or a held-out :func:`accuracy` check. Reusing the same
    ``Optimizer`` across ``fit`` calls resumes its state (momentum / Adam moments / step
    count); pass a fresh optimizer to restart.
    """
    if len(arrays) == 1 and isinstance(arrays[0], DataLoader):
        if batch_size is not None:
            raise ValueError(
                "fit: pass either a prebuilt DataLoader or batch_size, not both"
            )
        loader: Iterable[Batch] = arrays[0]
    else:
        if batch_size is None:
            raise ValueError(
                "fit: batch_size is required unless a DataLoader is passed"
            )
        loader = DataLoader(*arrays, batch_size=batch_size, shuffle=shuffle, rng=rng)

    history: list[float] = []
    for epoch in range(epochs):
        total = 0.0
        count = 0
        for batch in loader:  # a DataLoader re-shuffles on each pass (one epoch)
            # ``batch`` is a single array (one input) or a tuple of aligned arrays
            # (inputs + labels). It is fed to ``objective`` *as one argument* -- a
            # multi-array objective splats it with ``*|>`` (e.g. ``$ *|> loss(...)``).
            first = batch[0] if isinstance(batch, tuple) else batch
            n = len(first)
            value, grads = weights.grad(objective, batch, backend=backend)
            weights.step(grads, opt)
            total += float(value) * n
            count += n
        mean = total / count if count else 0.0
        history.append(mean)
        if on_epoch is not None:
            on_epoch(epoch, mean)
    return history


def accuracy(logits: Array, labels: Array, axis: int = -1) -> float:
    """Mean ``argmax(logits) == labels`` accuracy, as a float."""
    preds = np.argmax(np.asarray(logits), axis=axis)
    return float(np.mean(preds == labels))

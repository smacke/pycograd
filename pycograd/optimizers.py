# -*- coding: utf-8 -*-
"""Stateful optimizers over parameter pytrees.

``value_and_grad`` hands back gradients with the same pytree structure as the
parameters; an :class:`Optimizer` consumes a ``(params, grads)`` pair and returns
updated params, exactly like :func:`pycograd.tree.sgd_update` -- but carrying its
own state (momentum buffers, Adam moments, a step counter) so it can implement
richer update rules.

    opt = Adam(lr=1e-3)
    for Xb, yb in batches(X, y, batch_size=32, shuffle=True, rng=rng):
        loss, (g,) = value_and_grad(loss_fn)(params, Xb, yb)
        params = opt.step(params, g)

State lives in the optimizer instance as a flat list aligned to
``tree_leaves(params)``, built lazily on the first :meth:`Optimizer.step`. Leaves
are stepped leafwise with the same ``Param``-wrapper / ``None``-gradient handling
as :func:`pycograd.tree._sgd_step`: a frozen ``Param`` (or any leaf whose gradient
is ``None``) is carried through untouched, and trainable ``Param`` wrappers are
preserved across the update.

There is deliberately no ``zero_grad``: gradients are recomputed fresh by each
``value_and_grad`` call rather than accumulated in the params, so there is no
buffer to clear between steps.

``lr`` may be a float or a schedule -- any ``callable(step) -> float`` (see
:func:`step_decay` / :func:`cosine_decay`); it is evaluated once per step with the
optimizer's 1-based step count.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable, Union

from pycograd._typing import Array
from pycograd.dtypes import current_dtype
from pycograd.params import Param
from pycograd.tensor import _is_numeric, _xp
from pycograd.tree import (
    Leaf,
    PyTree,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_unflatten,
)

LearningRate = Union[float, Callable[[int], float]]
# Per-leaf optimizer state -- shape is private to each optimizer (a velocity
# buffer for SGD, an ``[m, v]`` pair for Adam), so it is left untyped here.
State = Any


def _leaf_value(leaf: Leaf) -> Array:
    """The numeric value array of a parameter leaf (unwrapping a ``Param``).

    Coerced with the active array module so optimizer state created from these values
    (momentum / Adam moment buffers) lives on the same device as the parameters. The
    leaf's own dtype is preserved -- a float32/bfloat16 parameter keeps its precision
    through the step, and its moment buffers (``zeros_like(value)``) match it.
    """
    value = leaf.value if isinstance(leaf, Param) else leaf
    arr = _xp().asarray(value)
    # Keep an existing float precision (f64/f32/f16/bf16) or complex (c64/c128); promote a
    # non-float/complex leaf (a bare int) to the working dtype so the update is well typed.
    return arr if arr.dtype.kind in "fc" else arr.astype(current_dtype())


class Optimizer:
    """Base class: leafwise parameter updates with per-leaf state.

    Subclasses override :meth:`_init_state` (the per-leaf state created lazily on
    the first step) and :meth:`_update_leaf` (the numeric update rule, which may
    mutate its state buffers in place). ``step`` handles flattening, the step
    counter, learning-rate scheduling, and ``Param``/``None`` bookkeeping.
    """

    def __init__(self, lr: LearningRate) -> None:
        self.lr: LearningRate = lr
        self.t: int = 0  # 1-based step count after the first step()
        self._state: list[State] | None = None

    # --- overridden per optimizer -----------------------------------------
    def _init_state(self, value: Array) -> State:
        """Per-leaf state for a trainable leaf of the given value array."""
        return None

    def _update_leaf(self, value: Array, grad: Array, state: State, lr: float) -> Array:
        raise NotImplementedError

    # --- the shared driver -------------------------------------------------
    def _trainable(self, leaf: Leaf, grad: Leaf) -> bool:
        if grad is None:
            return False
        if isinstance(leaf, Param):
            return leaf.trainable
        return _is_numeric(leaf)

    def _apply(self, leaf: Leaf, grad: Leaf, state: State, lr: float) -> Leaf:
        if not self._trainable(leaf, grad):
            return leaf  # frozen / no gradient / non-numeric: held fixed
        value = _leaf_value(leaf)
        # Match the gradient to the parameter's dtype so the update stays in precision.
        new_value = self._update_leaf(
            value, _xp().asarray(grad, dtype=value.dtype), state, lr
        )
        if isinstance(leaf, Param):
            return replace(leaf, value=new_value)
        return new_value

    def step(self, params: PyTree, grads: PyTree) -> PyTree:
        """Return ``params`` advanced one step using ``grads`` (same structure).

        ``grads`` is the gradient pytree ``value_and_grad`` returns for ``params``
        (with ``None`` at frozen/non-numeric leaves). The optimizer's state is
        built on the first call and reused thereafter, so the same param structure
        must be passed every step.
        """
        leaves, treedef = tree_flatten(params)
        grad_leaves, _ = tree_flatten(grads)
        if len(leaves) != len(grad_leaves):
            raise ValueError(
                "optimizer: params and grads must have the same pytree structure "
                f"({len(leaves)} vs {len(grad_leaves)} leaves)"
            )
        if self._state is None:
            self._state = [
                self._init_state(_leaf_value(p)) if self._trainable(p, g) else None
                for p, g in zip(leaves, grad_leaves)
            ]
        self.t += 1
        lr = self.lr(self.t) if callable(self.lr) else self.lr
        new_leaves = [
            self._apply(p, g, s, lr)
            for p, g, s in zip(leaves, grad_leaves, self._state)
        ]
        return tree_unflatten(treedef, new_leaves)


class SGD(Optimizer):
    """Stochastic gradient descent, with optional momentum and weight decay.

    ``momentum=0`` (the default) is plain ``p <- p - lr * g``. With momentum the
    per-leaf state is a velocity buffer ``v <- momentum*v + g``; ``nesterov`` looks
    ahead with ``g + momentum*v``. ``weight_decay`` adds an L2 term ``wd * p`` to
    the gradient before the step.
    """

    def __init__(
        self,
        lr: LearningRate = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if nesterov and momentum <= 0:
            raise ValueError("SGD: nesterov momentum requires momentum > 0")
        super().__init__(lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def _init_state(self, value: Array) -> State:
        return _xp().zeros_like(value) if self.momentum else None

    def _update_leaf(self, value: Array, grad: Array, state: State, lr: float) -> Array:
        g = grad
        if self.weight_decay:
            g = g + self.weight_decay * value
        if self.momentum:
            v = state  # in-place velocity buffer: v <- momentum*v + g
            v *= self.momentum
            v += g
            step = g + self.momentum * v if self.nesterov else v
            return value - lr * step
        return value - lr * g


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2014), with optional (coupled, L2) weight decay.

    Per-leaf state holds the first/second moment buffers ``m``, ``v``; both are
    bias-corrected by the global step count. ``weight_decay`` is added to the
    gradient (L2). For *decoupled* weight decay, use :class:`AdamW`.
    """

    _decoupled = False

    def __init__(
        self,
        lr: LearningRate = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def _init_state(self, value: Array) -> State:
        xp = _xp()
        return [xp.zeros_like(value), xp.zeros_like(value)]  # [m, v]

    def _update_leaf(self, value: Array, grad: Array, state: State, lr: float) -> Array:
        b1, b2 = self.betas
        m, v = state
        wd = self.weight_decay
        g = grad + wd * value if (wd and not self._decoupled) else grad
        m *= b1
        m += (1 - b1) * g  # m <- b1*m + (1-b1)*g  (in place)
        v *= b2
        # Second moment is ``|g|^2``: for complex grads ``g*conj(g)`` (real-valued) rather
        # than the complex square ``g*g``, so the adaptive scale ``sqrt(v_hat)`` is real.
        g2 = g * _xp().conj(g) if g.dtype.kind == "c" else g * g
        v += (1 - b2) * g2  # v <- b2*v + (1-b2)*|g|^2  (in place)
        m_hat = m / (1 - b1**self.t)
        v_hat = v / (1 - b2**self.t)
        if wd and self._decoupled:  # AdamW: decay the weight directly
            value = value - lr * wd * value
        return value - lr * m_hat / (_xp().sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Adam with *decoupled* weight decay (Loshchilov & Hutter, 2017).

    The decay term ``lr * weight_decay * p`` is applied to the parameter directly
    rather than folded into the gradient, so it does not interact with the adaptive
    per-coordinate scaling.
    """

    _decoupled = True


# ---------------------------------------------------------------------------
# Gradient clipping.
# ---------------------------------------------------------------------------
def clip_grad_norm(grads: PyTree, max_norm: float) -> PyTree:
    """Rescale a gradient pytree so its global L2 norm is at most ``max_norm``.

    The norm is taken over all leaves jointly (``None`` leaves -- frozen params --
    are skipped). Below the threshold the tree is returned unchanged.
    """
    xp = _xp()
    # The norm is accumulated in float64 for stability regardless of the grads' dtype...
    total = math.sqrt(
        sum(
            float(xp.sum(xp.abs(xp.asarray(g)) ** 2))
            for g in tree_leaves(grads)
            if g is not None
        )
    )
    if total <= max_norm or total == 0.0:
        return grads
    scale = max_norm / total
    # ...but the rescaled grads preserve each leaf's own precision: multiplying an array
    # by the Python scalar ``scale`` keeps the array's (float) dtype.
    return tree_map(lambda g: None if g is None else xp.asarray(g) * scale, grads)


# ---------------------------------------------------------------------------
# Learning-rate schedules: ``callable(step) -> lr`` over a 1-based step count.
# ---------------------------------------------------------------------------
def constant_lr(lr: float) -> Callable[[int], float]:
    """A flat schedule -- exactly equivalent to passing the float ``lr``."""
    return lambda step: lr


def step_decay(lr0: float, factor: float, every: int) -> Callable[[int], float]:
    """Multiply the rate by ``factor`` every ``every`` steps (staircase decay)."""
    return lambda step: lr0 * factor ** ((step - 1) // every)


def cosine_decay(
    lr0: float, total_steps: int, min_lr: float = 0.0
) -> Callable[[int], float]:
    """Cosine anneal from ``lr0`` down to ``min_lr`` over ``total_steps`` steps.

    The rate reaches ``min_lr`` at ``step == total_steps`` and stays there after.
    """

    def schedule(step: int) -> float:
        progress = min(step, total_steps) / total_steps
        return min_lr + 0.5 * (lr0 - min_lr) * (1 + math.cos(math.pi * progress))

    return schedule

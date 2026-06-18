# -*- coding: utf-8 -*-
"""The high-level differentiation API.

``value_and_grad`` / ``grad`` wrap a function so it returns gradients with the same
pytree structure as its arguments; ``gradient_descent`` is a plain-SGD training
loop. Argument leaves are lifted onto the tape (``_wrap_leaf``), the function is run
under the pyccolo tracer (via :mod:`pycograd.tracer`), and ``Var.backward`` walks
the graph.
"""
from __future__ import annotations

from typing import Callable, cast

import numpy as np

from pycograd._typing import Array, ArrayLike, Operand
from pycograd.params import Param, _TieRef
from pycograd.tensor import Var, _is_numeric
from pycograd.tracer import _INSTRUMENTED, _make_runner
from pycograd.tree import (
    Leaf,
    PyTree,
    TreeDef,
    tree_flatten,
    tree_leaves,
    tree_unflatten,
)


def _dup_param_msg(p: Param) -> str:
    where = f" (declared in params block {p.origin!r})" if p.origin is not None else ""
    return (
        f"autodiff: a parameter{where} appears more than once across the "
        "differentiated arguments; a weight must have a single owner -- declare it "
        "once, and use tied(key, value) to share one weight across positions"
    )


def _check_param_ownership(args: tuple[PyTree, ...]) -> None:
    """Reject a ``params{...}``-declared weight that is *also* handed in as a
    separate ``Param`` leaf.

    A weight with two owners would be lifted onto the tape -- and stepped by the
    optimizer -- through both paths. We flag the same ``Param`` object reused in
    two leaf positions, and a distinct ``Param`` aliasing a block-owned weight's
    value array. (Sharing one weight across positions is what ``tied`` is for.)
    """
    seen_obj: set[int] = set()
    val_owner: dict[int, Param] = {}
    for a in args:
        for leaf in tree_leaves(a):
            if not isinstance(leaf, Param):
                continue
            if id(leaf) in seen_obj and leaf.origin is not None:
                raise ValueError(_dup_param_msg(leaf))
            seen_obj.add(id(leaf))
            other = val_owner.get(id(leaf.value))
            tied_pair = (
                other is not None and leaf.tie is not None and leaf.tie == other.tie
            )
            if (
                other is not None
                and other is not leaf
                and not tied_pair  # tied params legitimately share one weight
                and (leaf.origin is not None or other.origin is not None)
            ):
                raise ValueError(
                    _dup_param_msg(leaf if leaf.origin is not None else other)
                )
            val_owner.setdefault(id(leaf.value), leaf)


def _wrap_leaf(leaf: Leaf, tie_vars: dict[object, Var]) -> tuple[Var | None, Leaf]:
    """Lift one argument leaf onto the tape.

    Returns ``(var, call_value)``: ``var`` is the tape node whose ``.grad`` becomes
    this leaf's gradient (``None`` when the leaf is frozen or non-numeric, so its
    gradient comes back ``None``); ``call_value`` is what to pass into the
    differentiated function in this leaf's place. Trainable ``Param``s and bare
    numerics become fresh ``Var``s; a frozen ``Param`` passes its raw value
    through; ``Param``s sharing a ``tie`` key share a single ``Var``.
    """
    if isinstance(leaf, _TieRef):
        raise ValueError(
            "autodiff: tied[...] is only meaningful inside params(...), where it "
            "references a sibling parameter; it reached value_and_grad unresolved"
        )
    if isinstance(leaf, Param):
        if not leaf.trainable:
            return None, leaf.value
        if leaf.tie is not None:
            shared = tie_vars.get(leaf.tie)
            if shared is None:
                shared = Var(leaf.value)
                tie_vars[leaf.tie] = shared
            return shared, shared
        v = Var(leaf.value)
        return v, v
    if _is_numeric(leaf):
        v = Var(cast(ArrayLike, leaf))
        return v, v
    return None, leaf


def value_and_grad(
    f: Callable[..., object],
) -> Callable[..., tuple[Array, tuple[PyTree, ...]]]:
    """Wrap ``f`` so that calling it returns ``(value, grads)``.

    ``grads`` is a tuple with one entry per positional argument, holding the
    gradient of the (scalar) output w.r.t. that argument. Each argument may be a
    pytree (e.g. a dict of weights); its gradient comes back as a matching pytree,
    with ``None`` at any non-numeric or frozen leaf. A bare array/scalar is just a
    pytree with one leaf, so it yields a bare gradient (backward compatible).
    Leaves may be ``Param``s to opt into freezing (``frozen``) or tying (``tied``).
    ``f`` may be an ordinary function or a pipescript ``|>`` pipe lambda (run it
    under ``PipelineTracer``).
    """
    runner = _INSTRUMENTED.get(f)
    if runner is None:
        runner = _make_runner(f)
        _INSTRUMENTED[f] = runner

    def wrapped(*args: PyTree) -> tuple[Array, tuple[PyTree, ...]]:
        call_args: list[PyTree] = []
        per_arg: list[tuple[TreeDef, list[tuple[Leaf, Var | None]]]] = []
        _check_param_ownership(args)
        tie_vars: dict[object, Var] = {}
        for a in args:
            leaves, treedef = tree_flatten(a)
            info: list[tuple[Leaf, Var | None]] = []
            wrapped_leaves: list[Leaf] = []
            for leaf in leaves:
                var, call_value = _wrap_leaf(leaf, tie_vars)
                info.append((leaf, var))
                wrapped_leaves.append(call_value)
            call_args.append(tree_unflatten(treedef, wrapped_leaves))
            per_arg.append((treedef, info))

        out = runner(*call_args)
        if isinstance(out, Var):
            out.backward()  # otherwise each leaf's grad stays at its init zeros
            value: Array = out.value
        else:
            value = np.asarray(out, dtype=float)

        grads = tuple(
            tree_unflatten(
                treedef,
                [None if v is None else _match_arg(orig, v.grad) for orig, v in info],
            )
            for treedef, info in per_arg
        )
        return value, grads

    return wrapped


def _match_arg(orig: Leaf, grad: Array) -> Operand:
    value = orig.value if isinstance(orig, Param) else orig
    return grad if isinstance(value, np.ndarray) else float(grad)


def grad(f: Callable[..., object]) -> Callable[..., tuple[PyTree, ...]]:
    """Return a function computing just the gradient tuple of ``f`` (one entry per
    argument; each matches that argument's pytree structure)."""
    vg = value_and_grad(f)
    return lambda *args: vg(*args)[1]


def gradient_descent(
    loss_fn: Callable[..., object],
    init_params: tuple[ArrayLike, ...],
    lr: float = 0.1,
    steps: int = 100,
) -> tuple[tuple[Array, ...], list[float]]:
    """Minimize ``loss_fn(*params)`` by gradient descent; return (params, history).

    Each step replaces ``p`` with ``p - lr * grad``; since a number/array minus an
    array is always an array, the returned params are ``Array``s (a scalar init
    like ``b=0.0`` is promoted on the first update). A positional ``frozen`` param
    is held fixed (its gradient is ``None``); ``Param`` wrappers are preserved.
    """
    from pycograd.tree import _sgd_step

    vg = value_and_grad(loss_fn)
    params: list[Leaf] = [cast(Leaf, p) for p in init_params]
    history: list[float] = []
    for _ in range(steps):
        loss, grads = vg(*params)
        history.append(float(loss))
        params = [_sgd_step(p, cast(Leaf, g), lr) for p, g in zip(params, grads)]
    return tuple(cast("list[Array]", params)), history

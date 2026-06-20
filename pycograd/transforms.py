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

from pycograd import ops
from pycograd._typing import Array, ArrayLike, Operand
from pycograd.backends import activate, current_backend, get_backend
from pycograd.batching import BatchedArray
from pycograd.params import Param, _TieRef
from pycograd.tensor import Var, _is_array, _is_numeric, _lift, _value, _xp
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
            value = _xp().asarray(out, dtype=float)

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
    # Array-valued argument -> array gradient (on the active backend); a bare Python
    # scalar -> a Python float (``float`` of a device scalar pulls it back to host).
    return grad if _is_array(value) else float(grad)


def grad(f: Callable[..., object]) -> Callable[..., tuple[PyTree, ...]]:
    """Return a function computing just the gradient tuple of ``f`` (one entry per
    argument; each matches that argument's pytree structure)."""
    vg = value_and_grad(f)
    return lambda *args: vg(*args)[1]


def _moveaxis_perm(src: int, dst: int, ndim: int) -> tuple:
    order = list(range(ndim))
    order.insert(dst, order.pop(src))
    return tuple(order)


def _to_front(v: object, ax: int) -> object:
    """Move axis ``ax`` to 0, staying on the tape for a ``Var`` (grad-aware)."""
    if ax == 0:
        return v
    if isinstance(v, Var):
        return ops.d_transpose(v, _moveaxis_perm(ax, 0, v.value.ndim))
    return np.moveaxis(np.asarray(v), ax, 0)


def _finish_vmap_leaf(leaf: object, batch: int, out_axis: int, nested: bool) -> object:
    """Produce one vmap output leaf. *Nested* under another transform (inputs were tape
    ``Var``s) -> keep the result on the tape (return a ``Var``). *Top level* -> return a
    materialized array, broadcasting an unbatched (batch-independent) output."""
    if nested:
        if isinstance(leaf, BatchedArray):
            v = leaf.inner
            if out_axis != 0 and isinstance(v, Var):
                v = ops.d_transpose(v, _moveaxis_perm(0, out_axis, v.value.ndim))
            return v
        return leaf  # batch-independent: pass through unchanged
    if isinstance(leaf, BatchedArray):
        arr = np.asarray(_value(cast(Operand, leaf.inner)))
        return np.moveaxis(arr, 0, out_axis)
    arr = np.asarray(_value(cast(Operand, leaf)))
    arr = np.broadcast_to(arr, (batch,) + arr.shape)
    return np.moveaxis(arr, 0, out_axis)


def vmap(
    f: Callable[..., object], in_axes: object = 0, out_axes: int = 0
) -> Callable[..., object]:
    """Vectorize ``f`` over a batch axis: ``vmap(f)(xs)`` applies ``f`` to each slice of
    ``xs`` along ``in_axes`` and stacks the results along ``out_axes`` -- but vectorized
    (one batched pass), not looped.

    ``in_axes`` is an int (the mapped axis of every argument), ``None`` (an argument
    shared across the batch -- e.g. a weight), or a tuple with one such entry per
    positional argument. Non-array arguments pass through unbatched.

    Composes with :func:`grad`/:func:`value_and_grad` (``grad(vmap(f))``); for per-sample
    gradients of the *data* argument see :func:`per_example_grad`. Not yet supported
    (raise ``NotImplementedError``): nesting ``vmap(vmap(...))`` and per-sample gradients
    of a *shared* parameter -- both need a trace-level interpreter stack.
    """
    runner = _INSTRUMENTED.get(f)
    if runner is None:
        runner = _make_runner(f)
        _INSTRUMENTED[f] = runner

    def wrapped(*args: PyTree) -> object:
        # A batch backend already active means we are inside another vmap; nesting needs
        # a trace-level stack we don't have yet. (Checked first, using only module-level
        # names, so it survives the closure-unaware re-instrumentation of this wrapper.)
        if current_backend().name == "batch":
            raise NotImplementedError(
                "vmap: nested vmap(vmap(...)) is not supported yet"
            )
        axes = in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(args)
        call_args: list[PyTree] = []
        batch = -1
        nested = False
        for a, ax in zip(args, cast("tuple", axes)):
            leaves, treedef = tree_flatten(a)
            new_leaves: list[Leaf] = []
            for leaf in leaves:
                if isinstance(leaf, BatchedArray):
                    raise NotImplementedError(
                        "vmap: nested vmap(vmap(...)) is not supported yet"
                    )
                if ax is None or not (_is_array(leaf) or isinstance(leaf, Var)):
                    new_leaves.append(leaf)  # shared / non-array: unbatched
                    continue
                if isinstance(leaf, Var):
                    nested = (
                        True  # already on a tape (we're inside grad/value_and_grad)
                    )
                    v: object = _to_front(leaf, cast(int, ax))
                    batch = cast(Var, v).value.shape[0]
                else:
                    arr = np.moveaxis(np.asarray(leaf), cast(int, ax), 0)
                    batch = arr.shape[0]
                    v = Var(arr)
                new_leaves.append(cast(Leaf, BatchedArray(v)))
            call_args.append(tree_unflatten(treedef, new_leaves))
        if batch < 0:
            raise ValueError("vmap: no batched (array) argument to map over")

        with activate(get_backend("batch")):
            out = runner(*call_args)

        out_leaves, out_def = tree_flatten(cast(PyTree, out))
        finished = [
            _finish_vmap_leaf(leaf, batch, out_axes, nested) for leaf in out_leaves
        ]
        return tree_unflatten(out_def, cast("list[Leaf]", finished))

    return wrapped


def per_example_grad(
    f: Callable[..., object], in_axes: int = 0
) -> Callable[..., object]:
    """Per-sample gradients of a per-example scalar ``f`` w.r.t. its batched input.

    ``f`` maps one example to a scalar; ``per_example_grad(f)(xs)`` returns one gradient
    per example, stacked. Implemented via the sum-of-losses identity (the examples are
    independent, so ``d sum_i f(x_i) / d x = [d f(x_i)/d x_i]``) -- a single batched
    forward and one backward, no Python loop.
    """
    runner = _INSTRUMENTED.get(f)
    if runner is None:
        runner = _make_runner(f)
        _INSTRUMENTED[f] = runner

    def wrapped(x: PyTree) -> object:
        arr = np.moveaxis(np.asarray(_value(cast(Operand, x))), in_axes, 0)
        xv = Var(arr)
        with activate(get_backend("batch")):
            out = runner(BatchedArray(xv))
        inner = out.inner if isinstance(out, BatchedArray) else out
        total = ops.d_sum(_lift(cast(Operand, inner)))
        total.backward()
        return np.moveaxis(xv.grad, 0, in_axes)

    return wrapped


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

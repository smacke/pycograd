# -*- coding: utf-8 -*-
"""The high-level differentiation API.

``value_and_grad`` / ``grad`` wrap a function so it returns gradients with the same
pytree structure as its arguments; ``gradient_descent`` is a plain-SGD training
loop. Argument leaves are lifted onto the tape (``_wrap_leaf``), the function is run
under the pyccolo tracer (via :mod:`pycograd.tracer`), and ``Var.backward`` walks
the graph.
"""
from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np

from pycograd import ops
from pycograd._typing import Array, ArrayLike, Operand
from pycograd.batching import BatchTrace, BatchTracer
from pycograd.params import Param, _TieRef
from pycograd.tensor import Var, _is_array, _is_numeric, _lift, _value, _xp
from pycograd.trace import bind, new_main
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

    # Tag so ``vmap`` can recognize ``vmap(value_and_grad(f))`` / ``vmap(grad(f))`` and
    # take the per-sample path (a single batched forward + one batched-cotangent backward),
    # and so the tracer never re-instruments this closure (it closes over ``f``/``runner``).
    wrapped._pycograd_vag_of = f  # type: ignore[attr-defined]
    wrapped._pycograd_run_directly = True  # type: ignore[attr-defined]
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

    def wrapped(*args: PyTree) -> tuple[PyTree, ...]:
        return vg(*args)[1]

    # Carry the underlying ``f`` (not ``vg``) so ``vmap(grad(f))`` reconstructs the
    # per-sample path from the original function.
    wrapped._pycograd_grad_of = f  # type: ignore[attr-defined]
    wrapped._pycograd_run_directly = True  # type: ignore[attr-defined]
    return wrapped


def _moveaxis_perm(src: int, dst: int, ndim: int) -> tuple:
    order = list(range(ndim))
    order.insert(dst, order.pop(src))
    return tuple(order)


def _mappable(leaf: object) -> bool:
    """A leaf ``vmap`` can map a batch axis over: a real array, a tape ``Var`` (nested
    under ``grad``), or a :class:`BatchTracer` (nested under an outer ``vmap``)."""
    return _is_array(leaf) or isinstance(leaf, (Var, BatchTracer))


def _move_out(v: object, src: int, dst: int) -> object:
    """Move a batched value's batch axis from ``src`` to ``dst`` on the way out, keeping
    a ``Var``/``BatchTracer`` on the tape/level (grad- and nesting-aware)."""
    if src == dst:
        return v
    if isinstance(v, Var):
        return ops.d_transpose(v, _moveaxis_perm(src, dst, v.value.ndim))
    if isinstance(v, BatchTracer):
        ndim = len(cast(tuple, v.shape)) + 1
        return bind(ops.d_transpose, v, _moveaxis_perm(src, dst, ndim))
    return np.moveaxis(np.asarray(cast(Any, v)), src, dst)


def _finish_vmap_leaf(
    leaf: object, level: int, batch: int, out_axis: int, nested: bool
) -> object:
    """Produce one ``vmap`` output leaf from this level's result.

    A :class:`BatchTracer` *at this level* carries the mapped output; its batch axis
    (``bdim``) is moved to ``out_axis`` and the physical value extracted. *Nested* under
    another transform (the result stays a ``Var``/lower-level ``BatchTracer`` so the tape
    keeps flowing); *top level* materializes to an array. A batch-independent output
    (not a tracer at this level) is broadcast over the batch (top level) or passed
    through (nested)."""
    if isinstance(leaf, BatchTracer) and leaf._trace.main.level == level:
        if leaf.bdim is None:
            # Batch-independent inside this level. Nested: hand the value down
            # unchanged. Top level: broadcast it back over the batch at ``out_axis``.
            if nested:
                return leaf.value
            arr = np.asarray(_value(cast(Operand, leaf.value)))
            return np.moveaxis(np.broadcast_to(arr, (batch,) + arr.shape), 0, out_axis)
        v = _move_out(leaf.value, leaf.bdim, out_axis)
        if nested:
            return v
        return np.asarray(_value(cast(Operand, v)))  # batch already at out_axis
    if nested:
        return leaf  # batch-independent: pass through unchanged
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

    Realized as one level of the trace-level interpreter stack (a :class:`BatchTrace`
    pushed via ``new_main``), so it *nests*: ``vmap(vmap(f))`` pushes two batch levels
    and the per-primitive rules peel one level at a time. Composes with
    :func:`grad`/:func:`value_and_grad` (``grad(vmap(f))``).

    **Per-sample gradients.** ``vmap(grad(g))`` (or ``vmap(value_and_grad(g))``) computes
    *per-sample* gradients: with ``in_axes=(0, None)`` the mapped (data) argument's
    gradient is stacked over the batch, and the *shared* parameter's gradient comes back
    per-sample as ``(B, *param.shape)`` -- ``d g(x_i, w)/dw`` for each example ``i`` --
    instead of being summed over the batch. (For just the data argument's per-sample
    gradient of a per-example scalar, :func:`per_example_grad` is the lighter entry point.)
    """
    # ``vmap(grad(g))`` / ``vmap(value_and_grad(g))``: a single batched forward plus one
    # batched-cotangent backward yields per-sample gradients (incl. of shared params).
    grad_of = getattr(f, "_pycograd_grad_of", None)
    if grad_of is not None:
        return _vmap_of_grad(grad_of, in_axes, return_value=False)
    vag_of = getattr(f, "_pycograd_vag_of", None)
    if vag_of is not None:
        return _vmap_of_grad(vag_of, in_axes, return_value=True)

    runner = _INSTRUMENTED.get(f)
    if runner is None:
        runner = _make_runner(f)
        _INSTRUMENTED[f] = runner

    def wrapped(*args: PyTree) -> object:
        axes = in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(args)
        with new_main(BatchTrace) as main:
            trace = BatchTrace(main)
            level = main.level
            call_args: list[PyTree] = []
            batch = -1
            nested = False
            for a, ax in zip(args, cast("tuple", axes)):
                leaves, treedef = tree_flatten(a)
                new_leaves: list[Leaf] = []
                for leaf in leaves:
                    if ax is None or not _mappable(leaf):
                        # A shared (unbatched) leaf that is itself on a tape/level means
                        # we are nested under ``grad`` (a ``Var``) or an outer ``vmap`` (a
                        # ``BatchTracer``): keep the output on that tape/level so a
                        # gradient wrt a *shared* operand (e.g. gathering a shared table by
                        # a per-example index) still flows.
                        if ax is None and isinstance(leaf, (Var, BatchTracer)):
                            nested = True
                        new_leaves.append(leaf)  # shared / non-array: unbatched
                        continue
                    if isinstance(leaf, (Var, BatchTracer)):
                        # Already on the tape / an outer level: keep it; the batch axis
                        # lives at ``ax`` in this leaf's *physical* layout.
                        nested = True
                        b = (
                            leaf.value.shape[ax]
                            if isinstance(leaf, Var)
                            else leaf.shape[ax]
                        )
                        batch = int(b)
                        bt = BatchTracer(trace, leaf, cast(int, ax))
                    else:
                        arr = np.asarray(leaf)
                        batch = arr.shape[cast(int, ax)]
                        bt = BatchTracer(trace, Var(arr), cast(int, ax))
                    new_leaves.append(cast(Leaf, bt))
                call_args.append(tree_unflatten(treedef, new_leaves))
            if batch < 0:
                raise ValueError("vmap: no batched (array) argument to map over")

            out = runner(*call_args)

            out_leaves, out_def = tree_flatten(cast(PyTree, out))
            finished = [
                _finish_vmap_leaf(leaf, level, batch, out_axes, nested)
                for leaf in out_leaves
            ]
            return tree_unflatten(out_def, cast("list[Leaf]", finished))

    # Mark the wrapper so the tracer never re-instruments it (it is a closure over
    # ``in_axes``/``runner``/``f`` -- recompiling from source would drop the closure,
    # raising ``NameError: in_axes``). It manages its own tracing; when it is the ``f``
    # of an *outer* ``vmap`` (nested ``vmap(vmap(...))``) the outer ``runner`` calls it
    # directly, and it pushes its own ``BatchTrace`` level internally.
    wrapped._pycograd_run_directly = True  # type: ignore[attr-defined]
    return wrapped


class _BatchTracerLater:
    """A placeholder pairing a :class:`~pycograd.tensor.Var` with its ``bdim`` until the
    :class:`~pycograd.batching.BatchTrace` exists (it can only be built inside ``new_main``).
    """

    __slots__ = ("var", "bdim")

    def __init__(self, var: Var, bdim: int) -> None:
        self.var = var
        self.bdim = bdim


def _mapped_batch_size(args: tuple[PyTree, ...], axes: tuple) -> int:
    """The batch size, read from the first *mapped* (``ax`` is an int) numeric leaf."""
    for a, ax in zip(args, axes):
        if ax is None:
            continue
        for leaf in tree_leaves(a):
            v = leaf.value if isinstance(leaf, Param) else leaf
            if _is_array(v) or _is_numeric(v):
                return int(np.asarray(_value(cast(Operand, v))).shape[cast(int, ax)])
    raise ValueError("vmap(grad(f)): no mapped (array) argument to map over")


def _grad_leaf(
    leaf: Leaf, ax: object, batch: int, tie_vars: dict[object, Var]
) -> tuple[Var | None, object, bool]:
    """Lift one ``vmap(grad(f))`` argument leaf onto the tape *and* into the batch level.

    Returns ``(var, tracer_later, batched)``: ``var`` is the tape node whose ``.grad`` is
    this leaf's per-sample gradient (``None`` for a frozen/non-numeric leaf, whose gradient
    comes back ``None``); ``tracer_later`` is a :class:`_BatchTracerLater` (or the raw
    value) handed to ``f``; ``batched`` flags the *mapped* (data) argument vs a *shared*
    parameter.

    A *mapped* leaf enters with its batch axis moved to the front (``bdim=0``). A *shared*
    trainable leaf is **tiled** across the batch to a genuine ``bdim=0`` operand of shape
    ``(B, *param.shape)``: each example touches its own copy, so the shared parameter's
    gradient accumulates per-example -- yielding ``d f(x_i, w)/dw`` for each ``i`` -- with
    no cross-example contraction anywhere. Tiling reduces the whole pass to the ordinary
    batched ``Var`` tape, so every primitive's backward is sound by construction (it is the
    Phase-2 batched-forward path); a genuine constant inside ``f`` stays unbatched and its
    gradient still collapses, as it should.
    """
    var, call_value = _wrap_leaf(leaf, tie_vars)
    if var is None:  # frozen / non-numeric: pass the raw value through, no gradient
        return None, call_value, ax is not None
    if ax is None:
        tiled = Var(np.broadcast_to(np.asarray(var.value), (batch,) + var.value.shape))
        return tiled, _BatchTracerLater(tiled, 0), False
    arr = np.moveaxis(np.asarray(var.value), cast(int, ax), 0)
    mapped = Var(arr)
    return mapped, _BatchTracerLater(mapped, 0), True


def _finish_grad_leaf(g: Array, batched: bool, ax: object) -> Operand:
    """One per-sample gradient leaf. A *mapped* leaf's per-example gradient is at axis 0;
    move it back to that argument's ``in_axes`` position. A *shared* leaf's gradient is
    already ``(B, *param.shape)`` (the per-sample stack); return it as-is."""
    arr = np.asarray(g)
    if batched and ax is not None and ax != 0:
        arr = np.moveaxis(arr, 0, cast(int, ax))
    return arr


def _vmap_of_grad(
    f: Callable[..., object], in_axes: object, return_value: bool
) -> Callable[..., object]:
    """``vmap(grad(f))`` / ``vmap(value_and_grad(f))``: per-sample gradients in one pass.

    A single batched forward computes the per-example output; one *batched-cotangent*
    ``backward`` (seeded with ones over the batch) then yields each argument's per-sample
    gradient -- the mapped (data) argument's gradient stacked over the batch, and a
    *shared* parameter's gradient kept per-sample as ``(B, *param.shape)`` rather than
    summed (because the shared parameter is tiled into a genuine batched operand; see
    :func:`_grad_leaf`). ``return_value`` also returns the per-example output values.
    """
    from pycograd.batching import BatchTrace, BatchTracer

    runner = _INSTRUMENTED.get(f)
    if runner is None:
        runner = _make_runner(f)
        _INSTRUMENTED[f] = runner

    def wrapped(*args: PyTree) -> object:
        axes = in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(args)
        _check_param_ownership(args)
        batch = _mapped_batch_size(args, cast("tuple", axes))
        tie_vars: dict[object, Var] = {}
        per_arg: list[tuple[TreeDef, list[tuple[Var | None, bool]], object]] = []
        flat_args: list[tuple[TreeDef, list[Leaf]]] = []
        for a, ax in zip(args, cast("tuple", axes)):
            leaves, treedef = tree_flatten(a)
            info: list[tuple[Var | None, bool]] = []
            new_leaves: list[Leaf] = []
            for leaf in leaves:
                var, call_value, batched = _grad_leaf(leaf, ax, batch, tie_vars)
                info.append((var, batched))
                new_leaves.append(cast(Leaf, call_value))
            flat_args.append((treedef, new_leaves))
            per_arg.append((treedef, info, ax))

        with new_main(BatchTrace) as main:
            trace = BatchTrace(main)

            def _realize(leaf: object) -> object:
                if isinstance(leaf, _BatchTracerLater):
                    return BatchTracer(trace, leaf.var, leaf.bdim)
                return leaf

            call_args = [
                tree_unflatten(
                    treedef, cast("list[Leaf]", [_realize(leaf) for leaf in leaves])
                )
                for treedef, leaves in flat_args
            ]
            out = runner(*call_args)
            inner = out.value if isinstance(out, BatchTracer) else out

        out_var = _lift(cast(Operand, inner))
        if out_var.value.ndim != 1 or out_var.value.shape[0] != batch:
            raise ValueError(
                "vmap(grad(f)): f must return a per-example scalar (one value per "
                f"example); got per-example output shape {out_var.value.shape[1:]}"
            )
        cotangent = _xp().ones((batch,), dtype=out_var.value.dtype)
        out_var.backward(cotangent=cotangent)

        grads = tuple(
            tree_unflatten(
                treedef,
                [
                    None if v is None else _finish_grad_leaf(v.grad, batched, ax)
                    for v, batched in info
                ],
            )
            for treedef, info, ax in per_arg
        )
        if return_value:
            return np.asarray(out_var.value), grads
        return grads

    wrapped._pycograd_run_directly = True  # type: ignore[attr-defined]
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
        with new_main(BatchTrace) as main:
            out = runner(BatchTracer(BatchTrace(main), xv, 0))
        inner = out.value if isinstance(out, BatchTracer) else out
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

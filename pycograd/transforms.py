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
from pycograd.forward import JVPTrace, JVPTracer
from pycograd.params import Param, _TieRef
from pycograd.tensor import Var, _is_array, _is_numeric, _lift, _value, _xp
from pycograd.trace import ReverseTrace, Tracer, _get_stack, bind, new_main
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
    if isinstance(leaf, Tracer):
        # ``grad`` is running under a live higher level (``jvp(grad(f))``): the leaf is a
        # ``JVPTracer`` pairing a primal ``Var`` with its tangent. Keep the tracer as the
        # call value so ``f``'s forward runs at that level; the *primal* ``Var`` is the tape
        # node whose ``.grad`` the differentiable backward fills with a level-connected
        # cotangent (the second-order information rides the tracer).
        primal = getattr(leaf, "primal", None)
        if isinstance(primal, Var):
            return primal, leaf
        return None, leaf
    if isinstance(leaf, Var):
        # ``grad`` is running inside an enclosing ``grad`` (``grad(grad(f))``): the leaf is
        # already a tape node on the outer ``grad``'s graph. Use it as both the tape node
        # (its ``.grad`` is filled with a cotangent that is itself a level-connected
        # ``Var`` chained back to it) and the call value, so the inner forward builds on
        # top of the outer's graph and the outer ``grad`` can differentiate the inner
        # gradient.
        return leaf, leaf
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

        # Is a differentiation context already enclosing this call? -- an outer ``jvp``
        # (Phase 1: forward-over-reverse) or an outer ``grad``'s reverse marker (Phase 2:
        # reverse-over-reverse). Measured *before* pushing this call's own marker, so a
        # single top-level ``grad`` reads ``higher=False`` and behaves byte-for-byte. When
        # enclosed, the backward runs differentiably (its cotangents are level-connected
        # ``Var``s the enclosing transform can keep differentiating) and the leaf grads are
        # returned as those ``Var``s rather than materialized arrays.
        higher = len(_get_stack()) > 1
        # Push a reverse marker so a *nested* ``grad`` (``grad(grad(f))``) detects it is
        # enclosed; the marker carries no tracer, so dispatch is unchanged (see
        # ``trace.ReverseTrace``). Forward and backward both run inside it.
        with new_main(ReverseTrace):
            out = runner(*call_args)
            if isinstance(out, Var):
                root: Var = out
            elif isinstance(out, Tracer) and isinstance(
                getattr(out, "primal", None), Var
            ):
                root = cast(Var, getattr(out, "primal"))
            else:
                root = cast(Var, None)

            if root is not None:
                # otherwise each leaf's grad stays at its init zeros
                root.backward(differentiable=higher)
                value: Array = out.value if isinstance(out, Var) else root.value
            elif isinstance(out, (tuple, list, dict)) and any(
                isinstance(leaf, (Var, Tracer)) for leaf in tree_leaves(out)
            ):
                # The differentiated function returned a *container* of tape values rather
                # than a single scalar -- almost always a nested ``grad`` whose inner
                # gradient wasn't scalarized.
                raise TypeError(
                    "grad/value_and_grad differentiates a function returning a single "
                    f"scalar, but it returned a {type(out).__name__} of differentiable "
                    "values. For a Hessian use jacfwd(grad(f)) or jacrev(grad(f)); to nest "
                    "grad, scalarize the inner gradient first, e.g. "
                    "grad(lambda x: np.sum(grad(f)(x)[0]))."
                )
            else:
                value = _xp().asarray(_value(cast(Operand, out)), dtype=float)

        def _grad_leaf(orig: Leaf, v: Var | None) -> object:
            if v is None:
                return None
            return v.grad if higher else _match_arg(orig, v.grad)

        grads = tuple(
            tree_unflatten(
                treedef,
                cast("list[Leaf]", [_grad_leaf(orig, v) for orig, v in info]),
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


def jvp(
    f: Callable[..., object],
    primals: tuple[PyTree, ...],
    tangents: tuple[PyTree, ...],
) -> tuple[object, object]:
    """Forward-mode AD: ``(f(*primals), df(*primals) . tangents)`` in one pass.

    JAX-style: ``primals`` and ``tangents`` are tuples with one entry per positional
    argument of ``f``; corresponding leaves must match in shape. Returns
    ``(primal_out, tangent_out)`` -- the primal output of ``f`` and its directional
    derivative along ``tangents`` -- both as pytrees matching ``f``'s output.

    Realized as one level of the trace-level interpreter stack (a
    :class:`~pycograd.forward.JVPTrace` pushed via ``new_main``): each argument leaf
    enters as a :class:`~pycograd.forward.JVPTracer` pairing its primal with its tangent,
    every primitive computes ``primal_out`` one level down and ``tangent_out`` via its
    forward-derivative rule, and the result leaves carry the propagated tangent. Because
    the tangent arithmetic itself flows through ``bind``, ``jvp`` *composes* with
    :func:`vmap` (``vmap(lambda x: jvp(g, (x,), (v,))[1])``) and nests.
    """
    if len(primals) != len(tangents):
        raise ValueError("jvp: primals and tangents must have the same length")
    runner = _INSTRUMENTED.get(f)
    if runner is None:
        runner = _make_runner(f)
        _INSTRUMENTED[f] = runner

    from pycograd.forward import _HOF_TRACER_FOR

    hof_token = _HOF_TRACER_FOR.set({})
    try:
        return _run_jvp(f, runner, primals, tangents)
    finally:
        _HOF_TRACER_FOR.reset(hof_token)


def _run_jvp(
    f: Callable[..., object],
    runner: Callable[..., object],
    primals: tuple[PyTree, ...],
    tangents: tuple[PyTree, ...],
) -> tuple[object, object]:
    with new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        level = main.level
        call_args: list[PyTree] = []
        for p, t in zip(primals, tangents):
            p_leaves, p_def = tree_flatten(p)
            t_leaves, _t_def = tree_flatten(t)
            if len(p_leaves) != len(t_leaves):
                raise ValueError(
                    "jvp: a primal argument and its tangent have different pytree "
                    "structure"
                )
            new_leaves: list[Leaf] = []
            for pl, tl in zip(p_leaves, t_leaves):
                primal_inner, tangent_inner = _jvp_inputs(pl, tl)
                if primal_inner is None:
                    new_leaves.append(pl)  # non-numeric leaf: pass through, no tangent
                    continue
                new_leaves.append(
                    cast(Leaf, JVPTracer(trace, primal_inner, tangent_inner))
                )
            call_args.append(tree_unflatten(p_def, new_leaves))

        out = runner(*call_args)
        out_leaves, out_def = tree_flatten(cast(PyTree, out))
        primal_leaves = [_jvp_primal(leaf, level) for leaf in out_leaves]
        tangent_leaves = [_jvp_tangent(leaf, level) for leaf in out_leaves]

    # Outside the ``with`` block the ``jvp`` level is popped. If no *other* level remains
    # live (this ``jvp`` was outermost), coerce any ``Var`` that rode out -- e.g. a reverse
    # cotangent's propagated tangent under ``jvp(grad(f))`` -- to a concrete array. If a
    # higher level is still live (``vmap``/``grad`` around this ``jvp``), keep it flowing.
    if len(_get_stack()) == 1:
        primal_leaves = [_coerce_top(v) for v in primal_leaves]
        tangent_leaves = [_coerce_top(v) for v in tangent_leaves]
    primal_out = tree_unflatten(out_def, cast("list[Leaf]", primal_leaves))
    tangent_out = tree_unflatten(out_def, cast("list[Leaf]", tangent_leaves))
    return primal_out, tangent_out


def _coerce_top(v: object) -> object:
    if isinstance(v, Var):
        return np.asarray(_value(cast(Operand, v)))
    return v


def _jvp_inputs(pl: object, tl: object) -> tuple[object | None, object]:
    """The (primal, tangent) inner values to seed a ``JVPTracer`` for one argument leaf.

    A bare numeric leaf becomes a fresh ``Var`` primal and ``Var`` tangent. A leaf that is
    *already* on a tape / an enclosing level (a :class:`~pycograd.tensor.Var` under
    ``grad``, or a :class:`~pycograd.batching.BatchTracer` under an outer ``vmap``) is kept
    as the primal so its level keeps flowing; its tangent is lifted to the same level by
    ``bind`` when used. A non-numeric, non-tracer leaf yields ``(None, _)`` so the caller
    passes it through untouched (no tangent)."""
    if isinstance(pl, (Var, Tracer)):
        return pl, tl
    if not _is_numeric(pl):
        return None, tl
    pv = _value(cast(Operand, pl))
    if isinstance(tl, (Var, Tracer)):
        # The tangent is itself on an enclosing level (``vmap`` over the tangent, or
        # ``jvp`` of ``jvp``); keep the tracer so its level keeps flowing rather than
        # coercing it to a concrete array.
        return Var(pv), tl
    tv = _xp().asarray(_value(cast(Operand, tl)), dtype=np.asarray(pv).dtype)
    return Var(pv), Var(tv)


def _jvp_materialize(v: object) -> object:
    """Pull one output value to a concrete array (top level) or hand it down unchanged
    when it is still on an enclosing tape/level, so nesting keeps flowing.

    A :class:`Tracer` (``vmap``/``jvp``) is handed down as-is. A :class:`Var` is also
    passed through (not coerced to an array): under ``jvp(grad(f))`` the propagated
    tangent of a reverse cotangent is a ``Var`` on the (enclosing ``grad``'s) tape, and it
    must ride out of the ``jvp`` so the surrounding transform keeps differentiating it.
    """
    if isinstance(v, Tracer):  # still on an enclosing level (vmap/jvp) -> keep flowing
        return v
    if isinstance(
        v, Var
    ):  # a reverse cotangent on an enclosing grad tape -> keep flowing
        return v
    return np.asarray(_value(cast(Operand, v)))


def _jvp_primal(leaf: object, level: int) -> object:
    if isinstance(leaf, JVPTracer) and leaf._trace.main.level == level:
        return _jvp_materialize(leaf.primal)
    return _jvp_materialize(leaf)


def _jvp_tangent(leaf: object, level: int) -> object:
    """One tangent output leaf. A tracer at this level carries the propagated tangent; an
    output that does not depend on the input (not a tracer at this level) has a zero
    tangent shaped like the primal."""
    if isinstance(leaf, JVPTracer) and leaf._trace.main.level == level:
        return _jvp_materialize(leaf.tangent)
    arr = np.asarray(_value(cast(Operand, leaf)))
    return np.zeros_like(arr)


def jacfwd(f: Callable[..., object], argnum: int = 0) -> Callable[..., object]:
    """Forward-mode Jacobian of ``f`` w.r.t. its ``argnum``-th argument.

    Builds the Jacobian by pushing one-hot tangent basis vectors through :func:`jvp` --
    one column per input coordinate -- and stacking the resulting output-tangents. The
    basis sweep is vectorized with :func:`vmap` over the JVP, demonstrating that forward
    mode composes with batching: ``vmap`` maps the (flat) one-hot index, ``jvp`` carries
    each basis tangent forward, and the per-basis output tangents stack into the Jacobian.

    For a scalar-output ``f`` this agrees with reverse-mode :func:`grad` (the gradient is
    the single Jacobian row); for vector output it returns the full ``(out, in)``
    Jacobian shaped ``(*out_shape, *in_shape)``.
    """

    def jacobian(*args: PyTree) -> object:
        x = args[argnum]
        flat, treedef = tree_flatten(x)
        if len(flat) != 1 or not _is_numeric(flat[0]):
            raise ValueError(
                "jacfwd: the differentiated argument must be a single array"
            )
        x_arr = np.asarray(_value(cast(Operand, flat[0])))
        n = int(x_arr.size)
        basis = np.eye(n, dtype=x_arr.dtype).reshape((n,) + x_arr.shape)

        def column(e: object) -> object:
            tan_args = tuple(
                tree_unflatten(treedef, [cast(Leaf, e)]) if i == argnum else a
                for i, a in enumerate(args)
            )
            zeros = tuple(
                tree_unflatten(
                    tree_flatten(a)[1],
                    [
                        (
                            np.zeros_like(np.asarray(_value(cast(Operand, leaf))))
                            if _is_numeric(leaf)
                            else leaf
                        )
                        for leaf in tree_flatten(a)[0]
                    ],
                )
                for a in args
            )
            primals = tuple(args)
            tangents = tuple(
                tan_args[i] if i == argnum else zeros[i] for i in range(len(args))
            )
            return jvp(f, primals, tangents)[1]

        try:
            cols = vmap(column)(basis)
            stacked = np.asarray(cols)
        except Exception:
            # Fall back to a Python loop if the vmap-over-jvp composition trips on a
            # particular function (kept available so jacfwd is always usable).
            stacked = np.stack([np.asarray(column(basis[i])) for i in range(n)], axis=0)
        # ``stacked`` is (n_in, *out_shape); move the input axis last -> (*out, *in).
        out_shape = stacked.shape[1:]
        jac = np.moveaxis(stacked, 0, -1)
        return jac.reshape(out_shape + x_arr.shape)

    return jacobian


def jacrev(f: Callable[..., object], argnum: int = 0) -> Callable[..., object]:
    """Reverse-mode Jacobian of ``f`` w.r.t. its ``argnum``-th argument.

    Builds the Jacobian one *row* at a time: the gradient of each scalar output component
    ``sum(f(x) * e_i)`` w.r.t. the input is that row of the Jacobian, computed by reverse
    mode (:func:`grad`). The rows stack into the full ``(*out_shape, *in_shape)`` Jacobian
    (for scalar output it is the single-row gradient, agreeing with :func:`grad` and
    :func:`jacfwd`).

    Composes with :func:`grad` for *reverse-over-reverse* Hessians: ``jacrev(grad(f))``
    differentiates the gradient of ``f`` again -- each row's reverse pass runs while the
    enclosing ``grad`` is live, so its inner backward records a cotangent graph the outer
    backward differentiates. The result agrees with the forward-over-reverse
    ``jacfwd(grad(f))`` Hessian.
    """

    def jacobian(*args: PyTree) -> object:
        x = args[argnum]
        flat, treedef = tree_flatten(x)
        if len(flat) != 1 or not _is_numeric(flat[0]):
            raise ValueError(
                "jacrev: the differentiated argument must be a single array"
            )
        x_arr = np.asarray(_value(cast(Operand, flat[0])))
        # Probe the output shape with a plain (untraced) call.
        y0 = np.asarray(_value(cast(Operand, f(*args))))
        out_shape = y0.shape
        m = int(y0.size)
        basis = np.eye(m, dtype=x_arr.dtype).reshape((m,) + out_shape)

        def make_component(e: Array) -> Callable[..., object]:
            # ``sum(f(x) * e_i)`` -- a scalar whose gradient is one Jacobian row. Tagged to
            # run directly so the tracer keeps it (and its closure over ``f``/``e``) intact
            # while still intercepting the ``np.*`` calls inside.
            def component(xa: object) -> object:
                replaced = tuple(
                    tree_unflatten(treedef, [cast(Leaf, xa)]) if i == argnum else a
                    for i, a in enumerate(args)
                )
                weighted = _lift(cast(Operand, f(*replaced))) * cast(Operand, e)
                return ops.d_sum(weighted)

            component._pycograd_run_directly = True  # type: ignore[attr-defined]
            return component

        rows = [np.asarray(grad(make_component(basis[i]))(x)[argnum]) for i in range(m)]
        stacked = np.stack(rows, axis=0)  # (m, *in_shape)
        jac = stacked.reshape(out_shape + x_arr.shape)
        return jac

    return jacobian


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

# -*- coding: utf-8 -*-
"""Gradient checkpointing (activation rematerialization).

``checkpoint(f)`` wraps a segment of a model so that its *intermediate* activations are
**not** retained on the tape. The forward keeps only the segment's inputs (and the small
output boundary); the activations are **recomputed during backward** by re-running ``f``.
This trades ~one extra forward pass for a peak-memory drop from "every segment at once" to
"one segment at a time" -- the deep-net / long-sequence OOM relief the closure-tape needs.

Design (see ROADMAP, Phase 2):

* A single **boundary node** ``B`` stands in for ``outputs = f(inputs)``. ``B``'s value is
  the *flat concatenation* of the segment's output-leaf values; the user-visible outputs
  are real differentiable ``slice``+``reshape`` views of ``B`` (built from ``d_getitem`` /
  ``d_reshape``). Because slice+concat is a bijective rearrangement, downstream cotangents
  flow back through the existing slice/reshape VJPs into a single cotangent on ``B`` that is
  exactly the concatenation of the per-output-leaf cotangents -- recoverable by split. This
  makes the multi-output cotangent join automatic, reusing the library's own VJP rules.
* ``B``'s backward **rematerializes**: it lifts the saved input/weight values into fresh
  leaf ``Var``s, re-runs ``f`` to rebuild the inner tape, contracts each fresh output leaf
  with its cotangent into a scalar, runs an inner backward, and scatters the resulting
  input/weight gradients onto ``B``'s parents. The inner tape lives only for the duration of
  this backward and is freed immediately after.
* **Ambient weights** (``with weights:`` / ``weights.grad``) enter ``f`` via globals, not
  args, and the live binding is gone by backward time, so ``checkpoint`` discovers the
  weight ``Var``s a segment touches (via :func:`pycograd.params.active_weight_bindings`),
  makes them boundary parents, and re-binds them for the remat.

Constraints (documented, enforced where cheap):

* ``f`` must be **deterministic** in its inputs+weights -- the backward remat must reproduce
  the forward activations. RNG/dropout inside a checkpointed ``f`` is out of scope for v1.
* ``f`` takes its differentiable inputs as positional args and reaches weights via the
  ambient DSL -- not via closure free vars (instrumentation drops those).
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

import numpy as np

from pycograd._typing import Boxed
from pycograd.tensor import Var, _unbroadcast, _xp, grad_recording
from pycograd.tree import Leaf, PyTree, TreeDef, tree_flatten, tree_unflatten

if TYPE_CHECKING:
    from pycograd.params import ParamDict


def _collect_leaf_vars(roots: Sequence[object]) -> list[Var]:
    """Every leaf ``Var`` (no ``_parents``) reachable from ``roots`` over the tape graph."""
    seen: set[int] = set()
    out: list[Var] = []
    stack: list[Var] = [r for r in roots if isinstance(r, Var)]
    while stack:
        v = stack.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        if v._parents:
            stack.extend(p for p in v._parents if isinstance(p, Var))
        else:
            out.append(v)
    return out


class _Remat:
    """Per-instance rematerialization state for one ``checkpoint`` boundary node ``B``.

    Holds everything the backward needs to rebuild the segment's forward: the instrumented
    ``runner``, the input pytree layout and saved values, the ambient-weight snapshots, the
    flat output layout, and the real outer ``Var`` parents (inputs then weights) the
    recomputed gradients are scattered onto. :meth:`raw_backward` drives the ordinary
    numpy-``.grad`` reverse pass (where the boundary is built and the memory is saved);
    :meth:`differentiable_vjp` is only reached if a built boundary is differentiated a
    second time in reverse (unsupported -- see its docstring).
    """

    def __init__(
        self,
        runner: Callable[..., PyTree],
        in_treedef: TreeDef,
        in_leaf_kinds: list[tuple[str, object]],
        input_vars: list[Var],
        weight_bindings: list[tuple["ParamDict", dict[str, Var]]],
        used_weight_vars: list[Var],
        out_treedef: TreeDef,
        leaf_shapes: list[tuple[int, ...]],
        slices: list[tuple[int, int]],
    ) -> None:
        self.runner = runner
        self.in_treedef = in_treedef
        # One entry per flattened input leaf: ("var", i) -> input_vars[i]; ("const", value).
        self.in_leaf_kinds = in_leaf_kinds
        self.input_vars = input_vars
        self.input_values = [v.value for v in input_vars]
        self.weight_bindings = weight_bindings
        self.used_weight_vars = used_weight_vars
        self.out_treedef = out_treedef
        self.leaf_shapes = leaf_shapes
        self.slices = slices
        self.B: Var | None = None  # set once the boundary node is built

    # -- rematerialization shared by both paths ------------------------------
    def _fresh_inputs(self) -> tuple[list[PyTree], list[Var]]:
        """Fresh leaf ``Var``s for the inputs + the reassembled positional ``call_args``."""
        fresh = [Var(v) for v in self.input_values]
        leaves: list[Leaf] = []
        for kind, payload in self.in_leaf_kinds:
            if kind == "var":
                leaves.append(fresh[cast(int, payload)])
            else:
                leaves.append(cast(Leaf, payload))
        call_args = cast("list[PyTree]", tree_unflatten(self.in_treedef, leaves))
        return call_args, fresh

    def _rebind_and_run(
        self, call_args: Sequence[PyTree], weight_for: dict[int, Boxed]
    ) -> list[Boxed]:
        """Re-run the segment with each used weight rebound to ``weight_for[id(orig)]`` --
        a fresh leaf ``Var`` (raw path) or a level-connected operand (differentiable path) --
        and return the inner output leaves. Rebinds each active model's full snapshot (so
        *tied* keys sharing one ``Var`` rebind consistently) for the dynamic extent of the
        call."""
        saved: list[tuple["ParamDict", dict[str, object] | None]] = []
        try:
            for model, snap in self.weight_bindings:
                live = {
                    key: weight_for[id(orig)]
                    for key, orig in snap.items()
                    if id(orig) in weight_for
                }
                saved.append((model, getattr(model, "_live", None)))
                model._live = cast("dict[str, object]", live)
            with grad_recording():
                out = self.runner(*call_args)
        finally:
            for model, prev in saved:
                model._live = prev
        return cast(list[Boxed], tree_flatten(out)[0])

    def _split_cotangent(self, g_flat: Any) -> list[Any]:
        """Split a flat boundary cotangent back into per-output-leaf cotangents."""
        return [
            g_flat[start:end].reshape(shape)
            for (start, end), shape in zip(self.slices, self.leaf_shapes)
        ]

    # -- raw path (numpy .grad) ----------------------------------------------
    def raw_backward(self) -> None:
        from pycograd import ops

        B = cast(Var, self.B)
        cots = self._split_cotangent(B.grad)
        call_args, fresh_inputs = self._fresh_inputs()
        fresh_weight_for: dict[int, Boxed] = {
            id(o): Var(o.value) for o in self.used_weight_vars
        }
        inner_leaves = self._rebind_and_run(call_args, fresh_weight_for)

        s: Var | None = None
        for leaf, cot in zip(inner_leaves, cots):
            if not isinstance(leaf, Var):
                continue
            term = ops.d_sum(ops.d_mul(leaf, cot))
            s = term if s is None else ops.d_add(s, term)
        if s is None:
            return  # no output leaf depends on any input: gradients stay zero
        s.backward()

        for real, fresh in zip(self.input_vars, fresh_inputs):
            real.grad = real.grad + _unbroadcast(fresh.grad, real.value.shape)
        for orig in self.used_weight_vars:
            fv = cast(Var, fresh_weight_for[id(orig)])
            orig.grad = orig.grad + _unbroadcast(fv.grad, orig.value.shape)

    # -- differentiable path (higher-order reverse) --------------------------
    def differentiable_vjp(self, operands: tuple[Boxed, ...], g: Boxed) -> list[Boxed]:
        """Reached only when a *built* boundary is differentiated a second time in reverse
        (``grad(grad(...))`` / ``jacrev`` of a checkpointed gradient): the boundary is only
        constructed under a single, non-nested reverse pass, so any path here is
        reverse-over-reverse. The segment's tape was discarded, and re-running it into the
        live cotangent graph re-enters the in-progress backward -- a blow-up -- so fail
        clearly and point at the supported forward-over-reverse route, which gives the same
        Hessian/HVP. (Under a live ``jvp``/``vmap``, checkpoint passes through and no
        boundary exists, so those compositions never reach here.)"""
        raise NotImplementedError(
            "checkpoint does not support reverse-over-reverse differentiation "
            "(grad(grad(...)) or jacrev of a checkpointed gradient). Use the "
            "forward-over-reverse route instead -- jvp(grad(f)) for an HVP, "
            "jacfwd(grad(f)) for the Hessian -- which checkpoint supports."
        )


def _flatten_args(args: tuple[PyTree, ...]) -> tuple[list[Leaf], TreeDef]:
    return tree_flatten(list(args))


def _checkpoint_call(
    f: Callable[..., PyTree],
    runner: Callable[..., PyTree],
    args: tuple[PyTree, ...],
) -> PyTree:
    in_leaves, in_treedef = _flatten_args(args)
    from pycograd.params import active_weight_bindings
    from pycograd.trace import Tracer

    # A live ``jvp``/``vmap`` presents the segment's inputs as *Tracers*. A plain-``Var``
    # rematerialization boundary cannot be built at that level (it would drop the batch /
    # tangent axis), so checkpoint is *transparent* there: run the segment instrumented at
    # the live level so it differentiates correctly (no memory saving in that nested case).
    if any(isinstance(leaf, Tracer) for leaf in in_leaves):
        return runner(*args)

    bindings = active_weight_bindings()
    input_vars = [leaf for leaf in in_leaves if isinstance(leaf, Var)]
    recording = bool(input_vars) or bool(bindings)
    if not recording:
        # Inference (no tape): nothing to checkpoint. Call ``f`` *directly* (not the
        # instrumented runner, which would swap ``np.*`` to taping ``d_*`` and build a Var
        # from plain arrays) so the result is a plain array, exactly as un-checkpointed.
        return f(*args)

    # Classify each input leaf: a Var becomes a boundary parent (and is rematerialized from
    # a fresh leaf); anything else is a constant passed through verbatim on the remat.
    in_leaf_kinds: list[tuple[str, object]] = []
    var_index: dict[int, int] = {}
    for leaf in in_leaves:
        if isinstance(leaf, Var):
            idx = var_index.setdefault(id(leaf), len(var_index))
            in_leaf_kinds.append(("var", idx))
        else:
            in_leaf_kinds.append(("const", leaf))
    # ``input_vars`` deduplicated in first-seen order, aligned with the "var" indices.
    seen: dict[int, Var] = {}
    for leaf in in_leaves:
        if isinstance(leaf, Var) and id(leaf) not in seen:
            seen[id(leaf)] = leaf
    input_vars = list(seen.values())

    # Capture forward: run the segment on *detached* fresh inputs (so the inner tape is
    # disjoint from the outer graph) while letting ambient weights resolve to their real
    # live Vars (so we can discover which the segment touches). Keep only the output values
    # and the touched weight Vars; the activations are dropped with the inner tape.
    cap_leaves: list[Leaf] = []
    for kind, payload in in_leaf_kinds:
        if kind == "var":
            cap_leaves.append(Var(input_vars[cast(int, payload)].value))
        else:
            cap_leaves.append(cast(Leaf, payload))
    cap_args = cast("list[PyTree]", tree_unflatten(in_treedef, cap_leaves))
    with grad_recording():
        out_cap = runner(*cap_args)
    out_leaves, out_treedef = tree_flatten(out_cap)

    weight_var_ids = {id(v) for _, snap in bindings for v in snap.values()}
    leaf_vars = _collect_leaf_vars(out_leaves)
    used_weight_vars: list[Var] = []
    used_ids: set[int] = set()
    for v in leaf_vars:
        if id(v) in weight_var_ids and id(v) not in used_ids:
            used_ids.add(id(v))
            used_weight_vars.append(v)

    # Boundary node: flat concatenation of the output-leaf values.
    xp = _xp()
    leaf_shapes = [tuple(np.shape(_leaf_value(leaf))) for leaf in out_leaves]
    flats = [xp.reshape(xp.asarray(_leaf_value(leaf)), (-1,)) for leaf in out_leaves]
    flat_value = xp.concatenate(flats) if flats else xp.zeros((0,))
    slices: list[tuple[int, int]] = []
    start = 0
    for fl in flats:
        slices.append((start, start + int(fl.shape[0])))
        start += int(fl.shape[0])

    parents = tuple(input_vars) + tuple(used_weight_vars)
    box = _Remat(
        runner=runner,
        in_treedef=in_treedef,
        in_leaf_kinds=in_leaf_kinds,
        input_vars=input_vars,
        weight_bindings=bindings,
        used_weight_vars=used_weight_vars,
        out_treedef=out_treedef,
        leaf_shapes=leaf_shapes,
        slices=slices,
    )
    B = Var(flat_value, _parents=parents)
    box.B = B
    B._backward = box.raw_backward

    from pycograd import ops

    B._vjp_prim = ops._remat
    B._vjp_operands = parents
    B._vjp_params = {"remat": box}

    # User-visible outputs: real slice+reshape views of B (so cotangents rejoin into B).
    out_views: list[Leaf] = []
    for (st, en), shape in zip(slices, leaf_shapes):
        sl = ops.d_getitem(B, slice(st, en))
        out_views.append(ops.d_reshape(sl, shape))
    return tree_unflatten(out_treedef, out_views)


def _leaf_value(leaf: object) -> Any:
    return leaf.value if isinstance(leaf, Var) else leaf


def checkpoint(f: Callable[..., PyTree]) -> Callable[..., PyTree]:
    """Wrap ``f`` so its intermediate activations are recomputed during backward instead of
    retained on the tape (gradient checkpointing / activation rematerialization).

    ``checkpoint(f)(*args)`` is a drop-in for ``f(*args)`` *inside* a model/objective being
    differentiated. Its differentiable inputs are the positional ``args`` (a Var/array, or a
    pytree of them); ambient ``with weights:`` parameters it uses are handled automatically.
    Gradients match those of the un-checkpointed ``f`` (verified against finite differences);
    only peak memory differs. ``f`` must be deterministic in its inputs+weights.

    Composition. The rematerialization boundary is built under a single reverse pass --
    ``grad`` / ``value_and_grad`` / ``weights.grad`` -- which is where the memory saving
    applies. Under a live ``jvp`` or ``vmap`` (e.g. ``jvp(grad(f))`` HVPs, ``vmap(f)``)
    checkpoint is *transparent*: it passes through and differentiates correctly, but does
    not save memory in that nested case (a boundary can't be built at the tracer level
    without dropping the tangent/batch axis). Reverse-over-reverse differentiation of a
    checkpointed segment (``grad(grad(...))`` / ``jacrev`` of its gradient) is unsupported
    and raises; use the forward-over-reverse route (``jacfwd(grad(f))``) for a Hessian.
    """
    from pycograd.tracer import _INSTRUMENTED, _make_runner

    runner = _INSTRUMENTED.get(f)
    if runner is None:
        runner = _make_runner(f)
        _INSTRUMENTED[f] = runner

    @functools.wraps(f)
    def wrapped(*args: PyTree) -> PyTree:
        return _checkpoint_call(f, runner, args)

    # Manage our own tracing/boundary construction: keep this orchestration wrapper out of
    # the interception path (as vmap/grad wrappers do), so it isn't itself instrumented.
    wrapped._pycograd_run_directly = True  # type: ignore[attr-defined]
    return wrapped

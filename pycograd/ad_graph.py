# -*- coding: utf-8 -*-
"""Reverse-mode autodiff *on the capture IR*: ``grad_graph(forward) -> graph`` turns a
captured forward :class:`~pycograd.capture.Graph` into one graph that computes the
output **and** its gradients w.r.t. the inputs -- forward and backward together.

This is what lets optimization passes work *across* the forward/backward boundary
(e.g. CSE merging a ``sigmoid`` the backward recomputes with the forward's value).
It works by reusing the existing higher-order VJP rules (``ops._VJP_FOR``) as graph
builders: those rules build cotangents with ``bind``, so run under a ``GraphTrace``
with ``GraphTracer`` operands, each ``bind`` records a backward node. The eager
``.grad`` tape engine is untouched.

The one impedance-match: a captured node stores ``bind``'s positional args, but a VJP
rule wants the ``_record_vjp`` (operands, params) split -- e.g. ``einsum``'s subscripts
is a positional arg in the node but a ``param`` to the rule. ``_decompose`` recovers
that split per primitive.
"""
from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np

from pycograd import ops
from pycograd._typing import Boxed
from pycograd.capture import (
    _CONST,
    _INPUT,
    Const,
    Graph,
    GraphTrace,
    GraphTracer,
    Ref,
    _Builder,
    _is_numeric,
    _rebuild,
    capture,
    eval_graph,
)
from pycograd.shapes import ShapedArray
from pycograd.tensor import _d_unbroadcast, _value
from pycograd.trace import bind, new_main
from pycograd.tree import PyTree, tree_flatten, tree_map, tree_unflatten


# ---------------------------------------------------------------------------
# Map a captured node's (prim, args, params) to the (operand specs, vjp params) a
# ``_VJP_FOR`` rule expects -- recovering what ``_record_vjp`` encoded eagerly.
# Default: every arg is a (differentiable) operand and params pass through.
# ---------------------------------------------------------------------------
def _const(spec: Any) -> Any:
    return spec.value if isinstance(spec, Const) else spec


def _decompose(prim: Any, args: tuple, params: dict) -> "tuple[list, dict]":
    if prim is ops.d_einsum:  # (subscripts, *operands); subscripts is a param
        return list(args[1:]), {"subscripts": _const(args[0]), **params}
    if prim is ops.d_getitem:  # (x, key); key is a param
        return [args[0]], {"key": _const(args[1]), **params}
    if prim is ops._scatter:  # (g, key, shape, dtype); only g is an operand
        return [args[0]], {"key": _const(args[1]), **params}
    if prim is ops.d_reshape or prim is ops.d_expand_dims:  # (x, shape/axis)
        return [args[0]], {}  # VJP reshapes the cotangent to the primal's shape
    if prim is ops.d_transpose:  # (x[, axes])
        axes = _const(args[1]) if len(args) > 1 else None
        return [args[0]], {"axes": axes}
    if prim is ops.d_concatenate or prim is ops.d_stack:  # ([parts], axis=...)
        return list(args[0]), dict(params)
    if prim is ops.d_where:  # (cond, a, b); cond is a param, a/b the operands
        return [args[1], args[2]], {"cond": _const(args[0]), **params}
    return list(args), dict(params)


# ---------------------------------------------------------------------------
# Per-primitive graph-building VJP rules for the ops whose eager rule reads a primal's
# *data* (a mask), or which are composed (not in ``_VJP_FOR``) yet recorded as one node
# because the trace routes them through ``bind`` (``stack``, ``mean``). ``_vjp_on_graph``
# consults this first, falling back to ``_VJP_FOR`` for everything else.
# ---------------------------------------------------------------------------
GraphVJP = Callable[..., "list[Boxed]"]


def _b(prim: Any, *args: Any, **kw: Any) -> Boxed:
    return bind(prim, *args, **kw)


def _mask_to_float(cond: Boxed) -> Boxed:
    return _b(ops.d_mul, cond, 1.0)  # bool comparison node -> float, in the graph


def _g_abs(operands: tuple, params: dict, g: Boxed, out: Boxed) -> "list[Boxed]":
    # d|x|/dx = sign(x), built as graph nodes (where(x>0,1,where(x<0,-1,0))).
    (x,) = operands
    sign = _b(
        ops.d_where,
        _b(ops.d_gt, x, 0.0),
        1.0,
        _b(ops.d_where, _b(ops.d_lt, x, 0.0), -1.0, 0.0),
    )
    return [_b(ops.d_mul, g, sign)]


def _g_select(operands: tuple, params: dict, g: Boxed, out: Boxed) -> "list[Boxed]":
    # maximum/minimum: gradient flows to whichever operand equals the output (ties to a),
    # the mask `(a == out)` built from the operand and output graph values.
    a, _b_op = operands
    mask = _mask_to_float(_b(ops.d_eq, a, out))
    return [_b(ops.d_mul, g, mask), _b(ops.d_mul, g, _b(ops.d_sub, 1.0, mask))]


def _g_reduce_select(reducer_prim: Any) -> GraphVJP:
    def rule(operands: tuple, params: dict, g: Boxed, out: Boxed) -> "list[Boxed]":
        # max/min reduction: gradient flows to the arg-extremum, split on ties. Recompute
        # the keepdims extremum so the mask `(x == kept)` broadcasts against x.
        (x,) = operands
        axis = params.get("axis")
        keepdims = params.get("keepdims", False)
        kept = _b(reducer_prim, x, axis=axis, keepdims=True)
        mask = _mask_to_float(_b(ops.d_eq, x, kept))
        count = _b(ops.d_sum, mask, axis=axis, keepdims=True)
        norm = _b(ops.d_div, mask, count)
        gg = g if (axis is None or keepdims) else ops._expand_dims_multi(g, axis)
        return [_b(ops.d_mul, gg, norm)]

    return rule


def _reduced_count(shape: "tuple[int, ...]", axis: Any) -> int:
    if axis is None:
        return int(np.prod(shape, dtype=np.int64)) if shape else 1
    axes = axis if isinstance(axis, tuple) else (axis,)
    return int(np.prod([shape[a] for a in axes], dtype=np.int64)) if axes else 1


def _g_mean(operands: tuple, params: dict, g: Boxed, out: Boxed) -> "list[Boxed]":
    # mean is sum / count -- not in _VJP_FOR, so build it here: broadcast the cotangent
    # back over the reduced axes and divide by the number of elements averaged.
    (x,) = operands
    axis = params.get("axis")
    keepdims = params.get("keepdims", False)
    xshape = _shape_of(x)
    n = float(_reduced_count(xshape, axis))
    gg = g if (axis is None or keepdims) else ops._expand_dims_multi(g, axis)
    return [_b(ops.d_div, _b(ops.d_broadcast_to, gg, xshape), n)]


def _g_stack(operands: tuple, params: dict, g: Boxed, out: Boxed) -> "list[Boxed]":
    # stack(parts, axis): the cotangent of part i is g sliced at index i along axis (an
    # int index removes that axis -- the inverse of the new axis stack inserted). Not in
    # _VJP_FOR (stack is composed eagerly), so built here.
    axis = params.get("axis", 0)
    gnd = len(cast(Any, g).aval.shape)
    ax = axis % gnd
    grads: list[Boxed] = []
    for i in range(len(operands)):
        key = tuple(i if d == ax else slice(None) for d in range(gnd))
        grads.append(_b(ops.d_getitem, g, key))
    return grads


def _g_pow(operands: tuple, params: dict, g: Boxed, out: Boxed) -> "list[Boxed]":
    # x**p with constant exponent p: d/dx = p * x**(p-1). The exponent is a Const operand
    # (a value, not a graph node), matching the eager rule (no grad to the exponent).
    a, b = operands
    if isinstance(b, GraphTracer):
        raise NotImplementedError("grad_graph: pow with a non-constant exponent")
    ga = _b(ops.d_mul, g, _b(ops.d_mul, b, _b(ops.d_pow, a, b - 1)))
    return [ga, None]


_VJP_GRAPH: dict[Any, GraphVJP] = {
    ops.d_abs: _g_abs,
    ops.d_maximum: _g_select,
    ops.d_minimum: _g_select,
    ops.d_max: _g_reduce_select(ops.d_max),
    ops.d_min: _g_reduce_select(ops.d_min),
    ops.d_mean: _g_mean,
    ops.d_stack: _g_stack,
    ops.d_pow: _g_pow,
}


def _vjp_on_graph(
    prim: Any, operands: list, params: dict, g: Boxed, out: Boxed
) -> "list[Boxed]":
    rule = _VJP_GRAPH.get(prim)
    if rule is not None:
        return rule(tuple(operands), params, g, out)
    base = ops._VJP_FOR.get(prim)
    if base is None:
        raise NotImplementedError(
            f"grad_graph: no graph VJP rule for {getattr(prim, '__name__', prim)!r}"
        )
    return base(tuple(operands), tuple(operands), params, g)


# ---------------------------------------------------------------------------
def _shape_of(v: Boxed) -> "tuple[Any, ...]":
    if isinstance(v, GraphTracer):
        return tuple(v.aval.shape)
    return np.shape(cast(Any, v))


def _ishape(shape: Any) -> "tuple[int, ...]":
    return cast("tuple[int, ...]", tuple(shape))


def _operand_val(spec: Any, env: dict) -> Boxed:
    return env[spec.id] if isinstance(spec, Ref) else _const(spec)


def grad_graph(forward: Graph) -> Graph:
    """Differentiate a captured forward graph. Returns a :class:`Graph` whose output is
    ``(value, grads)`` -- ``value`` the original (scalar) output and ``grads`` a flat
    tuple of cotangents, one per input leaf. Replays the forward (recording it), then a
    reverse pass over its nodes builds the backward with the VJP rules."""
    if len(forward.outputs) != 1:
        raise ValueError("grad_graph: expected a single (scalar) output")

    builder = _Builder()
    with new_main(GraphTrace, builder) as main:
        trace = GraphTrace(main)
        # Forward replay: record each node, env maps fwd id -> GraphTracer / const value.
        env: dict[int, Boxed] = {}
        for node in forward.nodes:
            if node.prim is _INPUT:
                env[node.id] = trace.add_input(
                    ShapedArray(node.aval.shape, node.aval.dtype)
                )
            elif node.prim is _CONST:
                env[node.id] = node.params["value"]
            else:
                envo = cast("dict[int, object]", env)
                args = [_rebuild(s, envo) for s in node.args]
                env[node.id] = bind(node.prim, *args, **node.params)

        # Seed: cotangent of the scalar output is ones.
        out_id = forward.outputs[0]
        out_aval = forward.nodes[out_id].aval if out_id < len(forward.nodes) else None
        seed_shape = out_aval.shape if out_aval is not None else ()
        ct: dict[int, Boxed] = {
            out_id: trace.pure(np.ones(_ishape(seed_shape), dtype=np.float64))
        }

        # Reverse pass: distribute each node's cotangent to its operands.
        for node in reversed(forward.nodes):
            if node.prim is _INPUT or node.prim is _CONST:
                continue
            g = ct.get(node.id)
            if g is None:
                continue
            operand_specs, vjp_params = _decompose(node.prim, node.args, node.params)
            operand_vals = [_operand_val(s, env) for s in operand_specs]
            contribs = _vjp_on_graph(
                node.prim, operand_vals, vjp_params, g, env[node.id]
            )
            for spec, contrib in zip(operand_specs, contribs):
                if contrib is None or not isinstance(spec, Ref):
                    continue
                want = _shape_of(env[spec.id])
                if _shape_of(contrib) != want:
                    contrib = _d_unbroadcast(contrib, want)
                ct[spec.id] = (
                    contrib
                    if spec.id not in ct
                    else bind(ops.d_add, ct[spec.id], contrib)
                )

        # Outputs: (value, grads) with grads a flat tuple, one per input leaf.
        grads: list[Boxed] = []
        for inp in forward.inputs:
            gco = ct.get(inp)
            if gco is None:
                aval = forward.nodes[inp].aval
                gco = trace.pure(np.zeros(_ishape(aval.shape), dtype=aval.dtype))
            grads.append(gco)
        out_ids = [trace.output_id(env[out_id])] + [
            trace.output_id(grad) for grad in grads
        ]

    _, out_treedef = tree_flatten((None, tuple(None for _ in grads)))
    inputs = [nd.id for nd in builder.nodes if nd.prim is _INPUT]
    in_avals = [builder.nodes[i].aval for i in inputs]
    return Graph(builder.nodes, inputs, out_ids, out_treedef, in_avals)


# ---------------------------------------------------------------------------
# jit: the ergonomic graph-mode entrypoint.
# ---------------------------------------------------------------------------
def _cache_key(args: tuple) -> tuple:
    """A key over the inputs' shapes/dtypes (and any non-numeric *static* leaf values,
    which select the captured graph). Same key => reuse the optimized graph; a new
    shape re-captures."""
    key: list = []
    for a in args:
        leaves, treedef = tree_flatten(a)
        key.append(repr(treedef))
        for leaf in leaves:
            if _is_numeric(leaf):
                arr = np.asarray(cast(Any, leaf))
                key.append((tuple(arr.shape), str(arr.dtype)))
            else:  # a bool flag / None / string baked into the graph
                key.append(("static", repr(leaf)))
    return tuple(key)


def _regroup_grads(flat: list, args: tuple) -> tuple:
    """Reshape ``grad_graph``'s flat per-leaf grads back into per-arg pytrees matching
    the inputs (``None`` for any non-numeric, non-differentiated leaf), so the output
    mirrors ``value_and_grad``."""
    it = iter(flat)
    out = []
    for a in args:
        leaves, treedef = tree_flatten(a)
        rebuilt = [next(it) if _is_numeric(leaf) else None for leaf in leaves]
        out.append(tree_unflatten(treedef, rebuilt))
    return tuple(out)


def jit(f: Callable[..., PyTree], grad: bool = False) -> Callable[..., PyTree]:
    """Graph-mode wrapper: capture ``f`` once per input shape/dtype, optimize the graph,
    cache it, and replay on later calls -- so the optimization passes (CSE/DCE/fusion)
    amortize over a training/inference loop. Falls back to eager if ``f`` can't be traced
    (e.g. data-dependent control flow).

    ``grad=False`` returns ``f``'s output. ``grad=True`` returns ``(value, grads)`` like
    :func:`value_and_grad`, but computed from a single *optimized forward+backward* graph
    (via :func:`grad_graph`) -- the gradient comes from the graph's backward nodes, with
    cross-pass CSE applied, and no eager ``.backward()`` pass.
    """
    cache: dict[tuple, Graph] = {}

    def _eager(args: tuple) -> PyTree:
        if not grad:
            return f(*args)
        from pycograd.transforms import value_and_grad

        return value_and_grad(f)(*args)

    def run(*args: PyTree) -> PyTree:
        key = _cache_key(args)
        graph = cache.get(key)
        if graph is None:
            try:
                from pycograd.passes import optimize

                fwd = capture(f, *args)
                graph = optimize(grad_graph(fwd) if grad else fwd)
            except Exception:
                return _eager(args)  # untraceable (dynamic control flow) -> eager
            cache[key] = graph
        out = tree_map(_value, eval_graph(graph, *args))
        if grad:
            value, flat = cast("tuple[Any, Any]", out)
            return value, _regroup_grads(list(flat), args)
        return out

    return run

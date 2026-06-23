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
    _rebuild,
)
from pycograd.shapes import ShapedArray
from pycograd.tensor import _d_unbroadcast
from pycograd.trace import bind, new_main
from pycograd.tree import tree_flatten


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
    if prim is ops.d_concatenate:  # ([parts], axis=...)
        return list(args[0]), dict(params)
    if prim is ops.d_where:  # (cond, a, b); cond is a param, a/b the operands
        return [args[1], args[2]], {"cond": _const(args[0]), **params}
    return list(args), dict(params)


# ---------------------------------------------------------------------------
# Per-primitive graph-building VJP rules for the ops whose eager rule reads a primal's
# *data* (a mask) rather than just its shape. Populated by ``ad_graph_mask`` (G2);
# falls back to ``_VJP_FOR`` for everything else.
# ---------------------------------------------------------------------------
GraphVJP = Callable[..., "list[Boxed]"]
_VJP_GRAPH: dict[Any, GraphVJP] = {}


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

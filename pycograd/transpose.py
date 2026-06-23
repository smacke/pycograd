# -*- coding: utf-8 -*-
"""Reverse mode derived from forward mode: ``vjp = transpose ∘ linearize``.

``linearize(f, *primals)`` reuses the JVP rules (``_JVP_FOR``) — it runs ``jvp`` under
``capture``, recording the tangent computation as a graph that is *linear* in the
tangents (primal-derived values become residual constants). ``transpose`` then flips
that linear graph with a small, derivative-free ``_TRANSPOSE`` table. The local
derivatives (``cos``, ``1-tanh²``, …) therefore live only in the JVP rules; reverse
mode is a transpose of them, not a second copy.

This is the *graph-mode* reverse path (mechanism #3). For why it coexists with the two
eager reverse paths -- the fast base ``.grad`` tape and the higher-order
``_backward_differentiable`` -- see the "Three reverse-mode mechanisms" overview in
:mod:`pycograd.tensor`.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, cast

import numpy as np

from pycograd import ops
from pycograd._typing import Boxed
from pycograd.ad_graph import _decompose, _operand_val, _shape_of, _vjp_on_graph
from pycograd.capture import (
    _CONST,
    _INPUT,
    Graph,
    GraphTrace,
    Ref,
    _Builder,
    _rebuild,
    capture,
)
from pycograd.passes import _spec_refs as _refs
from pycograd.passes import dce
from pycograd.shapes import ShapedArray
from pycograd.tensor import _d_unbroadcast
from pycograd.trace import bind, new_main
from pycograd.transforms import jvp
from pycograd.tree import PyTree, tree_flatten, tree_map


def _ishape(shape: Any) -> "tuple[int, ...]":
    return cast("tuple[int, ...]", tuple(shape))


def linearize(f: Callable[..., PyTree], *primals: PyTree) -> "tuple[Graph, int]":
    """Capture the JVP of ``f`` at ``primals`` as a graph. Returns ``(graph, n_primal)``
    where ``graph`` has inputs ``[*primal leaves, *tangent leaves]`` and outputs
    ``[*primal-output leaves, *tangent-output leaves]``, and ``n_primal`` is the number
    of primal input leaves (so the caller knows which inputs are the tangents). The
    tangent-output leaves are *linear* in the tangent inputs."""
    # Placeholder tangents: capture only reads input shapes, so zeros of the right shape.
    tangents = tuple(
        tree_map(lambda a: np.zeros_like(np.asarray(a)), p) for p in primals
    )

    def runner(ps: tuple, ts: tuple) -> PyTree:
        return jvp(f, ps, ts)  # (primal_out, tangent_out)

    # Run directly (not instrumented) so the closure over ``f`` survives; the inner
    # ``jvp`` instruments ``f``, and with the capture ``GraphTrace`` active every op is
    # still recorded (validated by the jvp-under-capture spike).
    runner._pycograd_run_directly = True  # type: ignore[attr-defined]

    graph = capture(runner, tuple(primals), tangents)
    n_primal = sum(len(tree_flatten(p)[0]) for p in primals)
    return graph, n_primal


def _forward_reachable(graph: Graph, seeds: "set[int]") -> "set[int]":
    """The nodes tangent-*linear*: reachable forward from the tangent inputs ``seeds``.
    Everything else is a residual constant (a primal-derived value)."""
    reach = set(seeds)
    for node in graph.nodes:  # SSA order, so producers precede consumers
        if node.prim is _INPUT or node.prim is _CONST:
            continue
        if any(r in reach for s in node.args for r in _refs(s)):
            reach.add(node.id)
    return reach


def vjp_graph(f: Callable[..., PyTree], *primals: PyTree) -> Graph:
    """Reverse mode as ``transpose ∘ linearize``: linearize ``f`` to a graph linear in
    the tangents, then flip that linear part. Returns a :class:`Graph` whose output is
    ``(value, grads)`` (grads a flat tuple, one per primal leaf) -- the same shape as
    :func:`pycograd.ad_graph.grad_graph`, but reverse mode is derived from forward mode.
    Only the *linear* VJP rules are exercised; the nonlinear derivatives entered as
    residual constants during linearize."""
    lin, n_primal = linearize(f, *primals)
    if len(lin.outputs) != 2:
        raise ValueError("vjp_graph: expected a single scalar output to differentiate")
    primal_out_id, tangent_out_id = lin.outputs
    tangent_ids = lin.inputs[n_primal:]
    linear = _forward_reachable(lin, set(tangent_ids))

    builder = _Builder()
    with new_main(GraphTrace, builder) as main:
        trace = GraphTrace(main)
        # Replay the linearized graph (records primal + residual nodes; the linear
        # forward nodes are recorded too but die in DCE below).
        env: dict[int, Boxed] = {}
        for node in lin.nodes:
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

        # Seed the output tangent's cotangent with ones, transpose the linear nodes.
        seed_shape = _ishape(lin.nodes[tangent_out_id].aval.shape)
        ct: dict[int, Boxed] = {tangent_out_id: trace.pure(np.ones(seed_shape))}
        for node in reversed(lin.nodes):
            if node.prim is _INPUT or node.prim is _CONST or node.id not in linear:
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
                # cotangent flows only to *linear* operands; residuals are constants.
                if (
                    contrib is None
                    or not isinstance(spec, Ref)
                    or spec.id not in linear
                ):
                    continue
                want = _shape_of(env[spec.id])
                if _shape_of(contrib) != want:
                    contrib = _d_unbroadcast(contrib, want)
                ct[spec.id] = (
                    contrib
                    if spec.id not in ct
                    else bind(ops.d_add, ct[spec.id], contrib)
                )

        grads: list[Boxed] = []
        for t in tangent_ids:
            gc = ct.get(t)
            if gc is None:
                gc = trace.pure(np.zeros(_ishape(lin.nodes[t].aval.shape)))
            grads.append(gc)
        out_ids = [trace.output_id(env[primal_out_id])] + [
            trace.output_id(g) for g in grads
        ]

    _, out_treedef = tree_flatten((None, tuple(None for _ in grads)))
    all_inputs = [nd.id for nd in builder.nodes if nd.prim is _INPUT]
    in_avals = [builder.nodes[i].aval for i in all_inputs]
    graph = Graph(builder.nodes, all_inputs, out_ids, out_treedef, in_avals)
    # The tangent-forward nodes are dead (grads depend only on residuals + the seed);
    # DCE removes them, leaving the tangent inputs unreferenced -- so the gradient graph
    # takes only the primal inputs (the first ``n_primal`` in replay order).
    graph = dce(graph)
    primal_inputs = all_inputs[:n_primal]
    return replace(graph, inputs=primal_inputs, in_avals=in_avals[:n_primal])

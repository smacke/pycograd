# -*- coding: utf-8 -*-
"""Reverse mode derived from forward mode: ``vjp = transpose ∘ linearize``.

``linearize(f, *primals)`` reuses the JVP rules (``_JVP_FOR``) — it runs ``jvp`` under
``capture``, recording the tangent computation as a graph that is *linear* in the
tangents (primal-derived values become residual constants). ``transpose`` then flips
that linear graph with a small, derivative-free ``_TRANSPOSE`` table. The local
derivatives (``cos``, ``1-tanh²``, …) therefore live only in the JVP rules; reverse
mode is a transpose of them, not a second copy. See the plan for the full rationale.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from pycograd.capture import Graph, capture
from pycograd.transforms import jvp
from pycograd.tree import PyTree, tree_flatten, tree_map


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

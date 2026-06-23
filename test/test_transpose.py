# -*- coding: utf-8 -*-
"""Spike + tests for linearize/transpose. The linchpin: does ``jvp`` compose *under*
``capture`` (jvp inner, capture outer), so the tangent computation gets recorded as a
linear graph?"""
import pytest

np = pytest.importorskip("numpy")

from pycograd import jvp  # noqa: E402
from pycograd.capture import capture, eval_graph  # noqa: E402
from pycograd.tensor import _value  # noqa: E402
from pycograd.transpose import linearize  # noqa: E402


def _smooth(x):
    return np.sum(np.tanh(x) * np.tanh(x))


def _tangent_of(x, t):
    # the tangent output of jvp -- a function of (primal x, tangent t), LINEAR in t.
    _, tangent_out = jvp(_smooth, (x,), (t,))
    return tangent_out


def test_jvp_under_capture_spike():
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((3, 4))
    t0 = rng.standard_normal((3, 4))

    # linearize: capture the tangent computation as a graph in (x, t).
    lin = capture(_tangent_of, x0, t0)

    # at fixed x0, evaluating the graph at t0 matches jvp's tangent.
    got = np.asarray(_value(eval_graph(lin, x0, t0)))
    _, ref = jvp(_smooth, (x0,), (t0,))
    assert np.allclose(got, np.asarray(_value(ref))), "graph tangent != jvp tangent"

    # ...and it is LINEAR in t (the property transpose relies on).
    got2 = np.asarray(_value(eval_graph(lin, x0, 2.0 * t0)))
    assert np.allclose(got2, 2.0 * got), "tangent graph is not linear in t"


def test_linearize_matches_jvp():
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((3, 4))
    t0 = rng.standard_normal((3, 4))
    graph, n_primal = linearize(_smooth, x0)
    assert n_primal == 1  # one primal leaf

    po, to = eval_graph(graph, (x0,), (t0,))
    rpo, rto = jvp(_smooth, (x0,), (t0,))
    assert np.allclose(float(_value(po)), float(_value(rpo)))  # primal value
    assert np.allclose(float(_value(to)), float(_value(rto)))  # tangent at t0

    _, to2 = eval_graph(graph, (x0,), (2.0 * t0,))
    assert np.allclose(float(_value(to2)), 2.0 * float(_value(to)))  # linear in t

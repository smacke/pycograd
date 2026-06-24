# -*- coding: utf-8 -*-
"""Tests for the ``%load_ext pycograd`` IPython/Jupyter extension.

Two layers:

* a fast **fake-shell wiring test** that exercises ``load_ipython_extension`` /
  ``unload_ipython_extension`` against a stub shell (needs pipescript on the path
  but not IPython);
* **real-notebook subprocess tests** that drive a fresh in-subprocess IPython
  shell (modeled on pipescript's ``test_reexecution.py``) so the singleton shell +
  tracer state can't leak across the session.

Both layers ``importorskip`` their dependencies, so they skip cleanly when
pipescript / IPython aren't installed rather than failing.
"""
import subprocess
import sys
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Fake-shell wiring test (no IPython required).
# ---------------------------------------------------------------------------
class _FakeExtensionManager:
    def __init__(self) -> None:
        self.loaded = set()
        self.requests = []

    def load_extension(self, name):
        # Record the request; we can't actually load pipescript's IPython
        # extension without a real shell, but the tracers it registers are
        # importable, which is all the wiring below needs.
        self.requests.append(name)
        self.loaded.add(name)
        return None


class _FakeShell:
    def __init__(self) -> None:
        self.user_ns = {}
        self.extension_manager = _FakeExtensionManager()


def test_extension_wiring_with_fake_shell():
    pytest.importorskip("pipescript")
    from pipescript.tracers.macro_tracer import MacroTracer
    from pipescript.tracers.pipeline_tracer import PipelineTracer

    import pycograd
    from pycograd import frozen, tied
    from pycograd.extension import _autodiff_hook

    shell = _FakeShell()
    pycograd.load_ipython_extension(shell)
    try:
        # 1. pipescript was requested; 2. the autodiff pipe hook is registered once;
        # 3. the params{} macro is wired; 4. frozen/tied are in the user namespace.
        assert "pipescript" in shell.extension_manager.requests
        assert _autodiff_hook in PipelineTracer.application_hooks
        assert PipelineTracer.application_hooks.count(_autodiff_hook) == 1
        assert "params" in MacroTracer.namespace_block_macros
        assert shell.user_ns.get("frozen") is frozen
        assert shell.user_ns.get("tied") is tied

        # Re-loading must not duplicate the hook.
        pycograd.load_ipython_extension(shell)
        assert PipelineTracer.application_hooks.count(_autodiff_hook) == 1
    finally:
        pycograd.unload_ipython_extension(shell)

    # Unload reverses everything we wired up.
    assert _autodiff_hook not in PipelineTracer.application_hooks
    assert "params" not in MacroTracer.namespace_block_macros
    assert "frozen" not in shell.user_ns and "tied" not in shell.user_ns


def test_is_user_function_accepts_ipython_cells():
    # A function defined in an IPython cell has a ``<ipython-input-N-...>``
    # filename; its source is retrievable via linecache, so it must be treated as
    # instrumentable user code (this is what makes ``x |> my_helper`` differentiate
    # when ``my_helper`` is defined in a notebook cell). Synthesize one with that
    # filename so the check runs without a live IPython shell.
    from pycograd.tracer import _is_user_function

    ns = {}
    exec(
        compile("def cell_fn(z):\n    return z", "<ipython-input-1-deadbeef>", "exec"),
        ns,
    )
    assert _is_user_function(ns["cell_fn"]) is True

    exec(compile("def repl_fn(z):\n    return z", "<stdin>", "exec"), ns)
    assert _is_user_function(ns["repl_fn"]) is False  # <stdin>/<string> still rejected


def test_autodiff_hook_resolves_bare_stage_under_delegate_backend():
    # A bare-function pipe stage (`x |> relu`) is left raw during plain inference (fast
    # path) but must be instrumented when a delegate backend is active -- otherwise the
    # staged helper's `np.maximum` would meet a raw framework tensor when compiling an
    # ambient DSL net. (The Var/Tracer cases are covered by the autodiff/vmap suites.)
    import numpy as np

    from pycograd.backends import activate, get_backend
    from pycograd.extension import _autodiff_hook

    def relu(z):
        return np.maximum(0.0, z)

    assert (
        _autodiff_hook(relu, np.zeros(3)) is relu
    )  # no backend -> raw, fast inference
    pytest.importorskip("torch")
    with activate(get_backend("torch")):
        resolved = _autodiff_hook(relu, np.zeros(3))  # delegate active -> instrumented
    assert resolved is not relu


def test_extension_unload_preserves_user_named_frozen():
    # If the user already bound ``frozen``/``tied``, we must not clobber or remove
    # their value on load/unload.
    pytest.importorskip("pipescript")
    import pycograd

    shell = _FakeShell()
    sentinel = object()
    shell.user_ns["frozen"] = sentinel
    pycograd.load_ipython_extension(shell)
    try:
        assert shell.user_ns["frozen"] is sentinel  # setdefault did not clobber
    finally:
        pycograd.unload_ipython_extension(shell)
    assert shell.user_ns["frozen"] is sentinel  # unload left the user's value alone


# ---------------------------------------------------------------------------
# Real-notebook subprocess tests.
# ---------------------------------------------------------------------------
# Each test runs in a *fresh subprocess* because IPython's singleton shell and the
# process-wide pyccolo tracer stacks leak state across a session.
_PROLOGUE = """
import sys
from IPython.testing.globalipapp import get_ipython

ip = get_ipython()
ip.run_line_magic("load_ext", "pycograd")
ip.run_cell("pass")  # warm up: let pipescript finish first-cell initialization
"""


def _run_probe(body: str):
    pytest.importorskip("IPython")
    pytest.importorskip("pipescript")
    probe = _PROLOGUE + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0 and proc.stdout.strip().endswith("OK"), (
        proc.stdout + proc.stderr
    )


def test_load_ext_auto_loads_pipescript():
    _run_probe(
        """
        assert "pipescript" in ip.extension_manager.loaded, (
            "pipescript was not auto-loaded by %load_ext pycograd"
        )
        print("OK")
        """
    )


def test_params_brace_block_in_cell():
    _run_probe(
        """
        ip.run_cell("import numpy as np")
        from pycograd import Param
        r = ip.run_cell("model = params{\\n  w = np.zeros(3)\\n  b = frozen[np.ones(2)]\\n}")
        assert r.error_in_exec is None, r.error_in_exec
        model = ip.user_ns["model"]
        assert isinstance(model["w"], Param) and model["w"].trainable
        assert isinstance(model["b"], Param) and not model["b"].trainable
        print("OK")
        """
    )


def test_autodiff_pipe_in_cell():
    _run_probe(
        """
        import numpy as np
        ip.run_cell("import numpy as np; from pycograd import Var")
        ip.run_cell("x = Var(np.array([1.0, 2.0, 3.0]))")
        r = ip.run_cell("loss = (x |> np.exp |> np.sum)")
        assert r.error_in_exec is None, r.error_in_exec
        ip.run_cell("loss.backward()")
        g = ip.user_ns["x"].grad
        assert np.allclose(g, np.exp([1.0, 2.0, 3.0])), g
        print("OK")
        """
    )


def test_autodiff_pointfree_power_in_bare_pipe_lambda():
    # A *bare top-level pipe lambda* using a point-free binop: ``np.tanh ** 2`` means the
    # stage ``x -> np.tanh(x) ** 2``. The lambda runs un-instrumented by default, so this
    # exercises both halves of the fix: (1) ``AutodiffTracer._augmented_definition_for``
    # recovers the lambda's retained augmented AST so ``value_and_grad``/``capture``
    # re-instrument it (instead of run-directly), and (2) the ``before_binop`` handler then
    # lifts ``np.tanh ** 2`` to a differentiable stage. Value and gradient must be exact.
    _run_probe(
        """
        import numpy as np
        ip.run_cell("import numpy as np; from pycograd import value_and_grad, capture")
        ip.run_cell("rng = np.random.default_rng(0); xs = rng.standard_normal((3, 4))")
        ip.run_cell("redundant = ($ |> np.tanh ** 2 |> np.sum)")
        r = ip.run_cell("v, (g,) = value_and_grad(redundant)(xs)")
        assert r.error_in_exec is None, r.error_in_exec
        xs = ip.user_ns["xs"]
        assert np.allclose(ip.user_ns["v"], np.sum(np.tanh(xs) ** 2))
        expected = 2 * np.tanh(xs) * (1 - np.tanh(xs) ** 2)
        assert np.allclose(ip.user_ns["g"], expected), ip.user_ns["g"]
        # and it captures into a graph with an explicit pow node (not a degraded op)
        gr = ip.run_cell("graph = capture(redundant, xs)")
        assert gr.error_in_exec is None, gr.error_in_exec
        assert "pow" in ip.user_ns["graph"].pretty()
        print("OK")
        """
    )


def test_autodiff_compose_power_distinct_from_power():
    # pipescript's compose op ``.**`` must *compose* (``np.tanh .** 2`` -> tanh(tanh(x))),
    # distinct from the point-free power ``**`` (``np.tanh ** 2`` -> tanh(x)**2). pycograd
    # tells them apart by the ``.**`` augmentation on the (lowered) Pow node.
    _run_probe(
        """
        import numpy as np
        ip.run_cell("import numpy as np; from pycograd import capture, value_and_grad")
        ip.run_cell("rng = np.random.default_rng(0); xs = rng.standard_normal((3, 4))")
        xs = np.random.default_rng(0).standard_normal((3, 4))
        ip.run_cell("comp = ($ |> np.tanh .** 2 |> np.sum)")
        ip.run_cell("gc = capture(comp, xs)")
        pretty = ip.user_ns["gc"].pretty()
        assert pretty.count("tanh") == 2 and "pow" not in pretty, pretty  # composition
        r = ip.run_cell("vc, (gxc,) = value_and_grad(comp)(xs)")
        assert r.error_in_exec is None, r.error_in_exec
        assert np.allclose(ip.user_ns["vc"], np.sum(np.tanh(np.tanh(xs))))
        expected = (1 - np.tanh(np.tanh(xs)) ** 2) * (1 - np.tanh(xs) ** 2)
        assert np.allclose(ip.user_ns["gxc"], expected), ip.user_ns["gxc"]
        # the plain power form stays a single tanh -> pow
        ip.run_cell("pf = ($ |> np.tanh ** 2 |> np.sum)")
        ip.run_cell("gp = capture(pf, xs)")
        pp = ip.user_ns["gp"].pretty()
        assert pp.count("tanh") == 1 and "pow" in pp, pp
        print("OK")
        """
    )


def test_leading_value_pipe_stage_with_integer_index_under_backend():
    # ``|> idx |> forward`` must equal ``forward(idx)`` even when ``forward`` gathers a table
    # with the integer index ``idx`` and runs under a delegate backend. Regression for a bug
    # where the binop-arg coercion handler ran on the pipe seed ``idx`` (the ``|>`` lowers to
    # a ``BitOr`` BinOp), the torch backend lifted that int index to a *float* tensor, and
    # ``table[idx]`` then raised "tensors used as indices must be long, int, byte or bool".
    pytest.importorskip("torch")
    _run_probe(
        """
        import numpy as np
        ip.run_cell("import numpy as np")
        ip.run_cell("from pycograd import params")
        ip.run_cell("rng = np.random.default_rng(0)")
        ip.run_cell("idx = np.array([0, 3, 7, 2, 5, 1])")
        ip.run_cell("Yt = np.eye(3)[[0, 1, 2, 0, 1, 2]]")
        ip.run_cell(
            "def softmax_ce(logits, onehot):\\n"
            " z = logits - np.max(logits, axis=1, keepdims=True)\\n"
            " logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))\\n"
            " return -np.mean(np.sum(onehot * logp, axis=1))"
        )
        src = (
            "with params{\\n"
            "    table = rng.standard_normal((16, 8))\\n"
            "    w = 0.3 * rng.standard_normal((8, 3))\\n"
            "    b = np.zeros(3)\\n"
            "} as weights:\\n"
            "    forward = $ |> table[$] |> $ @ w + b\\n"
            "    v_call, _ = weights.grad(|> forward(idx) |> softmax_ce($, Yt), backend='torch')\\n"
            "    v_pipe, _ = weights.grad(|> idx |> forward |> softmax_ce($, Yt), backend='torch')\\n"
        )
        r = ip.run_cell(src)
        assert r.error_in_exec is None, r.error_in_exec
        v_call, v_pipe = float(ip.user_ns["v_call"]), float(ip.user_ns["v_pipe"])
        assert np.allclose(v_call, v_pipe), (v_call, v_pipe)
        print("OK")
        """
    )


def test_autodiff_edited_pipe_lambda_not_stale():
    # Editing a pipe lambda and re-running must use the NEW pipeline, not a stale one.
    # The lambda's augmented node is recovered from the shared registry to re-instrument it;
    # since re-running a cell yields a *new* co_filename, the recovery is scoped per-file so
    # it never resolves to the previous run's lambda. Each capture must reflect its own op.
    _run_probe(
        """
        import numpy as np
        ip.run_cell("import numpy as np; from pycograd import capture")
        ip.run_cell("rng = np.random.default_rng(0); xs = rng.standard_normal((3, 4))")
        ip.run_cell("redundant = ($ |> np.tanh ** 2 |> np.sum)")
        ip.run_cell("gA = capture(redundant, xs)")
        assert "tanh" in ip.user_ns["gA"].pretty() and "sin" not in ip.user_ns["gA"].pretty()
        ip.run_cell("redundant = ($ |> np.sin ** 3 |> np.sum)")  # edit + re-run
        ip.run_cell("gB = capture(redundant, xs)")
        pretty = ip.user_ns["gB"].pretty()
        assert "sin" in pretty and "tanh" not in pretty, pretty  # NEW pipeline, not stale
        print("OK")
        """
    )


def test_autodiff_pipe_through_user_helper_in_cell():
    # `y |> relu` instruments the helper on demand (exercises instrumented()
    # self-activation), so the subgradient flows without an always-on tracer.
    _run_probe(
        """
        import numpy as np
        ip.run_cell("import numpy as np; from pycograd import Var")
        ip.run_cell("def relu(z):\\n    return np.maximum(z, 0.0)")
        ip.run_cell("y = Var(np.array([-1.0, 2.0, -3.0, 4.0]))")
        r = ip.run_cell("loss = (y |> relu |> np.sum)")
        assert r.error_in_exec is None, r.error_in_exec
        ip.run_cell("loss.backward()")
        g = ip.user_ns["y"].grad
        expected = (np.array([-1.0, 2.0, -3.0, 4.0]) > 0).astype(float)
        assert np.allclose(g, expected), g
        print("OK")
        """
    )


def test_autodiff_pipe_helper_with_pipescript_body():
    # A helper whose *own body* uses pipescript syntax. The helper is woven by
    # pipescript at cell-compile time, so its linecache source is the lowered ``|``
    # form; re-instrumenting that source would degrade the pipes to bitwise-or. The
    # fix re-instruments from the retained *augmented* AST, so the pipes survive and
    # the gradient flows through ``z |> np.exp |> np.sum``.
    _run_probe(
        """
        import numpy as np
        ip.run_cell("import numpy as np; from pycograd import Var")
        ip.run_cell("def stage(z):\\n    return z |> np.exp |> np.sum")
        ip.run_cell("x = Var(np.array([1.0, 2.0, 3.0]))")
        r = ip.run_cell("loss = (x |> stage)")
        assert r.error_in_exec is None, r.error_in_exec
        ip.run_cell("loss.backward()")
        g = ip.user_ns["x"].grad
        assert np.allclose(g, np.exp([1.0, 2.0, 3.0])), g
        print("OK")
        """
    )


def test_autodiff_pipe_helper_with_placeholder_macro_body():
    # Same path, but the woven body uses a pipescript placeholder lambda (``$``)
    # rather than a bare function reference -- exercises augmentation preservation for
    # a macro node, not just a plain pipe.
    _run_probe(
        """
        import numpy as np
        ip.run_cell("import numpy as np; from pycograd import Var")
        ip.run_cell("def scale_sum(z):\\n    return z |> ($ * 2.0) |> np.sum")
        ip.run_cell("x = Var(np.array([1.0, 2.0, 3.0]))")
        r = ip.run_cell("loss = (x |> scale_sum)")
        assert r.error_in_exec is None, r.error_in_exec
        ip.run_cell("loss.backward()")
        g = ip.user_ns["x"].grad
        assert np.allclose(g, [2.0, 2.0, 2.0]), g
        print("OK")
        """
    )


def test_autodiff_woven_helper_calls_nested_plain_helper():
    # Regression for the notebook's Transformer block: a woven helper (``outer``, whose
    # body uses ``|>``) calls a *nested plain* helper (``inner``, a bare-numpy reduction)
    # and is reached through an objective (``f``) that is instrumented first. Recompiling
    # ``f`` rebuilds its cell's per-file bookkeeping, so ``outer`` must be recovered from
    # the global node table -- otherwise it runs un-instrumented, ``inner`` is never
    # intercepted, and ``np.max`` on a Var blows up (``d_max() got an unexpected kwarg``).
    # All three defs share one cell (one co_filename); the call happens in a later cell.
    _run_probe(
        """
        import numpy as np
        ip.run_cell("import numpy as np; from pycograd import Var, value_and_grad")
        ip.run_cell(
            "def inner(z):\\n    return z - np.max(z)\\n"
            "def outer(z):\\n    return inner(z) |> np.sum\\n"
            "def f(z):\\n    return outer(z)"
        )
        r = ip.run_cell("v, (g,) = value_and_grad(f)(np.array([1.0, 2.0, 3.0]))")
        assert r.error_in_exec is None, r.error_in_exec
        v, g = ip.user_ns["v"], ip.user_ns["g"]
        assert np.isclose(v, -3.0), v               # sum(z - max(z))
        assert np.allclose(g, [1.0, 1.0, -2.0]), g  # 1 - n*[i == argmax], so inner ran
        print("OK")
        """
    )


def test_autodiff_pipe_helper_mixes_pipe_and_plain_numpy():
    # The hard case: a woven helper mixing an *un-piped* numpy call on a Var with a
    # pipe in the same body. The un-piped ``np.exp(z)`` needs before_call woven in,
    # while the pipe needs its augmentation preserved -- re-instrumenting from the
    # retained augmented AST satisfies both at once.
    _run_probe(
        """
        import numpy as np
        ip.run_cell("import numpy as np; from pycograd import Var")
        ip.run_cell("def stage(z):\\n    a = np.exp(z)\\n    return a |> np.sum")
        ip.run_cell("x = Var(np.array([1.0, 2.0, 3.0]))")
        r = ip.run_cell("loss = (x |> stage)")
        assert r.error_in_exec is None, r.error_in_exec
        ip.run_cell("loss.backward()")
        g = ip.user_ns["x"].grad
        assert np.allclose(g, np.exp([1.0, 2.0, 3.0])), g
        print("OK")
        """
    )

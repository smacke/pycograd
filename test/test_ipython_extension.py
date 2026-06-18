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

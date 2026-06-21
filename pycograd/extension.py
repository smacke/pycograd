# -*- coding: utf-8 -*-
"""IPython/Jupyter extension: ``%load_ext pycograd``.

Loading the extension turns a notebook into a pycograd DSL session. It:

1. loads `pipescript <https://github.com/smacke/pipescript>`_ if it isn't already
   (so ``|>`` pipes, brace blocks, and macros are available);
2. registers an autodiff hook on ``PipelineTracer.application_hooks`` so a pipe
   over a ``Var`` differentiates -- ``x |> np.exp`` swaps in the differentiable
   primitive, and ``x |> my_helper`` instruments the helper on demand;
3. wires the ``params{ w = ...; b = frozen[...] }`` brace surface via
   :func:`pycograd.params.register_pipescript_params_macro`;
4. injects the two names a ``params{...}`` block needs to resolve (``frozen`` /
   ``tied``) into the user namespace. ``params`` itself is exposed as a builtin by
   the macro; import the rest of the API (``Var``, ``value_and_grad``, ...) as
   usual.

This is the DSL surface only: ordinary ``np.exp(x)`` outside a pipe is *not*
auto-differentiated (use ``value_and_grad``/``grad`` or a pipe). pyccolo's
``instrumented()`` self-activates per call, so the pipe hook alone is enough --
no always-on per-cell tracer.

pipescript is an optional dependency (the ``pipescript`` / ``notebook`` extra);
all pipescript imports here are deferred so importing pycograd never requires it.
"""
from __future__ import annotations

from typing import Any

from pycograd._typing import Boxed, Prim
from pycograd.params import frozen as _frozen
from pycograd.params import tied as _tied
from pycograd.tensor import Var
from pycograd.tracer import resolve_call

# The DSL essentials a ``params{...}`` block references by name. ``params`` is
# bound as a builtin by the macro registration, so it is not listed here.
_DSL_NAMES = {"frozen": _frozen, "tied": _tied}

_PIPESCRIPT_MISSING_MSG = (
    "pycograd's IPython extension needs pipescript; install it with "
    "`pip install pycograd[notebook]` (or `pip install pipescript`)."
)


def _autodiff_hook(func: Prim, value: Boxed) -> Prim:
    """A ``PipelineTracer`` application hook: when a ``Var`` flows into a pipe,
    resolve the applied function to its autodiff-aware form (numpy/math swap, or
    on-demand helper instrumentation). Defined at module scope so it has a stable
    identity for idempotent registration and clean removal."""
    return resolve_call(func) if isinstance(value, Var) else func


def load_ipython_extension(shell: Any) -> None:
    """Entry point for ``%load_ext pycograd`` -- see the module docstring."""
    try:
        import pipescript  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without pipescript
        raise ImportError(_PIPESCRIPT_MISSING_MSG) from exc

    from pipescript.tracers.pipeline_tracer import PipelineTracer

    from pycograd.params import register_pipescript_params_macro

    # 1. Ensure pipescript's tracers are active per cell. ExtensionManager.load_
    #    extension no-ops if pipescript is already loaded, so this is idempotent.
    shell.extension_manager.load_extension("pipescript")

    # 2. Autodiff through pipes (guard against a duplicate on re-load).
    if _autodiff_hook not in PipelineTracer.application_hooks:
        PipelineTracer.application_hooks.append(_autodiff_hook)

    # 3. The ``params{ ... }`` brace surface.
    register_pipescript_params_macro()

    # 4. Make ``frozen``/``tied`` resolvable inside ``params{...}`` blocks without
    #    clobbering a user variable of the same name (removed on unload only if it
    #    is still our object).
    user_ns = getattr(shell, "user_ns", None)
    if user_ns is not None:
        for name, obj in _DSL_NAMES.items():
            user_ns.setdefault(name, obj)


def unload_ipython_extension(shell: Any) -> None:
    """Reverse :func:`load_ipython_extension`. Leaves pipescript itself loaded."""
    try:
        from pipescript.tracers.macro_tracer import MacroTracer
        from pipescript.tracers.pipeline_tracer import PipelineTracer
    except ImportError:  # pragma: no cover - nothing was wired up without pipescript
        return

    while _autodiff_hook in PipelineTracer.application_hooks:
        PipelineTracer.application_hooks.remove(_autodiff_hook)

    MacroTracer.namespace_block_macros.pop("params", None)
    MacroTracer.static_macros.pop("params", None)

    import builtins

    from pycograd.params import params as _params

    if getattr(builtins, "params", None) is _params:
        delattr(builtins, "params")

    user_ns = getattr(shell, "user_ns", None)
    if user_ns is not None:
        for name, obj in _DSL_NAMES.items():
            if user_ns.get(name) is obj:
                del user_ns[name]

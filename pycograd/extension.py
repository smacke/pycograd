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
4. injects the ``params`` / ``frozen`` / ``tied`` names into the user namespace --
   ``frozen`` / ``tied`` so a ``params{...}`` block resolves them, and ``params`` so
   it autocompletes (the macro also binds ``params`` as a builtin, but a runtime-set
   builtin is invisible to Jedi, the default completer). Import the rest of the API
   (``Var``, ``value_and_grad``, ...) as usual.

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
from pycograd.backends import active_backend_or_none
from pycograd.params import frozen as _frozen
from pycograd.params import params as _params
from pycograd.params import tied as _tied
from pycograd.tensor import Var
from pycograd.trace import Tracer
from pycograd.tracer import resolve_call

# The DSL essentials made resolvable by name in the user namespace. ``params`` is
# also bound as a builtin by the macro registration (so the bare call/brace form
# works), but a runtime-``setattr``'d builtin is invisible to Jedi -- IPython/Jupyter's
# default, static-analysis completer -- so we inject it into ``user_ns`` here as well to
# make ``params``/``frozen``/``tied`` autocomplete after ``%load_ext pycograd``.
_DSL_NAMES = {"params": _params, "frozen": _frozen, "tied": _tied}

_PIPESCRIPT_MISSING_MSG = (
    "pycograd's IPython extension needs pipescript; install it with "
    "`pip install pycograd[notebook]` (or `pip install pipescript`)."
)


def _autodiff_hook(func: Prim, value: Boxed) -> Prim:
    """A ``PipelineTracer`` application hook: when a managed value flows into a *bare*
    function pipe stage (``x |> relu``), resolve the applied function to its autodiff-
    aware form (numpy/math swap, or on-demand helper instrumentation) so the call routes
    through the trace stack instead of hitting the raw value.

    A ``Var`` is the base-level (reverse-mode) case. A :class:`~pycograd.trace.Tracer`
    -- a ``vmap`` ``BatchTracer``, a ``jvp`` ``JVPTracer``, or an ``eval_shape``
    ``ShapedArray`` -- is a higher trace level: resolving here is what lets a bare pipe
    stage vectorize/differentiate under those transforms too (without it the staged
    function runs un-instrumented and its inner ``np.*`` meets a raw tracer, e.g.
    ``BatchTracer does not support ufuncs``). A plain array/number is left alone, so pure
    inference keeps the fast path. Defined at module scope for a stable identity (idempotent
    registration, clean removal).

    Under an active *delegate* backend (a ``compile_to`` / ``weights.grad(backend=...)`` of
    an ambient net), the piped value is the backend's own tensor -- neither a ``Var`` nor a
    ``Tracer`` -- so we resolve there too, otherwise a bare ``|> relu`` runs un-instrumented
    and its ``np.maximum`` meets a raw framework tensor (e.g. numpy on a grad tensor).
    """
    if isinstance(value, (Var, Tracer)):
        return resolve_call(func)
    be = active_backend_or_none()
    if be is not None and be.is_delegate:
        return resolve_call(func)
    return func


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

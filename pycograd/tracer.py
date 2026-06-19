# -*- coding: utf-8 -*-
"""The pyccolo seam: intercept numpy/math calls and route them to the ``d_*``
primitives.

This is the *only* module that imports pyccolo. ``AutodiffTracer`` registers a
``before_call`` handler that, for each call inside an instrumented function, either
swaps a numpy/math function for its differentiable primitive, instruments a user
helper on demand so the tape flows through its body, or wraps an un-ruled
numpy/math function so it warns if a ``Var`` flows in. ``_make_runner`` is the
bridge that instruments (or directly runs, for pyccolo ``|>`` pipe lambdas) the
function being differentiated.
"""
from __future__ import annotations

import functools
import inspect
import sysconfig
from typing import Callable

import pyccolo as pyc

from pycograd.backends import current_backend
from pycograd.ops import _is_mathy

# Directories holding stdlib / installed packages -- functions defined here are
# treated as "library" code and left alone (only *your* code is instrumented).
_LIB_DIRS = tuple(
    p
    for p in {
        sysconfig.get_paths().get("stdlib"),
        sysconfig.get_paths().get("platstdlib"),
        sysconfig.get_paths().get("purelib"),
        sysconfig.get_paths().get("platlib"),
    }
    if p
)


def _is_user_function(func: Callable[..., object]) -> bool:
    """True for a plain Python function defined in user (non-library) code.

    The autodiff primitives (``d_*`` etc.) live in :mod:`pycograd.ops` but are only
    ever reached via ``_INTERCEPT`` / operator dispatch, never as a call site inside
    instrumented user code, so they are never handed to this predicate. pycograd's
    own internals (and pyccolo) are excluded as a safety net; ``pycograd.examples``
    helpers are *not* -- they are user-level demo code and should differentiate.
    """
    if not inspect.isfunction(func):  # excludes C builtins, numpy ufuncs, methods
        return False
    module = getattr(func, "__module__", "") or ""
    if module.startswith("numpy") or module.startswith("pyccolo."):
        return False
    if module.startswith("pycograd.") and not module.startswith("pycograd.examples"):
        return False
    filename = getattr(getattr(func, "__code__", None), "co_filename", "") or ""
    # An IPython/Jupyter cell (``<ipython-input-N-...>``) is user code whose source
    # is retrievable via linecache, so instrument it -- this is what lets a helper
    # defined in a notebook cell differentiate when piped (``x |> my_helper``).
    if filename.startswith("<ipython-input-"):
        return True
    if not filename or filename.startswith("<"):  # <stdin>, <string>, ...
        return False
    return not any(filename.startswith(d) for d in _LIB_DIRS)


class AutodiffTracer(pyc.BaseTracer):
    # Instrument whichever file the differentiated function lives in.
    instrument_all_files = True

    # Opt in to weaving bare ``lambda`` targets, so ``value_and_grad`` / ``grad`` /
    # ``ParamDict.grad`` can differentiate a loss written as a lambda (including one
    # defined in a notebook cell) -- its ``np.*`` calls are intercepted like any def.
    instrument_lambdas = True

    # Never persist instrumented bytecode to ``__pycache__``. With
    # ``instrument_all_files`` on, a module first imported inside this tracer's
    # scope would otherwise have its rewritten bytecode (carrying pyccolo's emit
    # builtins) cached; a later plain import would load that stale ``.pyc`` with no
    # tracer active and fail with ``NameError: ..._PYCCOLO_EVT_EMIT``. The autodiff
    # tape is built per call via ``instrumented(func)``, so we never need the cache.
    bytecode_caching_allowed = False

    # ``instrumented`` recompiles a helper from source into a fresh function, so
    # cache each helper's instrumented version and reuse it rather than recompiling
    # on every call.
    _helpers: dict[Callable[..., object], Callable[..., object]] = {}

    def _instrument_helper(self, func: Callable[..., object]) -> Callable[..., object]:
        cached = self._helpers.get(func)
        if cached is not None:
            return cached
        # Same instrument-or-run-directly decision as the top-level differentiated
        # function (``_make_runner``); cache the result so each helper is built once.
        # ``instrumented`` recompiles a *plain* helper from source and a *pipescript*
        # helper from its retained augmented AST, so a helper whose body uses ``|>``
        # differentiates through the pipe just like one written with bare ``np.*`` calls.
        runner = _make_runner(func)
        self._helpers[func] = runner
        return runner

    def resolve_call(self, func: Callable[..., object]) -> Callable[..., object]:
        """Map a callable to its autodiff-aware version.

        Swap an intercepted numpy/math function for its differentiable primitive;
        instrument a user helper on demand so the tape flows through its body; or
        wrap an un-ruled numpy/math function so it warns if a Var flows in. This is
        the logic ``before_call`` applies; it is exposed so other call mechanisms
        (e.g. pipescript's ``|>`` via its application hooks) can participate too.

        The swap target is the *active backend* (numpy by default): its ``intercept``
        table and un-mapped fallback. With the numpy backend this is exactly the old
        ``_INTERCEPT`` / ``_warn_wrapper`` behavior; a compile backend swaps the same
        calls for another framework's functions instead. Only the swap target varies --
        the user-helper instrumentation and the mathy predicate are backend-agnostic.
        """
        backend = current_backend()
        replacement = backend.intercept.get(func)
        if replacement is not None:
            return replacement
        if _is_user_function(func):
            return self._instrument_helper(func)
        if _is_mathy(func):
            return backend.on_unmapped(func)
        return func  # builtins, methods, etc. pass through

    @pyc.register_handler(pyc.before_call)
    def handle_before_call(
        self, func: Callable[..., object], node: object, *_: object, **__: object
    ) -> Callable[..., object]:
        return self.resolve_call(func)

    @pyc.register_handler(pyc.after_left_binop_arg)
    def handle_left_binop_arg(self, ret: object, *_: object, **__: object) -> object:
        # Let the active backend promote a binop operand (e.g. a numpy data global
        # meeting a torch/tf tensor). numpy/jax leave it unchanged.
        return current_backend().coerce_operand(ret)

    @pyc.register_handler(pyc.after_right_binop_arg)
    def handle_right_binop_arg(self, ret: object, *_: object, **__: object) -> object:
        return current_backend().coerce_operand(ret)


# ``tracer.instrumented`` recompiles ``f`` from source into a fresh function, so
# build each function's runner once and reuse it rather than recompiling per call.
_INSTRUMENTED: dict[Callable[..., object], Callable[..., object]] = {}


def _is_already_woven(f: Callable[..., object]) -> bool:
    """True if ``f``'s bytecode already emits before_call -- i.e. it was instrumented
    when its defining cell/module was compiled (notably a pipescript ``|>`` pipe
    lambda). Such a function must be run *directly* under the tracer (so its existing
    emits are handled), not re-instrumented from source -- its linecache source is the
    lowered placeholder form and would recompile to different semantics. Detected by
    the pyccolo emit builtin appearing among the names the code loads."""
    code = getattr(f, "__code__", None)
    if code is None:
        return False
    return any(name.endswith("_PYCCOLO_EVT_EMIT") for name in code.co_names)


def _make_runner(f: Callable[..., object]) -> Callable[..., object]:
    """A callable that runs ``f`` with autodiff interception active.

    A function that isn't yet woven (an ordinary ``def`` or a plain ``lambda``,
    including one defined in a notebook cell) is instrumented so its calls emit
    before_call. A function whose body uses pipescript syntax *is* woven, but
    ``instrumented`` re-instruments it from its retained augmented AST -- preserving
    the pipe/macro markings while still weaving before_call -- so it differentiates
    too. Only a function that's already woven yet has *no* retained augmented
    definition (e.g. a pipescript ``|>`` pipe lambda, or anything with no recompilable
    source) is run directly instead, with the autodiff tracer enabled so the tape is
    built as it executes; recompiling its lowered source would corrupt it.
    """
    tracer = AutodiffTracer.instance()
    if not (_is_already_woven(f) and tracer._augmented_definition_for(f) is None):
        try:
            return tracer.instrumented(f)
        except Exception:
            # No recompilable source (e.g. a closure over free vars, or a REPL/eval
            # function with no linecache entry): run it directly instead.
            pass

    @functools.wraps(f)
    def run_directly(*args: object, **kwargs: object) -> object:
        with tracer.tracing_enabled():
            return f(*args, **kwargs)

    return run_directly


def resolve_call(func: Callable[..., object]) -> Callable[..., object]:
    """Autodiff-aware version of ``func`` (the logic ``before_call`` applies).

    Exposed so other call mechanisms can opt into interception -- e.g. wiring a
    pipescript pipe hook so ``x |> np.exp`` differentiates when ``x`` is a Var::

        from pipescript.tracers.pipeline_tracer import PipelineTracer
        PipelineTracer.application_hooks.append(
            lambda f, v: resolve_call(f) if isinstance(v, Var) else f
        )
    """
    return AutodiffTracer.instance().resolve_call(func)

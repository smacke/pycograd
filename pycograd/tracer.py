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

import ast
import functools
import inspect
import sysconfig
from typing import Callable

import numpy as np
import pyccolo as pyc

from pycograd.backends import current_backend
from pycograd.ops import _is_mathy
from pycograd.tensor import Var
from pycograd.trace import (
    _BINOP_PRIM,
    _COMPARE_PRIM,
    _UNARYOP_PRIM,
    _SubscriptProxy,
    bind,
    num_transform_levels,
)

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
            # When a transform level (e.g. ``vmap``'s ``BatchTrace``) is live above the
            # base, route the intercepted call through the trace-level stack so the top
            # level processes it -- this is what makes ``vmap`` (and nested ``vmap``)
            # vectorize ``np.*`` calls. With only the base level active, ``bind`` lands
            # on ``EvalTrace`` and runs the primitive directly, identical to today. A
            # bare reverse pass (``grad``'s own ``ReverseTrace`` marker) is *not* such a
            # level, so a single top-level ``grad`` stays on the direct path.
            if num_transform_levels() > 0:
                return functools.partial(bind, replacement)
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

    # -- operator interception: route ops through the trace-level stack ---------
    # pyccolo hands each operator handler a default callable (``ret``) implementing the
    # original op and the op's AST node. For an op pycograd has a primitive for, we
    # return a callable that routes the operands through ``bind`` -- so a base-level
    # ``Var``/array runs the ``d_*`` primitive (identical to its dunder today), a
    # ``BatchTracer`` selects its ``vmap`` level, and an unmanaged value (the abstract
    # ``ShapedArray``) falls through to its own operator inside ``bind``. An op with no
    # primitive (e.g. ``%``, ``&``) keeps the original callable, so it is untouched.
    @pyc.register_handler(
        pyc.before_binop, when=lambda node: type(node.op) in _BINOP_PRIM
    )
    def handle_before_binop(
        self, ret: Callable[..., object], node: ast.BinOp, *_: object, **__: object
    ) -> Callable[..., object]:
        prim, _raw = _BINOP_PRIM[type(node.op)]
        return lambda x, y: bind(prim, x, y)

    @pyc.register_handler(
        pyc.before_unaryop, when=lambda node: type(node.op) in _UNARYOP_PRIM
    )
    def handle_before_unaryop(
        self, ret: Callable[..., object], node: ast.UnaryOp, *_: object, **__: object
    ) -> Callable[..., object]:
        prim, _raw = _UNARYOP_PRIM[type(node.op)]
        return lambda x: bind(prim, x)

    @pyc.register_handler(
        pyc.before_compare,
        # Only a *single* comparison (``a < b``) maps to a primitive; a chained compare
        # (``a < b < c``) has multiple ops and Python-level short-circuit semantics, so
        # leave it to pyccolo's default callable.
        when=lambda node: len(node.ops) == 1 and type(node.ops[0]) in _COMPARE_PRIM,
    )
    def handle_before_compare(
        self, ret: Callable[..., object], node: ast.Compare, *_: object, **__: object
    ) -> Callable[..., object]:
        prim, _raw = _COMPARE_PRIM[type(node.ops[0])]
        return lambda x, y: bind(prim, x, y)

    @pyc.register_handler(pyc.before_subscript_load)
    def handle_before_subscript_load(
        self, ret: object, node: object, *_: object, **__: object
    ) -> object:
        # pyccolo replaces the *subscripted object* (then performs ``[key]`` on it), so we
        # wrap the subscripted object in a proxy whose ``__getitem__`` *may* route through
        # ``bind``. We wrap a ``Var`` (the base level pycograd manages) and a raw
        # ``np.ndarray`` (so a shared/unbatched table gathered with a per-example batched
        # index -- ``table[batched_idx]`` -- reaches ``bind``). The proxy only binds when
        # the object is a ``Var`` or the key involves a ``Tracer``; otherwise it falls
        # through to plain ``obj[key]``, so ordinary array indexing by non-tracer keys --
        # and every dict/list/ParamDict subscript (never wrapped here) -- is unchanged.
        if isinstance(ret, Var) or isinstance(ret, np.ndarray):
            return _SubscriptProxy(ret)
        return ret


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
    # A function explicitly tagged to run directly (e.g. ``vmap``'s wrapper, a closure
    # over its config) manages its own tracing; instrumenting it from source would drop
    # its closure. Run it under the tracer so any ``np.*`` it makes is still intercepted.
    if getattr(f, "_pycograd_run_directly", False):

        @functools.wraps(f)
        def run_tagged(*args: object, **kwargs: object) -> object:
            with tracer.tracing_enabled():
                return f(*args, **kwargs)

        return run_tagged
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

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
from types import CodeType
from typing import cast

import numpy as np
import pyccolo as pyc

from pycograd._typing import Boxed, Prim
from pycograd.backends import Backend, current_backend
from pycograd.ops import _is_mathy
from pycograd.tensor import Var
from pycograd.trace import (
    _BINOP_PRIM,
    _COMPARE_PRIM,
    _UNARYOP_PRIM,
    Tracer,
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


def _is_user_function(func: Prim) -> bool:
    """True for a plain Python function defined in user (non-library) code.

    The autodiff primitives (``d_*`` etc.) live in :mod:`pycograd.ops` but are only
    ever reached via ``_INTERCEPT`` / operator dispatch, never as a call site inside
    instrumented user code, so they are never handed to this predicate. pycograd's
    own internals (and pyccolo) are excluded as a safety net; the compose-from-``np.*``
    op libraries (``pycograd.functional``, ``pycograd.examples``) are *not* -- they are
    user-level code whose bodies must be instrumented so their ``np.*`` calls route to
    the differentiable primitives (a plain ``np.max(var)`` would otherwise bypass the
    tape, since ``Var`` disables ``__array_ufunc__``).
    """
    if not inspect.isfunction(func):  # excludes C builtins, numpy ufuncs, methods
        return False
    module = getattr(func, "__module__", "") or ""
    if module.startswith("numpy") or module.startswith("pyccolo."):
        return False
    if module.startswith("pycograd.") and not module.startswith(
        ("pycograd.functional", "pycograd.examples")
    ):
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


def _unmapped_mathy(func: Prim, fallback: Prim) -> Prim:
    """Wrap an un-ruled numpy/math call. If a trace-stack :class:`Tracer`
    (a ``ShapedArray`` under ``eval_shape``, a ``BatchTracer`` under ``vmap``, a
    ``JVPTracer`` under ``jvp``) flows in, the active transform has no rule for ``func``,
    so raise a clear error instead of letting numpy fail obscurely on the abstract value.
    Otherwise defer to the backend's fallback (e.g. the numpy backend's warn-if-a-``Var``
    path, which still runs the call -- a non-differentiable op on a ``Var`` is allowed).
    """
    name = getattr(func, "__name__", repr(func))

    def _wrapped(*args: object, **kwargs: object) -> object:
        if any(isinstance(a, Tracer) for a in args) or any(
            isinstance(v, Tracer) for v in kwargs.values()
        ):
            raise NotImplementedError(
                f"pycograd has no rule for {name!r}, so the active transform "
                "(eval_shape / vmap / jvp) can't trace it. Rewrite using an op pycograd "
                "supports, or add a rule for it."
            )
        return fallback(*args, **kwargs)

    return _wrapped


def _is_deferred_operand(z: object) -> bool:
    """True for a *bare callable* used point-free in a binop -- e.g. ``np.tanh`` in a
    pipeline stage ``np.tanh ** 2`` (meaning ``x -> np.tanh(x) ** 2``, with ``x`` supplied
    later by the pipe). A ``Var``/``Tracer`` (or a plain number/array) is a *value*, not a
    deferred stage, so it takes the ordinary binop path."""
    return callable(z) and not isinstance(z, (Var, Tracer))


# pipescript lowers its *compose* operator ``.**`` to a ``**`` (Pow) token tagged with this
# augmentation. We duck-type on the token (rather than import pipescript) so a ``.**`` node
# composes functions while a plain ``**`` raises to a power. See ``handle_before_binop``.
_COMPOSE_TOKEN = ".**"


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
    _helpers: dict[Prim, Prim] = {}

    def _instrument_helper(self, func: Prim) -> Prim:
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

    def _augmented_definition_for(self, f: Prim) -> ast.stmt | None:
        """Like pyccolo's, but also recovers a *bare pipe lambda*'s augmented AST.

        pyccolo's base bails on ``co_name == "<lambda>"`` and only scans named ``def``s, so
        a top-level pipe lambda (``redundant = $ |> np.tanh ** 2 |> np.sum``) has no retained
        definition and falls back to *run-directly* -- leaving its binops (our point-free
        ``np.tanh ** 2`` stage) un-instrumented. But the lambda's augmented ``ast.Lambda``
        node *is* retained in the shared ``ast_node_by_id`` (pipescript wove it). Find it and
        lift it into a synthetic ``def`` -- named ``<lambda>`` so ``instrumented``'s by-name
        code-object extraction still matches -- mirroring pyccolo's own source-path lambda
        lifting, but from the *augmented* node so the pipe markings (and the ``**``) survive.
        A named def is handled by the base path unchanged.
        """
        node = super()._augmented_definition_for(f)
        if node is not None:
            return node
        code = getattr(f, "__code__", None)
        if (
            code is None
            or code.co_name != "<lambda>"
            or not any(name.endswith("_PYCCOLO_EVT_EMIT") for name in code.co_names)
        ):
            return None
        lam = self._retained_augmented_lambda(code)
        if lam is None:
            return None
        template = cast(ast.FunctionDef, ast.parse("def _l(): return None").body[0])
        template.name = "<lambda>"  # match ``instrumented``'s ``target_name`` (co_name)
        template.args = lam.args
        template.body = [ast.Return(value=lam.body)]
        ast.copy_location(template, lam)
        ast.fix_missing_locations(template)
        return template

    def _retained_augmented_lambda(self, code: CodeType) -> ast.Lambda | None:
        """The retained augmentation-annotated ``ast.Lambda`` for a woven lambda ``code``:
        an ``ast.Lambda`` whose subtree carries augmentations and whose arity matches,
        preferring an exact ``lineno`` match over a bare arity match.

        Scope the search to ``code``'s *source file* first (``ast_bookkeeper_by_fname``):
        each cell run gets a distinct ``co_filename``, so an edited-and-re-run pipeline (a
        new file) must not resolve to a *stale* lambda left in the shared global table by a
        prior run. Only if the per-file table has no match do we fall back to the shared
        table, scanning newest-first so the freshest definition still wins."""
        argcount, firstlineno = code.co_argcount, code.co_firstlineno

        def match(table: dict[int, ast.AST], newest_first: bool) -> ast.Lambda | None:
            nodes = reversed(list(table.values())) if newest_first else table.values()
            fallback: ast.Lambda | None = None
            for node in nodes:
                if (
                    isinstance(node, ast.Lambda)
                    and len(node.args.args) == argcount
                    and any(self.get_augmentations(id(n)) for n in ast.walk(node))
                ):
                    if getattr(node, "lineno", None) == firstlineno:
                        return node
                    fallback = fallback or node
            return fallback

        bk = self.ast_bookkeeper_by_fname.get(code.co_filename)
        if bk is not None:
            hit = match(bk.ast_node_by_id, newest_first=False)
            if hit is not None:
                return hit
        return match(self.ast_node_by_id, newest_first=True)

    def resolve_call(self, func: Prim) -> Prim:
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
            if backend.is_delegate:
                # On a compile backend an intercepted call (``np.exp(w)``) may receive a
                # bare ``Weight`` proxy directly as an operand -- pyccolo swapped the call
                # before the proxy's ``__array_ufunc__`` could resolve it. Binops coerce via
                # the binop-arg handlers; mirror that here so a unary ``np.*`` over a bare
                # weight resolves it to the live backend tensor too (no-op when no proxy is
                # present, so the dict-param compile path is unchanged).
                return _delegating_call(replacement, backend)
            return replacement
        if _is_user_function(func):
            return self._instrument_helper(func)
        if _is_mathy(func):
            return _unmapped_mathy(func, backend.on_unmapped(func))
        return func  # builtins, methods, etc. pass through

    @pyc.register_handler(pyc.before_call)
    def handle_before_call(
        self, func: Prim, node: ast.AST, *_: object, **__: object
    ) -> Prim:
        return self.resolve_call(func)

    def _operand_of_pipe(self, operand_node: object) -> bool:
        """True if ``operand_node`` is an operand of a pipescript ``|>`` pipe.

        pipescript lowers ``a |> f`` to a ``BitOr`` (``|``) ``BinOp`` tagged with its pipe
        augmentations, so a pipe operand is the *piped value* (``a``) or the *stage function*
        (``f``) -- neither is an arithmetic operand, so coercing it (below) is wrong: it would
        e.g. float an integer index seed and break ``table[idx]`` downstream. We duck-type on
        the parent ``BinOp`` carrying augmentations (mirroring pipescript's own
        ``node_is_pipeline_bitor_op``) rather than importing pipescript, exactly as the
        ``_COMPOSE_TOKEN`` check above does. A plain bitwise-or carries no augmentations, so
        it is still coerced."""
        parent = self.containing_ast_by_id.get(id(operand_node))
        return (
            isinstance(parent, ast.BinOp)
            and isinstance(parent.op, ast.BitOr)
            and bool(self.get_augmentations(id(parent)))
        )

    @pyc.register_handler(pyc.after_left_binop_arg)
    def handle_left_binop_arg(
        self, ret: Boxed, node: object, *_: object, **__: object
    ) -> Boxed:
        # Let the active backend promote a binop operand (e.g. a numpy data global
        # meeting a torch/tf tensor). numpy/jax leave it unchanged. A pipescript pipe
        # stage is not an arithmetic operand, so leave its piped value/function alone.
        if self._operand_of_pipe(node):
            return ret
        return current_backend().coerce_operand(ret)

    @pyc.register_handler(pyc.after_right_binop_arg)
    def handle_right_binop_arg(
        self, ret: Boxed, node: object, *_: object, **__: object
    ) -> Boxed:
        if self._operand_of_pipe(node):
            return ret
        return current_backend().coerce_operand(ret)

    # -- operator interception: route ops through the trace-level stack ---------
    # pyccolo hands each operator handler a default callable (``ret``) implementing the
    # original op and the op's AST node. For an op pycograd has a primitive for, we
    # return a callable that routes the operands through ``bind`` -- so a base-level
    # ``Var``/array runs the ``d_*`` primitive (identical to its dunder today), a
    # ``BatchTracer`` selects its ``vmap`` level, and an ``eval_shape`` ``ShapedArray``
    # selects its ``AbstractTrace`` level (its shape rule runs). An op with no primitive
    # (e.g. ``%``, ``&``) keeps the original callable, so it is untouched.
    @pyc.register_handler(
        pyc.before_binop, when=lambda node: type(node.op) in _BINOP_PRIM
    )
    def handle_before_binop(
        self, ret: Prim, node: ast.BinOp, *_: object, **__: object
    ) -> Prim:
        prim, _raw = _BINOP_PRIM[type(node.op)]
        # pipescript's compose op ``.**`` lowers to a Pow node carrying ``_COMPOSE_TOKEN``;
        # such a node *composes* functions instead of raising to a power.
        compose = isinstance(node.op, ast.Pow) and any(
            getattr(spec, "token", None) == _COMPOSE_TOKEN
            for spec in self.get_augmentations(id(node))
        )

        def op(x: Boxed | Prim, y: Boxed | Prim) -> Boxed | Prim:
            xd, yd = _is_deferred_operand(x), _is_deferred_operand(y)
            if compose and xd:
                # ``f .** g``: function composition (each call routed through
                # ``resolve_call`` so it differentiates). ``g`` a function -> ``f ∘ g``;
                # ``g`` an int -> the composition *power* ``f ∘ f ∘ … (g times)``.
                rf = self.resolve_call(cast(Prim, x))
                if yd:
                    rg = self.resolve_call(cast(Prim, y))
                    return lambda v: rf(rg(v))
                n = y if isinstance(y, int) else int(cast(int, y))

                def composed(v: Boxed) -> Boxed:
                    for _ in range(n):
                        v = rf(v)
                    return v

                return composed
            if not (xd or yd):
                # Ordinary value binop (a Var/array/number, no bare function): route
                # through ``bind`` exactly as before -- base ``Var``/array runs the ``d_*``
                # primitive, a ``BatchTracer``/``ShapedArray`` selects its transform level.
                return bind(prim, x, y)
            # Point-free: an operand is a bare function (``np.tanh ** 2``). Defer the op to a
            # one-argument *stage* ``v -> prim(x(v), y(v))`` -- each function operand is run
            # through ``resolve_call`` so its call differentiates (``np.tanh`` -> ``d_tanh``)
            # and rides any live transform level, exactly like an intercepted call would; a
            # non-function operand (the ``2``) is held constant.
            fx = self.resolve_call(cast(Prim, x)) if xd else None
            fy = self.resolve_call(cast(Prim, y)) if yd else None

            def stage(v: Boxed) -> Boxed:
                a = fx(v) if fx is not None else x
                b = fy(v) if fy is not None else y
                return bind(prim, a, b)

            return stage

        return op

    @pyc.register_handler(
        pyc.before_unaryop, when=lambda node: type(node.op) in _UNARYOP_PRIM
    )
    def handle_before_unaryop(
        self, ret: Prim, node: ast.UnaryOp, *_: object, **__: object
    ) -> Prim:
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
        self, ret: Prim, node: ast.Compare, *_: object, **__: object
    ) -> Prim:
        prim, _raw = _COMPARE_PRIM[type(node.ops[0])]
        return lambda x, y: bind(prim, x, y)

    @pyc.register_handler(pyc.before_subscript_load)
    def handle_before_subscript_load(
        self, ret: object, node: ast.AST, *_: object, **__: object
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


def _delegating_call(replacement: Prim, backend: Backend) -> Prim:
    """Wrap a delegate-backend replacement so it resolves bare ``Weight`` operands.

    Only rewrites the args when a proxy is actually present (directly or nested in a
    list/tuple, as in ``np.concatenate([w1, w2])``); otherwise it calls ``replacement``
    with the original args, so the dict-param compile path is byte-for-byte unchanged.
    The resolution mirrors :func:`pycograd.params._delegate_dispatch`."""

    def call(*args: object, **kwargs: object) -> object:
        from pycograd.params import Weight, _deep_unwrap

        def _has_weight(x: object) -> bool:
            return isinstance(x, Weight) or (
                isinstance(x, (list, tuple)) and any(_has_weight(e) for e in x)
            )

        if any(_has_weight(a) for a in args):
            args = tuple(backend.coerce_operand(_deep_unwrap(a)) for a in args)
        return replacement(*args, **kwargs)

    return call


# ``tracer.instrumented`` recompiles ``f`` from source into a fresh function, so
# build each function's runner once and reuse it rather than recompiling per call.
_INSTRUMENTED: dict[Prim, Prim] = {}


def _is_already_woven(f: Prim) -> bool:
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


def _make_runner(f: Prim) -> Prim:
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
    # A function explicitly tagged to run directly (e.g. ``vmap``'s wrapper) manages its
    # own tracing -- it pushes its own trace level -- so its body should stay out of the
    # interception path. Run it under the tracer so any ``np.*`` it makes is still
    # intercepted, but without instrumenting (recompiling) it.
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
            # No recompilable source (e.g. a REPL/eval function with no linecache entry,
            # or a dynamically built code object): run it directly instead.
            pass

    @functools.wraps(f)
    def run_directly(*args: object, **kwargs: object) -> object:
        with tracer.tracing_enabled():
            return f(*args, **kwargs)

    return run_directly


def resolve_call(func: Prim) -> Prim:
    """Autodiff-aware version of ``func`` (the logic ``before_call`` applies).

    Exposed so other call mechanisms can opt into interception -- e.g. wiring a
    pipescript pipe hook so ``x |> np.exp`` differentiates when ``x`` is a ``Var`` (the
    base reverse-mode level) or a ``Tracer`` (a ``vmap``/``jvp``/``eval_shape`` level)::

        from pipescript.tracers.pipeline_tracer import PipelineTracer
        PipelineTracer.application_hooks.append(
            lambda f, v: resolve_call(f) if isinstance(v, (Var, Tracer)) else f
        )
    """
    return AutodiffTracer.instance().resolve_call(func)

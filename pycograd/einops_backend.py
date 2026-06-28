# -*- coding: utf-8 -*-
"""einops <-> pycograd bridge.

`einops <https://einops.rocks>`_ dispatches ``rearrange`` / ``reduce`` / ``repeat`` /
``einsum`` through a *backend registry* (:mod:`einops._backends`) keyed by tensor type.
Out of the box it has no backend for our :class:`~pycograd.tensor.Var`, so
``einops.rearrange(var, ...)`` raises ``RuntimeError: Tensor type unknown to einops``.

This module registers a backend so einops operations run on ``Var`` (and on the
transform-level ``Tracer`` types, so they compose with ``grad`` / ``vmap`` / ``jvp``).
einops is an *optional* dependency: importing pycograd never imports einops. Instead
:func:`install` (called once from ``pycograd/__init__.py``) wires registration to happen
the moment both pycograd and einops are imported, in *either* order -- registering
immediately if einops is already loaded, otherwise on the first ``import einops``.

The backend dispatches each operation exactly the way the tracer's
:meth:`~pycograd.tracer.AutodiffTracer.resolve_call` does: swap the numpy callable for the
active backend's differentiable primitive, and -- when a transform level (``vmap`` /
``jvp``) is live -- route it through ``bind`` so that level processes it. With only a plain
reverse pass (or no transform) live, it calls the ``d_*`` primitive directly. This is what
makes the eager (``grad``) and the vectorized/forward (``vmap`` / ``jvp``) paths both work.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from pycograd._typing import Boxed, Prim
from pycograd.backends import current_backend
from pycograd.tensor import Var
from pycograd.trace import Tracer, bind, num_transform_levels

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

# einops names a backend by ``framework_name`` and only instantiates it when that name is
# in ``sys.modules`` -- "pycograd" always is here, so the backend is discoverable. We also
# insert the instance into einops' ``_loaded_backends`` directly (below) so ``get_backend``
# finds it without relying on the subclass scan.
_FRAMEWORK_NAME = "pycograd"

# einops reductions map to numpy callables; the active backend's intercept table swaps each
# for the matching ``d_*`` primitive (``np.mean`` -> ``d_mean`` etc.).
_REDUCE: dict[str, Prim] = {
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "mean": np.mean,
    "prod": np.prod,
    "any": np.any,
    "all": np.all,
}

_registered = False


def _dispatch(np_fn: Prim, *args: Any, **params: Any) -> Boxed:
    """Run ``np_fn`` through pycograd's dispatch, mirroring ``resolve_call``.

    Swap ``np_fn`` for the active backend's primitive, then route through ``bind`` when a
    transform level (``vmap`` / ``jvp``) is live so the top level vectorizes/differentiates
    it; otherwise call the primitive directly (the eager ``grad`` / no-transform path).
    """
    backend = current_backend()
    prim = backend.intercept.get(np_fn, np_fn)
    if num_transform_levels() > 0:
        return bind(prim, *args, **params)
    return prim(*args, **params)


def _build_backend_class() -> type:
    """Define the ``AbstractBackend`` subclass (deferred: needs einops imported)."""
    from einops._backends import AbstractBackend

    class PycogradEinopsBackend(AbstractBackend):  # type: ignore[misc]
        framework_name = _FRAMEWORK_NAME

        # Recognize the tape node *and* the transform-level tracers, so einops ops compose
        # under ``vmap`` / ``jvp`` (where the value is a BatchTracer / JVPTracer, not a Var).
        def is_appropriate_type(self, tensor: Any) -> bool:
            return isinstance(tensor, (Var, Tracer))

        def from_numpy(self, x: Any) -> Var:
            return Var(np.asarray(x))

        def to_numpy(self, x: Any) -> np.ndarray:
            return np.asarray(getattr(x, "value", x))

        def shape(self, x: Any) -> tuple[int, ...]:
            return tuple(x.shape)

        def reshape(self, x: Any, shape: Any) -> Boxed:
            return _dispatch(np.reshape, x, tuple(shape))

        def transpose(self, x: Any, axes: Any) -> Boxed:
            return _dispatch(np.transpose, x, tuple(axes))

        def reduce(self, x: Any, operation: str, axes: Any) -> Boxed:
            return _dispatch(_REDUCE[operation], x, axis=tuple(axes))

        def stack_on_zeroth_dimension(self, tensors: list) -> Boxed:
            return _dispatch(np.stack, list(tensors), axis=0)

        def add_axis(self, x: Any, new_position: int) -> Boxed:
            return _dispatch(np.expand_dims, x, new_position)

        def tile(self, x: Any, repeats: Any) -> Boxed:
            return _dispatch(np.tile, x, tuple(repeats))

        def concat(self, tensors: list, axis: int) -> Boxed:
            return _dispatch(np.concatenate, list(tensors), axis=axis)

        def is_float_type(self, x: Any) -> bool:
            return np.dtype(x.dtype).kind == "f"

        def einsum(self, pattern: str, *x: Any) -> Boxed:
            # einops pre-converts named patterns to numpy form ("a b, a c -> b c" ->
            # "ab,ac->bc"), so this is a plain ``np.einsum`` call.
            return _dispatch(np.einsum, pattern, *x)

    return PycogradEinopsBackend


def register_einops_backend() -> None:
    """Register the pycograd backend with einops. Idempotent; safe to call repeatedly.

    Requires einops to be importable. Raises ``ImportError`` if it is not -- call this
    only when you know einops is installed, or rely on :func:`install` (which never fails).
    """
    global _registered
    if _registered:
        return
    import einops._backends as eb

    backend = _build_backend_class()()
    # Insert directly so ``get_backend`` finds it via its ``_loaded_backends`` scan without
    # depending on the recursive ``__subclasses__`` discovery.
    eb._loaded_backends[_FRAMEWORK_NAME] = backend
    _registered = True


def install() -> None:
    """Arrange for registration once both pycograd and einops are imported (either order).

    Never raises and never imports einops itself: if einops is already loaded we register
    now; otherwise we install a one-shot ``builtins.__import__`` shim that registers on the
    first ``import einops`` and then restores the original import. (If einops is never
    imported the shim stays installed for the process -- a single name compare per import.)
    """
    if _registered:
        return
    if "einops" in sys.modules:
        register_einops_backend()
        return

    import builtins as _builtins

    _orig_import = _builtins.__import__

    def _import_hook(
        name: str,
        globals: "Mapping[str, object] | None" = None,
        locals: "Mapping[str, object] | None" = None,
        fromlist: "Sequence[str] | None" = (),
        level: int = 0,
    ) -> "ModuleType":
        module = _orig_import(name, globals, locals, fromlist, level)
        if name == "einops" or name.startswith("einops."):
            _builtins.__import__ = _orig_import  # one-shot: restore before registering
            try:
                register_einops_backend()
            except Exception:  # never let our hook break a user's ``import einops``
                pass
        return module

    _builtins.__import__ = _import_hook

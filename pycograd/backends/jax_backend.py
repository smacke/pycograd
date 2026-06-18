# -*- coding: utf-8 -*-
"""JAX backend: swap numpy/math calls for ``jax.numpy`` and differentiate with JAX.

``jax.numpy`` mirrors numpy almost name-for-name, so the intercept table is derived
directly from the numpy backend's coverage (:data:`pycograd.ops._INTERCEPT`) -- every
numpy/math callable pycograd knows how to differentiate is mapped to the same-named
``jnp`` function (with a tiny override for ``math.atan`` -> ``jnp.arctan``). Gradients
come from ``jax.value_and_grad``. x64 is enabled so results match pycograd's float64.

Constraint: JAX traces the Python forward, so *data-dependent* Python control flow
won't ``jit`` -- static nets (the common case) are fine.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from pycograd.backends import Backend
from pycograd.ops import _INTERCEPT

# math.* names that don't exist verbatim on jnp.
_NAME_OVERRIDE = {"atan": "arctan"}


def _build_intercept(jnp: "Any") -> dict:
    """Map every numpy/math callable pycograd differentiates to its ``jnp`` twin."""
    table: dict = {}
    for fn in _INTERCEPT:
        name = getattr(fn, "__name__", None)
        if name is None:
            continue
        repl = getattr(jnp, _NAME_OVERRIDE.get(name, name), None)
        if repl is not None:
            table[fn] = repl
    return table


def _unmapped(func: Callable[..., object], is_tensor: Callable[[object], bool]):
    """Wrap a mathy call with no jnp twin: raise clearly if a live tensor flows in."""
    name = getattr(func, "__name__", repr(func))

    def _wrapped(*args: object, **kwargs: object) -> object:
        if any(is_tensor(a) for a in args) or any(
            is_tensor(v) for v in kwargs.values()
        ):
            raise NotImplementedError(
                f"compile(jax): no JAX mapping for {name!r}; cannot differentiate "
                "this call. Rewrite the net using ops with a jax.numpy equivalent."
            )
        return func(*args, **kwargs)

    return _wrapped


class JaxBackend(Backend):
    name = "jax"

    def __init__(self) -> None:
        import jax
        import jax.numpy as jnp

        # Match pycograd's float64 tape so gradients agree to tight tolerance.
        jax.config.update("jax_enable_x64", True)
        self._jax = jax
        self._jnp = jnp
        self._intercept = _build_intercept(jnp)

    def _is_tensor(self, x: object) -> bool:
        return isinstance(x, (self._jax.Array, self._jax.core.Tracer))

    @property
    def intercept(self) -> Mapping[object, Callable[..., object]]:
        return self._intercept

    def on_unmapped(self, func: Callable[..., object]) -> Callable[..., object]:
        return _unmapped(func, self._is_tensor)

    def lift(self, array: object) -> object:
        return self._jnp.asarray(np.asarray(array, dtype=np.float64))

    def const(self, array: object) -> object:
        return self._jnp.asarray(np.asarray(array, dtype=np.float64))

    def to_numpy(self, tensor: object) -> object:
        return np.asarray(tensor)

    def grad_and_value(
        self, scalar_fn: Callable[[list], object], leaves: list
    ) -> tuple[object, list]:
        jax, jnp = self._jax, self._jnp
        arrs = [jnp.asarray(np.asarray(leaf, dtype=np.float64)) for leaf in leaves]

        def f(ts: list) -> object:
            return jnp.asarray(scalar_fn(ts)).reshape(())

        if not arrs:
            return np.asarray(f(arrs)), []
        value, grads = jax.value_and_grad(f)(arrs)
        return np.asarray(value), [np.asarray(g) for g in grads]

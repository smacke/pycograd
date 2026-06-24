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

from types import ModuleType
from typing import Callable, Mapping, Optional

import numpy as np

from pycograd._typing import Array, BackendArray, Prim
from pycograd.backends import Backend
from pycograd.dtypes import current_dtype
from pycograd.ops import _INTERCEPT, d_gated_act, d_logsumexp, d_sigmoid, d_softmax

# math.* names that don't exist verbatim on jnp.
_NAME_OVERRIDE = {"atan": "arctan"}


def _build_intercept(jnp: ModuleType) -> dict[Prim, Prim]:
    """Map every numpy/math callable pycograd differentiates to its ``jnp`` twin."""
    table: dict[Prim, Prim] = {}
    for fn in _INTERCEPT:
        name = getattr(fn, "__name__", None)
        if name is None:
            continue
        repl = getattr(jnp, _NAME_OVERRIDE.get(name, name), None)
        if repl is not None:
            table[fn] = repl
    return table


def _unmapped(func: Prim, is_tensor: Callable[[object], bool]) -> Prim:
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
    is_delegate = True

    def __init__(self) -> None:
        import jax
        import jax.nn
        import jax.numpy as jnp

        # Match pycograd's float64 tape so gradients agree to tight tolerance.
        jax.config.update("jax_enable_x64", True)
        self._jax = jax
        self._jnp = jnp
        self._intercept = _build_intercept(jnp)
        # ``d_sigmoid`` is tape-only (no numpy callable, so ``_build_intercept``
        # skips it); map the primitive directly to jax.nn.sigmoid so a direct call
        # lowers instead of running its xp-based body.
        self._intercept[d_sigmoid] = jax.nn.sigmoid
        # ``d_gated_act`` (tanh(f)*sigmoid(s)) is likewise tape-only; lower it natively.
        self._intercept[d_gated_act] = lambda f, s: jnp.tanh(f) * jax.nn.sigmoid(s)
        # Fused stable softmax / logsumexp (tape-only): lower to jax's native ops.
        import jax.scipy.special as _jsp

        self._intercept[d_softmax] = lambda x, axis=-1: jax.nn.softmax(x, axis=axis)
        self._intercept[d_logsumexp] = (
            lambda x, axis=None, keepdims=False: _jsp.logsumexp(
                x, axis=axis, keepdims=keepdims
            )
        )
        # Lower the composed im2col ``conv2d`` to XLA's native conv via
        # ``lax.conv_general_dilated`` (NCHW input / OIHW kernel, matching pycograd's
        # layout; ``rhs_dilation`` = kernel dilation, ``feature_group_count`` = groups),
        # so the compiled net runs an XLA convolution and jax autodiff supplies the
        # backward -- instead of tracing the gather + einsum. The numpy path keeps the
        # composed conv. ``conv1d`` / ``causal_conv1d`` route through ``conv2d``.
        from pycograd.functional import conv2d as _conv2d

        def _jax_conv2d(
            x: BackendArray,
            w: BackendArray,
            b: Optional[BackendArray] = None,
            stride: int = 1,
            pad: int = 0,
            dilation: int = 1,
            groups: int = 1,
        ) -> BackendArray:
            out = jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(stride, stride),
                padding=[(pad, pad), (pad, pad)],
                rhs_dilation=(dilation, dilation),
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
                feature_group_count=groups,
            )
            return out if b is None else out + jnp.reshape(b, (1, -1, 1, 1))

        self._intercept[_conv2d] = _jax_conv2d

    def _is_tensor(self, x: object) -> bool:
        return isinstance(x, (self._jax.Array, self._jax.core.Tracer))

    @property
    def intercept(self) -> Mapping[Prim, Prim]:
        return self._intercept

    def on_unmapped(self, func: Prim) -> Prim:
        return _unmapped(func, self._is_tensor)

    def lift(self, array: BackendArray) -> BackendArray:
        return self._jnp.asarray(np.asarray(array, dtype=current_dtype()))

    def const(self, array: BackendArray) -> BackendArray:
        return self._jnp.asarray(np.asarray(array, dtype=current_dtype()))

    def coerce_operand(self, value: BackendArray) -> BackendArray:
        # The tracer's ``after_{left,right}_binop_arg`` handlers run each ``@``/``+``/...
        # operand through here. A concrete jax array shares numpy's operators, but a
        # *trace-time* tracer (under jax.grad) cannot convert to numpy -- so a binop where a
        # numpy data global (e.g. an input baked into an ambient-weights net) meets a weight
        # bound to a tracer would raise. Promote that numpy operand to jnp so jax's own op
        # runs. Existing jnp arrays, tracers, and python scalars are left untouched. (torch
        # and tf override this too; the numpy/cupy tape needs no coercion, hence the identity
        # default on the base ``Backend``.)
        if isinstance(value, (np.ndarray, np.generic)):
            return self._jnp.asarray(np.asarray(value, dtype=current_dtype()))
        return value

    def to_numpy(self, tensor: BackendArray) -> Array:
        # Preserves dtype, including bfloat16 (jax's bf16 is the ml_dtypes numpy dtype).
        return np.asarray(tensor)

    def grad_and_value(
        self,
        scalar_fn: Callable[[list[BackendArray]], BackendArray],
        leaves: list[BackendArray],
    ) -> tuple[BackendArray, list[BackendArray]]:
        jax, jnp = self._jax, self._jnp
        arrs = [jnp.asarray(np.asarray(leaf, dtype=current_dtype())) for leaf in leaves]

        def f(ts: list) -> object:
            return jnp.asarray(scalar_fn(ts)).reshape(())

        if not arrs:
            return np.asarray(f(arrs)), []
        value, grads = jax.value_and_grad(f)(arrs)
        return np.asarray(value), [np.asarray(g) for g in grads]

    def compile_grad(
        self, scalar_fn: Callable[[list[BackendArray]], BackendArray]
    ) -> Callable[[list[BackendArray]], tuple[BackendArray, list[BackendArray]]]:
        # jit the value-and-grad once; XLA caches the compiled program keyed by the leaves'
        # shapes/dtypes, so reusing this closure across training steps traces the net a
        # single time. (scalar_fn closes over the weights' ambient binding, but jit re-runs
        # the Python only on the first trace -- later calls feed the leaf tensors straight
        # into the compiled graph.)
        jax, jnp = self._jax, self._jnp

        def f(ts: list) -> object:
            return jnp.asarray(scalar_fn(ts)).reshape(())

        compiled = jax.jit(jax.value_and_grad(f))

        def run(
            leaves: list[BackendArray],
        ) -> tuple[BackendArray, list[BackendArray]]:
            arrs = [jnp.asarray(np.asarray(x, dtype=current_dtype())) for x in leaves]
            if not arrs:
                return np.asarray(f(arrs)), []
            value, grads = compiled(arrs)
            return np.asarray(value), [np.asarray(g) for g in grads]

        return run

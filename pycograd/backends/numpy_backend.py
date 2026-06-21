# -*- coding: utf-8 -*-
"""The default backend: pycograd's own numpy tape.

This is the behavior that existed before backends were introduced, expressed behind
the :class:`Backend` protocol. Its ``intercept`` table and un-mapped fallback are the
very ``_INTERCEPT`` / ``_warn_wrapper`` the tracer used directly, so routing through
``current_backend()`` is byte-for-byte identical to the original path.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, cast

import numpy as np

from pycograd._typing import Array, BackendArray, Index, Operand, Prim
from pycograd.backends import Backend, activate
from pycograd.dtypes import current_dtype
from pycograd.ops import _INTERCEPT, _warn_wrapper
from pycograd.tensor import Var, _lift


class NumpyBackend(Backend):
    name = "numpy"
    array_module = np

    @property
    def intercept(self) -> Mapping[Prim, Prim]:
        return _INTERCEPT

    def on_unmapped(self, func: Prim) -> Prim:
        return _warn_wrapper(func)

    def scatter_add(self, out: BackendArray, key: Index, vals: BackendArray) -> None:
        # scatter-add handles repeated indices; ``out`` is a numpy array at runtime.
        np.add.at(cast(Any, out), cast(Any, key), cast(Any, vals))

    def lift(self, array: BackendArray) -> Var:
        return Var(np.asarray(array, dtype=current_dtype()))

    def const(self, array: BackendArray) -> BackendArray:
        # A raw numpy value: Var's operators auto-lift it when it meets a tape node,
        # so it participates in the forward without ever getting a gradient slot.
        return np.asarray(array, dtype=current_dtype())

    def to_numpy(self, tensor: BackendArray) -> Array:
        # Preserve the tensor's dtype (a float32/bfloat16 tape stays in its precision)
        # rather than upcasting back to float64.
        return tensor.value if isinstance(tensor, Var) else np.asarray(tensor)

    def grad_and_value(
        self,
        scalar_fn: Callable[[list[BackendArray]], BackendArray],
        leaves: list[BackendArray],
    ) -> tuple[BackendArray, list[BackendArray]]:
        # Activate self across the whole forward + backward so the tape's primitives
        # resolve their array module (``_xp()``) to this backend during *both* passes --
        # essential for cupy, where the compile path calls this outside its inner
        # ``activate``; harmless (already numpy) here.
        with activate(self):
            vars_ = [self.lift(leaf) for leaf in leaves]
            out = _lift(cast(Operand, scalar_fn(vars_)))
            out.backward()
        return out.value, [v.grad for v in vars_]

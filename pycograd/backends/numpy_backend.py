# -*- coding: utf-8 -*-
"""The default backend: pycograd's own numpy tape.

This is the behavior that existed before backends were introduced, expressed behind
the :class:`Backend` protocol. Its ``intercept`` table and un-mapped fallback are the
very ``_INTERCEPT`` / ``_warn_wrapper`` the tracer used directly, so routing through
``current_backend()`` is byte-for-byte identical to the original path.
"""
from __future__ import annotations

from typing import Callable, Mapping, cast

import numpy as np

from pycograd._typing import Operand
from pycograd.backends import Backend
from pycograd.ops import _INTERCEPT, _warn_wrapper
from pycograd.tensor import Var, _lift


class NumpyBackend(Backend):
    name = "numpy"

    @property
    def intercept(self) -> Mapping[object, Callable[..., object]]:
        return _INTERCEPT

    def on_unmapped(self, func: Callable[..., object]) -> Callable[..., object]:
        return _warn_wrapper(func)

    def lift(self, array: object) -> Var:
        return Var(np.asarray(array, dtype=float))

    def const(self, array: object) -> object:
        # A raw numpy value: Var's operators auto-lift it when it meets a tape node,
        # so it participates in the forward without ever getting a gradient slot.
        return np.asarray(array, dtype=float)

    def to_numpy(self, tensor: object) -> object:
        return (
            tensor.value if isinstance(tensor, Var) else np.asarray(tensor, dtype=float)
        )

    def grad_and_value(
        self, scalar_fn: Callable[[list], object], leaves: list
    ) -> tuple[object, list]:
        vars_ = [Var(np.asarray(leaf, dtype=float)) for leaf in leaves]
        out = _lift(cast(Operand, scalar_fn(vars_)))
        out.backward()
        return out.value, [v.grad for v in vars_]

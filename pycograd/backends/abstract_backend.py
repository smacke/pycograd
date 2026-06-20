# -*- coding: utf-8 -*-
"""The abstract backend: swap numpy/math calls for *shape rules*, not numbers.

**Legacy.** ``eval_shape`` no longer runs through this backend: abstract evaluation is
now a *trace level* (:class:`pycograd.shapes.AbstractTrace`), the same way ``vmap`` and
``jvp`` are, so ``eval_shape`` pushes that level and routes ops through
:func:`pycograd.trace.bind` rather than activating a backend swap. This backend is kept
only because the ``"abstract"`` / ``"shape"`` backend names remain registered (and a
test asserts constructing it imports no framework); it still works if activated directly
-- its ``intercept`` swaps ``np.exp``/``np.matmul``/... for the shape rules in
:mod:`pycograd.shapes` over :class:`~pycograd.shapes.ShapedArray` values -- but nothing
in the shape-inference path uses it anymore.

It imports only numpy (no jax/torch/tf): shape inference is framework-free.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from pycograd.backends import Backend
from pycograd.shapes import _ABSTRACT, ShapedArray


def _unmapped(func: Callable[..., object]) -> Callable[..., object]:
    """Wrap a mathy call with no shape rule: raise clearly if an aval flows in."""
    name = getattr(func, "__name__", repr(func))

    def _wrapped(*args: object, **kwargs: object) -> object:
        if any(isinstance(a, ShapedArray) for a in args) or any(
            isinstance(v, ShapedArray) for v in kwargs.values()
        ):
            raise NotImplementedError(
                f"eval_shape: no shape rule for {name!r}; cannot infer its output "
                "shape. Rewrite the net using ops pycograd has a rule for."
            )
        return func(*args, **kwargs)

    return _wrapped


class AbstractBackend(Backend):
    name = "abstract"

    @property
    def intercept(self) -> Mapping[object, Callable[..., object]]:
        return _ABSTRACT

    def on_unmapped(self, func: Callable[..., object]) -> Callable[..., object]:
        return _unmapped(func)

    def lift(self, array: object) -> ShapedArray:
        arr = np.asarray(array)
        return ShapedArray(arr.shape, arr.dtype)

    def const(self, array: object) -> ShapedArray:
        return self.lift(array)

    def to_numpy(self, tensor: object) -> object:
        raise NotImplementedError(
            "the abstract backend has no data to convert; read .shape/.dtype instead"
        )

    def grad_and_value(
        self, scalar_fn: Callable[[list], object], leaves: list
    ) -> tuple[object, list]:
        raise NotImplementedError("the abstract backend computes shapes, not gradients")

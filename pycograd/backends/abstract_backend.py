# -*- coding: utf-8 -*-
"""The abstract backend: swap numpy/math calls for *shape rules*, not numbers.

Where the numpy backend swaps ``np.exp``/``np.matmul``/... for differentiable
primitives and the compile backends swap them for another framework's ops, this
backend swaps them for the shape rules in :mod:`pycograd.shapes` operating on
:class:`~pycograd.shapes.ShapedArray` values. Running a net under it -- which is what
``eval_shape(..., method="abstract")`` does -- propagates ``(shape, dtype)`` with no
data, so it allocates nothing and raises on shapes that depend on data values.

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

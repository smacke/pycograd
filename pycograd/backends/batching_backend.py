# -*- coding: utf-8 -*-
"""The batching backend: swap numpy/math calls for *batching rules* (``vmap``).

Where the abstract backend swaps ops for shape rules on :class:`ShapedArray`, this one
swaps them for the batching rules in :mod:`pycograd.batching` operating on
:class:`~pycograd.batching.BatchedArray` values. A rule adjusts axis arguments to skip
the materialized batch axis and calls the underlying differentiable primitive on real
arrays, so an ordinary `Var` tape is built and ``backward()`` differentiates it.

It runs on numpy (``array_module = np``): a batched forward is just numpy work on
arrays that happen to carry a leading batch axis.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, cast

import numpy as np

from pycograd.backends import Backend
from pycograd.batching import _BATCH, BatchedArray
from pycograd.tensor import Var


def _unmapped(func: Callable[..., object]) -> Callable[..., object]:
    """Wrap a mathy call with no batching rule: raise clearly if a batched value flows
    in; otherwise run it (an unbatched call during the vmap trace is fine)."""
    name = getattr(func, "__name__", repr(func))

    def _wrapped(*args: object, **kwargs: object) -> object:
        if any(isinstance(a, BatchedArray) for a in args) or any(
            isinstance(v, BatchedArray) for v in kwargs.values()
        ):
            raise NotImplementedError(
                f"vmap: no batching rule for {name!r}; cannot vectorize it. "
                "Rewrite the net using ops pycograd has a rule for."
            )
        return func(*args, **kwargs)

    return _wrapped


class BatchingBackend(Backend):
    name = "batch"
    array_module = np

    @property
    def intercept(self) -> Mapping[object, Callable[..., object]]:
        return _BATCH

    def on_unmapped(self, func: Callable[..., object]) -> Callable[..., object]:
        return _unmapped(func)

    def scatter_add(self, out: object, key: object, vals: object) -> None:
        # The batched getitem prepends a full slice over the batch axis, so the key
        # scatters into the batched-shaped gradient just as on the numpy backend.
        np.add.at(cast(Any, out), cast(Any, key), cast(Any, vals))

    def lift(self, array: object) -> Var:
        return Var(np.asarray(array))

    def const(self, array: object) -> object:
        return np.asarray(array)

    def to_numpy(self, tensor: object) -> object:
        raise NotImplementedError(
            "the batching backend operates on batched tape values, not host arrays"
        )

    def grad_and_value(
        self, scalar_fn: Callable[[list], object], leaves: list
    ) -> tuple[object, list]:
        raise NotImplementedError(
            "the batching backend vectorizes; it does not differentiate"
        )

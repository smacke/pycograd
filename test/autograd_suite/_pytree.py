# -*- coding: utf-8 -*-
"""A tiny real-vector-space view over pytrees (arrays / python scalars / nested
list-tuple-dict), standing in for autograd's ``vspace`` so the ported ``check_grads``
can run unchanged. Real-only: ``covector`` is the identity (pycograd has no complex
support). Leaves are flattened with pycograd's own ``tree_flatten`` so the structure
matches what ``grad``/``value_and_grad`` produce.
"""
from __future__ import annotations

import numpy as np

from pycograd.tree import tree_flatten, tree_unflatten


def _is_num(leaf: object) -> bool:
    return isinstance(leaf, (int, float, complex, np.ndarray, np.generic))


def _as_arr(leaf: object) -> np.ndarray:
    return np.asarray(leaf, dtype=float)


class VSpace:
    """The flat real vector space of a pytree value: its numeric leaves' shapes."""

    def __init__(self, value: object) -> None:
        leaves, self.treedef = tree_flatten(value)
        self.leaves = leaves
        self.shapes = [(_as_arr(x).shape if _is_num(x) else None) for x in leaves]
        self.is_scalar = [
            (_is_num(x) and not isinstance(x, np.ndarray)) for x in leaves
        ]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VSpace) and self.shapes == other.shapes

    def __repr__(self) -> str:
        return f"VSpace(shapes={self.shapes})"

    @property
    def size(self) -> int:
        return int(sum(int(np.prod(s)) for s in self.shapes if s is not None))

    # --- constructors -----------------------------------------------------
    def _rebuild(self, arrays: list) -> object:
        out: list = []
        ai = iter(arrays)
        for leaf, shape, scal in zip(self.leaves, self.shapes, self.is_scalar):
            if shape is None:
                out.append(leaf)  # non-numeric leaf: passed through
            else:
                a = next(ai)
                out.append(float(a) if scal else np.asarray(a, dtype=float))
        return tree_unflatten(self.treedef, out)

    def zeros(self) -> object:
        return self._rebuild([np.zeros(s) for s in self.shapes if s is not None])

    def ones(self) -> object:
        return self._rebuild([np.ones(s) for s in self.shapes if s is not None])

    def randn(self, rng: np.random.Generator | None = None) -> object:
        r = rng if rng is not None else np.random.default_rng()
        return self._rebuild(
            [r.standard_normal(s) for s in self.shapes if s is not None]
        )

    # --- vector operations ------------------------------------------------
    def covector(self, x: object) -> object:
        return x  # real inner product: the covector is the vector itself

    def add(self, x: object, y: object) -> object:
        xs = [l for l in tree_flatten(x)[0] if _is_num(l)]
        ys = [l for l in tree_flatten(y)[0] if _is_num(l)]
        return self._rebuild([_as_arr(a) + _as_arr(b) for a, b in zip(xs, ys)])

    def scalar_mul(self, x: object, a: float) -> object:
        xs = [l for l in tree_flatten(x)[0] if _is_num(l)]
        return self._rebuild([_as_arr(l) * a for l in xs])

    def inner_prod(self, x: object, y: object) -> float:
        xs = [l for l in tree_flatten(x)[0] if _is_num(l)]
        ys = [l for l in tree_flatten(y)[0] if _is_num(l)]
        return float(sum(np.sum(_as_arr(a) * _as_arr(b)) for a, b in zip(xs, ys)))


def vspace(value: object) -> VSpace:
    return VSpace(value)

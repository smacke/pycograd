# -*- coding: utf-8 -*-
"""A tiny real-vector-space view over pytrees (arrays / python scalars / nested
list-tuple-dict), standing in for autograd's ``vspace`` so the ported ``check_grads``
can run unchanged. Complex-aware: a complex leaf is treated as a real 2N-vector, so the
``covector`` conjugates and the inner product is Hermitian (``Re(sum(conj(a)*b))``) --
exactly the convention pycograd's complex ``grad`` satisfies. Leaves are flattened with
pycograd's own ``tree_flatten`` so the structure matches what ``grad``/``value_and_grad``
produce.
"""
from __future__ import annotations

import numpy as np

from pycograd.tree import tree_flatten, tree_unflatten


def _is_num(leaf: object) -> bool:
    return isinstance(leaf, (int, float, complex, np.ndarray, np.generic))


def _as_arr(leaf: object) -> np.ndarray:
    # Preserve a complex leaf's dtype (so the Hermitian inner product / conj covector are
    # meaningful); a real leaf is carried as float.
    arr = np.asarray(leaf)
    return arr if arr.dtype.kind == "c" else np.asarray(leaf, dtype=float)


class VSpace:
    """The flat (real or complex) vector space of a pytree value: its numeric leaves'
    shapes and dtypes."""

    def __init__(self, value: object) -> None:
        leaves, self.treedef = tree_flatten(value)
        self.leaves = leaves
        self.shapes = [(_as_arr(x).shape if _is_num(x) else None) for x in leaves]
        self.dtypes = [(_as_arr(x).dtype if _is_num(x) else None) for x in leaves]
        self.is_scalar = [
            (_is_num(x) and not isinstance(x, np.ndarray)) for x in leaves
        ]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, VSpace)
            and self.shapes == other.shapes
            and self.dtypes == other.dtypes
        )

    def __repr__(self) -> str:
        return f"VSpace(shapes={self.shapes}, dtypes={self.dtypes})"

    @property
    def size(self) -> int:
        return int(sum(int(np.prod(s)) for s in self.shapes if s is not None))

    # --- constructors -----------------------------------------------------
    def _rebuild(self, arrays: list) -> object:
        out: list = []
        ai = iter(arrays)
        for leaf, shape, dt, scal in zip(
            self.leaves, self.shapes, self.dtypes, self.is_scalar
        ):
            if shape is None:
                out.append(leaf)  # non-numeric leaf: passed through
            else:
                a = np.asarray(next(ai), dtype=dt)
                out.append(
                    complex(a)
                    if (scal and dt.kind == "c")
                    else (float(a) if scal else a)
                )
        return tree_unflatten(self.treedef, out)

    def _draw(self, shape, dt, rng):
        if dt is not None and dt.kind == "c":
            r = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            return r.astype(dt)
        return rng.standard_normal(shape)

    def zeros(self) -> object:
        return self._rebuild(
            [np.zeros(s, d) for s, d in zip(self.shapes, self.dtypes) if s is not None]
        )

    def ones(self) -> object:
        return self._rebuild(
            [np.ones(s, d) for s, d in zip(self.shapes, self.dtypes) if s is not None]
        )

    def randn(self, rng: np.random.Generator | None = None) -> object:
        r = rng if rng is not None else np.random.default_rng()
        return self._rebuild(
            [
                self._draw(s, d, r)
                for s, d in zip(self.shapes, self.dtypes)
                if s is not None
            ]
        )

    # --- vector operations ------------------------------------------------
    def covector(self, x: object) -> object:
        # Hermitian inner product: the covector of a complex vector is its conjugate (a real
        # vector is its own covector).
        xs = [l for l in tree_flatten(x)[0] if _is_num(l)]
        return self._rebuild([np.conj(_as_arr(a)) for a in xs])

    def add(self, x: object, y: object) -> object:
        xs = [l for l in tree_flatten(x)[0] if _is_num(l)]
        ys = [l for l in tree_flatten(y)[0] if _is_num(l)]
        return self._rebuild([_as_arr(a) + _as_arr(b) for a, b in zip(xs, ys)])

    def scalar_mul(self, x: object, a: float) -> object:
        xs = [l for l in tree_flatten(x)[0] if _is_num(l)]
        return self._rebuild([_as_arr(l) * a for l in xs])

    def inner_prod(self, x: object, y: object) -> float:
        # Real inner product on the underlying 2N-real space: Re(sum(conj(x) * y)).
        xs = [l for l in tree_flatten(x)[0] if _is_num(l)]
        ys = [l for l in tree_flatten(y)[0] if _is_num(l)]
        return float(
            sum(
                np.real(np.sum(np.conj(_as_arr(a)) * _as_arr(b)))
                for a, b in zip(xs, ys)
            )
        )


def vspace(value: object) -> VSpace:
    return VSpace(value)

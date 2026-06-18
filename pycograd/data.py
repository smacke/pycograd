# -*- coding: utf-8 -*-
"""Minibatching: iterate over arrays in chunks for stochastic optimization.

``batches`` slices one or more equal-length arrays into minibatches for a single
epoch (optionally shuffled); :class:`DataLoader` wraps the same logic so each
iteration re-shuffles for a fresh epoch.

    opt = Adam(lr=1e-3)
    for Xb, yb in batches(X, y, batch_size=32, shuffle=True, rng=rng):
        loss, (g,) = value_and_grad(loss_fn)(params, Xb, yb)
        params = opt.step(params, g)

A single array yields bare minibatch arrays; several arrays (e.g. inputs and
labels) are sliced with one shared index, so corresponding rows stay aligned, and
each minibatch comes back as a tuple. Pass an ``rng`` (a ``np.random.Generator``)
for reproducible shuffling -- shuffling never touches the global RNG.
"""
from __future__ import annotations

from typing import Iterator, Optional, Union

import numpy as np

from pycograd._typing import Array
from pycograd.tensor import _xp

Batch = Union[Array, tuple[Array, ...]]


def _check(arrays: tuple[Array, ...], batch_size: int) -> int:
    if not arrays:
        raise ValueError("batches: needs at least one array")
    if batch_size <= 0:
        raise ValueError(f"batches: batch_size must be positive, got {batch_size}")
    n = len(arrays[0])
    for a in arrays[1:]:
        if len(a) != n:
            raise ValueError(
                "batches: all arrays must share the leading dimension "
                f"(got {[len(a) for a in arrays]})"
            )
    return n


def batches(
    *arrays: Array,
    batch_size: int,
    shuffle: bool = False,
    rng: Optional[np.random.Generator] = None,
    drop_last: bool = False,
) -> Iterator[Batch]:
    """Yield minibatches over one epoch of ``arrays`` (sliced on the first axis).

    With ``shuffle`` the rows are permuted once for this epoch (via ``rng`` if
    given, else a fresh default generator). ``drop_last`` discards a final partial
    batch. Each item is a single array when one array was passed, else a tuple of
    aligned minibatch arrays.
    """
    n = _check(arrays, batch_size)
    if shuffle:
        gen = rng if rng is not None else np.random.default_rng()
        order = gen.permutation(n)
    else:
        order = np.arange(n)
    # The permutation/index math is host-side (numpy); the gather uses the active array
    # module so on-device arrays (e.g. cupy under ``device("cupy")``) are sliced without
    # a host round-trip, while plain numpy arrays behave exactly as before.
    xp = _xp()
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        if drop_last and len(idx) < batch_size:
            break
        # ``xp.asarray(idx)`` keeps the (host) index on the same device as the data so
        # cupy's fancy indexing applies; for numpy it is a no-op and behavior is unchanged.
        dev_idx = xp.asarray(idx)
        slices = tuple(xp.asarray(a)[dev_idx] for a in arrays)
        yield slices[0] if len(slices) == 1 else slices


class DataLoader:
    """Reusable epoch iterator over a fixed set of arrays.

    Iterating a ``DataLoader`` runs one shuffled epoch via :func:`batches`; iterate
    it again (e.g. once per training epoch) for a freshly shuffled pass. ``len`` is
    the number of minibatches per epoch.
    """

    def __init__(
        self,
        *arrays: Array,
        batch_size: int,
        shuffle: bool = False,
        rng: Optional[np.random.Generator] = None,
        drop_last: bool = False,
    ) -> None:
        self._n = _check(arrays, batch_size)
        self.arrays = arrays
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Batch]:
        return batches(
            *self.arrays,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rng=self.rng,
            drop_last=self.drop_last,
        )

    def __len__(self) -> int:
        if self.drop_last:
            return self._n // self.batch_size
        return (self._n + self.batch_size - 1) // self.batch_size

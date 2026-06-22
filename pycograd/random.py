# -*- coding: utf-8 -*-
"""Splittable, pure PRNG keys -- randomness without a hidden global generator.

A ``Key`` is a small immutable ``uint32`` array (not a stateful
``np.random.Generator``). Every draw and every derived key is a pure function of
its input key, so randomness is *threaded explicitly* and reproducible:

    k = key(0)
    k1, k2 = split(k)            # two independent child keys
    mask = bernoulli(k1, 0.9, x.shape)

This mirrors JAX's ``jax.random``: the same key always yields the same draw, and
to get *different* randomness you ``split``/``fold_in`` to derive fresh keys
rather than mutating one generator. That is what makes per-sample randomness
under ``vmap`` correct -- ``split(k, B)`` gives one key per batch row, mapped with
``in_axes`` -- where a shared generator would hand every row the same draw.

Built on numpy's counter-based, splittable ``SeedSequence`` + ``Philox`` (both are
deterministic and carry no process-global state). Draws are produced host-side as
plain numpy arrays; a per-backend (cupy / torch / jax / tf) RNG seam is a separate
roadmap item, so on a delegate/device backend a key-sampled value is a constant.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from pycograd._typing import Array, Key

# A reshape/sample shape: a single length or a tuple of dims (runtime-referenced
# only in annotations, but kept as ``Union`` for parity with the codebase style).
Shape = Union[int, Tuple[int, ...]]

# A key is this many ``uint32`` words of entropy -- enough headroom that derived
# keys do not collide in practice.
_KEY_WORDS = 4
# Distinct salts so ``split`` and ``fold_in`` derive *different* children from the
# same parent key (otherwise ``split(k)[1]`` and ``fold_in(k, 1)`` would coincide).
_SPLIT_SALT = 0x5F1D
_FOLD_SALT = 0xF01D


def key(seed: int) -> Key:
    """Create a root PRNG key from an integer ``seed``."""
    return np.random.SeedSequence(int(seed)).generate_state(_KEY_WORDS, dtype=np.uint32)


def _derive(k: Key, salt: int, idx: int) -> Key:
    """Deterministically derive a child key from ``(k, salt, idx)`` by re-hashing
    the parent's entropy -- pure, so the same inputs always give the same child."""
    entropy = [int(w) for w in np.asarray(k).ravel()] + [int(salt), int(idx)]
    return np.random.SeedSequence(entropy).generate_state(_KEY_WORDS, dtype=np.uint32)


def split(k: Key, num: int = 2) -> Key:
    """Split ``k`` into ``num`` independent child keys, returned stacked as
    ``(num, _KEY_WORDS)``. Pure: the same key always splits the same way."""
    return np.stack([_derive(k, _SPLIT_SALT, i) for i in range(num)])


def fold_in(k: Key, data: int) -> Key:
    """Fold an integer ``data`` (a step index, a layer id, ...) into ``k``, giving a
    new key -- the idiom for per-step / per-layer streams off one root key."""
    return _derive(k, _FOLD_SALT, data)


def _generator(k: Key) -> "np.random.Generator":
    """A fresh ``Generator`` seeded *only* by ``k`` -- no shared/global state, so
    the draw is a pure function of the key."""
    entropy = [int(w) for w in np.asarray(k).ravel()]
    return np.random.Generator(np.random.Philox(np.random.SeedSequence(entropy)))


def bernoulli(k: Key, p: float, shape: Shape) -> Array:
    """Sample a 0/1 float array (1 with probability ``p``) of the given ``shape``."""
    return (_generator(k).random(shape) < p).astype(float)


def uniform(k: Key, shape: Shape, low: float = 0.0, high: float = 1.0) -> Array:
    """Sample a uniform ``[low, high)`` float array of the given ``shape``."""
    return _generator(k).uniform(low, high, shape)


def normal(k: Key, shape: Shape) -> Array:
    """Sample a standard-normal float array of the given ``shape``."""
    return _generator(k).standard_normal(shape)


def randint(k: Key, low: int, high: int, shape: Shape) -> Array:
    """Sample integers in ``[low, high)`` of the given ``shape``."""
    return _generator(k).integers(low, high, shape)

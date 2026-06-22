# -*- coding: utf-8 -*-
"""Tests for ``pycograd.random``: splittable PRNG keys and the key-threaded
``dropout``. Keys are pure -- the same key always yields the same draw and the
same children -- so the tests assert determinism, independence of split children,
and that ``dropout`` is reproducible from a key with no hidden global generator."""
import pytest

np = pytest.importorskip("numpy")

from pycograd import dropout  # noqa: E402
from pycograd import value_and_grad  # noqa: E402
from pycograd import random as R  # noqa: E402


# --- keys -------------------------------------------------------------------
def test_key_is_uint32_and_deterministic():
    k = R.key(0)
    assert k.dtype == np.uint32
    assert np.array_equal(R.key(0), k)  # same seed -> same key
    assert not np.array_equal(R.key(1), k)  # different seed -> different key


def test_split_is_deterministic_and_children_independent():
    k = R.key(7)
    a = R.split(k, 3)
    b = R.split(k, 3)
    assert a.shape == (3, R._KEY_WORDS)
    assert np.array_equal(a, b)  # pure: same key splits the same way
    # The three children are mutually distinct.
    assert not np.array_equal(a[0], a[1])
    assert not np.array_equal(a[1], a[2])


def test_fold_in_differs_from_split_and_is_deterministic():
    k = R.key(7)
    assert np.array_equal(R.fold_in(k, 5), R.fold_in(k, 5))
    assert not np.array_equal(R.fold_in(k, 5), R.fold_in(k, 6))
    # Distinct salts: fold_in(k, i) must not collide with split(k)[i].
    assert not np.array_equal(R.fold_in(k, 1), R.split(k, 2)[1])


# --- samplers ---------------------------------------------------------------
def test_samplers_shapes_determinism_and_stats():
    k = R.key(3)
    assert R.normal(k, (3, 2)).shape == (3, 2)
    assert R.uniform(k, (4,), 1.0, 2.0).shape == (4,)
    assert R.randint(k, 0, 5, (10,)).shape == (10,)
    assert np.array_equal(R.normal(k, (5,)), R.normal(k, (5,)))  # pure
    # bernoulli(p) is 1 with probability ~p.
    draws = R.bernoulli(k, 0.7, (20000,))
    assert set(np.unique(draws)).issubset({0.0, 1.0})
    assert abs(draws.mean() - 0.7) < 0.02
    u = R.uniform(k, (20000,), 1.0, 2.0)
    assert u.min() >= 1.0 and u.max() < 2.0


def test_distinct_keys_give_distinct_draws():
    k1, k2 = R.split(R.key(0))
    assert not np.allclose(R.normal(k1, (100,)), R.normal(k2, (100,)))


# --- key-threaded dropout ---------------------------------------------------
def test_dropout_key_is_reproducible():
    k1, k2 = R.split(R.key(0))
    x = np.ones((4, 5))
    assert np.array_equal(
        np.asarray(dropout(x, 0.5, True, key=k1)),
        np.asarray(dropout(x, 0.5, True, key=k1)),  # same key -> same mask
    )
    assert not np.array_equal(
        np.asarray(dropout(x, 0.5, True, key=k1)),
        np.asarray(dropout(x, 0.5, True, key=k2)),  # different key -> different mask
    )


def test_dropout_full_batch_masks_each_element_independently():
    # One key already gives per-sample dropout over a (B, d) batch -- no vmap needed.
    k = R.split(R.key(1))[0]
    X = np.ones((200, 50))
    m = np.asarray(dropout(X, 0.5, True, key=k))
    distinct_rows = {tuple((row != 0).tolist()) for row in m}
    assert len(distinct_rows) > 100  # rows are not all the same mask
    assert abs(m.mean() - 1.0) < 0.05  # inverted dropout preserves the mean


def test_dropout_grad_routes_through_key_mask():
    k = R.split(R.key(2))[0]
    x = np.array([1.0, 2.0, 3.0, 4.0])
    _, (g,) = value_and_grad(lambda a: dropout(a, 0.5, True, key=k))(x)
    mask = R.bernoulli(k, 0.5, x.shape) / 0.5
    assert np.allclose(g, mask)


def test_dropout_requires_explicit_rng_when_training():
    x = np.ones((3,))
    assert np.allclose(np.asarray(dropout(x, 0.5, training=False)), x)  # eval: no rng
    assert np.allclose(np.asarray(dropout(x, 0.0, training=True)), x)  # p=0: no rng
    with pytest.raises(ValueError, match="no global generator"):
        dropout(x, 0.5, training=True)  # training with neither key nor rng

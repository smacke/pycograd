# -*- coding: utf-8 -*-
"""The symbolic-dimension algebra: canonicalization, simplification, and the
shape-rule helpers (broadcast / product / contract gate / slice)."""
import numpy as np
import pytest

from pycograd._dims import (
    Dim,
    broadcast_dim,
    broadcast_shapes,
    naming_scope,
    prod_dims,
    provably_unequal,
    slice_dim,
    symbol,
)


def test_symbol_is_a_dim():
    n = symbol("a")
    assert isinstance(n, Dim)


def test_constants_collapse_to_int():
    n = symbol("a")
    assert n - n == 0 and isinstance(n - n, int)
    assert n * 0 == 0 and isinstance(n * 0, int)
    assert isinstance(n + 1 - 1, Dim)  # still symbolic
    assert (n + 1 - 1) == n


def test_like_terms_combine():
    n = symbol("a")
    assert n + n == 2 * n
    assert 3 * n - n == 2 * n
    assert n + n + n == 3 * n


def test_distinct_symbols_stay_distinct():
    a, b = symbol("a"), symbol("b")
    assert a != b
    assert a + b != 2 * a
    assert hash(a) != hash(b) or a != b  # different by value


def test_products_of_symbols():
    a, b = symbol("a"), symbol("b")
    assert a * b == b * a
    assert (a * b) * a == a * a * b


def test_floordiv_exact_simplifies():
    n = symbol("a")
    assert (2 * n) // 2 == n
    assert (6 * n) // 3 == 2 * n
    assert (n * 4) // 2 == 2 * n


def test_floordiv_inexact_is_opaque_but_stable():
    n = symbol("a")
    a = (n + 1) // 2
    b = (n + 1) // 2
    assert isinstance(a, Dim)
    assert a == b  # structurally identical -> equal
    assert (n + 1) // 2 != (n + 2) // 2


def test_floordiv_of_equal_dims():
    n = symbol("a")
    assert n // n == 1
    assert n // 1 == n


def test_mod_divisible_is_zero():
    n = symbol("a")
    assert (2 * n) % 2 == 0


def test_dim_not_equal_to_int():
    n = symbol("a")
    assert (n == 5) is False
    assert (n != 5) is True


def test_prod_dims_concrete_is_int():
    assert prod_dims((2, 3, 4)) == 24 and isinstance(prod_dims((2, 3, 4)), int)
    assert prod_dims(()) == 1


def test_prod_dims_symbolic():
    n = symbol("a")
    assert prod_dims((2, n, 3)) == 6 * n


def test_provably_unequal():
    n = symbol("a")
    assert provably_unequal(3, 4) is True
    assert provably_unequal(3, 3) is False
    assert provably_unequal(n, 4) is False  # symbolic -> never proven unequal
    assert provably_unequal(n, n) is False


def test_broadcast_dim_rules():
    n = symbol("a")
    assert broadcast_dim(1, 5) == 5
    assert broadcast_dim(5, 1) == 5
    assert broadcast_dim(1, n) == n  # concrete 1 yields the other
    assert broadcast_dim(n, 1) == n
    assert broadcast_dim(7, n) == 7  # concrete >1 pins the symbol
    assert broadcast_dim(n, n) == n  # same symbol
    assert broadcast_dim(3, 3) == 3


def test_broadcast_dim_concrete_mismatch_raises():
    with pytest.raises(ValueError):
        broadcast_dim(3, 4)


def test_broadcast_two_distinct_symbols_interns():
    a, b = symbol("a"), symbol("b")
    r1 = broadcast_dim(a, b)
    r2 = broadcast_dim(b, a)  # commutative -> same symbol
    assert isinstance(r1, Dim) and r1 == r2


def test_broadcast_shapes_concrete_matches_numpy():
    got = broadcast_shapes((3, 1, 5), (4, 5))
    assert got == tuple(np.broadcast_shapes((3, 1, 5), (4, 5)))
    assert all(isinstance(d, int) for d in got)


def test_broadcast_shapes_symbolic():
    n = symbol("a")
    assert broadcast_shapes((n,), (1,)) == (n,)
    assert broadcast_shapes((4, n), (n,)) == (4, n)
    assert broadcast_shapes((), (n,)) == (n,)


def test_slice_dim_concrete():
    assert slice_dim(10, slice(1, 3)) == 2
    assert slice_dim(8, slice(None, None, 2)) == 4


def test_slice_dim_symbolic():
    n = symbol("a")
    assert slice_dim(n, slice(None, None, None)) == n
    assert slice_dim(n, slice(2, None, None)) == n - 2
    assert slice_dim(n, slice(None, None, 2)) == (n + 1) // 2


def test_naming_is_scoped_and_deterministic():
    with naming_scope():
        a = symbol(("x", 0))
        b = symbol(("y", 1))
        assert str(a) == "n0" and str(b) == "n1"
        assert str(a) == "n0"  # stable on re-render
    with naming_scope():
        c = symbol(("z", 2))
        assert str(c) == "n0"  # restarts each scope


def test_render_expressions():
    with naming_scope():
        n = symbol("a")  # -> n0
        assert str(2 * n) == "2*n0"
        assert str(n + 1) == "n0 + 1"


def test_named_symbol_renders_by_name_and_unifies():
    b1 = symbol("B", name="B")
    b2 = symbol("B", name="B")
    assert str(b1) == "B"
    assert b1 == b2  # same key -> same symbol regardless of scope
    assert str(2 * b1 + 1) == "2*B + 1"


def test_subs_and_symbol_keys():
    b = symbol("B", name="B")
    k = symbol("K", name="K")
    assert (2 * b + 1).subs({"B": 4}) == 9
    assert (b * k).subs({"K": 3}) == 3 * b
    assert (b * k).symbol_keys() == {"B", "K"}
    # unmapped symbols stay symbolic
    assert isinstance((b + k).subs({"B": 2}), Dim)


def test_as_symbol():
    b = symbol("B", name="B")
    assert b.as_symbol() == ("B", "B")
    assert (2 * b).as_symbol() is None
    assert (b + 1).as_symbol() is None

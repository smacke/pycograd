# -*- coding: utf-8 -*-
"""The dimension-equality store (union-find) behind shape polymorphism."""
from pycograd._constraints import ConstraintEnv
from pycograd._dims import symbol


def _sym(name):
    return symbol(name, name=name)


def test_concrete_equality():
    env = ConstraintEnv()
    assert env.assert_eq(4, 4) is True
    assert env.assert_eq(4, 5) is False


def test_bind_symbol_to_concrete():
    env = ConstraintEnv()
    assert env.assert_eq(_sym("K"), 768) is True
    # rebinding the same class to a different concrete is a contradiction
    assert env.assert_eq(_sym("K"), 512) is False
    assert env.mapping()["K"] == 768


def test_union_two_symbols_then_pin():
    env = ConstraintEnv()
    assert env.assert_eq(_sym("A"), _sym("B")) is True
    assert env.assert_eq(_sym("B"), 32) is True
    m = env.mapping()
    # both names resolve to the pinned concrete
    assert m["A"] == 32 and m["B"] == 32


def test_union_conflicting_concretes():
    env = ConstraintEnv()
    env.assert_eq(_sym("A"), 8)
    env.assert_eq(_sym("B"), 16)
    assert env.assert_eq(_sym("A"), _sym("B")) is False


def test_data_dependent_symbol_is_not_bound():
    # A tuple-keyed (data-dependent) symbol is opaque: equating it with a concrete is
    # accepted but never pins it (its value is a runtime fact).
    env = ConstraintEnv()
    mask = symbol(("nonzero", 1))  # tuple key -> not solvable
    assert env.assert_eq(mask, 768) is True
    assert mask.as_symbol()[0] not in env.mapping()


def test_unmerged_symbol_absent_from_mapping():
    env = ConstraintEnv()
    env.assert_eq(_sym("A"), _sym("B"))  # A,B merge; representative stays symbolic
    m = env.mapping()
    # the non-representative aliases to the representative; neither is pinned
    assert "A" in m or "B" in m  # at least the aliased one maps to the rep

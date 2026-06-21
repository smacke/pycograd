# -*- coding: utf-8 -*-
"""Dimension-equality constraints for shape polymorphism.

When shape inference runs over *symbolic* input dims (e.g. a batch ``B`` declared via
``ShapeDtypeStruct(("B", 768))``), each contraction registers an equality: a matmul
asserts its inner dims equal, concatenate asserts its non-axis dims equal, broadcasting
asserts compatible dims equal. :class:`ConstraintEnv` is the union-find that records
those equalities, refines a symbol pinned to a concrete (``K`` forced to ``4``), and
reports a contradiction (two concretes forced equal) as a shape error.

Only *solvable* symbols -- caller-declared input dims, whose key is a ``str`` -- get
bound to concretes or merged. *Data-dependent* symbols (a mask count, a broadcast;
their key is a tuple) are runtime facts, not statically known, so they are left opaque,
preserving the optimistic "carry it forward" behavior of plain symbolic inference.
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Hashable, Iterator, cast

from pycograd import _dims
from pycograd._dims import Dim


def _as_atom(x: int | Dim) -> tuple:
    """Classify a dim as ``("int", v)``, ``("sym", key, name)``, or ``("expr",)``."""
    if isinstance(x, Dim):
        s = x.as_symbol()
        return ("sym", s[0], s[1]) if s is not None else ("expr",)
    if _dims._is_int(x):
        return ("int", int(cast(Any, x)))
    return ("expr",)


class ConstraintEnv:
    """Union-find over symbol keys with at most one concrete value per class."""

    def __init__(self) -> None:
        self.parent: dict[Hashable, Hashable] = {}  # key -> parent key
        self.value: dict[Hashable, int] = {}  # root key -> concrete int
        self.name: dict[Hashable, str] = {}  # key -> rendered name

    def _add(self, key: Hashable, name: str) -> None:
        if key not in self.parent:
            self.parent[key] = key
            self.name[key] = name

    def _find(self, key: Hashable) -> Hashable:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:  # path compression
            self.parent[key], key = root, self.parent[key]
        return root

    @staticmethod
    def _solvable(key: Hashable) -> bool:
        # Caller-declared input dims have string keys; data-dependent symbols (nonzero,
        # bcast, slice) have tuple keys and are never bound/merged.
        return isinstance(key, str)

    def assert_eq(self, a: int | Dim, b: int | Dim) -> bool:
        """Record ``a == b``; return ``False`` if that is a provable contradiction."""
        ta, tb = _as_atom(a), _as_atom(b)
        if ta[0] == "int" and tb[0] == "int":
            return ta[1] == tb[1]
        if ta[0] == "int" and tb[0] == "sym":
            return self._bind(tb, ta[1])
        if ta[0] == "sym" and tb[0] == "int":
            return self._bind(ta, tb[1])
        if ta[0] == "sym" and tb[0] == "sym":
            return self._union(ta, tb)
        return True  # an expression is involved -- can't reason, stay optimistic

    def _bind(self, sym: tuple, val: int) -> bool:
        _, key, name = sym
        if not self._solvable(key):
            return True  # data-dependent: a runtime fact, never statically pinned
        self._add(key, name)
        root = self._find(key)
        cur = self.value.get(root)
        if cur is not None and cur != val:
            return False
        self.value[root] = val
        return True

    def _union(self, s1: tuple, s2: tuple) -> bool:
        _, k1, n1 = s1
        _, k2, n2 = s2
        if not (self._solvable(k1) and self._solvable(k2)):
            return True  # leave data-dependent symbols opaque
        self._add(k1, n1)
        self._add(k2, n2)
        r1, r2 = self._find(k1), self._find(k2)
        if r1 == r2:
            return True
        v1, v2 = self.value.get(r1), self.value.get(r2)
        if v1 is not None and v2 is not None and v1 != v2:
            return False
        self.parent[r2] = r1
        if v1 is None and v2 is not None:
            self.value[r1] = v2
        return True

    def mapping(self) -> dict[Hashable, int | Dim]:
        """A substitution mapping each known symbol key to its concrete value (if its
        class is pinned) or to its class representative symbol (if merged)."""
        m: dict[Hashable, int | Dim] = {}
        for key in self.parent:
            root = self._find(key)
            v = self.value.get(root)
            if v is not None:
                m[key] = v
            elif root != key:
                m[key] = _dims.symbol(root, name=self.name[root])
        return m


# ---------------------------------------------------------------------------
# Active environment (entered for the duration of an abstract inference run).
# ---------------------------------------------------------------------------
_env: "contextvars.ContextVar[ConstraintEnv | None]" = contextvars.ContextVar(
    "dim_env", default=None
)


@contextlib.contextmanager
def constraint_scope() -> Iterator[ConstraintEnv]:
    env = ConstraintEnv()
    token = _env.set(env)
    try:
        yield env
    finally:
        _env.reset(token)


def active_env() -> "ConstraintEnv | None":
    return _env.get()


def register_eq(a: int | Dim, b: int | Dim) -> bool:
    """Register ``a == b`` with the active env (if any); ``False`` on a provable
    contradiction. With no active env, falls back to the concrete-only check."""
    env = _env.get()
    if env is None:
        return not _dims.provably_unequal(a, b)
    return env.assert_eq(a, b)

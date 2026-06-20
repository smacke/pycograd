# -*- coding: utf-8 -*-
"""Symbolic dimensions for shape inference.

A shape dimension is ``int | Dim``. Concrete dims stay plain ints -- :class:`Dim`
appears only for *data-dependent* sizes (e.g. the length of ``x[x > 0]``, which
depends on values, not shapes). A ``Dim`` is an immutable, canonicalized polynomial
over atomic unknowns (:class:`Symbol`) and opaque :class:`FloorDiv` atoms, so equal
expressions compare equal (``n0 + n0 == 2 * n0``) and arithmetic simplifies
(``(2 * n0) // 2 == n0``). That canonical form is what lets a shape rule carry an
unknown dimension through matmul/reshape/broadcast and recognize when two of them
are the same.

The helpers mirror the numeric operations the shape rules need but tolerate symbols:
:func:`prod_dims` (for ``.size``), :func:`broadcast_shapes` / :func:`broadcast_dim`,
:func:`provably_unequal` (gate a contract check so only a *provable* concrete
mismatch raises), and :func:`slice_dim`. Each fast-paths to plain ints when no
symbol is present, so concrete inference is byte-for-byte unchanged.
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Iterator, cast

import numpy as np

# A *monomial* is a tuple of ``(atom, power)`` pairs sorted by the atom's sort key
# (power > 0); the empty tuple ``()`` is the constant monomial. A ``Dim``'s value is a
# mapping ``monomial -> int coefficient`` (all nonzero) -- i.e. the polynomial
# ``sum(coeff * prod(atom ** power))``. Constants live in the ``()`` monomial.

Monomial = tuple  # tuple[tuple["_Atom", int], ...]


def _is_int(x: object) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


# ---------------------------------------------------------------------------
# Atoms: the irreducible factors a monomial is built from.
# ---------------------------------------------------------------------------
class _Atom:
    """An irreducible factor: a :class:`Symbol` or an opaque :class:`FloorDiv`."""

    __slots__ = ()

    @property
    def sort_key(self) -> tuple:
        return ()


class Symbol(_Atom):
    """An atomic unknown dimension, identified by a hashable ``key``.

    Two symbols are equal iff their keys are equal. The rendered ``name`` (``n0``,
    ``n1``, ...) is cosmetic, assigned *at creation* from the active
    :func:`naming_scope` (so reprs are deterministic and survive the scope exit);
    equal keys created within one scope share a name.
    """

    __slots__ = ("key", "name")

    def __init__(self, key: object) -> None:
        self.key = key
        self.name = _name_of(key)

    @property
    def sort_key(self) -> tuple:
        return (0, repr(self.key))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Symbol) and other.key == self.key

    def __hash__(self) -> int:
        return hash(("Symbol", self.key))

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__


class FloorDiv(_Atom):
    """An opaque ``num // den`` that did not simplify (carried as a single atom)."""

    __slots__ = ("num", "den")

    def __init__(self, num: object, den: object) -> None:
        self.num = num
        self.den = den

    @property
    def sort_key(self) -> tuple:
        return (1, repr(self.num), repr(self.den))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FloorDiv)
            and other.num == self.num
            and other.den == self.den
        )

    def __hash__(self) -> int:
        return hash(("FloorDiv", self.num, self.den))

    def __str__(self) -> str:
        n = f"({self.num})" if isinstance(self.num, Dim) else str(self.num)
        d = f"({self.den})" if isinstance(self.den, Dim) else str(self.den)
        return f"{n}//{d}"

    __repr__ = __str__


# ---------------------------------------------------------------------------
# Polynomial helpers (operate on the ``monomial -> coeff`` dicts).
# ---------------------------------------------------------------------------
def _as_poly(x: object) -> dict | None:
    if isinstance(x, Dim):
        return x._poly
    if _is_int(x):
        xi = int(cast(Any, x))
        return {} if xi == 0 else {(): xi}
    return None


def _make(poly: dict) -> int | Dim:
    """Canonicalize a poly dict, collapsing a pure constant back to a plain int."""
    poly = {m: c for m, c in poly.items() if c != 0}
    if not poly:
        return 0
    if tuple(poly) == ((),):
        return poly[()]
    return Dim(poly)


def _add_poly(p: dict, q: dict) -> dict:
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, 0) + c
    return out


def _mul_mono(m1: Monomial, m2: Monomial) -> Monomial:
    powers: dict = {}
    for atom, p in m1 + m2:
        powers[atom] = powers.get(atom, 0) + p
    return tuple(sorted(powers.items(), key=lambda ap: ap[0].sort_key))


def _mul_poly(p: dict, q: dict) -> dict:
    out: dict = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = _mul_mono(m1, m2)
            out[m] = out.get(m, 0) + c1 * c2
    return out


# ---------------------------------------------------------------------------
# Dim: a canonicalized polynomial over atoms.
# ---------------------------------------------------------------------------
class Dim:
    """A symbolic, unknown dimension -- a canonical polynomial over :class:`Symbol`s.

    Construct one with :func:`symbol`; combine with ``+``, ``-``, ``*``, ``//``. A
    result that reduces to a constant comes back as a plain ``int`` (so concrete
    arithmetic never leaks a ``Dim``). Equality and hashing are structural over the
    canonical form, so ``n0 + n0`` and ``2 * n0`` are the same object-by-value.
    """

    __slots__ = ("_poly", "_hash")

    def __init__(self, poly: dict) -> None:
        # poly is assumed already canonical (no zero coeffs, never pure-constant).
        self._poly = poly
        self._hash = hash(frozenset(poly.items()))

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, o: object) -> int | Dim:
        q = _as_poly(o)
        return NotImplemented if q is None else _make(_add_poly(self._poly, q))

    __radd__ = __add__

    def __neg__(self) -> int | Dim:
        return _make({m: -c for m, c in self._poly.items()})

    def __sub__(self, o: object) -> int | Dim:
        q = _as_poly(o)
        if q is None:
            return NotImplemented
        return _make(_add_poly(self._poly, {m: -c for m, c in q.items()}))

    def __rsub__(self, o: object) -> int | Dim:
        q = _as_poly(o)
        if q is None:
            return NotImplemented
        return _make(_add_poly({m: -c for m, c in self._poly.items()}, q))

    def __mul__(self, o: object) -> int | Dim:
        q = _as_poly(o)
        return NotImplemented if q is None else _make(_mul_poly(self._poly, q))

    __rmul__ = __mul__

    def __floordiv__(self, o: object) -> int | Dim:
        if _is_int(o):
            d = int(cast(Any, o))
            if d == 0:
                raise ZeroDivisionError("floor division of a Dim by zero")
            if all(c % d == 0 for c in self._poly.values()):
                return _make({m: c // d for m, c in self._poly.items()})
            return _opaque_floordiv(self, d)
        if isinstance(o, Dim):
            if self == o:
                return 1
            return _opaque_floordiv(self, o)
        return NotImplemented

    def __rfloordiv__(self, o: object) -> int | Dim:
        if _is_int(o):
            oi = int(cast(Any, o))
            return 0 if oi == 0 else _opaque_floordiv(oi, self)
        return NotImplemented

    def __mod__(self, o: object) -> int | Dim:
        # a % b == a - (a // b) * b; divisible cases collapse to 0.
        q = self // o
        return self - q * cast(Any, o)

    # -- equality / hashing (structural over the canonical form) -------------
    def __eq__(self, o: object) -> bool:
        if isinstance(o, Dim):
            return self._poly == o._poly
        if _is_int(o):
            return False  # a real Dim is never a pure constant
        return NotImplemented

    def __ne__(self, o: object) -> bool:
        eq = self.__eq__(o)
        return eq if eq is NotImplemented else not eq

    def __hash__(self) -> int:
        return self._hash

    # -- rendering -----------------------------------------------------------
    def __str__(self) -> str:
        terms = [
            _fmt_term(m, c) for m, c in sorted(self._poly.items(), key=_term_sort_key)
        ]
        return " + ".join(terms) if terms else "0"

    __repr__ = __str__


def _opaque_floordiv(num: object, den: object) -> int | Dim:
    return _make({((FloorDiv(num, den), 1),): 1})


def _term_sort_key(item: tuple) -> tuple:
    mono, _ = item
    # higher-degree terms first, constant (degree 0) last, then by atom keys.
    degree = sum(p for _, p in mono)
    return (-degree, tuple(a.sort_key for a, _ in mono))


def _fmt_term(mono: Monomial, c: int) -> str:
    if not mono:
        return str(c)
    factors = [str(a) if p == 1 else f"{a}**{p}" for a, p in mono]
    body = "*".join(factors)
    if c == 1:
        return body
    if c == -1:
        return f"-{body}"
    return f"{c}*{body}"


def symbol(key: object) -> Dim:
    """A fresh symbolic dimension identified by the hashable ``key``."""
    return Dim({((Symbol(key), 1),): 1})


# ---------------------------------------------------------------------------
# Lazy, scoped naming so reprs are deterministic per inference run.
# ---------------------------------------------------------------------------
_naming: contextvars.ContextVar[tuple[dict, list] | None] = contextvars.ContextVar(
    "dim_naming", default=None
)
_GLOBAL_NAMES: dict = {}
_GLOBAL_CTR: list = [0]


@contextlib.contextmanager
def naming_scope() -> Iterator[None]:
    """Within this scope, symbol names restart at ``n0`` (so reprs are stable)."""
    token = _naming.set(({}, [0]))
    try:
        yield
    finally:
        _naming.reset(token)


def _name_of(key: object) -> str:
    """Assign (or reuse) a name for ``key`` from the active scope's counter."""
    reg = _naming.get()
    names, ctr = reg if reg is not None else (_GLOBAL_NAMES, _GLOBAL_CTR)
    if key not in names:
        names[key] = f"n{ctr[0]}"
        ctr[0] += 1
    return names[key]


# ---------------------------------------------------------------------------
# Numeric helpers the shape rules call (symbol-aware, int fast paths).
# ---------------------------------------------------------------------------
def has_symbol(shape: tuple) -> bool:
    return any(isinstance(d, Dim) for d in shape)


def prod_dims(shape: tuple) -> int | Dim:
    """The product of a shape's dims -- a plain int unless any dim is symbolic."""
    shape = tuple(shape)
    if not has_symbol(shape):
        return int(np.prod(shape, dtype=np.int64)) if shape else 1
    acc: int | Dim = 1
    for d in shape:
        acc = acc * d
    return acc


def provably_unequal(a: object, b: object) -> bool:
    """True only when both dims are concrete ints and differ -- the safe gate for a
    contract check: a symbolic dim is never *proven* unequal, so it never raises."""
    if isinstance(a, Dim) or isinstance(b, Dim):
        return False
    return _is_int(a) and _is_int(b) and a != b


def broadcast_dim(a: object, b: object) -> int | Dim:
    """Broadcast two dims. Concrete ``1`` yields the other; a concrete ``>1`` pins a
    symbolic partner; two distinct symbolics yield an interned ``bcast`` symbol."""
    if isinstance(a, Dim) or isinstance(b, Dim):
        if not isinstance(a, Dim) and cast(int, a) == 1:
            return cast("int | Dim", b)
        if not isinstance(b, Dim) and cast(int, b) == 1:
            return cast("int | Dim", a)
        if isinstance(a, Dim) and isinstance(b, Dim):
            if a == b:
                return a
            return symbol(("bcast", frozenset((a, b))))
        return cast("int | Dim", a if not isinstance(a, Dim) else b)  # concrete pins it
    ai, bi = cast(int, a), cast(int, b)
    if ai == bi or bi == 1:
        return ai
    if ai == 1:
        return bi
    raise ValueError(f"cannot broadcast dims {a} and {b}")


def broadcast_shapes(*shapes: tuple) -> tuple:
    """Right-aligned broadcast tolerant of symbolic dims. Falls back to numpy (and
    its exact error) when every shape is concrete."""
    shapes = tuple(tuple(s) for s in shapes)
    if not any(has_symbol(s) for s in shapes):
        return tuple(int(d) for d in np.broadcast_shapes(*shapes))
    ndim = max((len(s) for s in shapes), default=0)
    out: list = [1] * ndim
    for s in shapes:
        offset = ndim - len(s)
        for i, d in enumerate(s):
            out[offset + i] = broadcast_dim(out[offset + i], d)
    return tuple(out)


def slice_dim(dim: object, s: slice) -> int | Dim:
    """Length of ``range``-applied ``s`` over a dimension. Concrete when ``dim`` is an
    int; otherwise resolved symbolically for the common slices, else a fresh symbol."""
    if not isinstance(dim, Dim):
        return len(range(*s.indices(cast(int, dim))))
    start, stop, step = s.start, s.stop, s.step
    step = 1 if step is None else step
    if (
        step == 1
        and stop is None
        and (start is None or (_is_int(start) and start >= 0))
    ):
        a = int(start) if start else 0
        return dim - a if a else dim
    if start is None and stop is None and _is_int(step) and step > 0:
        return (dim + (int(step) - 1)) // int(step)  # ceil(dim / step)
    return symbol(("slice", dim, (start, stop, step)))

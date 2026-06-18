# -*- coding: utf-8 -*-
"""The reverse-mode tape node.

``Var`` wraps a numpy array; arithmetic operators are overloaded so that running a
program builds a computation graph, and ``Var.backward()`` walks it in reverse to
accumulate gradients. The convenience methods (``sum``/``mean``/``__pow__``/``T``/
...) delegate to the differentiable primitives in :mod:`pycograd.ops`; those
imports are deferred to function bodies so this module has no import-time
dependency on ``ops`` (which imports ``Var`` from here).
"""
from __future__ import annotations

import functools
from typing import Any, Callable

import numpy as np

from pycograd._typing import Array, ArrayLike, Index, Operand
from pycograd.backends import current_backend


# ---------------------------------------------------------------------------
# The array-module seam: data is computed with the *active* backend's array
# library (numpy by default, cupy under ``device("cupy")``), so the same tape and
# the same VJP rules run on whatever device that backend lives on.
# ---------------------------------------------------------------------------
def _xp() -> Any:
    """The active backend's array module -- ``numpy`` unless a device is activated.

    Typed ``Any`` because the module is chosen at runtime (numpy / cupy / ...); the
    surface pycograd uses is the shared array-API subset both provide.
    """
    return current_backend().array_module


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _unbroadcast(grad: Array, shape: tuple[int, ...]) -> Array:
    """Sum ``grad`` over axes that were broadcast, so it matches ``shape``."""
    grad = _xp().asarray(grad, dtype=float)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def _unwrap_weight(x: Operand) -> Var | ArrayLike:
    """Resolve a ``Weight`` proxy to its current value; leave anything else."""
    # Deferred import: params imports Var/_lift/_unbroadcast from this module.
    from pycograd.params import Weight

    return x._live() if isinstance(x, Weight) else x


def _value(x: Operand) -> ArrayLike:
    x = _unwrap_weight(x)
    return x.value if isinstance(x, Var) else x


def _is_array(x: object) -> bool:
    """True for a numpy array or the active backend's array type (e.g. a cupy array)."""
    if isinstance(x, np.ndarray):
        return True
    ndarray = getattr(_xp(), "ndarray", None)
    return ndarray is not None and isinstance(x, ndarray)


def _is_numeric(x: object) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)) or _is_array(x)


# ---------------------------------------------------------------------------
# The reverse-mode tape node.
# ---------------------------------------------------------------------------
class Var:
    # Tell numpy to defer operators (``ndarray + Var``, ``ndarray @ Var``) to our
    # reflected methods, and to fail loudly on un-intercepted ufuncs.
    __array_ufunc__ = None

    def __init__(self, value: ArrayLike, _parents: tuple[Var, ...] = ()) -> None:
        xp = _xp()
        self.value: Array = xp.asarray(value, dtype=float)
        self.grad: Array = xp.zeros_like(self.value)
        self._parents = _parents
        self._backward: Callable[[], None] = lambda: None

    # -- construction helpers ------------------------------------------------
    def _unary(self, value: ArrayLike, grad_fn: Callable[[Array, Array], Array]) -> Var:
        out = Var(value, _parents=(self,))

        def _backward() -> None:
            self.grad = self.grad + _unbroadcast(
                grad_fn(self.value, out.grad), self.value.shape
            )

        out._backward = _backward
        return out

    def _binary(
        self,
        other: Operand,
        value_fn: Callable[[Array, Array], Array],
        grad_fn: Callable[[Array, Array, Array], tuple[Array, Array]],
    ) -> Var:
        other = _lift(other)
        out = Var(value_fn(self.value, other.value), _parents=(self, other))

        def _backward() -> None:
            ga, gb = grad_fn(self.value, other.value, out.grad)
            self.grad = self.grad + _unbroadcast(ga, self.value.shape)
            other.grad = other.grad + _unbroadcast(gb, other.value.shape)

        out._backward = _backward
        return out

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, o: Operand) -> Var:
        return self._binary(o, lambda a, b: a + b, lambda a, b, g: (g, g))

    __radd__ = __add__  # addition is commutative

    def __mul__(self, o: Operand) -> Var:
        return self._binary(o, lambda a, b: a * b, lambda a, b, g: (g * b, g * a))

    __rmul__ = __mul__  # multiplication is commutative

    def __sub__(self, o: Operand) -> Var:
        return self._binary(o, lambda a, b: a - b, lambda a, b, g: (g, -g))

    def __rsub__(self, o: Operand) -> Var:
        return self._binary(o, lambda a, b: b - a, lambda a, b, g: (-g, g))

    def __truediv__(self, o: Operand) -> Var:
        return self._binary(
            o, lambda a, b: a / b, lambda a, b, g: (g / b, -g * a / (b * b))
        )

    def __rtruediv__(self, o: Operand) -> Var:
        return self._binary(
            o, lambda a, b: b / a, lambda a, b, g: (-g * b / (a * a), g / a)
        )

    def __neg__(self) -> Var:
        return self._unary(-self.value, lambda a, g: -g)

    def __pow__(self, p: Operand) -> Var:
        from pycograd import ops

        if isinstance(p, Var):
            # general x ** y == exp(y * log x)
            return ops.d_exp(p * ops.d_log(self))
        return self._unary(self.value**p, lambda a, g: g * p * a ** (p - 1))

    def __abs__(self) -> Var:
        from pycograd import ops

        return ops.d_abs(self)

    def __matmul__(self, o: Operand) -> Var:
        from pycograd import ops

        return ops._matmul(self, o)

    def __rmatmul__(self, o: Operand) -> Var:
        from pycograd import ops

        return ops._matmul(o, self)

    # -- comparisons return plain (non-differentiable) booleans --------------
    def __lt__(self, o: Operand) -> Array:
        return self.value < _value(o)

    def __le__(self, o: Operand) -> Array:
        return self.value <= _value(o)

    def __gt__(self, o: Operand) -> Array:
        return self.value > _value(o)

    def __ge__(self, o: Operand) -> Array:
        return self.value >= _value(o)

    def __eq__(self, o: Operand) -> Array:  # type: ignore[override]
        return self.value == _value(o)

    def __ne__(self, o: Operand) -> Array:  # type: ignore[override]
        return self.value != _value(o)

    __hash__ = None  # type: ignore[assignment]

    # -- indexing (gather forward, scatter-add backward) ---------------------
    def __getitem__(self, key: Index) -> Var:
        out = Var(self.value[key], _parents=(self,))

        def _backward() -> None:
            grad = _xp().zeros_like(self.value)
            # scatter-add (np.add.at / cupyx.scatter_add) handles repeated indices
            current_backend().scatter_add(grad, key, out.grad)
            self.grad = self.grad + grad

        out._backward = _backward
        return out

    # -- numpy-style methods -------------------------------------------------
    def __getattr__(self, name: str) -> object:
        # Provide numpy's array-method surface (``x.sum(...)``, ``x.mean(...)``,
        # ``x.reshape(...)``, ``x.clip(...)``, ...) generically rather than enumerating
        # each: route a numpy function name we have a VJP rule for to its
        # differentiable primitive. Only reached for names not resolved normally, so
        # ``value`` / ``grad`` / ``T`` / dunders are unaffected; an unknown name (or
        # one with no rule) raises ``AttributeError``.
        if name.startswith("__"):
            raise AttributeError(name)
        from pycograd import ops

        np_fn = getattr(np, name, None)
        prim = ops._INTERCEPT.get(np_fn) if callable(np_fn) else None
        if prim is None:
            raise AttributeError(name)
        return functools.partial(prim, self)

    @property
    def T(self) -> Var:
        from pycograd import ops

        return ops.d_transpose(self)

    # -- non-differentiable metadata (read-only views of the underlying array) -
    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def ndim(self) -> int:
        return self.value.ndim

    @property
    def size(self) -> int:
        return self.value.size

    def __array__(self, *args: object, **kwargs: object) -> Array:
        # No concrete array view: a numpy call reaching here was NOT intercepted,
        # so fail loudly instead of silently building an object array. Defining it
        # also makes Var statically array-like, so numpy-typed code checks.
        raise TypeError(
            "Var has no array conversion; this numpy call should have been "
            "intercepted -- is the calling function instrumented?"
        )

    def __repr__(self) -> str:
        return f"Var({self.value!r})"

    # -- reverse pass --------------------------------------------------------
    def backward(self) -> None:
        topo: list[Var] = []
        visited = set()

        def build(v: Var) -> None:
            if id(v) in visited:
                return
            visited.add(id(v))
            for parent in v._parents:
                build(parent)
            topo.append(v)

        build(self)
        xp = _xp()
        for v in topo:
            v.grad = xp.zeros_like(v.value)
        self.grad = xp.ones_like(self.value)
        for v in reversed(topo):
            v._backward()


def _lift(x: Operand) -> Var:
    x = _unwrap_weight(x)
    return x if isinstance(x, Var) else Var(x)


def detach(x: Operand) -> Var:
    """A fresh leaf with the same value but no gradient history (stop-gradient)."""
    x = _unwrap_weight(x)
    return Var(x.value if isinstance(x, Var) else x)

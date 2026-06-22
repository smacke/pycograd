# -*- coding: utf-8 -*-
"""Shared type aliases.

``Var`` is the reverse-mode tape node; an ``Operand`` is anything the ops and
primitives accept -- a ``Var`` or a plain number/array that is lifted to one, or a
``Weight`` proxy (a late-bound reference to an ambient parameter).

The aliases below are ordinary runtime objects (they are referenced in
``cast(...)`` calls), so they keep ``Union``/``Optional`` rather than PEP 604
``|`` -- which is not valid at runtime on Python 3.9. Annotations elsewhere use
``X | Y`` freely, since ``from __future__ import annotations`` makes those strings.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Hashable, Optional, Union

import numpy as np
from numpy.typing import DTypeLike

if TYPE_CHECKING:
    from pycograd.params import Weight
    from pycograd.tensor import Var
    from pycograd.trace import Tracer

Scalar = Union[int, float]
Array = np.ndarray
ArrayLike = Union[Scalar, Array]
# A splittable PRNG key (see :mod:`pycograd.random`): a small immutable ``uint32``
# array that deterministically seeds a draw or derives child keys, in place of a
# stateful ``np.random.Generator``.
Key = np.ndarray
# An operand is a Var, a plain number/array, or a ``Weight`` proxy (a late-bound
# reference to an ambient parameter; see ``with`` support near ``ParamDict``).
Operand = Union["Var", ArrayLike, "Weight"]
# A differentiable tensor value: a ``Var`` during tracing, or a plain ndarray
# when the same helper is run outside the tape (e.g. at eval time).
Tensor = Union["Var", np.ndarray]
Axis = Optional[Union[int, tuple[int, ...]]]

# A *primitive*: a differentiable ``d_*`` op or operator primitive that the
# trace-level stack dispatches through ``bind`` (see :mod:`pycograd.trace`). Each
# primitive has its own heterogeneous call signature, so only the callable-ness is
# spelled out here -- the unavoidable dynamism, localized to one named alias instead
# of a bare ``Callable`` / ``Callable[..., object]`` repeated everywhere.
Prim = Callable[..., Any]

# A value flowing through the trace-level interpreter stack and its per-primitive
# rules (``bind`` / ``pure`` / ``lift`` / ``process_primitive`` / the vmap & jvp
# rules): a level ``Tracer`` (BatchTracer / JVPTracer / ShapedArray), a tape ``Var``,
# or a raw scalar/array lifted into one. ``None`` rides the same path as a structural
# operand (an absent ``clip`` bound, an omitted axis). Broader than ``Operand`` -- it
# adds ``Tracer`` -- and the precise replacement for the ``object`` annotations the
# dispatch core used to carry.
Boxed = Union["Tracer", "Var", ArrayLike, None]

# A per-primitive *rule* (vmap batch rule / jvp rule / abstract shape rule): called
# with a leading trace plus the primitive's own operands, returning a level value.
Rule = Callable[..., Boxed]

# A raw operand at the ``bind`` dispatch boundary: a level value, an index key
# (slice / int / ellipsis / array / a tuple thereof), or a *sequence* of operands for
# the join primitives (concatenate / stack). Deliberately broad -- the dispatch core
# only *inspects* these (ranks tracer levels, falls through raw operators) rather than
# computing on them -- but named so the boundary reads as "a bind operand" instead of
# a bare ``object``.
BindArg = Any

# A backend-native array or tensor: a numpy / cupy ``ndarray`` for the tape backends,
# or a foreign framework's tensor (jax / torch / tf) for the delegate backends. The
# backends bridge pycograd's operands to these duck-typed values, so the concrete type
# is deliberately open (the frameworks are typed ``Any`` in ``setup.cfg``) -- but named
# so a backend signature reads as "a backend array" rather than a bare ``object`` /
# ``Any``.
BackendArray = Any

# Named aliases for numpy interop whose precise types are intractable to spell
# out (so the unavoidable ``Any`` is localized and documented, not bare).
Index = Any  # a NumPy __getitem__ key: int / slice / ndarray / tuple / None / ...
Shape = Any  # dim(s) for reshape: an int or a tuple of ints (np.reshape overloads)

__all__ = [
    "Array",
    "ArrayLike",
    "Axis",
    "BackendArray",
    "BindArg",
    "Boxed",
    "DTypeLike",
    "Hashable",
    "Index",
    "Operand",
    "Prim",
    "Rule",
    "Scalar",
    "Shape",
    "Tensor",
]

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

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from pycograd.params import Weight
    from pycograd.tensor import Var

Scalar = Union[int, float]
Array = np.ndarray
ArrayLike = Union[Scalar, Array]
# An operand is a Var, a plain number/array, or a ``Weight`` proxy (a late-bound
# reference to an ambient parameter; see ``with`` support near ``ParamDict``).
Operand = Union["Var", ArrayLike, "Weight"]
# A differentiable tensor value: a ``Var`` during tracing, or a plain ndarray
# when the same helper is run outside the tape (e.g. at eval time).
Tensor = Union["Var", np.ndarray]
Axis = Optional[Union[int, tuple[int, ...]]]
# Named aliases for numpy interop whose precise types are intractable to spell
# out (so the unavoidable ``Any`` is localized and documented, not bare).
Index = Any  # a NumPy __getitem__ key: int / slice / ndarray / tuple / None / ...
Shape = Any  # dim(s) for reshape: an int or a tuple of ints (np.reshape overloads)

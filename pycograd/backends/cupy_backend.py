# -*- coding: utf-8 -*-
"""The cupy tape backend: pycograd's own tape, on the GPU.

A *sibling* of :class:`NumpyBackend`, not a compile target. It runs the exact same
``Var`` tape, the same ``_INTERCEPT`` table, and the same ``d_*`` VJP rules -- only the
underlying array library is swapped from numpy to cupy (via ``array_module``). cupy has
no autodiff of its own; pycograd's reverse-mode tape supplies the gradients, now
computing on device. So everything is inherited from :class:`NumpyBackend` except what
genuinely differs on a GPU: leaf conversion (host<->device) and scatter-add.

Importing this module imports cupy (and cupyx), so it is reached only through the lazy
``get_backend("cupy")`` factory -- never at ``import pycograd`` time.
"""
from __future__ import annotations

import cupy
import cupyx

from pycograd.backends.numpy_backend import NumpyBackend
from pycograd.tensor import Var


class CupyBackend(NumpyBackend):
    name = "cupy"
    array_module = cupy

    def scatter_add(self, out: object, key: object, vals: object) -> None:
        # cupy has no ``cupy.add.at``; cupyx.scatter_add is the GPU scatter-add.
        cupyx.scatter_add(out, key, vals)

    def lift(self, array: object) -> Var:
        return Var(cupy.asarray(array, dtype=float))  # host -> device

    def const(self, array: object) -> object:
        return cupy.asarray(array, dtype=float)

    def to_numpy(self, tensor: object) -> object:
        value = tensor.value if isinstance(tensor, Var) else tensor
        return cupy.asnumpy(value)  # device -> host (no-op for a numpy array)

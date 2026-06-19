# -*- coding: utf-8 -*-
"""The working-dtype seam: what precision the tape computes in.

pycograd's tape historically forced float64 at every array-creation point. This
module makes that dtype a single, overridable choice -- defaulting to float64 (so
existing code is byte-for-byte unchanged) but switchable to float32/float16/bfloat16
for faster, lower-memory experiments.

The selected dtype is held in a context variable, mirroring the device seam in
:mod:`pycograd.backends`: a :func:`dtype` ``with`` block (the public, ambient API) or a
``dtype=`` keyword on the transforms (which sets the same variable for the duration of a
pass) chooses it; every ``Var``/lifted-leaf/parameter creation reads it via
:func:`current_dtype`. The deep creation points live inside instrumented user code
(``np.exp(z)`` builds a ``Var``), so an ambient variable is the only thing that can reach
them.

bfloat16 is not a native numpy dtype; like JAX, we get it from the small ``ml_dtypes``
package, imported lazily so the base install never depends on it.
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Iterator, Optional, cast

import numpy as np

# The active working dtype, per execution context. ``None`` means "use the default
# float64" -- resolved lazily so merely importing pycograd forces no choice.
_DTYPE: contextvars.ContextVar[Optional[np.dtype]] = contextvars.ContextVar(
    "pycograd_working_dtype", default=None
)

# Friendly spellings -> the numpy dtype name to resolve. bfloat16 is handled separately
# (it needs ml_dtypes), so it is deliberately absent here.
_ALIASES = {
    "float64": "float64",
    "f64": "float64",
    "double": "float64",
    "float32": "float32",
    "f32": "float32",
    "single": "float32",
    "float16": "float16",
    "f16": "float16",
    "half": "float16",
}
_BFLOAT16_NAMES = frozenset({"bfloat16", "bf16"})


def _bfloat16() -> np.dtype:
    """The numpy ``bfloat16`` dtype from ``ml_dtypes`` (imported only when bf16 is used)."""
    try:
        import ml_dtypes
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "bfloat16 needs the 'ml_dtypes' package, which is not installed; "
            "install it with `pip install pycograd[bf16]` (or `pip install ml_dtypes`)."
        ) from e
    return np.dtype(ml_dtypes.bfloat16)


def resolve_dtype(spec: object) -> np.dtype:
    """Resolve a dtype spec to a concrete (floating) numpy dtype.

    ``spec`` may be ``None`` (-> float64, the default working dtype), a numpy dtype or
    scalar type, or a friendly string (``"float32"``/``"f32"``, ``"bf16"``, ...). bf16
    resolves via ``ml_dtypes``. Non-floating dtypes are rejected -- the tape
    differentiates real-valued tensors.
    """
    if spec is None:
        return np.dtype(np.float64)
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in _BFLOAT16_NAMES:
            return _bfloat16()
        name = _ALIASES.get(key)
        if name is None:
            raise ValueError(
                f"unknown dtype {spec!r}; expected one of "
                f"{sorted(set(_ALIASES) | _BFLOAT16_NAMES)}"
            )
        return np.dtype(name)
    dt = np.dtype(
        cast(Any, spec)
    )  # a numpy dtype / scalar type / "<f4" / an ml_dtypes one
    # Accept the native floats (kind "f") and bfloat16 -- whose ml_dtypes dtype reports
    # kind "V" yet is the floating type we resolve "bf16" to. Reject ints/bools/etc.
    if dt.kind != "f" and dt.name != "bfloat16":
        raise ValueError(
            f"dtype {dt.name!r} is not a floating-point dtype; the tape computes "
            "gradients over real-valued tensors (use float64/float32/float16/bfloat16)"
        )
    return dt


def current_dtype() -> np.dtype:
    """The working dtype the tape should create arrays in right now (float64 unless set)."""
    return resolve_dtype(_DTYPE.get())


@contextlib.contextmanager
def dtype(spec: object) -> Iterator[np.dtype]:
    """Run the enclosed tape in a given working dtype (``"float32"``, ``"bf16"``, ...).

    Inside the block, every ``Var``, lifted leaf, parameter, and gradient is created in
    ``spec`` rather than float64, so the forward and backward passes -- and the
    optimizers that consume the resulting params -- compute in that precision::

        with dtype("float32"):
            params = pg.params(w=..., b=...)   # float32 weights
            value, (g,) = value_and_grad(loss)(params)   # float32 grads

    Composes with :func:`pycograd.backends.device`, e.g. ``with device("cupy"),
    dtype("float16"):`` keeps a float16 tape on the GPU.
    """
    token = _DTYPE.set(resolve_dtype(spec))
    try:
        yield resolve_dtype(_DTYPE.get())
    finally:
        _DTYPE.reset(token)


@contextlib.contextmanager
def _maybe_dtype(spec: object) -> Iterator[None]:
    """Apply :func:`dtype` for ``spec``, or do nothing when ``spec is None``.

    The ``dtype=`` keyword on the transforms passes ``None`` to mean "inherit", so a
    ``value_and_grad(f)`` call nested inside a ``with dtype("bf16"):`` block still runs in
    bf16; an explicit ``dtype=`` overrides it for that call.
    """
    if spec is None:
        yield
        return
    with dtype(spec):
        yield

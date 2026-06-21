# -*- coding: utf-8 -*-
"""PyTorch backend: swap numpy/math calls for ``torch`` and differentiate with autograd.

torch's op set maps onto numpy's almost one-to-one, but a handful of calls need a
thin adapter: torch spells the reduction axis ``dim`` (not ``axis``) and ``keepdim``
(not ``keepdims``), uses ``amax``/``amin`` for value-only max/min, ``clamp`` for clip,
``unsqueeze`` for ``expand_dims``, and requires tensor (not python-scalar) operands for
``maximum``/``minimum``/``where``. Those adapters live here; everything else maps to the
same-named ``torch`` function. Gradients come from ``torch.autograd.grad``.

Tensors are created in the active working dtype (:func:`~pycograd.dtypes.current_dtype`),
which defaults to float64 to match pycograd's float64 tape (torch otherwise defaults
float32) so gradients agree to tight tolerance; under ``dtype("float32")`` /
``dtype("bf16")`` the compiled forward runs in that precision instead. bfloat16 needs a
detour: torch cannot read or write an ``ml_dtypes.bfloat16`` numpy buffer, so bf16 leaves
are routed in through float32 and bf16 results back out through float32.
"""
from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Callable, Mapping

import numpy as np

from pycograd._typing import Array, Axis, BackendArray, DTypeLike, Prim, Shape
from pycograd.backends import Backend
from pycograd.dtypes import current_dtype
from pycograd.ops import _INTERCEPT

if TYPE_CHECKING:
    import torch


def _torch_dtype(torch: ModuleType, np_dtype: np.dtype) -> "torch.dtype":
    """The ``torch`` dtype matching a numpy dtype (names line up: float32, bfloat16, ...)."""
    return getattr(torch, np_dtype.name)


def _as_torch(torch: ModuleType, x: BackendArray) -> BackendArray:
    """Convert ``x`` to a torch tensor in the active working dtype (bf16 via float32)."""
    if isinstance(x, torch.Tensor):
        return x
    dt = current_dtype()
    if dt.name == "bfloat16":
        # torch can't ingest an ml_dtypes.bfloat16 buffer; stage through float32.
        return torch.as_tensor(np.asarray(x, dtype=np.float32)).to(torch.bfloat16)
    return torch.as_tensor(np.asarray(x, dtype=dt))


def _torch_to_numpy(torch: ModuleType, t: BackendArray) -> Array:
    """A torch tensor back to numpy, preserving bfloat16 via ``ml_dtypes`` (float32 bridge)."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
        if t.dtype == torch.bfloat16:
            import ml_dtypes

            return t.to(torch.float32).numpy().astype(ml_dtypes.bfloat16)
        return t.numpy()
    return np.asarray(t)


def _make_adapters(torch: ModuleType) -> dict[str, Prim]:
    """Build ``{numpy/math fn name: torch replacement}`` for the whole intercept set."""

    def as_t(x: BackendArray) -> BackendArray:
        return _as_torch(torch, x)

    def unary(name: str) -> Prim:
        fn = getattr(torch, name)
        return lambda x: fn(as_t(x))

    def matmul(a: BackendArray, b: BackendArray) -> BackendArray:
        return torch.matmul(as_t(a), as_t(b))

    def maximum(a: BackendArray, b: BackendArray) -> BackendArray:
        return torch.maximum(as_t(a), as_t(b))

    def minimum(a: BackendArray, b: BackendArray) -> BackendArray:
        return torch.minimum(as_t(a), as_t(b))

    def where(cond: BackendArray, a: BackendArray, b: BackendArray) -> BackendArray:
        c = (
            cond
            if isinstance(cond, torch.Tensor)
            else torch.as_tensor(np.asarray(cond))
        )
        return torch.where(c, as_t(a), as_t(b))

    def clip(
        x: BackendArray, a_min: BackendArray = None, a_max: BackendArray = None
    ) -> BackendArray:
        return torch.clamp(as_t(x), min=a_min, max=a_max)

    def reduce(name: str) -> Prim:
        fn = getattr(torch, name)

        def _r(
            x: BackendArray, axis: Axis = None, keepdims: bool = False, **_: object
        ) -> BackendArray:
            if axis is None:
                return fn(as_t(x))
            return fn(as_t(x), dim=axis, keepdim=keepdims)

        return _r

    def variance(
        x: BackendArray,
        axis: Axis = None,
        dtype: DTypeLike | None = None,
        out: BackendArray = None,
        ddof: int = 0,
        keepdims: bool = False,
        **_: object,
    ) -> BackendArray:
        if axis is None:
            return torch.var(as_t(x), correction=ddof)
        return torch.var(as_t(x), dim=axis, correction=ddof, keepdim=keepdims)

    def std(
        x: BackendArray,
        axis: Axis = None,
        dtype: DTypeLike | None = None,
        out: BackendArray = None,
        ddof: int = 0,
        keepdims: bool = False,
        **_: object,
    ) -> BackendArray:
        if axis is None:
            return torch.std(as_t(x), correction=ddof)
        return torch.std(as_t(x), dim=axis, correction=ddof, keepdim=keepdims)

    def transpose(x: BackendArray, axes: tuple[int, ...] | None = None) -> BackendArray:
        t = as_t(x)
        return t.permute(*(axes if axes is not None else range(t.ndim - 1, -1, -1)))

    def reshape(x: BackendArray, *shape: Shape) -> BackendArray:
        newshape = shape[0] if len(shape) == 1 else shape
        if isinstance(newshape, int):
            newshape = (newshape,)
        return torch.reshape(as_t(x), tuple(newshape))

    def expand_dims(x: BackendArray, axis: int) -> BackendArray:
        return torch.unsqueeze(as_t(x), axis)

    def concatenate(seq: BackendArray, axis: int = 0) -> BackendArray:
        return torch.cat([as_t(s) for s in seq], dim=axis)

    def stack(seq: BackendArray, axis: int = 0) -> BackendArray:
        return torch.stack([as_t(s) for s in seq], dim=axis)

    def listwise(name: str) -> Prim:
        fn = getattr(torch, name)
        return lambda seq: fn([as_t(s) for s in seq])

    by_name: dict[str, Prim] = {
        "abs": unary("abs"),
        "square": unary("square"),
        "reciprocal": unary("reciprocal"),
        "maximum": maximum,
        "minimum": minimum,
        "where": where,
        "clip": clip,
        "sum": reduce("sum"),
        "mean": reduce("mean"),
        "var": variance,
        "std": std,
        "max": reduce("amax"),
        "amax": reduce("amax"),
        "min": reduce("amin"),
        "amin": reduce("amin"),
        "dot": matmul,
        "matmul": matmul,
        "transpose": transpose,
        "reshape": reshape,
        "expand_dims": expand_dims,
        "concatenate": concatenate,
        "stack": stack,
        "vstack": listwise("vstack"),
        "hstack": listwise("hstack"),
        "column_stack": listwise("column_stack"),
        "dstack": listwise("dstack"),
    }
    for nm in (
        "exp",
        "log",
        "sin",
        "cos",
        "tanh",
        "sqrt",
        "sinh",
        "cosh",
        "arctan",
        "log1p",
        "expm1",
    ):
        by_name[nm] = unary(nm)
    by_name["atan"] = unary("arctan")  # math.atan
    return by_name


def _unmapped(func: Prim, is_tensor: Callable[[object], bool]) -> Prim:
    name = getattr(func, "__name__", repr(func))

    def _wrapped(*args: object, **kwargs: object) -> object:
        if any(is_tensor(a) for a in args) or any(
            is_tensor(v) for v in kwargs.values()
        ):
            raise NotImplementedError(
                f"compile(torch): no PyTorch mapping for {name!r}; cannot differentiate "
                "this call. Rewrite the net using ops with a torch equivalent."
            )
        return func(*args, **kwargs)

    return _wrapped


class TorchBackend(Backend):
    name = "torch"

    def __init__(self) -> None:
        import torch

        self._torch = torch
        adapters = _make_adapters(torch)
        self._intercept = {
            fn: adapters[getattr(fn, "__name__")]
            for fn in _INTERCEPT
            if getattr(fn, "__name__", None) in adapters
        }

    def _as_tensor(self, x: BackendArray) -> BackendArray:
        return _as_torch(self._torch, x)

    @property
    def intercept(self) -> Mapping[Prim, Prim]:
        return self._intercept

    def on_unmapped(self, func: Prim) -> Prim:
        return _unmapped(func, lambda x: isinstance(x, self._torch.Tensor))

    def lift(self, array: BackendArray) -> BackendArray:
        return _as_torch(self._torch, array)

    def const(self, array: BackendArray) -> BackendArray:
        return _as_torch(self._torch, array)

    def coerce_operand(self, value: BackendArray) -> BackendArray:
        # Promote a numpy constant (e.g. a data global) so it can share an operator
        # with a torch tensor; leave python scalars and existing tensors untouched.
        if isinstance(value, (np.ndarray, np.generic)):
            return _as_torch(self._torch, value)
        return value

    def to_numpy(self, tensor: BackendArray) -> Array:
        return _torch_to_numpy(self._torch, tensor)

    def grad_and_value(
        self,
        scalar_fn: Callable[[list[BackendArray]], BackendArray],
        leaves: list[BackendArray],
    ) -> tuple[BackendArray, list[BackendArray]]:
        torch = self._torch
        ts = [_as_torch(torch, leaf).requires_grad_(True) for leaf in leaves]
        out = self._as_tensor(scalar_fn(ts)).reshape(())
        if ts:
            grads = torch.autograd.grad(out, ts, allow_unused=True)
            grads = [
                g if g is not None else torch.zeros_like(t) for g, t in zip(grads, ts)
            ]
        else:
            grads = []
        return _torch_to_numpy(torch, out), [_torch_to_numpy(torch, g) for g in grads]

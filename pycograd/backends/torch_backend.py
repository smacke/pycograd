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
from typing import TYPE_CHECKING, Callable, Mapping, Optional, cast

import numpy as np

from pycograd._typing import Array, Axis, BackendArray, DTypeLike, Prim, Shape
from pycograd.backends import Backend
from pycograd.dtypes import current_dtype
from pycograd.ops import _INTERCEPT, d_gated_act, d_logsumexp, d_sigmoid, d_softmax

if TYPE_CHECKING:
    import torch


def _torch_dtype(torch: ModuleType, np_dtype: np.dtype) -> "torch.dtype":
    """The ``torch`` dtype matching a numpy dtype (names line up: float32, bfloat16, ...)."""
    return getattr(torch, np_dtype.name)


def _as_torch(
    torch: ModuleType,
    x: BackendArray,
    device: Optional[str] = None,
    np_dtype: Optional[np.dtype] = None,
) -> BackendArray:
    """Convert ``x`` to a torch tensor in the working dtype (bf16 via float32).

    ``np_dtype`` overrides the active :func:`~pycograd.dtypes.current_dtype` (a device
    backend like ``mps`` uses this to substitute float32 for the unsupported float64),
    and ``device`` places the result on that torch device (``None`` keeps it on the CPU,
    the historical behavior; a no-op move when ``x`` is already a tensor on it)."""
    dt = current_dtype() if np_dtype is None else np_dtype
    if isinstance(x, torch.Tensor):
        return x if device is None else x.to(device)
    if dt.name == "bfloat16":
        # torch can't ingest an ml_dtypes.bfloat16 buffer; stage through float32.
        t = torch.as_tensor(np.asarray(x, dtype=np.float32)).to(torch.bfloat16)
    else:
        t = torch.as_tensor(np.asarray(x, dtype=dt))
    return t if device is None else t.to(device)


def _torch_to_numpy(torch: ModuleType, t: BackendArray) -> Array:
    """A torch tensor back to numpy, preserving bfloat16 via ``ml_dtypes`` (float32 bridge)."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
        if t.dtype == torch.bfloat16:
            import ml_dtypes

            return t.to(torch.float32).numpy().astype(ml_dtypes.bfloat16)
        return t.numpy()
    return np.asarray(t)


def _make_adapters(
    torch: ModuleType,
    as_t: Callable[[BackendArray], BackendArray],
    device: Optional[str] = None,
) -> dict[str, Prim]:
    """Build ``{numpy/math fn name: torch replacement}`` for the whole intercept set.

    ``as_t`` is the backend's device/dtype-aware tensor converter (so every operand the
    adapters touch lands on the right device); ``device`` is threaded into the one spot
    that builds a tensor directly rather than through ``as_t`` (the ``where`` condition).
    """

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
        if isinstance(cond, torch.Tensor):
            c = cond
        else:
            c = torch.as_tensor(np.asarray(cond))
            if device is not None:
                c = c.to(device)
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
        "sigmoid",
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
    is_delegate = True

    def __init__(self, device: Optional[str] = None) -> None:
        import torch

        self._torch = torch
        # The torch device every tensor is placed on. ``None`` keeps tensors on the CPU
        # (the default torch backend); a subclass (e.g. ``mps``) names a device.
        self._device = device
        adapters = _make_adapters(torch, self._as_tensor, device)
        self._intercept = {
            fn: adapters[getattr(fn, "__name__")]
            for fn in _INTERCEPT
            if getattr(fn, "__name__", None) in adapters
        }
        # ``d_sigmoid`` is a tape-only primitive (no numpy callable, so it is absent
        # from ``_INTERCEPT``); map the primitive itself to torch.sigmoid so a direct
        # ``d_sigmoid`` call lowers instead of running its xp-based body (xp is None
        # under a delegate backend).
        self._intercept[d_sigmoid] = adapters["sigmoid"]
        # ``d_gated_act`` (tanh(f)*sigmoid(s)) is likewise tape-only; lower natively.
        self._intercept[d_gated_act] = lambda f, s: adapters["tanh"](f) * adapters[
            "sigmoid"
        ](s)
        # Fused stable softmax / logsumexp (tape-only): lower natively. torch spells the
        # axis ``dim`` and needs a concrete dim (no ``None``), so default softmax to -1
        # and reduce logsumexp over all axes when ``axis is None``.
        self._intercept[d_softmax] = lambda x, axis=-1: torch.softmax(
            self._as_tensor(x), dim=-1 if axis is None else axis
        )

        def _torch_logsumexp(
            x: BackendArray, axis: object = None, keepdims: bool = False
        ) -> BackendArray:
            t = self._as_tensor(x)
            dim = tuple(range(t.ndim)) if axis is None else axis
            return torch.logsumexp(t, dim=dim, keepdim=keepdims)

        self._intercept[d_logsumexp] = _torch_logsumexp
        # Lower the composed im2col ``conv2d`` to torch's *native* conv (a direct
        # NCHW / OIHW map), so the compiled net runs cuDNN/MKLDNN convolutions and
        # torch autograd supplies the backward -- instead of tracing the gather +
        # einsum. Keyed by the ``functional.conv2d`` object, so a call site swaps to
        # this; the numpy path (no intercept entry) keeps the composed conv and its
        # im2col autodiff. ``conv1d`` / ``causal_conv1d`` route through ``conv2d``.
        from pycograd.functional import conv2d as _conv2d

        def _torch_conv2d(
            x: BackendArray,
            w: BackendArray,
            b: Optional[BackendArray] = None,
            stride: int = 1,
            pad: int = 0,
            dilation: int = 1,
            groups: int = 1,
        ) -> BackendArray:
            return torch.nn.functional.conv2d(x, w, b, stride, pad, dilation, groups)

        self._intercept[_conv2d] = _torch_conv2d

    def _working_np_dtype(self) -> np.dtype:
        """The numpy dtype tensors are created in -- the active working dtype by default.

        A device backend that cannot run the float64 default (``mps``) overrides this to
        substitute a supported dtype; everything else inherits :func:`current_dtype`."""
        return current_dtype()

    def _as_tensor(self, x: BackendArray) -> BackendArray:
        return _as_torch(
            self._torch, x, device=self._device, np_dtype=self._working_np_dtype()
        )

    @property
    def intercept(self) -> Mapping[Prim, Prim]:
        return self._intercept

    def on_unmapped(self, func: Prim) -> Prim:
        return _unmapped(func, lambda x: isinstance(x, self._torch.Tensor))

    def lift(self, array: BackendArray) -> BackendArray:
        return self._as_tensor(array)

    def const(self, array: BackendArray) -> BackendArray:
        return self._as_tensor(array)

    def coerce_operand(self, value: BackendArray) -> BackendArray:
        # Promote a numpy constant (e.g. a data global) so it can share an operator
        # with a torch tensor; leave python scalars and existing tensors untouched.
        if isinstance(value, (np.ndarray, np.generic)):
            return self._as_tensor(value)
        return value

    def to_numpy(self, tensor: BackendArray) -> Array:
        return _torch_to_numpy(self._torch, tensor)

    def grad_and_value(
        self,
        scalar_fn: Callable[[list[BackendArray]], BackendArray],
        leaves: list[BackendArray],
    ) -> tuple[BackendArray, list[BackendArray]]:
        torch = self._torch
        ts = [self._as_tensor(leaf).requires_grad_(True) for leaf in leaves]
        out = self._as_tensor(scalar_fn(ts)).reshape(())
        if ts:
            grads = torch.autograd.grad(out, ts, allow_unused=True)
            grads = [
                g if g is not None else torch.zeros_like(t) for g, t in zip(grads, ts)
            ]
        else:
            grads = []
        return _torch_to_numpy(torch, out), [_torch_to_numpy(torch, g) for g in grads]

    def compile_grad(
        self, scalar_fn: Callable[[list[BackendArray]], BackendArray]
    ) -> Callable[[list[BackendArray]], tuple[BackendArray, list[BackendArray]]]:
        # torch.compile can't trace *through* pyccolo's dispatch directly -- Dynamo drops
        # the ``activate()`` contextvar, so a binop falls through to a numpy op on a grad
        # tensor. Instead we capture the value+grad graph ONCE the way jax/tf do: run the
        # net eagerly under ``torch.func.grad_and_value`` (functional autodiff -- forward and
        # backward) inside ``make_fx``, which records it into a clean ATen ``GraphModule``
        # with no pyccolo / contextvar left. ``torch.compile`` then optimizes that graph, and
        # every step reuses it. (``torch.jit.trace`` can't be used here: it would run the
        # backward during tracing and freeze the first step's gradients as constants.)
        torch = self._torch
        state: dict[str, object] = {"fn": None}

        def build(example: list[BackendArray]) -> object:
            from torch.func import grad_and_value
            from torch.fx.experimental.proxy_tensor import make_fx

            graph = make_fx(grad_and_value(scalar_fn))(example)
            try:
                return torch.compile(graph)
            except Exception:
                return graph  # the captured graph already avoids per-step re-tracing

        def run(
            leaves: list[BackendArray],
        ) -> tuple[BackendArray, list[BackendArray]]:
            ts = [self._as_tensor(x) for x in leaves]
            if not ts:
                out = self._as_tensor(scalar_fn(ts)).reshape(())
                return _torch_to_numpy(torch, out), []
            fn = state["fn"]
            if fn is None:
                try:
                    fn = build(ts)
                except Exception:
                    fn = "eager"  # robust fallback: correct, just not compiled
                state["fn"] = fn
            if fn == "eager":
                return self.grad_and_value(scalar_fn, leaves)
            grads, value = cast(Callable, fn)(ts)  # grad_and_value -> (grads, value)
            return _torch_to_numpy(torch, value), [
                _torch_to_numpy(torch, g) for g in grads
            ]

        return run

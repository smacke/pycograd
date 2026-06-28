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
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, cast

import numpy as np

from pycograd._typing import Array, Axis, BackendArray, DTypeLike, Index, Prim, Shape
from pycograd.backends import Backend
from pycograd.dtypes import current_dtype, is_integral_array
from pycograd.ops import _INTERCEPT, d_gated_act, d_logsumexp, d_sigmoid, d_softmax

if TYPE_CHECKING:
    import torch

# Sentinel for ``_as_tensor``'s ``device`` argument meaning "use the backend's compute
# device". Distinct from ``None``, which is a *real* device (CPU) a leaf may be pinned to.
_DEFAULT: object = object()


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
    if is_integral_array(x):
        # An integer array is an index/label, not a differentiable float operand;
        # preserve its dtype rather than casting to the working float dtype (else
        # ``table[idx]`` sees a float index and torch rejects it).
        t = torch.as_tensor(np.asarray(x))
    elif np.iscomplexobj(x):
        # A complex leaf keeps its complex dtype (torch has native complex64/128); casting
        # to the float working dtype would drop the imaginary part.
        t = torch.as_tensor(np.asarray(x))
    elif dt.name == "bfloat16":
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

    ``as_t`` is the backend's operand converter: a fresh tensor (from numpy/scalar) lands
    on the compute ``device``, but an *existing* tensor is left on its current device so
    per-leaf placement (a CPU-resident leaf under a GPU compute device) survives. Where two
    operands then disagree on device, ``_unify`` moves them to the compute device -- the
    auto-unify that lets a CPU slice meet a GPU weight in a binary op. ``device`` is also
    threaded into the one spot that builds a tensor directly (the ``where`` condition).
    """
    _target = torch.device(device) if device is not None else torch.device("cpu")

    def _unify(*ts: BackendArray) -> tuple[BackendArray, ...]:
        """Move operands onto the compute device iff they span more than one device (so a
        single-device subgraph -- e.g. the CPU embedding table -- never moves wholesale).
        """
        if len({t.device for t in ts if isinstance(t, torch.Tensor)}) <= 1:
            return ts
        return tuple(t.to(_target) if isinstance(t, torch.Tensor) else t for t in ts)

    def unary(name: str) -> Prim:
        fn = getattr(torch, name)
        return lambda x: fn(as_t(x))

    def matmul(a: BackendArray, b: BackendArray) -> BackendArray:
        return torch.matmul(*_unify(as_t(a), as_t(b)))

    def maximum(a: BackendArray, b: BackendArray) -> BackendArray:
        return torch.maximum(*_unify(as_t(a), as_t(b)))

    def minimum(a: BackendArray, b: BackendArray) -> BackendArray:
        return torch.minimum(*_unify(as_t(a), as_t(b)))

    def where(cond: BackendArray, a: BackendArray, b: BackendArray) -> BackendArray:
        if isinstance(cond, torch.Tensor):
            c = cond
        else:
            c = torch.as_tensor(np.asarray(cond))
            if device is not None:
                c = c.to(device)
        return torch.where(*_unify(c, as_t(a), as_t(b)))

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

    def astype(x: BackendArray, dtype: DTypeLike, **_kw: Any) -> BackendArray:
        from pycograd.dtypes import resolve_dtype

        return as_t(x).to(_torch_dtype(torch, resolve_dtype(dtype)))

    def concatenate(seq: BackendArray, axis: int = 0) -> BackendArray:
        return torch.cat(list(_unify(*[as_t(s) for s in seq])), dim=axis)

    def stack(seq: BackendArray, axis: int = 0) -> BackendArray:
        return torch.stack(list(_unify(*[as_t(s) for s in seq])), dim=axis)

    def listwise(name: str) -> Prim:
        fn = getattr(torch, name)
        return lambda seq: fn(list(_unify(*[as_t(s) for s in seq])))

    by_name: dict[str, Prim] = {
        "abs": unary("abs"),
        # ``np.abs``/``np.fabs`` carry __name__ "absolute"/"fabs"; torch.abs covers both
        # (and is the magnitude for complex inputs).
        "absolute": unary("abs"),
        "fabs": unary("abs"),
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
        "astype": astype,
        # complex component ops (``np.conj``/``np.conjugate`` share __name__ "conjugate").
        "conjugate": lambda x: torch.conj_physical(as_t(x)),
        "real": lambda x: torch.real(as_t(x)),
        "imag": lambda x: torch.imag(as_t(x)),
        "angle": lambda x: torch.angle(as_t(x)),
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
        adapters = _make_adapters(torch, self._operand, device)
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
        # axis ``dim`` and needs a concrete dim (no ``None``), and ``axis=None`` means
        # "over all axes" in the numpy reference -- so flatten/softmax/reshape for that
        # case rather than silently reducing only the last axis. logsumexp likewise
        # reduces over all axes when ``axis is None``.
        def _torch_softmax(x: BackendArray, axis: object = -1) -> BackendArray:
            t = self._operand(x)
            if axis is None:
                return torch.softmax(t.reshape(-1), dim=0).reshape(t.shape)
            return torch.softmax(t, dim=axis)

        self._intercept[d_softmax] = _torch_softmax

        def _torch_logsumexp(
            x: BackendArray, axis: object = None, keepdims: bool = False
        ) -> BackendArray:
            t = self._operand(x)
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
        # Lower the ``embedding`` row-gather to torch's native ``F.embedding`` (which
        # honors ``padding_idx`` by zeroing that row's gradient -- exactly our primitive's
        # semantics), instead of fancy-indexing a ``Var``. Keyed by ``functional.embedding``
        # so the call site swaps; the numpy path keeps ``ops.d_embedding`` and its
        # scatter-add VJP. A table with more than one feature dim is flattened to 2-D for
        # ``F.embedding`` (which wants a 2-D weight) and the result is restored.
        from pycograd.functional import embedding as _embedding

        def _torch_embedding(
            table: BackendArray,
            indices: BackendArray,
            padding_idx: Optional[int] = None,
        ) -> BackendArray:
            t = self._operand(table)
            # Gather on the *table's* device (it may be a CPU-resident offload leaf while
            # compute is on the GPU): align the index to ``t`` rather than pulling the
            # whole table to the compute device, mirroring ``align_key``.
            idx = self._operand(indices).long().to(t.device)
            feat = tuple(t.shape[1:])
            flat = t.reshape(t.shape[0], -1)
            out = torch.nn.functional.embedding(idx, flat, padding_idx=padding_idx)
            return out.reshape(*tuple(idx.shape), *feat)

        self._intercept[_embedding] = _torch_embedding

    def _working_np_dtype(self) -> np.dtype:
        """The numpy dtype tensors are created in -- the active working dtype by default.

        A device backend that cannot run the float64 default (``mps``) overrides this to
        substitute a supported dtype; everything else inherits :func:`current_dtype`."""
        return current_dtype()

    def _as_tensor(self, x: BackendArray, device: object = _DEFAULT) -> BackendArray:
        """Lift ``x`` onto a device (the compute device by default; an explicit home device
        for a leaf/const). An existing tensor is *moved* to the target -- so a leaf is
        placed where it belongs, and a 0-d loss is pulled to the compute device."""
        dev = self._device if device is _DEFAULT else cast("Optional[str]", device)
        return _as_torch(self._torch, x, device=dev, np_dtype=self._working_np_dtype())

    def _operand(self, x: BackendArray) -> BackendArray:
        """Convert an *operand* for an op: a fresh tensor (from numpy/scalar) is created on
        the compute device, but an existing tensor is left on its current device, so a
        CPU-resident leaf survives until ``_unify`` reconciles it at a mixed-device op.
        """
        if isinstance(x, self._torch.Tensor):
            return x
        return self._as_tensor(x)

    @property
    def intercept(self) -> Mapping[Prim, Prim]:
        return self._intercept

    def on_unmapped(self, func: Prim) -> Prim:
        return _unmapped(func, lambda x: isinstance(x, self._torch.Tensor))

    def lift(self, array: BackendArray) -> BackendArray:
        return self._as_tensor(array)

    def const(self, array: BackendArray, device: str | None = None) -> BackendArray:
        # ``device`` (from a ``frozen``/plain leaf wrapped in ``on_cpu``/``on_device``) pins
        # the constant to its home device; ``None`` uses the compute device.
        return self._as_tensor(array, device=_DEFAULT if device is None else device)

    def coerce_operand(self, value: BackendArray) -> BackendArray:
        # Promote a numpy constant (e.g. a data global) so it can share an operator
        # with a torch tensor; leave python scalars and existing tensors untouched.
        if isinstance(value, (np.ndarray, np.generic)):
            return self._as_tensor(value)
        return value

    def colocate(
        self, a: BackendArray, b: BackendArray
    ) -> tuple[BackendArray, BackendArray]:
        # Two operands of an ambient ``Weight`` binop that disagree on device (a CPU slice
        # meeting a GPU weight): move both to the compute device so the torch op succeeds.
        torch = self._torch
        if (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.device != b.device
        ):
            target = torch.device(self._device) if self._device else torch.device("cpu")
            return a.to(target), b.to(target)
        return a, b

    def align_key(self, data: BackendArray, key: Index) -> Index:
        # A tensor index for a gather must sit on the indexed tensor's device; move it
        # there (e.g. a GPU-lifted index gathering a CPU-resident embedding table). A
        # host/int/slice key is left untouched.
        torch = self._torch
        if (
            isinstance(data, torch.Tensor)
            and isinstance(key, torch.Tensor)
            and key.device != data.device
        ):
            return key.to(data.device)
        return key

    def to_numpy(self, tensor: BackendArray) -> Array:
        return _torch_to_numpy(self._torch, tensor)

    def _lift_leaves(
        self, leaves: list[BackendArray], devices: "list[str | None] | None"
    ) -> list[BackendArray]:
        """Lift each leaf onto its home device (``devices[i]``, or the compute device when
        ``None``) -- so a CPU-tagged leaf is created on the CPU and keeps its gradient there
        while the rest of the net runs on the compute device."""
        devs = devices if devices is not None else [None] * len(leaves)
        return [
            self._as_tensor(leaf, device=_DEFAULT if d is None else d)
            for leaf, d in zip(leaves, devs)
        ]

    def grad_and_value(
        self,
        scalar_fn: Callable[[list[BackendArray]], BackendArray],
        leaves: list[BackendArray],
        devices: "list[str | None] | None" = None,
    ) -> tuple[BackendArray, list[BackendArray]]:
        torch = self._torch
        ts = [t.requires_grad_(True) for t in self._lift_leaves(leaves, devices)]
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
        self,
        scalar_fn: Callable[[list[BackendArray]], BackendArray],
        devices: "list[str | None] | None" = None,
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
        # ``fn`` is the compiled callable once built, the ``"eager"`` sentinel if
        # compilation fell back, or ``None`` before the first call.
        state: dict[str, "Callable[..., Any] | str | None"] = {"fn": None}

        def build(example: list[BackendArray]) -> "Callable[..., Any]":
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
            ts = self._lift_leaves(leaves, devices)
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
                return self.grad_and_value(scalar_fn, leaves, devices)
            grads, value = cast(Callable, fn)(ts)  # grad_and_value -> (grads, value)
            return _torch_to_numpy(torch, value), [
                _torch_to_numpy(torch, g) for g in grads
            ]

        return run

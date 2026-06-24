# -*- coding: utf-8 -*-
"""TensorFlow backend: swap numpy/math calls for ``tf`` and differentiate with a tape.

TF diverges from numpy more than torch/jax do: math functions are split across ``tf``
and ``tf.math``, reductions are ``reduce_*`` (with ``axis``/``keepdims``), ``matmul``
needs explicit handling for vector operands, and clip is ``clip_by_value``. Those
adapters live here. Gradients come from ``tf.GradientTape`` over ``tf.Variable`` leaves in
the active working dtype (:func:`~pycograd.dtypes.current_dtype`), defaulting to float64 so
results match pycograd's float64 tape; ``dtype("float32")`` / ``dtype("bf16")`` compile the
forward in that precision instead. bfloat16 needs a float32 bridge in and out, since TF
cannot read or write an ``ml_dtypes.bfloat16`` numpy buffer.

Two TF *operator* limitations are inherent (not pycograd's): ``tensor.T`` does not exist
on a ``tf.Tensor`` (use ``np.transpose``), and the ``@`` operator requires rank >= 2 on
both sides (write ``np.dot`` for matrix-vector products). Standard Linear/LayerNorm/
attention nets that matmul rank-2 weights compile cleanly.
"""
from __future__ import annotations

from types import ModuleType
from typing import Callable, Mapping

import numpy as np

from pycograd._typing import Array, Axis, BackendArray, DTypeLike, Prim, Shape
from pycograd.backends import Backend
from pycograd.dtypes import current_dtype, is_integral_array
from pycograd.ops import _INTERCEPT, d_gated_act, d_logsumexp, d_sigmoid, d_softmax


def _as_tf(tf: ModuleType, x: BackendArray) -> BackendArray:
    """Convert ``x`` to a tf tensor in the active working dtype (bf16 via float32)."""
    if tf.is_tensor(x) or isinstance(x, tf.Variable):
        return x
    if is_integral_array(x):
        # An integer array is an index/label, not a differentiable float operand;
        # preserve its dtype rather than casting to the working float dtype.
        return tf.constant(np.asarray(x))
    dt = current_dtype()
    if dt.name == "bfloat16":
        # TF can't ingest an ml_dtypes.bfloat16 buffer; stage through float32.
        return tf.cast(tf.constant(np.asarray(x, dtype=np.float32)), tf.bfloat16)
    return tf.constant(np.asarray(x, dtype=dt))


def _tf_to_numpy(tf: ModuleType, t: BackendArray) -> Array:
    """A tf tensor back to numpy, preserving bfloat16 via ``ml_dtypes`` (float32 bridge)."""
    if tf.is_tensor(t) or isinstance(t, tf.Variable):
        if t.dtype == tf.bfloat16:
            import ml_dtypes

            return np.asarray(tf.cast(t, tf.float32)).astype(ml_dtypes.bfloat16)
        return np.asarray(t)
    return np.asarray(t)


def _make_adapters(tf: ModuleType) -> dict[str, Prim]:
    def as_t(x: BackendArray) -> BackendArray:
        return _as_tf(tf, x)

    def unary(fn: Prim) -> Prim:
        return lambda x: fn(as_t(x))

    def matmul(a: BackendArray, b: BackendArray) -> BackendArray:
        a, b = as_t(a), as_t(b)
        ar, br = len(a.shape), len(b.shape)
        if ar == 1 and br == 1:
            return tf.tensordot(a, b, 1)
        if ar == 2 and br == 1:
            return tf.linalg.matvec(a, b)
        if ar == 1 and br == 2:
            return tf.linalg.matvec(b, a, transpose_a=True)
        return tf.matmul(a, b)

    def maximum(a: BackendArray, b: BackendArray) -> BackendArray:
        return tf.maximum(as_t(a), as_t(b))

    def minimum(a: BackendArray, b: BackendArray) -> BackendArray:
        return tf.minimum(as_t(a), as_t(b))

    def where(cond: BackendArray, a: BackendArray, b: BackendArray) -> BackendArray:
        return tf.where(cond, as_t(a), as_t(b))

    def clip(
        x: BackendArray, a_min: BackendArray = None, a_max: BackendArray = None
    ) -> BackendArray:
        out = as_t(x)
        if a_min is not None:
            out = tf.maximum(out, as_t(a_min))
        if a_max is not None:
            out = tf.minimum(out, as_t(a_max))
        return out

    def reduce(fn: Prim) -> Prim:
        def _r(
            x: BackendArray, axis: Axis = None, keepdims: bool = False, **_: object
        ) -> BackendArray:
            return fn(as_t(x), axis=axis, keepdims=keepdims)

        return _r

    def _count(x: BackendArray, axis: Axis) -> int:
        shape = x.shape
        if axis is None:
            return int(np.prod(shape))
        axes = axis if isinstance(axis, tuple) else (axis,)
        return int(np.prod([shape[a] for a in axes]))

    def variance(
        x: BackendArray,
        axis: Axis = None,
        dtype: DTypeLike | None = None,
        out: BackendArray = None,
        ddof: int = 0,
        keepdims: bool = False,
        **_: object,
    ) -> BackendArray:
        x = as_t(x)
        m = tf.reduce_mean(x, axis=axis, keepdims=True)
        c = x - m
        ssq = tf.reduce_sum(c * c, axis=axis, keepdims=keepdims)
        return ssq / (_count(x, axis) - ddof)

    def std(
        x: BackendArray,
        axis: Axis = None,
        dtype: DTypeLike | None = None,
        out: BackendArray = None,
        ddof: int = 0,
        keepdims: bool = False,
        **_: object,
    ) -> BackendArray:
        return tf.sqrt(variance(x, axis=axis, ddof=ddof, keepdims=keepdims))

    def transpose(x: BackendArray, axes: Axis = None) -> BackendArray:
        return tf.transpose(as_t(x), perm=axes)

    def reshape(x: BackendArray, *shape: Shape) -> BackendArray:
        newshape = shape[0] if len(shape) == 1 else shape
        if isinstance(newshape, int):
            newshape = (newshape,)
        return tf.reshape(as_t(x), tuple(newshape))

    def expand_dims(x: BackendArray, axis: int) -> BackendArray:
        return tf.expand_dims(as_t(x), axis)

    def concatenate(seq: BackendArray, axis: int = 0) -> BackendArray:
        return tf.concat([as_t(s) for s in seq], axis=axis)

    def stack(seq: BackendArray, axis: int = 0) -> BackendArray:
        return tf.stack([as_t(s) for s in seq], axis=axis)

    # numpy's vstack/hstack/column_stack/dstack are concatenations after a per-array
    # at-least-Nd promotion (TF has no direct equivalents). Ranks are static under
    # ``tf.function`` tracing, so promote off the concrete rank, mirroring numpy.
    def _atleast_2d(t: BackendArray) -> BackendArray:
        r = len(t.shape)
        if r == 0:
            return tf.reshape(t, (1, 1))
        if r == 1:
            return tf.reshape(t, (1, -1))  # (N,) -> (1, N), like np.atleast_2d
        return t

    def _atleast_3d(t: BackendArray) -> BackendArray:
        r = len(t.shape)
        if r == 0:
            return tf.reshape(t, (1, 1, 1))
        if r == 1:
            return tf.reshape(t, (1, -1, 1))  # (N,) -> (1, N, 1)
        if r == 2:
            return tf.expand_dims(t, -1)  # (M, N) -> (M, N, 1)
        return t

    def vstack(seq: BackendArray) -> BackendArray:
        return tf.concat([_atleast_2d(as_t(s)) for s in seq], axis=0)

    def hstack(seq: BackendArray) -> BackendArray:
        ts = [as_t(s) for s in seq]
        # 1-D arrays concatenate along axis 0; otherwise along axis 1 (np.hstack).
        axis = 0 if all(len(t.shape) <= 1 for t in ts) else 1
        return tf.concat(ts, axis=axis)

    def column_stack(seq: BackendArray) -> BackendArray:
        # 1-D (N,) become columns (N, 1); 2-D pass through; then concat along axis 1.
        cols = [
            tf.reshape(t, (-1, 1)) if len(t.shape) == 1 else t
            for t in (as_t(s) for s in seq)
        ]
        return tf.concat(cols, axis=1)

    def dstack(seq: BackendArray) -> BackendArray:
        return tf.concat([_atleast_3d(as_t(s)) for s in seq], axis=2)

    m = tf.math
    by_name: dict[str, Prim] = {
        "exp": unary(tf.exp),
        "log": unary(m.log),
        "sin": unary(tf.sin),
        "cos": unary(tf.cos),
        "tanh": unary(tf.tanh),
        "sqrt": unary(tf.sqrt),
        "sigmoid": unary(m.sigmoid),
        "sinh": unary(tf.sinh),
        "cosh": unary(tf.cosh),
        "arctan": unary(tf.atan),
        "atan": unary(tf.atan),
        "log1p": unary(m.log1p),
        "expm1": unary(m.expm1),
        "abs": unary(tf.abs),
        "square": unary(tf.square),
        "reciprocal": unary(m.reciprocal),
        "maximum": maximum,
        "minimum": minimum,
        "where": where,
        "clip": clip,
        "sum": reduce(tf.reduce_sum),
        "mean": reduce(tf.reduce_mean),
        "max": reduce(tf.reduce_max),
        "amax": reduce(tf.reduce_max),
        "min": reduce(tf.reduce_min),
        "amin": reduce(tf.reduce_min),
        "var": variance,
        "std": std,
        "dot": matmul,
        "matmul": matmul,
        "transpose": transpose,
        "reshape": reshape,
        "expand_dims": expand_dims,
        "concatenate": concatenate,
        "stack": stack,
        "vstack": vstack,
        "hstack": hstack,
        "column_stack": column_stack,
        "dstack": dstack,
    }
    return by_name


def _unmapped(func: Prim, is_tensor: Callable[[object], bool]) -> Prim:
    name = getattr(func, "__name__", repr(func))

    def _wrapped(*args: object, **kwargs: object) -> object:
        if any(is_tensor(a) for a in args) or any(
            is_tensor(v) for v in kwargs.values()
        ):
            raise NotImplementedError(
                f"compile(tf): no TensorFlow mapping for {name!r}; cannot differentiate "
                "this call. Rewrite the net using ops with a tf equivalent."
            )
        return func(*args, **kwargs)

    return _wrapped


class TFBackend(Backend):
    name = "tf"
    is_delegate = True

    def __init__(self) -> None:
        import tensorflow as tf

        self._tf = tf
        adapters = _make_adapters(tf)
        self._intercept = {
            fn: adapters[getattr(fn, "__name__")]
            for fn in _INTERCEPT
            if getattr(fn, "__name__", None) in adapters
        }
        # ``d_sigmoid`` is tape-only (no numpy callable in ``_INTERCEPT``); map the
        # primitive directly to tf.math.sigmoid so a direct call lowers.
        self._intercept[d_sigmoid] = adapters["sigmoid"]
        # ``d_gated_act`` (tanh(f)*sigmoid(s)) is likewise tape-only; lower natively.
        self._intercept[d_gated_act] = lambda f, s: adapters["tanh"](f) * adapters[
            "sigmoid"
        ](s)

        # Fused stable softmax / logsumexp (tape-only): lower to tf's native ops.
        # ``axis=None`` means "over all axes" in the numpy reference; tf.nn.softmax needs
        # a concrete axis, so flatten/softmax/reshape for that case rather than silently
        # reducing only the last axis.
        def _tf_softmax(x: BackendArray, axis: Axis = -1) -> BackendArray:
            t = _as_tf(tf, x)
            if axis is None:
                flat = tf.nn.softmax(tf.reshape(t, [-1]), axis=0)
                return tf.reshape(flat, tf.shape(t))
            return tf.nn.softmax(t, axis=axis)

        self._intercept[d_softmax] = _tf_softmax
        self._intercept[d_logsumexp] = (
            lambda x, axis=None, keepdims=False: tf.reduce_logsumexp(
                _as_tf(tf, x), axis=axis, keepdims=keepdims
            )
        )
        # Lower the composed im2col ``conv2d`` to tf's native conv (as torch/jax do), so
        # the compiled net runs a cuDNN/oneDNN convolution and the tape supplies the
        # backward. pycograd is NCHW / OIHW; tf.nn.conv2d is NHWC / HWIO, so transpose in
        # and out. The OIHW kernel already encodes ``in_channels = Cin/groups``, which tf
        # reads back as the group count. The numpy path keeps the composed conv.
        # NB: TF's *CPU* conv backward rejects dilation > 1 and grouped convs; those run
        # on GPU (and the forward is correct everywhere). Other backends have no such gap.
        from pycograd.functional import conv2d as _conv2d

        def _tf_conv2d(
            x: BackendArray,
            w: BackendArray,
            b: BackendArray = None,
            stride: int = 1,
            pad: int = 0,
            dilation: int = 1,
            groups: int = 1,
        ) -> BackendArray:
            xn = tf.transpose(_as_tf(tf, x), [0, 2, 3, 1])  # NCHW -> NHWC
            wn = tf.transpose(_as_tf(tf, w), [2, 3, 1, 0])  # OIHW -> HWIO
            padding = [[0, 0], [pad, pad], [pad, pad], [0, 0]] if pad else "VALID"
            out = tf.nn.conv2d(
                xn,
                wn,
                strides=[1, stride, stride, 1],
                padding=padding,
                dilations=[1, dilation, dilation, 1],
            )
            out = tf.transpose(out, [0, 3, 1, 2])  # NHWC -> NCHW
            return out if b is None else out + tf.reshape(_as_tf(tf, b), (1, -1, 1, 1))

        self._intercept[_conv2d] = _tf_conv2d

    def _is_tensor(self, x: object) -> bool:
        tf = self._tf
        return tf.is_tensor(x) or isinstance(x, tf.Variable)

    def _as_tensor(self, x: BackendArray) -> BackendArray:
        return _as_tf(self._tf, x)

    @property
    def intercept(self) -> Mapping[Prim, Prim]:
        return self._intercept

    def on_unmapped(self, func: Prim) -> Prim:
        return _unmapped(func, self._is_tensor)

    def lift(self, array: BackendArray) -> BackendArray:
        return _as_tf(self._tf, array)

    def const(self, array: BackendArray, device: str | None = None) -> BackendArray:
        self._reject_device(device)  # per-leaf placement is torch-only for now
        return _as_tf(self._tf, array)

    def coerce_operand(self, value: BackendArray) -> BackendArray:
        if isinstance(value, (np.ndarray, np.generic)):
            return _as_tf(self._tf, value)
        return value

    def to_numpy(self, tensor: BackendArray) -> Array:
        return _tf_to_numpy(self._tf, tensor)

    def grad_and_value(
        self,
        scalar_fn: Callable[[list[BackendArray]], BackendArray],
        leaves: list[BackendArray],
        devices: "list[str | None] | None" = None,
    ) -> tuple[BackendArray, list[BackendArray]]:
        self._reject_devices(devices)  # per-leaf placement is torch-only for now
        tf = self._tf
        ts = [tf.Variable(_as_tf(tf, leaf)) for leaf in leaves]
        with tf.GradientTape() as tape:
            out = self._as_tensor(scalar_fn(ts))
        grads = tape.gradient(out, ts) if ts else []
        grads = [g if g is not None else tf.zeros_like(t) for g, t in zip(grads, ts)]
        return _tf_to_numpy(tf, out), [_tf_to_numpy(tf, g) for g in grads]

    def compile_grad(
        self,
        scalar_fn: Callable[[list[BackendArray]], BackendArray],
        devices: "list[str | None] | None" = None,
    ) -> Callable[[list[BackendArray]], tuple[BackendArray, list[BackendArray]]]:
        self._reject_devices(devices)  # per-leaf placement is torch-only for now
        # Stage the tape into a static graph with tf.function: it traces the net once
        # (keyed by the leaves' shapes/dtypes) and reruns the graph thereafter. autograph
        # is off -- the net is already lowered by pyccolo, and AutoGraph's source rewrite
        # both fails on it and is unnecessary.
        tf = self._tf

        @tf.function(autograph=False)
        def step(
            tensors: list[BackendArray],
        ) -> tuple[BackendArray, list[BackendArray]]:
            with tf.GradientTape() as tape:
                for t in tensors:
                    tape.watch(t)
                out = self._as_tensor(scalar_fn(tensors))
            grads = tape.gradient(out, tensors)
            grads = [
                g if g is not None else tf.zeros_like(t) for g, t in zip(grads, tensors)
            ]
            return out, grads

        def run(
            leaves: list[BackendArray],
        ) -> tuple[BackendArray, list[BackendArray]]:
            ts = [_as_tf(tf, x) for x in leaves]
            if not ts:
                out = self._as_tensor(scalar_fn(ts))
                return _tf_to_numpy(tf, out), []
            out, grads = step(ts)
            return _tf_to_numpy(tf, out), [_tf_to_numpy(tf, g) for g in grads]

        return run

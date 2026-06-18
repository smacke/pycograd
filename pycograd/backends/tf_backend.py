# -*- coding: utf-8 -*-
"""TensorFlow backend: swap numpy/math calls for ``tf`` and differentiate with a tape.

TF diverges from numpy more than torch/jax do: math functions are split across ``tf``
and ``tf.math``, reductions are ``reduce_*`` (with ``axis``/``keepdims``), ``matmul``
needs explicit handling for vector operands, and clip is ``clip_by_value``. Those
adapters live here. Gradients come from ``tf.GradientTape`` over float64 ``tf.Variable``
leaves, so results match pycograd's float64 tape.

Two TF *operator* limitations are inherent (not pycograd's): ``tensor.T`` does not exist
on a ``tf.Tensor`` (use ``np.transpose``), and the ``@`` operator requires rank >= 2 on
both sides (write ``np.dot`` for matrix-vector products). Standard Linear/LayerNorm/
attention nets that matmul rank-2 weights compile cleanly.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from pycograd.backends import Backend
from pycograd.ops import _INTERCEPT


def _make_adapters(tf: "Any") -> dict:
    def as_t(x: Any) -> Any:
        if tf.is_tensor(x) or isinstance(x, tf.Variable):
            return x
        return tf.constant(np.asarray(x, dtype=np.float64))

    def unary(fn: Callable[..., object]) -> Callable[..., object]:
        return lambda x: fn(as_t(x))

    def matmul(a: Any, b: Any) -> object:
        a, b = as_t(a), as_t(b)
        ar, br = len(a.shape), len(b.shape)
        if ar == 1 and br == 1:
            return tf.tensordot(a, b, 1)
        if ar == 2 and br == 1:
            return tf.linalg.matvec(a, b)
        if ar == 1 and br == 2:
            return tf.linalg.matvec(b, a, transpose_a=True)
        return tf.matmul(a, b)

    def maximum(a: object, b: object) -> object:
        return tf.maximum(as_t(a), as_t(b))

    def minimum(a: object, b: object) -> object:
        return tf.minimum(as_t(a), as_t(b))

    def where(cond: object, a: object, b: object) -> object:
        return tf.where(cond, as_t(a), as_t(b))

    def clip(x: object, a_min: object = None, a_max: object = None) -> object:
        out = as_t(x)
        if a_min is not None:
            out = tf.maximum(out, as_t(a_min))
        if a_max is not None:
            out = tf.minimum(out, as_t(a_max))
        return out

    def reduce(fn: Callable[..., object]) -> Callable[..., object]:
        def _r(
            x: object, axis: object = None, keepdims: bool = False, **_: object
        ) -> object:
            return fn(as_t(x), axis=axis, keepdims=keepdims)

        return _r

    def _count(x: Any, axis: object) -> int:
        shape = x.shape
        if axis is None:
            return int(np.prod(shape))
        axes = axis if isinstance(axis, tuple) else (axis,)
        return int(np.prod([shape[a] for a in axes]))

    def variance(
        x: object,
        axis: object = None,
        dtype: object = None,
        out: object = None,
        ddof: int = 0,
        keepdims: bool = False,
        **_: object,
    ) -> object:
        x = as_t(x)
        m = tf.reduce_mean(x, axis=axis, keepdims=True)
        c = x - m
        ssq = tf.reduce_sum(c * c, axis=axis, keepdims=keepdims)
        return ssq / (_count(x, axis) - ddof)

    def std(
        x: object,
        axis: object = None,
        dtype: object = None,
        out: object = None,
        ddof: int = 0,
        keepdims: bool = False,
        **_: object,
    ) -> object:
        return tf.sqrt(variance(x, axis=axis, ddof=ddof, keepdims=keepdims))

    def transpose(x: object, axes: object = None) -> object:
        return tf.transpose(as_t(x), perm=axes)

    def reshape(x: object, *shape: Any) -> object:
        newshape = shape[0] if len(shape) == 1 else shape
        if isinstance(newshape, int):
            newshape = (newshape,)
        return tf.reshape(as_t(x), tuple(newshape))

    def expand_dims(x: object, axis: int) -> object:
        return tf.expand_dims(as_t(x), axis)

    def concatenate(seq: Any, axis: int = 0) -> object:
        return tf.concat([as_t(s) for s in seq], axis=axis)

    def stack(seq: Any, axis: int = 0) -> object:
        return tf.stack([as_t(s) for s in seq], axis=axis)

    m = tf.math
    by_name: dict = {
        "exp": unary(tf.exp),
        "log": unary(m.log),
        "sin": unary(tf.sin),
        "cos": unary(tf.cos),
        "tanh": unary(tf.tanh),
        "sqrt": unary(tf.sqrt),
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
    }
    return by_name


def _unmapped(func: Callable[..., object], is_tensor: Callable[[object], bool]):
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

    def __init__(self) -> None:
        import tensorflow as tf

        self._tf = tf
        adapters = _make_adapters(tf)
        self._intercept = {
            fn: adapters[getattr(fn, "__name__")]
            for fn in _INTERCEPT
            if getattr(fn, "__name__", None) in adapters
        }

    def _is_tensor(self, x: object) -> bool:
        tf = self._tf
        return tf.is_tensor(x) or isinstance(x, tf.Variable)

    def _as_tensor(self, x: Any) -> Any:
        tf = self._tf
        if self._is_tensor(x):
            return x
        return tf.constant(np.asarray(x, dtype=np.float64))

    @property
    def intercept(self) -> Mapping[object, Callable[..., object]]:
        return self._intercept

    def on_unmapped(self, func: Callable[..., object]) -> Callable[..., object]:
        return _unmapped(func, self._is_tensor)

    def lift(self, array: object) -> object:
        return self._tf.constant(np.asarray(array, dtype=np.float64))

    def const(self, array: object) -> object:
        return self._tf.constant(np.asarray(array, dtype=np.float64))

    def coerce_operand(self, value: object) -> object:
        if isinstance(value, (np.ndarray, np.generic)):
            return self._tf.constant(np.asarray(value, dtype=np.float64))
        return value

    def to_numpy(self, tensor: object) -> object:
        return np.asarray(tensor)

    def grad_and_value(
        self, scalar_fn: Callable[[list], object], leaves: list
    ) -> tuple[object, list]:
        tf = self._tf
        ts = [tf.Variable(np.asarray(leaf, dtype=np.float64)) for leaf in leaves]
        with tf.GradientTape() as tape:
            out = self._as_tensor(scalar_fn(ts))
        grads = tape.gradient(out, ts) if ts else []
        grads = [g if g is not None else tf.zeros_like(t) for g, t in zip(grads, ts)]
        return np.asarray(out), [np.asarray(g) for g in grads]

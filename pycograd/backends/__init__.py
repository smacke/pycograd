# -*- coding: utf-8 -*-
"""Pluggable backends: where the tracer's call-interception swap-target lives.

pycograd executes every forward pass through one seam -- ``AutodiffTracer.resolve_call``
(:mod:`pycograd.tracer`) consults a table to swap ``np.exp``/``np.sum``/... for a
replacement. With the default :class:`NumpyBackend` the replacement is a differentiable
``d_*`` primitive that builds a numpy tape. A *compile* backend swaps the same calls
for another framework's functions (``jnp.exp``/``torch.exp``/``tf.exp``) so the forward
runs natively there and that framework's own autodiff produces gradients.

This module is deliberately framework-free: importing it pulls in **no** jax/torch/
tensorflow. Each non-numpy backend is registered as a factory that imports its
framework only when first constructed (via :func:`get_backend`).
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Callable, Iterator, Mapping


class Backend:
    """A swap-target plus the glue to differentiate through it.

    Subclasses provide:

    * ``intercept`` -- the table the tracer consults: ``{numpy_or_math_fn: replacement}``.
    * ``on_unmapped(func)`` -- fallback for a mathy call with no entry (numpy warns and
      runs anyway; compile backends raise if a live tensor flows in).
    * ``lift(array)`` / ``to_numpy(tensor)`` -- leaf conversion to/from the framework.
    * ``grad_and_value(scalar_fn, leaves)`` -- run ``scalar_fn(lifted_leaves) -> scalar``
      under the framework's autodiff and return ``(value, [grad per leaf])`` as numpy.
    """

    name: str = "?"

    @property
    def intercept(self) -> Mapping[object, Callable[..., object]]:
        raise NotImplementedError

    def on_unmapped(self, func: Callable[..., object]) -> Callable[..., object]:
        raise NotImplementedError

    def lift(self, array: object) -> object:
        raise NotImplementedError

    def const(self, array: object) -> object:
        """Lift ``array`` as a non-differentiated constant (frozen params, literals)."""
        return self.lift(array)

    def coerce_operand(self, value: object) -> object:
        """Coerce a binary-operator operand for this backend, or return it unchanged.

        Frameworks whose tensors refuse to share a Python operator with a raw numpy
        array (torch, tf) override this to promote a numpy constant -- e.g. a data
        global baked into the net -- to a non-grad tensor, so ``data @ weight`` and
        ``x * mask`` work. numpy and jax need no coercion (their arrays already share
        operators with numpy), so the default is identity and the numpy tape path is
        unchanged."""
        return value

    def to_numpy(self, tensor: object) -> object:
        raise NotImplementedError

    def grad_and_value(
        self, scalar_fn: Callable[[list], object], leaves: list
    ) -> tuple[object, list]:
        raise NotImplementedError


# The active backend the tracer reads, per execution context. ``None`` means "use the
# default numpy backend" -- resolved lazily so merely importing pycograd never forces a
# NumpyBackend construction before it is needed.
_active: contextvars.ContextVar["Backend | None"] = contextvars.ContextVar(
    "pycograd_active_backend", default=None
)


def current_backend() -> Backend:
    """The backend the tracer should swap against right now (numpy unless overridden)."""
    be = _active.get()
    return get_backend("numpy") if be is None else be


@contextlib.contextmanager
def activate(backend: Backend) -> Iterator[Backend]:
    """Make ``backend`` the active swap-target for the duration of the ``with`` block."""
    token = _active.set(backend)
    try:
        yield backend
    finally:
        _active.reset(token)


# --- lazy registry ---------------------------------------------------------
_INSTANCES: dict[str, Backend] = {}
_FACTORIES: dict[str, Callable[[], Backend]] = {}


def register_backend(name: str, factory: Callable[[], Backend]) -> None:
    """Register a backend factory under ``name``. The factory is called (and its
    framework imported) only on the first :func:`get_backend` for that name."""
    _FACTORIES[name] = factory


def get_backend(name: "str | Backend") -> Backend:
    """Resolve a backend name to its (cached) instance, constructing it on first use.

    Passing a :class:`Backend` returns it unchanged, so callers can accept either a
    name or an instance. Constructing a non-numpy backend imports its framework here
    and nowhere earlier."""
    if isinstance(name, Backend):
        return name
    cached = _INSTANCES.get(name)
    if cached is not None:
        return cached
    factory = _FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"unknown backend {name!r}; known backends: {sorted(_FACTORIES)}"
        )
    be = factory()
    _INSTANCES[name] = be
    return be


def _make_numpy() -> Backend:
    from pycograd.backends.numpy_backend import NumpyBackend

    return NumpyBackend()


def _make_jax() -> Backend:
    from pycograd.backends.jax_backend import JaxBackend

    return JaxBackend()


def _make_torch() -> Backend:
    from pycograd.backends.torch_backend import TorchBackend

    return TorchBackend()


def _make_tf() -> Backend:
    from pycograd.backends.tf_backend import TFBackend

    return TFBackend()


def _make_abstract() -> Backend:
    from pycograd.backends.abstract_backend import AbstractBackend

    return AbstractBackend()


register_backend("numpy", _make_numpy)
register_backend("abstract", _make_abstract)
register_backend("shape", _make_abstract)
register_backend("jax", _make_jax)
register_backend("torch", _make_torch)
register_backend("pytorch", _make_torch)
register_backend("tf", _make_tf)
register_backend("tensorflow", _make_tf)

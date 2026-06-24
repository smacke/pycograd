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
from types import ModuleType
from typing import Callable, Iterator, Mapping

from pycograd._typing import Array, BackendArray, Index, Prim


class Backend:
    """A swap-target plus the glue to differentiate through it.

    Subclasses provide:

    * ``intercept`` -- the table the tracer consults: ``{numpy_or_math_fn: replacement}``.
    * ``on_unmapped(func)`` -- fallback for a mathy call with no entry (numpy warns and
      runs anyway; compile backends raise if a live tensor flows in).
    * ``lift(array)`` / ``to_numpy(tensor)`` -- leaf conversion to/from the framework.
    * ``grad_and_value(scalar_fn, leaves)`` -- run ``scalar_fn(lifted_leaves) -> scalar``
      under the framework's autodiff and return ``(value, [grad per leaf])`` as numpy.

    Backends come in two flavors. A *tape* backend (numpy, cupy) runs pycograd's own
    ``Var`` tape over an array library; it additionally sets ``array_module`` (the ``xp``
    the primitives compute with) and implements ``scatter_add``. A *delegate* backend
    (jax/torch/tf) swaps the intercept table for a foreign framework's ops and lets that
    framework differentiate, so it never invokes a ``d_*`` primitive and leaves
    ``array_module`` / ``scatter_add`` unused.
    """

    name: str = "?"

    # Tape backends only: the array module (``numpy``/``cupy``) the ``d_*`` primitives
    # compute with. ``None`` on the base and on delegate backends, which never read it.
    array_module: ModuleType | None = None

    # True for *delegate* backends (jax/torch/tf): a foreign framework does the autodiff.
    # The ambient-weights proxy reads this to route a bare-weight numpy ufunc / array-func
    # onto the backend -- needed when the net is captured into a graph (torch ``make_fx``'s
    # proxy mode dispatches a binop over a weight through ``__array_ufunc__``).
    is_delegate: bool = False

    @property
    def intercept(self) -> Mapping[Prim, Prim]:
        raise NotImplementedError

    def scatter_add(self, out: BackendArray, key: Index, vals: BackendArray) -> None:
        """Scatter-add ``vals`` into ``out`` at ``key`` (tape backends only).

        Backs the indexing VJP (``Var.__getitem__``'s backward) where repeated indices
        must accumulate. numpy uses ``np.add.at``; cupy uses ``cupyx.scatter_add``.
        """
        raise NotImplementedError

    def on_unmapped(self, func: Prim) -> Prim:
        raise NotImplementedError

    def lift(self, array: BackendArray) -> BackendArray:
        raise NotImplementedError

    def const(self, array: BackendArray) -> BackendArray:
        """Lift ``array`` as a non-differentiated constant (frozen params, literals)."""
        return self.lift(array)

    def coerce_operand(self, value: BackendArray) -> BackendArray:
        """Coerce a binary-operator operand for this backend, or return it unchanged.

        Frameworks whose tensors refuse to share a Python operator with a raw numpy
        array (torch, tf) override this to promote a numpy constant -- e.g. a data
        global baked into the net -- to a non-grad tensor, so ``data @ weight`` and
        ``x * mask`` work. numpy and jax need no coercion (their arrays already share
        operators with numpy), so the default is identity and the numpy tape path is
        unchanged."""
        return value

    def to_numpy(self, tensor: BackendArray) -> Array:
        raise NotImplementedError

    def grad_and_value(
        self,
        scalar_fn: Callable[[list[BackendArray]], BackendArray],
        leaves: list[BackendArray],
    ) -> tuple[BackendArray, list[BackendArray]]:
        raise NotImplementedError

    def compile_grad(
        self, scalar_fn: Callable[[list[BackendArray]], BackendArray]
    ) -> Callable[[list[BackendArray]], tuple[BackendArray, list[BackendArray]]]:
        """Return a *reusable* ``leaves -> (value, [grad per leaf])`` callable.

        The default just re-runs :meth:`grad_and_value` on every call (correct, but no
        speedup). Every framework backend overrides it to build the gradient **once** and
        reuse the result: ``jax.jit`` / ``tf.function`` trace the net a single time (keyed by
        the leaves' shapes), and torch captures the value+grad into an ATen graph via
        ``make_fx`` then ``torch.compile``\\ s it -- so a training loop stops re-tracing
        every step. Callers cache the returned closure across steps and feed it the current
        leaf values; it is valid only while the net's structure and the scalar_fn's non-leaf
        inputs (e.g. data baked in by closure) stay fixed."""
        return lambda leaves: self.grad_and_value(scalar_fn, leaves)


# The active backend the tracer reads, per execution context. ``None`` means "use the
# default numpy backend" -- resolved lazily so merely importing pycograd never forces a
# NumpyBackend construction before it is needed.
_active: contextvars.ContextVar[Backend | None] = contextvars.ContextVar(
    "pycograd_active_backend", default=None
)


def current_backend() -> Backend:
    """The backend the tracer should swap against right now (numpy unless overridden)."""
    be = _active.get()
    return get_backend("numpy") if be is None else be


def active_backend_or_none() -> "Backend | None":
    """The active backend, or ``None`` if none is overridden (the default numpy tape).

    Unlike :func:`current_backend`, this never constructs the numpy backend, so the hot
    ``Var``-tape path (no backend activated) pays only a contextvar read -- letting the
    ambient-weights proxy cheaply detect a delegate backend swap."""
    return _active.get()


@contextlib.contextmanager
def activate(backend: Backend) -> Iterator[Backend]:
    """Make ``backend`` the active swap-target for the duration of the ``with`` block."""
    token = _active.set(backend)
    try:
        yield backend
    finally:
        _active.reset(token)


@contextlib.contextmanager
def device(name: str | Backend) -> Iterator[Backend]:
    """Run the enclosed tape on a named array backend (``"numpy"``, ``"cupy"``, ...).

    A thin, friendly wrapper over :func:`activate` / :func:`get_backend` for the device
    seam: inside the block, ``value_and_grad``/``grad`` and the optimizers compute on that
    backend's array library (so e.g. ``device("cupy")`` keeps the tape, optimizer state,
    and gradients on the GPU)::

        with device("cupy"):
            value, (g,) = value_and_grad(loss)(w)
            w = opt.step(w, g)
    """
    with activate(get_backend(name)) as be:
        yield be


# --- lazy registry ---------------------------------------------------------
_INSTANCES: dict[str, Backend] = {}
_FACTORIES: dict[str, Callable[[], Backend]] = {}


def register_backend(name: str, factory: Callable[[], Backend]) -> None:
    """Register a backend factory under ``name``. The factory is called (and its
    framework imported) only on the first :func:`get_backend` for that name."""
    _FACTORIES[name] = factory


def get_backend(name: str | Backend) -> Backend:
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


def _make_mps() -> Backend:
    from pycograd.backends.mps_backend import MpsBackend

    return MpsBackend()


def _make_abstract() -> Backend:
    from pycograd.backends.abstract_backend import AbstractBackend

    return AbstractBackend()


def _make_cupy() -> Backend:
    from pycograd.backends.cupy_backend import CupyBackend

    return CupyBackend()


register_backend("numpy", _make_numpy)
register_backend("abstract", _make_abstract)
register_backend("shape", _make_abstract)
register_backend("jax", _make_jax)
register_backend("torch", _make_torch)
register_backend("pytorch", _make_torch)
register_backend("mps", _make_mps)
register_backend("metal", _make_mps)
register_backend("tf", _make_tf)
register_backend("tensorflow", _make_tf)
register_backend("cupy", _make_cupy)
register_backend("gpu", _make_cupy)
register_backend("cuda", _make_cupy)

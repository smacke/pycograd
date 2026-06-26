# -*- coding: utf-8 -*-
"""Compile a pycograd net onto another framework (PyTorch / TensorFlow / JAX).

A pycograd net is an ordinary numpy function over pytrees of arrays. Because every
forward pass runs through one call-interception seam (:mod:`pycograd.tracer`), we can
swap the seam's target for another framework's functions and let *that* framework
execute -- and differentiate -- the net.

* :func:`compile_to` returns a forward callable that runs ``fn`` on a backend's tensors
  (pass that backend's tensors in, get one out, autograd-ready if the inputs are).
* :func:`value_and_grad` mirrors :func:`pycograd.transforms.value_and_grad` but lifts
  leaves onto the backend and reads gradients from the backend's own autodiff. The
  ``frozen`` / ``tied`` parameter semantics carry over: frozen leaves become non-grad
  constants, tied leaves share one tensor (so their gradient accumulates once).

No framework is imported until ``backend=`` selects it.
"""
from __future__ import annotations

from typing import Callable, cast

import numpy as np

from pycograd._typing import Array, BackendArray, DTypeLike, Prim
from pycograd.backends import Backend, activate, get_backend
from pycograd.dtypes import _maybe_dtype, current_dtype
from pycograd.params import Param, _TieRef
from pycograd.tensor import _is_numeric
from pycograd.tracer import _INSTRUMENTED, _make_runner
from pycograd.transforms import _check_param_ownership, _match_arg
from pycograd.tree import Leaf, PyTree, TreeDef, tree_flatten, tree_unflatten


def _leaf_array(value: object) -> Array:
    """A trainable leaf as a numpy array in the working dtype, preserving complex.

    A complex leaf keeps its complex dtype (casting to the float working dtype would drop
    the imaginary part); everything else is carried in :func:`~pycograd.dtypes.current_dtype`
    (the precision seam), mirroring ``Var.__init__`` and the backends' cast helpers."""
    arr = np.asarray(value)
    return arr if arr.dtype.kind == "c" else np.asarray(value, dtype=current_dtype())


def _runner_for(f: Prim) -> Prim:
    """The instrumented version of ``f`` (cached, shared with the numpy path)."""
    runner = _INSTRUMENTED.get(f)
    if runner is None:
        runner = _make_runner(f)
        _INSTRUMENTED[f] = runner
    return runner


def compile_to(
    fn: Prim, backend: str | Backend, *, dtype: DTypeLike | None = None
) -> Callable[..., BackendArray]:
    """Return a forward callable running ``fn`` with ``backend`` as the swap target.

    Selecting a non-numpy backend imports its framework here (and nowhere earlier).
    The returned callable expects that backend's tensors and returns one; gradients
    are the framework's job (e.g. wrap params with ``requires_grad`` for torch, or
    ``jax.grad`` the result). For an all-in-one gradient driver, use
    :func:`value_and_grad`.

    ``dtype`` selects the working precision (``"float32"``, ``"bf16"``, ...) the backend
    lifts constants in; ``None`` (the default) inherits any enclosing ``dtype(...)`` block
    or float64.
    """
    be = get_backend(backend)
    runner = _runner_for(fn)

    def forward(*args: BackendArray, **kwargs: BackendArray) -> BackendArray:
        with _maybe_dtype(dtype), activate(be):
            return runner(*args, **kwargs)

    return forward


# A leaf "slot" records how to fill one argument-leaf position during the forward, and
# how to report its gradient afterwards. One of:
#   ("train", idx, orig)     -- differentiated; tensor/grad live at index ``idx``; the
#                               leaf's home device travels in the parallel ``devices`` list
#   ("const", value, device) -- frozen Param; a non-grad constant on ``device`` (or None)
#   ("none", leaf)           -- non-numeric; passed through; gradient is None
_Slot = tuple


def _plan_leaf(
    leaf: Leaf,
    trainable: list[Array],
    tie_slot: dict[object, int],
    devices: list[str | None],
) -> _Slot:
    """Classify one leaf; for a trainable leaf record its raw value in ``trainable`` and its
    home device (``Param.device``, else ``None``) in the parallel ``devices`` list."""
    if isinstance(leaf, _TieRef):
        raise ValueError(
            "compile: tied[...] is only meaningful inside params(...), where it "
            "references a sibling parameter; it reached value_and_grad unresolved"
        )
    if isinstance(leaf, Param):
        if not leaf.trainable:
            return ("const", leaf.value, leaf.device)
        if leaf.tie is not None:
            idx = tie_slot.get(leaf.tie)
            if idx is None:
                idx = len(trainable)
                trainable.append(_leaf_array(leaf.value))
                devices.append(leaf.device)
                tie_slot[leaf.tie] = idx
            elif devices[idx] != leaf.device:
                raise ValueError(
                    "tied params share one tensor and so must share a device; got "
                    f"{devices[idx]!r} and {leaf.device!r}"
                )
            return ("train", idx, leaf)
        idx = len(trainable)
        trainable.append(_leaf_array(leaf.value))
        devices.append(leaf.device)
        return ("train", idx, leaf)
    if _is_numeric(leaf):
        idx = len(trainable)
        trainable.append(_leaf_array(leaf))
        devices.append(None)
        return ("train", idx, leaf)
    return ("none", leaf)


def _fill(slot: _Slot, tensors: list[BackendArray], be: Backend) -> BackendArray:
    if slot[0] == "train":
        return tensors[slot[1]]
    if slot[0] == "const":
        return be.const(slot[1], device=slot[2])
    return slot[1]


def _grad_for(slot: _Slot, grads: list[BackendArray], be: Backend) -> PyTree | None:
    if slot[0] == "train":
        return _match_arg(slot[2], np.asarray(be.to_numpy(grads[slot[1]])))
    return None


def value_and_grad(
    fn: Prim,
    *,
    backend: str | Backend = "numpy",
    dtype: DTypeLike | None = None,
) -> Callable[..., tuple[BackendArray, tuple[PyTree, ...]]]:
    """Wrap ``fn`` so calling it returns ``(value, grads)`` computed on ``backend``.

    Same contract as :func:`pycograd.transforms.value_and_grad`: ``grads`` is a tuple
    with one matching pytree per positional argument (``None`` at frozen / non-numeric
    leaves). The gradients come from the target framework's autodiff -- which, on the
    finite-difference-checked example models, agrees with pycograd's own numpy tape.

    ``dtype`` selects the working precision (``"float32"``, ``"bf16"``, ...) leaves are
    lifted onto the backend in; ``None`` (the default) inherits any enclosing
    ``dtype(...)`` block, else float64.
    """
    be = get_backend(backend)
    runner = _runner_for(fn)

    def wrapped(*args: PyTree) -> tuple[BackendArray, tuple[PyTree, ...]]:
        with _maybe_dtype(dtype):
            _check_param_ownership(args)
            trainable: list[Array] = []
            tie_slot: dict[object, int] = {}
            devices: list[str | None] = []  # home device per trainable leaf (parallel)
            per_arg: list[tuple[TreeDef, list[_Slot]]] = []  # (treedef, [slot])
            for a in args:
                leaves, treedef = tree_flatten(a)
                slots = [
                    _plan_leaf(leaf, trainable, tie_slot, devices) for leaf in leaves
                ]
                per_arg.append((treedef, slots))

            def scalar_fn(tensors: list[BackendArray]) -> BackendArray:
                with activate(be):
                    call_args = [
                        tree_unflatten(
                            treedef,
                            cast("list[Leaf]", [_fill(s, tensors, be) for s in slots]),
                        )
                        for treedef, slots in per_arg
                    ]
                    return runner(*call_args)

            value, grad_leaves = be.grad_and_value(scalar_fn, trainable, devices)

            grads = tuple(
                tree_unflatten(
                    treedef,
                    cast("list[Leaf]", [_grad_for(s, grad_leaves, be) for s in slots]),
                )
                for treedef, slots in per_arg
            )
            return value, grads

    return wrapped


def grad(
    fn: Prim,
    *,
    backend: str | Backend = "numpy",
    dtype: DTypeLike | None = None,
) -> Callable[..., tuple[PyTree, ...]]:
    """Like :func:`value_and_grad` but returns only the gradient tuple."""
    vg = value_and_grad(fn, backend=backend, dtype=dtype)
    return lambda *args: vg(*args)[1]

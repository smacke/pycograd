# -*- coding: utf-8 -*-
"""Static export: turn a pycograd net into a standalone, pycograd-free artifact.

Rather than build a custom graph IR, this reuses the PyTorch retarget
(:func:`pycograd.compile.compile_to`) and PyTorch's *own* exporters. :func:`to_torch_module`
wraps a net ``fn(params, *inputs)`` as a real ``torch.nn.Module`` (weight leaves become
``Parameter``s; frozen leaves become buffers), which then:

* trains with any torch optimizer (``loss.backward()``),
* serializes to TorchScript via :func:`export_torchscript` (``torch.jit.trace``), and
* exports to ONNX via :func:`export_onnx` (``torch.onnx.export``).

A TorchScript / ONNX file produced this way runs with no pycograd (or numpy-autodiff)
dependency at inference time -- the forward graph has been captured by torch's tracer.

torch is imported only when one of these functions is called.
"""
from __future__ import annotations

from typing import Callable, Sequence, cast

from pycograd.compile import compile_to
from pycograd.dtypes import _maybe_dtype
from pycograd.params import Param
from pycograd.tree import PyTree, tree_flatten, tree_unflatten


def to_torch_module(
    fn: Callable[..., object], params: PyTree, *, dtype: object = None
) -> object:
    """Wrap a net ``fn(params, *inputs)`` as a ``torch.nn.Module``.

    The returned module holds ``params``' leaves as trainable ``Parameter``s (frozen
    ``Param`` leaves become non-trainable buffers). Calling ``module(*inputs)`` runs the
    compiled-to-torch forward, reconstructing the original param pytree from the live
    tensors -- so weight structure (nested dicts/lists) is preserved.

    ``dtype`` selects the precision (``"float32"``, ``"bf16"``, ...) the weights and the
    compiled forward use; ``None`` (the default) keeps float64.
    """
    import torch

    from pycograd.backends.torch_backend import _as_torch

    run = compile_to(fn, "torch", dtype=dtype)
    leaves, treedef = tree_flatten(params)

    class PycogradModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._treedef = treedef
            self._slots: list[tuple[str, object]] = []
            self._weights = torch.nn.ParameterList()
            for i, leaf in enumerate(leaves):
                value = leaf.value if isinstance(leaf, Param) else leaf
                with _maybe_dtype(dtype):
                    tensor = _as_torch(torch, value)
                frozen = isinstance(leaf, Param) and not leaf.trainable
                if frozen:
                    name = f"_frozen_{i}"
                    self.register_buffer(name, tensor)
                    self._slots.append(("buffer", name))
                else:
                    self._slots.append(("weight", len(self._weights)))
                    self._weights.append(torch.nn.Parameter(tensor))

        def _live_leaves(self) -> list:
            return [
                (
                    self._weights[cast(int, ref)]
                    if kind == "weight"
                    else getattr(self, cast(str, ref))
                )
                for kind, ref in self._slots
            ]

        def forward(self, *inputs: object) -> object:
            live = tree_unflatten(self._treedef, self._live_leaves())
            return run(live, *inputs)

    return PycogradModule()


def export_torchscript(
    module: object, example_inputs: Sequence[object], path: str | None = None
) -> object:
    """Trace ``module`` (e.g. from :func:`to_torch_module`) to TorchScript.

    Returns the ``ScriptModule``; if ``path`` is given, also saves it there. The saved
    file loads and runs via ``torch.jit.load`` with no pycograd dependency.
    """
    import torch

    traced = torch.jit.trace(module, tuple(example_inputs))
    if path is not None:
        traced.save(path)
    return traced


def export_onnx(
    module: object, example_inputs: Sequence[object], path: str, **kwargs: object
) -> str:
    """Export ``module`` to ONNX at ``path`` (via ``torch.onnx.export``); returns ``path``.

    The resulting ``.onnx`` runs under any ONNX runtime with no pycograd dependency.
    Extra keyword arguments are forwarded to ``torch.onnx.export`` (e.g. ``opset_version``,
    ``input_names``, ``dynamic_axes``).
    """
    import torch

    torch.onnx.export(module, tuple(example_inputs), path, **kwargs)
    return path

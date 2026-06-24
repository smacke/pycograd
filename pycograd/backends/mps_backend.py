# -*- coding: utf-8 -*-
"""The MPS backend: PyTorch's autodiff, run on Apple Silicon's Metal GPU.

A *delegate* backend like :class:`~pycograd.backends.torch_backend.TorchBackend`, and in
fact a thin subclass of it: MPS (Metal Performance Shaders) is just a torch *device*, so
every op table, adapter, and the value+grad / compiled-grad machinery is inherited
unchanged -- only the device tensors live on (``"mps"``) and the working dtype differ.

The one real difference from the CPU torch backend is precision. pycograd's tape, and
the torch backend mirroring it, default to float64 so gradients agree to tight
tolerance; **the MPS framework does not support float64**, so this backend computes the
float64 default in float32 instead (gradients then match the numpy tape to a float32
tolerance, ~1e-6 in practice). An explicit ``with dtype("float32" | "float16" | "bf16")``
block is honored as-is -- MPS supports those -- and only the float64 default is
downgraded.

Importing this module imports torch; it is reached only through the lazy
``get_backend("mps")`` factory, never at ``import pycograd`` time.
"""
from __future__ import annotations

import numpy as np

from pycograd.backends.torch_backend import TorchBackend
from pycograd.dtypes import current_dtype


class MpsBackend(TorchBackend):
    name = "mps"

    def __init__(self) -> None:
        import torch

        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(
                "the 'mps' backend needs a Mac with Metal-capable PyTorch "
                "(torch.backends.mps.is_available() is False); use backend='torch' "
                "for the CPU path, or 'cuda'/'cupy' for an NVIDIA GPU"
            )
        super().__init__(device="mps")

    def _working_np_dtype(self) -> np.dtype:
        # MPS has no float64; compute the float64 default in float32 instead. Other
        # working dtypes (float32/float16/bfloat16) are supported and pass through.
        dt = current_dtype()
        return np.dtype(np.float32) if dt == np.dtype(np.float64) else dt

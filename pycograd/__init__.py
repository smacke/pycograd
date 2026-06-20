# -*- coding: utf-8 -*-
"""pycograd: a small, readable reverse-mode autograd built on numpy and pyccolo.

Write ordinary numeric Python -- including ``numpy`` calls like ``np.exp``,
``np.dot``, ``np.sum`` and operators like ``@`` -- and get correct gradients.
``Var`` is the reverse-mode tape node; ``value_and_grad`` / ``grad`` wrap a
function to return gradients with the same pytree structure as its arguments.
"""
from importlib.metadata import PackageNotFoundError, version

from pycograd._typing import Operand, Tensor
from pycograd.backends import activate, device, get_backend
from pycograd.compile import compile_to
from pycograd.data import DataLoader, batches
from pycograd.dtypes import current_dtype, dtype, resolve_dtype
from pycograd.export import export_onnx, export_torchscript, to_torch_module
from pycograd.extension import load_ipython_extension, unload_ipython_extension
from pycograd.ops import (
    AutodiffWarning,
    d_abs,
    d_arctan,
    d_clip,
    d_column_stack,
    d_concatenate,
    d_cos,
    d_cosh,
    d_dstack,
    d_exp,
    d_expand_dims,
    d_expm1,
    d_hstack,
    d_log,
    d_log1p,
    d_max,
    d_maximum,
    d_mean,
    d_min,
    d_minimum,
    d_reciprocal,
    d_reshape,
    d_sin,
    d_sinh,
    d_sqrt,
    d_square,
    d_stack,
    d_std,
    d_sum,
    d_tanh,
    d_transpose,
    d_var,
    d_vstack,
    d_where,
)
from pycograd.optimizers import (
    SGD,
    Adam,
    AdamW,
    Optimizer,
    clip_grad_norm,
    constant_lr,
    cosine_decay,
    step_decay,
)
from pycograd.params import (
    Param,
    ParamDict,
    Weight,
    frozen,
    param_values,
    params,
    register_pipescript_params_macro,
    tied,
)
from pycograd.shapes import (
    Dim,
    ShapedArray,
    ShapeDtypeStruct,
    ShapeError,
    Summary,
    bind,
    eval_shape,
    infer_shapes,
    substitute,
    summary,
)
from pycograd.tensor import Var, detach
from pycograd.tracer import AutodiffTracer, resolve_call
from pycograd.transforms import grad, gradient_descent, value_and_grad
from pycograd.tree import (
    sgd_update,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
)

try:
    __version__ = version("pycograd")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    # core
    "Var",
    "detach",
    "Tensor",
    "Operand",
    # parameters
    "Param",
    "ParamDict",
    "Weight",
    "frozen",
    "tied",
    "params",
    "param_values",
    # transforms / training
    "value_and_grad",
    "grad",
    "gradient_descent",
    "sgd_update",
    # shape inference
    "eval_shape",
    "infer_shapes",
    "substitute",
    "bind",
    "summary",
    "Summary",
    "ShapeDtypeStruct",
    "ShapedArray",
    "ShapeError",
    "Dim",
    # compile to other frameworks (torch / tf / jax)
    "compile_to",
    "get_backend",
    # device / array backend seam (numpy default, cupy for GPU)
    "device",
    "activate",
    # working-dtype seam (float64 default; float32 / float16 / bfloat16)
    "dtype",
    "current_dtype",
    "resolve_dtype",
    # static export (standalone artifacts)
    "to_torch_module",
    "export_torchscript",
    "export_onnx",
    # optimizers
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "clip_grad_norm",
    "constant_lr",
    "step_decay",
    "cosine_decay",
    # data / batching
    "batches",
    "DataLoader",
    # pytrees
    "tree_flatten",
    "tree_unflatten",
    "tree_leaves",
    "tree_structure",
    "tree_map",
    # tracer / interception
    "AutodiffTracer",
    "resolve_call",
    "AutodiffWarning",
    "register_pipescript_params_macro",
    # ipython / jupyter extension
    "load_ipython_extension",
    "unload_ipython_extension",
    # differentiable primitives
    "d_exp",
    "d_log",
    "d_sin",
    "d_cos",
    "d_tanh",
    "d_sqrt",
    "d_abs",
    "d_square",
    "d_sinh",
    "d_cosh",
    "d_arctan",
    "d_log1p",
    "d_expm1",
    "d_reciprocal",
    "d_maximum",
    "d_minimum",
    "d_clip",
    "d_where",
    "d_sum",
    "d_mean",
    "d_var",
    "d_std",
    "d_max",
    "d_min",
    "d_concatenate",
    "d_transpose",
    "d_reshape",
    "d_expand_dims",
    "d_stack",
    "d_vstack",
    "d_hstack",
    "d_column_stack",
    "d_dstack",
]

# pycograd

[![pycograd](https://github.com/smacke/pycograd/actions/workflows/ci.yml/badge.svg)](https://github.com/smacke/pycograd/actions/workflows/ci.yml)
[![checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![License: BSD3](https://img.shields.io/badge/License-BSD3-maroon.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python versions](https://img.shields.io/pypi/pyversions/pycograd.svg)](https://pypi.org/project/pycograd)
[![PyPI version](https://img.shields.io/pypi/v/pycograd.svg)](https://pypi.org/project/pycograd)

A small, readable reverse-mode automatic-differentiation library, built on numpy
and [pyccolo](https://github.com/smacke/pyccolo). Write *ordinary* numeric Python
— including `numpy` calls like `np.exp`, `np.dot`, `np.sum` and operators like
`@` — and get correct gradients, with **no special "autodiff namespace."**

It's small enough to read end to end — the kind of autograd you can use to
explain backprop on one slide — but the machinery around the core scales *up*:
auto-batching (`vmap`), forward-mode (`jvp`) and Hessians, graph capture and
optimization, gradient checkpointing, and a one-call **compile to PyTorch / JAX /
TensorFlow** — enough to write a Transformer or an RWKV recurrent net (see
[`notebooks/`](notebooks/)) and have one forward pass serve all of them.

There are **two co-equal ways to write a model**, and the same transforms work on
both:

- a **functional** surface — plain numpy functions you hand to `grad` /
  `value_and_grad` / `vmap`;
- an **ambient-weights DSL** — a `params{ ... } as weights:` block plus `|>`
  pipelines, so you write the forward *once*, by bare name, and `weights.grad`
  differentiates it.

## Install

```bash
pip install pycograd
```

## Quickstart

### Functional — `grad` over ordinary numpy

Hand any numpy function to `grad` / `value_and_grad`; the array argument is lifted
onto the tape for you.

```python
import numpy as np
from pycograd import value_and_grad

def f(x):
    return np.sum(np.sin(x * x))          # ordinary numpy -- and it differentiates

x = np.array([0.5, 1.0, 1.5])
value, (g,) = value_and_grad(f)(x)
# g == 2 * x * cos(x * x)
```

### The ambient-weights DSL — write the forward once

In a notebook or IPython session, `%load_ext pycograd` turns on the DSL: a
`params{ ... } as weights:` block builds a parameter pytree and injects the
weights as ambient names, and `|>` pipelines differentiate when a model runs
through them. Here is a 2-layer MLP classifier trained by SGD:

```python
%load_ext pycograd
import numpy as np
from pycograd import relu, softmax, cross_entropy

rng = np.random.default_rng(42)
X, Y = ...                                  # features and one-hot labels (3 classes)

with params{
    w1 = 0.3 * rng.standard_normal((2, 16)); b1 = np.zeros(16)
    w2 = 0.3 * rng.standard_normal((16, 3)); b2 = np.zeros(3)
} as weights:
    logits  = $ |> $ @ w1 + b1 |> relu |> $ @ w2 + b2     # the model, written once
    forward = $ |> logits |> softmax
    obj     = |> X |> logits |> cross_entropy($, Y)        # mean softmax cross-entropy
    for _ in range(10):
        value, grads = weights.grad(obj)                   # bind weights -> Vars, backprop
        weights.step(grads, 0.5)                           # in-place SGD
```

Weights are referenced by bare name; `relu` / `softmax` / `cross_entropy` are
first-class, finite-difference-checked fused ops imported straight from
`pycograd` (there is no op library to import for the *model* — a linear layer is
just `$ @ w + b`). `frozen[...]` holds a weight fixed (its gradient comes back
`None`), `tied(...)` shares one. `weights.grad` only *computes* gradients, so any
optimizer can consume them — swap the loop for `train(weights, obj, 200,
Adam(lr=cosine_decay(0.05, 200)))`. The very same `forward` is what `vmap` and
`compile` consume below.

## One forward, many uses

The payoff of writing the forward once is that the autodiff transforms compose
over it with no rewrites.

### Per-sample gradients with `vmap`

`vmap` turns a function written for **one** example into one that runs over a
whole batch in a single vectorized pass. Composed with `grad`, it gives something
broadcasting *cannot*: the gradient of each example separately, stacked over the
batch.

```python
from pycograd import grad, vmap

def per_example_loss(w, b, x, y):           # ONE (2,) point + ONE label -> scalar
    return x |> $ @ w + b |> cross_entropy($, y)

w = np.zeros((2, 3)); b = np.zeros(3)
gw, gb, _, _ = vmap(grad(per_example_loss), in_axes=(None, None, 0, 0))(w, b, X, Y)
# gw: (N, 2, 3)   gb: (N, 3)   -- one gradient per example
# their batch-mean is exactly the ordinary full-batch gradient
```

Per-sample gradients are exactly what gradient clipping and DP-SGD need: bound
each example's gradient norm *before* averaging. `vmap` is one trace level in the
interpreter stack, so it composes every which way — `vmap(grad(f))` gives the
per-sample gradients above, `grad(vmap(f))` runs straight through a batched
forward, and `vmap(vmap(f))` nests.

### Compile to PyTorch / JAX / TensorFlow

The same forward can be handed to *another framework's* autodiff. Pass
`backend=` to `weights.grad` (or `train`) and gradients come back from torch /
jax / tf instead of pycograd's numpy tape — matching to floating-point tolerance:

```python
v_np, g_np = weights.grad(objective)                         # pycograd's numpy tape
for backend in ("torch", "jax", "tf"):
    v_be, g_be = weights.grad(objective, backend=backend, jit=True)
    worst = max(np.max(np.abs(np.asarray(g_be[k]) - np.asarray(g_np[k]))) for k in weights)
    print("%-5s  max|grad - grad_numpy| = %.1e" % (backend, worst))   # ~1e-6
```

`compile_to(forward, "torch")` instead returns a function over the framework's
own tensors, and `weights.to_torch_module(forward)` / `export_torchscript` /
`export_onnx` package a trained net for shipping with no pycograd dependency. A
GRU, an attention block, or an RWKV cell written once thus trains on numpy,
batches under `vmap`, and compiles to three frameworks with zero rewrites
(see the notebooks below).

## Shape inference

Because a net is just a numpy function, you can ask what shapes it produces
*without* training it. `eval_shape` runs the function over abstract `(shape,
dtype)` values — no data, no allocation, so a `100000×100000` matmul is sized
instantly — and `summary` tabulates the parameters:

```python
from pycograd import eval_shape, summary, ShapeDtypeStruct as S

eval_shape(mlp_forward, S((5, 2)), S((2, 16)), S((16,)), S((16, 3)), S((3,)))  # -> f64[5,3]
summary(mlp_batch_loss, params, (5, 2), (5, 3))            # per-weight table + total params
```

Shape mismatches raise a `ShapeError` that names the op and operand shapes
(`matmul: incompatible shapes (3, 4) and (5, 6)`) instead of an opaque numpy
message; a shape that genuinely depends on data values is reported as such rather
than silently mis-sized.

## Gradient checkpointing

The tape keeps every intermediate alive until `backward`, so a deep net can run
out of memory. `checkpoint(f)` wraps a segment so its activations are **dropped on
the forward and recomputed in backward** — trading ~one extra forward pass for a
large peak-memory drop. It's a drop-in: gradients are unchanged.

```python
from pycograd import checkpoint, value_and_grad

def loss(x):
    y = checkpoint(block)(x)               # block's activations are rematerialized in backward
    return np.sum(y * y)

value, (g,) = value_and_grad(loss)(x)      # same gradient, less memory
```

It composes with positional `grad` / `value_and_grad`, the ambient
`weights.grad` loop, and `vmap` (`vmap(checkpoint(f)) == checkpoint(vmap(f))`);
`f` must be deterministic in its inputs/weights, since it is re-run to recover the
activations.

## Devices / backends

The tape runs on a pluggable **array backend** (NumPy by default). A `device(...)`
block swaps the array library the primitives, the tape, and the optimizers compute
with — so the same net trains on a GPU with no code changes, gradients and
optimizer state living on-device:

```python
from pycograd import device, value_and_grad, Adam

with device("cupy"):                       # requires a CUDA GPU + cupy (`pip install pycograd[cupy]`)
    value, (g,) = value_and_grad(loss)(w)  # tape + grads on the GPU
    w = Adam(lr=1e-3).step(w, g)           # Adam moments on the GPU too
```

CuPy mirrors NumPy's API, so the `np.exp` / `@` / `np.sum` code you already wrote
"just works." For finer control, `on_cpu[...]` / `on_device(...)` pin individual
leaves — e.g. a large embedding table on the CPU while the classifier trains on
the GPU, one autograd graph straddling both (see the device-placement notebook).
This is distinct from `compile_to`, which hands the net to *another framework's*
autodiff.

## Graph capture & rematerialization

`capture(forward, x)` records a `|>` pipeline into a flat SSA graph you can print,
`grad_graph` differentiates it, and `optimize` runs passes over it (CSE across the
forward/backward boundary, dead-code elimination). On top of that, `plan_remat`
fits a value+gradient pass under a memory budget by deciding per activation
whether to keep, spill, or recompute it — `eval_scheduled` then runs the plan to
the identical answer. See the [graph-viz](notebooks/pycograd_graph_viz_demo.ipynb)
and [remat](notebooks/pycograd_remat_demo.ipynb) notebooks.

## Examples & notebooks

The bundled demos (logistic regression, MLP, LayerNorm/Dropout, single-head
Transformer block, GRU/LSTM) train from scratch and are gradient-checked against
finite differences. Run them with:

```bash
python -m pycograd.examples
```

The [`notebooks/`](notebooks/) directory walks through the library end to end:

- [`pycograd_demo`](notebooks/pycograd_demo.ipynb) — the DSL tour: linear
  classifier → MLP → highway net → self-attention → a Transformer encoder block.
- [`pycograd_vmap_demo`](notebooks/pycograd_vmap_demo.ipynb) — where `vmap`
  earns its keep: per-sample gradients, gradient clipping, batched attention.
- [`pycograd_rnn_demo`](notebooks/pycograd_rnn_demo.ipynb) /
  [`pycograd_rwkv_demo`](notebooks/pycograd_rwkv_demo.ipynb) — GRU/LSTM and
  RWKV (trained in parallel, sampled in O(1)-per-token recurrent form).
- [`pycograd_compile_*`](notebooks/) — compile/parity against PyTorch, JAX,
  TensorFlow, and Apple MPS, plus TorchScript/ONNX export.
- [`pycograd_device_placement_demo`](notebooks/pycograd_device_placement_demo.ipynb) —
  a single pass split across CPU and GPU.
- [`pycograd_graph_viz_demo`](notebooks/pycograd_graph_viz_demo.ipynb) /
  [`pycograd_remat_demo`](notebooks/pycograd_remat_demo.ipynb) — the graph IR,
  optimization passes, and the cost-model-driven spill/remat planner.

`value_and_grad` / `grad` work the same in a notebook as anywhere else; the DSL is
the only part that needs `%load_ext pycograd`.

## How it works

* `Var` is a reverse-mode tape node wrapping a numpy array. Arithmetic operators
  are overloaded so that running a program builds a computation graph;
  `Var.backward()` then walks it in reverse to accumulate gradients.

* Operator overloading alone is *not enough*. The moment user code calls a numpy
  function — `np.exp(x)` — numpy's ufunc machinery takes over and the gradient
  link is lost. (`Var` sets `__array_ufunc__ = None` so this fails loudly instead
  of silently producing a wrong gradient.) pyccolo supplies the missing piece: its
  `before_call` event lets a handler *replace the function being called*, swapping
  `np.exp` for a differentiable `d_exp` transparently — so idiomatic numpy code
  "just differentiates." The same trick routes scalar `math.*` through the
  numpy-backed primitives, differentiates through your own helper functions by
  instrumenting them on demand, and powers the `|>` DSL.

## License

[BSD-3-Clause](docs/LICENSE.txt).

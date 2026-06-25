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

The transform API is modelled on [JAX](https://github.com/jax-ml/jax): `grad`,
`vmap`, and `jvp` are function-to-function transforms you compose freely, and a
program can be captured into an inspectable, optimizable graph — a typed SSA form
much like a JAX *jaxpr*. The difference is that pycograd is small enough to read
in an afternoon, and it differentiates the numpy you already write rather than a
look-alike array API.

## Install

```bash
pip install pycograd
```

## Quickstart

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

## Composable transforms

The transforms are borrowed from JAX, and like JAX's they compose. `grad` and
`value_and_grad` differentiate; `vmap` vectorizes a function written for **one**
example over a whole batch in a single pass; `jvp` (with `jacfwd` / `jacrev`)
gives forward-mode and Jacobians. Composing `vmap` with `grad` yields something a
plain batched backward cannot — the gradient of *each* example separately,
stacked over the batch (what gradient clipping and DP-SGD need):

```python
from pycograd import grad, vmap, cross_entropy

def per_example_loss(w, b, x, y):          # one (2,) point + one label -> scalar
    return cross_entropy(x @ w + b, y)

# in_axes maps over X and Y, holds w and b shared:
gw, gb, _, _ = vmap(grad(per_example_loss), in_axes=(None, None, 0, 0))(w, b, X, Y)
# gw: (N, 2, 3)   gb: (N, 3)   -- one gradient per example
# their batch-mean is exactly the ordinary full-batch gradient
```

`relu`, `softmax`, `cross_entropy`, `layer_norm`, and scaled-dot-product
`attention` ship as first-class, finite-difference-checked ops, so models stay
plain numpy and the transforms see straight through them.

## Inspecting the graph

A numpy function can be *captured* into a graph instead of run — the same idea as
a JAX jaxpr. `capture` records the forward, `grad_graph` differentiates it into a
combined forward+backward graph, and `optimize` cleans that up.

```python
import numpy as np
from pycograd import capture, grad_graph, optimize

def forward(x, w, b):
    h = np.tanh(x @ w + b)
    return np.sum(h * h)

g = capture(forward, x, w, b)              # trace once over (shape, dtype) inputs
```

```text
graph(%0:f64[4,3], %1:f64[3,2], %2:f64[2]) {
  %3 = matmul %0 %1 -> f64[4,2]
  %4 = add %3 %2 -> f64[4,2]
  %5 = tanh %4 -> f64[4,2]
  %6 = mul %5 %5 -> f64[4,2]
  %7 = sum %6 -> f64[]
  outputs: %7
}
```

`grad_graph(g)` returns one graph holding the value **and** the gradient w.r.t.
every input. Written naïvely, the backward pass is wasteful — it recomputes
`tanh` (`%13`, `%14`), doubles a multiply (`%10`, `%11`), and broadcasts a
constant 1.0 (`%8`, `%9`):

```text
# grad_graph(g) -- BEFORE
graph(%0:f64[4,3], %1:f64[3,2], %2:f64[2]) {
  %3 = matmul %0 %1 -> f64[4,2]
  %4 = add %3 %2 -> f64[4,2]
  %5 = tanh %4 -> f64[4,2]
  %6 = mul %5 %5 -> f64[4,2]
  %7 = sum %6 -> f64[]
  %8 = const 1.0 -> f64[]
  %9 = broadcast_to %8 [4, 2] -> f64[4,2]
  %10 = mul %9 %5 -> f64[4,2]
  %11 = mul %9 %5 -> f64[4,2]
  %12 = add %10 %11 -> f64[4,2]
  %13 = tanh %4 -> f64[4,2]              # recomputes %5
  %14 = tanh %4 -> f64[4,2]              # recomputes %5
  %15 = mul %13 %14 -> f64[4,2]          # recomputes %6
  %16 = sub 1.0 %15 -> f64[4,2]
  %17 = mul %12 %16 -> f64[4,2]
  %18 = sum %17 {axis=0} -> f64[2]
  %19 = transpose %1 [1, 0] -> f64[2,3]
  %20 = matmul %17 %19 -> f64[4,3]
  %21 = transpose %0 [1, 0] -> f64[3,4]
  %22 = matmul %21 %17 -> f64[3,2]
  outputs: %7, %20, %22, %18
}
```

`optimize` removes the redundancy by common-subexpression elimination, constant
folding, and dead-code elimination — the recomputed `tanh`/`mul` collapse back
onto `%5`/`%6` and the broadcast folds away:

```python
opt = optimize(grad_graph(g))
```

```text
# optimize(grad_graph(g)) -- AFTER
graph(%0:f64[4,3], %1:f64[3,2], %2:f64[2]) {
  %3 = matmul %0 %1 -> f64[4,2]
  %4 = add %3 %2 -> f64[4,2]
  %5 = tanh %4 -> f64[4,2]
  %6 = mul %5 %5 -> f64[4,2]
  %7 = sum %6 -> f64[]
  %12 = add %5 %5 -> f64[4,2]            # was mul %9 %5 twice; 1.0 broadcast folded away
  %16 = sub 1.0 %6 -> f64[4,2]          # reuses %6 = tanh^2 instead of recomputing tanh
  %17 = mul %12 %16 -> f64[4,2]
  %18 = sum %17 {axis=0} -> f64[2]      # grad wrt b
  %19 = transpose %1 [1, 0] -> f64[2,3]
  %20 = matmul %17 %19 -> f64[4,3]      # grad wrt x
  %21 = transpose %0 [1, 0] -> f64[3,4]
  %22 = matmul %21 %17 -> f64[3,2]      # grad wrt w
  outputs: %7, %20, %22, %18
}
```

Because the graph carries `(shape, dtype)` for every value, `eval_shape` /
`summary` can report a net's output shapes and parameter counts without running
it, and a captured forward can be handed to another framework — see below.

## Training models

For writing models, `%load_ext pycograd` enables a small DSL (built on
[pipescript](https://github.com/smacke/pipescript)): a `params{ ... }` block
declares the weights, a `|>` pipeline is the forward written once, and
`weights.grad` differentiates it. Here is a 2-layer MLP classifier:

```python
%load_ext pycograd
import numpy as np
from pycograd import relu, softmax, cross_entropy

with params{
    w1 = 0.3 * rng.standard_normal((2, 16)); b1 = np.zeros(16)
    w2 = 0.3 * rng.standard_normal((16, 3)); b2 = np.zeros(3)
} as weights:
    logits  = $ |> $ @ w1 + b1 |> relu |> $ @ w2 + b2     # the model, written once
    forward = $ |> logits |> softmax
    obj     = |> X |> logits |> cross_entropy($, Y)
    for _ in range(200):
        value, grads = weights.grad(obj)                  # backprop
        weights.step(grads, 0.5)                          # in-place SGD
```

Weights are referred to by name, `frozen[...]` holds one fixed, and any optimizer
can consume the gradients — swap the loop for `train(weights, obj, 200,
Adam(lr=cosine_decay(0.05, 200)))`. The same `forward` is also what `vmap` and
the compiler below consume.

## Compile to PyTorch / JAX / TensorFlow

The captured graph can be lowered onto another framework's autodiff. Pass
`backend=` and gradients come back from torch / jax / tf instead of the numpy
tape, matching to floating-point tolerance:

```python
for backend in ("torch", "jax", "tf"):
    v, grads = weights.grad(obj, backend=backend, jit=True)   # same model, framework autodiff
```

`compile_to(forward, "torch")` instead returns a plain function over the
framework's own tensors, and `to_torch_module` / `export_torchscript` /
`export_onnx` package a trained net for shipping with no pycograd dependency.

## Examples & notebooks

The bundled demos (logistic regression, MLP, LayerNorm/Dropout, single-head
Transformer block, GRU/LSTM) train from scratch and are gradient-checked against
finite differences:

```bash
python -m pycograd.examples
```

The [`notebooks/`](notebooks/) directory goes deeper, each as an executable
walk-through:

- [`pycograd_demo`](notebooks/pycograd_demo.ipynb) — linear classifier → MLP →
  highway net → self-attention → a Transformer encoder block.
- [`pycograd_vmap_demo`](notebooks/pycograd_vmap_demo.ipynb) — where `vmap`
  earns its keep: per-sample gradients, gradient clipping, batched attention.
- [`pycograd_rnn_demo`](notebooks/pycograd_rnn_demo.ipynb) /
  [`pycograd_rwkv_demo`](notebooks/pycograd_rwkv_demo.ipynb) — GRU/LSTM and
  RWKV (trained in parallel, sampled one token at a time).
- [`pycograd_compile_*`](notebooks/) — parity against PyTorch, JAX, TensorFlow,
  and Apple MPS, plus TorchScript / ONNX export.
- [`pycograd_graph_viz_demo`](notebooks/pycograd_graph_viz_demo.ipynb) — the
  graph IR, its rendering, and the optimization passes shown above.

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
  "just differentiates." The same mechanism routes scalar `math.*` through the
  numpy-backed primitives and powers the `|>` training DSL.

## License

[BSD-3-Clause](docs/LICENSE.txt).

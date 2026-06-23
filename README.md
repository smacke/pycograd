# pycograd

[![pycograd](https://github.com/smacke/pycograd/actions/workflows/ci.yml/badge.svg)](https://github.com/smacke/pycograd/actions/workflows/ci.yml)
[![checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![License: BSD3](https://img.shields.io/badge/License-BSD3-maroon.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python versions](https://img.shields.io/pypi/pyversions/pycograd.svg)](https://pypi.org/project/pycograd)
[![PyPI version](https://img.shields.io/pypi/v/pycograd.svg)](https://pypi.org/project/pycograd)

A small, readable reverse-mode automatic-differentiation library, built on numpy
and [pyccolo](https://github.com/smacke/pyccolo). Write *ordinary* numeric Python
— including `numpy` calls like `np.exp`, `np.dot`, `np.sum` and operators like
`@` — and get correct gradients, with no special "autodiff namespace."

pycograd grew out of the autodiff example in pyccolo. It is meant to scale *down*
to "explain backprop on one slide" and *up*, eventually, to training real models —
see [`ROADMAP.md`](ROADMAP.md).

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
  numpy-backed primitives, and differentiates through your own helper functions by
  instrumenting them on demand.

## Install

```bash
pip install pycograd
```

## Quickstart

```python
import numpy as np
from pycograd import value_and_grad, grad

def f(x):
    return np.sum(np.sin(x * x))          # ordinary numpy -- and it differentiates

x = np.array([0.5, 1.0, 1.5])
value, (g,) = value_and_grad(f)(x)
# g == 2 * x * cos(x * x)
```

Train a logistic-regression model from scratch by gradient descent:

```python
import numpy as np
from pycograd import gradient_descent

X, y = ...                                # your data
def loss(w, b):
    p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

(w, b), history = gradient_descent(loss, (np.zeros(X.shape[1]), 0.0), lr=0.5, steps=200)
```

Parameters can be structured pytrees (`params(...)` / `ParamDict`), with `frozen`
and `tied` weights, and gradients come back with the same structure. See
[`pycograd/examples/`](pycograd/examples/) for worked MLP, LayerNorm/Dropout, and
Transformer-block demos — each gradient-checked against finite differences. Run
them with:

```bash
python -m pycograd.examples
```

## Shape inference

Because a net is just a numpy function, you can ask what shapes it produces
*without* training it — for a quick sanity check or a parameter count. `eval_shape`
runs the function over abstract `(shape, dtype)` values (no data, no allocation, so a
`100000×100000` matmul is sized instantly), and `summary` tabulates the parameters:

```python
import numpy as np
from pycograd import eval_shape, summary, ShapeDtypeStruct as S

# the output shape of a forward, from input shapes alone
eval_shape(mlp_forward, S((5, 2)), S((2, 16)), S((16,)), S((16, 3)), S((3,)))
# -> f64[5,3]

summary(mlp_batch_loss, params, (5, 2), (5, 3))   # per-weight table + total params
```

Shape mismatches raise a `ShapeError` that names the op and the operand shapes
(`matmul: incompatible shapes (3, 4) and (5, 6)`) instead of an opaque numpy message,
and a shape that genuinely depends on data values (e.g. boolean-mask indexing) is
reported as such rather than silently mis-sized.

## Gradient checkpointing

The closure-tape keeps *every* intermediate alive until `backward`, so a deep net or a
long sequence can run out of memory. `checkpoint(f)` wraps a segment so its
intermediate activations are **dropped on the forward and recomputed in backward** —
trading ~one extra forward pass for a peak-memory drop from "every segment at once" to
"one segment at a time". It's a drop-in: wrap the call, gradients are unchanged.

```python
from pycograd import checkpoint, value_and_grad

def block(x):                      # a chunk of the model
    h = np.tanh(x @ W1)
    return np.tanh(h @ W2)

def loss(x):
    y = checkpoint(block)(x)       # block's activations are rematerialized in backward
    return np.sum(y * y)

value, (g,) = value_and_grad(loss)(x)   # same gradient as without checkpoint, less memory
```

Works with positional `grad` / `value_and_grad` and the ambient `weights.grad` training
loop, over arbitrary pytree outputs, and nests. `f` must be deterministic in its
inputs/weights (it is re-run to recover the activations). It also saves memory **under
`vmap`** — `vmap(checkpoint(f))` lowers the batch into the boundary
(`vmap(checkpoint(f)) == checkpoint(vmap(f))`), so `grad(vmap(checkpoint(f)))` and the
per-sample `vmap(grad(checkpoint(f)))` rematerialize the batched activations. Under a live
`jvp` checkpoint is transparent (correct gradients, no memory saving in that case).

## Devices / backends

The tape runs on a pluggable **array backend** (NumPy by default). A `device(...)`
block swaps the array library the primitives, the tape, and the optimizers compute
with — so the same net trains on a GPU with no code changes, gradients and optimizer
state living on-device:

```python
import numpy as np
from pycograd import device, value_and_grad, Adam

with device("cupy"):               # requires a CUDA GPU + cupy (`pip install pycograd[cupy]`)
    value, (g,) = value_and_grad(loss)(w)   # tape + grads on the GPU
    w = Adam(lr=1e-3).step(w, g)             # Adam moments on the GPU too
```

CuPy mirrors NumPy's API, so the same `np.exp` / `@` / `np.sum` code you already wrote
"just works"; pycograd keeps its own reverse-mode tape and only swaps the array library
underneath. (This is distinct from `compile_to(fn, "torch"|"jax"|"tf")`, which instead
hands the net to *another framework's* autodiff.)

## Notebooks / Jupyter

In a Jupyter or IPython session, `%load_ext pycograd` turns on the pycograd DSL —
it loads [pipescript](https://github.com/smacke/pipescript) (if not already
loaded) and enables the `params{ ... }` block surface and autodiff through `|>`
pipes. Requires the `pipescript` extra (`pip install pycograd[notebook]`):

```python
%load_ext pycograd
import numpy as np
from pycograd import Var

# `params{ ... }` builds a parameter pytree; `frozen` / `tied` are in scope.
model = params{
    w = 0.1 * np.random.default_rng(0).standard_normal((3, 2))
    b = frozen[np.zeros(2)]
}

# `|>` pipes differentiate when a Var flows through them.
x = Var(np.array([1.0, 2.0, 3.0]))
loss = (x |> np.exp |> np.sum)
loss.backward()
x.grad                                   # == np.exp(x)
```

`value_and_grad` / `grad` work the same in a notebook as anywhere else.

## License

[BSD-3-Clause](docs/LICENSE.txt).

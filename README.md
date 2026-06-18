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

# `autograd_suite` — HIPS autograd's test suite, ported onto pycograd

This directory is a port of the test suite from [HIPS **autograd**](https://github.com/HIPS/autograd)
(`../autograd/tests/`, MIT-licensed) onto **pycograd**. It serves two purposes:

1. **A parity / regression harness** — a large, battle-tested body of gradient-correctness
   tests run against pycograd.
2. **A test-driven gap map** — what autograd covers that pycograd does not yet. The failing
   tests are *skipped with a reason*, and [`REPORT.md`](REPORT.md) aggregates those reasons
   into a prioritized bridging plan.

## Layout

| File | Role |
|---|---|
| `_pytree.py` | `VSpace` — a tiny real-vector-space view over pytrees, replacing autograd's `vspace`. |
| `_test_util.py` | `check_grads` / `check_equivalent` / `combo_check`, ported from `autograd/test_util.py`, with `make_vjp`/`make_jvp` re-backed by pycograd. |
| `_compat.py` | autograd-shaped operators (`grad`, `value_and_grad`, `jacobian`, `hessian`, `elementwise_grad`, `make_jvp`, `make_vjp`, `deriv`, `grad_and_aux`, `make_hvp`, `hessian_tensor_product`, `tensor_jacobian_product`, `make_ggnvp`) + container/`isinstance` shims. |
| `numpy_utils.py` | `stat_check`/`unary_ufunc_check`/`binary_ufunc_check` shape sweeps. |
| `_skips.py` | Centralized skip registry for the byte-faithful op-coverage ports. |
| `conftest.py` | Per-test seed + applies `_skips.py` via `pytest_collection_modifyitems`. |
| `test_*.py` | The ported test files (one per autograd original). |

## How the port differs from autograd (read before trusting a "pass")

* **`import numpy as np`, not `autograd.numpy`.** pycograd intercepts *real* numpy inside an
  instrumented function, so the test bodies use plain numpy.
* **`grad` returns the bare gradient** of `argnum` (default 0), matching autograd/JAX — this
  is pycograd's new `argnum` convention (see `pycograd.grad`).
* **`check_grads` is first-order.** autograd recurses into the gradient function for
  `order>1`; this port does not (pycograd's eager reverse pass detaches, so an autograd-style
  reverse-over-reverse *numerical* check would read zero). pycograd's genuine higher-order AD
  is tested natively in `test/test_highorder.py`. The `order=` argument is accepted but only
  first-order checks run.
* **Real-only.** No complex numbers, scipy, fft, or linalg gradients.

## Running

```bash
pytest test/autograd_suite/ -q -rs      # -rs prints every skip reason
PYCOGRAD_RUN_SKIPS=1 pytest test/autograd_suite/ -q   # ignore the skip registry (re-triage)
```

Set `PYCOGRAD_RUN_SKIPS=1` after adding a missing VJP rule to pycograd to see which
registry-skipped tests now pass (then prune `_skips.py`).

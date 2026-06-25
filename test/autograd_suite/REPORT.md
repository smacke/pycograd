# pycograd ↔ autograd parity report & bridging plan

This is the gap report for the ported autograd test suite (see [`README.md`](README.md)).
It records what passes, what is skipped and why, and a prioritized plan for closing the gaps
in pycograd.

## How to read this

Every ported test either **passes** (pycograd matches autograd) or is **skipped with a
reason**. Two reason prefixes:

* `pycograd-gap:` — a real capability autograd has and pycograd does not (the actionable list).
* `autograd-internal:` — the test pokes an autograd implementation detail (VSpace,
  `@primitive`/`defvjp`, `const_graph`, graph naming) with no pycograd analog; not a defect,
  documented so it isn't mistaken for one.

## Summary

The suite runs **clean (zero failures/errors): ~172 passed, ~226 skipped.** Per file:

| File | passed | skipped | notes |
|---|---:|---:|---|
| test_core.py | 11 | 1 | only `%` (mod operator) missing |
| test_truediv.py | 1 | 0 | |
| test_binary_ops.py | 7 | 3 | mod / arctan2 / hypot |
| test_scalar_ops.py | 15 | 27 | unary-rule + numpy-function-form gaps |
| test_jacobian.py | 1 | 3 | `np.array`-of-Var + higher-order |
| test_dict.py | 7 | 0 | pytree grads fully work |
| test_list.py | 6 | 0 | incl. slicing |
| test_tuple.py | 3 | 1 | nested higher-order skipped |
| test_builtins.py | 1 | 0 | box-transparent `isinstance` shim |
| test_logic.py | 0 | 6 | `@primitive` / complex / `np.allclose`-in-trace |
| test_graphs.py | 11 | 8 | higher-order + complex skipped |
| test_wrappers.py | 17 | 9 | full operator surface; HVP/Hessian/jacobian/ggnvp(non-default) pass |
| test_systematic.py | 45 | 94 | op-coverage sweep |
| test_numpy.py | 42 | 65 | op-coverage sweep |
| test_complex / scipy / fft / linalg / ufunc_dispatch / performance / vspaces / tests / misc | — | module-skipped | wholesale unsupported / autograd-internal |

The **core differentiation surface works**: elementwise ufuncs (sin/cos/exp/log/tanh/sqrt/…),
reductions (sum/mean/var/max/min), `maximum`/`minimum`/`where`, matmul/`@`, broadcasting,
`einsum` (subscript form), concatenate/stack/vstack/hstack/transpose, **pytree (dict/list/tuple)
gradients**, `argnum`/`**kwargs`, and the operator family `jacobian`/`hessian`/
`elementwise_grad`/`make_jvp`/`make_vjp`/`make_hvp`/`hessian_vector_product`/
`tensor_jacobian_product`/`grad_and_aux` (the last several **landed in pycograd as part of
this work** — see below).

## What was landed in pycograd (cheap API ergonomics)

`pycograd/transforms.py` + `__init__.py`:

* `grad` / `value_and_grad` gained **`argnum`** (int → bare gradient; sequence → tuple in the
  given order; default `None` keeps the existing tuple-over-all-args behavior) and **`**kwargs`**
  passthrough (held fixed), via selective lifting — only the differentiated argument is put on
  the tape, so an op applied to a *held* argument (e.g. `np.tan`) needs no rule.
* New public operators **`jacobian`, `hessian`, `elementwise_grad`** (alias **`egrad`**),
  **`make_jvp`, `make_vjp`**. `make_vjp` is a new *public, eager, function-level* VJP transform
  (`make_vjp(f)(x) -> (vjp_fn, ans)`, vector output, reusable cotangent); it is **not** a new
  core capability — it builds on the pre-existing `Var.backward(cotangent=...)` (which already
  accepted an arbitrary output cotangent) and overlaps with the existing graph-form
  `transpose.vjp_graph(f, *primals)` (which returns a captured `Graph` rather than an eager
  closure). Covered by `test/test_autograd_api.py`.

## Gaps, prioritized

### 1. Missing numpy-op VJP rules (highest leverage)

Ranked by how many skipped tests each unblocks. These are reverse-mode (`grad`) gaps — adding
a `d_*` rule + registering it for the op would flip the corresponding tests green.

| Op family | ~tests | Notes |
|---|---:|---|
| **array-manipulation**: `repeat`, `tile`, `diff`, `gradient`, `roll`, `moveaxis`, `swapaxes`, `rollaxis`, `pad`, `select`, `sort`, `partition`, `atleast_{1,2,3}d`, `squeeze`, `ravel`, `append` | ~30 | mostly index/gather-shaped backward |
| **tensor contraction**: `np.dot` (general/≥2-D), `inner`, `outer`, `tensordot`, `kron`, `cross`, `trace`, `matmul` (general/broadcast) | ~21 | `einsum` already covers the math; these are the high-value linear-algebra entry points |
| **numpy *function* forms of arithmetic**: `np.add`/`subtract`/`multiply`/`divide`/`true_divide`/`power`/`mod`/`remainder` and `op.*` | ~15 | the *operators* (`+ - * / **`) work in reverse; the function aliases don't dispatch to a rule. Also forward-mode of a **two-tracer bilinear op** (`x*y` differentiating both args) is unsupported. |
| **triangular / diagonal**: `tril`, `triu`, `diag` | ~11 | |
| **split family**: `split`, `vsplit`, `hsplit`, `dsplit`, `array_split` | ~11 | inverse of concatenate |
| **unary ufuncs**: `tan`, `sign`, `ceil`, `floor`, `fabs`, `exp2`, `log2`, `log10`, `arcsin`, `arccos`, `arccosh`, `arcsinh`, `arctanh`, `rad2deg`/`deg2rad`/`radians`/`degrees`, `sinc` | ~17 (also in test_scalar_ops) | each is a one-line `d_*` rule |
| `np.prod`, `np.std` (ddof), `np.cumsum` (function form), `nan_to_num`, integer cast, `np.array([...])` of boxes | ~10 | misc reductions/constructors |
| `einsum` with ≥3 operands / explicit-axes (operand-index tuple) form | ~6 | the 2-operand subscript form works |
| `fmax`/`fmin`/`logaddexp`/`logaddexp2`/`arctan2`/`hypot` | ~6 | binary, no rule |
| `Var.flatten()` / `Var.squeeze()` methods, `max`/`min` equal-value tie-break | ~6 | method coverage + argmax tie semantics |

### 2. Forward-mode (`jvp`) is narrower than reverse

Several tests fail *only* on the forward check: pycograd's `jvp` lacks rules for many ops, and
does not support a **two-tracer bilinear op** (e.g. `JVPTracer * JVPTracer`, `np.dot(x, x)`).
Reverse mode for the same ops often works. This also blocks `make_ggnvp` with the default
quadratic `g` (`jvp(grad(0.5·dot(x,x)))`). Where the ported `check_grads` runs both modes, a
forward gap currently skips the whole test even though reverse would pass — so adding JVP rules
(or running these in reverse-only) would recover a large block at once.

### 3. Missing subsystems (wholesale, module-skipped)

| Subsystem | Effort | Roadmap status |
|---|---|---|
| **Complex numbers** (`real`/`imag`/`conj`/`angle`, complex dtypes, `holomorphic_grad`) | large (a parallel cotangent convention) | explicit non-goal currently |
| **`np.linalg`** gradients (`inv`/`pinv`/`solve`/`det`/`slogdet`/`eigh`/`svd`/`qr`/`cholesky`) | large | not yet started |
| **`np.fft`** | medium | not yet started |
| **`scipy`** (`special`/`stats`/`linalg`/`signal`/`integrate`) | large | no scipy backend |
| **In-place / scatter** (`A[i] = b`) | medium | Phase 2 (WIP per ROADMAP) |
| **dtype preservation** (grad keeps float32/float16; longdouble/clongdouble) | medium | pycograd defaults to a float64 working dtype |

### Future cleanup: unify `grad` and `make_vjp`

Unlike autograd (where `grad` literally *is* `make_vjp`-of-ones), pycograd's
`grad`/`value_and_grad` and `make_vjp` are **separate reverse-mode paths** that both bottom
out at `Var.backward` but neither builds on the other: `value_and_grad`'s `_run` is the
pytree-aware path (dict/list/tuple args, `Param`/frozen/tied, the higher-order `differentiable`
flag, ones-seeded scalar backward), while `make_vjp`'s `_forward_for_vjp` is a simpler
single-array-arg lift with a caller-supplied cotangent. They were left separate to avoid
disturbing the proven, hot `_run` path. **Future work:** generalize `make_vjp` to pytree
inputs and express `grad`/`value_and_grad` as `make_vjp(...)[0](ones)` for scalar output, so
there is a single reverse-mode core (benchmark first — the base `.grad` closure path was kept
over the `bind`-riding one precisely because it was ~2x faster, see `ops.py` `_VJP_FOR`).

### 4. Semantic divergences (not bugs)

Documented so they aren't mistaken for regressions:

* **No public `vspace`** — `test_vspaces.py` tests autograd's VSpace axioms + `standard_basis`;
  pycograd works directly on pytrees. (The suite's `_pytree.VSpace` is a minimal test shim.)
* **No `@primitive`/`defvjp`/`defgrad` custom-VJP API** — `test_tests.py`, the deprecated
  wrappers in `test_wrappers.py`, and the `@primitive` cases in `test_logic.py` rely on
  registering a (sometimes deliberately wrong) VJP; pycograd has no such registration surface.
* **No `const_graph` / `flatten`-to-vector** — `test_misc.py` (`tree_flatten` returns
  `(leaves, treedef)`, not a ravelled vector + unflattener).
* **`check_grads` is first-order** — see README; pycograd higher-order is covered natively in
  `test/test_highorder.py`, so `test_graphs.py`/`test_tuple.py`/`test_jacobian.py` higher-order
  cases are skipped here rather than duplicated.
* **`np.allclose` (and other non-differentiable predicates) cannot appear inside a traced
  function** — pycograd raises rather than treating them as non-traced primitives.

## Suggested first PRs (best parity-per-line)

1. **Unary ufunc rules** (`tan`, `arcsin`/`arccos`/`arctanh`/`arccosh`/`arcsinh`, `exp2`/`log2`/
   `log10`, `sign`/`ceil`/`floor`/`fabs`, deg/rad conversions, `sinc`). Each is a one-line
   derivative; flips ~17 scalar_ops + systematic/numpy tests.
2. **numpy *function* forms** of the arithmetic operators + `np.prod` — wire `np.add`/`multiply`/
   `divide`/`power`/`mod`/`negative`/`true_divide` to the existing operator rules.
3. **Tensor contraction** (`np.dot` general, `tensordot`, `inner`, `outer`) — reuse the existing
   `einsum` backward; unblocks ~21 tests and most of `test_wrappers`' jacobian-product family.
4. **Array-manipulation gather/scatter rules** (`repeat`, `tile`, `split` family, `tril`/`triu`/
   `diag`, `roll`, `moveaxis`/`swapaxes`) — the largest single bucket (~50 tests across the two
   op-coverage files).

After landing any of these, run `PYCOGRAD_RUN_SKIPS=1 pytest test/autograd_suite/` to see what
newly passes and prune `_skips.py`.

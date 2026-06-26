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
* **Unary ufunc VJP rules** (first bridging PR): `tan`, `arcsin`, `arccos`, `arctanh`,
  `arcsinh`, `arccosh`, `exp2`, `log2`, `log10`, `deg2rad`/`rad2deg`/`radians`/`degrees`,
  the zero-gradient step ufuncs `sign`/`ceil`/`floor`, and `fabs` (→ `d_abs`). Each is a
  `d_*` primitive registered across all four parity-checked tables (`_RULES`/`_UNARY_DERIV`,
  forward `_JVP`, `_BATCH`, `_ABSTRACT`); the local derivative is written once in
  `_UNARY_DERIV` and shared by the reverse and forward rules. This flipped ~29 previously
  skipped tests green (14 in `test_scalar_ops`, ~15 across `test_systematic`/`test_numpy`).
  Native regression coverage: `test/test_unary_ops.py`.
* **numpy function-form arithmetic + mod + prod** (second bridging PR): mapped the numpy
  *and* `operator` function forms (`np.add`/`operator.add`, `multiply`, `subtract`,
  `divide`/`true_divide`, `negative`, `power`) onto the existing operator primitives; added a
  new binary primitive **`d_mod`** (`np.mod`/`np.remainder`/`operator.mod` and the `%` operator
  via `ast.Mod` in `_BINOP_PRIM` + `Var.__mod__`) and a new reduction primitive **`d_prod`**
  (`np.prod`), each with reverse, forward (jvp), batching (vmap) and **shape-inference
  (eval_shape)** rules. Flipped ~44 more skipped tests green (the suite is now 224 passed /
  169 skipped). Native regression: `test/test_arith_reduce_ops.py`.
* **Tensor contraction** (third bridging PR): `np.dot` (general -- routed off `_matmul` to a
  new `d_dot`), `np.inner`, `np.tensordot`. Each **lowers to `d_einsum`**: the eager call and
  all three transform rules (forward jvp, batching/vmap, abstract/eval_shape) build the einsum
  subscript from the operand ranks and re-bind `d_einsum`, so they reuse einsum's reverse rule
  on the tape and need no separate `_VJP_FOR` entry. Flipped ~11 more tests green (suite now
  235 passed / 158 skipped). `np.outer`/`np.trace`/`np.kron`/general broadcast `np.matmul`
  remain (see gaps). Native regression: `test/test_contraction_ops.py`.
* **Array-manipulation (axis reorder + triangular)** (fourth bridging PR, first batch):
  `np.moveaxis`/`np.swapaxes`/`np.rollaxis` lower to `d_transpose` (a permutation built from
  the operand rank), and `np.tril`/`np.triu` lower to `d_mul` against a constant triangular
  mask -- same einsum-style delegation (eager + forward/vmap/eval_shape rules, no separate
  `_VJP_FOR`). Flipped ~10 more tests green (suite now 245 passed / 148 skipped). Remaining in
  this bucket: `repeat`/`tile`/`split` family/`diag`/`pad`/`diff`/`sort`/etc. Native
  regression: `test/test_manip_ops.py`.
* **Array-manipulation (roll + reshape-lowered)** (fourth PR, second batch): `np.roll` (a
  genuine linear primitive -- the VJP rolls the cotangent back by the negated shift; full
  reverse/forward/vmap/eval_shape rules), and `np.ravel`/`np.squeeze`/`np.atleast_{1,2,3}d`
  lowering to `d_reshape` (target shape computed from the operand shape). Flipped ~9 more
  tests green (suite now 254 passed / 139 skipped).
* **Array-manipulation (segment/scatter)** (fourth PR, third batch): `np.pad` (constant mode --
  linear; the VJP slices the padded cotangent back via `d_getitem`), `np.repeat`, and `np.tile`
  (the VJP is the matching sum-over-copies), each with full reverse/forward/vmap/eval_shape
  rules. The new structural primitives pass a *plain* (non-tape) input straight through to
  numpy -- matching autograd, and a correctness fix so the conv `im2col` index arrays
  (`np.repeat(np.arange(...))`) stay plain. Flipped ~8 more tests green (suite now 262 passed /
  131 skipped). The forward checker (`_jvp`) was also adjusted to hold *structural positional*
  args (e.g. `pad`'s `pad_width`) fixed rather than lifting them to tracers.
* **The split family** (fourth PR, fourth batch): `np.split`/`np.array_split`/`np.vsplit`/
  `np.hsplit`/`np.dsplit` -- the inverse of concatenate, lowered to `d_getitem` slices, so the
  op returns a *list* of pieces whose getitem VJPs scatter-add back into `x`. A single rule
  factory serves both forward and vmap (both delegate to `d_getitem`); the abstract
  (`eval_shape`) path gained list-output support (`AbstractTrace.process_primitive` tags each
  element). Flipped ~11 more tests green (suite now 273 passed / 120 skipped). Native
  regression: `test/test_split_ops.py`.
* **Binary ufuncs + diff + diag** (fifth PR): `np.logaddexp`/`logaddexp2` (smooth -- the
  softmax-weight VJP, computed stably) and `np.fmax`/`np.fmin` (reuse the maximum/minimum
  selection machinery); `np.diff` (a `getitem`/`sub` composition); `np.diag`/`np.diagonal`
  (extract = gather the diagonal indices via `d_getitem`, construct = scatter onto a zero
  diagonal via `_scatter`). `eval_shape` for all. Flipped ~8 more tests green (suite now 281
  passed / 112 skipped). (`np.make_diagonal` is autograd-specific, not numpy; `vmap` of
  diag-*construct* is unsupported -- `_scatter` has no batch rule.) Native regression:
  `test/test_more_ops.py`.
* **Gather/selection ops** (sixth PR): `np.sort`/`np.partition` (permute by the stop-gradient
  `argsort`/`argpartition` -- a take-along-axis gather whose adjoint scatters back via
  `put_along_axis`), `np.select` (a right-fold of `where`), and `np.gradient` (central
  difference, unit spacing, `edge_order=1` -- a `getitem`/`concatenate` composition; returns a
  list for `axis=None`/a tuple of axes). Full reverse/forward/vmap/eval_shape rules. Flipped ~5
  more tests green (suite now 286 passed / 107 skipped). Native regression:
  `test/test_gather_ops.py`.
* **ndarray methods + np.append** (seventh PR): the `Var`/`JVPTracer`/`BatchTracer` method
  surface gained **`.flatten()`** (aliased to `d_ravel`, since there is no `np.flatten`);
  `.ravel()`/`.squeeze()` already routed via the `np.*` names. **`np.append`** is a
  `concatenate` composition (`axis=None` ravels both operands first). Flipped ~2 more tests
  green (suite now 288 passed / 105 skipped). The python-*list*-operand `append` cases stay
  skipped (the np.array-of-boxes gap). Native regression in `test/test_manip_ops.py`.
* **Interleaved einsum** (ninth PR): numpy's explicit-axes form
  `np.einsum(op0, sublist0, op1, sublist1, ..., out_sublist)` (integer index labels instead of a
  subscript string) now normalizes to the subscript form and reuses the full einsum machinery, so
  reverse/forward/vmap/eval_shape all work over any operand count. Flipped 4 more tests green (suite
  now 301 passed / 92 skipped). The two remaining `einsum*_three_args` tests use a *repeated label
  within one operand* (a diagonal/trace inside einsum), whose adjoint needs a scatter-to-diagonal
  the reverse einsum can't express -- still skipped, with an accurate reason. Native regression:
  `test/test_einsum_interleaved.py`.
* **Flips / trace / cumsum-flatten** (eighth PR): `np.flipud`/`np.fliplr`/`np.rot90` (an axis
  `::-1` slice, plus a transpose for rot90 -- getitem/transpose compositions), `np.trace` (gather
  the diagonal indices over the leading two axes, then sum -- works for any ndim with the default
  axes), and `np.cumsum(axis=None)` (now ravels first, returning a 1-D cumulative sum like numpy,
  instead of raising). Full forward/vmap/eval_shape. Flipped ~9 more tests green (suite now 297
  passed / 96 skipped). Native regression: `test/test_flip_trace_ops.py`.
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
| **tensor contraction** — ✅ *mostly landed* (`np.dot` general, `inner`, `tensordot` lower to einsum); **`outer`/`trace`/`kron`/general-broadcast `matmul`** remain (`trace` needs a diagonal einsum einsum rejects; `outer` needs a flatten+abstract-reshape path) | ~6 left | `einsum` covers the math; the rest are follow-ups |
| **numpy *function* forms of arithmetic** — ✅ *landed* (`np.add`/`subtract`/`multiply`/`divide`/`true_divide`/`power`/`negative`, `op.*`, `mod`/`remainder`, `%`) | 2 left | only the `x**0`/`0**y` power-at-zero edge (autograd #116) remains |
| **triangular / diagonal**: `tril`, `triu`, `diag` | ~11 | |
| **split family**: `split`, `vsplit`, `hsplit`, `dsplit`, `array_split` | ~11 | inverse of concatenate |
| **unary ufuncs** — ✅ *landed* (`tan`, inverse-trig, `exp2`/`log2`/`log10`, deg/rad, `sign`/`ceil`/`floor`, `fabs`); only **`sinc`** remains (fiddly derivative, deferred) | 1 left | done in the first bridging PR |
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

These are simply **unimplemented** — none is a decided "won't support." The column below is
an *effort/shape-of-work* estimate (what new machinery each needs), not a scope judgement.

| Subsystem | Effort | What it would take |
|---|---|---|
| **Complex numbers** (`real`/`imag`/`conj`/`angle`, complex dtypes, `holomorphic_grad`) | large | a parallel cotangent convention; pycograd is real-only today |
| **`np.linalg`** gradients (`inv`/`pinv`/`solve`/`det`/`slogdet`/`eigh`/`svd`/`qr`/`cholesky`) | large | a bespoke VJP per decomposition |
| **`np.fft`** | medium | FFT/IFFT primitives + their (complex) adjoints |
| **`scipy`** (`special`/`stats`/`linalg`/`signal`/`integrate`) | large | a scipy backend + per-fn rules |
| **In-place / scatter** (`A[i] = b`) | medium | a forward scatter-add primitive |
| **dtype preservation** (grad keeps float32/float16; longdouble/clongdouble) | medium | thread the input dtype through; pycograd defaults to a float64 working dtype |

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

1. ~~**Unary ufunc rules**~~ — ✅ **done** (see "What was landed"); only `sinc` left.
2. ~~**numpy *function* forms** + `np.prod` + `mod`~~ — ✅ **done** (see "What was landed").
3. ~~**Tensor contraction**~~ — ✅ **done** for `dot`/`inner`/`tensordot` (lower to einsum);
   `outer`/`trace`/`kron`/broadcast-`matmul` are follow-ups.
4. **Array-manipulation gather/scatter rules** (`repeat`, `tile`, `split` family, `tril`/`triu`/
   `diag`, `roll`, `moveaxis`/`swapaxes`) — the largest single bucket (~50 tests across the two
   op-coverage files).

After landing any of these, run `PYCOGRAD_RUN_SKIPS=1 pytest test/autograd_suite/` to see what
newly passes and prune `_skips.py`.

# pycograd — Roadmap

> Working name. A small, readable reverse-mode automatic-differentiation library
> for inventing neural-net architectures, grown out of the autodiff example in
> [pyccolo](https://github.com/smacke/pyccolo)
> (`pyccolo/examples/autodiff.py`).

## Vision

A NumPy-backed autograd you can actually read, that scales *down* to "explain
backprop on one slide" and *up*, eventually, to training real models. The
reverse-mode core is already correct and tiny; pycograd is about building the
framework around it without losing the legibility.

## Where this comes from (current state)

The starting point is the pyccolo autodiff example, which already has:

- A reverse-mode tape node (`Var`) wrapping a NumPy array, with operator
  overloading, broadcasting (gradients reduced back over broadcast axes), and
  VJP rules for the common elementwise / reduction / linear-algebra / shape ops.
- Indexed reads (gather forward, scatter-add backward), `where`/`clip`,
  `var`/`std`, `max`/`min` with subgradients, `concatenate`, `detach`.
- `value_and_grad` / `grad`, a plain-SGD `gradient_descent`, and demos that train
  logistic regression, an MLP, an MLP+LayerNorm+Dropout, and a single-head
  Transformer block — each gradient-checked against finite differences.
- Transparent interception of `numpy`/`math` calls via pyccolo's `before_call`
  (so you write ordinary `np.exp(x)` and it differentiates), helper
  instrumentation on demand, and a pipescript `|>` integration.

So the **autograd core ships working**. Everything below is the surrounding
system.

## Guiding decision: drop the interception layer from the core

pyccolo's `before_call` interception is a great *teaching device* — it shows that
operator overloading alone can't make `np.exp(x)` differentiable and that a
tracer can fix that transparently. But it is the wrong foundation for a real
library: it only works *inside an instrumented function under the tracer*, and it
adds per-call overhead.

pycograd's `Tensor` (the productized `Var`) should instead implement NumPy's own
extension protocols — `__array_ufunc__` and `__array_function__` — so a `Tensor`
is differentiable *everywhere*, with no tracer required. (The example deliberately
sets `__array_ufunc__ = None` to motivate pyccolo; pycograd reverses that.)

pyccolo can remain an **optional "transparent mode"** for niceties the protocols
can't reach — routing scalar `math.*` through tensor ops, differentiating through
un-annotated helper bodies, and the pipescript `|>` syntax — but nothing in the
core should depend on it.

The existing demos (logistic regression → MLP → LayerNorm/Dropout → Transformer)
become the first integration tests / examples and the bar for "don't regress."

---

## Roadmap

Phased so each layer rests on the one before it. Each item is tagged with how
**crucial** it is to the goal (developing novel architectures and/or training at
scale) and a rough **feasibility / effort**. Note the tension: the single most
crucial thing *for scale* (hardware + performance) is intentionally last because
it is gated by the foundation.

### Phase 0 — Extraction & foundation

- **Carve out into `pycograd`.** Lift `Var` + the VJP rules out of the pyccolo
  example into a standalone package (`pycograd.tensor`, `pycograd.ops`), with its
  own tests/CI. Crucial: prerequisite. Feasibility: high (days).
- **A real `Tensor` type via NumPy protocols.** Implement `__array_ufunc__` /
  `__array_function__` so `np.*` on a `Tensor` dispatches to our VJPs natively;
  drop the `__array_ufunc__ = None` fail-loud stance. Crucial: foundational.
  Feasibility: medium (the protocols are fiddly but well-documented).
- **dtype & a device seam.** The array-backend seam is **done**: `device("cupy")`
  swaps the array library the tape, primitives, and optimizers compute with (NumPy
  default, CuPy for GPU), so a net trains on-device unchanged. Still open: dtype —
  stop forcing `float64`, track dtype, default `float32`, support `bf16`/`float16`.
  Crucial: high (memory + speed + the GPU on-ramp). Feasibility: medium.
- **Keep pyccolo as optional transparent mode** (math.* routing, helper
  instrumentation, pipescript `|>`), gated behind an extra. Crucial: low.
  Feasibility: high (already built).

### Phase 1 — A usable research framework (highest impact-per-effort)

- **Module / pytree parameter abstraction.** Named, nested params + state; stop
  threading bare arrays positionally (the Transformer demo takes 14 positional
  args). Crucial: **highest for developing architectures.** Feasibility: high
  (pure Python).
- **Real optimizers.** Adam/AdamW, SGD+momentum, weight decay, gradient
  clipping, LR schedules, `zero_grad`. Crucial: high. Feasibility: high.
- **Fused, numerically-stable primitives.** `log_softmax`, `logsumexp`,
  `softmax`, `cross_entropy`, `layer_norm`/`batch_norm` (with running stats) as
  first-class stable ops rather than hand-composed. Crucial: high. Feasibility:
  high (mostly composition + care).
- **Broader op coverage.** `einsum` (general contractions / attention variants),
  `gather`/`scatter` + embeddings, `sort`/`argsort`/`topk`, `cumsum`, padding,
  one-hot. Crucial: high for novel architectures. Feasibility: medium (`einsum`
  is real work).
- **Convolution / pooling.** Conv1d/2d, pooling. Crucial: high (CNNs).
  Feasibility: medium — naive is easy but slow; fast conv = im2col/FFT.
- **RNG management.** Splittable/threaded PRNG (JAX-style keys) instead of a
  hidden global generator (dropout currently uses `np.random`). Crucial: medium
  (reproducibility). Feasibility: high.
- **Training glue.** Data loading/batching/shuffling, checkpoint save/load
  (params + optimizer state), basic metrics/logging. Crucial: medium.
  Feasibility: high (orthogonal).

### Phase 2 — Depth & memory (correctness at real model sizes)

- **Gradient checkpointing.** Recompute activations in backward instead of
  retaining the whole tape; the closure-tape keeps *every* intermediate alive
  until `backward`, so deep nets / long sequences OOM. Crucial: high for scale.
  Feasibility: medium (known technique, fiddly with the closure design).
- **Explicit tape lifetime.** Free graphs deterministically; `no_grad` context;
  guard against the tape/`_INSTRUMENTED` caches growing unbounded. Crucial:
  medium. Feasibility: medium.
- **In-place / scatter updates.** `__setitem__`, masked assignment, scatter-add
  for embeddings / KV-caches / efficient optimizer steps. Prefer a functional
  `x.at[i].add(...)` form to sidestep aliasing hazards (true in-place + autograd
  needs version tracking). Crucial: medium. Feasibility: medium.
- **Mixed-precision training.** Loss scaling, autocast. Crucial: medium-high for
  scale. Feasibility: medium.

### Phase 3 — Advanced autodiff transforms

- **Higher-order gradients.** Make `backward` itself differentiable
  (`create_graph`) for Hessians, meta-learning, gradient penalties. Crucial:
  research-high. Feasibility: medium-hard.
- **Forward-mode / `jvp`.** Jacobians, some second-order methods. Crucial:
  medium. Feasibility: medium.
- **`vmap` (auto-batching) / per-sample gradients.** Big efficiency + many
  methods. Crucial: high for research. Feasibility: **low** — the eager tape
  makes auto-batching genuinely hard; this is where a trace-and-transform model
  (cf. JAX) earns its keep, and may force a design rethink.

### Phase 4 — Scale (the hard ceiling)

- **GPU / accelerator backend.** The execution model is CPU-bound and allocates a
  Python `Var` per op (~2–4 orders of magnitude too slow/heavy for real
  training). A CuPy backend (its NumPy-mirroring API means much "just works")
  gets GPU at moderate effort. Crucial: **highest for scale.** Feasibility:
  medium for CuPy; the per-op Python overhead is *not* fixed by this.
- **Graph capture + compilation / fusion.** Eliminate per-op Python overhead by
  tracing to a graph IR and compiling/fusing (XLA-style). This is a fundamentally
  different execution model from the eager tape and likely a separate engine.
  Crucial: highest for throughput. Feasibility: **low** (research-grade).
- **Distributed / multi-device.** Data / model / pipeline parallelism,
  collectives. Crucial: only for truly large scale. Feasibility: **very low**;
  pointless before the above.

---

## Non-goals (for now)

- A PyTorch-compatible API surface. Legibility over familiarity.
- Beating PyTorch/JAX on speed. The point is a system you can fully read.
- Being a hard dependency of pyccolo or pipescript — the relationship is reversed:
  pycograd may *optionally* use pyccolo for transparent mode.

## Open questions

- Eager tape vs. trace-and-compile: Phase 4 (and `vmap`) may not be reachable
  from the eager design without a second execution mode. Decide early whether
  pycograd is "always eager, readable" or grows a compiled path.
- How much of the pyccolo transparent-interception story to keep front-and-center
  vs. relegate to an optional extra once `Tensor` works natively with NumPy.
- Backend abstraction: design the array seam now (Phase 0) so CuPy/other backends
  drop in later without touching the VJP rules.

# pycograd

A small, readable reverse-mode autograd library built on numpy and pyccolo.

## Typing: avoid bare `object` and `Any`

Treat a bare `object`/`Any` annotation as a smell, not a precedent. When you
touch code annotated this way — or add new code — reach for a more informative
type. `pycograd/_typing.py` is the single home for the shared aliases below; use
them rather than re-spelling a union or falling back to `object`.

Order of preference:

1. **A concrete type or an existing alias from `pycograd/_typing.py`:**
   - `Operand` — a `Var`, a plain number/array, or a `Weight` proxy (what the
     public ops/primitives accept).
   - `Boxed` — a value flowing through the trace-level stack: a `Tracer`
     (BatchTracer/JVPTracer/ShapedArray), a `Var`, a raw scalar/array, or `None`.
     This is the precise type for `pure`/`lift`/`process_primitive`, `bind`'s
     result, and the per-primitive vmap/jvp rule operands and returns.
   - `Prim` — a primitive / numpy-or-math callable the trace stack dispatches
     through `bind`. Use it for `Callable[..., object]` that holds a function to
     call, and for the registry maps (`dict[Prim, Prim]` / `dict[Prim, Rule]`).
   - `Rule` — a per-primitive vmap/jvp/abstract rule (`Callable[..., Boxed]`).
   - `BindArg` — a raw operand at the `bind` dispatch boundary (a value, an index
     key, or a sequence of operands); deliberately broad but named.
   - `BackendArray` — a backend-native array / foreign framework tensor; the
     duck-typed bridge value in `pycograd/backends/*` and `compile.py`/`export.py`.
   - `Tensor` (`Var | ndarray`), `ArrayLike` / `Array` / `Scalar`, `Axis`
     (`int | tuple[int, ...] | None`), `Index` (a numpy `__getitem__` key),
     `Shape` (a reshape spec), `DTypeLike`, `Hashable` (a dict-key / tie / prov).
   - A `Tracer` / `Var` / `Trace` subclass, or `ShapedArray`/`ShapeDtypeStruct`,
     when the value is genuinely that object. `shapes.py` adds a local
     `AbstractVal` for the broader set its shape rules consume.
2. **A `TypeVar`, `Protocol`, or `Union`** when the value is generic or
   duck-typed but still has a knowable shape.
3. **`Any`, only when the precise type is genuinely intractable.** Prefer a
   *named* alias that documents intent (`Index`, `Shape`, `BackendArray`,
   `BindArg`) over a bare `Any`, with a comment on why it's unavoidable.

Bare `object` is almost never the right answer: it accepts everything but lets
you call nothing, so it neither documents intent nor catches mistakes. A few
legitimate exceptions remain and should stay `object`: runtime type-test
predicates (`def _is_array(x: object) -> bool`), `*args`/`**kwargs` that forward
arbitrary values verbatim, opaque identity tokens (`Param.origin`), and the
numpy-dispatch protocol slots (`__array_ufunc__`'s `*inputs`). Say so in a
comment rather than leaving a silent `object`.

The per-primitive VJP rule bodies in `ops.py` are typed
`(primals: tuple[Var, ...], operands: tuple[Boxed, ...], params: dict[str, Any],
g: Boxed) -> list[Boxed]` — match that shape when adding a rule.

The compile backends (`pycograd/backends`, `compile.py`) bridge our operands to
an optional framework's duck-typed API; `setup.cfg` keeps `follow_imports = skip`
for `torch`/`jax`/`tf` so the glue stays duck-typed (do not remove it — letting
mypy follow the stubs surfaces errors throughout the bridges). Use `BackendArray`
for tensor values, `ModuleType` for a framework-module parameter, and a
`if TYPE_CHECKING: import torch` plus a `"torch.dtype"`-style string annotation
for a framework object — never a bare `object`/`Any`.

## Conventions

- New aliases go in `pycograd/_typing.py` (note its header on why the runtime-
  referenced aliases keep `Union`/`Optional` rather than PEP 604 `|`).
- `mypy` and `ruff` are configured; run `make check` (blackcheck + lint +
  typecheck) before declaring done.

## New ops & integrations: exercise every transform surface

A new op, primitive, or third-party bridge (e.g. the einops backend in
`pycograd/einops_backend.py`) is not "done" when `grad` works — it must compose
with *all* the trace/graph surfaces, because each is a separate interpreter:

- **`grad` / `value_and_grad`** — eager reverse mode (the base tape).
- **`vmap`** (batching) and **`jvp`** (forward mode) — these run with a transform
  level live, so the value is a `BatchTracer`/`JVPTracer`, *not* a `Var`. Code that
  calls eager `d_*` ops directly (which build a `Var`) breaks here; route through
  `bind` when `num_transform_levels() > 0`, mirroring `tracer.resolve_call`.
- **Graph mode** — `capture(f, *args)` → `Graph`, then `eval_graph` and
  `value_and_grad(capture(...))` (graph-grad lowering). Confirm the op lowers to
  primitives the graph understands.
- **Shape inference / aval** — `eval_shape(f, *args)` runs an abstract `ShapedArray`
  trace (a `Tracer` subclass). Any `isinstance` type-gate must admit `Tracer`, not
  just `Var`.
- **Cost modeling** — `cost_report(graph)` / `graph.cost()` over a captured graph;
  the op's lowered primitives should show up as `NodeCost` entries.

Add regression tests for each surface (see `test/test_einops.py`: grad-vs-FD, jvp,
vmap, `vmap(grad)`, capture+`eval_graph`, `value_and_grad(capture)`, `eval_shape`,
and `cost_report`). If a pass skipped any of these surfaces, treat it as a gap to
close, not an optional extra.

## Notebooks / demos: use pipescript syntax

Write the examples in the `notebooks/` demos using the pipescript ambient DSL
(the `|>` pipe surface), not plain `def f(x): ...`. Match the existing demos:
a `params{ w = ...; b = frozen[...] } as weights:` block for parameters, and
`|>` pipelines for the forward — e.g.
`forward = $ |> $ @ w + b |> np.maximum(0.0, $) |> np.sum($)`.

- `$` is the piped value (the hole) for that stage. **Repeated anonymous `$`
  in one stage makes *separate* positional holes** (`$ * $` is a 2-arg lambda).
  To reuse the *same* piped value within a stage, use a **named** placeholder:
  `$v * $v` (square), `np.tanh($v) * np.tanh($v)` (same input twice).
- The graph/transform tools compose with pipes: `capture(forward, x)`,
  `grad_graph(...)`, `vjp_graph(forward, x)`, and `weights.grad(objective)` all
  work on a `|>` pipeline (validate new notebooks by executing them with
  `jupyter nbconvert --execute`). `%load_ext pycograd` enables the DSL.
- **Point-free binops over a bare function.** Any arithmetic (`+ - * / **`) or
  comparison (`< <= > >= == !=`) binop where an operand is a *bare function*
  becomes a deferred one-arg stage: `np.tanh ** 2` is `v -> np.tanh(v) ** 2`,
  `np.tanh > np.exp` is `v -> np.tanh(v) > np.exp(v)` (a mask), with the other
  operand a function or a held-constant value, in either order. This only works
  under instrumentation (inside `grad`/`vmap`/`capture`/compile), not at module
  level. Two carve-outs: `.**` is the distinct *compose* operator (`f .** g` =>
  `f ∘ g`, `f .** n` => n-fold), not a power; and point-free `@` with a *function*
  right operand (`f @ g`) raises — matmul a function's output against a value
  (`f @ w`) instead. (`handle_before_binop` / `handle_before_compare` in
  `tracer.py`.) A comparison stage in a pipe needs parens — `|>` lowers to `|`,
  which binds tighter than `>`: `$ |> (np.tanh > 0.0) |> $.astype(float)`.

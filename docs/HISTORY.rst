History
=======

0.0.3 (2026-06-27)
------------------
This release is driven largely by porting the HIPS ``autograd`` test suite and
closing the parity gaps it surfaced: a much broader numpy op library, complex
numbers, faithful dtypes, and wider graph-capture coverage.

Autograd parity and the ``grad`` API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Ported the HIPS ``autograd`` test suite (``test/autograd_suite``) as a
  pass-or-skip parity harness, with a ``REPORT.md`` gap map of what does and
  doesn't yet differentiate; skips reclassified accurately as gaps were closed.
* ``grad`` gained ``argnum`` / keyword-argument support, plus ``jacobian`` /
  ``hessian`` / ``make_vjp``; ``egrad`` alias for ``elementwise_grad``.

Op library expansion (numpy coverage)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* VJP rules for ~15 unary numpy ufuncs, numpy function-form arithmetic, ``d_mod``
  and ``d_prod``.
* Linear algebra: general ``np.dot`` / ``np.inner`` / ``np.tensordot`` lowered via
  einsum, ``np.outer``, ``np.cross`` and ``np.kron`` (bilinear compositions), and
  a general ``np.matmul`` VJP across all rank regimes.
* einsum: numpy's interleaved (explicit-axes) form, and a repeated label within
  one operand (diagonal).
* Shape / reordering: ``np.roll``, reshape-lowered ``ravel`` / ``squeeze`` /
  ``atleast_1d`` / ``2d`` / ``3d``, axis-reorder (``moveaxis`` / ``swapaxes`` /
  ``rollaxis``), triangular (``tril`` / ``triu``), ``np.flipud`` / ``fliplr`` /
  ``rot90``, ``np.trace``, ``Var.flatten()``.
* Construction / joining: the stack family (1-D single-array, ``dtype`` kwarg,
  ``np.row_stack``), ``np.append`` (including a python-list operand),
  positional-axis ``np.concatenate``, ``np.r_`` / ``np.c_`` index-expression
  construction, ``np.linspace``, and ``np.array`` over differentiable leaves
  (array-of-boxes).
* Splitting / segments: the split family (``split`` / ``array_split`` / ``vsplit``
  / ``hsplit`` / ``dsplit``), ``np.pad``, ``np.repeat``, ``np.tile`` (segment /
  scatter adjoints).
* Reductions / misc: ``logaddexp`` / ``logaddexp2`` / ``fmax`` / ``fmin``,
  ``np.diff``, ``np.diag`` / ``np.diagonal``, ``np.sort`` / ``np.partition``,
  ``np.select``, ``np.gradient``, ``cumsum(axis=None)``, ``np.nan_to_num`` /
  ``np.real_if_close``, and a degenerate-safe ``np.std`` gradient at zero
  variance.
* Python-builtin bridges: list-of-box reductions, ``len(box)``, and builtin
  ``sum`` over boxes; ``Var.dtype``.

Complex-number autodiff
~~~~~~~~~~~~~~~~~~~~~~~~~
* Complex (``complex64`` / ``complex128``) autodiff across all modes, with the
  real / non-holomorphic convention and a ``holomorphic_grad`` entrypoint;
  ``conj`` / ``real`` / ``imag`` / ``angle`` ops.

dtype fidelity
~~~~~~~~~~~~~~~
* Consistent dtype support across modes and a graph-differentiable ``astype``.
* The working dtype is a *creation default*, not a propagation cast: existing
  float / bf16 arrays keep their dtype through the tape, and capture stays
  faithful (no spurious ``float64``); ``constant_fold`` pins folded constants to
  the recorded dtype. The ``params{ ... }`` block honors the ambient dtype
  context.

Graph capture and graph-grad coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Ambient weights flow through ``capture`` (recorded as weight leaves); graph
  gradients return as a ``ParamDict`` / per-arg pytrees, and numpy
  ``weights.grad(jit=True)`` is supported.
* Composition ops are lowered at capture so they're graph-differentiable, with
  graph-grad coverage extended to ``roll`` / ``repeat`` / ``tile`` / ``pad`` /
  ``select``, interleaved einsum, and ``cumsum(axis=None)``; a constant index /
  shape tuple is stored whole rather than element-wise.

NumPy 2.0 and tooling
~~~~~~~~~~~~~~~~~~~~~~~
* NumPy 2.0 compatibility (the ``row_stack`` removal), while still intercepting
  ``np.row_stack`` where it exists.
* The cost model classifies every capture-surviving primitive, with an
  exhaustiveness guard.
* ``Makefile`` prefers ``.venv`` tools so ``make`` works unactivated; editable
  install in compatibility mode; bumped pyccolo / pipescript and other deps.

0.0.1 (2026-06-24)
------------------
Initial public release of pycograd: a small, readable reverse-mode
automatic-differentiation library built on numpy and pyccolo. Write *ordinary*
numeric Python — ``np.exp``, ``np.dot``, ``np.sum``, the ``@`` operator — and get
correct gradients with no special "autodiff namespace."

Core engine
~~~~~~~~~~~
* Initial extraction of the reverse-mode autodiff engine from the pyccolo
  ``autodiff`` example into a standalone package.
* ``Var`` reverse-mode tape node wrapping a numpy array, with broadcasting-aware
  VJP rules for the common elementwise, reduction, linear-algebra, and shape ops;
  ``Var.backward()`` walks the graph in reverse to accumulate gradients.
* ``value_and_grad`` / ``grad`` / ``gradient_descent``, pytree-structured
  gradients, and a clear error when ``grad``'s function returns a non-scalar.
* Transparent numpy/``math`` call interception via pyccolo's ``before_call``,
  swapping e.g. ``np.exp`` for a differentiable ``d_exp`` so idiomatic numpy
  "just differentiates"; the same seam differentiates through your own helper
  functions by instrumenting them on demand. (``Var`` sets
  ``__array_ufunc__ = None`` so an un-intercepted ufunc fails loudly.)

Parameters and the ambient-weights DSL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``Param`` / ``ParamDict`` parameter system with frozen / tied / ambient
  ``with weights:`` proxies.
* Ambient-weights DSL (``%load_ext pycograd``): a ``params{ ... } as weights:``
  block builds a parameter pytree and injects weights as ambient names, and
  ``|>`` pipelines differentiate when a model runs through them — write the
  forward *once*, by bare name, and ``weights.grad`` differentiates it.
* Pipescript surface: ``|>`` pipes with ``$`` holes, named ``$v`` placeholders to
  reuse a piped value, point-free function binops/comparisons, and the ``.**``
  compose operator.
* First-class minibatch SGD for the DSL; ``train`` / ``accuracy`` promoted to top
  level; autocompletable ``params{}`` blocks.

Composable transforms (trace-level dispatch stack)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Vectorized ``vmap`` (auto-batching), including batched gather (``x[idx]``),
  with a trace-level dispatch stack so transforms compose: nested ``vmap``,
  ``vmap(grad(f))`` for per-sample gradients (gradient clipping / DP-SGD),
  ``grad(vmap(f))``, and per-sample shared-param grads.
* Forward-mode AD (``jvp``) as a new trace level.
* Higher-order reverse-mode AD: differentiable backward and reverse-over-reverse
  ``grad(grad)``; vmap-composed higher-order AD for per-sample Hessians /
  batched HVPs.
* ``vmap`` composes with the ambient-weights DSL.
* Clear error for an un-ruled op encountered under a transform.

Shape inference
~~~~~~~~~~~~~~~
* ``eval_shape`` / ``summary`` / ``ShapeDtypeStruct`` run a net over abstract
  ``(shape, dtype)`` values with no data or allocation, reframed as a trace level
  (``AbstractTrace``).
* Symbolic / polymorphic dimensions (symbolic input dims + an equality store).
* ``ShapeError`` that names the op and operand shapes; a shape that genuinely
  depends on data values is reported as such rather than silently mis-sized.

Op library and layers
~~~~~~~~~~~~~~~~~~~~~~
* First-class functional op library: the softmax family (``softmax`` /
  ``logsumexp`` / ``cross_entropy``), ``einsum`` (with ellipsis + numpy
  broadcasting parity), convolution, and pooling.
* Common NN layers promoted to first-class functional ops; RNN / GRU / LSTM
  recurrent cells as ambient-DSL building blocks; an RWKV example.
* Splittable PRNG keys, ``batch_norm`` + buffers.
* Convolutions: streaming (incremental) 1-D and single-axis 2-D convs (state
  threaded through ``ParamDict`` buffers), dilated transposed convs, and grouped
  / depthwise convs.

Devices / backends
~~~~~~~~~~~~~~~~~~
* Pluggable array backend via a ``device(...)`` block (CuPy support), so the same
  net trains on a GPU with gradients and optimizer state on-device.
* Custom-dtype seam (float32 / float16 / bfloat16).
* Per-leaf device placement (``on_cpu[...]`` / ``on_device(...)``): part-CPU /
  part-GPU in one autograd pass.

Compile / export
~~~~~~~~~~~~~~~~
* One-call compile of the same forward to PyTorch / JAX / TensorFlow:
  ``weights.grad(objective, backend=..., jit=True)`` and ``compile_to``, matching
  pycograd's numpy tape to floating-point tolerance.
* An MPS (Apple Metal) compile backend.
* Conv lowering to native backend convolutions on the compile path.
* ``weights.to_torch_module`` / ``export_torchscript`` / ``export_onnx`` package
  a trained net for shipping with no pycograd dependency.

Graph capture, optimization, and rematerialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``capture`` records a ``|>`` pipeline into a flat SSA graph; a captured
  ``Graph`` is callable for inference.
* ``optimize`` passes: DCE / CSE / constant-folding, shape-aware algebraic
  simplification, gated-activation fusion, stable softmax/logsumexp fusion, and
  matmul-chain reordering.
* ``grad_graph`` (autodiff on the capture IR) and ``vjp_graph`` =
  transpose ∘ linearize; cross-pass CSE across the forward/backward boundary, and
  backward fusion via the optimized forward graph.
* Graph visualization: ``Graph.pretty()`` / ``to_dot()`` / ``render()`` with
  Jupyter auto-render.
* Gradient checkpointing (``checkpoint``): activations dropped on the forward and
  recomputed in backward; composes with ``grad`` / ``value_and_grad`` /
  ``weights.grad`` / ``vmap``.
* Static cost model plus a rematerialization/spill planner (``plan_remat``) over
  the capture IR; ``eval_scheduled`` runs the plan to the identical answer.

Demos and tooling
~~~~~~~~~~~~~~~~~
* Worked demos (logistic regression, MLP, LayerNorm/Dropout, single-head
  Transformer block, GRU/LSTM), each gradient-checked against finite differences;
  ``python -m pycograd.examples``.
* Notebook tour of the DSL, ``vmap``, RNN/RWKV, compile/parity (torch / jax / tf /
  MPS + TorchScript/ONNX export), device placement, graph viz, and remat.
* ``mypy`` / ``ruff`` configured; informative shared type aliases throughout.

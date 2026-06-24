History
=======

0.0.1
-----
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

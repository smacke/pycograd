History
=======

0.0.1 (unreleased)
------------------
* Initial extraction of the reverse-mode autodiff engine from the pyccolo
  ``autodiff`` example into a standalone package;
* ``Var`` tape node with broadcasting-aware VJP rules for the common elementwise,
  reduction, linear-algebra, and shape ops;
* ``value_and_grad`` / ``grad`` / ``gradient_descent``, pytree-structured
  gradients, and a ``Param`` / ``ParamDict`` parameter system (frozen / tied /
  ambient ``with weights:`` proxies);
* Transparent numpy/math call interception via pyccolo's ``before_call``;
* Worked demos (logistic regression, MLP, LayerNorm+Dropout, Transformer block),
  each gradient-checked against finite differences.

# -*- coding: utf-8 -*-
"""Worked autodiff demos -- logistic regression, MLPs, and a Transformer block.

These models are the package's first integration tests and the "don't regress"
bar (see the test suite, which imports them). Run the training demos with::

    python -m pycograd.examples

The training-loop helpers the demos use are top-level exports: ``from pycograd import
train, fit, accuracy`` -- ``train`` for a full-batch loop, ``fit`` for minibatch
(stochastic) gradient descent over a dataset.
"""

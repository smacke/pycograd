# -*- coding: utf-8 -*-
"""Ported placeholder for autograd tests/test_complex.py (MIT).

The op-level complex coverage now lives in test_systematic.py (test_real/imag/conj/
conjugate/angle, validated by the finite-difference adjoint check with the complex-aware
VSpace in _pytree.py) and in the pycograd-native test/test_complex.py. This module's
original autograd bodies poke autograd-internal complex machinery (VSpace/box internals)
with no pycograd analog, so it stays a documented placeholder rather than a real-only skip.
"""
import pytest

pytest.skip(
    "autograd-internal: test_complex.py exercises autograd's complex VSpace/box internals; "
    "pycograd's complex op coverage lives in test_systematic.py + test/test_complex.py",
    allow_module_level=True,
)

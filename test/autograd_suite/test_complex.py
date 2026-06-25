# -*- coding: utf-8 -*-
"""Ported placeholder for autograd tests/test_complex.py (MIT).

Skipped wholesale: complex numbers (np.real/imag/conj/angle, complex dtypes) — pycograd is real-only. See REPORT.md (missing subsystems) for the bridging plan.
"""
import pytest

pytest.skip(
    "pycograd-gap: complex numbers (np.real/imag/conj/angle, complex dtypes) — pycograd is real-only",
    allow_module_level=True,
)

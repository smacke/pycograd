# -*- coding: utf-8 -*-
"""Ported placeholder for autograd tests/test_scipy.py (MIT).

Skipped wholesale: scipy.special/stats/linalg/signal/integrate VJP rules — pycograd has no scipy backend. See REPORT.md (missing subsystems) for the bridging plan.
"""
import pytest

pytest.skip(
    "pycograd-gap: scipy.special/stats/linalg/signal/integrate VJP rules — pycograd has no scipy backend",
    allow_module_level=True,
)

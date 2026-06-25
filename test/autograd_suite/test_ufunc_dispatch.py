# -*- coding: utf-8 -*-
"""Ported placeholder for autograd tests/test_ufunc_dispatch.py (MIT).

Skipped wholesale: xarray/foreign-container ufunc dispatch — pycograd disables __array_ufunc__ on Var by design. See REPORT.md (missing subsystems) for the bridging plan.
"""
import pytest

pytest.skip(
    "pycograd-gap: xarray/foreign-container ufunc dispatch — pycograd disables __array_ufunc__ on Var by design",
    allow_module_level=True,
)

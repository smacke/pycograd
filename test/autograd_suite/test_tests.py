# -*- coding: utf-8 -*-
"""Ported placeholder for autograd ``tests/test_tests.py`` (MIT).

Skipped wholesale: autograd-internal. These tests assert that ``check_grads`` *detects* a
deliberately-wrong gradient, injected via autograd's ``@primitive`` + ``defvjp`` custom-VJP
API. pycograd has no equivalent mechanism to register an incorrect VJP for a primitive, so
the failure cannot be constructed. See REPORT.md (semantic divergences).
"""
import pytest

pytest.skip(
    "autograd-internal: needs @primitive/defvjp to inject a wrong VJP for check_grads to catch",
    allow_module_level=True,
)

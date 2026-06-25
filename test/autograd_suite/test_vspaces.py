# -*- coding: utf-8 -*-
"""Ported placeholder for autograd ``tests/test_vspaces.py`` (MIT).

Skipped wholesale: autograd-internal. It exercises autograd's ``VSpace`` abstraction and
its ``standard_basis`` axioms, which pycograd has no public analog for (pycograd works
directly on pytrees of arrays; the suite's ``_pytree.VSpace`` is a minimal test-only shim
without a standard basis). See REPORT.md (semantic divergences).
"""
import pytest

pytest.skip(
    "autograd-internal: pycograd has no public VSpace / standard_basis abstraction",
    allow_module_level=True,
)

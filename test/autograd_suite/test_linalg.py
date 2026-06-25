# -*- coding: utf-8 -*-
"""Ported placeholder for autograd tests/test_linalg.py (MIT).

Skipped wholesale: np.linalg VJP rules (inv/pinv/solve/det/slogdet/eigh/svd/qr/cholesky) — pycograd has no linalg gradients. See REPORT.md (missing subsystems) for the bridging plan.
"""
import pytest

pytest.skip(
    "pycograd-gap: np.linalg VJP rules (inv/pinv/solve/det/slogdet/eigh/svd/qr/cholesky) — pycograd has no linalg gradients",
    allow_module_level=True,
)

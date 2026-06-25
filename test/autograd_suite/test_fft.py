# -*- coding: utf-8 -*-
"""Ported placeholder for autograd tests/test_fft.py (MIT).

Skipped wholesale: np.fft transforms (fft/ifft/rfft/irfft) — pycograd has no FFT VJP rules. See REPORT.md (missing subsystems) for the bridging plan.
"""
import pytest

pytest.skip(
    "pycograd-gap: np.fft transforms (fft/ifft/rfft/irfft) — pycograd has no FFT VJP rules",
    allow_module_level=True,
)

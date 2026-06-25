# -*- coding: utf-8 -*-
"""Shared fixtures for the ported autograd suite: a per-test deterministic seed, applied to
both numpy's global RNG (used by the ported tests' ``npr``) and the checker's RNG."""
import os

import numpy as np
import pytest

from ._skips import SKIPS
from ._test_util import _reseed


@pytest.fixture(autouse=True)
def random_seed():
    np.random.seed(42)
    _reseed(42)


def pytest_collection_modifyitems(config, items):
    """Apply the centralized skip registry to the byte-faithful op-coverage ports
    (``test_systematic.py`` / ``test_numpy.py``), so those files stay verbatim. Set the
    env var ``PYCOGRAD_RUN_SKIPS=1`` to *not* apply them (e.g. to re-triage after adding a
    rule and see what now passes)."""
    if os.environ.get("PYCOGRAD_RUN_SKIPS"):
        return
    for item in items:
        fname = os.path.basename(str(item.fspath))
        reason = SKIPS.get((fname, item.name))
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))

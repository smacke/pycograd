# -*- coding: utf-8 -*-
"""Ported from autograd ``tests/test_logic.py`` (MIT)."""
import numpy as np
import pytest

from ._test_util import check_grads


@pytest.mark.skip(
    reason="pycograd-gap: np.allclose has no rule, so it cannot appear inside a traced "
    "(differentiated) function -- the assert breaks tracing"
)
def test_assert():
    # An assertion inside the differentiated function must not disturb the gradient.
    def fun(x):
        assert np.allclose(x, (x * 3.0) / 3.0)
        return np.sum(x)

    check_grads(fun)(np.array([1.0, 2.0, 3.0]))


@pytest.mark.skip(
    reason="autograd-internal: relies on autograd's non-differentiability TypeError for "
    "np.allclose output; pycograd has different non-diff behavior"
)
def test_nograd():
    from ._compat import grad

    fun = lambda x: np.allclose(x, (x * 3.0) / 3.0)
    with pytest.raises(TypeError):
        grad(fun)(np.array([1.0, 2.0, 3.0]))


@pytest.mark.skip(
    reason="autograd-internal: uses @primitive (no VJP defined) to assert a "
    "NotImplementedError; pycograd has no custom-primitive registration API"
)
def test_no_vjp_def():
    pass


@pytest.mark.skip(
    reason="autograd-internal: uses @primitive (no JVP defined) to assert a "
    "NotImplementedError; pycograd has no custom-primitive registration API"
)
def test_no_jvp_def():
    pass


@pytest.mark.skip(
    reason="pycograd-gap: complex numbers (np.iscomplex / complex inputs)"
)
def test_falseyness():
    pass


@pytest.mark.skip(
    reason="autograd-internal: pokes autograd's primitive_vjps registry + complex inputs"
)
def test_unimplemented_falseyness():
    pass

# -*- coding: utf-8 -*-
"""autograd-shaped differential-operator surface, backed by pycograd.

The ported test files import this in place of ``from autograd import ...`` so their bodies
read like autograd's. ``grad``/``value_and_grad``/``jacobian``/``hessian``/
``elementwise_grad``/``make_jvp``/``make_vjp`` are pycograd's now-autograd-shaped operators
(``argnum`` defaults to 0, an int returns the bare gradient). The composed operators
(``deriv``/``grad_and_aux``/``make_hvp``/``hessian_tensor_product``/
``tensor_jacobian_product``/``make_ggnvp``) mirror ``autograd/differential_operators.py``
but are expressed with pycograd's native forward-over-reverse HVP (``jvp(grad(.))``) and the
pytree-capable ``_vjp`` from :mod:`_test_util`, since pycograd's reverse pass detaches and so
does not support autograd's reverse-over-reverse composition.

Container/identity shims: pycograd already treats list/tuple/dict as pytrees, so the
``autograd.tuple``/``list``/``dict`` constructors are the plain builtins here, and
``autograd.isinstance``/``type`` are the builtins.
"""
from __future__ import annotations

import builtins

import numpy as np

import pycograd as pg
from pycograd import jvp as _pg_jvp

from ._test_util import _vjp


# --- direct pycograd operators (autograd argnum/return convention) -----------
# autograd defaults argnum=0 and returns the *bare* gradient; pycograd defaults
# argnum=None (a tuple over all args), so grad/value_and_grad pin argnum=0 here.
def grad(fun, argnum=0):
    return pg.grad(fun, argnum)


def value_and_grad(fun, argnum=0):
    return pg.value_and_grad(fun, argnum)


jacobian = pg.jacobian
hessian = pg.hessian
elementwise_grad = pg.elementwise_grad
make_jvp = pg.make_jvp


egrad = pg.egrad  # autograd nickname for elementwise_grad


def make_vjp(fun, argnum=0):
    """``make_vjp(fun)(x) -> (vjp_fn, ans)`` (autograd ordering), pytree-capable."""

    def maker(*args, **kwargs):
        vjp_fn, ans = _vjp(fun, args, kwargs, argnum)
        return vjp_fn, ans

    return maker


def deriv(fun, argnum=0):
    """Forward-mode scalar derivative ``df/dx`` (``make_jvp`` along the ones tangent)."""

    def d(*args, **kwargs):
        x = np.asarray(args[argnum], dtype=float)
        v = np.ones_like(x)
        _, tangent = make_jvp(fun, argnum)(*args, **kwargs)(v if v.ndim else float(v))
        return tangent

    return d


def holomorphic_grad(fun, argnum=0):  # pragma: no cover - pycograd is real-only
    raise NotImplementedError(
        "holomorphic_grad needs complex support, which pycograd does not have"
    )


def grad_and_aux(fun, argnum=0):
    """For ``fun`` returning ``(scalar, aux)``: ``(grad_of_scalar, aux)``."""

    def ga(*args, **kwargs):
        vjp_fn, ans = _vjp(fun, args, kwargs, argnum)
        value, aux = ans
        cot = (
            np.ones_like(np.asarray(value, dtype=float)),
            np.zeros_like(np.asarray(aux, dtype=float)),
        )
        return vjp_fn(cot), aux

    return ga


def _tangents(args, argnum, v):
    return tuple(
        v if i == argnum else np.zeros_like(np.asarray(a, dtype=float))
        for i, a in enumerate(args)
    )


def make_hvp(fun, argnum=0):
    """``make_hvp(fun)(x) -> (hvp_fn, grad)``; ``hvp_fn(v)`` is ``H v`` via forward-over-
    reverse (``jvp(grad(fun))``), pycograd's native second-order path."""

    def maker(*args, **kwargs):
        x = args[argnum]
        g0 = grad(fun, argnum)(*args, **kwargs)

        def hvp(v):
            return _pg_jvp(grad(fun, argnum), (x,), (v,))[1]

        return hvp, g0

    return maker


def hessian_tensor_product(fun, argnum=0):
    """``htp(*args, tensor)`` = ``tensordot(H, tensor)`` over ``ndim(tensor)`` axes."""

    def htp(*all_args, **kwargs):
        args, vector = all_args[:-1], all_args[-1]
        H = np.asarray(hessian(fun, argnum)(*args, **kwargs))
        return np.tensordot(H, np.asarray(vector), np.ndim(vector))

    return htp


hessian_vector_product = hessian_tensor_product


def tensor_jacobian_product(fun, argnum=0):
    """``tjp(*args, tensor)`` = ``tensordot(tensor, J)`` over ``ndim(tensor)`` axes."""

    def tjp(*all_args, **kwargs):
        args, vector = all_args[:-1], all_args[-1]
        J = np.asarray(jacobian(fun, argnum)(*args, **kwargs))
        return np.tensordot(np.asarray(vector), J, np.ndim(vector))

    return tjp


def make_ggnvp(f, g=lambda x: 0.5 * np.dot(x, x), f_argnum=0):
    """Generalized Gauss-Newton vector product ``Jᵀ (H_g) J v`` at a point."""

    def maker(*args, **kwargs):
        x = args[f_argnum]
        fx = f(*args, **kwargs)

        def ggnvp(v):
            jv = _pg_jvp(f, tuple(args), _tangents(args, f_argnum, v))[1]
            ghjv = _pg_jvp(grad(g), (fx,), (np.asarray(jv),))[1]
            vjp_fn, _ = _vjp(f, args, kwargs, f_argnum)
            return vjp_fn(np.asarray(ghjv))

        return ggnvp

    return maker


# --- container / identity shims (pycograd treats containers as pytrees) ------
ag_tuple = tuple
ag_list = list
ag_dict = dict
checkpoint = pg.checkpoint


def _unbox(x):
    """The underlying value of a tape ``Var`` / ``Tracer`` (else ``x`` itself) -- so the
    ``isinstance``/``type`` shims see the *array* a leaf was lifted from, matching autograd's
    box-transparent ``isinstance``."""
    from pycograd.tensor import Var
    from pycograd.trace import Tracer

    while isinstance(x, (Var, Tracer)):
        x = x.value
    return x


def ag_isinstance(x, type_):
    return builtins.isinstance(_unbox(x), type_)


def ag_type(x):
    return builtins.type(_unbox(x))

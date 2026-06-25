# -*- coding: utf-8 -*-
"""``check_grads`` / ``check_equivalent`` / ``combo_check`` -- ported from HIPS autograd's
``autograd/test_util.py`` (MIT-licensed), re-backed by pycograd.

The shape of the check is autograd's: a central-difference numerical JVP, the reverse-mode
VJP inner-product identity ``<u, J v> == <Jᵀ u, v>``, the forward-mode JVP comparison, and
the order>1 recursion into the gradient function. Two things differ for pycograd:

* The **numerical** derivative calls the function on plain perturbed arrays, so it uses a
  simple unary closure that holds the non-differentiated arguments fixed.
* The **analytic** VJP/JVP must *instrument* the real function (pycograd disables the numpy
  array protocol on ``Var``, so an op like ``np.sin`` is only intercepted inside a woven
  function). We therefore run the user's ``f`` through ``_make_runner`` and lift only the
  differentiated argument's leaves onto the tape, leaving the others (and ``**kwargs``) as
  plain constants -- the same selective-lifting ``pycograd.grad`` uses for ``argnum``.

``vspace`` is a small pytree real-vector-space view (:mod:`_pytree`); pycograd is real-only.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from pycograd import jvp as _pg_jvp
from pycograd import ops
from pycograd.tensor import Var, _lift, _value
from pycograd.trace import ReverseTrace, new_main
from pycograd.tracer import _INSTRUMENTED, _make_runner
from pycograd.transforms import _traceable
from pycograd.tree import tree_flatten, tree_unflatten

from ._pytree import _is_num, vspace

TOL = 1e-6
RTOL = 1e-6
EPS = 1e-6

_RNG = np.random.default_rng(0)


def _reseed(seed: int) -> None:
    global _RNG
    _RNG = np.random.default_rng(seed)


def scalar_close(a: float, b: float) -> bool:
    return abs(a - b) < TOL or abs(a - b) / (abs(a + b) + 1e-300) < RTOL


# ---------------------------------------------------------------------------
# Helpers over the (fun, args, kwargs, argnum) call shape.
# ---------------------------------------------------------------------------
def _idxs(argnum):
    return (argnum,) if isinstance(argnum, int) else tuple(argnum)


def _select(args, argnum):
    """The differentiated argument value: the single arg (int argnum) or a tuple."""
    if isinstance(argnum, int):
        return args[argnum]
    return tuple(args[i] for i in argnum)


def _unary(fun, args, kwargs, argnum):
    """A plain (untraced) one-argument view of ``fun`` -- for the numerical derivative,
    which only ever calls it on concrete arrays."""

    def unary_f(z):
        sub = list(args)
        if isinstance(argnum, int):
            sub[argnum] = z
        else:
            for i, zi in zip(argnum, z):
                sub[i] = zi
        return fun(*sub, **kwargs)

    return unary_f


def _runner_for(fun):
    r = _INSTRUMENTED.get(fun)
    if r is None:
        r = _make_runner(_traceable(fun))
        _INSTRUMENTED[fun] = r
    return r


def _zeros_like_tree(value):
    leaves, td = tree_flatten(value)
    return tree_unflatten(
        td, [np.zeros(np.asarray(l).shape) if _is_num(l) else l for l in leaves]
    )


# ---------------------------------------------------------------------------
# Analytic VJP / JVP, instrumenting the real ``fun`` with selective lifting.
# ---------------------------------------------------------------------------
def _vjp(fun, args, kwargs, argnum):
    """``(vjp_fn, ans)``: instrument ``fun``, lift only the ``argnum`` argument(s) onto the
    tape, run forward; ``vjp_fn(cot)`` pulls the output cotangent back to the selected
    argument's structure (a single pytree for int ``argnum``, a tuple for a sequence).
    """
    runner = _runner_for(fun)
    idxs = _idxs(argnum)
    call_args = list(args)
    in_info: dict[int, tuple] = {}
    for i in idxs:
        leaves, td = tree_flatten(args[i])
        vs = [Var(np.asarray(l, dtype=float)) if _is_num(l) else None for l in leaves]
        in_info[i] = (td, vs)
        call_args[i] = tree_unflatten(
            td, [v if v is not None else l for v, l in zip(vs, leaves)]
        )
    with new_main(ReverseTrace):
        out = runner(*call_args, **kwargs)
    out_leaves, out_def = tree_flatten(out)
    out_vars = [ol if isinstance(ol, Var) else _lift(ol) for ol in out_leaves]
    ans = tree_unflatten(out_def, [np.asarray(_value(ov)) for ov in out_vars])

    def vjp_fn(cot):
        cot_leaves, _ = tree_flatten(cot)
        scalar = None
        for ov, c in zip(out_vars, cot_leaves):
            term = ops.d_sum(ov * np.asarray(c, dtype=float))
            scalar = term if scalar is None else scalar + term
        scalar.backward(differentiable=False)
        grads = {
            i: tree_unflatten(
                td, [np.asarray(v.grad) if v is not None else None for v in vs]
            )
            for i, (td, vs) in in_info.items()
        }
        if isinstance(argnum, int):
            return grads[argnum]
        return tuple(grads[i] for i in argnum)

    return vjp_fn, ans


def _jvp(fun, args, kwargs, argnum, v):
    """Forward-mode tangent ``df(args) . v`` w.r.t. the ``argnum`` argument(s)."""
    idxs = _idxs(argnum)
    primitive = (
        _traceable(fun) is not fun
    )  # a bare ufunc/builtin (no instrumentable source)
    if kwargs or primitive:
        # Hold every non-``argnum`` argument (positional *and* keyword) fixed in a closure and
        # ``jvp`` only over the differentiated argument(s). This keeps a *structural*
        # positional argument -- ``np.pad(x, pad_width, mode)``'s ``pad_width``/``mode`` -- a
        # plain constant instead of letting ``jvp`` lift it to a tracer. For a primitive
        # ``fun`` the single woven call routes through the rule with those constants intact.
        def picked(*da):
            full = list(args)
            for i, d in zip(idxs, da):
                full[i] = d
            return fun(*full, **kwargs)

        diff = tuple(args[i] for i in idxs)
        tang = (v,) if isinstance(argnum, int) else tuple(v)
        _, tangent = _pg_jvp(picked, diff, tang)
        return tangent
    # A composite (source-bearing) ``fun`` with no kwargs: instrument it directly so its
    # internal ``np.*`` calls are woven; the held numeric args carry a zero tangent.
    parts = {argnum: v} if isinstance(argnum, int) else dict(zip(argnum, v))
    tangents = tuple(
        parts[i] if i in idxs else _zeros_like_tree(a) for i, a in enumerate(args)
    )
    _, tangent = _pg_jvp(fun, tuple(args), tangents)
    return tangent


# ---------------------------------------------------------------------------
# The checks (autograd's structure).
# ---------------------------------------------------------------------------
def make_numerical_jvp(unary_f, x):
    x_vs = vspace(x)
    y_vs = vspace(unary_f(x))

    def jvp(v):
        plus = unary_f(x_vs.add(x, x_vs.scalar_mul(v, EPS / 2)))
        minus = unary_f(x_vs.add(x, x_vs.scalar_mul(v, -EPS / 2)))
        return y_vs.scalar_mul(y_vs.add(plus, y_vs.scalar_mul(minus, -1.0)), 1.0 / EPS)

    return jvp


def check_equivalent(x, y):
    x_vs, y_vs = vspace(x), vspace(y)
    assert x_vs == y_vs, f"VSpace mismatch:\n  x: {x_vs}\n  y: {y_vs}"
    v = x_vs.randn(_RNG)
    assert scalar_close(
        x_vs.inner_prod(x, v), x_vs.inner_prod(y, v)
    ), f"Value mismatch:\n  x: {x}\n  y: {y}"


def _check_vjp(fun, args, kwargs, argnum):
    unary_f = _unary(fun, args, kwargs, argnum)
    x = _select(args, argnum)
    vjp, ans = _vjp(fun, args, kwargs, argnum)
    num_jvp = make_numerical_jvp(unary_f, x)
    x_vs, y_vs = vspace(x), vspace(ans)
    x_v, y_v = x_vs.randn(_RNG), y_vs.randn(_RNG)
    vjp_y = vjp(y_v)
    assert vspace(vjp_y) == x_vs, f"VJP vspace mismatch:\n  {vspace(vjp_y)}\n  {x_vs}"
    vjv_exact = x_vs.inner_prod(x_v, vjp_y)
    vjv_numeric = y_vs.inner_prod(y_v, num_jvp(x_v))
    assert scalar_close(vjv_numeric, vjv_exact), (
        f"Reverse (VJP) check of {getattr(fun, '__name__', fun)} failed:\n"
        f"  analytic: {vjv_exact}\n  numeric:  {vjv_numeric}"
    )


def _check_jvp(fun, args, kwargs, argnum):
    unary_f = _unary(fun, args, kwargs, argnum)
    x = _select(args, argnum)
    x_v = vspace(x).randn(_RNG)
    analytic = _jvp(fun, args, kwargs, argnum, x_v)
    numeric = make_numerical_jvp(unary_f, x)(x_v)
    check_equivalent(analytic, numeric)


def check_grads(f, argnum=0, modes=("fwd", "rev"), order=2):
    """Numerically check ``f``'s gradients (forward and/or reverse mode). Same call surface
    as autograd's ``check_grads(f, argnum=, modes=, order=)(*args, **kwargs)``.

    **First-order only.** autograd's ``check_grads`` recurses into the gradient function for
    ``order``>1; this port does not. autograd realizes that recursion with ``make_vjp`` /
    ``make_jvp`` of throwaway *closures* under nested traces -- a path pycograd's eager,
    instrument-from-source tape does not compose with (its reverse pass detaches, so a
    reverse-over-reverse numerical check would silently read zero). pycograd's genuine
    higher-order AD (``grad(grad)``, ``jvp(grad)``, ``jacfwd(grad)``) is exercised directly
    in ``test/test_highorder.py``. The ``order`` argument is accepted (so the ported test
    bodies are unchanged) but only the first-order checks run. This is called out in
    ``REPORT.md`` as the suite's main fidelity caveat.
    """
    modes = list(modes)
    assert all(m in ("fwd", "rev") for m in modes)
    del order  # accepted for call-compatibility; see the docstring

    def checker(*args, **kwargs):
        if "fwd" in modes:
            _check_jvp(f, args, kwargs, argnum)
        if "rev" in modes:
            _check_vjp(f, args, kwargs, argnum)

    return checker


def combo_check(fun, *args, **kwargs):
    """Run ``check_grads(fun, *args, **kwargs)`` over the cartesian product of the given
    positional-argument lists and keyword-argument value lists."""
    _check = lambda f: check_grads(f, *args, **kwargs)

    def _combo(*arg_lists, **kwarg_lists):
        kw_pairs = [[(k, val) for val in vs] for k, vs in kwarg_lists.items()]
        for cur_args in product(*arg_lists):
            for cur_kwargs in product(*kw_pairs):
                _check(fun)(*cur_args, **dict(cur_kwargs))

    return _combo

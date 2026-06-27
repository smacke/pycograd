# -*- coding: utf-8 -*-
"""vmap (auto-batching): the vectorized transform must match a plain Python loop.

``_loop_vmap`` (map ``f`` over the batch with a list comprehension, then stack) is the
oracle, exactly as the dummy-array path is the oracle for shape inference. Every
vectorized ``vmap(f)`` must equal it; gradients of vmapped functions are checked
against finite differences.
"""
import numpy as np

from pycograd import grad, params, value_and_grad, vmap
from pycograd.examples import models as M
from pycograd.transforms import per_example_grad


def _rng(seed=0):
    return np.random.default_rng(seed)


def _loop_vmap(f, in_axes=0):
    def wrapped(*args):
        axes = in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(args)
        batch = next(a.shape[ax] for a, ax in zip(args, axes) if ax is not None)
        outs = []
        for i in range(batch):
            sliced = [
                np.take(a, i, axis=ax) if ax is not None else a
                for a, ax in zip(args, axes)
            ]
            outs.append(np.asarray(f(*sliced)))
        return np.stack(outs)

    return wrapped


def _check(f, args, in_axes=0):
    got = np.asarray(vmap(f, in_axes=in_axes)(*args))
    ref = _loop_vmap(f, in_axes=in_axes)(*args)
    assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
    assert np.allclose(got, ref), f"value mismatch\n{got}\n{ref}"


def _check_nested(f, args, in_axes=0):
    """Vectorized ``vmap(vmap(f))`` must equal a double Python loop (the oracle for the
    inner ``vmap`` composed with the oracle for the outer ``vmap``)."""
    got = np.asarray(vmap(vmap(f, in_axes=in_axes), in_axes=in_axes)(*args))
    ref = _loop_vmap(_loop_vmap(f, in_axes=in_axes), in_axes=in_axes)(*args)
    assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
    assert np.allclose(got, ref), f"value mismatch\n{got}\n{ref}"


# ---------------------------------------------------------------------------
# forward conformance vs the loop oracle
# ---------------------------------------------------------------------------
def elementwise(x):
    return np.exp(x) * 2.0 - 1.0


def reduce_all(x):
    return np.sum(x)


def reduce_axis(x):
    return np.sum(x, axis=0)


def matvec(x):
    return x @ np.ones(4)


def per_sample_dot(x):
    return x @ x


def reshape_fn(x):
    return x.reshape(2, 2)


def transpose_fn(x):
    return x.reshape(2, 2).T


def getitem_fn(x):
    return x[1:3]


def mlp(x, w1, b1, w2, b2):
    return np.tanh(x @ w1 + b1) @ w2 + b2


def test_vmap_elementwise():
    _check(elementwise, (_rng().standard_normal((6, 4)),))


def test_vmap_reduce_all():
    _check(reduce_all, (_rng().standard_normal((6, 4)),))


def test_vmap_reduce_axis():
    _check(reduce_axis, (_rng().standard_normal((6, 3, 5)),))


def test_vmap_matvec():
    _check(matvec, (_rng().standard_normal((6, 4)),))


def test_vmap_per_sample_dot():
    _check(per_sample_dot, (_rng().standard_normal((6, 4)),))


def test_vmap_reshape():
    _check(reshape_fn, (_rng().standard_normal((6, 4)),))


def test_vmap_transpose():
    _check(transpose_fn, (_rng().standard_normal((6, 4)),))


def test_vmap_getitem():
    _check(getitem_fn, (_rng().standard_normal((6, 5)),))


def matmul_shared(x, w):
    return x @ w


def test_vmap_matmul_shared_weight():
    r = _rng()
    _check(
        matmul_shared,
        (r.standard_normal((6, 4)), r.standard_normal((4, 3))),
        in_axes=(0, None),
    )


def test_vmap_matmul_both_batched():
    r = _rng()
    _check(
        matmul_shared,
        (r.standard_normal((6, 2, 4)), r.standard_normal((6, 4, 3))),
        in_axes=(0, 0),
    )


def test_vmap_mlp_shared_params():
    r = _rng()
    X = r.standard_normal((6, 3))
    w1, b1 = r.standard_normal((3, 5)), r.standard_normal((5,))
    w2, b2 = r.standard_normal((5, 2)), r.standard_normal((2,))
    _check(mlp, (X, w1, b1, w2, b2), in_axes=(0, None, None, None, None))


def test_vmap_out_axes():
    f = elementwise
    X = _rng().standard_normal((6, 4))
    got = np.asarray(vmap(f, out_axes=1)(X))
    ref = np.moveaxis(_loop_vmap(f)(X), 0, 1)
    assert got.shape == ref.shape and np.allclose(got, ref)


# ---------------------------------------------------------------------------
# gradient composition
# ---------------------------------------------------------------------------
def _fd(f, x, h=1e-5):
    g = np.zeros_like(x)
    flat = x.reshape(-1)
    for i in range(flat.size):
        xp = flat.copy()
        xp[i] += h
        xm = flat.copy()
        xm[i] -= h
        g.reshape(-1)[i] = (f(xp.reshape(x.shape)) - f(xm.reshape(x.shape))) / (2 * h)
    return g


def batch_loss(x):
    # x: (B, d) -> scalar mean of per-sample squared norms
    return np.sum(vmap(lambda r: r @ r)(x)) / x.shape[0]


def test_grad_of_vmap():
    X = _rng().standard_normal((5, 4))
    (g,) = grad(batch_loss)(X)
    expected = _fd(lambda z: float(np.sum(np.sum(z * z, axis=1)) / z.shape[0]), X)
    assert np.allclose(g, expected, atol=1e-5)


def test_value_and_grad_of_vmap_unwraps():
    X = _rng().standard_normal((5, 4))
    val, (g,) = value_and_grad(batch_loss)(X)
    assert np.isscalar(val) or np.ndim(val) == 0
    assert g.shape == X.shape


def per_sample_sq(x):
    return x @ x  # scalar per example


def test_per_example_grad_matches_loop():
    X = _rng().standard_normal((5, 4))
    g = per_example_grad(per_sample_sq)(X)
    # d/dx (x . x) = 2x, per sample
    assert g.shape == X.shape
    assert np.allclose(g, 2 * X, atol=1e-6)


def pow_sq(x):
    return (
        x**2
    )  # a *constant* exponent -- must stay on the safe power path under vmap


def pow_sum(x):
    return np.sum(x**3)


def test_vmap_pow_constant_exponent_negative_base():
    # Regression: the generic elementwise batch rule lifted ``d_pow``'s exponent to a
    # tracer, flipping ``x**k`` onto ``exp(k*log x)`` -- nan for a negative base. The
    # dedicated ``_pow_rule`` keeps the exponent constant, so both forward and gradient
    # are finite and correct for negative bases.
    X = np.array([[-1.0, 2.0], [-3.0, 0.5], [1.5, -2.0]])
    _check(pow_sq, (X,))  # forward matches the loop oracle (no nan)
    g = per_example_grad(pow_sum)(X)
    assert np.all(np.isfinite(g))
    assert np.allclose(g, 3 * X**2, atol=1e-6)  # d/dx sum(x**3) = 3x**2, per sample


# ---------------------------------------------------------------------------
# batched (per-example) gather: x[idx] where each example indexes its own data
# ---------------------------------------------------------------------------
def gather(x, idx):
    return x[idx]


def test_vmap_batched_gather_1d():
    r = _rng()
    X = r.standard_normal((4, 6))
    IDX = r.integers(0, 6, size=(4, 3))
    _check(gather, (X, IDX), in_axes=(0, 0))


def test_vmap_batched_gather_rows():
    r = _rng()
    X = r.standard_normal((4, 5, 2))  # gather whole rows per example
    IDX = r.integers(0, 5, size=(4, 3))
    _check(gather, (X, IDX), in_axes=(0, 0))


# Module-level so the differentiated function closes over globals only (pyccolo
# re-instruments it from source, which doesn't preserve enclosing-function closures).
_GIDX = _rng(7).integers(0, 6, size=(4, 3))


def _gather_loss(x):
    return np.sum(vmap(gather, in_axes=(0, 0))(x, _GIDX))


def test_grad_through_batched_gather():
    X = _rng().standard_normal((4, 6))
    (g,) = grad(_gather_loss)(X)
    # d/dx of summing gathered entries = count of how many times each (row, col) is picked
    expected = np.zeros_like(X)
    for i in range(4):
        for j in _GIDX[i]:
            expected[i, j] += 1.0
    assert np.allclose(g, expected)


# ---------------------------------------------------------------------------
# shared/unbatched-table gather: table[idx] where the *index* is the batched thing
# and the table is shared across the batch (in_axes=(0, None)). Functions/constants
# are module-level so pyccolo re-instruments them from source (no closure capture).
# ---------------------------------------------------------------------------
_ETABLE = _rng(3).standard_normal((10, 3))  # shared embedding table
_TIDS = _rng(4).integers(0, 10, size=8)  # one index per example


def shared_gather(i, t):
    return t[i]


def global_gather(i):
    return _ETABLE[i]


def test_vmap_shared_table_gather():
    # table[batched_index] -> (B, *table.shape[1:]), matching the loop oracle.
    got = np.asarray(vmap(shared_gather, in_axes=(0, None))(_TIDS, _ETABLE))
    ref = np.stack([_ETABLE[_TIDS[k]] for k in range(len(_TIDS))])
    assert got.shape == ref.shape
    assert np.allclose(got, ref)


def test_vmap_shared_table_gather_global():
    # Same gather, but the shared table is a module global closed over by the mapped fn.
    got = np.asarray(vmap(global_gather)(_TIDS))
    ref = np.stack([_ETABLE[_TIDS[k]] for k in range(len(_TIDS))])
    assert got.shape == ref.shape
    assert np.allclose(got, ref)


def _shared_gather_loss(t):
    return np.sum(vmap(shared_gather, in_axes=(0, None))(_TIDS, t))


def test_grad_through_shared_table_gather():
    (g,) = grad(_shared_gather_loss)(_ETABLE)
    # d/dtable of summing the gathered rows = count of how often each row is selected
    # (scatter-add accumulates 1 per occurrence into that row).
    expected = np.zeros_like(_ETABLE)
    for k in range(len(_TIDS)):
        expected[_TIDS[k]] += 1.0
    assert g.shape == _ETABLE.shape
    assert np.allclose(g, expected)


# ---------------------------------------------------------------------------
# nested vmap(vmap(f)) vs a double Python loop
# (functions are module-level so re-instrumentation from source works -- pyccolo does
# not preserve enclosing-function closures, so they must close over module globals only)
# ---------------------------------------------------------------------------
def scale2(x):
    return x * 2.0


def nested_elementwise(x):
    return np.exp(x) * 2.0 - 1.0


def nested_reduce_last(x):
    return np.sum(x, axis=-1)


def nested_reduce_all(x):
    return np.sum(x)


def nested_dot(x):
    return x @ x


def nested_matvec(x):
    return x @ np.ones(4)


def test_nested_vmap_elementwise_scalar():
    X = _rng().standard_normal((3, 5))
    _check_nested(scale2, (X,))


def test_nested_vmap_elementwise():
    X = _rng().standard_normal((3, 5, 4))
    _check_nested(nested_elementwise, (X,))


def test_nested_vmap_reduce_axis():
    X = _rng().standard_normal((3, 5, 4))
    _check_nested(nested_reduce_last, (X,))


def test_nested_vmap_reduce_all():
    X = _rng().standard_normal((3, 5, 4))
    _check_nested(nested_reduce_all, (X,))


def test_nested_vmap_dot():
    X = _rng().standard_normal((3, 5, 4))
    _check_nested(nested_dot, (X,))


def test_nested_vmap_matvec():
    X = _rng().standard_normal((3, 5, 4))
    _check_nested(nested_matvec, (X,))


def nested_gather(x, idx):
    return x[idx]


def test_nested_vmap_gather():
    r = _rng()
    X = r.standard_normal((2, 3, 6))
    IDX = r.integers(0, 6, size=(2, 3, 4))
    got = np.asarray(vmap(vmap(nested_gather, in_axes=(0, 0)), in_axes=(0, 0))(X, IDX))
    ref = _loop_vmap(_loop_vmap(nested_gather, in_axes=(0, 0)), in_axes=(0, 0))(X, IDX)
    assert got.shape == ref.shape
    assert np.allclose(got, ref)


def _nested_scaled_loss(x):
    # sum over a doubly-vmapped scale-by-2: loss = sum(2 * x_ij), so d loss / dx = 2.
    return np.sum(vmap(vmap(scale2))(x))


def test_nested_vmap_grad_composes():
    X = _rng().standard_normal((3, 5))
    (g,) = grad(_nested_scaled_loss)(X)
    assert np.allclose(g, 2 * np.ones_like(X))


def test_coverage_matches_intercept():
    from pycograd.batching import _BATCH
    from pycograd.ops import _INTERCEPT

    assert set(_BATCH) == set(_INTERCEPT)


# ---------------------------------------------------------------------------
# per-sample gradients of a SHARED parameter: vmap(grad(f), in_axes=(0, None))
# yields d f(x_i, w)/dw stacked to (B, *w.shape), not summed over the batch.
# Oracle: a Python loop of grad(f) over each example.
# ---------------------------------------------------------------------------
def _stack_loop_grad(f, X, *shared, in_axes):
    """stack([grad(f)(x_i, *shared) for i]) -- the per-sample-grad oracle."""
    axes = in_axes if isinstance(in_axes, tuple) else (in_axes,) * (1 + len(shared))
    batch = X.shape[axes[0]]
    per_arg = [[] for _ in range(1 + len(shared))]
    for i in range(batch):
        xi = np.take(X, i, axis=axes[0])
        gs = grad(f)(xi, *shared)
        for k, g in enumerate(gs):
            per_arg[k].append(np.asarray(g))
    return tuple(np.stack(col) for col in per_arg)


def mlp_scalar(x, w1, b1, w2):
    # one example -> scalar; w1/b1/w2 shared across the batch
    h = np.tanh(x @ w1 + b1)
    return h @ w2


def test_per_sample_grad_shared_weight_vs_loop():
    r = _rng()
    B = 6
    X = r.standard_normal((B, 3))
    w1, b1 = r.standard_normal((3, 5)), r.standard_normal((5,))
    w2 = r.standard_normal((5,))
    in_axes = (0, None, None, None)
    gx, gw1, gb1, gw2 = vmap(grad(mlp_scalar), in_axes=in_axes)(X, w1, b1, w2)
    ogx, ogw1, ogb1, ogw2 = _stack_loop_grad(mlp_scalar, X, w1, b1, w2, in_axes=in_axes)
    assert gx.shape == (B, 3) and gw1.shape == (B, 3, 5)
    assert gb1.shape == (B, 5) and gw2.shape == (B, 5)
    assert np.allclose(gx, ogx)
    assert np.allclose(gw1, ogw1)
    assert np.allclose(gb1, ogb1)
    assert np.allclose(gw2, ogw2)


def linear_scalar(x, w):
    return x @ w  # x:(d,), w:(d,) -> scalar; shared w


def test_per_sample_grad_shared_weight_fd():
    r = _rng(3)
    B = 5
    X = r.standard_normal((B, 4))
    w = r.standard_normal((4,))
    (_gx, gw) = vmap(grad(linear_scalar), in_axes=(0, None))(X, w)
    # per-sample finite differences of f(x_i, w) w.r.t. w
    fd = np.stack([_fd(lambda ww, xi=X[i]: float(xi @ ww), w) for i in range(B)])
    assert gw.shape == (B, 4)
    assert np.allclose(gw, fd, atol=1e-5)


def test_value_and_grad_per_sample_shared_weight():
    r = _rng(1)
    B = 4
    X = r.standard_normal((B, 4))
    w = r.standard_normal((4,))
    val, (gx, gw) = vmap(value_and_grad(linear_scalar), in_axes=(0, None))(X, w)
    assert val.shape == (B,)
    assert np.allclose(val, X @ w)
    assert gw.shape == (B, 4) and np.allclose(gw, X)  # d (x.w)/dw = x, per sample


# -- per-primitive backward-batch-preservation: a shared-weight grad through each of
# max / where / matmul / reduction must stay per-sample (shape (B, *w.shape)). --------
def prim_max(x, w):
    return np.max(x * w)  # shared w; max over the per-example vector


def prim_where(x, w):
    return np.sum(np.where(x > 0, x * w, w))  # shared w through a select


def prim_matmul(x, w):
    return x @ (w @ x)  # shared matrix w:(d,d), scalar out


def prim_reduction(x, w):
    return np.sum(x * w)  # shared w through a reduction


def _check_per_sample_shared(f, X, w, in_axes=(0, None)):
    (_gx, gw) = vmap(grad(f), in_axes=in_axes)(X, w)
    (_ogx, ogw) = _stack_loop_grad(f, X, w, in_axes=in_axes)
    assert gw.shape == ogw.shape == (X.shape[0],) + np.asarray(w).shape
    assert np.allclose(gw, ogw, atol=1e-6)


def test_per_sample_shared_through_max():
    r = _rng(4)
    _check_per_sample_shared(
        prim_max, r.standard_normal((6, 4)), r.standard_normal((4,))
    )


def test_per_sample_shared_through_where():
    r = _rng(5)
    _check_per_sample_shared(
        prim_where, r.standard_normal((6, 4)), r.standard_normal((4,))
    )


def test_per_sample_shared_through_matmul():
    r = _rng(6)
    _check_per_sample_shared(
        prim_matmul, r.standard_normal((5, 3)), r.standard_normal((3, 3))
    )


def test_per_sample_shared_through_reduction():
    r = _rng(8)
    _check_per_sample_shared(
        prim_reduction, r.standard_normal((6, 4)), r.standard_normal((4,))
    )


def test_unbroadcast_keep_axes_preserves_batch():
    """_unbroadcast(keep_axes=(0,)) keeps a size-1 batch axis instead of summing it."""
    from pycograd.tensor import _unbroadcast

    g = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    # default: a size-1 leading target collapses the batch axis
    assert _unbroadcast(g, (1, 3, 4)).shape == (1, 3, 4)
    assert np.allclose(_unbroadcast(g, (1, 3, 4)), g.sum(axis=0, keepdims=True))
    # keep_axes=(0,): the per-example axis survives
    kept = _unbroadcast(g, (1, 3, 4), keep_axes=(0,))
    assert kept.shape == (2, 3, 4)
    assert np.allclose(kept, g)


def test_backward_batched_cotangent_keep_batch_axis():
    """Var.backward(cotangent, keep_batch_axis=0): a batched cotangent over a per-example
    output, with a size-1-batch shared leaf, yields a per-sample gradient (B, *w.shape).
    """
    from pycograd import ops
    from pycograd.tensor import Var

    B, d = 5, 4
    x = _rng(2).standard_normal((B, d))
    w = _rng(9).standard_normal((d,))
    xv = Var(x)
    wv = Var(w[np.newaxis, :])  # size-1 batch axis so it broadcasts over the batch
    out = ops.d_sum(xv * wv, axis=1)  # per-example dot -> shape (B,)
    out.backward(cotangent=np.ones(B), keep_batch_axis=0)
    assert wv.grad.shape == (B, d)  # kept per-sample, not summed to (1, d)
    assert np.allclose(wv.grad, x)  # d (x_i . w)/dw = x_i, per sample
    # without keep_batch_axis the shared leaf's grad collapses (default behavior)
    xv2, wv2 = Var(x), Var(w[np.newaxis, :])
    out2 = ops.d_sum(xv2 * wv2, axis=1)
    out2.backward(cotangent=np.ones(B))
    assert wv2.grad.shape == (1, d)
    assert np.allclose(wv2.grad, x.sum(axis=0, keepdims=True))


def test_per_example_grad_still_works():
    # the existing data-arg per-sample path must keep working unchanged
    X = _rng().standard_normal((5, 4))
    g = per_example_grad(per_sample_sq)(X)
    assert g.shape == X.shape
    assert np.allclose(g, 2 * X, atol=1e-6)


# ---------------------------------------------------------------------------
# vmap composed with ambient-weight grad (``weights.grad`` of a ``vmap`` forward)
# ---------------------------------------------------------------------------
# The weights arrive through a closure (the ``with weights:`` proxies), not as mapped
# args, so vmap never sees their ``Var``s. Two seams make the gradient survive: a live
# grad pass keeps the vmap output on the tape (``grad_is_recording``), and ``full_raise``
# resolves a ``Weight`` proxy that meets a ``BatchTracer`` to its live ``Var``. The
# data ``X`` and the per-example forward live at module scope so the recompiled-on-
# instrumentation objective resolves them as globals (a local would be dropped).
_AMBIENT_X = _rng(0).standard_normal((5, 2))


def _ambient_per_example(x):
    return np.sum(x @ aw + ab)  # noqa: F821 -- aw/ab injected by ``with weights:``


def _ambient_vmap_objective():
    return np.sum(vmap(_ambient_per_example)(_AMBIENT_X))


def test_grad_flows_through_vmap_with_ambient_weights():
    weights = params(aw=np.arange(6.0).reshape(2, 3) * 0.1, ab=np.zeros(3))
    with weights:
        value, grads = weights.grad(_ambient_vmap_objective)

    # Oracle: the identical scalar differentiated wrt explicit-arg weights (no vmap).
    def ref(p):
        return np.sum(np.sum(_AMBIENT_X @ p["aw"] + p["ab"], axis=1))

    rvalue, (rgrads,) = value_and_grad(ref)(weights)
    np.testing.assert_allclose(float(value), float(rvalue), rtol=1e-10)
    np.testing.assert_allclose(grads["aw"], rgrads["aw"], rtol=1e-9)
    np.testing.assert_allclose(grads["ab"], rgrads["ab"], rtol=1e-9)
    # the gradient must actually be non-trivial (regression: it came back all-zero)
    assert np.linalg.norm(grads["aw"]) > 1e-6


# ---------------------------------------------------------------------------
# the pipescript application hook resolves a bare ``|> fn`` stage under any trace
# ---------------------------------------------------------------------------
def test_autodiff_hook_resolves_for_batch_tracer_not_plain():
    # Regression: a bare function pipe stage (``x |> relu``) under vmap used to run the
    # staged function un-instrumented, because the hook only fired for a ``Var`` -- a
    # ``BatchTracer`` then met a raw ufunc inside it. The hook must resolve for any tracer.
    from pycograd.batching import BatchTrace, BatchTracer
    from pycograd.extension import _autodiff_hook
    from pycograd.tensor import Var
    from pycograd.trace import new_main

    # plain array: leave the function untouched (pure-inference fast path)
    assert _autodiff_hook(np.exp, np.array([1.0])) is np.exp
    # base-level Var: resolve (np.exp -> its differentiable primitive)
    assert _autodiff_hook(np.exp, Var(np.array([1.0]))) is not np.exp
    # vmap BatchTracer: must resolve too (the fix)
    with new_main(BatchTrace) as main:
        bt = BatchTracer(BatchTrace(main), Var(np.zeros(3)), 0)
        assert _autodiff_hook(np.exp, bt) is not np.exp


# ---------------------------------------------------------------------------
# RWKV: the recurrence + token-shift must vectorize over a batch of sequences
# (the recurrent loop reads ``k.shape[0]`` -- the *logical* per-example length --
# so it unrolls correctly under a batch level).
# ---------------------------------------------------------------------------
_RWKV_BLK = M._init_rwkv_block(_rng(1), 4)


def rwkv_block_fwd(x):  # x: (T, 4) one sequence
    return M.rwkv_block(x, _RWKV_BLK)


def rwkv_block_energy(x):
    return np.sum(M.rwkv_block(x, _RWKV_BLK) ** 2)


def test_vmap_rwkv_block_forward():
    _check(rwkv_block_fwd, (_rng(2).standard_normal((5, 6, 4)),))


def test_vmap_rwkv_block_per_sample_grad():
    X = _rng(3).standard_normal((5, 6, 4))
    got = per_example_grad(rwkv_block_energy)(X)
    ref = np.stack(
        [np.asarray(grad(rwkv_block_energy)(X[i])[0]) for i in range(len(X))]
    )
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# GRU / LSTM: the gated scans seed their state from a bias ``* 0.0`` (not a shape
# read off the input), so the per-timestep recurrence vectorizes over a batch of
# sequences exactly like RWKV's.
# ---------------------------------------------------------------------------
_GRU_CELL = M._init_gru_cell(_rng(1), 4, 4)
_LSTM_CELL = M._init_lstm_cell(_rng(2), 4, 4)


def gru_scan_fwd(x):  # x: (T, 4) one sequence
    return M.gru_scan(x, _GRU_CELL)


def gru_scan_energy(x):
    return np.sum(M.gru_scan(x, _GRU_CELL) ** 2)


def lstm_scan_fwd(x):
    return M.lstm_scan(x, _LSTM_CELL)


def lstm_scan_energy(x):
    return np.sum(M.lstm_scan(x, _LSTM_CELL) ** 2)


def test_vmap_gru_scan_forward():
    _check(gru_scan_fwd, (_rng(3).standard_normal((5, 6, 4)),))


def test_vmap_lstm_scan_forward():
    _check(lstm_scan_fwd, (_rng(4).standard_normal((5, 6, 4)),))


def test_vmap_gru_scan_per_sample_grad():
    X = _rng(5).standard_normal((5, 6, 4))
    got = per_example_grad(gru_scan_energy)(X)
    ref = np.stack([np.asarray(grad(gru_scan_energy)(X[i])[0]) for i in range(len(X))])
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-6)


def test_vmap_lstm_scan_per_sample_grad():
    X = _rng(6).standard_normal((5, 6, 4))
    got = per_example_grad(lstm_scan_energy)(X)
    ref = np.stack([np.asarray(grad(lstm_scan_energy)(X[i])[0]) for i in range(len(X))])
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-6)

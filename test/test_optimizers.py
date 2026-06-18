# -*- coding: utf-8 -*-
"""Tests for the stateful optimizers (:mod:`pycograd.optimizers`) and the
minibatch utilities (:mod:`pycograd.data`).

Loss/target functions are module-level so ``value_and_grad`` can recompile them
under the tracer (``inspect.getsource``), mirroring ``test_autodiff.py``.
"""
import pytest

np = pytest.importorskip("numpy")

from pycograd import (  # noqa: E402
    SGD,
    Adam,
    AdamW,
    DataLoader,
    batches,
    clip_grad_norm,
    constant_lr,
    cosine_decay,
    frozen,
    params,
    step_decay,
    tied,
    value_and_grad,
)
from pycograd.examples.models import (  # noqa: E402
    _accuracy,
    _init_mlp_tree,
    _mlp_tree_accuracy,
    logistic_loss,
    logistic_param_loss,
    mlp_tree_loss,
)

# --- module-level targets (so getsource works under instrumentation) --------
_QUAD_TARGET = np.array([3.0, -2.0, 0.5])


def quad(p):
    return np.sum((p - _QUAD_TARGET) ** 2)


def tied_sum_sq(m):
    return np.sum((m["a"] + m["b"]) ** 2)


# --- optimizer convergence --------------------------------------------------
def test_adam_logistic_convergence():
    p = (np.zeros(2), 0.0)
    opt = Adam(lr=0.1)
    for _ in range(300):
        _loss, g = value_and_grad(logistic_loss)(*p)
        p = opt.step(p, g)
    w, b = p
    assert _accuracy(w, b) > 0.9


def test_sgd_momentum_mlp_convergence():
    p = _init_mlp_tree(np.random.default_rng(1))
    opt = SGD(lr=0.3, momentum=0.9)
    for _ in range(300):
        _loss, (g,) = value_and_grad(mlp_tree_loss)(p)
        p = opt.step(p, g)
    assert _mlp_tree_accuracy(p) > 0.9


@pytest.mark.parametrize("opt_factory", [lambda: Adam(lr=0.2), lambda: AdamW(lr=0.2)])
def test_adam_reaches_quadratic_minimum(opt_factory):
    p = np.zeros(3)
    opt = opt_factory()
    for _ in range(2000):
        _loss, (g,) = value_and_grad(quad)(p)
        p = opt.step(p, g)
    np.testing.assert_allclose(p, _QUAD_TARGET, atol=1e-2)


def test_sgd_plain_matches_manual_step():
    # momentum=0 SGD must equal the hand-written p <- p - lr*g for one step.
    p = np.array([1.0, 2.0, 3.0])
    _loss, (g,) = value_and_grad(quad)(p)
    stepped = SGD(lr=0.1).step(p, g)
    np.testing.assert_allclose(stepped, p - 0.1 * np.asarray(g))


# --- Param-wrapper handling -------------------------------------------------
def test_frozen_param_is_held_fixed():
    m = params(w=np.zeros(2), b=frozen(7.0))
    opt = Adam(lr=0.1)
    for _ in range(50):
        _loss, (g,) = value_and_grad(logistic_param_loss)(m)
        m = opt.step(m, g)
    assert float(m["b"].value) == 7.0  # frozen bias never moved
    assert np.any(m["w"].value != 0.0)  # trainable weight did


def test_tied_params_stay_equal():
    m = params(a=tied("s", np.array([1.0, 2.0])), b=tied("s", np.array([1.0, 2.0])))
    opt = Adam(lr=0.05)
    for _ in range(30):
        _loss, (g,) = value_and_grad(tied_sum_sq)(m)
        m = opt.step(m, g)
    np.testing.assert_allclose(m["a"].value, m["b"].value)


def test_step_raises_on_structure_mismatch():
    opt = SGD(lr=0.1)
    with pytest.raises(ValueError):
        opt.step((np.zeros(2), np.zeros(3)), (np.zeros(2),))


# --- gradient clipping ------------------------------------------------------
def test_clip_grad_norm_scales_to_max():
    grads = {"x": np.array([3.0, 4.0]), "y": None}  # global norm 5
    clipped = clip_grad_norm(grads, 1.0)
    total = np.linalg.norm(clipped["x"])
    assert clipped["y"] is None
    np.testing.assert_allclose(total, 1.0)


def test_clip_grad_norm_leaves_small_grads_unchanged():
    grads = {"x": np.array([0.1, 0.2])}
    assert clip_grad_norm(grads, 1.0) is grads


# --- learning-rate schedules ------------------------------------------------
def test_constant_lr_matches_float():
    p = np.array([1.0, 2.0, 3.0])
    _loss, (g,) = value_and_grad(quad)(p)
    a = SGD(lr=0.1).step(p, g)
    b = SGD(lr=constant_lr(0.1)).step(p, g)
    np.testing.assert_allclose(a, b)


def test_step_decay_values():
    sched = step_decay(1.0, 0.5, every=10)
    assert sched(1) == 1.0
    assert sched(10) == 1.0
    assert sched(11) == 0.5
    assert sched(21) == 0.25


def test_cosine_decay_reaches_min():
    sched = cosine_decay(1.0, total_steps=100, min_lr=0.1)
    assert sched(1) == pytest.approx(1.0, abs=1e-3)
    assert sched(100) == pytest.approx(0.1)
    assert sched(200) == pytest.approx(0.1)  # clamps after total_steps


def test_schedule_applied_by_optimizer():
    # A schedule that drops lr to 0 after the first step freezes the params.
    p = np.array([1.0, 2.0, 3.0])
    opt = SGD(lr=lambda step: 0.1 if step == 1 else 0.0)
    _loss, (g,) = value_and_grad(quad)(p)
    p1 = opt.step(p, g)
    _loss, (g,) = value_and_grad(quad)(p1)
    p2 = opt.step(p1, g)
    np.testing.assert_allclose(p1, p2)  # second step had lr=0


# --- batching ---------------------------------------------------------------
def test_batches_cover_all_rows_once():
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    seen = np.concatenate([yb for _xb, yb in batches(X, y, batch_size=3)])
    np.testing.assert_array_equal(np.sort(seen), np.arange(10))


def test_batches_drop_last():
    X = np.arange(10).reshape(10, 1)
    full = list(batches(X, batch_size=3, drop_last=False))
    dropped = list(batches(X, batch_size=3, drop_last=True))
    assert len(full) == 4 and len(dropped) == 3
    assert all(len(b) == 3 for b in dropped)


def test_batches_single_array_is_bare():
    X = np.arange(6).reshape(6, 1)
    batch = next(batches(X, batch_size=2))
    assert isinstance(batch, np.ndarray)


def test_batches_shuffle_is_reproducible_and_aligned():
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10) * 100
    b1 = list(batches(X, y, batch_size=3, shuffle=True, rng=np.random.default_rng(0)))
    b2 = list(batches(X, y, batch_size=3, shuffle=True, rng=np.random.default_rng(0)))
    for (x1, y1), (x2, y2) in zip(b1, b2):
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(x1[:, 0] * 100, y1)  # rows stay aligned


def test_batches_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        list(batches(np.zeros(4), np.zeros(5), batch_size=2))


def test_dataloader_len_and_reuse():
    X = np.arange(10).reshape(10, 1)
    loader = DataLoader(X, batch_size=3)
    assert len(loader) == 4
    assert len(DataLoader(X, batch_size=3, drop_last=True)) == 3
    # iterating twice yields the full epoch each time
    assert sum(len(b) for b in loader) == 10
    assert sum(len(b) for b in loader) == 10

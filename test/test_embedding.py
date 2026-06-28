# -*- coding: utf-8 -*-
"""The ``embedding`` primitive (``ops.d_embedding``): a row gather of a parameter table at
integer indices, whose backward scatter-adds into the looked-up rows. ``padding_idx`` holds
a row fixed by zeroing only its gradient. Exercises every transform surface a primitive must
compose with -- grad (vs finite differences), jvp, vmap, vmap(grad), capture + eval_graph,
value_and_grad(capture), eval_shape, and cost_report -- plus the compile backends.

Module-level functions so pyccolo can re-instrument them from source."""
import numpy as np
import pytest

from pycograd import (
    capture,
    cost_report,
    embedding,
    eval_graph,
    eval_shape,
    grad,
    jvp,
    value_and_grad,
    vmap,
)
from pycograd.tensor import _value

_rng = np.random.default_rng(0)
_TABLE = _rng.standard_normal((8, 4))
_IDX = np.array([[1, 3], [3, 7]])  # row 3 looked up twice
_COEF = _rng.standard_normal((2, 2, 4))


def _fd(f, x, eps=1e-6):
    out = np.zeros(np.shape(x))
    for i in range(np.size(x)):
        xp, xm = x.copy(), x.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        out.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return out


def f_embed(table):
    return np.sum(embedding(table, _IDX) * _COEF)


def f_embed_pad(table):
    return np.sum(embedding(table, _IDX, padding_idx=3) * _COEF)


# --- forward / grad ---------------------------------------------------------
def test_forward_matches_fancy_index_and_returns_array():
    out = embedding(_TABLE, _IDX)
    assert isinstance(out, np.ndarray)  # untraced -> plain array, like other ops
    assert out.shape == (2, 2, 4) and np.allclose(out, _TABLE[_IDX])


def test_grad_vs_finite_difference():
    (g,) = grad(f_embed)(_TABLE)
    assert np.allclose(np.asarray(g), _fd(f_embed, _TABLE), atol=1e-5)


def test_grad_scatter_adds_repeated_rows():
    _, (g,) = value_and_grad(lambda t: embedding(t, _IDX))(_TABLE)
    expected = np.zeros_like(_TABLE)
    for i in _IDX.ravel():
        expected[i] += 1.0
    assert np.allclose(np.asarray(g), expected)


# --- padding_idx ------------------------------------------------------------
def test_padding_idx_forward_is_a_plain_gather():
    # padding_idx is reverse-only: the forward still returns the table's own pad row.
    assert np.allclose(np.asarray(embedding(_TABLE, _IDX, padding_idx=3)), _TABLE[_IDX])


def test_padding_idx_zeroes_only_that_rows_gradient():
    _, (g,) = value_and_grad(lambda t: embedding(t, _IDX, padding_idx=3))(_TABLE)
    g = np.asarray(g)
    assert np.allclose(g[3], 0.0)  # pad row held fixed
    # every other looked-up row is unchanged from the no-padding gradient
    _, (g0,) = value_and_grad(lambda t: embedding(t, _IDX))(_TABLE)
    g0 = np.asarray(g0)
    assert np.allclose(g[1], g0[1]) and np.allclose(g[7], g0[7])


def test_padding_idx_grad_matches_fd_off_the_pad_row():
    # padding_idx makes the analytic grad *deliberately* differ from the true derivative
    # at the pad row (the forward still gathers it), so they agree only off that row.
    (g,) = grad(f_embed_pad)(_TABLE)
    g, fd = np.asarray(g), _fd(f_embed_pad, _TABLE)
    off = [r for r in range(_TABLE.shape[0]) if r != 3]
    assert np.allclose(g[off], fd[off], atol=1e-5)
    assert np.allclose(g[3], 0.0) and not np.allclose(fd[3], 0.0)


# --- shape inference --------------------------------------------------------
def test_eval_shape():
    assert eval_shape(lambda t: embedding(t, _IDX), _TABLE).shape == (2, 2, 4)


def test_eval_shape_multidim_feature():
    table = _rng.standard_normal((6, 3, 5))  # feature dims (3, 5)
    assert eval_shape(lambda t: embedding(t, _IDX), table).shape == (2, 2, 3, 5)


# --- forward mode -----------------------------------------------------------
def test_jvp_is_linear_in_the_table():
    dt = _rng.standard_normal(_TABLE.shape)
    p, t = jvp(lambda T: embedding(T, _IDX), (_TABLE,), (dt,))
    assert np.allclose(np.asarray(p), _TABLE[_IDX])
    assert np.allclose(np.asarray(t), dt[_IDX])  # tangent gathers identically


# --- batching ---------------------------------------------------------------
def test_vmap_over_a_batched_table():
    tables = _rng.standard_normal((5, 8, 4))
    out = np.asarray(vmap(lambda T: embedding(T, _IDX))(tables))
    assert out.shape == (5, 2, 2, 4) and np.allclose(out, tables[:, _IDX])


def test_vmap_of_grad_over_a_batched_table():
    tables = _rng.standard_normal((5, 8, 4))
    g = vmap(grad(lambda T: embedding(T, _IDX).sum()))(tables)
    g = np.asarray(g[0] if isinstance(g, tuple) else g)
    expected = np.zeros((8, 4))
    for i in _IDX.ravel():
        expected[i] += 1.0
    assert g.shape == (5, 8, 4) and all(np.allclose(g[b], expected) for b in range(5))


def test_vmap_batched_table_with_padding_idx_is_rejected():
    tables = _rng.standard_normal((5, 8, 4))
    with pytest.raises(NotImplementedError, match="padding_idx"):
        vmap(lambda T: embedding(T, _IDX, padding_idx=3))(tables)


# --- graph capture ----------------------------------------------------------
def test_capture_and_eval_graph():
    g = capture(lambda t: embedding(t, _IDX).sum(), _TABLE)
    got = float(_value(eval_graph(g, _TABLE)))
    assert np.allclose(got, float(embedding(_TABLE, _IDX).sum()))


def test_value_and_grad_of_capture():
    v, (gc,) = value_and_grad(capture(lambda t: embedding(t, _IDX).sum(), _TABLE))(
        _TABLE
    )
    expected = np.zeros_like(_TABLE)
    for i in _IDX.ravel():
        expected[i] += 1.0
    assert np.allclose(np.asarray(gc), expected)


def test_cost_report_classifies_the_embedding_node():
    g = capture(lambda t: embedding(t, _IDX).sum(), _TABLE)
    cost_report(g)  # must not raise (every survivable primitive is classified)


# --- compile backends -------------------------------------------------------
@pytest.mark.parametrize("padding_idx", [None, 3])
def test_torch_backend_parity(padding_idx):
    pytest.importorskip("torch")
    import pycograd.compile as C

    def loss(p):
        return np.sum(embedding(p["table"], _IDX, padding_idx=padding_idx) * _COEF)

    model = {"table": _TABLE}
    v_np, g_np = C.value_and_grad(loss, backend="numpy")(model)
    v_t, g_t = C.value_and_grad(loss, backend="torch")(model)
    assert np.allclose(v_np, v_t, atol=1e-5)
    assert np.allclose(np.asarray(g_np["table"]), np.asarray(g_t["table"]), atol=1e-5)

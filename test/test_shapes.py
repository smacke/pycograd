# -*- coding: utf-8 -*-
"""Shape inference: ``eval_shape``/``summary`` must report the shapes a real run produces.

The real (finite-difference-checked) example models are the oracle: for each, the shape
``eval_shape`` infers from dummy arrays must equal the shape the model actually returns
when run on real data. ``summary`` is checked for correct parameter counts, and the
eager ``ShapeError`` for naming the op and the offending operand shapes.
"""
import time

import numpy as np
import pytest

from pycograd import (
    ShapedArray,
    ShapeDtypeStruct,
    ShapeError,
    Var,
    eval_shape,
    frozen,
    infer_shapes,
    summary,
)
from pycograd.examples import models as M
from pycograd.tree import tree_leaves

SDS = ShapeDtypeStruct


def _rng(seed):
    return np.random.default_rng(seed)


def _flat_spec_shapes(specs):
    return [tuple(s.shape) for s in tree_leaves(specs) if s is not None]


def _flat_real_shapes(out):
    return [np.asarray(leaf).shape for leaf in tree_leaves(out) if leaf is not None]


# (id, loss_fn, args_factory) -- args are passed splatted to both fn and eval_shape, so
# the same factory drives the real run and the inferred run.
_CASES = [
    (
        "logistic",
        M.logistic_param_loss,
        lambda: ({"w": _rng(0).standard_normal((2,)), "b": 0.0},),
    ),
    ("mlp_tree", M.mlp_tree_loss, lambda: (M._init_mlp_tree(_rng(1)),)),
    ("mlp_batch", M.mlp_batch_loss, lambda: (M._init_mlp_tree(_rng(1)), M._Xc, M._Yoh)),
    ("deep_ln_dropout", M.deep_loss, lambda: M._init_deep(_rng(2))),
    ("transformer", M.transformer_loss, lambda: M._init_transformer(_rng(3))),
]


@pytest.mark.parametrize("method", ["dummy", "abstract"])
@pytest.mark.parametrize("cid,fn,argf", _CASES, ids=[c[0] for c in _CASES])
def test_output_shape_matches_real_run(cid, fn, argf, method):
    real = fn(*argf())
    inferred = eval_shape(fn, *argf(), method=method)
    assert _flat_real_shapes(real) == _flat_spec_shapes(inferred), f"{cid}/{method}"


@pytest.mark.parametrize("cid,fn,argf", _CASES, ids=[c[0] for c in _CASES])
def test_abstract_matches_dummy(cid, fn, argf):
    # The dummy (plain-numpy) path is the oracle; the data-free abstract path must agree.
    dummy = eval_shape(fn, *argf(), method="dummy")
    abstract = eval_shape(fn, *argf(), method="abstract")
    assert _flat_spec_shapes(dummy) == _flat_spec_shapes(abstract), cid


def test_abstract_rule_coverage_matches_intercept():
    # Every numpy/math call pycograd can differentiate must have a shape rule, so a
    # newly added op can't silently lack one.
    from pycograd.ops import _INTERCEPT
    from pycograd.shapes import _ABSTRACT

    assert set(_ABSTRACT) == set(_INTERCEPT)


def test_abstract_is_o1_on_huge_shapes():
    # A (100000, 100000) matmul would be ~80GB if materialized; the abstract path
    # never allocates, so it returns instantly.
    t = time.time()
    out = eval_shape(
        lambda a, b: a @ b,
        SDS((100_000, 100_000)),
        SDS((100_000, 7)),
        method="abstract",
    )
    assert out.shape == (100_000, 7)
    assert time.time() - t < 1.0


def test_abstract_backend_imports_no_framework():
    import importlib
    import sys

    from pycograd.backends import get_backend

    # Construct the backend fresh and assert the construction imports no framework
    # (order-independent: only the *delta* across the call is checked).
    sys.modules.pop("pycograd.backends.abstract_backend", None)
    frameworks = ("jax", "torch", "tensorflow")
    before = {m for m in frameworks if m in sys.modules}
    importlib.import_module("pycograd.backends.abstract_backend").AbstractBackend()
    get_backend("abstract")
    after = {m for m in frameworks if m in sys.modules}
    assert before == after


def test_abstract_matmul_error_matches_eager_message():
    # The same ShapeError text from both the eager Var path and the abstract path.
    with pytest.raises(ShapeError) as eager:
        (Var(np.zeros((3, 4))) @ Var(np.zeros((5, 6)))).value
    with pytest.raises(ShapeError) as abstract:
        eval_shape(lambda a, b: a @ b, SDS((3, 4)), SDS((5, 6)), method="abstract")
    assert str(eager.value) == str(abstract.value)


def test_abstract_boolean_mask_is_symbolic():
    # A boolean mask is data-dependent: its length flows through as a symbolic Dim
    # (was a ShapeError before symbolic dimensions existed).
    from pycograd import Dim

    out = eval_shape(lambda x: x[x > 0], SDS((10,)), method="abstract")
    assert out.ndim == 1 and isinstance(out.shape[0], Dim)
    assert repr(out) == "f64[n0]"


def test_abstract_boolean_mask_keeps_trailing_axes():
    from pycograd import Dim

    out = eval_shape(lambda x: x[x[:, 0] > 0], SDS((10, 3)))
    assert isinstance(out.shape[0], Dim) and out.shape[1:] == (3,)


def test_abstract_mask_relationship_tracking():
    # Two structurally identical masks share one symbol, so adding their selections
    # broadcasts to that same symbol rather than a fresh one.
    def f(x):
        a = x[x > 0]
        b = x[x > 0]
        return {"a": a, "sum": a + b}

    out = eval_shape(f, SDS((10,)))
    assert out["a"].shape[0] == out["sum"].shape[0]


def test_abstract_distinct_predicates_dont_merge():
    # Soundness: different masks must NOT be claimed equal.
    out = eval_shape(lambda x: {"gt": x[x > 0], "lt": x[x < 0]}, SDS((10,)))
    assert out["gt"].shape[0] != out["lt"].shape[0]


def test_abstract_symbol_names_restart_per_run():
    a = eval_shape(lambda x: x[x > 0], SDS((10,)))
    b = eval_shape(lambda x: x[x > 0], SDS((10,)))
    assert repr(a) == repr(b) == "f64[n0]"


def test_abstract_advanced_integer_index_is_determinable():
    # An integer index array's result shape comes from the *key's* shape, not its
    # values -- so it is exact, with no symbol.
    out = eval_shape(lambda x, idx: x[idx], SDS((10, 3)), SDS((4,), np.intp))
    assert out.shape == (4, 3)


def test_abstract_advanced_index_after_slice():
    out = eval_shape(lambda x, idx: x[:, idx], SDS((5, 6)), SDS((4,), np.intp))
    assert out.shape == (5, 4)


def test_abstract_advanced_non_contiguous_goes_to_front():
    # Separated advanced indices put their broadcast block first (numpy semantics).
    out = eval_shape(
        lambda x, i, j: x[i, :, j],
        SDS((5, 6, 7)),
        SDS((4,), np.intp),
        SDS((4,), np.intp),
    )
    assert out.shape == (4, 6)


def test_abstract_symbolic_dim_flows_through_reshape():
    from pycograd import Dim

    out = eval_shape(lambda x: x[x > 0].reshape(-1, 1), SDS((12,)))
    assert isinstance(out.shape[0], Dim) and out.shape[1] == 1


def test_abstract_symbolic_contract_dim_does_not_raise():
    # A matmul whose contract dim is symbolic can't be proven incompatible -> no error.
    out = eval_shape(lambda x, w: x[x > 0] @ w, SDS((10,)), SDS((10, 4)))
    assert out.shape == (4,)


def test_abstract_static_index_ok():
    out = eval_shape(lambda x: x[1:3, ::2], SDS((10, 8)), method="abstract")
    assert out.shape == (2, 4)


def test_shapedarray_metadata_is_concrete():
    a = ShapedArray((3, 4))
    assert a.shape == (3, 4) and a.ndim == 2 and a.size == 12
    assert a.shape[-1] ** -0.5 == 0.5  # the attention-style read works
    assert (a @ ShapedArray((4, 5))).shape == (3, 5)
    assert a.sum(axis=1).shape == (3,)
    assert a.T.shape == (4, 3)


def test_intermediate_output_shape():
    # mlp_forward returns the (N, n_classes) probability matrix, not just a scalar.
    out = infer_shapes(
        M.mlp_forward, SDS((5, 2)), SDS((2, 16)), SDS((16,)), SDS((16, 3)), SDS((3,))
    )
    assert out == (5, 3)


def test_attention_reads_shape_abstractly():
    # attention uses ``q.shape[-1] ** -0.5`` -- a data-independent shape read that must
    # survive dummy inference; output is (seq, d_v).
    out = eval_shape(M.attention, SDS((3, 4)), SDS((3, 4)), SDS((3, 4)))
    assert out.shape == (3, 4)


def test_nonarray_args_pass_through():
    # A bool flag (``training``) is not numeric, so it reaches the function unchanged.
    def fn(x, training):
        return x * 2.0 if training else x

    assert eval_shape(fn, SDS((4, 5)), True).shape == (4, 5)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def test_summary_param_counts():
    params = M._init_mlp_tree(_rng(1))
    s = summary(M.mlp_batch_loss, params, (5, 2), (5, 3), print_fn=None)
    # w:(2,16)=32, b:(16,)=16, w:(16,3)=48, b:(3,)=3  ->  99
    assert s.total == 99
    assert s.trainable == 99
    assert s.output.shape == ()  # scalar loss
    assert {r.name for r in s.rows} == {"hidden.w", "hidden.b", "out.w", "out.b"}


def test_summary_frozen_excluded_from_trainable():
    model = {"w": np.zeros((2, 3)), "b": frozen(np.zeros(3))}
    s = summary(lambda m: np.sum(m["w"]) + np.sum(m["b"]), model, print_fn=None)
    assert s.total == 9  # 6 + 3
    assert s.trainable == 6  # frozen b excluded
    (b_row,) = [r for r in s.rows if r.name == "b"]
    assert b_row.trainable is False
    assert "(frozen)" in str(s)


# ---------------------------------------------------------------------------
# friendly shape errors (eager Var path)
# ---------------------------------------------------------------------------
def test_shape_error_matmul_names_op_and_shapes():
    with pytest.raises(ShapeError) as ei:
        (np.zeros((3, 4)) @ Var(np.zeros((5, 6)))).value
    msg = str(ei.value)
    assert "matmul" in msg and "(3, 4)" in msg and "(5, 6)" in msg


def test_shape_error_is_a_value_error():
    # Subclassing ValueError keeps existing ``except ValueError`` handlers working.
    with pytest.raises(ValueError):
        Var(np.zeros((2, 3))).reshape((4, 4))


def test_shape_error_concatenate():
    from pycograd.ops import d_concatenate

    with pytest.raises(ShapeError) as ei:
        d_concatenate([Var(np.zeros((2, 3))), Var(np.zeros((2, 4)))], axis=0)
    assert "concatenate" in str(ei.value)


def test_shape_error_reshape():
    with pytest.raises(ShapeError) as ei:
        Var(np.zeros((2, 3))).reshape((4, 4))
    assert "reshape" in str(ei.value) and "(2, 3)" in str(ei.value)


# ---------------------------------------------------------------------------
# ShapeDtypeStruct
# ---------------------------------------------------------------------------
def test_shapedtypestruct_basics():
    s = SDS((2, 3))
    assert s.shape == (2, 3) and s.ndim == 2 and s.size == 6
    assert repr(s) == "f64[2,3]"
    assert SDS(5).shape == (5,)  # bare int normalizes to a 1-tuple
    assert SDS((), np.float32).dtype == np.dtype(np.float32)


def test_eval_shape_rejects_unknown_method():
    with pytest.raises(ValueError):
        eval_shape(lambda x: x, SDS((2,)), method="bogus")

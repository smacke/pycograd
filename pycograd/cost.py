# -*- coding: utf-8 -*-
"""A static cost model over the capture IR (:mod:`pycograd.capture`).

Given a captured :class:`~pycograd.capture.Graph`, this estimates -- without
running it -- how expensive each intermediate is along three axes:

* **CPU** -- a roofline estimate ``max(flops / flops_per_sec, traffic / bandwidth)``
  per node, summed over the graph. ``flops`` come from a per-primitive rule that
  reads the node's abstract shapes (so a ``matmul`` is ``2*M*N*K``, an elementwise
  op is ``size`` times a transcendental/cheap weight, a movement op is free).
* **Memory** -- the bytes each intermediate materializes (``size * itemsize``) and
  the **peak live memory** of the whole graph under SSA execution order, computed
  from last-use liveness (a value is live from when its node runs until its last
  consumer / a graph output).
* **Disk (SSD)** -- the round-trip cost of *spilling* an intermediate to SSD and
  reading it back, so a caller deciding keep-in-RAM vs recompute vs spill can
  compare ``recompute_time`` (the node's own CPU cost) against ``spill_time``.

This is an *analysis* -- it reports costs and never rewrites the graph. A scheduler
that inserts remat/spill nodes (ROADMAP Phase 4) is the natural consumer.

The hardware constants live on :class:`CostModel` (sensible NVMe-class defaults,
fully overridable). :func:`calibrate` micro-benchmarks the host to fill them in.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import numpy as np

from pycograd.capture import _CONST, _INPUT, Const, Graph, Node, Ref
from pycograd.shapes import ShapeDtypeStruct

if TYPE_CHECKING:
    from pycograd._typing import Prim


# ---------------------------------------------------------------------------
# Hardware parameters.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CostModel:
    """Hardware constants the cost model is parameterized by (all SI units: seconds,
    bytes, FLOPs). The defaults are a conservative modern single-host estimate --
    construct your own with measured numbers, or use :func:`calibrate` to fill them
    in from the machine you are on.

    ``flops_per_sec`` is effective f64 throughput (BLAS-class, so ``matmul`` is
    realistic; the elementwise weights below fold transcendental cost into a node's
    FLOP count). ``mem_bandwidth`` bounds memory-bound ops via the roofline.
    ``ssd_*`` model spilling an intermediate to SSD and reading it back.
    ``symbolic_dim_size`` is the assumed extent of a data-dependent (symbolic) dim
    so a graph with masks still costs to a finite number.
    """

    flops_per_sec: float = 5.0e10  # ~50 GFLOP/s effective f64
    mem_bandwidth: float = 2.0e10  # ~20 GB/s RAM read+write
    ssd_read_bandwidth: float = 3.0e9  # ~3 GB/s NVMe read
    ssd_write_bandwidth: float = 1.5e9  # ~1.5 GB/s NVMe write
    ssd_latency: float = 1.0e-4  # ~100 us fixed per spill/restore
    ram_capacity: Optional[int] = None  # bytes; None = unbounded
    symbolic_dim_size: int = 64

    def spill_time(self, nbytes: int) -> float:
        """Seconds to write ``nbytes`` to SSD and read them back (a spill round-trip)."""
        write = nbytes / self.ssd_write_bandwidth + self.ssd_latency
        read = nbytes / self.ssd_read_bandwidth + self.ssd_latency
        return write + read


DEFAULT_COST_MODEL = CostModel()


# ---------------------------------------------------------------------------
# Per-primitive FLOP weights. Keyed by ``prim.__name__`` (matching the IR's own
# rendering) so the table reads at a glance and needs no op imports.
# ---------------------------------------------------------------------------
# Elementwise: flops = output_size * weight.
_TRANSCENDENTAL = {  # ~one libm call per element
    "d_exp",
    "d_log",
    "d_sin",
    "d_cos",
    "d_tanh",
    "d_sqrt",
    "d_sinh",
    "d_cosh",
    "d_arctan",
    "d_log1p",
    "d_expm1",
    "d_sigmoid",
}
_POW = {"d_pow"}  # x**y via exp/log -> transcendental-class
_CHEAP_ELEMENTWISE = {  # one or two arithmetic ops per element
    "d_neg",
    "d_abs",
    "d_square",
    "d_reciprocal",
    "d_add",
    "d_sub",
    "d_mul",
    "d_div",
    "d_maximum",
    "d_minimum",
    "d_lt",
    "d_le",
    "d_gt",
    "d_ge",
    "d_eq",
    "d_ne",
    "d_where",
    "d_clip",
}
_GATED = {"d_gated_act"}  # fused tanh(f)*sigmoid(s): two transcendentals + a mul

# Reductions: flops = input_size * weight (one pass; var/std take ~two).
_REDUCE = {"d_sum", "d_mean", "d_max", "d_min", "d_cumsum"}
_REDUCE_HEAVY = {"d_var", "d_std"}

# Pure data movement / views: no arithmetic, only the memory traffic of the copy.
_MOVEMENT = {
    "d_reshape",
    "d_transpose",
    "d_expand_dims",
    "d_broadcast_to",
    "d_getitem",
    "d_concatenate",
    "d_stack",
    "d_vstack",
    "d_hstack",
    "d_column_stack",
    "d_dstack",
}

_W_TRANSCENDENTAL = 8.0
_W_CHEAP = 1.0
_W_GATED = 18.0
_W_REDUCE_HEAVY = 2.0


# ---------------------------------------------------------------------------
# Shape / size helpers.
# ---------------------------------------------------------------------------
def _int_size(aval: ShapeDtypeStruct, symbolic: int) -> int:
    """Element count of ``aval``, substituting ``symbolic`` for any data-dependent dim."""
    n = 1
    for d in aval.shape:
        n *= d if isinstance(d, int) else symbolic
    return n


def _nbytes(aval: ShapeDtypeStruct, symbolic: int) -> int:
    """Bytes to materialize ``aval`` (``size * itemsize``)."""
    return _int_size(aval, symbolic) * aval.dtype.itemsize


def _aval_of(spec: object, by_id: "dict[int, Node]") -> Optional[ShapeDtypeStruct]:
    """The abstract value of a *numeric* operand (a ``Ref`` producer or an array/scalar
    ``Const``), or ``None`` for a non-array operand (an einsum subscript string, a slice,
    an index tuple) that carries no tensor traffic."""
    if isinstance(spec, Ref):
        nd = by_id.get(spec.id)
        return nd.aval if nd is not None else None
    if isinstance(spec, Const):
        try:
            arr = np.asarray(spec.value)
        except Exception:  # pragma: no cover - genuinely opaque operand
            return None
        if arr.dtype.kind in "fiub":
            return ShapeDtypeStruct(arr.shape, arr.dtype)
    return None


def _operand_avals(node: Node, by_id: "dict[int, Node]") -> "list[ShapeDtypeStruct]":
    """The numeric operands' avals, flattening structural (list/tuple) args."""
    out: list[ShapeDtypeStruct] = []
    for spec in node.args:
        items = spec if isinstance(spec, (list, tuple)) else (spec,)
        for it in items:
            av = _aval_of(it, by_id)
            if av is not None:
                out.append(av)
    return out


# ---------------------------------------------------------------------------
# FLOP estimation.
# ---------------------------------------------------------------------------
def _einsum_flops(node: Node, operands: "list[ShapeDtypeStruct]", symbolic: int) -> int:
    """``2 * prod(size of every distinct index label)`` -- one multiply-add per point of
    the full iteration space. Falls back to a generic estimate on ``...`` / a parse miss.
    """
    subs = node.args[0].value if node.args and isinstance(node.args[0], Const) else None
    if not isinstance(subs, str) or "." in subs:
        return 2 * max((_int_size(a, symbolic) for a in operands), default=1)
    sizes: dict[str, int] = {}
    for sub, av in zip(subs.split("->")[0].split(","), operands):
        for label, dim in zip(sub.strip(), av.shape):
            sizes[label] = dim if isinstance(dim, int) else symbolic
    prod = 1
    for v in sizes.values():
        prod *= v
    return 2 * prod


def node_flops(node: Node, by_id: "dict[int, Node]", model: CostModel) -> int:
    """Estimated floating-point operations to evaluate ``node`` (0 for inputs/consts and
    pure data movement)."""
    if node.prim is _INPUT or node.prim is _CONST:
        return 0
    name = getattr(node.prim, "__name__", str(node.prim))
    sym = model.symbolic_dim_size
    out = _int_size(node.aval, sym)
    operands = _operand_avals(node, by_id)

    if name == "_matmul":
        # out is [..., M, N]; the contracted dim K is the inner dim of the first operand.
        k = 1
        if operands and operands[0].shape:
            d = operands[0].shape[-1]
            k = d if isinstance(d, int) else sym
        return 2 * out * k
    if name == "d_einsum":
        return _einsum_flops(node, operands, sym)
    if name in _MOVEMENT:
        return 0
    if name in _TRANSCENDENTAL:
        return int(out * _W_TRANSCENDENTAL)
    if name in _POW:
        return int(out * _W_TRANSCENDENTAL)
    if name in _GATED:
        return int(out * _W_GATED)
    if name in _REDUCE or name in _REDUCE_HEAVY:
        # cost scales with the (larger) input swept, not the reduced output
        in_size = max((_int_size(a, sym) for a in operands), default=out)
        w = _W_REDUCE_HEAVY if name in _REDUCE_HEAVY else 1.0
        return int(in_size * w)
    # default: a cheap elementwise op (also the conservative fallback for any
    # primitive not yet classified above).
    return int(out * _W_CHEAP)


# ---------------------------------------------------------------------------
# Per-node and whole-graph cost.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeCost:
    """The estimated cost of one IR node."""

    id: int
    prim: str
    aval: ShapeDtypeStruct
    flops: int
    out_bytes: int  # bytes the node's output occupies
    read_bytes: int  # input bytes read (roofline traffic = read + out)
    compute_time: float  # flops / flops_per_sec
    memory_time: float  # traffic / mem_bandwidth
    time: float  # roofline: max(compute_time, memory_time)
    recompute_time: float  # cost to recompute this value (== time); for cache decisions
    spill_time: float  # SSD write-then-read round-trip for this value


@dataclass(frozen=True)
class GraphCost:
    """Whole-graph cost estimate (see :func:`cost_report`)."""

    nodes: "list[NodeCost]"
    model: CostModel
    total_flops: int
    total_compute_time: float  # sum of per-node roofline times
    peak_memory_bytes: int  # max simultaneously-live bytes under SSA order
    peak_memory_node: int  # id of the node whose execution hits the peak
    over_budget: bool  # peak_memory_bytes > model.ram_capacity (False if unbounded)

    def by_id(self) -> "dict[int, NodeCost]":
        return {nc.id: nc for nc in self.nodes}

    def hotspots(self, k: int = 5) -> "list[NodeCost]":
        """The ``k`` costliest nodes by roofline time (compute hotspots)."""
        return sorted(self.nodes, key=lambda nc: nc.time, reverse=True)[:k]

    def spill_candidates(self, k: int = 5) -> "list[NodeCost]":
        """The ``k`` largest intermediates that are *cheaper to spill than to recompute*
        -- the values a memory-constrained scheduler should prefer to page to SSD rather
        than rematerialize."""
        cands = [nc for nc in self.nodes if nc.spill_time < nc.recompute_time]
        return sorted(cands, key=lambda nc: nc.out_bytes, reverse=True)[:k]

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"GraphCost({self.total_flops:.3g} flops, "
            f"{self.total_compute_time * 1e3:.3g} ms, "
            f"peak {self.peak_memory_bytes / 1e6:.3g} MB)"
        )

    def __str__(self) -> str:  # pragma: no cover - debug aid
        return pretty_cost(self)


def live_intervals(graph: Graph, model: CostModel) -> "dict[int, tuple[int, int, int]]":
    """Per-node live range: node id -> ``(produce_step, last_use_step, bytes)``, where the
    steps are positions in SSA order. A value is live across ``[produce_step,
    last_use_step]`` inclusive; a graph output's last use is the final step. This is the
    raw material for both the peak-memory analysis and the remat planner."""
    last = len(graph.nodes) - 1
    last_use: dict[int, int] = {}
    for i, nd in enumerate(graph.nodes):
        for r in _refs(nd.args):
            last_use[r] = i
    for o in graph.outputs:
        last_use[o] = last  # an output stays live through the whole run

    sym = model.symbolic_dim_size
    out: dict[int, tuple[int, int, int]] = {}
    for i, nd in enumerate(graph.nodes):
        out[nd.id] = (i, last_use.get(nd.id, i), _nbytes(nd.aval, sym))
    return out


def memory_timeline(
    intervals: "dict[int, tuple[int, int, int]]", n_steps: int
) -> "list[int]":
    """Simultaneously-live bytes after each of ``n_steps`` SSA steps, summing every node
    whose ``[produce, last_use]`` range covers that step (see :func:`live_intervals`).
    """
    live = [0] * n_steps
    for produce, last, nbytes in intervals.values():
        for s in range(produce, last + 1):
            live[s] += nbytes
    return live


def peak_of(timeline: "list[int]") -> "tuple[int, int]":
    """``(peak_bytes, step)`` of a memory timeline -- the first step attaining the max."""
    peak = 0
    step = -1
    for s, v in enumerate(timeline):
        if v > peak:
            peak, step = v, s
    return peak, step


def _peak_memory(graph: Graph, model: CostModel) -> "tuple[int, int]":
    """Peak simultaneously-live bytes under SSA execution order, and the id of the node
    whose execution reaches it. Thin wrapper over the timeline helpers."""
    if not graph.nodes:
        return 0, -1
    intervals = live_intervals(graph, model)
    peak, step = peak_of(memory_timeline(intervals, len(graph.nodes)))
    return peak, graph.nodes[step].id if step >= 0 else graph.nodes[0].id


def _refs(args: tuple) -> "set[int]":
    out: set[int] = set()
    stack = list(args)
    while stack:
        x = stack.pop()
        if isinstance(x, Ref):
            out.add(x.id)
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
    return out


def cost_report(graph: Graph, model: CostModel = DEFAULT_COST_MODEL) -> GraphCost:
    """Estimate the CPU / memory / disk cost of ``graph`` under ``model`` (no execution)."""
    by_id = {nd.id: nd for nd in graph.nodes}
    sym = model.symbolic_dim_size
    nodes: list[NodeCost] = []
    total_flops = 0
    total_time = 0.0
    for nd in graph.nodes:
        flops = node_flops(nd, by_id, model)
        out_bytes = _nbytes(nd.aval, sym)
        read_bytes = sum(_nbytes(a, sym) for a in _operand_avals(nd, by_id))
        compute_time = flops / model.flops_per_sec
        memory_time = (read_bytes + out_bytes) / model.mem_bandwidth
        t = max(compute_time, memory_time)
        nodes.append(
            NodeCost(
                id=nd.id,
                prim=_prim_name(nd.prim),
                aval=nd.aval,
                flops=flops,
                out_bytes=out_bytes,
                read_bytes=read_bytes,
                compute_time=compute_time,
                memory_time=memory_time,
                time=t,
                recompute_time=t,
                spill_time=model.spill_time(out_bytes),
            )
        )
        total_flops += flops
        total_time += t

    peak, peak_node = _peak_memory(graph, model)
    over = model.ram_capacity is not None and peak > model.ram_capacity
    return GraphCost(
        nodes=nodes,
        model=model,
        total_flops=total_flops,
        total_compute_time=total_time,
        peak_memory_bytes=peak,
        peak_memory_node=peak_node,
        over_budget=over,
    )


def _prim_name(prim: "Prim") -> str:
    if prim is _INPUT:
        return "input"
    if prim is _CONST:
        return "const"
    name = getattr(prim, "__name__", str(prim))
    return name[2:] if name.startswith("d_") else name.lstrip("_")


# ---------------------------------------------------------------------------
# Human-readable rendering.
# ---------------------------------------------------------------------------
def _fmt_time(s: float) -> str:
    if s >= 1e-3:
        return f"{s * 1e3:.2f}ms"
    if s >= 1e-6:
        return f"{s * 1e6:.2f}us"
    return f"{s * 1e9:.1f}ns"


def _fmt_bytes(b: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= scale:
            return f"{b / scale:.2f}{unit}"
    return f"{b}B"


def pretty_cost(report: GraphCost) -> str:
    """A per-node cost listing plus the whole-graph totals (see :meth:`GraphCost`)."""
    lines = ["cost {"]
    for nc in report.nodes:
        if nc.prim in ("input", "const"):
            continue
        lines.append(
            f"  %{nc.id} = {nc.prim} -> {nc.aval}  "
            f"[{nc.flops:.3g} flops, {_fmt_time(nc.time)}, {_fmt_bytes(nc.out_bytes)}]"
        )
    cap = (
        ""
        if report.model.ram_capacity is None
        else (
            " / cap "
            + _fmt_bytes(report.model.ram_capacity)
            + (" OVER" if report.over_budget else "")
        )
    )
    lines.append(
        f"  total: {report.total_flops:.3g} flops, "
        f"{_fmt_time(report.total_compute_time)} compute, "
        f"peak mem {_fmt_bytes(report.peak_memory_bytes)} @%{report.peak_memory_node}{cap}"
    )
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Host calibration -- micro-benchmark to fill in real constants.
# ---------------------------------------------------------------------------
def calibrate(
    base: CostModel = DEFAULT_COST_MODEL, *, disk_path: Optional[str] = None
) -> CostModel:
    """Micro-benchmark the host and return a :class:`CostModel` with measured constants.

    Measures ``flops_per_sec`` from a square ``matmul`` (``2 n^3`` flops),
    ``mem_bandwidth`` from a large elementwise pass (read + write), and -- when
    ``disk_path`` is given -- ``ssd_*`` from writing/reading a temp file there
    (defaults stay if the probe fails). Fields not measured are copied from ``base``.

    Caveats: the FLOP/bandwidth numbers reflect this host *now* (turbo, load, NUMA);
    the SSD read figure is optimistic when the file is still in the OS page cache.
    Write to a path on the SSD you actually intend to spill to.
    """
    from time import perf_counter

    # CPU FLOP/s: a BLAS matmul dominated by the 2 n^3 multiply-adds.
    n = 1024
    a = np.random.default_rng(0).standard_normal((n, n))
    b = np.random.default_rng(1).standard_normal((n, n))
    a @ b  # warm up BLAS / allocate
    t0 = perf_counter()
    a @ b
    dt = perf_counter() - t0
    flops_per_sec = (2.0 * n**3) / dt if dt > 0 else base.flops_per_sec

    # Memory bandwidth: a large elementwise add streams the input and the output.
    big = np.ones(1 << 24)  # 16M f64 = 128 MB
    big + 1.0  # warm up / page in
    t0 = perf_counter()
    out = big + 1.0
    dt = perf_counter() - t0
    traffic = 2 * big.nbytes  # read `big`, write `out`
    mem_bandwidth = traffic / dt if dt > 0 else base.mem_bandwidth
    del out

    measured = replace(base, flops_per_sec=flops_per_sec, mem_bandwidth=mem_bandwidth)
    if disk_path is not None:
        measured = _calibrate_disk(measured, disk_path)
    return measured


def _calibrate_disk(model: CostModel, disk_path: str) -> CostModel:
    """Fill in ``ssd_*`` from a write/read probe under ``disk_path`` (best effort)."""
    import os
    import tempfile
    from time import perf_counter

    payload = np.random.default_rng(2).standard_normal(1 << 22).tobytes()  # 32 MB
    fd, path = tempfile.mkstemp(dir=disk_path)
    try:
        t0 = perf_counter()
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        wt = perf_counter() - t0
        t0 = perf_counter()
        with open(path, "rb") as fh:
            fh.read()
        rt = perf_counter() - t0
    except OSError:  # pragma: no cover - probe is best effort
        return model
    finally:
        try:
            os.unlink(path)
        except OSError:  # pragma: no cover
            pass
    nbytes = len(payload)
    write_bw = nbytes / wt if wt > 0 else model.ssd_write_bandwidth
    read_bw = nbytes / rt if rt > 0 else model.ssd_read_bandwidth
    return replace(model, ssd_write_bandwidth=write_bw, ssd_read_bandwidth=read_bw)

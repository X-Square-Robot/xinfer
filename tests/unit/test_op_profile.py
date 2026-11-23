"""Operator profiling: attribution of a synthetic kineto trace, the tables, the roofline and the cost recorder."""

import contextlib

import pytest
import torch

from xinfer import ops
from xinfer.observability import annotate
from xinfer.observability.op_cost import OpCost
from xinfer.observability.op_profile import (OUTSIDE, CostRecorder, attribute, coverage, op_rows,
                                             require_cuda_activity, stage_rows)
from xinfer.observability.roofline import PEAKS, Bound, roofline
from xinfer.platforms import resolve_platform

LINEAR = "op::linear x=bfloat16[4,8] weight=bfloat16[16,8]"
NORM = "op::rms_norm x=bfloat16[4,8] weight=float32[8] eps=1e-06"
ENGINE, MAIN, RUNTIME = (1, 10), (1, 20), (1, 11)


def span(name, ts, dur, thread=ENGINE):
    return {"ph": "X", "cat": "user_annotation", "name": name, "pid": thread[0], "tid": thread[1], "ts": ts, "dur": dur}


def launch(correlation, ts, thread=ENGINE, external=None):
    args = {"correlation": correlation} if external is None else {"correlation": correlation, "External id": external}
    return {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": thread[0], "tid": thread[1],
            "ts": ts, "dur": 1, "args": args}


def aten(external, ts, thread=ENGINE):
    return {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "pid": thread[0], "tid": thread[1], "ts": ts, "dur": 1,
            "args": {"External id": external}}


def gpu(correlation, ts, dur, category="kernel"):
    return {"ph": "X", "cat": category, "name": "k", "pid": 0, "tid": 7, "ts": ts, "dur": dur,
            "args": {"correlation": correlation}}


# Engine thread: vision dispatch (100-200 µs on the host) with a linear call (two kernels), an RMSNorm call and a copy
# outside operators; action dispatch (210-260) with a linear call whose two kernels overlap by 5 µs. An upload
# before the stages, a kernel without a launch in the trace, and an op span on another thread that launches nothing.
# The action dispatch's runtime calls carry another thread id, as kineto writes after the thread changed; the aten
# event their External id names maps them back to the spans' thread (one call has no External id, like a graph launch).
EVENTS = [
    span("stage::vision batch_size=1", 100, 100), span(LINEAR, 110, 20), launch(1, 115), launch(2, 120),
    span(NORM, 140, 10), launch(3, 145), launch(4, 160),
    span("stage::action batch_size=1", 210, 50), span(LINEAR, 215, 10), aten(9, 217),
    launch(5, 218, RUNTIME, external=9), launch(6, 220, RUNTIME),
    launch(7, 50), span(LINEAR, 115, 50, MAIN),
    gpu(1, 300, 10), gpu(2, 312, 6), gpu(3, 320, 4), gpu(4, 330, 5, "gpu_memcpy"),
    gpu(5, 400, 10), gpu(6, 405, 10), gpu(7, 60, 3, "gpu_memcpy"), gpu(99, 500, 2),
]
COSTS = {LINEAR: OpCost(2 * 4 * 16 * 8, 4 * 8 * 2 + 16 * 8 * 2, 4 * 16 * 2, "bf16"),
         NORM: OpCost(0, 64 + 32, 64, "memory")}
PEAKS_ = {"bf16": 1e12, "dram": 1e9}


def test_activities_go_to_the_innermost_spans_around_their_launches():
    attribution = attribute(EVENTS)
    assert attribution.dispatches == [("vision", 100, 200), ("action", 210, 260)]
    assert sorted(attribution.calls, key=str) == sorted([(0, LINEAR), (0, NORM), (1, LINEAR), (None, LINEAR)], key=str)
    assert sorted(attribution.work, key=lambda w: w[2]) == [
        (None, None, 60, 3), (0, LINEAR, 300, 10), (0, LINEAR, 312, 6), (0, NORM, 320, 4), (0, None, 330, 5),
        (1, LINEAR, 400, 10), (1, LINEAR, 405, 10), (None, None, 500, 2)]
    assert coverage(attribution) == {"activities": 8, "in_stages": 6, "in_operators": 5, "us": 50,
                                     "in_stages_us": 45, "in_operators_us": 40}


def test_operator_rows_average_per_request_with_bounds():
    measured = op_rows(attribute(EVENTS), 1, COSTS, PEAKS_, {NORM: "selected reference"})
    rows = {(r["stage"], r["op"]): r for r in measured}
    assert list(rows) == [("vision", "linear"), ("vision", "rms_norm"), ("vision", OUTSIDE), ("action", "linear"),
                          ("action", OUTSIDE)]
    linear = rows[("vision", "linear")]
    # 448 bytes at 1 GB/s (0.448 µs) outweigh 1024 FLOPs at 1 TFLOP/s; 16 µs of kernels out of the stage's 25 µs.
    assert linear == pytest.approx({
        "stage": "vision", "op": "linear", "shape": "x=bfloat16[4,8] weight=bfloat16[16,8]", "calls": 1.0,
        "us_per_call": 16.0, "total_us": 16.0, "share": 16 / 25, "gflop_per_call": 1024e-9, "mb_per_call": 448e-6,
        "bound_us_per_call": 0.448, "efficiency": 0.448 / 16, "limiter": "memory", "cost": "formula", "reference": ""})
    assert rows[("vision", "rms_norm")]["reference"] == "selected reference"
    assert rows[("vision", "rms_norm")]["cost"] == "generic" and rows[("vision", "rms_norm")]["share"] == 4 / 25
    outside = rows[("vision", OUTSIDE)]
    assert (outside["calls"], outside["us_per_call"], outside["total_us"], outside["share"], outside["cost"]) == \
        (0, None, 5.0, 0.2, None)
    assert rows[("action", "linear")]["total_us"] == 20.0 and rows[("action", OUTSIDE)]["total_us"] == 0.0
    halved = {(r["stage"], r["op"]): r for r in op_rows(attribute(EVENTS), 2, COSTS, None)}[("vision", "linear")]
    assert (halved["calls"], halved["us_per_call"], halved["total_us"], halved["bound_us_per_call"]) == \
        (0.5, 16.0, 8.0, None)


def test_stage_rows_reconcile_the_gpu_span():
    attribution = attribute(EVENTS)
    vision, action = stage_rows(attribution, 1, replay=attribution)
    assert vision == {"stage": "vision", "dispatches": 1, "host_us": 100, "gpu_span_us": 35, "operators_us": 20,
                      "outside_us": 5, "overlap_us": 0, "idle_us": 10, "graph_gpu_span_us": 35}
    assert action == {"stage": "action", "dispatches": 1, "host_us": 50, "gpu_span_us": 15, "operators_us": 20,
                      "outside_us": 0, "overlap_us": 5, "idle_us": 0, "graph_gpu_span_us": 15}
    for row in (vision, action):
        assert row["operators_us"] + row["outside_us"] - row["overlap_us"] + row["idle_us"] == row["gpu_span_us"]
    assert stage_rows(attribution, 1)[0]["graph_gpu_span_us"] is None


def test_roofline_takes_the_larger_of_compute_and_memory_time():
    compute = roofline(OpCost(2 * 10**12, 10**9, 0, "bf16"), PEAKS_ | {"dram": 1e12, "bf16": 2e14})
    assert compute.limiter == "compute" and compute.seconds == pytest.approx(0.01)  # 10 ms vs 1 ms
    memory = roofline(OpCost(2 * 10**9, 3 * 10**9, 10**9, "bf16"), {"bf16": 2e14, "dram": 1e12})
    assert memory.limiter == "memory" and memory.seconds == pytest.approx(0.004)  # 10 µs vs 4 ms
    assert roofline(OpCost(10**15, 3000, 1000, "memory"), {"dram": 1e12}) == Bound(pytest.approx(4e-9), "memory")
    with pytest.raises(KeyError):
        roofline(OpCost(1, 0, 0, "nvfp4"), {"bf16": 1.0, "dram": 1.0})
    assert set(PEAKS["NVIDIA GeForce RTX 5090"]) == {"bf16", "fp32", "mxfp8", "nvfp4", "dram"}


def test_a_trace_without_cuda_activity_is_an_error():
    with pytest.raises(RuntimeError, match="no CUDA activity.*torch 2.10"):
        require_cuda_activity([span("stage::vision batch_size=1", 0, 1), launch(1, 0)])
    require_cuda_activity(EVENTS)


def test_cost_recorder_keys_costs_by_span_label():
    labels = []

    def sink(name, args):
        labels.append(annotate.label(name, args))
        return contextlib.nullcontext()

    recorder = CostRecorder()
    base = ops.registry.build_table(resolve_platform("cpu"), reference_only=True)
    table = annotate.profiled_ops(base).wrapped(recorder)
    annotate.configure(sink)
    try:
        table.linear(torch.ones(4, 8), torch.ones(16, 8))
        recorder.active = False
        table.linear(torch.ones(2, 8), torch.ones(16, 8))
    finally:
        annotate.configure()
    assert labels[0] == "op::linear x=float32[4,8] weight=float32[16,8]"
    assert recorder.costs == {labels[0]: OpCost(2 * 4 * 16 * 8, (4 * 8 + 16 * 8) * 4, 4 * 16 * 4, "fp32")}

"""Per-operator GPU time from a torch profiler trace, with costs and roofline bounds (docs/profiling.md).

``attribute`` assigns every GPU activity of a kineto Chrome trace (kernels, memory copies and sets) to the innermost
``op::`` span and the ``stage::`` span around its launch: the runtime call that launched it has the same correlation
id and runs on the thread that holds the spans. Kineto may label a thread's CPU events with a stale thread id (one
of an ended thread) while its runtime calls carry the current one; runtime calls whose ``External id`` names a CPU
event map their thread to that event's. ``op_rows`` and ``stage_rows`` turn the attribution into the
per-operator table and the per-stage reconciliation, as averages per request. ``CostRecorder`` collects the
``OpCost`` of every operator call shape while a model runs.
"""

from __future__ import annotations

import functools
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch

from xinfer.observability import annotate
from xinfer.observability.op_cost import FORMULAS, OpCost, op_cost
from xinfer.observability.roofline import roofline
from xinfer.ops import is_slow_fallback
from xinfer.ops.registry import observe_reference_calls

GPU = ("kernel", "gpu_memcpy", "gpu_memset")
LAUNCH = ("cuda_runtime", "cuda_driver")
OUTSIDE = "(outside operators)"


class ReferenceRecorder:
    """Collect the operator and call signature on the eager execution thread of this table."""

    def __init__(self):
        self.seen = set()
        self.references = {}
        self.active = True

    def __call__(self, implementation):
        function, op = implementation.function, implementation.op
        params = annotate.parameter_names(function)
        selected_reference = is_slow_fallback(implementation, "cuda")

        @functools.wraps(function)
        def recorded(*args, **kwargs):
            if not self.active:
                return function(*args, **kwargs)
            self.seen.add(op)

            def reference(source):
                label = annotate.label(f"op::{op}", annotate.call_args(params, args, kwargs))
                self.references[label] = source

            if selected_reference:
                reference("selected reference")
            with observe_reference_calls(reference):
                return function(*args, **kwargs)

        return recorded


class CostRecorder:
    """An ``OpTable.wrapped`` wrapper that records the ``OpCost`` of each operator call shape while ``active``.

    Costs are keyed by the call's span label (``annotate.label``), so they match the ``op::`` spans of a trace.

    Attributes:
        costs: span label -> ``OpCost``, first call of each label.
        active: whether calls are recorded; set False before timing, so calls only pay one attribute check.
    """

    def __init__(self):
        self.costs: dict[str, OpCost] = {}
        self.active = True

    def __call__(self, implementation):
        function, op = implementation.function, implementation.op
        params, name = annotate.parameter_names(function), f"op::{op}"

        @functools.wraps(function)
        def recorded(*args, **kwargs):
            outputs = function(*args, **kwargs)
            if self.active:
                key = annotate.label(name, annotate.call_args(params, args, kwargs))
                if key not in self.costs:
                    self.costs[key] = op_cost(op, args, kwargs, outputs)
            return outputs

        return recorded


@dataclass
class Attribution:
    """GPU activity of a trace by stage dispatch and operator call; times in µs.

    Attributes:
        dispatches: ``(stage, start, end)`` on the host of every ``stage::`` span, in time order.
        calls: ``(dispatch index or None, op label)`` of every ``op::`` span.
        work: ``(dispatch index or None, op label or None, start, duration)`` of every GPU activity; both are None
            for an activity whose launch is not in the trace.
    """

    dispatches: list[tuple[str, float, float]]
    calls: list[tuple[int | None, str]]
    work: list[tuple[int | None, str | None, float, float]]


def require_cuda_activity(events: list[dict]) -> None:
    """Raise ``RuntimeError`` when a trace has CPU events but no GPU activity (CUPTI delivered nothing)."""
    if not any(event.get("cat") in GPU for event in events):
        raise RuntimeError(
            f"torch.profiler recorded {len(events)} events but no CUDA activity: CUPTI delivered no activity records "
            f"to torch {torch.__version__} (CUDA {torch.version.cuda}). torch 2.9.1+cu130 does this; profile with "
            "torch 2.10 or later.")


def _innermost(ranges: list[tuple[float, float]], points: list[float]) -> list[int | None]:
    """For each point, the index of the innermost range containing it (ranges nest properly, as on one thread)."""
    order = sorted(range(len(ranges)), key=lambda i: (ranges[i][0], -ranges[i][1]))
    found, stack, j = [None] * len(points), [], 0
    for p in sorted(range(len(points)), key=points.__getitem__):
        while j < len(order) and ranges[order[j]][0] <= points[p]:
            while stack and ranges[stack[-1]][1] < ranges[order[j]][0]:
                stack.pop()
            stack.append(order[j])
            j += 1
        while stack and ranges[stack[-1]][1] < points[p]:
            stack.pop()
        found[p] = stack[-1] if stack else None
    return found


def attribute(events: list[dict]) -> Attribution:
    """Attribute the GPU activity of a kineto Chrome trace (``traceEvents``) to stage dispatches and operator calls.

    Spans are ``user_annotation`` events named ``stage::<stage> ...`` and ``op::<op> ...``; launches are runtime or
    driver calls with a ``correlation`` id; activities are ``kernel``, ``gpu_memcpy`` and ``gpu_memset`` events.
    """
    spans, launches, activity, cpu_thread = defaultdict(lambda: ([], [])), {}, [], {}
    for event in events:
        category, name = event.get("cat"), event.get("name", "")
        if category == "user_annotation" and name.startswith(("stage::", "op::")):
            spans[(event["pid"], event["tid"])][name.startswith("op::")].append(event)
        elif category in LAUNCH and "correlation" in event.get("args", {}):
            launches[event["args"]["correlation"]] = event
        elif category in GPU:
            activity.append(event)
        if category in ("cpu_op", "user_annotation") and "External id" in event.get("args", {}):
            cpu_thread[event["args"]["External id"]] = (event["pid"], event["tid"])
    votes = defaultdict(Counter)
    for launch in launches.values():
        if launch["args"].get("External id") in cpu_thread:
            votes[(launch["pid"], launch["tid"])][cpu_thread[launch["args"]["External id"]]] += 1
    thread_of = {thread: counter.most_common(1)[0][0] for thread, counter in votes.items()}
    stages = sorted((s for thread in spans.values() for s in thread[0]), key=lambda s: s["ts"])
    index = {id(s): i for i, s in enumerate(stages)}
    launched = defaultdict(list)
    for event in activity:
        launch = launches.get(event.get("args", {}).get("correlation"))
        thread = (launch["pid"], launch["tid"]) if launch else None
        launched[thread_of.get(thread, thread)].append((launch["ts"] if launch else None, event))
    calls, work = [], [(None, None, event["ts"], event["dur"]) for _, event in launched.pop(None, [])]
    for thread in set(spans) | set(launched):
        stage_spans, op_spans = spans.get(thread, ([], []))
        items = launched.get(thread, [])
        times = [t for t, _ in items]
        op_at = _innermost([(s["ts"], s["ts"] + s["dur"]) for s in op_spans], times)
        stage_at = _innermost([(s["ts"], s["ts"] + s["dur"]) for s in stage_spans],
                              times + [s["ts"] for s in op_spans])
        dispatch = [None if i is None else index[id(stage_spans[i])] for i in stage_at]
        for (_, event), o, d in zip(items, op_at, dispatch):
            work.append((d, None if o is None else op_spans[o]["name"], event["ts"], event["dur"]))
        calls.extend(zip(dispatch[len(items):], (s["name"] for s in op_spans)))
    dispatches = [(s["name"].split()[0].removeprefix("stage::"), s["ts"], s["ts"] + s["dur"]) for s in stages]
    return Attribution(dispatches, calls, work)


def coverage(attribution: Attribution) -> dict:
    """Counts and GPU time (µs) of all activities, of those inside a stage span and of those inside an op span."""
    work = attribution.work
    inside = [w for w in work if w[0] is not None]
    under_ops = [w for w in work if w[1] is not None]
    return {"activities": len(work), "in_stages": len(inside), "in_operators": len(under_ops),
            "us": sum(w[3] for w in work), "in_stages_us": sum(w[3] for w in inside),
            "in_operators_us": sum(w[3] for w in under_ops)}


def op_rows(attribution: Attribution, requests: int, costs: dict[str, OpCost], peaks: dict | None,
            references: dict[str, str] | None = None) -> list[dict]:
    """Per stage, one row per operator call shape and one for the activity outside operators; per request.

    Rows are sorted by GPU time within each stage (stages in the order they first run). ``share`` is of the stage's
    GPU time; the bound columns are None without ``peaks``, a cost, or a peak for the cost's pipe.
    """
    references = references or {}
    stage_of = [d[0] for d in attribution.dispatches]
    calls, time, stage_total = defaultdict(int), defaultdict(float), defaultdict(float)
    for dispatch, label in attribution.calls:
        if dispatch is not None:
            calls[(stage_of[dispatch], label)] += 1
    for dispatch, label, _, duration in attribution.work:
        if dispatch is not None:
            time[(stage_of[dispatch], label or OUTSIDE)] += duration
            stage_total[stage_of[dispatch]] += duration
    rows = []
    for stage in dict.fromkeys(stage_of):
        keys = sorted((k for k in set(calls) | set(time) if k[0] == stage and k[1] != OUTSIDE),
                      key=lambda k: -time[k]) + [(stage, OUTSIDE)]
        for key in keys:
            label, n, us = key[1], calls[key], time[key]
            op, _, shape = (OUTSIDE, "", "") if label == OUTSIDE else label.removeprefix("op::").partition(" ")
            cost, bound = costs.get(label), None
            if cost is not None and peaks is not None:
                try:
                    bound = roofline(cost, peaks)
                except KeyError:
                    pass
            per_call = us / n if n else None
            rows.append({
                "stage": stage, "op": op, "shape": shape, "calls": n / requests, "us_per_call": per_call,
                "total_us": us / requests, "share": us / stage_total[stage] if stage_total[stage] else 0.0,
                "gflop_per_call": cost.flops / 1e9 if cost else None, "mb_per_call": cost.bytes / 1e6 if cost else None,
                "bound_us_per_call": bound.seconds * 1e6 if bound else None,
                "efficiency": bound.seconds * 1e6 / per_call if bound and per_call else None,
                "limiter": bound.limiter if bound else None,
                "cost": None if label == OUTSIDE else "formula" if op in FORMULAS else "generic",
                "reference": references.get(label, "")})
    return rows


def stage_rows(attribution: Attribution, requests: int, replay: Attribution | None = None) -> list[dict]:
    """Per stage, per request: host span, GPU span and how operators, outside activity, overlap and idle gaps fill it.

    Within each dispatch, ``operators_us + outside_us - overlap_us + idle_us == gpu_span_us``: overlap is activity
    time that runs concurrently with other activity, idle the time within the GPU span with no activity.
    ``graph_gpu_span_us`` is the stage's GPU span in ``replay`` (the same requests under CUDA graph replay).
    """
    def spans(attr: Attribution):
        out = defaultdict(lambda: defaultdict(float))
        work = defaultdict(list)
        for dispatch, label, start, duration in attr.work:
            if dispatch is not None:
                work[dispatch].append((start, start + duration, label))
        for dispatch, (stage, host_start, host_end) in enumerate(attr.dispatches):
            items, total = sorted(work[dispatch], key=lambda w: w[0]), out[stage]
            total["dispatches"] += 1
            total["host_us"] += host_end - host_start
            if not items:
                continue
            busy, end = 0.0, items[0][0]
            for start, stop, _ in items:
                busy += max(0.0, stop - max(start, end))
                end = max(end, stop)
            ops = sum(stop - start for start, stop, label in items if label is not None)
            outside = sum(stop - start for start, stop, label in items if label is None)
            total["gpu_span_us"] += end - items[0][0]
            total["operators_us"] += ops
            total["outside_us"] += outside
            total["overlap_us"] += ops + outside - busy
            total["idle_us"] += end - items[0][0] - busy
        return out

    eager, graphs = spans(attribution), spans(replay) if replay is not None else {}
    return [{"stage": stage, **{k: v / requests for k, v in values.items()},
             "graph_gpu_span_us": graphs[stage]["gpu_span_us"] / requests if stage in graphs else None}
            for stage, values in eager.items()]

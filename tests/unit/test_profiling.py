"""Annotation spans and profiled op tables (docs/profiling.md)."""

import contextlib
import inspect

import pytest
import torch

from xinfer import ops
from xinfer.models.layers.basic import RMSNorm
from xinfer.observability import annotate
from xinfer.ops import Contract, OpRegistry, OpSpec, use_ops
from xinfer.platforms import resolve_platform


def _recorder(events, tag=None):
    """A sink appending ``(tag, "enter", name, args)`` and ``(tag, "exit", name)`` to ``events``."""

    @contextlib.contextmanager
    def sink(name, args):
        events.append((tag, "enter", name, dict(args)))
        yield
        events.append((tag, "exit", name))

    return sink


@pytest.fixture
def events():
    recorded = []
    annotate.configure(_recorder(recorded))
    yield recorded
    annotate.configure()


def _registry():
    """``mm`` with a backend that logs its calls."""
    reg, calls = OpRegistry(), []
    reg.define(OpSpec("mm", "mm(x, w, *, scale=1.0)", lambda x, w, *, scale=1.0: x @ w * scale))

    def fast(x, w, *, scale=1.0):
        calls.append("fast")
        return x @ w * scale

    reg.add("mm", fast, backend="fast", contract=Contract(numerics="tolerance"), priority=5)

    return reg, fast, calls


def test_plain_tables_bind_the_selected_functions_without_a_wrapper():
    reg, fast, _ = _registry()
    assert reg.build_table(resolve_platform("cpu")).mm is fast
    reference = ops.registry.build_table(resolve_platform("cpu"), reference_only=True)
    for row in reference.report():
        name = row["op"]
        assert getattr(reference, name) is ops.registry.spec(name).reference, name


def test_profiled_tables_call_the_same_implementations_inside_spans(events):
    reg, fast, calls = _registry()
    table = reg.build_table(resolve_platform("cpu"))
    profiled = annotate.profiled_ops(table)
    x, w = torch.ones(2, 3), torch.ones(3, 4)
    assert torch.equal(profiled.mm(x, w, scale=2.0), torch.full((2, 4), 6.0))
    assert torch.equal(profiled.mm(x, w=w), torch.full((2, 4), 3.0))
    assert calls == ["fast", "fast"]
    assert events == [(None, "enter", "op::mm", {"x": "float32[2,3]", "w": "float32[3,4]", "scale": 2.0}),
                      (None, "exit", "op::mm"),
                      (None, "enter", "op::mm", {"x": "float32[2,3]", "w": "float32[3,4]"}),
                      (None, "exit", "op::mm")]
    assert table.mm is fast  # the plain table is unchanged
    assert profiled.report() == table.report()
    assert inspect.signature(profiled.mm) == inspect.signature(fast)


def test_sequences_of_tensors_are_described_by_dtype_and_shape(events):
    reg = OpRegistry()
    reg.define(OpSpec("total", "total(parts, bounds)", lambda parts, bounds: sum(p.sum() for p in parts)))
    profiled = annotate.profiled_ops(reg.build_table(resolve_platform("cpu")))
    profiled.total((torch.ones(2, 3), torch.ones(4, dtype=torch.int64)), [0, 3, 7])
    assert events[0] == (None, "enter", "op::total", {"parts": "[float32[2,3],int64[4]]", "bounds": "[0,3,7]"})


def test_layers_built_under_a_profiled_table_emit_operator_spans(events):
    table = ops.registry.build_table(resolve_platform("cpu"), reference_only=True)
    with use_ops(table):
        assert RMSNorm(8, 1e-6)._rms_norm is ops.reference.rms_norm
    with use_ops(annotate.profiled_ops(table)):
        norm = RMSNorm(8, 1e-6)
    norm.weight = torch.nn.Parameter(torch.ones(8), requires_grad=False)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    assert torch.equal(norm(x), ops.reference.rms_norm(x, norm.weight, 1e-6))
    assert events[0] == (None, "enter", "op::rms_norm", {"x": "bfloat16[2,8]", "weight": "float32[8]", "eps": 1e-6})


def test_spans_without_sinks_share_one_no_op_context():
    annotate.configure()
    assert annotate.span("a", n=1) is annotate.span("b")


def test_spans_enter_sinks_in_order_and_exit_in_reverse():
    recorded = []
    annotate.configure(_recorder(recorded, "first"), _recorder(recorded, "second"))
    try:
        with annotate.span("region", n=1, kind="a"):
            recorded.append("body")
    finally:
        annotate.configure()
    args = {"n": 1, "kind": "a"}
    assert recorded == [("first", "enter", "region", args), ("second", "enter", "region", args), "body",
                        ("second", "exit", "region"), ("first", "exit", "region")]


def test_label_carries_the_arguments():
    assert annotate.label("op::linear", {"x": "bfloat16[2,3]", "eps": 1e-6}) == "op::linear x=bfloat16[2,3] eps=1e-06"
    assert annotate.label("region", {}) == "region"


def test_kineto_sink_names_a_profiler_user_annotation():
    annotate.configure(annotate.kineto)
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            with annotate.span("op::linear", x="bfloat16[2,3]"):
                torch.ones(2).add_(1)
    finally:
        annotate.configure()
    assert "op::linear x=bfloat16[2,3]" in {event.name for event in prof.events()}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nvtx_sink_pushes_and_pops_a_range():
    annotate.configure(annotate.nvtx)
    try:
        with annotate.span("op::linear", x="bfloat16[2,3]"):
            torch.ones(2, device="cuda").add_(1)
    finally:
        annotate.configure()

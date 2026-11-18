"""Configuration reports the model's effective precision, independent of its op binding."""

from dataclasses import replace
import importlib
from types import SimpleNamespace

import pytest

from xinfer.models.interface import Capabilities, Stage, StageGraph
from xinfer.ops.registry import OpTable
from xinfer.platforms.cpu import cpu_platform


@pytest.mark.parametrize(("selection", "effective", "options", "expected"), [
    ("model-bound", "bf16", {}, "bf16"),
    ("model-bound", "fp16", {}, "fp16"),
    ("table", "fp16", {}, "fp16"),
    ("model-bound", "bf16", {"quantization": "auto"}, "bf16"),
    ("table", None, {"quantization": "mxfp8"}, "mxfp8"),
    ("model-bound", None, {}, "bf16"),
])
def test_policy_reports_effective_precision_without_choosing_a_model_default(
        monkeypatch, selection, effective, options, expected):
    from xinfer.entrypoints import policy as policy_module

    platform = cpu_platform()
    platform = replace(platform, capabilities=replace(platform.capabilities, op_selection=selection))
    graph = StageGraph(Stage("step", "step"))
    model = SimpleNamespace(
        stage_graph=graph,
        capabilities=lambda: Capabilities(batch_sizes={"step": (1,)}, num_steps=1),
        shape_buckets=lambda: {}, io_schema=lambda: None,
        input_processor=lambda: None, output_processor=lambda: None,
    )
    if effective is not None:
        model.quantization = effective
    loaded_options, tuned_precisions = [], []

    def load(checkpoint, *, device, ops, **model_options):
        loaded_options.append(model_options)
        return model

    def tuned(model_name, arch, quantization):
        tuned_precisions.append(quantization)
        return None

    monkeypatch.setattr(policy_module, "resolve_platform", lambda device: platform)
    monkeypatch.setattr(policy_module, "get_model_class", lambda name: SimpleNamespace(from_checkpoint=load))
    monkeypatch.setattr(policy_module.ops.registry, "load_plugins", lambda: None)
    monkeypatch.setattr(policy_module.ops.registry, "build_table", lambda *args, **kwargs: OpTable(platform.name, {}))
    monkeypatch.setattr(policy_module._engine_factory, "create_engine",
                        lambda *args: SimpleNamespace(shutdown=lambda: None))
    monkeypatch.setattr(importlib.import_module("xinfer.config.resolve"), "find_tuned", tuned)

    policy = policy_module.Policy("test", "unused", device="cpu", **options)
    try:
        assert loaded_options == [options]
        assert tuned_precisions == [expected]
    finally:
        policy.close()


def test_policy_routes_rpu_profile_defaults_to_the_rpu_engine(monkeypatch):
    from xinfer.entrypoints import policy as policy_module
    from xinfer.engine.rpu import RpuEngine

    graph = StageGraph(Stage("step", "step"))
    model = SimpleNamespace(
        stage_graph=graph,
        capabilities=lambda: Capabilities(batch_sizes={"step": (1,)}, num_steps=1),
        shape_buckets=lambda: {}, io_schema=lambda: None,
        input_processor=lambda: None, output_processor=lambda: None,
        create_rpu_request=lambda prepared, context: None,
    )
    closed, tuned_precisions = [], []
    context = SimpleNamespace(close=lambda: closed.append(True))

    def load(checkpoint, *, device, ops, quantization="fp16", **options):
        assert device == "rpu:0"
        model.quantization = quantization
        return model

    def tuned(model_name, arch, quantization):
        tuned_precisions.append(quantization)
        return None

    monkeypatch.setattr(policy_module, "get_model_class", lambda name: SimpleNamespace(from_checkpoint=load))
    monkeypatch.setattr(policy_module._engine_factory, "_rpu_context",
                        lambda **kwargs: context)
    monkeypatch.setattr(importlib.import_module("xinfer.config.rpu"), "find_tuned", tuned)

    with policy_module.Policy("test", "unused", device="rpu") as policy:
        assert isinstance(policy.engine, RpuEngine)
        assert model.quantization == "fp16" and tuned_precisions == ["fp16"]
        assert policy.engine.config.max_running_requests == policy.engine.config.max_queued_requests == 1
        assert policy.engine.config.lanes == {"step": 1}
        assert policy.engine.config.max_batch == {"step": 1}
    assert closed == [True]

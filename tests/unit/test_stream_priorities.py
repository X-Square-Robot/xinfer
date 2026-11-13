"""Stream priority routing follows capabilities and preserves exact requested values."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from xinfer.config import ConfigError, ExplicitSettings, resolve
from xinfer.engine.executor import Lane
from xinfer.models.interface import Capabilities, Stage, StageGraph, VLAModel
from xinfer.platforms.interface import Platform, PlatformCapabilities
from xinfer.platforms.npu import npu_stream_priority_range


class Pipeline(VLAModel):
    stage_graph = StageGraph(Stage("input", "step"), Stage("output", "step", depends_on="input"))

    def capabilities(self):
        return Capabilities(batch_sizes={}, num_steps=1)


def _platform(*, supported=True, bounds=(-2, 0)):
    caps = PlatformCapabilities("ascend", "test", False, supported, False, None, None, None,
                                stream_priority_range=bounds)
    return Platform("ascend-test", "npu:0", caps)


def test_non_cuda_defaults_and_explicit_values_follow_platform_capabilities():
    platform = _platform()
    default = resolve(Pipeline(), platform, model_name="test", tuned=False)
    assert default.engine_config().stream_priorities == {"input": 0, "output": -1}
    configured = resolve(Pipeline(), platform, model_name="test", tuned=False,
                         explicit=ExplicitSettings(stream_priorities={"output": -1}))
    assert configured.engine_config().stream_priorities["output"] == -1
    positive = replace(platform, capabilities=replace(platform.capabilities, stream_priority_range=(0, 7)))
    assert resolve(Pipeline(), positive, model_name="test", tuned=False).engine_config().stream_priorities == {
        "input": 0, "output": 0}
    assert resolve(Pipeline(), positive, model_name="test", tuned=False,
                   explicit=ExplicitSettings(stream_priorities={"input": 7})).engine_config().stream_priorities == {
        "input": 7, "output": 0}
    with pytest.raises(ConfigError, match="outside"):
        resolve(Pipeline(), positive, model_name="test", tuned=False,
                explicit=ExplicitSettings(stream_priorities={"output": -1}))


@pytest.mark.parametrize(("bounds", "priority"), [((-2, 0), -2), ((0, 7), 7)])
def test_lane_passes_non_cuda_priorities_and_keeps_current_stream_ordering(monkeypatch, bounds, priority):
    created, waited = [], []
    current = object()

    def stream(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(priority=kwargs["priority"], wait_stream=waited.append)

    module = SimpleNamespace(Stream=stream, current_stream=lambda device: current)
    monkeypatch.setattr("xinfer.engine.executor.device_module_for", lambda device: module)
    lane = Lane("output", 0, _platform(bounds=bounds), priority=priority)
    assert lane.stream.priority == priority
    assert created and all(item["priority"] == priority for item in created)
    assert current in waited


@pytest.mark.parametrize("priority", [-3, 1, True, 0.5])
def test_invalid_priority_is_rejected_before_creating_a_stream(monkeypatch, priority):
    module = SimpleNamespace(Stream=lambda **kwargs: pytest.fail("stream created"))
    monkeypatch.setattr("xinfer.engine.executor.device_module_for", lambda device: module)
    with pytest.raises(ValueError):
        Lane("output", 0, _platform(), priority=priority)


def test_non_cuda_clamping_and_unverifiable_priority_are_rejected(monkeypatch):
    stream = SimpleNamespace(priority=0)
    module = SimpleNamespace(Stream=lambda **kwargs: stream)
    monkeypatch.setattr("xinfer.engine.executor.device_module_for", lambda device: module)
    with pytest.raises(ValueError, match="clamp"):
        Lane("output", 0, _platform(), priority=-1)

    class Unverifiable:
        @property
        def priority(self):
            raise RuntimeError("not supported")

    stream = Unverifiable()
    with pytest.raises(ValueError, match="cannot verify"):
        Lane("output", 0, _platform(), priority=-1)


def test_unsupported_non_cuda_runtime_keeps_default_and_rejects_nonzero(monkeypatch):
    created = []
    stream = SimpleNamespace(wait_stream=lambda current: None)

    def create(**kwargs):
        created.append(kwargs)
        return stream

    module = SimpleNamespace(Stream=create, current_stream=lambda device: None)
    monkeypatch.setattr("xinfer.engine.executor.device_module_for", lambda device: module)
    platform = _platform(supported=False, bounds=None)
    Lane("output", 0, platform)
    assert created and all(item.get("priority", 0) == 0 for item in created)
    with pytest.raises(ValueError, match="does not support nonzero"):
        Lane("output", 0, platform, priority=-1)


def _runtime(bounds=(0, -1), priority=0):
    return SimpleNamespace(device=lambda index: nullcontext(), Stream=SimpleNamespace(priority_range=lambda: bounds),
                           current_stream=lambda index: SimpleNamespace(priority=priority))


def test_npu_priority_capability_requires_working_range_and_readback():
    assert npu_stream_priority_range(_runtime(), 0) == (-1, 0)
    assert npu_stream_priority_range(_runtime((7, 0)), 0) == (0, 7)
    assert npu_stream_priority_range(_runtime((0, 0)), 0) is None
    assert npu_stream_priority_range(_runtime((0, -1), priority=3), 0) is None
    assert npu_stream_priority_range(_runtime((False, -1)), 0) is None

    def unavailable():
        raise RuntimeError("NPU does not support Stream.priority_range() currently")

    runtime = _runtime()
    runtime.Stream.priority_range = unavailable
    assert npu_stream_priority_range(runtime, 0) is None


@pytest.mark.parametrize("failure", ["device_enter", "device_exit", "current_stream"])
def test_npu_priority_probe_propagates_device_errors(failure):
    runtime = _runtime()

    @contextmanager
    def device(index):
        if failure == "device_enter":
            raise RuntimeError("device binding failed")
        yield
        if failure == "device_exit":
            raise RuntimeError("device restoration failed")

    def current_stream(index):
        raise RuntimeError("device stream lookup failed")

    runtime.device = device
    if failure == "current_stream":
        runtime.current_stream = current_stream
    with pytest.raises(RuntimeError, match="device .* failed"):
        npu_stream_priority_range(runtime, 0)


def test_unknown_priority_range_does_not_invent_a_final_stage_priority():
    config = resolve(Pipeline(), _platform(bounds=None), model_name="test", tuned=False)
    assert config.engine_config().stream_priorities == {"input": 0, "output": 0}


def test_cuda_capability_uses_the_active_device_range(monkeypatch):
    import torch
    from xinfer.platforms.cuda import cuda_platform

    visited = []

    def device(index):
        visited.append(index)
        return nullcontext()

    props = SimpleNamespace(major=12, minor=0, multi_processor_count=170, total_memory=32 << 30)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props)
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "Stream", SimpleNamespace(priority_range=lambda: (0, -2)))
    platform = cuda_platform("cuda:2")
    assert platform.capabilities.stream_priorities and platform.capabilities.stream_priority_range == (-2, 0)
    assert 2 in visited
    config = resolve(Pipeline(), platform, model_name="test", tuned=False)
    assert config.engine_config().stream_priorities == {"input": 0, "output": -1}

"""RPU request limits are isolated from generic configuration resolution."""

from dataclasses import replace

import pytest

from xinfer.config import ConfigError, ExplicitSettings, TunedTable, resolve
from xinfer.config.rpu import resolve as resolve_rpu
from xinfer.entrypoints._factory import resolve_config
from xinfer.models.interface import Capabilities, Stage, StageGraph, VLAModel
from xinfer.platforms import resolve_platform
from xinfer.platforms.interface import Platform, PlatformCapabilities


class Model(VLAModel):
    stage_graph = StageGraph(Stage("vision", "vision"), Stage("action", "action", depends_on="vision"))

    def capabilities(self):
        return Capabilities(batch_sizes={"vision": (1,), "action": (1,)}, num_steps=1)


def _resolve(**kwargs):
    return resolve_config(Model(), resolve_platform("rpu"), model_name="test", cpus=32, **kwargs)


def test_rpu_defaults_keep_backend_model_and_platform_sources():
    result = _resolve(tuned=None)
    for name in ("max_running_requests", "max_queued_requests", "per_robot_inflight", "lanes.vision", "lanes.action"):
        assert result.value(name) == 1
        assert result.settings[name].source == "platform:single-submission rpu-tp8"
    assert result.settings["batch_sizes.action"].source == "model:declares B1 only"
    assert result.value("batch_sizes.action") == [1]
    assert result.value("stream_priorities.action") == 0
    assert result.settings["cudagraphs"].source == "platform:lacks graph capture"
    assert result.engine_config().lanes == {"vision": 1, "action": 1}


def test_rpu_keeps_generic_frontend_and_shared_memory_defaults():
    result = _resolve(tuned=None)
    assert result.value("frontends") == 2
    assert result.settings["frontends"].source == "default, ≤ cpus/2 = 16"
    assert result.value("frontend_workers") == 4
    assert result.value("shm_buffers") == 1
    assert result.settings["shm_buffers"].source == "default:one per running request"


def test_rpu_accepts_explicit_frontend_and_shared_memory_counts():
    explicit = ExplicitSettings(frontends=3, frontend_workers=5, shm_buffers=7)
    result = _resolve(explicit=explicit, tuned=None)
    for name in ("frontends", "frontend_workers", "shm_buffers"):
        assert result.value(name) == getattr(explicit, name)
        assert result.settings[name].source == "explicit"


def test_rpu_preserves_explicit_and_tuned_sources_without_mutating_inputs():
    explicit = ExplicitSettings(max_running_requests=1, max_queued_requests=1, per_robot_inflight=1,
                                lanes={"action": 1})
    table = TunedTable("test/rpu-tp8-fp16", lanes={"vision": 1, "action": 1}, measured={"commit": "measured"})
    result = _resolve(explicit=explicit, tuned=table)
    for name in ("max_running_requests", "max_queued_requests", "per_robot_inflight", "lanes.action"):
        assert result.value(name) == 1
        assert result.settings[name].source == "explicit"
    assert result.settings["lanes.vision"].source == table.source
    assert result.tuned is table
    assert explicit.lanes == {"action": 1}
    assert table.lanes == {"vision": 1, "action": 1}


def test_rpu_looks_up_tuned_lanes_and_only_defaults_missing_stages(monkeypatch):
    import xinfer.config.rpu as rpu

    table = TunedTable("test/rpu-tp8-fp16", lanes={"vision": 1}, measured={"commit": "measured"})
    calls = []

    def lookup(*args):
        calls.append(args)
        return table

    monkeypatch.setattr(rpu, "find_tuned", lookup)
    result = _resolve(quantization="fp16")
    assert calls == [("test", "rpu-tp8", "fp16")]
    assert result.settings["lanes.vision"].source == table.source
    assert result.settings["lanes.action"].source == "platform:single-submission rpu-tp8"
    assert result.tuned is table


@pytest.mark.parametrize("name", ["max_running_requests", "max_queued_requests", "per_robot_inflight", "lanes"])
@pytest.mark.parametrize("value", [-1, 0, 2, True, 1.0, "1"])
def test_rpu_rejects_invalid_explicit_concurrency(name, value):
    explicit = ExplicitSettings(**{name: {"action": value} if name == "lanes" else value})
    with pytest.raises(ConfigError, match="requires .*1"):
        _resolve(explicit=explicit, tuned=None)


@pytest.mark.parametrize("value", [-1, 0, 2, True, 1.0, "1"])
def test_rpu_rejects_invalid_effective_tuned_lanes(value):
    table = TunedTable("test/rpu-tp8-fp16", lanes={"action": value})
    with pytest.raises(ConfigError, match="requires tuned lanes.action=1"):
        _resolve(tuned=table)


def test_rpu_explicit_lanes_override_an_unused_invalid_tuned_value():
    table = TunedTable("test/rpu-tp8-fp16", lanes={"vision": 1, "action": 2}, measured={"commit": "measured"})
    result = _resolve(explicit=ExplicitSettings(lanes={"action": 1}), tuned=table)
    assert result.value("lanes.action") == 1
    assert result.settings["lanes.action"].source == "explicit"
    assert result.settings["lanes.vision"].source == table.source
    assert result.tuned is table
    assert table.lanes["action"] == 2


@pytest.mark.parametrize("explicit, message", [
    (ExplicitSettings(max_batch={"action": 2}), "supports batch sizes"),
    (ExplicitSettings(stream_priorities={"action": -1}), "does not support nonzero"),
    (ExplicitSettings(cudagraphs=True), "does not support CUDA graph capture"),
    (ExplicitSettings(lanes={"unknown": 1}), "unknown stage"),
    (ExplicitSettings(frontends=0), "frontends must be at least 1"),
    (ExplicitSettings(shm_buffers=0), "shm_buffers must be at least 1"),
])
def test_rpu_reuses_generic_model_and_platform_validation(explicit, message):
    with pytest.raises(ConfigError, match=message):
        _resolve(explicit=explicit, tuned=None)


@pytest.mark.parametrize("kind, device", [("cpu", "cpu"), ("cuda", "cuda:0"), ("ascend", "npu:0")])
def test_other_execution_backends_keep_generic_settings(kind, device):
    caps = PlatformCapabilities(kind, None, False, False, False, None, None, None)
    platform = Platform(f"{kind}-test", device, caps)
    explicit = ExplicitSettings(max_running_requests=4, max_queued_requests=9, per_robot_inflight=2,
                                lanes={"action": 3}, frontends=3, shm_buffers=5)
    table = TunedTable("test/device-fp16", lanes={"vision": 4})
    expected = resolve(Model(), platform, model_name="test", cpus=32, explicit=explicit, tuned=table)
    actual = resolve_config(Model(), platform, model_name="test", cpus=32, explicit=explicit, tuned=table)
    assert actual.settings == expected.settings
    assert actual.tuned is table
    assert actual.value("lanes.vision") == 4
    assert actual.value("lanes.action") == 3
    assert actual.value("max_running_requests") == 4


def test_configuration_dispatch_rejects_unknown_execution_models():
    platform = resolve_platform("rpu")
    platform = replace(platform, capabilities=replace(platform.capabilities, execution_model="unknown"))
    with pytest.raises(ValueError, match="unknown execution_model 'unknown'"):
        resolve_config(Model(), platform, model_name="test")


def test_rpu_adapter_rejects_another_execution_model():
    with pytest.raises(ConfigError, match="requires execution_model='explicit-completion'"):
        resolve_rpu(Model(), resolve_platform("cpu"), model_name="test")

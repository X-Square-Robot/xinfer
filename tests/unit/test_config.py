"""Configuration resolution: explicit > tuned table > default (docs/configuration.md)."""

import json

import pytest

from xinfer.config import ConfigError, ExplicitSettings, TunedTable, load_tuned, resolve
from xinfer.config.tuned import TUNED_ROOT, find_tuned
from xinfer.models.interface import BucketSpec, Capabilities, Stage, StageGraph, VLAModel
from xinfer.platforms.interface import Platform, PlatformCapabilities


class ThreeStage(VLAModel):
    stage_graph = StageGraph(Stage("vision", "f"), Stage("prefill", "f", depends_on="vision"),
                             Stage("action", "f", depends_on="prefill"))

    def __init__(self, invariant=False):
        self.invariant = invariant

    def capabilities(self):
        return Capabilities(num_steps=5, batch_sizes={"vision": (1,), "prefill": (1,), "action": (1, 2, 4)},
                            batch_invariant={"vision": True, "prefill": True, "action": self.invariant})


def _platform(cudagraphs=True, arch="sm120"):
    capabilities = PlatformCapabilities(device_type="cuda" if cudagraphs else "cpu", arch=arch, cudagraphs=cudagraphs,
                        stream_priorities=cudagraphs, unified_memory=False, multiprocess_service=None, sm_count=170,
                        total_memory_bytes=32 << 30)
    return Platform(name=f"cuda-{arch}" if cudagraphs else "cpu", device="cuda:0" if cudagraphs else "cpu",
                    capabilities=capabilities)


TABLE = TunedTable("m/sm120-mxfp8", lanes={"vision": 2, "prefill": 3, "action": 2},
                   measured={"commit": "abc1234", "date": "2026-11-01", "method": "test", "evidence": "test"})


def test_bucket_policy_is_reported():
    class Bucketed(ThreeStage):
        def shape_buckets(self):
            return {"prompt": BucketSpec(16, max_length=200)}

    result = resolve(Bucketed(), _platform(), model_name="m", tuned=None)
    assert result.value("shape_buckets.prompt")["granularity"] == 16
    assert result.value("shape_buckets.prompt")["max_length"] == 200
    assert "shape_buckets.prompt" in result.report()


def test_defaults_do_not_depend_on_the_load():
    r = resolve(ThreeStage(), _platform(), model_name="m", tuned=None, cpus=24)
    assert r.value("per_robot_inflight") == 1 and r.settings["per_robot_inflight"].source == "default"
    assert r.value("max_running_requests") == 32 and r.value("max_queued_requests") == 256
    assert all(r.value(f"lanes.{s}") == 2 and r.settings[f"lanes.{s}"].source == "default"
               for s in ("vision", "prefill", "action"))
    assert r.value("cudagraphs") is True and r.settings["cudagraphs"].source == "platform:supports graph capture"
    assert r.value("frontends") == 2 and r.settings["frontends"].source == "default, ≤ cpus/2 = 12"
    assert r.value("frontend_workers") == 4
    assert r.settings["frontend_workers"].source == "default, frontends × workers ≤ cpus/2 = 12"
    assert r.value("shm_buffers") == 16  # one per running request: ceil(32 / 2)
    cpu = resolve(ThreeStage(), _platform(cudagraphs=False), model_name="m", tuned=None, cpus=2)
    assert cpu.value("cudagraphs") is False and cpu.settings["cudagraphs"].source == "platform:lacks graph capture"
    assert cpu.value("frontends") == 1 and cpu.value("shm_buffers") == 32  # at most half the CPUs
    assert cpu.value("frontend_workers") == 1
    assert resolve(ThreeStage(), _platform(), model_name="m", tuned=None, cpus=8).value("frontend_workers") == 2


def test_the_tuned_table_sets_the_lanes():
    r = resolve(ThreeStage(), _platform(), model_name="m", tuned=TABLE, cpus=24)
    assert (r.value("lanes.vision"), r.value("lanes.prefill"), r.value("lanes.action")) == (2, 3, 2)
    assert r.settings["lanes.prefill"].source == "tuned:m/sm120-mxfp8@abc1234"
    assert r.value("batch_sizes.action") == [1]
    assert r.settings["batch_sizes.action"].source == "model:not batch-invariant"
    engine_config = r.engine_config()
    assert engine_config.lanes == {"vision": 2, "prefill": 3, "action": 2}
    assert engine_config.max_running_requests == 32 and engine_config.cudagraphs is not None
    assert engine_config.max_batch == {"vision": 1, "prefill": 1, "action": 1}


def test_batch_invariant_stages_allow_every_declared_size():
    r = resolve(ThreeStage(invariant=True), _platform(), model_name="m", tuned=None)
    assert r.value("batch_sizes.action") == [1, 2, 4]
    assert r.settings["batch_sizes.action"].source == "model:batch-invariant, declared sizes"
    assert r.value("batch_sizes.vision") == [1] and r.settings["batch_sizes.vision"].source == "model:declares B1 only"
    assert r.engine_config().max_batch["action"] == 4


def test_explicit_values_win_and_are_checked():
    ex = ExplicitSettings(lanes={"action": 4}, max_running_requests=5, cudagraphs=False, frontends=3,
                          frontend_workers=6, shm_buffers=7)
    r = resolve(ThreeStage(), _platform(), model_name="m", explicit=ex, tuned=TABLE)
    assert r.value("lanes.action") == 4 and r.settings["lanes.action"].source == "explicit"
    assert r.value("lanes.vision") == 2  # others still resolved
    values = [r.value(k) for k in ("max_running_requests", "cudagraphs", "frontends", "frontend_workers",
                                   "shm_buffers")]
    assert values == [5, False, 3, 6, 7]
    with pytest.raises(ConfigError, match="not batch-invariant"):
        resolve(ThreeStage(), _platform(), model_name="m", tuned=None,
                explicit=ExplicitSettings(max_batch={"action": 2}))
    r = resolve(ThreeStage(), _platform(), model_name="m", tuned=None,
                explicit=ExplicitSettings(max_batch={"action": 2}, allow_non_invariant_batching=True))
    assert r.value("batch_sizes.action") == [1, 2] and r.value("allow_non_invariant_batching") is True
    with pytest.raises(ConfigError, match="supports batch sizes"):
        resolve(ThreeStage(invariant=True), _platform(), model_name="m",
                explicit=ExplicitSettings(max_batch={"action": 8}))
    with pytest.raises(ConfigError, match="unknown stage"):
        resolve(ThreeStage(), _platform(), model_name="m", explicit=ExplicitSettings(lanes={"decode": 2}))
    with pytest.raises(ConfigError, match="graph capture"):
        resolve(ThreeStage(), _platform(cudagraphs=False), model_name="m", explicit=ExplicitSettings(cudagraphs=True))
    with pytest.raises(ConfigError, match="at least 1"):
        resolve(ThreeStage(), _platform(), model_name="m", explicit=ExplicitSettings(max_running_requests=0))


def test_report_and_json(tmp_path):
    r = resolve(ThreeStage(), _platform(), model_name="m", tuned=TABLE)
    text = r.report()
    assert "lanes.prefill" in text and "tuned:m/sm120-mxfp8@abc1234" in text
    r.write(tmp_path / "resolved.json", model="m")
    data = json.loads((tmp_path / "resolved.json").read_text())
    expected = {"name": "max_running_requests", "value": 32, "source": "default"}
    assert data["model"] == "m" and expected in data["settings"]


def test_tuned_tables_load_with_provenance(tmp_path):
    for path in sorted(TUNED_ROOT.glob("*/*.json")):  # bundled tables, if any
        t = load_tuned(path)
        assert t.measured["commit"] and t.measured["evidence"]
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "sm120-mxfp8.json").write_text(json.dumps({
        "model": "m", "arch": "sm120", "quantization": "mxfp8", "lanes": {"vision": 2, "prefill": 2, "action": 2},
        "measured": {"commit": "abc1234", "date": "2026-11-01", "method": "test", "evidence": "none"}}))
    assert find_tuned("m", "sm120", "mxfp8", root=tmp_path).lanes == {"vision": 2, "prefill": 2, "action": 2}
    assert find_tuned("m", "sm110", "mxfp8", root=tmp_path) is None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"model": "m", "arch": "sm120", "quantization": "bf16", "measured": {"commit": "x"}}))
    with pytest.raises(ValueError, match="measured"):
        load_tuned(bad)


def test_stream_priorities_put_the_final_stage_first_and_preserve_explicit_values(tmp_path):
    default = resolve(ThreeStage(), _platform(), model_name="m", tuned=None)
    assert default.engine_config().stream_priorities == {"vision": 0, "prefill": 0, "action": -1}
    assert default.settings["stream_priorities.action"].source == "default:final stage"
    cpu = resolve(ThreeStage(), _platform(cudagraphs=False), model_name="m", tuned=None)
    assert cpu.engine_config().stream_priorities == {"vision": 0, "prefill": 0, "action": 0}
    equal = resolve(ThreeStage(), _platform(), model_name="m", tuned=None,
                    explicit=ExplicitSettings(stream_priorities={"action": 0}))
    assert equal.engine_config().stream_priorities == {"vision": 0, "prefill": 0, "action": 0}
    configured = resolve(ThreeStage(), _platform(), model_name="m", tuned=None,
                         explicit=ExplicitSettings(stream_priorities={"action": -1, "vision": 0}))
    assert configured.engine_config().stream_priorities == {"vision": 0, "prefill": 0, "action": -1}
    assert configured.settings["stream_priorities.action"].source == "explicit"
    assert configured.settings["stream_priorities.vision"].source == "explicit"
    assert configured.settings["stream_priorities.prefill"].source == "default"
    configured.write(tmp_path / "resolved.json")
    rows = json.loads((tmp_path / "resolved.json").read_text())["settings"]
    assert {"name": "stream_priorities.action", "value": -1, "source": "explicit"} in rows
    assert "stream_priorities.action" in configured.report()


@pytest.mark.parametrize("priorities,platform,message", [
    ({"unknown": -1}, _platform(), "unknown stage"),
    ({"action": 0.5}, _platform(), "must be an integer"),
    ({"action": True}, _platform(), "must be an integer"),
    ({"action": -1}, _platform(cudagraphs=False), "does not support nonzero stream priorities"),
])
def test_invalid_stream_priorities_are_rejected(priorities, platform, message):
    with pytest.raises(ConfigError, match=message):
        resolve(ThreeStage(), platform, model_name="m", tuned=None,
                explicit=ExplicitSettings(stream_priorities=priorities))


def test_policy_consumes_stream_priorities_as_runtime_settings(monkeypatch):
    from types import SimpleNamespace
    import xinfer.entrypoints.policy as policy_module

    model = SimpleNamespace(stage_graph=ThreeStage.stage_graph, capabilities=ThreeStage().capabilities,
                            shape_buckets=lambda: {}, io_schema=lambda: None,
                            input_processor=lambda: None, output_processor=lambda: None)
    received_options = []

    def load(checkpoint, *, device, ops, **options):
        received_options.append(options)
        return model

    monkeypatch.setattr(policy_module, "get_model_class", lambda name: SimpleNamespace(from_checkpoint=load))
    monkeypatch.setattr(policy_module.ops.registry, "load_plugins", lambda: None)
    monkeypatch.setattr(policy_module, "Engine", lambda model, platform, config: SimpleNamespace(config=config))
    policy = policy_module.Policy("test", "unused", device="cpu", stream_priorities={"action": 0}, scale=3)
    assert received_options == [{"scale": 3}]
    assert policy.engine.config.stream_priorities == {"vision": 0, "prefill": 0, "action": 0}
    assert policy.resolved.settings["stream_priorities.action"].source == "explicit"

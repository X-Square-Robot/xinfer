"""Ascend platform and engine behavior using only stock TorchNPU operators."""

import pytest
import torch

from xinfer.engine import Engine, EngineConfig
from xinfer.engine.cudagraph import CUDAGraphConfig, CUDAGraphMiss
from xinfer.config import resolve
from xinfer.models.interface import HandoffSpec, OutputSpec, PreparedRequest, Stage, StageGraph, TensorSpec, VLAModel
from xinfer.ops import reference
from xinfer.ops.backends.ascend import _torch_attn
from xinfer.platforms import resolve_platform
from xinfer.platforms.devices import device_module_for, graph_api_for


class Pipeline(VLAModel):
    stage_graph = StageGraph(Stage("a", "first"), Stage("b", "second", depends_on="a"))
    handoffs = (HandoffSpec("a", "b", {"h": TensorSpec()}),)

    def stage_outputs(self, stage, batch):
        return {"h" if stage == "a" else "y": OutputSpec(tuple(batch.inputs["x"].shape), torch.float32)}

    def first(self, batch):
        torch.mul(batch.inputs["x"], 2, out=batch.out["h"])

    def second(self, batch):
        torch.add(batch.inputs["h"], batch.inputs["x"], out=batch.out["y"])


@pytest.fixture
def npu_runtime():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("no NPU available")
    return torch.npu


def request(value, width=3):
    return PreparedRequest({"x": torch.full((1, width), float(value))})


def test_explicit_npu_dispatches_to_the_platform_resolver(monkeypatch):
    from xinfer.platforms import npu

    monkeypatch.setattr(npu, "npu_platform", lambda device: device)
    assert resolve_platform("npu:1") == "npu:1"


def test_discovery_and_eager_handoffs(npu_runtime):
    platform = resolve_platform("npu")
    assert platform.is_npu
    assert platform.device == f"npu:{npu_runtime.current_device()}"
    assert device_module_for(platform.device) is npu_runtime
    with pytest.raises(ValueError, match="out of range"):
        resolve_platform(f"npu:{npu_runtime.device_count()}")

    engine = Engine(Pipeline(), platform, EngineConfig(lanes={"a": 2, "b": 2}))
    reported = []
    engine.stats_sink = reported.append
    try:
        futures = [engine.submit(request(value), robot=value) for value in range(4)]
        for value, future in enumerate(futures):
            result = future.result(timeout=60)["y"]
            assert result.device.type == "cpu"
            torch.testing.assert_close(result, torch.full((1, 3), float(3 * value)), atol=0, rtol=0)
    finally:
        engine.shutdown()
    assert len(reported) == 4
    assert all(set(stats.stages) == {"a", "b"} for stats in reported)
    assert all(times.wait >= 0 and times.execution >= 0 for stats in reported for times in stats.stages.values())


@pytest.mark.parametrize("empty_warmup", [False, True])
def test_default_graphs_capture_the_cold_first_request_without_warmup(npu_runtime, empty_warmup):
    platform = resolve_platform("npu")
    if not platform.capabilities.cudagraphs:
        pytest.skip("this runtime does not support device graphs")
    model = Pipeline()
    config = resolve(model, platform, model_name="test", tuned=None).engine_config()
    assert config.cudagraphs is not None
    engine = Engine(model, platform, config)
    try:
        if empty_warmup:
            engine.warmup([])
        first = engine.submit(request(2)).result(timeout=60)["y"]
        torch.testing.assert_close(first, torch.full((1, 3), 6.0), atol=0, rtol=0)
        captured = engine.cudagraph_stats()
        assert set(captured) == {stage.name for stage in model.stage_graph}
        assert all(stage["captures"] == 1 and stage["eager"] == 0 for stage in captured.values())
        second = engine.submit(request(4)).result(timeout=60)["y"]
        torch.testing.assert_close(second, torch.full((1, 3), 12.0), atol=0, rtol=0)
        replayed = engine.cudagraph_stats()
        assert set(replayed) == set(captured)
        assert all(stage["captures"] == captured[name]["captures"] and stage["eager"] == 0
                   and stage["replays"] == captured[name]["replays"] + 1 for name, stage in replayed.items())
    finally:
        engine.shutdown()


@pytest.mark.parametrize("miss_policy", ["capture", "eager", "error"])
def test_after_warmup_new_shapes_follow_the_shared_graph_policy(npu_runtime, miss_policy):
    platform = resolve_platform("npu")
    if not platform.capabilities.cudagraphs:
        pytest.skip("this runtime does not support device graphs")
    config = EngineConfig(lanes={"a": 2, "b": 2}, cudagraphs=CUDAGraphConfig(on_miss_after_warmup=miss_policy))
    model = Pipeline()
    engine = Engine(model, platform, config)
    try:
        engine.warmup([request(1)])
        warmed = engine.cudagraph_stats()
        assert set(warmed) == {stage.name for stage in model.stage_graph}
        assert all(stage["captures"] == 2 for stage in warmed.values())
        for value in (2, -1, 5):
            result = engine.submit(request(value)).result(timeout=60)["y"]
            torch.testing.assert_close(result, torch.full((1, 3), float(3 * value)), atol=0, rtol=0)
        replayed = engine.cudagraph_stats()
        assert set(replayed) == set(warmed)
        assert all(stage["replays"] > warmed[name]["replays"] for name, stage in replayed.items())
        if miss_policy == "error":
            with pytest.raises(CUDAGraphMiss, match="no graph"):
                engine.submit(request(1, width=5)).result(timeout=60)
        else:
            result = engine.submit(request(1, width=5)).result(timeout=60)["y"]
            torch.testing.assert_close(result, torch.full((1, 5), 3.0), atol=0, rtol=0)
            missed = engine.cudagraph_stats()
            assert set(missed) == set(warmed)
            counter = "captures_after_warmup" if miss_policy == "capture" else "eager"
            assert all(stage[counter] == replayed[name][counter] + 1 for name, stage in missed.items())
            result = engine.submit(request(2, width=5)).result(timeout=60)["y"]
            torch.testing.assert_close(result, torch.full((1, 5), 6.0), atol=0, rtol=0)
            repeated = engine.cudagraph_stats()
            assert set(repeated) == set(warmed)
            if miss_policy == "capture":
                assert all(stage["captures"] == missed[name]["captures"]
                           and stage["replays"] == missed[name]["replays"] + 1
                           and stage["eager"] == replayed[name]["eager"] for name, stage in repeated.items())
            else:
                assert all(stage["captures"] == replayed[name]["captures"]
                           and stage["eager"] == replayed[name]["eager"] + 2 for name, stage in repeated.items())
    finally:
        engine.shutdown()


@pytest.mark.parametrize("kind", ["plain", "bool", "additive", "causal"])
def test_graph_attention_keeps_sdpa_mask_semantics(kind):
    rng = torch.Generator().manual_seed(12)
    q = torch.randn(1, 2, 4, 8, generator=rng)
    k, v = [torch.randn(1, 2, 6, 8, generator=rng) for _ in range(2)]
    mask = None
    if kind in ("bool", "additive"):
        mask = torch.ones(1, 1, 4, 6, dtype=torch.bool)
        mask[..., 0, 1:] = False
        mask[..., 1, :] = False
        if kind == "additive":
            mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, float("-inf"))
            mask[..., 2, 1] = -2
    options = {"mask": mask, "causal": kind == "causal", "scale": 0.21}
    expected = reference.attention(q, k, v, **options)
    result = _torch_attn._graph_safe_attention(q, k, v, attn_mask=mask, is_causal=kind == "causal", scale=0.21)
    torch.testing.assert_close(result, expected, atol=3e-7, rtol=2e-6)
    if mask is not None:
        assert torch.count_nonzero(result[..., 1, :]) == 0


def test_unsupported_npu_priorities_are_rejected_before_lane_creation(npu_runtime, monkeypatch):
    from xinfer.engine.executor import Lane

    platform = resolve_platform("npu")
    if platform.capabilities.stream_priorities:
        pytest.skip("this runtime exposes verifiable stream priorities")
    assert platform.capabilities.stream_priority_range is None
    monkeypatch.setattr(npu_runtime, "Stream", lambda **kwargs: pytest.fail("stream created"))
    with pytest.raises(ValueError, match="does not support nonzero stream priorities"):
        Lane("action", 0, platform, priority=-1)


@pytest.mark.parametrize("operator", ["varlen_attention", "causal_attention", "prefix_kvcache_attention"])
def test_bf16_attention_graph_replays_changed_inputs_and_masks(npu_runtime, operator):
    if not resolve_platform("npu").capabilities.cudagraphs:
        pytest.skip("this runtime does not support device graphs")
    rng = torch.Generator().manual_seed(43)

    def tensor(*shape):
        return torch.randn(*shape, generator=rng).to(device="npu", dtype=torch.bfloat16)

    options = {}
    if operator == "varlen_attention":
        inputs = (*tensor(2048, 3, 16, 80).unbind(1), (0, 1024, 2048))
    elif operator == "causal_attention":
        inputs = (tensor(1, 768, 16, 128), tensor(1, 768, 2, 128), tensor(1, 768, 2, 128))
    else:
        inputs = (tensor(1, 16, 16, 128), tensor(1, 2, 784, 128).transpose(1, 2),
                  tensor(1, 2, 784, 128).transpose(1, 2), torch.tensor([768], dtype=torch.int32, device="npu"), 768)
        mask = torch.zeros(1, 1, 784, 2, dtype=torch.int32, device="npu")
        mask[..., 0] = 16
        mask[:, :, 700:768, 0] = 0
        mask[:, :, 773:, 1] = 5
        options["flash_mask"] = mask
    function = getattr(_torch_attn, operator)
    stream = npu_runtime.Stream()
    stream.wait_stream(npu_runtime.current_stream())
    with torch.inference_mode(), npu_runtime.stream(stream):
        expected = function(*inputs, **options).clone()
        golden = getattr(reference, operator)(*inputs, **options)
    stream.synchronize()
    error = (expected.float() - golden.float()).abs().max()
    assert error <= 4 * 2 ** -8 * max(1.0, float(golden.float().abs().max()))
    tensors = [value for value in (*inputs, *options.values()) if torch.is_tensor(value)]
    saved = [value.clone() for value in tensors]
    api = graph_api_for("npu")
    graph = api.CUDAGraph()
    with torch.inference_mode(), api.graph(graph, stream=stream):
        actual = function(*inputs, **options)
    graph.replay()
    npu_runtime.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for value, original in zip(tensors, saved):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
    inputs[0].add_(0.125)
    if options:
        options["flash_mask"][:, :, 773:, 1] = 9
    expected_changed = function(*inputs, **options)
    npu_runtime.synchronize()
    graph.replay()
    npu_runtime.synchronize()
    torch.testing.assert_close(actual, expected_changed, atol=0, rtol=0)
    assert not torch.equal(expected, expected_changed)

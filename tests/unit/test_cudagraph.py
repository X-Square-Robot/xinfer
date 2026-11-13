"""CUDA graph capture/replay of stage dispatches (GPU only)."""

import threading

import pytest
import torch

from xinfer.engine import Engine, EngineConfig
from xinfer.engine.cudagraph import CUDAGraphMiss, CUDAGraphConfig, LaneCUDAGraphs
from xinfer.models.interface import (BucketSpec, Capabilities, DeviceImage, HandoffSpec, OutputSpec, PreparedRequest,
                                     Stage, StageGraph, TensorSpec, VLAModel)
from xinfer.platforms import resolve_platform

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


class Toy(VLAModel):
    """Two stages; the second's work depends on a host value ``steps``."""

    stage_graph = StageGraph(Stage("a", "first"), Stage("b", "second", depends_on="a"))
    handoffs = (HandoffSpec("a", "b", {"h": TensorSpec()}),)

    def stage_outputs(self, stage, batch):
        n = batch.inputs["x"].shape[0]
        return {"a": {"h": OutputSpec((n, 8), torch.float32)},
                "b": {"y": OutputSpec((n, 8), torch.float32)}}[stage]

    def first(self, batch):
        x = batch.inputs["x"]
        torch.matmul(x, batch.inputs["w"], out=batch.out["h"])

    def second(self, batch):
        h = batch.inputs["h"]
        for _ in range(int(batch.inputs["steps"])):
            h = torch.tanh(h) * 1.5 + batch.inputs["bias"]
        batch.out["y"].copy_(h)


def _request(n, steps=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return PreparedRequest(dict(x=torch.randn(n, 4, generator=g), w=torch.randn(4, 8, generator=g),
                                bias=torch.randn(8, generator=g), steps=steps, unused=torch.zeros(n + 1)))


def test_stage_stream_priorities_preserve_graph_replay_outputs():
    platform = resolve_platform("cuda")
    high = torch.cuda.Stream(device=platform.device, priority=-1).priority
    if high == 0:
        pytest.skip("CUDA device has no distinct high-priority stream")
    normal = Engine(Toy(), platform, EngineConfig(cudagraphs=CUDAGraphConfig()))
    prioritized = Engine(Toy(), platform, EngineConfig(lanes={"a": 2, "b": 2},
                         stream_priorities={"b": high}, cudagraphs=CUDAGraphConfig()))
    try:
        assert all(lane.stream.priority == 0 for lane in prioritized.lanes["a"])
        assert all(lane.stream.priority == high for lane in prioritized.lanes["b"])
        assert prioritized._upload.priority == 0
        requests = [_request(3, seed=i) for i in range(6)]
        expected = [normal.run(request)["y"] for request in requests]
        prioritized.warmup(requests)
        futures = [prioritized.submit(request) for request in requests]
        for future, value in zip(futures, expected):
            assert torch.equal(future.result(timeout=30)["y"], value)
        assert prioritized.cudagraph_stats()["b"]["replays"] >= len(requests)
    finally:
        prioritized.shutdown()
        normal.shutdown()


def test_unrepresentable_stream_priority_is_rejected_instead_of_clamped():
    with pytest.raises(ValueError, match="would clamp"):
        Engine(Toy(), resolve_platform("cuda"), EngineConfig(stream_priorities={"a": 1_000_000}), start=False)


def test_graph_replay_equals_eager_and_reuses_graphs():
    platform = resolve_platform("cuda")
    eager = Engine(Toy(), platform)
    graphed = Engine(Toy(), platform, EngineConfig(cudagraphs=CUDAGraphConfig()))
    # Largest shape first, so smaller shapes fit the static buffers without regrowth.
    for n, steps, seed in [(5, 3, 0), (3, 3, 1), (3, 3, 2), (3, 2, 3), (5, 3, 4)]:
        request = _request(n, steps, seed)
        assert torch.equal(graphed.run(request)["y"], eager.run(request)["y"])
    stats = graphed.cudagraph_stats()
    # stage a: n=5 and n=3; stage b: (5, steps 3), (3, 3), (3, 2)
    assert stats["a"]["captures"] == 2 and stats["b"]["captures"] == 3
    assert stats["a"]["replays"] == 5


def test_unread_inputs_do_not_split_graphs():
    graphed = Engine(Toy(), resolve_platform("cuda"), EngineConfig(cudagraphs=CUDAGraphConfig()))
    for seed in range(3):
        request = _request(3, seed=seed)
        request.tensors["unused"] = torch.zeros(seed + 1)  # shape varies but no stage reads it
        graphed.run(request)
    assert graphed.cudagraph_stats()["a"]["captures"] == 1


def test_miss_policies_after_warmup():
    platform = resolve_platform("cuda")
    for policy, expect in (("eager", "eager"), ("capture", "captures")):
        engine = Engine(Toy(), platform, EngineConfig(cudagraphs=CUDAGraphConfig(on_miss_after_warmup=policy)))
        engine.warmup([_request(3)])
        before = engine.cudagraph_stats()["a"][expect]
        engine.run(_request(6))
        assert engine.cudagraph_stats()["a"][expect] == before + 1
    engine = Engine(Toy(), platform, EngineConfig(cudagraphs=CUDAGraphConfig(on_miss_after_warmup="error")))
    engine.warmup([_request(3)])
    with pytest.raises(CUDAGraphMiss):
        engine.run(_request(6))


def test_lru_bound_and_static_buffer_growth():
    platform = resolve_platform("cuda")
    model = Toy()
    lane = LaneCUDAGraphs(model, torch.cuda.Stream(), platform.device, CUDAGraphConfig(max_graphs=2))
    w = torch.randn(4, 8, device="cuda")
    for n in (2, 3, 4):  # each larger n grows the static buffers and drops older graphs
        x = torch.randn(n, 4, device="cuda")
        with torch.cuda.stream(lane.stream):
            out = lane.run("a", 1, dict(x=x, w=w))
        lane.stream.synchronize()
        torch.testing.assert_close(out["h"], x @ w)
    assert lane.graph_count() <= 2
    for n in (2, 3):  # smaller shapes reuse the grown static buffers
        x = torch.randn(n, 4, device="cuda")
        with torch.cuda.stream(lane.stream):
            out = lane.run("a", 1, dict(x=x, w=w))
        lane.stream.synchronize()
        torch.testing.assert_close(out["h"], x @ w)


def test_result_freed_during_a_capture():
    """A result freed on another thread while its lane is being captured must not break pinned allocation."""
    engine = Engine(Toy(), resolve_platform("cuda"), EngineConfig(cudagraphs=CUDAGraphConfig()))
    held = [engine.run(_request(3, seed=0))]
    engine.run(_request(3, seed=1))  # the engine no longer refers to the held result
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph, stream=engine.lanes["b"][0].stream,
                                                  capture_error_mode="thread_local"):  # as the engine captures
        torch.ones(8, device="cuda") * 2
        release = threading.Thread(target=held.clear)
        release.start()
        release.join()
    del graph
    torch.empty(64, pin_memory=True)  # queries the events of freed pinned buffers
    engine.run(_request(3, seed=2))
    engine.shutdown()


class BatchedToy(Toy):
    """Toy whose second stage batches B1/B2/B4; it works row by row."""

    def capabilities(self):
        return Capabilities(batch_sizes={"a": (1,), "b": (1, 2, 4)}, num_steps=0)


def _row_request(seed, steps=200):
    """One-row request for ``BatchedToy`` (batch axis 0 has size 1, so rows can be stacked into a batch)."""
    g = torch.Generator().manual_seed(seed)
    return PreparedRequest(dict(x=torch.randn(1, 4, generator=g), w=torch.randn(4, 8, generator=g),
                                bias=torch.randn(1, 8, generator=g), steps=steps))


def test_warmup_captures_every_lane_and_batch_size():
    """After warmup, concurrent requests of the warmed shapes run on any lane and batch size without a capture
    (``on_miss_after_warmup="error"`` fails a request on a miss), with the bits of single eager runs."""
    platform = resolve_platform("cuda")
    engine = Engine(BatchedToy(), platform,
                    EngineConfig(lanes={"a": 2, "b": 2}, max_batch={"b": 4}, max_running_requests=16,
                                 cudagraphs=CUDAGraphConfig(on_miss_after_warmup="error")))
    engine.warmup([_row_request(0)])
    warmed = engine.cudagraph_stats()
    assert warmed["a"]["graphs"] == 2  # 2 lanes x B1
    assert warmed["b"]["graphs"] == 6  # 2 lanes x B1, B2, B4
    requests = [_row_request(seed) for seed in range(1, 41)]
    futures = [engine.submit(request) for request in requests]
    results = [future.result(timeout=60) for future in futures]
    served = engine.cudagraph_stats()
    batched = engine.stats["dispatch_b_b2"] + engine.stats["dispatch_b_b4"]
    engine.shutdown()
    assert served["a"]["captures"] == warmed["a"]["captures"]
    assert served["b"]["captures"] == warmed["b"]["captures"]
    assert batched > 0  # batches formed and replayed warmed graphs
    eager = Engine(BatchedToy(), platform)
    for request, result in zip(requests, results):
        assert torch.equal(result["y"], eager.run(request)["y"])
    eager.shutdown()


class RaggedToy(VLAModel):
    """a: kv = x (its length varies); b (B1/B2/B4): y = sum of the row's own entries of the left-padded kv."""

    stage_graph = StageGraph(Stage("a", "first"), Stage("b", "second", depends_on="a"))
    handoffs = (HandoffSpec("a", "b", {"kv": TensorSpec(batch_axis=0, seq_axis=1)}),)

    def capabilities(self):
        return Capabilities(batch_sizes={"a": (1,), "b": (1, 2, 4)}, num_steps=0)

    def stage_outputs(self, stage, batch):
        if stage == "a":
            return {"kv": OutputSpec((1, batch.inputs["x"].shape[1]), torch.float32)}
        return {"y": OutputSpec((batch.batch_size, 1), torch.float32)}

    def first(self, batch):
        batch.out["kv"].copy_(batch.inputs["x"] * 2)

    def second(self, batch):
        kv, lengths = batch.inputs["kv"], batch.inputs["kv_lengths"]
        start = kv.shape[1] - lengths.long()
        valid = torch.arange(kv.shape[1], device=kv.device)[None] >= start[:, None]
        torch.sum(kv * valid, dim=1, keepdim=True, out=batch.out["y"])


def test_warmed_lengths_cover_batches_of_mixed_lengths():
    """A batch of different lengths is padded to its longest member, so warming each length covers the mixes."""
    engine = Engine(RaggedToy(), resolve_platform("cuda"),
                    EngineConfig(lanes={"a": 2, "b": 2}, max_batch={"b": 4}, max_running_requests=16,
                                 cudagraphs=CUDAGraphConfig(on_miss_after_warmup="error")))
    lengths = (3, 5, 8)
    engine.warmup([PreparedRequest({"x": torch.ones(1, n)}) for n in lengths])
    warmed = engine.cudagraph_stats()
    assert warmed["b"]["graphs"] == 2 * 3 * len(lengths)  # lanes x batch sizes x lengths
    sizes = [lengths[i % 3] for i in range(48)]
    futures = [engine.submit(PreparedRequest({"x": torch.full((1, n), float(i + 1))})) for i, n in enumerate(sizes)]
    for i, (future, n) in enumerate(zip(futures, sizes)):
        assert float(future.result(timeout=60)["y"]) == 2.0 * (i + 1) * n
    padded = engine.stats["padded_dispatch_b"]
    served = engine.cudagraph_stats()
    engine.shutdown()
    assert served["b"]["captures"] == warmed["b"]["captures"]
    assert padded > 0  # some batches mixed lengths


class BucketedToy(RaggedToy):
    """RaggedToy whose inputs come in three lengths; warming up one request covers all of them."""

    def shape_buckets(self):
        return {"x": BucketSpec(3, max_length=8)}

    def pad(self, prepared, dimension, width):
        return PreparedRequest({"x": torch.ones(1, width)}, prepared.meta)


def test_warmup_requests_cover_every_bucket():
    engine = Engine(BucketedToy(), resolve_platform("cuda"),
                    EngineConfig(lanes={"a": 2, "b": 2}, max_batch={"b": 4}, max_running_requests=16,
                                 cudagraphs=CUDAGraphConfig(on_miss_after_warmup="error")))
    engine.warmup([PreparedRequest({"x": torch.ones(1, 3)}, {"x_len_range": (3, 8)})])
    warmed = engine.cudagraph_stats()
    assert warmed["a"]["graphs"] == 2 * 3 and warmed["b"]["graphs"] == 2 * 3 * 3  # lanes x lengths (x batch sizes)
    sizes = [(3, 6, 8)[i % 3] for i in range(48)]
    futures = [engine.submit(PreparedRequest({"x": torch.full((1, n), float(i + 1))})) for i, n in enumerate(sizes)]
    for i, (future, n) in enumerate(zip(futures, sizes)):
        assert float(future.result(timeout=60)["y"]) == 2.0 * (i + 1) * n
    served = engine.cudagraph_stats()
    engine.shutdown()
    assert served["b"]["captures"] == warmed["b"]["captures"]


class Views(Toy):
    """Toy whose ``x`` is a ``[4, 4, 3]`` uint8 view, summed over its channels."""

    def first(self, batch):
        torch.matmul(batch.inputs["x"].float().sum(-1), batch.inputs["w"], out=batch.out["h"])


def test_device_images_are_decoded_off_the_lanes_and_replay_one_graph(monkeypatch):
    """JPEGs of different lengths and a frame become ``x`` on the device-image thread's stream, not a lane's, from
    host memory; every request replays the graph warmed with the first and equals the eager engine's output."""
    if not resolve_platform("cuda").capabilities.device_images:
        pytest.skip("needs nvImageCodec")
    cv2 = pytest.importorskip("cv2")

    from xinfer.engine.device_images import DeviceImageDecoder

    streams, decode = [], DeviceImageDecoder.__call__

    def recording(self, inputs):
        assert all(value.data.device.type == "cpu" for value in inputs.values() if isinstance(value, DeviceImage))
        streams.append(torch.cuda.current_stream())
        return decode(self, inputs)

    monkeypatch.setattr(DeviceImageDecoder, "__call__", recording)
    platform = resolve_platform("cuda")
    engine = Engine(Views(), platform, EngineConfig(cudagraphs=CUDAGraphConfig(on_miss_after_warmup="error")))
    eager = Engine(Views(), platform, EngineConfig())
    try:
        def request(seed, jpeg=True):
            base = _request(4, seed=seed).tensors
            g = torch.Generator().manual_seed(seed)
            frame = torch.randint(0, 256, (12 + seed, 10 + seed, 3), dtype=torch.uint8, generator=g)
            data = torch.from_numpy(cv2.imencode(".jpg", frame.numpy())[1]) if jpeg else frame
            image = DeviceImage(data, ((6, 5), (4, 4)), (("bicubic", True), ("bilinear", True)))
            return PreparedRequest(dict(x=image, w=base["w"], bias=base["bias"], steps=3))

        engine.warmup([request(0)])
        assert len({request(seed).tensors["x"].data.numel() for seed in (1, 2)}) == 2
        for seed, jpeg in ((1, True), (2, True), (3, False)):
            assert torch.equal(engine.run(request(seed, jpeg))["y"], eager.run(request(seed, jpeg))["y"])
        assert engine.cudagraph_stats()["a"]["graphs"] == 1
        lanes = {lane.stream for e in (engine, eager) for lanes in e.lanes.values() for lane in lanes}
        assert set(streams) == {engine._device_stream, eager._device_stream} and not set(streams) & lanes
    finally:
        engine.shutdown()
        eager.shutdown()

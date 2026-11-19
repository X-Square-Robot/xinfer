"""Existing Ascend capture and image-stream contracts, exercised without a device."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from xinfer.engine import core, cudagraph, executor
from xinfer.models.interface import Stage, StageGraph, VLAModel
from xinfer.platforms.interface import Platform, PlatformCapabilities


class TwoDeviceNpu:
    def __init__(self):
        self.current = "npu:0"
        self.pending = {"npu:0": True, "npu:1": False}
        self.synchronized = []

    @contextmanager
    def device(self, device):
        previous, self.current = self.current, device
        try:
            yield
        finally:
            self.current = previous

    def synchronize(self, device=None):
        device = self.current if device is None else device
        self.synchronized.append(device)
        self.pending[device] = False

    def empty_cache(self):
        pass

    def Stream(self, device, priority=0):
        return SimpleNamespace(device=device, priority=priority, wait_stream=lambda stream: None)

    def current_stream(self, device):
        return SimpleNamespace(device=device)


def test_capture_batch_sizes_waits_on_the_requested_npu_after_stream_scope(monkeypatch):
    runtime = TwoDeviceNpu()
    captured = []

    def capture(stage, batch_size, batch):
        captured.append(runtime.current)
        runtime.pending[runtime.current] = True

    graphs = SimpleNamespace(warmup_inputs={"step": [{"x": torch.ones(1)}]},
                             known=lambda stage, batch_size, batch: False, run=capture)
    lane = SimpleNamespace(cudagraphs=graphs, _device_module=runtime,
                           use_stream=lambda: runtime.device("npu:1"))
    monkeypatch.setattr(cudagraph, "device_module_for", lambda device: runtime)

    cudagraph.capture_batch_sizes({"step": [lane]}, {"step": [1]},
                                 lambda stage, rows: rows[0], device="npu:1")

    assert captured == ["npu:1"]
    assert runtime.current == "npu:0"
    assert runtime.synchronized == ["npu:1"]
    assert not runtime.pending["npu:1"]
    assert runtime.pending["npu:0"]


class SingleStage(VLAModel):
    stage_graph = StageGraph(Stage("step", "step"))

    def step(self, batch):
        pass


def test_ascend_without_device_images_has_no_image_processing_stream(monkeypatch):
    runtime = TwoDeviceNpu()
    monkeypatch.setattr(core, "device_module_for", lambda device: runtime)
    monkeypatch.setattr(executor, "device_module_for", lambda device: runtime)
    platform = Platform("ascend-test", "npu:1", PlatformCapabilities(
        device_type="ascend", arch="test", cudagraphs=False, stream_priorities=False,
        unified_memory=False, multiprocess_service=False, sm_count=None,
        total_memory_bytes=None, device_images=False))
    engine = core.Engine(SingleStage(), platform, start=False)
    try:
        assert engine._device_stream is None
    finally:
        engine.shutdown()

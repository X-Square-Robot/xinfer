"""Platform JPEG decoder boundaries and CUDA stream ownership without accelerator hardware."""

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch

from xinfer.engine import device_images
from xinfer.models.interface import DeviceImage
from xinfer.platforms import cuda, devices


@pytest.mark.parametrize(("requested", "index"), [("cuda", 4), (torch.device("cuda:2"), 2)])
def test_jpeg_factory_normalizes_the_device_before_creating_its_decoder(monkeypatch, requested, index):
    created = []

    def create(device):
        created.append(device)
        return SimpleNamespace(device=device, backend="cuda")

    monkeypatch.setattr(devices, "device_module_for", lambda device: SimpleNamespace(current_device=lambda: 4))
    monkeypatch.setattr(cuda, "CudaJpegDecoder", create)
    decoder = devices.jpeg_decoder_for(requested)
    assert decoder.backend == "cuda" and decoder.device == torch.device("cuda", index)
    assert created and all(device == decoder.device for device in created)


def test_jpeg_factory_rejects_a_device_without_an_accelerator_runtime():
    with pytest.raises(ValueError, match="need a CUDA or NPU device"):
        devices.jpeg_decoder_for("cpu")


def test_engine_images_use_the_decoder_interface_for_mixed_frames_and_close(monkeypatch):
    requested, batches, uploaded, closed = [], [], [], []
    jpeg_a, jpeg_b = (torch.tensor(values, dtype=torch.uint8) for values in ([1, 2], [3, 4]))
    raw = torch.tensor([[[0, 10, 20], [20, 30, 40]], [[40, 50, 60], [60, 70, 80]]], dtype=torch.uint8)
    raw_before = raw.clone()
    first = torch.tensor([10, 20, 30], dtype=torch.uint8)[:, None, None].expand(3, 2, 2)
    second = torch.tensor([100, 110, 120], dtype=torch.uint8)[:, None, None].expand(3, 2, 2)
    target = torch.device("cpu")

    class Decoder:
        device = target

        def __call__(self, jpegs):
            batches.append(jpegs)
            return [first, second]

        def close(self):
            closed.append(True)

    def factory(device):
        requested.append(device)
        return Decoder()

    def upload(value, device):
        uploaded.append((value, device))
        return value

    monkeypatch.setattr(device_images, "jpeg_decoder_for", factory)
    monkeypatch.setattr(device_images, "_to_device", upload)
    # An opaque factory input proves the engine does not parse device kinds itself.
    decoder = device_images.DeviceImageDecoder("test-accelerator:7")
    assert requested[-1] == "test-accelerator:7" and decoder.device == target
    metadata = {"camera": 7}
    inputs = {"jpeg_a": DeviceImage(jpeg_a, ((1, 1),), (("bilinear", False),)),
              "raw": DeviceImage(raw, ((1, 1), (2, 3)), (("bilinear", False), ("bicubic", True))),
              "meta": metadata, "jpeg_b": DeviceImage(jpeg_b, (), ())}
    result = decoder(inputs)
    assert set(result) == set(inputs) and result["meta"] == metadata
    assert len(batches) == 1 and torch.equal(batches[0][0], jpeg_a) and torch.equal(batches[0][1], jpeg_b)
    assert uploaded and all(torch.equal(value, raw) and device == target for value, device in uploaded)
    assert torch.equal(result["jpeg_a"], torch.tensor([[[10, 20, 30]]], dtype=torch.uint8))
    assert torch.equal(result["raw"], torch.tensor([30, 40, 50], dtype=torch.uint8).expand(2, 3, 3))
    assert torch.equal(result["jpeg_b"], second.permute(1, 2, 0))
    assert all(result[name].is_contiguous() for name in ("jpeg_a", "raw", "jpeg_b"))
    assert torch.equal(inputs["raw"].data, raw_before)  # preparing outputs must not replace the caller's inputs
    decoder({"raw": inputs["raw"]})
    assert len(batches) == 1  # raw-only requests do not invoke the JPEG codec
    decoder.close()
    assert closed


def test_cuda_jpegs_are_ordered_and_retained_on_each_callers_stream(monkeypatch):
    created, decoded, used_devices = [], [], []
    allocation_stream = object()
    active = SimpleNamespace(stream=None)

    class Caller:
        def __init__(self):
            self.waited_generation = 0

        def wait_stream(self, stream):
            assert stream == allocation_stream
            self.waited_generation = len(decoded)

    class View:
        def __init__(self, value):
            self.value = value
            self.readers = []

        def record_stream(self, stream):
            self.readers.append(stream)

    def new_stream(device):
        created.append(device)
        return allocation_stream

    @contextmanager
    def use_stream(stream):
        previous = active.stream
        active.stream = stream
        try:
            yield
        finally:
            active.stream = previous

    rgb = "RGB"

    def decode_jpeg(jpegs, *, mode, device):
        assert mode == rgb and active.stream == allocation_stream
        used_devices.append(device)
        views = [View(int(jpeg[0])) for jpeg in jpegs]
        decoded.append(views)
        return views

    image_io = ModuleType("torchvision.io")
    image_io.ImageReadMode = SimpleNamespace(RGB=rgb)
    image_io.decode_jpeg = decode_jpeg
    monkeypatch.setitem(sys.modules, "torchvision", ModuleType("torchvision"))
    monkeypatch.setitem(sys.modules, "torchvision.io", image_io)
    monkeypatch.setattr(torch.cuda, "Stream", new_stream)
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: active.stream)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    decoder = devices.jpeg_decoder_for("cuda")
    assert created and all(device == decoder.device for device in created)
    assert decoder.device == torch.device("cuda:3")
    callers, results = [Caller(), Caller()], []
    for caller, values in zip(callers, ([17, 29], [47, 59])):
        active.stream = caller
        results.append(decoder([torch.tensor([value]) for value in values]))
        assert active.stream == caller
        assert caller.waited_generation == len(decoded) > 0  # dependency includes this call's decode
        assert all(caller in view.readers for view in results[-1])
        assert [view.value for view in results[-1]] == values
    assert all(device == decoder.device for device in used_devices)
    assert [view.value for view in results[0]] == [17, 29]
    decoder.close()
    decoder.close()

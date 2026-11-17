"""Platform JPEG decoder boundaries and CUDA stream ownership without accelerator hardware."""

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from xinfer.engine import device_images
from xinfer.models.interface import DeviceImage
from xinfer.ops.reference.image import image_resize
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


@pytest.fixture
def nvimgcodec_stub(monkeypatch):
    """Stub nvImageCodec: ``install(status, decoder_class=None)`` with status "decodes", "no-decoder" (its ``Decoder``
    raises for the device) or None (not installed). Returns the ``Decoder`` calls made."""
    calls = []

    def install(status, decoder_class=None):
        class Decoder:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))
                if status == "no-decoder":
                    raise RuntimeError("no decoder for this device")

            def decode(self, data, **kwargs):
                calls.append(("decode", kwargs))
                return [object() for _ in data]

        nvidia = ModuleType("nvidia")
        monkeypatch.setitem(sys.modules, "nvidia", nvidia)
        if status is None:  # not installed: the stub nvidia is no package, so the import fails
            monkeypatch.delitem(sys.modules, "nvidia.nvimgcodec", raising=False)
        else:
            nvimgcodec = ModuleType("nvidia.nvimgcodec")
            nvimgcodec.Decoder, nvimgcodec.DecodeParams = decoder_class or Decoder, lambda color_spec: None
            nvimgcodec.ColorSpec = SimpleNamespace(SRGB="srgb")
            nvidia.nvimgcodec = nvimgcodec
            monkeypatch.setitem(sys.modules, "nvidia.nvimgcodec", nvimgcodec)
        cuda._nvimgcodec_decoder.cache_clear()
        return calls

    yield install
    cuda._nvimgcodec_decoder.cache_clear()


@pytest.mark.parametrize("status", ["decodes", "no-decoder", None])
def test_the_cuda_platform_has_device_images_where_nvimgcodec_has_a_decoder(nvimgcodec_stub, status):
    """The capability only creates the decoder (one CPU thread) and decodes nothing; without one, the decoder names
    the package to install."""
    calls = nvimgcodec_stub(status)
    assert (cuda._nvimgcodec_decoder(0) is not None) is (status == "decodes")  # the capability device_images
    assert [name for name, _ in calls] == ([] if status is None else ["init"])
    if status is not None:
        assert calls[0][1] == {"device_id": 0, "max_num_cpu_threads": 1}
    if status != "decodes":
        with pytest.raises(RuntimeError, match=r"nvImageCodec.*xinfer\[gpu-image\]"):
            cuda.CudaJpegDecoder(torch.device("cuda", 0))


def test_cuda_jpegs_are_ordered_retained_and_decoded_into_kept_images(monkeypatch, nvimgcodec_stub):
    """Decode and copy run on the decoder's stream; each caller's stream waits for it and retains the views; a call
    decodes into the images of earlier calls once their copies have completed, never before."""
    created, decoded = [], []
    allocation_stream = SimpleNamespace(cuda_stream=7)
    active = SimpleNamespace(stream=None)
    copies_completed = SimpleNamespace(value=False)

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

        def permute(self, *dims):
            return self

        def clone(self, memory_format):
            assert active.stream == allocation_stream
            return View(self.value)

        def record_stream(self, stream):
            self.readers.append(stream)

    class Event:
        def record(self, stream):
            assert stream == allocation_stream

        def query(self):
            return copies_completed.value

    class Decoder:
        def __init__(self, device_id, max_num_cpu_threads):
            assert (device_id, max_num_cpu_threads) == (3, 1)

        def decode(self, data, params, cuda_stream, images=None):
            assert active.stream == allocation_stream and cuda_stream == 7
            out = images or [SimpleNamespace() for _ in data]
            for image, jpeg in zip(out, data):
                image.value = int(jpeg[0])
            decoded.append((out, images is not None))
            return out

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

    nvimgcodec_stub("decodes", Decoder)
    monkeypatch.setattr(torch, "as_tensor", lambda image, device: View(image.value))
    monkeypatch.setattr(torch.cuda, "Event", Event)
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
    assert [reused for _, reused in decoded] == [False, False]  # the first copies have not completed
    copies_completed.value = True
    assert [view.value for view in decoder([torch.tensor([71]), torch.tensor([73])])] == [71, 73]
    assert decoded[-1][1] and decoded[-1][0] == decoded[0][0]  # decoded into the first call's images
    assert [view.value for view in results[0]] == [17, 29]  # earlier views are copies
    decoder.close()
    decoder.close()


def test_nvimgcodec_decodes_into_its_earlier_images_within_one_level_of_opencv():
    """Batches of JPEGs of several sizes, decoded in turn by one decoder that reuses its earlier images once their
    copies have completed: every view equals a decode by a fresh nvImageCodec decoder and is within one level of
    OpenCV's (libjpeg-turbo) decode."""
    if not torch.cuda.is_available() or cuda._nvimgcodec_decoder(torch.cuda.current_device()) is None:
        pytest.skip("needs a CUDA device and nvImageCodec")
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from nvidia import nvimgcodec

    rng = np.random.default_rng(0)
    smooth = cv2.resize(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8), (640, 480), interpolation=cv2.INTER_CUBIC)
    images = [smooth, smooth[:200, :300], cv2.resize(smooth, (1280, 720)), rng.integers(0, 256, (32, 48, 3), np.uint8)]
    jpegs = [torch.from_numpy(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]) for image in images]
    expected = [torch.from_numpy(cv2.cvtColor(cv2.imdecode(jpeg.numpy(), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
                .permute(2, 0, 1) for jpeg in jpegs]
    device = torch.device("cuda", torch.cuda.current_device())
    decoder, fresh = cuda.CudaJpegDecoder(device), nvimgcodec.Decoder(device_id=device.index)
    params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.SRGB)
    for batch in ([0, 1, 2], [3, 0], [2, 1, 0], [1], [0, 1, 2, 3], [2, 3, 0]) * 3:
        views = decoder([jpegs[i] for i in batch])
        torch.cuda.synchronize()
        reference = fresh.decode([jpegs[i].numpy() for i in batch], params=params)
        for i, view, image in zip(batch, views, reference):
            assert torch.equal(view, torch.as_tensor(image, device=device).permute(2, 0, 1))
            assert (view.cpu().int() - expected[i].int()).abs().max() <= 1
    assert decoder._free or decoder._copying  # the decoder kept its images for reuse


def test_resize_rounds_each_intermediate_step_and_keeps_unchanged_sizes():
    view = torch.tensor([[[0, 255], [3, 128]], [[1, 254], [4, 127]], [[2, 253], [5, 126]]], dtype=torch.uint8)
    first = F.interpolate(view[None].float(), size=(3, 5), mode="bilinear", antialias=False, align_corners=False)
    first = (first + 0.5).floor().clamp(0, 255).to(torch.uint8)
    second = F.interpolate(first.float(), size=(2, 3), mode="bicubic", antialias=True, align_corners=False)
    expected = (second + 0.5).floor().clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).contiguous()
    result = image_resize(view, ((2, 2), (3, 5), (2, 3)),
                          (("bicubic", True), ("bilinear", False), ("bicubic", True)))
    assert torch.equal(result, expected) and result.is_contiguous()
    assert torch.equal(image_resize(view, ((2, 2),), (("bilinear", False),)), view.permute(1, 2, 0))


def test_op_table_exposes_the_image_contract_on_cpu():
    from xinfer.ops import registry
    from xinfer.platforms.cpu import cpu_platform

    table = registry.build_table(cpu_platform(), reference_only=True)
    view = torch.arange(12, dtype=torch.uint8).reshape(3, 2, 2)
    assert table.selection("image_resize").backend == "reference"
    assert torch.equal(table.image_resize(view, ((1, 1),), (("bilinear", False),)),
                       image_resize(view, ((1, 1),), (("bilinear", False),)))

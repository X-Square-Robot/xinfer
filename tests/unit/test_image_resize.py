"""The shared image operator preserves the established resize and per-step rounding contract."""

import torch
import torch.nn.functional as F
import pytest

from xinfer.ops.reference.image import image_resize


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


def test_decoder_instances_keep_their_selected_quantizers_and_reference_selection(monkeypatch):
    from types import SimpleNamespace

    from xinfer.engine import device_images
    from xinfer.ops import registry
    from xinfer.ops.backends.ascend import image as native
    from xinfer.platforms import npu
    from xinfer.platforms.cpu import cpu_platform

    table = registry.build_table(cpu_platform(), reference_only=True)
    assert table.selection("image_quantize").backend == "reference"
    calls = []

    def quantize(view):
        calls.append(True)
        return torch.full(view.shape, 17, dtype=torch.uint8)

    monkeypatch.setattr(native.soup2, "enabled", lambda family: True)
    monkeypatch.setattr(native.soup2, "quant_u8", quantize)
    monkeypatch.setattr(npu, "load_npu_media", lambda: SimpleNamespace(rt=SimpleNamespace()))
    monkeypatch.setattr(npu, "_aclfast", None)
    monkeypatch.setattr(device_images, "jpeg_decoder_for",
                        lambda device, image_quantize: npu.NpuJpegDecoder(torch.device("cpu"), quantize=image_quantize))
    decoders = [device_images.DeviceImageDecoder("cpu", image_quantize=callback)._decode
                for callback in (native.image_quantize, table.image_quantize)]
    fmt = next(fmt for fmt, (h, w, _) in npu._CHROMA.items() if h == w == 1)
    meta = dict(fmt=fmt, h=2, w=2, hs=2, ws=2, luma_size=4, start=4, end=12, ch=2, cs=4)
    data = torch.full((12,), 128, dtype=torch.uint8)
    for decoder in decoders:
        monkeypatch.setattr(decoder, "_plane_meta", lambda *args: meta)
    for index, expected in ((0, 17), (1, 128), (0, 17)):
        out = decoders[index]._frame_view({}, 0, 12, 2, 2, data)
        assert torch.equal(out, torch.full((3, 2, 2), expected, dtype=torch.uint8))
    assert len(calls) == 2


def test_selected_quantizer_propagates_execution_errors(monkeypatch):
    from xinfer.ops.backends.ascend import image

    monkeypatch.setattr(image.soup2, "enabled", lambda family: True)

    def fail(view):
        raise RuntimeError("image quantizer launch failed")

    monkeypatch.setattr(image.soup2, "quant_u8", fail)
    with pytest.raises(RuntimeError, match="image quantizer launch failed"):
        image.image_quantize(torch.ones(3, 2, 2))

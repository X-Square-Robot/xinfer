"""Device-image coverage with generated JPEGs and raw RGB frames."""

import io
from contextlib import closing, nullcontext

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from xinfer.engine.device_images import DeviceImageDecoder
from xinfer.models.processing import device_image
from xinfer.platforms import resolve_platform


@pytest.mark.parametrize(("height", "width", "subsampling", "progressive"), [
    (96, 128, 2, False), (96, 128, 1, False), (96, 128, 0, False),
    (95, 127, 2, False), (96, 128, 2, True),
])
def test_jpegd_and_raw_resize_on_npu(height, width, subsampling, progressive):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("no NPU is available")
    platform = resolve_platform("npu")
    if not platform.capabilities.device_images:
        pytest.skip("NPU device images require CANN DVPP V2 JPEGD")
    image_module = pytest.importorskip("PIL.Image")

    y, x = np.indices((height, width))
    rgb = np.stack((2 * x, 2 * y, x + y), axis=-1).astype(np.uint8)
    with io.BytesIO() as buffer:
        image_module.fromarray(rgb).save(buffer, format="JPEG", quality=91, subsampling=subsampling,
                                         progressive=progressive)
        jpeg = buffer.getvalue()
    with image_module.open(io.BytesIO(jpeg)) as decoded:
        expected_jpeg = torch.from_numpy(np.array(decoded.convert("RGB")))

    warning = pytest.warns(UserWarning, match="decoded on the CPU") if progressive else nullcontext()
    with closing(DeviceImageDecoder(platform.device)) as decode, warning:
        views = decode({
            "jpeg": device_image(jpeg, (), ()),
            "raw": device_image(rgb, ((48, 64), (24, 32)), (("bicubic", True), ("bilinear", True))),
            "meta": "kept",
        })
    assert views["meta"] == "kept"
    assert views["jpeg"].device.type == views["raw"].device.type == "npu"
    assert views["jpeg"].dtype == views["raw"].dtype == torch.uint8
    assert views["jpeg"].is_contiguous() and views["raw"].is_contiguous()
    difference = (views["jpeg"].cpu().int() - expected_jpeg.int()).abs()
    assert difference.float().mean() <= 1 and difference.max() <= 4

    reference = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
    for size, mode in (((48, 64), "bicubic"), ((24, 32), "bilinear")):
        reference = F.interpolate(reference, size=size, mode=mode, align_corners=False, antialias=True)
        reference = (reference + 0.5).floor().clamp(0, 255)
    difference = (views["raw"].cpu().int() - reference[0].permute(1, 2, 0).int()).abs()
    assert difference.max() <= 1

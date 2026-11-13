"""``DeviceImage`` inputs decoded and resized on the device (``image_preprocess_device="gpu"``).

The engine runs ``DeviceImageDecoder`` on its device-image thread before a request's first stage, outside the CUDA
graphs (docs/engine.md). The views it produces have the shapes of the CPU path's views, so the graphs and their keys
are the same in both modes; the variable-length JPEG bytes never reach a graph. The pixels are not bitwise equal to
the CPU path's: nvJPEG and libjpeg-turbo decode differently, mostly around sharp chroma edges, and the resize computes
in FP32 instead of OpenCV's or Pillow's fixed point.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from xinfer.engine.executor import _to_device
from xinfer.models.interface import DeviceImage
from xinfer.platforms.devices import jpeg_decoder_for


class DeviceImageDecoder:
    """Turns the ``DeviceImage`` inputs of one request into ``[h, w, 3]`` uint8 views on the device.

    The platform owns JPEG decoding and its stream ordering. Raw frames use the shared upload path;
    both kinds of image use the same resize steps.

    Args:
        device: the accelerator device.

    Raises:
        ValueError: ``device`` has no device-image decoder.
    """

    def __init__(self, device):
        self._decode = jpeg_decoder_for(device)
        self.device = self._decode.device

    def close(self) -> None:
        """Release the platform decoder's resources on its owning thread."""
        self._decode.close()

    def __call__(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """``inputs`` with every ``DeviceImage`` replaced by its view on the device; other values are kept.

        JPEG decode completes before the resize steps, which run on the caller's stream. Raw frames upload on that
        stream. JPEGs are submitted together to the platform decoder.
        """
        images = {name: value for name, value in inputs.items() if isinstance(value, DeviceImage)}
        jpegs = [name for name, image in images.items() if image.data.dim() == 1]
        decoded = {}
        if jpegs:
            decoded = dict(zip(jpegs, self._decode([images[name].data for name in jpegs])))
        out = dict(inputs)
        for name, image in images.items():
            view = decoded[name] if name in decoded else _to_device(image.data, self.device).permute(2, 0, 1)
            out[name] = _resize(view, image)
        return out


def _resize(view: torch.Tensor, image: DeviceImage) -> torch.Tensor:
    """``[3, H, W]`` uint8 → ``[h, w, 3]`` uint8 through ``image``'s resize steps."""
    x = view.unsqueeze(0)
    for (mode, antialias), size in zip(image.resize, image.sizes):
        if tuple(size) != tuple(x.shape[-2:]):
            x = F.interpolate(x.float(), size=tuple(size), mode=mode, antialias=antialias, align_corners=False)
            x = x.add_(0.5).floor_().clamp_(0, 255).to(torch.uint8)
    return x[0].permute(1, 2, 0).contiguous()

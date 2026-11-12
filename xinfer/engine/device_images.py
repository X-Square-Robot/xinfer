"""``DeviceImage`` inputs decoded and resized on the device (``image_preprocess_device="gpu"``).

The engine runs ``DeviceImageDecoder`` on its device-image thread before a request's first stage, outside the CUDA
graphs (docs/engine.md). The views it produces have the shapes of the CPU path's views, so the graphs and their keys
are the same in both modes; the variable-length JPEG bytes never reach a graph. The pixels are not bitwise equal to
the CPU path's: the resize computes in FP32 instead of OpenCV's or Pillow's fixed point, and the decoded JPEGs differ
slightly from libjpeg-turbo's (nvImageCodec's by at most one level; CANN JPEGD's mostly around sharp chroma edges).
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
        out = dict(inputs)
        if self.device.type == "npu":
            # Per-view: a prefetched view (its ticket ran ahead of the request's admission item,
            # on this same thread) is taken from its slot -- the decode that ran there -- and any
            # view without a completed slot decodes here in its own DVPP transaction, the same
            # transaction per JPEG in the same order as the former single batched call.
            for name, image in images.items():
                slot = getattr(image, "prefetch", None)
                if (slot is not None and slot.device == self.device
                        and (slot.view is not None or slot.error is not None)):
                    if slot.error is not None:
                        raise slot.error
                    out[name] = slot.view
                else:
                    out[name] = self.decode_one(image)
            return out
        jpegs = [name for name, image in images.items() if image.data.dim() == 1]
        decoded = {}
        if jpegs:
            decoded = dict(zip(jpegs, self._decode([images[name].data for name in jpegs])))
        for name, image in images.items():
            view = decoded[name] if name in decoded else _to_device(image.data, self.device).permute(2, 0, 1)
            out[name] = _resize(view, image)
        return out

    def decode_one(self, image: DeviceImage) -> torch.Tensor:
        """One view's ``[h, w, 3]`` uint8 tensor on the NPU: its decode or upload, then the resize steps.

        The engine's per-view prefetch and the admission-time fallback share this path: a JPEG
        takes one synchronous DVPP V2 transaction (the same transaction the former batched call
        ran, one JPEG at a time), a raw frame uploads through ``_to_device`` on the caller's
        stream, and both then run ``_resize`` on that stream.
        """
        if self.device.type != "npu":
            raise RuntimeError("per-view decode is the NPU path; CUDA batches its JPEGs in __call__")
        if image.data.dim() == 1:
            view = self._decode([image.data])[0]
        else:
            view = _to_device(image.data, self.device).permute(2, 0, 1)
        return _resize(view, image)


def _resize(view: torch.Tensor, image: DeviceImage) -> torch.Tensor:
    """``[3, H, W]`` uint8 → ``[h, w, 3]`` uint8 through ``image``'s resize steps."""
    x = view.unsqueeze(0)
    for (mode, antialias), size in zip(image.resize, image.sizes):
        if tuple(size) != tuple(x.shape[-2:]):
            x = F.interpolate(x.float(), size=tuple(size), mode=mode, antialias=antialias, align_corners=False)
            x = x.add_(0.5).floor_().clamp_(0, 255).to(torch.uint8)
    return x[0].permute(1, 2, 0).contiguous()

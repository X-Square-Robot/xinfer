"""``DeviceImage`` inputs decoded and resized on the CUDA device (``image_preprocess_device="gpu"``).

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


class DeviceImageDecoder:
    """Turns the ``DeviceImage`` inputs of one request into ``[h, w, 3]`` uint8 views on the device.

    torchvision keeps one nvJPEG decoder and builds a new one whenever it is called with a different
    ``torch.device`` (``cuda`` and ``cuda:0`` differ), leaking the old one's resources, so every call passes the
    same device object. torchvision creates its decoder on the first call.

    Args:
        device: the CUDA device.

    Raises:
        ValueError: ``device`` is not a CUDA device.
        ImportError: torchvision is not installed.
    """

    def __init__(self, device):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f'DeviceImage inputs (image_preprocess_device "gpu") need a CUDA device, not {device}')
        try:
            from torchvision.io import ImageReadMode, decode_jpeg
        except ImportError as exc:
            raise ImportError('DeviceImage inputs (image_preprocess_device "gpu") need torchvision '
                              '(pip install xinfer[gpu-image])') from exc
        self.device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        self._decode = lambda jpegs: decode_jpeg(jpegs, mode=ImageReadMode.RGB, device=self.device)
        # Decoded views are allocated on this stream: torchvision writes them on a stream of its own that does not
        # wait for the caller's stream, so they must not reuse memory that work queued on the caller's stream may
        # still read. The caller's later reads are recorded on each view (record_stream).
        self._stream = torch.cuda.Stream(self.device)

    def __call__(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """``inputs`` with every ``DeviceImage`` replaced by its view on the device; other values are kept.

        All JPEGs of the request are decoded in one batched call, which blocks the calling thread until they are
        decoded; frames are uploaded on the current stream. The resize steps run on the current stream.
        """
        images = {name: value for name, value in inputs.items() if isinstance(value, DeviceImage)}
        jpegs = [name for name, image in images.items() if image.data.dim() == 1]
        decoded = {}
        if jpegs:
            current = torch.cuda.current_stream(self.device)
            with torch.cuda.stream(self._stream):
                views = self._decode([images[name].data for name in jpegs])
            current.wait_stream(self._stream)
            for name, view in zip(jpegs, views):
                view.record_stream(current)
                decoded[name] = view
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

"""Map request-owned DeviceImage values to resized device tensors.

Platform pipelines own codec resources, stream ordering and optional image-graph replay. The
selected image_resize operator supplies the numerical resize contract for both raw and decoded views.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xinfer.engine.executor import _to_device
from xinfer.models.interface import DeviceImage
from xinfer.platforms.devices import image_pipeline_for, jpeg_decoder_for
from xinfer.platforms.images import resize_image


class DeviceImageDecoder:
    """Decode views on the engine's image thread, using the selected platform and resize operator."""

    def __init__(self, device, image_resize=None, prefetch=None, image_quantize=None):
        self._decode = (jpeg_decoder_for(device) if image_quantize is None
                        else jpeg_decoder_for(device, image_quantize=image_quantize))
        self.device = self._decode.device
        self._resize = image_resize or resize_image
        self._pipeline = image_pipeline_for(self._decode, self._resize, _to_device)
        self._prefetch = prefetch

    def close(self) -> None:
        self._decode.close()

    def capture_image_graphs(self) -> int:
        return 0 if self._pipeline is None else self._pipeline.capture_image_graphs()

    @property
    def image_graph_stats(self) -> dict:
        return {} if self._pipeline is None else self._pipeline.image_graph_stats

    def __call__(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        images = {name: value for name, value in inputs.items() if isinstance(value, DeviceImage)}
        out = dict(inputs)
        if self._pipeline is not None:
            pending = []
            for name, image in images.items():
                slot = image.prefetch
                if (slot is not None and slot.owner is self._prefetch and slot.device == self.device
                        and (slot.view is not None or slot.error is not None)):
                    if slot.error is not None:
                        raise slot.error
                    out[name] = slot.view
                else:
                    pending.append(name)
            if pending:
                # A prepared request may be replayed through another engine. Its outstanding codec
                # transactions and stream-local results still belong to the original engine.
                views = self._pipeline.decode_many([
                    image if image.prefetch is None or image.prefetch.owner is self._prefetch
                    else DeviceImage(image.data, image.sizes, image.resize)
                    for image in (images[name] for name in pending)])
                out.update(zip(pending, views))
            return out
        jpegs = [name for name, image in images.items() if image.data.dim() == 1]
        decoded = {}
        if jpegs:
            decoded = dict(zip(jpegs, self._decode([images[name].data for name in jpegs])))
        for name, image in images.items():
            view = decoded[name] if name in decoded else _to_device(image.data, self.device).permute(2, 0, 1)
            out[name] = self._resize(view, image.sizes, image.resize)
        return out

    def start(self, image):
        return None if self._pipeline is None else self._pipeline.start(image)

    def start_batch(self, images):
        return None if self._pipeline is None else self._pipeline.start_batch(images)

    def decode_many(self, images):
        if self._pipeline is not None:
            return self._pipeline.decode_many(images)
        raise RuntimeError("the platform has no per-view decode pipeline")

    def decode_one(self, image):
        if self._pipeline is not None:
            return self._pipeline.decode_one(image)
        raise RuntimeError("the platform has no per-view decode pipeline")

"""``DeviceImage`` inputs decoded and resized on the device (``image_preprocess_device="gpu"``).

The engine runs ``DeviceImageDecoder`` on its device-image thread before a request's first stage, outside the CUDA
graphs (docs/engine.md). The views it produces have the shapes of the CPU path's views, so the graphs and their keys
are the same in both modes; the variable-length JPEG bytes never reach a graph. The pixels are not bitwise equal to
the CPU path's: the resize computes in FP32 instead of OpenCV's or Pillow's fixed point, and the decoded JPEGs differ
slightly from libjpeg-turbo's (nvImageCodec's by at most one level; CANN JPEGD's mostly around sharp chroma edges).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from xinfer.engine.executor import _to_device
from xinfer.engine.image_graphs import ResizeChainGraphs, chain_view
from xinfer.models.interface import DeviceImage
from xinfer.platforms.devices import jpeg_decoder_for

try:  # aa-resample (XINFER_AA_RESAMPLE, default off)
    from xinfer.ops.backends.ascend import aa_resample
except Exception:  # pragma: no cover - optional mount
    aa_resample = None


def _quant_u8(x):
    """SOUP2 quant knife: one kernel for add+floor+clamp+cast; falls back."""
    try:
        from xinfer.ops.backends.ascend import soup2

        if soup2.enabled("quant"):
            out = soup2.quant_u8(x)
            if out is not None:
                return out
    except Exception:  # noqa: BLE001 - any miss runs the released chain
        pass
    return x.add_(0.5).floor_().clamp_(0, 255).to(torch.uint8)


def _image_graphs_enabled() -> bool:
    """Whether the post-decode chain may be captured (``XINFER_IMAGE_GRAPHS``, default on)."""
    if os.environ.get("XINFER_IMAGE_GRAPHS", "1") not in ("1", "true", "True"):
        return False
    npu = getattr(torch, "npu", None)
    return callable(getattr(npu, "NPUGraph", None)) and callable(getattr(npu, "graph", None))


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
        # The YUV->RGB + resize chain of each view runs from a captured graph when the engine
        # opens the capture window at the end of warmup (see image_graphs).
        self._graphs = (ResizeChainGraphs(self.device, self._decode.acl)
                        if self.device.type == "npu" and _image_graphs_enabled() else None)

    def close(self) -> None:
        """Release the platform decoder's resources on its owning thread."""
        self._decode.close()

    def capture_image_graphs(self) -> int:
        """Capture the post-decode chain graphs; the engine calls this in a capture-safe window."""
        return 0 if self._graphs is None else self._graphs.capture_all()

    @property
    def image_graph_stats(self) -> dict:
        """Counters of the captured chain (``{}`` when the chain is not captured)."""
        return {} if self._graphs is None else dict(self._graphs.stats)

    def _to_view(self, decoded, image: DeviceImage) -> torch.Tensor:
        """One decoded payload -> the view ``[h, w, 3]`` uint8, through the captured chain if it has one.

        ``decoded`` is ``(plane buffer, plane meta)`` for a JPEG or an uploaded ``[3, H, W]`` tensor
        for a raw frame. The JPEG chain (YUV->RGB plus the resize steps) is the one the graph cache
        replays; a bucket without a graph runs :func:`image_graphs.chain_view` eagerly, the very
        function the graph was captured from.
        """
        if isinstance(decoded, tuple):
            data, meta = decoded
            out = self._graphs.run(data, meta, image.sizes, image.resize) if self._graphs is not None else None
            return out if out is not None else chain_view(data, meta, image.sizes, image.resize)
        return _resize(decoded, image)

    def __call__(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """``inputs`` with every ``DeviceImage`` replaced by its view on the device; other values are kept.

        JPEG decode completes before the resize steps, which run on the caller's stream. Raw frames upload on that
        stream. The CUDA decoder submits the request's JPEGs as one batch.
        """
        images = {name: value for name, value in inputs.items() if isinstance(value, DeviceImage)}
        out = dict(inputs)
        if self.device.type == "npu":
            # Per-view: a prefetched view (its ticket ran ahead of the request's admission item,
            # on this same thread) is taken from its slot -- the decode that ran there -- and every
            # view without a completed slot decodes here, all of them in one DVPP wave.
            pending = []
            for name, image in images.items():
                slot = getattr(image, "prefetch", None)
                if (slot is not None and slot.device == self.device
                        and (slot.view is not None or slot.error is not None)):
                    if slot.error is not None:
                        raise slot.error
                    out[name] = slot.view
                else:
                    pending.append(name)
            if pending:
                views = self.decode_many([images[name] for name in pending])
                for name, view in zip(pending, views):
                    out[name] = view
            return out
        jpegs = [name for name, image in images.items() if image.data.dim() == 1]
        decoded = {}
        if jpegs:
            decoded = dict(zip(jpegs, self._decode([images[name].data for name in jpegs])))
        for name, image in images.items():
            view = decoded[name] if name in decoded else _to_device(image.data, self.device).permute(2, 0, 1)
            out[name] = _resize(view, image)
        return out

    def start(self, image: DeviceImage):
        """Start one prefetched view's decode now (``None`` when there is nothing to start)."""
        slot = getattr(image, "prefetch", None)
        if self.device.type != "npu" or image.data.dim() != 1 or slot is None or slot.txn is not None:
            return None
        slot.txn = self._decode.start(image.data)
        return slot.txn

    def decode_many(self, images) -> list[torch.Tensor]:
        """Several views' ``[h, w, 3]`` uint8 tensors on the NPU, decoded as one DVPP wave.

        All JPEG views' channel transactions start before any frame is fetched, so the media engine
        decodes them concurrently; raw frames upload on the caller's stream. Each view's decoded
        planes then take the same YUV->RGB + resize chain as ``decode_one`` -- from its captured
        graph when the engine captured that shape bucket.
        """
        if self.device.type != "npu":
            raise RuntimeError("decode_many is the NPU path; CUDA batches its JPEGs in __call__")
        views = [None] * len(images)
        jpegs = [index for index, image in enumerate(images) if image.data.dim() == 1]
        if jpegs:
            started = []
            for index in jpegs:
                slot = getattr(images[index], "prefetch", None)
                started.append(slot.txn if slot is not None else None)
                if slot is not None:
                    slot.txn = None
            if self._graphs is not None:
                def replay(offset, payload):
                    # WAVE-TAIL WT1: each view's captured chain replays the moment its
                    # planes are fetched -- while the wave's larger views are still
                    # decoding -- instead of after the last one's fetch. The same chain
                    # (graph replay or eager) runs on the same stream with the same bytes;
                    # only the enqueue moment moves between fetches.
                    views[jpegs[offset]] = self._to_view(payload, images[jpegs[offset]])
                decoded = self._decode([images[index].data for index in jpegs], started,
                                       planes=True, emit=replay)
            else:
                decoded = self._decode([images[index].data for index in jpegs], started)
            for offset, payload in enumerate(decoded):
                index = jpegs[offset]
                if views[index] is None and payload is not None:
                    views[index] = self._to_view(payload, images[index])
        for index, image in enumerate(images):
            if views[index] is None:
                views[index] = _to_device(image.data, self.device).permute(2, 0, 1)
                views[index] = _resize(views[index], image)
        return views

    def start_batch(self, images) -> list | None:
        """Start several prefetched views' decodes as one batch; per image this is ``start``.

        ``None`` when there is nothing to start. A view whose chain failed carries its exception
        into its slot, exactly as ``start`` raising inside ``_start_view`` did.
        """
        if self.device.type != "npu":
            return None
        todo = []
        for image in images:
            slot = getattr(image, "prefetch", None)
            if (image.data.dim() != 1 or slot is None or slot.txn is not None
                    or slot.view is not None or slot.error is not None):
                continue
            todo.append((image, slot))
        if not todo:
            return None
        results = self._decode.start_batch([image.data for image, _ in todo])
        for (image, slot), txn in zip(todo, results):
            if isinstance(txn, Exception):
                slot.error = txn
            elif txn is not None:
                slot.txn = txn
        return results

    def decode_one(self, image: DeviceImage) -> torch.Tensor:
        """One view's ``[h, w, 3]`` uint8 tensor on the NPU: its decode or upload, then the resize steps.

        A JPEG takes one synchronous DVPP V2 transaction (the same transaction the former batched call
        ran, one JPEG at a time -- a wave of one), a raw frame uploads through ``_to_device`` on the
        caller's stream, and both then run ``_resize`` on that stream. A decoded JPEG's chain -- the
        RGB conversion plus the resize steps -- runs from its captured graph when the engine captured
        that shape bucket; a bucket without a graph runs :func:`image_graphs.chain_view` eagerly, the
        very function the graph was captured from.
        """
        if self.device.type != "npu":
            raise RuntimeError("per-view decode is the NPU path; CUDA batches its JPEGs in __call__")
        if image.data.dim() == 1:
            slot = getattr(image, "prefetch", None)
            started = [slot.txn if slot is not None else None]
            if slot is not None:
                slot.txn = None
            decoded = self._decode([image.data], started, planes=self._graphs is not None)[0]
            if isinstance(decoded, tuple):
                return self._to_view(decoded, image)
            view = decoded
        else:
            view = _to_device(image.data, self.device).permute(2, 0, 1)
        return _resize(view, image)


def _resize(view: torch.Tensor, image: DeviceImage) -> torch.Tensor:
    """``[3, H, W]`` uint8 → ``[h, w, 3]`` uint8 through ``image``'s resize steps."""
    x = view.unsqueeze(0)
    for (mode, antialias), size in zip(image.resize, image.sizes):
        if tuple(size) != tuple(x.shape[-2:]):
            if (
                aa_resample is not None and mode == "bicubic" and antialias
                and x.dtype == torch.uint8
            ):
                routed = aa_resample.step(x, tuple(size))
                if routed is not None:
                    x = routed
                    continue
            x = F.interpolate(x.float(), size=tuple(size), mode=mode, antialias=antialias, align_corners=False)
            x = _quant_u8(x)
    return x[0].permute(1, 2, 0).contiguous()

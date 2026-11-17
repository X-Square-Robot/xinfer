"""Ascend image decode waves and post-decode graph replay.

The caller supplies the selected resize operator and upload function. This module owns only
platform codec/graph resources and never imports a model, engine or operator backend.
"""

from __future__ import annotations

import os

import torch

from xinfer.platforms.npu_image_graphs import ResizeChainGraphs, chain_view


def _image_graphs_enabled() -> bool:
    return (os.environ.get("XINFER_IMAGE_GRAPHS", "1") in ("1", "true", "True")
            and callable(getattr(torch.npu, "NPUGraph", None))
            and callable(getattr(torch.npu, "graph", None)))


class NpuImagePipeline:
    """The image pipeline of one NPU decoder, used only on its owning image thread."""

    def __init__(self, decoder, resize_image, upload):
        self._decode = decoder
        self.device = decoder.device
        self._resize = resize_image
        self._upload = upload
        self._graphs = (ResizeChainGraphs(self.device, decoder.acl, resize_image, decoder.quantize)
                        if _image_graphs_enabled() else None)

    def capture_image_graphs(self) -> int:
        """Capture the post-decode chain graphs; the engine calls this in a capture-safe window."""
        return 0 if self._graphs is None else self._graphs.capture_all()

    @property
    def image_graph_stats(self) -> dict:
        """Counters of the captured chain (``{}`` when the chain is not captured)."""
        return {} if self._graphs is None else dict(self._graphs.stats)

    def _to_view(self, decoded, image) -> torch.Tensor:
        """One decoded payload -> the view ``[h, w, 3]`` uint8, through the captured chain if it has one.

        ``decoded`` is ``(plane buffer, plane meta)`` for a JPEG or an uploaded ``[3, H, W]`` tensor
        for a raw frame. The JPEG chain (YUV->RGB plus the resize steps) is the one the graph cache
        replays; a bucket without a graph runs :func:`image_graphs.chain_view` eagerly, the very
        function the graph was captured from.
        """
        if isinstance(decoded, tuple):
            data, meta = decoded
            out = self._graphs.run(data, meta, image.sizes, image.resize) if self._graphs is not None else None
            return out if out is not None else chain_view(
                data, meta, image.sizes, image.resize, self._resize, self._decode.quantize)
        return self._resize(decoded, image.sizes, image.resize)

    def start(self, image):
        """Start one prefetched view's decode now (``None`` when there is nothing to start)."""
        slot = getattr(image, "prefetch", None)
        if image.data.dim() != 1 or slot is None or slot.txn is not None:
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
                    # Replay each fetched view while the larger views are still decoding.
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
                views[index] = self._upload(image.data, self.device).permute(2, 0, 1)
                views[index] = self._resize(views[index], image.sizes, image.resize)
        return views

    def start_batch(self, images) -> list | None:
        """Start several prefetched views' decodes as one batch; per image this is ``start``.

        ``None`` when there is nothing to start. A view whose chain failed carries its exception
        into its slot, exactly as ``start`` raising inside ``_start_view`` did.
        """
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

    def decode_one(self, image) -> torch.Tensor:
        """One view's ``[h, w, 3]`` uint8 tensor on the NPU: its decode or upload, then the resize steps.

        A JPEG takes one synchronous DVPP V2 transaction (the same transaction the former batched call
        ran, one JPEG at a time -- a wave of one), a raw frame uploads through ``_to_device`` on the
        caller's stream, and both then run ``_resize`` on that stream. A decoded JPEG's chain -- the
        RGB conversion plus the resize steps -- runs from its captured graph when the engine captured
        that shape bucket; a bucket without a graph runs :func:`image_graphs.chain_view` eagerly, the
        very function the graph was captured from.
        """
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
            view = self._upload(image.data, self.device).permute(2, 0, 1)
        return self._resize(view, image.sizes, image.resize)

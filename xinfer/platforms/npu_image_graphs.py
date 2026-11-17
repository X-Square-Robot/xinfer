"""Capture the decoded-plane conversion and selected resize chain in NPU graphs.

The graph retains static input/output addresses and copies each replay's result into a fresh
request-owned tensor. Capture runs only during the engine's warmup window: this runtime's graph
entry synchronizes device-wide, so it cannot overlap another thread's capture or synchronization.
Unseen buckets continue through the same eager chain.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

import torch

from xinfer.platforms.npu import _CHROMA, _DEVICE_TO_DEVICE, _call, _rgb

logger = logging.getLogger(__name__)


def chain_view(data, meta, sizes, resize, resize_image, quantize_image):
    """Convert decoded planes and run the same selected resize operator eagerly or in a graph."""
    luma = data[:meta["luma_size"]].view(meta["hs"], meta["ws"])[:meta["h"], :meta["w"]]
    chroma = None
    if meta["fmt"] in _CHROMA:
        chroma = data[meta["start"]:meta["end"]].view(meta["ch"], meta["cs"])
    return resize_image(_rgb(luma, chroma, meta["fmt"], quantize=quantize_image), sizes, resize)


def bucket_key(meta, sizes, resize):
    """What a captured chain depends on: the decoded buffer geometry and the resize recipe."""
    return (meta["out_size"], meta["fmt"], meta["h"], meta["w"], meta["hs"], meta["ws"],
            meta["luma_size"], meta["ch"], meta["cs"], meta["start"], meta["end"],
            tuple(tuple(size) for size in sizes), tuple(tuple(step) for step in resize))


def out_shape(meta, sizes):
    """``[h, w, 3]`` of the last resize step of a view whose decoded size is ``meta``."""
    height, width = meta["h"], meta["w"]
    for size in sizes:
        height, width = int(size[0]), int(size[1])
    return height, width, 3


class _Captured:
    """One bucket: its graph, the static input the graph reads and the pooled tensor it writes."""

    __slots__ = ("graph", "static_in", "out")

    def __init__(self, graph, static_in, out):
        self.graph = graph
        self.static_in = static_in
        self.out = out


class ResizeChainGraphs:
    """Graph replays of :func:`chain_view`, one graph per shape bucket.

    Attributes:
        stats: counters ``captures``, ``replays``, ``eager`` (no graph for the bucket),
            ``misses`` (a bucket first seen after the capture window) and ``capture_ms``.
    """

    def __init__(self, device, acl, resize_image, quantize_image, max_graphs: int = 8):
        self.device = device
        self._acl = acl
        self._resize = resize_image
        self._quantize = quantize_image
        self._max_graphs = max_graphs
        self._pending: OrderedDict = OrderedDict()   # key -> (meta, sizes, resize), seen eagerly
        self._graphs: OrderedDict = OrderedDict()    # key -> _Captured
        self._pool = None
        self._stream = None
        self._window_closed = False
        self._failed = False
        self._missing: set = set()
        self._copy = callable(getattr(acl.rt, "memcpy_async", None))
        self.stats = dict(captures=0, replays=0, eager=0, misses=0, capture_ms=0.0)

    # ------------------------------------------------------------------ capture
    def note(self, meta, sizes, resize) -> None:
        """Record a bucket seen on the eager path so :meth:`capture_all` can capture it."""
        if self._window_closed or self._failed:
            return
        key = bucket_key(meta, sizes, resize)
        if key in self._graphs or key in self._pending:
            return
        if len(self._pending) >= self._max_graphs:
            logger.warning("device image chain: more than %d shape buckets, %r runs eagerly",
                           self._max_graphs, key)
            return
        self._pending[key] = (dict(meta), tuple(tuple(s) for s in sizes), tuple(tuple(s) for s in resize))

    def capture_all(self) -> int:
        """Capture one graph per recorded bucket and close the capture window.

        Call it only when no other thread can be capturing or synchronizing the device (the engine
        calls it from the device-image thread while its own thread waits): graph capture is
        device-wide here, and ACL rejects any other stream's synchronize during it.
        """
        self._window_closed = True
        if self._failed or not self._pending:
            self._pending.clear()
            return 0
        captured = 0
        try:
            if self._pool is None:
                self._pool = torch.npu.graph_pool_handle()
                self._stream = torch.npu.Stream()
            for key, (meta, sizes, resize) in list(self._pending.items()):
                try:
                    self._capture(meta, sizes, resize)
                    captured += 1
                except Exception:  # noqa: BLE001 - one bad bucket must not cost the others
                    logger.exception("device image chain: could not capture %r; it stays eager", key)
        finally:
            self._pending.clear()
        if captured:
            logger.info("device image chain: captured %d graphs in %.1f ms; outputs %s",
                        captured, self.stats["capture_ms"],
                        [tuple(c.out.shape) for c in self._graphs.values()])
        return captured

    def _capture(self, meta, sizes, resize) -> None:
        start = time.perf_counter()
        static_in = torch.zeros(meta["out_size"], dtype=torch.uint8, device=self.device)
        # Stream-local operator buffers must be created on the stream captured below.
        self._stream.wait_stream(torch.npu.current_stream(self.device))
        with torch.inference_mode(), torch.npu.stream(self._stream):
            warm_output = chain_view(static_in, meta, sizes, resize, self._resize, self._quantize)
        self._stream.synchronize()
        del warm_output
        graph = torch.npu.NPUGraph()
        # inference_mode, as LaneCUDAGraphs captures the stage graphs: torch_npu's capture_begin
        # updates a graph-bookkeeping tensor in place, and once the engine's own captures (inside
        # inference mode) have run in this process that tensor carries the inference bit -- an
        # in-place update of it outside inference mode is rejected
        # ("Inplace update to inference tensor outside InferenceMode is not allowed").
        with torch.inference_mode(), torch.npu.graph(graph, pool=self._pool, stream=self._stream,
                                                     capture_error_mode="thread_local"):
            out = chain_view(static_in, meta, sizes, resize, self._resize, self._quantize)
        shape = out_shape(meta, sizes)
        if tuple(out.shape) != shape or out.dtype != torch.uint8:
            raise RuntimeError("captured chain returned %s/%s, expected %s/uint8"
                               % (tuple(out.shape), out.dtype, shape))
        self._graphs[bucket_key(meta, sizes, resize)] = _Captured(graph, static_in, out)
        self.stats["captures"] += 1
        self.stats["capture_ms"] += (time.perf_counter() - start) * 1000

    # ---------------------------------------------------------------------- run
    def run(self, data, meta, sizes, resize):
        """The captured chain's output for one decoded plane buffer, or ``None`` (run eagerly).

        Returns a fresh tensor holding the bytes: the graph writes into its pooled output, which a
        later replay overwrites, so the bytes are copied out on the caller's stream before the next
        view's replay is enqueued.
        """
        key = bucket_key(meta, sizes, resize)
        captured = self._graphs.get(key)
        if captured is None:
            self.stats["eager"] += 1
            if self._window_closed or self._failed:
                self.stats["misses"] += 1
                if key not in self._missing:
                    self._missing.add(key)
                    logger.warning("device image chain: shape bucket %r appeared after the capture window; "
                                   "it runs eagerly", key)
            else:
                self.note(meta, sizes, resize)
            return None
        self._copy_into(captured.static_in, data)
        captured.graph.replay()
        out = torch.empty(tuple(captured.out.shape), dtype=torch.uint8, device=self.device)
        self._copy_into(out, captured.out)
        self.stats["replays"] += 1
        return out

    def _copy_into(self, dst, src) -> None:
        """Stream-ordered device-to-device copy of ``src``'s bytes into ``dst``."""
        if self._copy:
            _call(self._acl.rt.memcpy_async, dst.data_ptr(), dst.numel(), src.data_ptr(), src.numel(),
                  _DEVICE_TO_DEVICE, torch.npu.current_stream().npu_stream)
        else:
            dst.copy_(src)

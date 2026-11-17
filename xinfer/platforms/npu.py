"""The Ascend device's platform: TorchNPU discovery and CANN JPEG decoding.

CANN DVPP V2 exposes JPEGD through its ``vdec_*`` channel APIs. ``NpuJpegDecoder`` selects the JPEG payload
in frame mode to decode still images into CHW uint8 RGB tensors.

The platform owns codec buffers, channels and session lifetime, just as ``platforms/cuda.py`` owns the
nvJPEG adapter and its stream ordering. Both implement the tensor-only ``jpeg_decoder_for`` contract;
the engine handles ``DeviceImage`` mapping, raw uploads and resize. The capability probe here checks
the same DVPP bindings the decoder uses, keeping device runtime details behind the platform boundary.
"""

from __future__ import annotations

import importlib
import io
import re
import warnings
from threading import Lock

import torch
import torch.nn.functional as F

from xinfer.platforms.devices import device_module_for
from xinfer.platforms.images import quantize_image

try:  # HOST-TAIL HT1: GIL-releasing wrappers for the hot acl host calls
    from xinfer import _aclfast
except ImportError:  # not built / frontend never imports this module
    _aclfast = None
from xinfer.platforms.interface import Platform, PlatformCapabilities

_MEDIA_APIS = (
    "sys_init", "sys_exit", "dvpp_get_image_info", "dvpp_malloc", "dvpp_free",
    "vdec_create_chn", "vdec_destroy_chn", "vdec_set_jpegd_precision_mode", "vdec_start_recv_stream",
    "vdec_stop_recv_stream", "vdec_send_stream", "vdec_get_frame", "vdec_release_frame",
)


def load_npu_media():
    """Return Python ACL when it exposes the required media and runtime bindings."""
    try:
        acl = importlib.import_module("acl")
    except (ImportError, OSError):
        return None
    media = getattr(acl, "himpi", None)
    runtime = getattr(acl, "rt", None)
    if callable(getattr(runtime, "memcpy", None)) and all(
        callable(getattr(media, name, None)) for name in _MEDIA_APIS
    ):
        return acl
    return None


def npu_device_images_available() -> bool:
    """Whether the installed Python ACL exposes the required DVPP V2 bindings."""
    return load_npu_media() is not None


def npu_stream_priority_range(module, index):
    """A verifiable TorchNPU priority range, or None when the runtime cannot report it.

    Some TorchNPU releases accept the constructor argument but expose neither a working range query nor
    priority readback. Constructor acceptance alone cannot establish that the runtime honors priorities.
    """
    with module.device(index):
        stream = module.current_stream(index)
        try:
            least, greatest = module.Stream.priority_range()
            current = stream.priority
        except (AttributeError, NotImplementedError, RuntimeError):
            return None
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in (least, greatest, current))
            or not greatest <= current <= least or not greatest <= 0 <= least or greatest == least):
        return None
    return greatest, least


def npu_platform(device: str) -> Platform:
    """Resolve an explicit Ascend device and its supported runtime features."""
    try:
        module = device_module_for(device)
    except ImportError as exc:
        raise RuntimeError(f"device {device!r} requires TorchNPU and its matching CANN runtime") from exc
    if not module.is_available():
        raise RuntimeError(f"device {device!r} requested but no NPU is available")

    index = torch.device(device).index
    if index is None:
        index = module.current_device()
    count = module.device_count()
    if not 0 <= index < count:
        raise ValueError(f"NPU device index {index} is out of range for {count} visible cards")
    props = module.get_device_properties(index)
    name = str(getattr(props, "name", "") or "Ascend")
    arch = re.sub(r"[^a-z0-9]", "", name.lower()) or "ascend"
    memory = getattr(props, "total_memory", None)
    graphs = all(callable(getattr(module, name, None)) for name in ("NPUGraph", "graph", "graph_pool_handle"))
    priority_range = npu_stream_priority_range(module, index)
    return Platform(
        # The platform is named after its hardware and backend (``ascend``), as ``cuda-<arch>`` is on the other
        # side; the torch device below stays ``npu:N``. The vendor's own name already starts with "ascend"
        # (``ascend950dt9582``), so the family prefix is dropped instead of yielding ``ascend-ascend950dt9582``.
        # docs/design.md §3.1 gives the rule.
        name=f"ascend-{arch.removeprefix('ascend')}",
        device=f"npu:{index}",
        capabilities=PlatformCapabilities(
            device_type="ascend",
            arch=arch,
            cudagraphs=graphs,
            stream_priorities=priority_range is not None,
            stream_priority_range=priority_range,
            unified_memory=False,
            multiprocess_service=None,
            sm_count=getattr(props, "cube_core_num", None) or getattr(props, "multi_processor_count", None),
            total_memory_bytes=int(memory) if memory is not None else None,
            device_images=npu_device_images_available(),
        ),
    )


# CANN's hi_payload_type, hi_vdec_send_mode, hi_pixel_format and aclrtMemcpyKind.
_JPEG = 26
_SEND_FRAME = 1
_ALIGN_UP = 1
_SOURCE_FORMAT = 10000
_HOST_TO_DEVICE = 1
_DEVICE_TO_DEVICE = 3
_CHANNEL_EXISTS = 0xA0058004
_FORMAT_UNSUPPORTED = 0xA0058008
_TIMEOUT_MS = 3000
_MAX_CHANNELS = 16
# Channel transactions a wave may keep in flight on one channel before the oldest is fetched:
# ``vdec_send_stream`` returns HI_ERR_VDEC_BUF_FULL (0xa005800f) once the channel's stream buffer
# cannot hold another pending stream, so a wave also tracks the pending JPEG bytes per channel.
_MAX_INFLIGHT = 3
_MEDIA_LOCK = Lock()
_MEDIA_USERS: dict[torch.device, int] = {}


def _check(status, operation):
    if status != 0:
        raise RuntimeError(f"CANN JPEGD {operation} failed (0x{status & 0xFFFFFFFF:08x})")


def _call(operation, *args):
    _check(operation(*args), operation.__name__)


# hi_pixel_format: the chroma subsampling factors (height, width) and order.
_CHROMA = {
    1: (2, 2, False), 2: (2, 2, True),        # YUV/YVU420SP
    3: (1, 2, False), 4: (1, 2, True),        # YUV/YVU422SP
    5: (1, 1, False), 6: (1, 1, True),        # YUV/YVU444SP
    1000: (2, 1, False), 1001: (2, 1, True),  # YUV/YVU440SP
}

def _rgb(luma: torch.Tensor, chroma: torch.Tensor | None, pixel_format: int, *,
         quantize=quantize_image) -> torch.Tensor:
    """Unpack a cropped Y plane and interleaved chroma plane as CHW uint8 RGB."""
    if chroma is None:
        return luma.expand(3, -1, -1).contiguous()
    height, width = luma.shape
    step_h, step_w, vu_order = _CHROMA[pixel_format]
    chroma_h = (height + step_h - 1) // step_h
    chroma_w = (width + step_w - 1) // step_w
    uv = chroma[:chroma_h, :2 * chroma_w].reshape(chroma_h, chroma_w, 2).permute(2, 0, 1).float()
    if (step_h, step_w) != (1, 1):
        uv = F.interpolate(uv[None], size=(chroma_h * step_h, chroma_w * step_w),
                           mode="bilinear", align_corners=False)[0, :, :height, :width]
    u, v = uv[1 if vu_order else 0] - 128, uv[0 if vu_order else 1] - 128
    y = luma.float()
    rgb = torch.stack((y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u))
    return quantize(rgb)


def rgb_of(data, meta, *, quantize=quantize_image):
    """``[3, H, W]`` uint8 RGB of one decoded plane buffer (``meta`` as ``_plane_meta`` builds it)."""
    luma = data[:meta["luma_size"]].view(meta["hs"], meta["ws"])[:meta["h"], :meta["w"]]
    chroma = None
    if meta["fmt"] in _CHROMA:
        chroma = data[meta["start"]:meta["end"]].view(meta["ch"], meta["cs"])
    return _rgb(luma, chroma, meta["fmt"], quantize=quantize)


class NpuJpegDecoder:
    """Decode JPEGs with a lazy DVPP lease shared by decoders on the same device."""

    def __init__(self, device: torch.device, *, quantize=quantize_image):
        self.acl = load_npu_media()
        if self.acl is None:
            raise RuntimeError('DeviceImage inputs (image_preprocess_device "gpu") need CANN Python '
                               'ACL with DVPP V2 JPEGD on NPU')
        self.device = device
        self.quantize = quantize
        self._leased = False
        self._closed = False
        # Persistent VDEC channels, one per (width, height, stream-buffer) shape: a fresh
        # channel costs ~1.5 ms inside the compiled binding and the per-JPEG create/destroy
        # made every decode pay it while holding the GIL (probe-gil-r12 iso). Channels keep
        # exactly the attributes the per-JPEG creation used, so every decode runs the same
        # send/get/release transaction; a channel that fails mid-transaction is dropped and
        # recreated by the next decode, and ``close`` tears them all down.
        self._channels: dict[tuple[int, int, int], int] = {}
        # HOST-TAIL HT1: the compiled acl binding holds the interpreter lock for
        # every himpi/rt call (~0.4 ms per request across the decode chain); when
        # _aclfast resolved the same native symbols it runs the identical call
        # chain under Py_BEGIN_ALLOW_THREADS. Any miss -> binding path below.
        self._fast = _aclfast is not None and bool(_aclfast.init())
        # Frames are copied out of DVPP memory with a stream-ordered copy, so a
        # transaction's DVPP buffers cannot be freed before that copy ran: they
        # wait in ``_deferred`` until the recorded copy event completed.
        self._async_copy = callable(getattr(self.acl.rt, "memcpy_async", None))
        self._deferred: list[tuple[int, int]] = []
        self._copy_event = None
        self._free_error = None
        self._copy_unfenced = False
        # Transactions started by ``start`` and not fetched yet: channel key -> (count, jpeg bytes).
        self._pending: dict[tuple[int, int, int], tuple[int, int]] = {}
        # WAVE-TAIL WT2b: this decoder's hot paths enter the device context once
        # per call on one thread; the cached context object skips its rebuild.
        self._device_ctx = None

    def _on_device(self):
        """The cached device context of this decoder (single thread; entered and exited in pairs)."""
        if self._device_ctx is None:
            self._device_ctx = torch.npu.device(self.device)
        return self._device_ctx

    def _flush_deferred(self) -> None:
        """Free the DVPP buffers of finished frames once their copies completed."""
        if self._free_error is not None:
            raise self._free_error
        if not self._deferred:
            return
        if self._copy_unfenced:
            # Event creation/recording failed after copies were queued. An older event cannot
            # fence those reads; retain the buffers until device completion is established.
            torch.npu.synchronize()
            self._copy_unfenced = False
        if self._copy_event is not None:
            if not self._copy_event.query():
                self._copy_event.synchronize()
            self._copy_event = None
        try:
            if self._fast and self._deferred:
                pointers = [pointer for pair in self._deferred for pointer in pair]
                if _aclfast.free_batch(pointers) != len(pointers):
                    raise RuntimeError("CANN JPEGD dvpp_free failed")
            else:
                media = self.acl.himpi
                for source, output in self._deferred:
                    _call(media.dvpp_free, source)
                    _call(media.dvpp_free, output)
        except Exception as exc:
            # The batch reports a success count, not which pointers were freed. A sequential
            # failure can also follow successful frees. Retain ownership and never retry an
            # unknown subset; completion-query failures above remain safely retryable.
            self._free_error = exc
            raise
        self._deferred.clear()

    def _record_copy_event(self) -> None:
        """Mark the point after the frame copies on the caller's stream."""
        self._copy_unfenced = True
        self._copy_event = None
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        self._copy_event = event
        self._copy_unfenced = False

    def __call__(self, jpegs, prestarted=None, planes=False, emit=None):
        """Decode a wave of JPEGs; ``prestarted[i]`` is a transaction already begun by ``start``.

        Every image's DVPP transaction starts first (the host copy plus ``vdec_send_stream``),
        then each image's frame is fetched and converted: the media engine decodes the wave on
        the cached channels concurrently instead of one image after the other, which is what a
        one-at-a-time decode paid for a request's three views (~3.6 ms summed serially, ~1.2 ms
        with the transactions started together). Every image still runs the same
        send/get/release transaction on a channel with the same attributes, so its decoded
        pixels are unchanged -- only the order in which the channel transactions are issued
        changes.

        With ``planes`` each result is ``(buffer, meta)`` -- the flat decoded planes copied onto
        the caller's stream and the geometry of the views inside them -- instead of the RGB view,
        so a caller can run the RGB + resize chain as one captured graph
        (:mod:`xinfer.platforms.npu_image_graphs`). The bytes are the same either way.

        With ``emit`` the wave's drain runs smallest decoded buffer first and calls
        ``emit(index, result)`` the moment each view's planes are fetched, so the caller replays
        that view's post-decode chain while the wave's larger views are still decoding. Views of
        one channel keep their start order (``vdec_get_frame`` hands a channel's frames out in
        the order it decoded them), and the drain's stable size sort never reorders equal-size
        views, so every transaction still fetches exactly its own frame -- only the interleaving
        of independent channels' fetches changes.
        """
        jpegs = list(jpegs)
        prestarted = list(prestarted) if prestarted is not None else [None] * len(jpegs)
        with _MEDIA_LOCK:
            if self._closed:
                raise RuntimeError("CANN JPEGD decoder is closed")
            with self._on_device():
                self._flush_deferred()
                if not self._leased:
                    users = _MEDIA_USERS.get(self.device, 0)
                    # Repeated sys_init/sys_exit cycles exhaust HDC descriptors on CANN 9.1.1.
                    if users == 0:
                        _call(self.acl.himpi.sys_init)
                    _MEDIA_USERS[self.device] = users + 1
                    self._leased = True
                results = [None] * len(jpegs)
                started = []      # [(index, context)] in start order
                # Keep active channel leases visible to the cache until their frames are fetched.
                pending = self._pending
                done = False
                try:
                    for index, jpeg in enumerate(jpegs):
                        jpeg = jpeg.contiguous()
                        prestarted_here = prestarted[index] is not None
                        context = prestarted[index] if prestarted_here else self._begin(jpeg)
                        if context is None:
                            results[index] = self._decode_on_host(jpeg)
                            continue
                        started.append((index, context))
                        if prestarted_here:
                            # A prestarted transaction is already counted in the ``_pending``
                            # snapshot this drain took; counting it again would trip the
                            # per-channel in-flight budget and finish the first view before the
                            # drain began.
                            continue
                        key = self._channel_key(context["width"], context["height"], context["size"])
                        count, size = pending.get(key, (0, 0))
                        pending[key] = (count + 1, size + context["size"])
                        if count + 1 > _MAX_INFLIGHT or size + context["size"] > key[2]:
                            index, result = self._finish_oldest(started, pending, results, planes)
                            if emit is not None and result is not None:
                                emit(index, result)
                    if emit is not None and len(started) > 1:
                        # Drain the smallest decoded buffers first so their post-decode chains
                        # overlap the largest view's remaining decode.
                        started.sort(key=lambda item: item[1]["out_size"])
                    while started:
                        index, result = self._finish_oldest(started, pending, results, planes)
                        if emit is not None and result is not None:
                            emit(index, result)
                    done = True
                finally:
                    try:
                        if not done:
                            for _, context in started:
                                self._abandon(context)
                    finally:
                        if self._deferred:
                            self._record_copy_event()
                return results

    def _finish_oldest(self, started, pending, results, planes=False):
        """Fetch and convert the transaction started longest ago, freeing its channel budget."""
        index, context = started.pop(0)
        key = self._channel_key(context["width"], context["height"], context["size"])
        count, size = pending[key]
        if count == 1:
            pending.pop(key)
        else:
            pending[key] = (count - 1, size - context["size"])
        results[index] = self._finish(context, planes)
        return index, results[index]

    def _release_pending(self, key, size):
        count, total = self._pending.get(key, (0, 0))
        if count <= 1:
            self._pending.pop(key, None)
        else:
            self._pending[key] = (count - 1, total - size)

    def start(self, jpeg):
        """Begin one JPEG's channel transaction now; a later ``__call__`` fetches its frame.

        The media engine decodes while the caller keeps working, so a view whose bytes are ready
        early does not wait for the rest of its wave. Returns the transaction, or ``None`` when
        the image needs the host decoder or its channel already holds as much pending input as
        its stream buffer takes; the wave that fetches it starts it then, under the same budget.
        """
        jpeg = jpeg.contiguous()
        with _MEDIA_LOCK:
            if self._closed:
                raise RuntimeError("CANN JPEGD decoder is closed")
            with self._on_device():
                self._flush_deferred()
                if not self._leased:
                    users = _MEDIA_USERS.get(self.device, 0)
                    if users == 0:
                        _call(self.acl.himpi.sys_init)
                    _MEDIA_USERS[self.device] = users + 1
                    self._leased = True
                probed = self._probe(jpeg)
                if probed is None:
                    return None
                info, stream = probed
                size = jpeg.numel()
                key = self._channel_key(info["width"], info["height"], size)
                count, pending_bytes = self._pending.get(key, (0, 0))
                if count + 1 > _MAX_INFLIGHT or pending_bytes + size > key[2]:
                    return None
                self._pending[key] = (count + 1, pending_bytes + size)
                try:
                    return self._launch(jpeg, info, stream)
                except Exception:
                    self._release_pending(key, size)
                    raise

    def start_batch(self, jpegs):
        """Begin several JPEGs' channel transactions now; a later ``__call__`` fetches the frames.

        Per image this is ``start`` -- ``None`` when the image needs the host decoder or its
        channel already holds as much pending input as its stream buffer takes, the same budget
        bookkeeping, the same error semantics -- but the startable views' [malloc + upload +
        malloc + send] chains run under one GIL-releasing ``_aclfast.start3`` call, in list order,
        the order per-ticket ``start`` calls would have issued them. What changes is only that
        the prepare window's interpreter slices cannot interleave between the chains; the sends
        reach the media engine exactly as before. A list entry may be an ``Exception`` when that
        view's chain failed mid-way (the caller fails that view's request with it, as ``start``
        raising would have).
        """
        jpegs = list(jpegs)
        results = [None] * len(jpegs)
        if not jpegs:
            return results
        with _MEDIA_LOCK:
            if self._closed:
                raise RuntimeError("CANN JPEGD decoder is closed")
            with self._on_device():
                self._flush_deferred()
                if not self._leased:
                    users = _MEDIA_USERS.get(self.device, 0)
                    if users == 0:
                        _call(self.acl.himpi.sys_init)
                    _MEDIA_USERS[self.device] = users + 1
                    self._leased = True
                ready = []   # [(position, jpeg, info, stream, channel)]
                try:
                    for index, jpeg in enumerate(jpegs):
                        jpeg = jpeg.contiguous()
                        probed = self._probe(jpeg)
                        if probed is None:
                            continue
                        info, stream = probed
                        size = jpeg.numel()
                        key = self._channel_key(info["width"], info["height"], size)
                        count, pending_bytes = self._pending.get(key, (0, 0))
                        if count + 1 > _MAX_INFLIGHT or pending_bytes + size > key[2]:
                            continue
                        channel = self._channel_for(info["width"], info["height"], size)
                        ready.append((index, jpeg, info, stream, channel))
                        self._pending[key] = (count + 1, pending_bytes + size)
                except Exception:
                    # No stream has been submitted while this list is being prepared.
                    for _, jpeg, info, _, _ in ready:
                        self._release_pending(self._channel_key(info["width"], info["height"],
                                                                jpeg.numel()), jpeg.numel())
                    raise
                if not ready:
                    return results
                if (self._fast and len(ready) == 3
                        and getattr(_aclfast, "start3", None) is not None):
                    started = _aclfast.start3(
                        self.device.index,
                        *[(channel, jpeg.data_ptr(), jpeg.numel(), info["img_buf_size"],
                           info["width"], info["height"], info["width_stride"],
                           info["height_stride"], _SOURCE_FORMAT)
                          for _, jpeg, info, _stream, channel in ready],
                        _TIMEOUT_MS)
                    if started is not None:
                        for (index, jpeg, info, _stream, channel), outcome in zip(ready, started):
                            status, phase, source, output = outcome
                            if status == 0:
                                results[index] = dict(channel=channel, source=source, output=output,
                                                      out_size=info["img_buf_size"],
                                                      width=info["width"], height=info["height"],
                                                      size=jpeg.numel())
                                continue
                            operation = ("dvpp_malloc input", "memcpy", "dvpp_malloc output",
                                         "vdec_send_stream")[phase]
                            self._release_pending(self._channel_key(info["width"], info["height"],
                                                                    jpeg.numel()), jpeg.numel())
                            if phase == 3:
                                self._drop_channel(self._channel_key(info["width"], info["height"],
                                                                     jpeg.numel()))
                                _aclfast.free_batch([source, output])
                            # this view's request fails with it, as ``start`` raising would have
                            results[index] = RuntimeError(
                                f"CANN JPEGD {operation} failed (0x{status & 0xFFFFFFFF:08x})")
                        return results
                for index, jpeg, info, stream, _channel in ready:
                    try:
                        results[index] = self._launch(jpeg, info, stream)
                    except Exception as exc:  # noqa: BLE001 - this view's request fails with it
                        self._release_pending(self._channel_key(info["width"], info["height"],
                                                                jpeg.numel()), jpeg.numel())
                        results[index] = exc
                return results

    def close(self) -> None:
        """Release this decoder's lease, ending the session on the last release."""
        with _MEDIA_LOCK:
            self._closed = True
            if not self._leased:
                return
            with torch.npu.device(self.device):
                self._flush_deferred()
                self._close_channels()
            self._leased = False
            users = _MEDIA_USERS[self.device] - 1
            if users:
                _MEDIA_USERS[self.device] = users
                return
            del _MEDIA_USERS[self.device]
            with torch.npu.device(self.device):
                _call(self.acl.himpi.sys_exit)

    def _decode_on_host(self, jpeg):
        # JPEGD only supports a subset of JPEG profiles and dimensions.
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(jpeg.numpy().tobytes())) as image:
            rgb = torch.from_numpy(np.array(image.convert("RGB")))
        warnings.warn("CANN JPEGD does not support this JPEG; decoded on the CPU before NPU resize", stacklevel=2)
        return rgb.permute(2, 0, 1).contiguous().to(self.device)

    def _probe(self, jpeg):
        """The image info of one JPEG plus its stream descriptor; ``None`` needs the host decoder."""
        if jpeg.device.type != "cpu" or jpeg.dtype != torch.uint8 or jpeg.ndim != 1 or not jpeg.numel():
            raise ValueError("CANN JPEGD needs a nonempty one-dimensional CPU uint8 tensor")
        size = jpeg.numel()
        stream = dict(end_of_frame=1, end_of_stream=0, need_display=1, pts=0, len=size, addr=jpeg.data_ptr())
        # HOST-TAIL HT1: the probe runs under a released GIL through the built _aclfast
        # extension; a miss falls to the binding chain below, which is unchanged.
        if self._fast:
            probed = _aclfast.probe(_JPEG, jpeg.data_ptr(), size)
            if probed is not None:
                status, width, height, wstride, hstride, buf_size, _fmt = probed
                if status & 0xFFFFFFFF == _FORMAT_UNSUPPORTED:
                    return None
                _check(status, "dvpp_get_image_info")
                if not (32 <= width <= 16384 and 32 <= height <= 16384):
                    return None
                return dict(width=width, height=height, width_stride=wstride,
                            height_stride=hstride, img_buf_size=buf_size), stream
        info, status = self.acl.himpi.dvpp_get_image_info(_JPEG, stream)
        if status & 0xFFFFFFFF == _FORMAT_UNSUPPORTED:
            return None
        _check(status, "dvpp_get_image_info")
        width, height = info["width"], info["height"]
        if not (32 <= width <= 16384 and 32 <= height <= 16384):
            return None
        return info, stream

    def _begin(self, jpeg):
        """Start one JPEG's channel transaction; ``None`` means the image needs the host decoder."""
        probed = self._probe(jpeg)
        return None if probed is None else self._launch(jpeg, *probed)

    def _launch(self, jpeg, info, stream):
        """Allocate, upload and send one probed JPEG's stream on its channel.

        The fast start chain (2x dvpp_malloc + the H2D copy + vdec_send_stream) runs under one
        released GIL through the built _aclfast extension; a miss falls to the binding chain,
        which is unchanged. Phase failures free exactly what the binding chain's excepts free;
        a send failure hands both pointers back for the drop+free below.
        """
        size = jpeg.numel()
        width, height = info["width"], info["height"]
        if self._fast:
            out_size = info["img_buf_size"]
            channel = self._channel_for(width, height, size)
            started = _aclfast.start(self.device.index, channel, jpeg.data_ptr(), size,
                                     out_size, width, height, info["width_stride"],
                                     info["height_stride"], _SOURCE_FORMAT, _TIMEOUT_MS)
            if started is not None:
                status, phase, source, output = started
                if status != 0:
                    operation = ("dvpp_malloc input", "memcpy", "dvpp_malloc output", "vdec_send_stream")[phase]
                    if phase == 3:
                        self._drop_channel(self._channel_key(width, height, size))
                        _aclfast.free_batch([source, output])
                    raise RuntimeError(f"CANN JPEGD {operation} failed (0x{status & 0xFFFFFFFF:08x})")
                return dict(channel=channel, source=source, output=output, out_size=out_size,
                            width=width, height=height, size=size)
        media, runtime = self.acl.himpi, self.acl.rt
        source, status = media.dvpp_malloc(self.device.index, size)
        _check(status, "dvpp_malloc input")
        try:
            _call(runtime.memcpy, source, size, jpeg.data_ptr(), size, _HOST_TO_DEVICE)
            stream["addr"] = source
            out_size = info["img_buf_size"]
            output, status = media.dvpp_malloc(self.device.index, out_size)
            _check(status, "dvpp_malloc output")
        except Exception:
            _call(media.dvpp_free, source)
            raise
        context = dict(channel=None, source=source, output=output, out_size=out_size,
                       width=width, height=height, size=size)
        picture = dict(width=width, height=height, width_stride=info["width_stride"],
                       height_stride=info["height_stride"], pixel_format=_SOURCE_FORMAT,
                       vir_addr=output, buffer_size=out_size)
        try:
            context["channel"] = self._channel_for(width, height, size)
            _call(media.vdec_send_stream, context["channel"], stream, picture, _TIMEOUT_MS)
        except Exception:
            # a channel that failed mid-transaction is not trusted again: drop it so the next
            # decode recreates a fresh one, as the per-JPEG lifecycle always did
            if context["channel"] is not None:
                self._drop_channel(self._channel_key(width, height, size))
            _call(media.dvpp_free, source)
            _call(media.dvpp_free, output)
            raise
        return context

    def _finish(self, context, planes=False):
        """Fetch one started transaction's frame, copy it off DVPP memory and convert it to RGB.

        With ``planes`` the copy stops after the bytes are on the caller's stream and returns
        ``(buffer, meta)`` for :func:`xinfer.platforms.npu_image_graphs.chain_view` instead of the RGB
        view. The frame copy is stream-ordered when the runtime has ``memcpy_async`` -- the fast
        path's form -- and the source and output buffers then wait in ``_deferred`` for the
        copy's recorded event instead of freeing here. Without it the synchronous copy runs and
        the buffers free here, the released form.
        """
        if self._fast and self._async_copy:
            done = self._finish_fast(context, planes)
            if done is not None:
                return done
        media = self.acl.himpi
        channel = context["channel"]
        try:
            frame, _, _, status = media.vdec_get_frame(channel, _TIMEOUT_MS)
            _check(status, "vdec_get_frame")
        except Exception:
            self._drop_channel(self._channel_key(context["width"], context["height"], context["size"]))
            _call(media.dvpp_free, context["source"])
            _call(media.dvpp_free, context["output"])
            context["channel"] = None
            raise
        view = frame["v_frame"]
        try:
            if view["frame_flag"] != 0:
                raise RuntimeError(f"CANN JPEGD rejected an image (frame_flag={view['frame_flag']})")
            data = self._planes(context["output"], context["out_size"])
        except Exception:
            _call(media.vdec_release_frame, channel, frame)
            _call(media.dvpp_free, context["source"])
            _call(media.dvpp_free, context["output"])
            context["channel"] = None
            raise
        if self._async_copy:
            self._deferred.append((context["source"], context["output"]))
        try:
            _call(media.vdec_release_frame, channel, frame)
        except Exception:
            if not self._async_copy:
                self._deferred.append((context["source"], context["output"]))
            context["channel"] = None
            self._drop_channel(self._channel_key(context["width"], context["height"], context["size"]))
            raise
        if not self._async_copy:
            _call(media.dvpp_free, context["source"])
            _call(media.dvpp_free, context["output"])
        context["channel"] = None
        return self._frame_view(view, context["output"], context["out_size"],
                                context["height"], context["width"], data, planes)

    def _finish_fast(self, context, planes):
        """``_finish`` through _aclfast: get_frame + memcpy_async + release_frame
        in one GIL-releasing C call (None = fall back to the binding chain).

        The C call attempts frame release once after get_frame succeeds. A successful copy
        keeps its source buffers deferred even when release or later metadata validation fails.
        """
        key = self._channel_key(context["width"], context["height"], context["size"])
        data = torch.empty(context["out_size"], dtype=torch.uint8, device=self.device)
        fetched = _aclfast.finish(context["channel"], context["out_size"], data.data_ptr(),
                                  torch.npu.current_stream().npu_stream, _TIMEOUT_MS)
        if fetched is None:
            return None
        (status, phase, pixel_format, frame_flag, va0, va1, va2,
         ws0, ws1, ws2, hs0, hs1, hs2) = fetched
        source, output = context["source"], context["output"]
        if phase == 0:
            self._drop_channel(key)
            _aclfast.free_batch([source, output])
            context["channel"] = None
            raise RuntimeError(f"CANN JPEGD vdec_get_frame failed (0x{status & 0xFFFFFFFF:08x})")
        if phase == 1:
            _aclfast.free_batch([source, output])
            context["channel"] = None
            raise RuntimeError(f"CANN JPEGD memcpy_async failed (0x{status & 0xFFFFFFFF:08x})")
        self._deferred.append((source, output))
        context["channel"] = None
        if phase == 2:
            self._drop_channel(key)
            raise RuntimeError(f"CANN JPEGD vdec_release_frame failed (0x{status & 0xFFFFFFFF:08x})")
        if frame_flag != 0:
            raise RuntimeError(f"CANN JPEGD rejected an image (frame_flag={frame_flag})")
        frame = dict(pixel_format=pixel_format, frame_flag=frame_flag,
                     virt_addr=[va0, va1, va2], width_stride=[ws0, ws1, ws2],
                     height_stride=[hs0, hs1, hs2])
        return self._frame_view(frame, output, context["out_size"], context["height"],
                                context["width"], data, planes)

    def _abandon(self, context) -> None:
        """Free a transaction that never reached ``_finish`` (a later view of the wave failed)."""
        if context is None:
            return
        key = self._channel_key(context["width"], context["height"], context["size"])
        self._release_pending(key, context["size"])
        media = self.acl.himpi
        if context["channel"] is not None:
            self._drop_channel(self._channel_key(context["width"], context["height"], context["size"]))
            context["channel"] = None
        _call(media.dvpp_free, context["source"])
        _call(media.dvpp_free, context["output"])

    def _planes(self, output, out_size):
        """Copy one decoded frame out of DVPP memory into a tensor on the caller's stream."""
        data = torch.empty(out_size, dtype=torch.uint8, device=self.device)
        if self._async_copy:
            _call(self.acl.rt.memcpy_async, data.data_ptr(), out_size, output, out_size,
                  _DEVICE_TO_DEVICE, torch.npu.current_stream().npu_stream)
        else:
            _call(self.acl.rt.memcpy, data.data_ptr(), out_size, output, out_size, _DEVICE_TO_DEVICE)
        return data

    def _open_channel(self, attrs):
        # DVPP channel ids are dense and small (the driver's VDEC channel pool); 256 is the probe ceiling, not a
        # documented count. A taken id answers _CHANNEL_EXISTS, so the loop finds the first free one.
        for channel in range(256):
            status = self.acl.himpi.vdec_create_chn(channel, attrs)
            if status & 0xFFFFFFFF != _CHANNEL_EXISTS:
                _check(status, "vdec_create_chn")
                return channel
        raise RuntimeError("no CANN JPEGD channel is available")

    @staticmethod
    def _channel_key(width: int, height: int, size: int) -> tuple[int, int, int]:
        """The channel shape one JPEG needs: its dimensions and the stream buffer it pays for."""
        return width, height, max(size, width * height)

    def _channel_for(self, width: int, height: int, size: int) -> int:
        """The started VDEC channel of one JPEG's shape, reused across decodes.

        Creating a channel costs ~1.5 ms inside the compiled binding (probe-gil-r12 iso), paid
        by every decode when each JPEG created and destroyed its own. The cached channel holds
        exactly the attributes that per-JPEG creation built -- type, mode, dimensions, stream
        buffer size, the odd-row/column JPEGD precision mode and the started receive state --
        so each decode runs the same send/get/release transaction on the same channel state.
        """
        media = self.acl.himpi
        key = self._channel_key(width, height, size)
        channel = self._channels.get(key)
        if channel is not None:
            return channel
        while len(self._channels) >= _MAX_CHANNELS:
            idle = next((cached for cached in self._channels
                         if self._pending.get(cached, (0, 0))[0] == 0), None)
            if idle is None:
                break  # Active transactions temporarily exceed the idle-cache limit.
            self._drop_channel(idle)
        attrs = dict(type=_JPEG, mode=_SEND_FRAME, pic_width=width, pic_height=height,
                     stream_buf_size=key[2], frame_buf_size=0, frame_buf_cnt=0)
        channel = self._open_channel(attrs)
        try:
            # Keep an odd final row or column in a subsampled JPEG.
            _call(media.vdec_set_jpegd_precision_mode, channel, _ALIGN_UP)
            _call(media.vdec_start_recv_stream, channel)
        except Exception:
            _call(media.vdec_destroy_chn, channel)
            raise
        self._channels[key] = channel
        return channel

    def _drop_channel(self, key) -> None:
        """Stop and destroy one cached channel, freeing its id for a fresh one."""
        media = self.acl.himpi
        channel = self._channels.get(key)
        if channel is None:
            return
        try:
            try:
                _call(media.vdec_stop_recv_stream, channel)
            finally:
                _call(media.vdec_destroy_chn, channel)
        except Exception as exc:
            self._free_error = exc
            raise
        self._channels.pop(key)

    def _close_channels(self) -> None:
        """Tear down every cached channel before the media session ends."""
        for key in list(self._channels):
            self._drop_channel(key)

    def _copy_frame(self, frame, output, out_size, height, width):
        data = torch.empty(out_size, dtype=torch.uint8, device=self.device)
        _call(self.acl.rt.memcpy, data.data_ptr(), out_size, output, out_size, _DEVICE_TO_DEVICE)
        return self._frame_view(frame, output, out_size, height, width, data)

    def _frame_view(self, frame, output, out_size, height, width, data, planes=False):
        """Validate one frame's JPEGD output layout and return the RGB view, or its plane payload.

        With ``planes`` the return is ``(data, meta)`` -- the flat decoded planes and the geometry
        of the views inside them -- instead of the RGB view, so a caller can run the RGB + resize
        chain as one captured graph (:mod:`xinfer.platforms.npu_image_graphs`). The bytes are the same
        either way.
        """
        meta = self._plane_meta(frame, output, height, width, out_size)
        return (data, meta) if planes else rgb_of(data, meta, quantize=self.quantize)

    def _plane_meta(self, frame, output, height, width, out_size):
        """Validate one frame's JPEGD output layout and return the plane geometry inside its buffer."""
        fmt, addresses = frame["pixel_format"], frame["virt_addr"]
        height_stride, width_stride = frame["height_stride"][0], frame["width_stride"][0]
        luma_size = height_stride * width_stride
        if addresses[0] != output or height > height_stride or width > width_stride or luma_size > out_size:
            raise RuntimeError("unexpected CANN JPEGD luma buffer layout")
        meta = dict(fmt=fmt, h=height, w=width, hs=height_stride, ws=width_stride, luma_size=luma_size,
                    ch=0, cs=0, start=luma_size, end=luma_size, out_size=out_size)
        if fmt == 0:  # YUV400
            return meta
        if fmt not in _CHROMA:
            raise RuntimeError(f"unsupported CANN JPEGD pixel format {fmt}")
        step_h, step_w, _ = _CHROMA[fmt]
        chroma_height = (height + step_h - 1) // step_h
        chroma_width = 2 * ((width + step_w - 1) // step_w)
        chroma_stride = frame["width_stride"][1]
        # JPEGD uses one contiguous output allocation. On 950DT the binding only fills the first address.
        start = addresses[1] - output if addresses[1] else luma_size
        end = start + chroma_height * chroma_stride
        if start < luma_size or chroma_width > chroma_stride or end > out_size:
            raise RuntimeError("unexpected CANN JPEGD chroma buffer layout")
        meta.update(ch=chroma_height, cs=chroma_stride, start=start, end=end)
        return meta

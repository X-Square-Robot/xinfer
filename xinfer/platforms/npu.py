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
from contextlib import ExitStack
from threading import Lock

import torch
import torch.nn.functional as F

from xinfer.platforms.devices import device_module_for

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


def _rgb(luma: torch.Tensor, chroma: torch.Tensor | None, pixel_format: int) -> torch.Tensor:
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
    return rgb.add_(0.5).floor_().clamp_(0, 255).to(torch.uint8)


class NpuJpegDecoder:
    """Decode JPEGs with a lazy DVPP lease shared by decoders on the same device."""

    def __init__(self, device: torch.device):
        self.acl = load_npu_media()
        if self.acl is None:
            raise RuntimeError('DeviceImage inputs (image_preprocess_device "gpu") need CANN Python '
                               'ACL with DVPP V2 JPEGD on NPU')
        self.device = device
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
        if not self._deferred:
            return
        if self._copy_event is not None:
            if not self._copy_event.query():
                self._copy_event.synchronize()
            self._copy_event = None
        if self._fast and self._deferred:
            pointers = [pointer for pair in self._deferred for pointer in pair]
            if _aclfast.free_batch(pointers) != len(pointers):
                raise RuntimeError("CANN JPEGD dvpp_free failed")
        else:
            media = self.acl.himpi
            for source, output in self._deferred:
                _call(media.dvpp_free, source)
                _call(media.dvpp_free, output)
        self._deferred.clear()

    def _record_copy_event(self) -> None:
        """Mark the point after the frame copies on the caller's stream."""
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        self._copy_event = event

    def __call__(self, jpegs):
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
                return [self._decode(jpeg.contiguous()) for jpeg in jpegs]

    def close(self) -> None:
        """Release this decoder's lease, ending the session on the last release."""
        with _MEDIA_LOCK:
            self._closed = True
            if not self._leased:
                return
            self._leased = False
            with torch.npu.device(self.device):
                self._flush_deferred()
                self._close_channels()
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

    def _decode(self, jpeg):
        if jpeg.device.type != "cpu" or jpeg.dtype != torch.uint8 or jpeg.ndim != 1 or not jpeg.numel():
            raise ValueError("CANN JPEGD needs a nonempty one-dimensional CPU uint8 tensor")
        media, runtime = self.acl.himpi, self.acl.rt
        size = jpeg.numel()
        stream = dict(end_of_frame=1, end_of_stream=0, need_display=1, pts=0, len=size, addr=jpeg.data_ptr())
        # HOST-TAIL HT1: the probe, the start chain (2x dvpp_malloc + H2D copy +
        # vdec_send_stream) and the finish chain (get_frame + stream-ordered copy
        # + vdec_release_frame) each run under one released GIL through the
        # built _aclfast extension; every miss falls to the binding chain below,
        # which is unchanged.
        if self._fast:
            probed = _aclfast.probe(_JPEG, jpeg.data_ptr(), size)
            if probed is not None:
                status, width, height, wstride, hstride, buf_size, _fmt = probed
                info = dict(width=width, height=height, width_stride=wstride,
                            height_stride=hstride, img_buf_size=buf_size)
                if status & 0xFFFFFFFF == _FORMAT_UNSUPPORTED:
                    return self._decode_on_host(jpeg)
                _check(status, "dvpp_get_image_info")
                if not (32 <= width <= 16384 and 32 <= height <= 16384):
                    return self._decode_on_host(jpeg)
                if self._async_copy:
                    fast = self._decode_fast(jpeg, info, size, width, height)
                    if fast is not None:
                        return fast
                return self._decode_binding(jpeg, info, stream, size, width, height, media, runtime)
        info, status = media.dvpp_get_image_info(_JPEG, stream)
        if status & 0xFFFFFFFF == _FORMAT_UNSUPPORTED:
            return self._decode_on_host(jpeg)
        _check(status, "dvpp_get_image_info")
        width, height = info["width"], info["height"]
        if not (32 <= width <= 16384 and 32 <= height <= 16384):
            return self._decode_on_host(jpeg)
        return self._decode_binding(jpeg, info, stream, size, width, height, media, runtime)

    def _decode_fast(self, jpeg, info, size, width, height):
        """One JPEG's DVPP transaction through _aclfast (None = the binding chain).

        The start chain frees exactly what the binding chain's excepts free on
        each phase; the finish chain releases the frame exactly once whenever
        get_frame succeeded. The source/output buffers join ``_deferred`` and
        free once the stream-ordered frame copy completed (the next call's
        flush), matching the binding chain's post-copy frees.
        """
        out_size = info["img_buf_size"]
        channel = self._channel_for(width, height, size)
        started = _aclfast.start(self.device.index, channel, jpeg.data_ptr(), size,
                                 out_size, width, height, info["width_stride"],
                                 info["height_stride"], _SOURCE_FORMAT, _TIMEOUT_MS)
        if started is None:
            return None
        status, phase, source, output = started
        if status != 0:
            operation = ("dvpp_malloc input", "memcpy", "dvpp_malloc output", "vdec_send_stream")[phase]
            if phase == 3:
                self._drop_channel(self._channel_key(width, height, size))
                _aclfast.free_batch([source, output])
            raise RuntimeError(f"CANN JPEGD {operation} failed (0x{status & 0xFFFFFFFF:08x})")
        data = torch.empty(out_size, dtype=torch.uint8, device=self.device)
        fetched = _aclfast.finish(channel, out_size, data.data_ptr(),
                                  torch.npu.current_stream().npu_stream, _TIMEOUT_MS)
        if fetched is None:
            return None
        (status, phase, pixel_format, frame_flag, va0, va1, va2,
         ws0, ws1, ws2, hs0, hs1, hs2) = fetched
        if phase == 0:
            self._drop_channel(self._channel_key(width, height, size))
            _aclfast.free_batch([source, output])
            raise RuntimeError(f"CANN JPEGD vdec_get_frame failed (0x{status & 0xFFFFFFFF:08x})")
        if phase == 1:
            _aclfast.free_batch([source, output])
            raise RuntimeError(f"CANN JPEGD memcpy_async failed (0x{status & 0xFFFFFFFF:08x})")
        if phase == 2:
            raise RuntimeError(f"CANN JPEGD vdec_release_frame failed (0x{status & 0xFFFFFFFF:08x})")
        if frame_flag != 0:
            _aclfast.free_batch([source, output])
            raise RuntimeError(f"CANN JPEGD rejected an image (frame_flag={frame_flag})")
        frame = dict(pixel_format=pixel_format, frame_flag=frame_flag,
                     virt_addr=[va0, va1, va2], width_stride=[ws0, ws1, ws2],
                     height_stride=[hs0, hs1, hs2])
        view = self._frame_view(frame, output, out_size, height, width, data)
        # each failure phase above freed exactly what the binding chain's excepts
        # free (phase 2 leaves the channel holding the frame, like the campaign)
        self._deferred.append((source, output))
        self._record_copy_event()
        return view

    def _decode_binding(self, jpeg, info, stream, size, width, height, media, runtime):
        """The released binding transaction, unchanged."""
        out_size = info["img_buf_size"]
        with ExitStack() as stack:
            source, status = media.dvpp_malloc(self.device.index, size)
            _check(status, "dvpp_malloc input")
            stack.callback(_call, media.dvpp_free, source)
            _call(runtime.memcpy, source, size, jpeg.data_ptr(), size, _HOST_TO_DEVICE)
            stream["addr"] = source
            output, status = media.dvpp_malloc(self.device.index, out_size)
            _check(status, "dvpp_malloc output")
            stack.callback(_call, media.dvpp_free, output)

            channel = self._channel_for(width, height, size)
            picture = dict(width=width, height=height, width_stride=info["width_stride"],
                           height_stride=info["height_stride"], pixel_format=_SOURCE_FORMAT,
                           vir_addr=output, buffer_size=out_size)
            try:
                _call(media.vdec_send_stream, channel, stream, picture, _TIMEOUT_MS)
                frame, _, _, status = media.vdec_get_frame(channel, _TIMEOUT_MS)
                _check(status, "vdec_get_frame")
            except Exception:
                # a channel that failed mid-transaction is not trusted again: drop it so the
                # next decode recreates a fresh one, as the per-JPEG lifecycle always did
                self._drop_channel(self._channel_key(width, height, size))
                raise
            stack.callback(_call, media.vdec_release_frame, channel, frame)
            view = frame["v_frame"]
            if view["frame_flag"] != 0:
                raise RuntimeError(f"CANN JPEGD rejected an image (frame_flag={view['frame_flag']})")
            return self._copy_frame(view, output, out_size, height, width)

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
        if len(self._channels) >= _MAX_CHANNELS:
            self._drop_channel(next(iter(self._channels)))
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
        channel = self._channels.pop(key, None)
        if channel is None:
            return
        try:
            _call(media.vdec_stop_recv_stream, channel)
        finally:
            _call(media.vdec_destroy_chn, channel)

    def _close_channels(self) -> None:
        """Tear down every cached channel before the media session ends."""
        for key in list(self._channels):
            self._drop_channel(key)

    def _copy_frame(self, frame, output, out_size, height, width):
        data = torch.empty(out_size, dtype=torch.uint8, device=self.device)
        _call(self.acl.rt.memcpy, data.data_ptr(), out_size, output, out_size, _DEVICE_TO_DEVICE)
        return self._frame_view(frame, output, out_size, height, width, data)

    def _frame_view(self, frame, output, out_size, height, width, data):
        """Validate one frame's JPEGD output layout and return the RGB view inside ``data``."""
        fmt, addresses = frame["pixel_format"], frame["virt_addr"]
        height_stride, width_stride = frame["height_stride"][0], frame["width_stride"][0]
        luma_size = height_stride * width_stride
        if addresses[0] != output or height > height_stride or width > width_stride or luma_size > out_size:
            raise RuntimeError("unexpected CANN JPEGD luma buffer layout")
        luma = data[:luma_size].view(height_stride, width_stride)[:height, :width]
        if fmt == 0:  # YUV400
            return _rgb(luma, None, fmt)
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
        return _rgb(luma, data[start:end].view(chroma_height, chroma_stride), fmt)

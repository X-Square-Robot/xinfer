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
        name=f"npu-{arch}",
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

    def __call__(self, jpegs):
        with _MEDIA_LOCK:
            if self._closed:
                raise RuntimeError("CANN JPEGD decoder is closed")
            with torch.npu.device(self.device):
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
        info, status = media.dvpp_get_image_info(_JPEG, stream)
        if status & 0xFFFFFFFF == _FORMAT_UNSUPPORTED:
            return self._decode_on_host(jpeg)
        _check(status, "dvpp_get_image_info")
        width, height = info["width"], info["height"]
        if not (32 <= width <= 16384 and 32 <= height <= 16384):
            return self._decode_on_host(jpeg)

        with ExitStack() as stack:
            source, status = media.dvpp_malloc(self.device.index, size)
            _check(status, "dvpp_malloc input")
            stack.callback(_call, media.dvpp_free, source)
            _call(runtime.memcpy, source, size, jpeg.data_ptr(), size, _HOST_TO_DEVICE)
            stream["addr"] = source
            out_size = info["img_buf_size"]
            output, status = media.dvpp_malloc(self.device.index, out_size)
            _check(status, "dvpp_malloc output")
            stack.callback(_call, media.dvpp_free, output)

            attrs = dict(type=_JPEG, mode=_SEND_FRAME, pic_width=width, pic_height=height,
                         stream_buf_size=max(size, width * height), frame_buf_size=0, frame_buf_cnt=0)
            channel = self._open_channel(attrs)
            stack.callback(_call, media.vdec_destroy_chn, channel)
            # Keep an odd final row or column in a subsampled JPEG.
            _call(media.vdec_set_jpegd_precision_mode, channel, _ALIGN_UP)
            _call(media.vdec_start_recv_stream, channel)
            stack.callback(_call, media.vdec_stop_recv_stream, channel)
            picture = dict(width=width, height=height, width_stride=info["width_stride"],
                           height_stride=info["height_stride"], pixel_format=_SOURCE_FORMAT,
                           vir_addr=output, buffer_size=out_size)
            _call(media.vdec_send_stream, channel, stream, picture, _TIMEOUT_MS)
            frame, _, _, status = media.vdec_get_frame(channel, _TIMEOUT_MS)
            _check(status, "vdec_get_frame")
            stack.callback(_call, media.vdec_release_frame, channel, frame)
            view = frame["v_frame"]
            if view["frame_flag"] != 0:
                raise RuntimeError(f"CANN JPEGD rejected an image (frame_flag={view['frame_flag']})")
            return self._copy_frame(view, output, out_size, height, width)

    def _open_channel(self, attrs):
        for channel in range(256):
            status = self.acl.himpi.vdec_create_chn(channel, attrs)
            if status & 0xFFFFFFFF != _CHANNEL_EXISTS:
                _check(status, "vdec_create_chn")
                return channel
        raise RuntimeError("no CANN JPEGD channel is available")

    def _copy_frame(self, frame, output, out_size, height, width):
        fmt, addresses = frame["pixel_format"], frame["virt_addr"]
        height_stride, width_stride = frame["height_stride"][0], frame["width_stride"][0]
        luma_size = height_stride * width_stride
        if addresses[0] != output or height > height_stride or width > width_stride or luma_size > out_size:
            raise RuntimeError("unexpected CANN JPEGD luma buffer layout")
        data = torch.empty(out_size, dtype=torch.uint8, device=self.device)
        _call(self.acl.rt.memcpy, data.data_ptr(), out_size, output, out_size, _DEVICE_TO_DEVICE)
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

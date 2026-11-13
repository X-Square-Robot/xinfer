"""The CUDA device's platform: discovery and JPEG decoding (NVIDIA's nvImageCodec)."""

from __future__ import annotations

import functools

from xinfer.platforms.interface import Platform, PlatformCapabilities

# Compute capabilities with known deployment roles. Others are accepted as
# generic CUDA platforms; capability-specific kernels decide their own support.
_KNOWN_ARCH = {
    (12, 0): "sm120",  # RTX 5090
    (11, 0): "sm110",  # Jetson Thor
    (10, 0): "sm100",
    (9, 0): "sm90",
    (8, 0): "sm80",
}


def cuda_platform(device: str) -> Platform:
    """Describe a CUDA device from its properties.

    The architecture is ``sm<major><minor>``; graph capture and stream
    priorities follow the runtime range, unified memory follows ``is_integrated``, MPS
    availability is left undetermined, and device images need nvImageCodec (``CudaJpegDecoder``).

    Args:
        device: ``"cuda"`` (current device) or ``"cuda:N"``.

    Returns:
        A ``Platform`` named ``cuda-<arch>`` with device ``cuda:<index>``.
    """
    import torch

    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    capability = (props.major, props.minor)
    arch = _KNOWN_ARCH.get(capability, f"sm{props.major}{props.minor}")
    with torch.cuda.device(index):
        least_priority, greatest_priority = torch.cuda.Stream.priority_range()
    capabilities = PlatformCapabilities(
        device_type="cuda",
        arch=arch,
        cudagraphs=True,
        stream_priorities=greatest_priority < least_priority,
        stream_priority_range=(greatest_priority, least_priority),
        unified_memory=bool(getattr(props, "is_integrated", False)),
        # MPS availability is a deployment property (daemon, driver); v1's
        # colocated layout does not use it, so it is left undetermined.
        multiprocess_service=None,
        sm_count=props.multi_processor_count,
        total_memory_bytes=props.total_memory,
        device_images=_nvimgcodec_decoder(index) is not None,
    )
    return Platform(name=f"cuda-{arch}", device=f"cuda:{index}", capabilities=capabilities)


@functools.lru_cache(maxsize=None)
def _nvimgcodec_decoder(index: int):
    """NVIDIA's nvImageCodec decoder for CUDA device ``index`` and its sRGB decode parameters, one per device; None
    where nvImageCodec is not installed or has no decoder for the device (the capability ``device_images``).

    One CPU thread: by default nvImageCodec starts one per visible core, and they spin while they wait; in a container
    with a CPU quota that slows the frontends and the engine under load. Nothing is decoded here: a decode at platform
    resolution left every later CUDA graph replay of the process slower (on an RTX 5090, up to 1 ms per Wall-OSS
    request).
    """
    try:
        from nvidia import nvimgcodec

        return (nvimgcodec.Decoder(device_id=index, max_num_cpu_threads=1),
                nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.SRGB))
    except (ImportError, OSError, RuntimeError):  # not installed, or no decoder for this device
        return None


class CudaJpegDecoder:
    """Batch JPEG decode with NVIDIA's nvImageCodec (``nvidia-nvimgcodec-cu13``), ordered before work on the caller's
    current stream. It decodes on the GPU's JPEG engine where there is one (NVJPG: RTX 5090, Jetson Thor).

    The decoded images are nvImageCodec's memory. The decoder copies them into tensors on its own stream, keeps them
    until an event after that copy has completed, and then decodes into them again instead of allocating new ones; the
    caller never waits. Called from one thread only (the engine's device-image thread).
    """

    def __init__(self, device):
        import torch

        decoder = _nvimgcodec_decoder(device.index)
        if decoder is None:
            raise RuntimeError(f'DeviceImage inputs (image_preprocess_device "gpu") need NVIDIA\'s nvImageCodec for '
                               f"{device} (pip install xinfer[gpu-image])")
        self.device = device
        self._decoder, self._params = decoder
        self._stream = torch.cuda.Stream(self.device)
        self._copying = []  # (event after the copy, images) of earlier calls
        self._free = []  # images whose copy has completed

    def __call__(self, jpegs):
        import torch

        copying = []
        for event, images in self._copying:
            if event.query():
                self._free += images
            else:
                copying.append((event, images))
        self._copying = copying
        reuse = {}
        if len(self._free) >= len(jpegs):
            reuse["images"], self._free = self._free[:len(jpegs)], self._free[len(jpegs):]
        current = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self._stream):
            images = self._decoder.decode([jpeg.numpy() for jpeg in jpegs], params=self._params,
                                          cuda_stream=self._stream.cuda_stream, **reuse)
            if any(image is None for image in images):
                raise RuntimeError("nvImageCodec could not decode a JPEG view")
            views = [torch.as_tensor(image, device=self.device).permute(2, 0, 1).clone(
                memory_format=torch.contiguous_format) for image in images]
            copied = torch.cuda.Event()
            copied.record(self._stream)
        self._copying.append((copied, images))
        current.wait_stream(self._stream)
        for view in views:
            view.record_stream(current)
        return views

    def close(self) -> None:
        """Drop the kept images; the nvImageCodec decoder itself is shared per device."""
        self._copying, self._free = [], []

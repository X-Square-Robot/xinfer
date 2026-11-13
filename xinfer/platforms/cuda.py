"""The CUDA device's platform."""

from __future__ import annotations

import importlib.util

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
    availability is left undetermined, and device images need torchvision (nvJPEG).

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
        device_images=importlib.util.find_spec("torchvision") is not None,
    )
    return Platform(name=f"cuda-{arch}", device=f"cuda:{index}", capabilities=capabilities)


class CudaJpegDecoder:
    """Batch JPEG decode with nvJPEG, ordered before work on the caller's current stream.

    torchvision shares one decoder per device and rebuilds it when the device object changes, so every
    call uses the same canonical ``torch.device`` supplied by ``jpeg_decoder_for``.
    """

    def __init__(self, device):
        import torch

        try:
            from torchvision.io import ImageReadMode, decode_jpeg
        except ImportError as exc:
            raise ImportError('DeviceImage inputs (image_preprocess_device "gpu") need torchvision '
                              '(pip install xinfer[gpu-image])') from exc
        self.device = device
        self._decode = lambda jpegs: decode_jpeg(jpegs, mode=ImageReadMode.RGB, device=self.device)
        # nvJPEG writes on an internal stream that does not wait for the caller. Allocate its outputs on
        # this separate stream so they cannot reuse storage the caller's queued work is still reading.
        self._stream = torch.cuda.Stream(self.device)

    def __call__(self, jpegs):
        import torch

        current = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self._stream):
            views = self._decode(jpegs)
        current.wait_stream(self._stream)
        for view in views:
            view.record_stream(current)
        return views

    def close(self) -> None:
        """No per-instance codec release: torchvision owns the shared nvJPEG decoder."""

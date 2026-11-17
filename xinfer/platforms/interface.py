"""Platform capabilities.

A platform describes what a device can do. The runtime derives defaults from
these capabilities and rejects explicit settings a platform cannot honor; it
never silently downgrades (docs/design.md §13).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformCapabilities:
    """Capabilities that change how the runtime may execute work.

    Attributes:
        device_type: ``"cuda"``, ``"ascend"`` or ``"cpu"``: which hardware and backend serves a request, so it
            names the platform layer and is also a backend selection key. The torch device strings stay
            ``"npu:N"``, and ``Platform.is_npu`` asks the device-layer question about them; docs/design.md §3.1
            gives the rule.
        arch: architecture, e.g. ``"sm120"`` (RTX 5090) or ``"sm110"`` (Jetson Thor); ``None`` on CPU.
            Selects tuned tables.
        cudagraphs: stream capture and replay are available; gates the ``cudagraphs`` setting.
        stream_priorities: streams can have priorities.
        unified_memory: host and device share physical memory (Jetson).
        multiprocess_service: whether MPS is available; ``None`` when not determined.
        sm_count: compute units (CUDA SMs or Ascend cube cores); ``None`` on CPU.
        total_memory_bytes: device memory; ``None`` on CPU.
        device_images: camera views can be decoded and resized on the device (it has a JPEG decoder: on CUDA,
            NVIDIA's nvImageCodec); gates ``image_preprocess_device="gpu"``.
        stream_priority_range: inclusive numeric (minimum, maximum) supported by the active runtime,
            or None when the range is unknown. Lower values have higher scheduling priority.
    """

    device_type: str  # "cuda", "ascend" or "cpu"
    arch: str | None  # e.g. "sm120", "sm110"; None on CPU
    cudagraphs: bool  # stream capture and replay
    stream_priorities: bool
    unified_memory: bool  # host and device share physical memory (Jetson)
    multiprocess_service: bool | None  # MPS; None when not determined
    sm_count: int | None
    total_memory_bytes: int | None
    device_images: bool = False
    stream_priority_range: tuple[int, int] | None = None


@dataclass(frozen=True)
class Platform:
    """A resolved execution platform.

    Attributes:
        name: platform name, e.g. ``"cuda-sm120"`` or ``"cpu"``: the hardware and
            backend identity, which docs/design.md §3.1 separates from the torch device.
        device: torch device string, e.g. ``"cuda:0"``, ``"npu:0"`` or ``"cpu"``.
        capabilities: the device's capabilities.
    """

    name: str  # "cuda-sm120", "ascend-<arch>", "cpu"
    device: str  # torch device string, e.g. "cuda:0", "npu:0" or "cpu"
    capabilities: PlatformCapabilities

    @property
    def is_cuda(self) -> bool:
        """Whether the platform is a CUDA device."""
        return self.capabilities.device_type == "cuda"

    @property
    def is_npu(self) -> bool:
        """Whether the platform is an Ascend NPU."""
        return self.capabilities.device_type == "ascend"

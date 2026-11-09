"""Resolve the platform for a torch device."""

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


def _cuda_platform(device: str) -> Platform:
    """Describe a CUDA device from its properties.

    The architecture is ``sm<major><minor>``; graph capture and stream
    priorities are assumed, unified memory follows ``is_integrated``, MPS
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
    capabilities = PlatformCapabilities(
        device_type="cuda",
        arch=arch,
        cudagraphs=True,
        stream_priorities=True,
        unified_memory=bool(getattr(props, "is_integrated", False)),
        # MPS availability is a deployment property (daemon, driver); v1's
        # colocated layout does not use it, so it is left undetermined.
        multiprocess_service=None,
        sm_count=props.multi_processor_count,
        total_memory_bytes=props.total_memory,
        device_images=importlib.util.find_spec("torchvision") is not None,
    )
    return Platform(name=f"cuda-{arch}", device=f"cuda:{index}", capabilities=capabilities)


def _cpu_platform() -> Platform:
    """The CPU platform: no graph capture, no stream priorities, unified memory."""
    capabilities = PlatformCapabilities(
        device_type="cpu",
        arch=None,
        cudagraphs=False,
        stream_priorities=False,
        unified_memory=True,
        multiprocess_service=False,
        sm_count=None,
        total_memory_bytes=None,
    )
    return Platform(name="cpu", device="cpu", capabilities=capabilities)


def resolve_platform(device: str | None = None) -> Platform:
    """Return the platform for ``device`` ("cuda", "cuda:N", "cpu" or None).

    ``None`` selects the current CUDA device when one is available, else CPU.

    Args:
        device: a torch device string, or ``None``.

    Returns:
        The ``Platform`` with its capabilities.

    Raises:
        RuntimeError: a CUDA device was requested but CUDA is not available.
        ValueError: the device type is neither ``cuda`` nor ``cpu``.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    kind = torch.device(device).type
    if kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {device!r} requested but CUDA is not available")
        return _cuda_platform(device)
    if kind == "cpu":
        return _cpu_platform()
    raise ValueError(f"unsupported device type {kind!r}")

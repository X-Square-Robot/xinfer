"""Resolve the platform for a torch device.

One module per device type holds its description — ``cuda``, ``npu`` and ``cpu``, alongside this package, the
way ``xinfer/ops/backends/`` holds one package per hardware — and this module only parses the device string and
dispatches to it, importing the module lazily so a process serving one device type never loads another's
runtime.
"""

from __future__ import annotations

from xinfer.platforms.interface import Platform


def resolve_platform(device: str | None = None) -> Platform:
    """Return the platform for ``device`` ("cuda", "cuda:N", "npu", "npu:N", "cpu" or None).

    ``None`` selects the current CUDA device when one is available, else CPU.

    Args:
        device: a torch device string, or ``None``.

    Returns:
        The ``Platform`` with its capabilities.

    Raises:
        RuntimeError: the requested accelerator is unavailable.
        ValueError: the device type is unsupported or its index is out of range.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).partition(":")[0] == "npu":
        from xinfer.platforms.npu import npu_platform

        return npu_platform(device)
    kind = torch.device(device).type
    if kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {device!r} requested but CUDA is not available")
        from xinfer.platforms.cuda import cuda_platform

        return cuda_platform(device)
    if kind == "cpu":
        from xinfer.platforms.cpu import cpu_platform

        return cpu_platform()
    raise ValueError(f"unsupported device type {kind!r}")

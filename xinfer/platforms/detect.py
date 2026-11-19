"""Resolve the platform for a device string.

One module or package per device kind holds its description — ``cuda``, ``npu``, ``rpu`` and ``cpu`` —
the way ``xinfer/ops/backends/`` holds one package per hardware. This module only parses the device
string and dispatches to it, importing the implementation lazily so a process serving one device
kind never loads another's runtime (including its vendor SDK).

To add another accelerator family:
  1. Create ``platforms/<kind>.py`` or ``platforms/<kind>/`` exposing
     ``<kind>_platform(device) -> Platform`` from its public module or package.
  2. Add one branch below dispatching by the device-string prefix to that factory.

Keep dispatch here; do not leak vendor-specific literals into callers.
"""

from __future__ import annotations

from xinfer.platforms.interface import Platform


def _device_kind(device) -> str:
    return str(device).partition(":")[0]


def resolve_platform(device: str | None = None) -> Platform:
    """Return the platform for ``device`` (CUDA, CPU, ``rpu[:0]``, ``npu[:N]``) or None.

    ``None`` selects the current CUDA device when one is available, else CPU.

    Args:
        device: a device string, or ``None``.

    Returns:
        The ``Platform`` with its capabilities.

    Raises:
        RuntimeError: a CUDA device was requested but CUDA is not available,
            or a vendor accelerator was requested without its runtime present.
        ValueError: unsupported device type or invalid device index.
    """
    if device is not None:
        kind = _device_kind(device)
        if kind == "rpu":
            from xinfer.platforms.rpu import rpu_platform

            return rpu_platform(device)
        if kind == "npu":
            from xinfer.platforms.npu import npu_platform

            return npu_platform(device)

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_kind = torch.device(device).type
    if torch_kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {device!r} requested but CUDA is not available")
        from xinfer.platforms.cuda import cuda_platform

        return cuda_platform(device)
    if torch_kind == "cpu":
        from xinfer.platforms.cpu import cpu_platform

        return cpu_platform()
    raise ValueError(f"unsupported device type {torch_kind!r}")

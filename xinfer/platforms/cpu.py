"""The CPU device's platform: the trivial fallback, with nothing to detect."""

from __future__ import annotations


def cpu_platform():
    """The CPU platform: no graph capture, no stream priorities, unified memory."""
    from xinfer.platforms.interface import Platform, PlatformCapabilities

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

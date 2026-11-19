"""Board-free RPU platform capabilities and device resolution."""

from __future__ import annotations


def rpu_platform(device: str):
    """Describe the fixed single-device TP8 execution contract, without loading SDKs.

    Resolution is also usable for board-free admission. Native availability and
    candidate ABI admission are checked when the execution context is created.
    An architecture match alone is not an operator support claim.

    RPU operators use the shared OpTable with prepared-operation signatures.
    Execution waits on explicit submission completion rather than torch streams
    and events, as declared through the platform capabilities.
    """
    from xinfer.platforms.interface import Platform, PlatformCapabilities

    if device not in {"rpu", "rpu:0"}:
        raise ValueError("the first RPU profile supports only device rpu:0")
    return Platform(
        name="rpu-tp8", device="rpu:0",
        capabilities=PlatformCapabilities(
            device_type="rpu", arch="rpu-tp8", cudagraphs=False,
            stream_priorities=False, unified_memory=False,
            multiprocess_service=False, sm_count=None, total_memory_bytes=None,
            execution_model="explicit-completion",
        ),
    )

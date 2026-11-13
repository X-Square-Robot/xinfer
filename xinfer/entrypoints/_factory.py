"""Single dispatch site for engine selection by platform capability.

This is the only place engine classes are picked from a resolved ``Platform``: vendor identity
never leaks into branches here, only ``capabilities.execution_model`` (docs/design.md §13.1).
The CUDA-lanes engine is the default; vendor SDKs with their own submission model register a
different value on their platform's capabilities and the matching engine class in a one-line
branch below. Keep dependency direction intact: this module lives in ``entrypoints/`` and is
allowed to import from ``engine/`` (``platforms/interface.py`` stays torch- and engine-free).
"""

from __future__ import annotations

from xinfer.platforms.interface import Platform


def create_engine(platform: Platform, model, config=None):
    """Construct the engine ``platform.capabilities.execution_model`` selects.

    Args:
        platform: the resolved device platform.
        model: the loaded model implementing the model interface.
        config: the resolved engine configuration (defaults by engine).

    Returns:
        An ``Engine``-shaped object with ``submit(prepared) -> Future`` and ``shutdown()``.

    Raises:
        ValueError: an unsupported ``execution_model`` value.
    """
    if config is not None and hasattr(config, "engine_config"):
        raise TypeError("create_engine takes an EngineConfig, not a ResolvedConfig; call engine_config() first")
    if platform.capabilities.execution_model == "explicit-completion":
        from xinfer.engine.rpu import RpuEngine

        return RpuEngine(model, platform, config)
    if platform.capabilities.execution_model == "cuda-lanes":
        from xinfer.engine.core import Engine

        return Engine(model, platform, config)
    raise ValueError(f"unknown execution_model {platform.capabilities.execution_model!r} on {platform.name}")

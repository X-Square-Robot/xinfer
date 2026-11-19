"""Backend assembly for the in-process Policy.

Policy selects its configuration and engine by execution capability. This entry
point may assemble ops, platform and engine objects without reversing package
dependencies. Serving and load profiling explicitly require the lane engine.
"""

from __future__ import annotations

from xinfer.platforms.interface import Platform


def resolve_config(model, platform: Platform, **kwargs):
    """Resolve settings with the configuration adapter for the execution model.

    ``kwargs`` are the generic resolver's model name, quantization, explicit
    settings, CPU quota and tuned-table options.
    """
    if platform.capabilities.execution_model == "explicit-completion":
        from xinfer.config.rpu import resolve

        return resolve(model, platform, **kwargs)
    if platform.capabilities.execution_model == "cuda-lanes":
        from xinfer.config import resolve

        return resolve(model, platform, **kwargs)
    raise ValueError(f"unknown execution_model {platform.capabilities.execution_model!r} on {platform.name}")


def _rpu_context(op_table, **options):
    """Assemble the in-tree operator provider and the platform resource owner."""
    from xinfer.ops.backends import rpu
    from xinfer.platforms.rpu import RpuExecutionContext

    operation_factory = rpu.bind_ops(op_table)
    backend = rpu.load_backend()
    if not backend.provenance["runtime_available"]:
        raise RuntimeError("the RPU library is host-only; build the runtime against Rhino Launch SDK for execution")
    context = RpuExecutionContext(backend_factory=backend.backend_factory,
                                 operation_factory=operation_factory, **options)
    context.provenance.update(backend.provenance)
    assets = getattr(context.native, "asset_provenance", None)
    if assets is not None:
        context.provenance["assets"] = dict(assets)
    return context


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
    if platform.capabilities.execution_model == "explicit-completion":
        from xinfer.engine.rpu import RpuEngine

        return RpuEngine(model, platform, config,
                         context_factory=lambda **options: _rpu_context(model.ops, **options))
    if platform.capabilities.execution_model == "cuda-lanes":
        from xinfer.engine.core import Engine

        return Engine(model, platform, config)
    raise ValueError(f"unknown execution_model {platform.capabilities.execution_model!r} on {platform.name}")

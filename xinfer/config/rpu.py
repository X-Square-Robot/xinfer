"""Single-request runtime settings for RPU prepared execution.

The generic resolver remains responsible for model batch sizes, platform graph
and stream capabilities, and frontend settings. This adapter supplies only the
RPU engine's one-request and one-lane defaults and validates those limits.
"""

from __future__ import annotations

from dataclasses import replace

from xinfer.config.resolve import ConfigError, ExplicitSettings, ResolvedConfig, Setting, resolve as resolve_generic
from xinfer.config.tuned import TunedTable, find_tuned


def _require_one(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ConfigError(f"RPU execution requires {name}=1, got {value!r}")


def resolve(model, platform, *, model_name: str, quantization: str | None = None,
            explicit: ExplicitSettings | None = None, cpus: int | None = None,
            tuned: TunedTable | None | bool = True) -> ResolvedConfig:
    """Apply RPU engine limits while preserving generic setting precedence.

    Operator settings and selected tuned lanes must satisfy the single-request contract.
    Only missing limits receive backend defaults; their reported source remains
    the platform rather than an explicit operator setting. Frontend and shared
    memory settings follow the generic resolver because RPU serving is rejected
    at the serving entry point.
    """
    if platform.capabilities.execution_model != "explicit-completion":
        raise ConfigError("RPU configuration requires execution_model='explicit-completion', "
                          f"got {platform.capabilities.execution_model!r}")
    explicit = explicit or ExplicitSettings()
    if tuned is True:
        tuned = find_tuned(model_name, platform.capabilities.arch, quantization)
    tuned = tuned or None
    if tuned is not None:
        for stage in model.stage_graph:
            if stage.name in tuned.lanes and stage.name not in explicit.lanes:
                _require_one(f"tuned lanes.{stage.name}", tuned.lanes[stage.name])

    defaults = {}
    for name in ("max_running_requests", "max_queued_requests", "per_robot_inflight"):
        value = getattr(explicit, name)
        if value is None:
            defaults[name] = 1
        else:
            _require_one(name, value)
    for stage, value in explicit.lanes.items():
        _require_one(f"lanes.{stage}", value)

    lanes = {stage.name: 1 for stage in model.stage_graph
             if stage.name not in explicit.lanes and (tuned is None or stage.name not in tuned.lanes)}
    resolved = resolve_generic(model, platform, model_name=model_name, quantization=quantization,
                               explicit=replace(explicit, **defaults, lanes={**lanes, **explicit.lanes}),
                               cpus=cpus, tuned=tuned)
    source = f"platform:single-submission {platform.name}"
    for name in (*defaults, *(f"lanes.{stage}" for stage in lanes)):
        resolved.settings[name] = Setting(1, source)
    return resolved

"""Configuration resolution (docs/configuration.md): explicit > tuned table > default.

The settings are upper bounds and fixed choices that suit any load: the engine
adapts to the requests that arrive (batches form only from requests already
queued, an unused lane stays idle, graphs are captured for the keys that
occur), so nothing is sized from the number of robots. Each setting is
resolved once at startup and recorded with its source:

| Source | Meaning |
|---|---|
| ``explicit`` | set by the operator; checked against the platform and model, never rewritten |
| ``tuned:<table>@<commit>`` | measured best value for (model, architecture, quantization) |
| ``model:<rule>`` | follows from the model's declarations (batch sizes) |
| ``platform:<rule>`` | follows from the platform (graph capture, CPU quota) |
| ``default`` | a fixed default |

Runtime behaviors (chained dispatch, zero-wait batching) are not settings.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from xinfer.config.tuned import TunedTable, find_tuned
from xinfer.engine.core import EngineConfig
from xinfer.engine.cudagraph import CUDAGraphConfig

DEFAULT_LANES = 2  # per stage, without a tuned value
DEFAULT_MAX_RUNNING_REQUESTS = 32
DEFAULT_MAX_QUEUED_REQUESTS = 256
# Each frontend adds a bridge thread in the engine process; at 4 robots, 4 frontends were slower than 1 or 2
# (docs/configuration.md).
DEFAULT_FRONTENDS = 2
# Processing threads per frontend: decode and resize release the GIL, so one frontend processes several
# connections' requests at once (docs/configuration.md).
DEFAULT_FRONTEND_WORKERS = 4
DEFAULT_FINAL_STAGE_PRIORITY = -1  # one level above ordinary streams, bounded by the runtime's reported range


@dataclass(frozen=True)
class Setting:
    """One resolved setting and why it has its value.

    Attributes:
        value: the resolved value.
        source: where it came from: ``explicit``, ``tuned:<table>@<commit>``, ``model:<rule>``,
            ``platform:<rule>`` or ``default`` (see the module docstring).
    """

    value: Any
    source: str


@dataclass(frozen=True)
class ExplicitSettings:
    """Operator-set values; ``None`` (or a missing stage) means "resolve".

    Explicit values are validated against the platform and the model and
    never rewritten; a conflict raises ``ConfigError``.

    Attributes:
        lanes: lanes (stream + graph pool) per stage name.
        max_batch: batch limit per stage name; it must not exceed the model's largest declared batch size.
        max_running_requests: admitted requests in flight in the engine.
        max_queued_requests: submitted requests waiting for admission.
        per_robot_inflight: unfinished requests per robot.
        cudagraphs: ``True`` requires a platform with graph capture; ``False`` runs every stage eagerly.
        frontends: frontend processes.
        frontend_workers: processing threads per frontend.
        shm_buffers: shared-memory buffers per frontend.
        stream_priorities: explicit accelerator stream priorities per stage, within the platform's range.
    """

    lanes: Mapping[str, int] = field(default_factory=dict)
    max_batch: Mapping[str, int] = field(default_factory=dict)
    max_running_requests: int | None = None
    max_queued_requests: int | None = None
    per_robot_inflight: int | None = None
    cudagraphs: bool | None = None
    frontends: int | None = None
    frontend_workers: int | None = None
    shm_buffers: int | None = None
    stream_priorities: Mapping[str, int] = field(default_factory=dict)


class ConfigError(ValueError):
    """An explicit value conflicts with the platform or the model."""


@dataclass
class ResolvedConfig:
    """The resolved settings of one engine process, in resolution order.

    Attributes:
        settings: setting name to ``Setting``. Names: ``per_robot_inflight``, ``max_running_requests``,
            ``max_queued_requests``, ``lanes.<stage>``, ``batch_sizes.<stage>``
            (list of allowed batch sizes), ``stream_priorities.<stage>``, ``cudagraphs``, ``frontends``,
            ``frontend_workers``, ``shm_buffers`` and declared ``shape_buckets.<dimension>`` policies.
        stages: the model's stage names in execution order.
        tuned: the tuned table consulted, or ``None``.
    """

    settings: dict[str, Setting]
    stages: tuple[str, ...]
    tuned: TunedTable | None = None

    def value(self, name: str) -> Any:
        """Return the value of setting ``name`` (``KeyError`` if there is none)."""
        return self.settings[name].value

    def engine_config(self, cudagraph_config: CUDAGraphConfig | None = None) -> EngineConfig:
        """Build the ``EngineConfig`` for these settings.

        Args:
            cudagraph_config: graph policy to use when ``cudagraphs`` resolved to True; defaults to
                ``CUDAGraphConfig()``. Ignored when graphs are off.

        Returns:
            An ``EngineConfig`` with lanes per stage, ``max_batch`` per stage set to the largest resolved
            batch size, the request limits and the graph policy (``None`` for eager execution).
        """
        value = self.value
        return EngineConfig(
            lanes={s: value(f"lanes.{s}") for s in self.stages},
            stream_priorities={s: value(f"stream_priorities.{s}") for s in self.stages},
            max_batch={s: max(value(f"batch_sizes.{s}")) for s in self.stages},
            max_running_requests=value("max_running_requests"),
            max_queued_requests=value("max_queued_requests"),
            per_robot_inflight=value("per_robot_inflight"),
            cudagraphs=(cudagraph_config or CUDAGraphConfig()) if value("cudagraphs") else None,
        )

    def rows(self) -> list[dict]:
        """Return the settings as ``[{"name", "value", "source"}, ...]`` in resolution order."""
        return [dict(name=k, value=s.value, source=s.source) for k, s in self.settings.items()]

    def report(self) -> str:
        """Format the settings as an aligned text table for the startup log.

        Returns:
            One line per setting: name, value (lists and dicts as JSON) and source.
        """
        width = max(len(k) for k in self.settings)
        lines = []
        for k, s in self.settings.items():
            value = json.dumps(s.value) if isinstance(s.value, (list, dict)) else str(s.value)
            lines.append(f"{k:<{width}}  {value:<10} {s.source}")
        return "\n".join(lines)

    def write(self, path: str | Path, **extra) -> None:
        """Write the settings and their sources as JSON (``--resolved-config``).

        Args:
            path: output file, overwritten.
            **extra: further top-level keys (``xinfer serve`` adds the model, platform, operator selection
                and tuned-table provenance); values JSON cannot encode are written as ``str``.
        """
        Path(path).write_text(json.dumps(dict(settings=self.rows(), **extra), indent=1, default=str))


def cpu_quota() -> int:
    """CPUs this process may use: the cgroup quota when set, else the affinity mask.

    Reads the cgroup v2 ``/sys/fs/cgroup/cpu.max``; without a quota (or the
    file) it falls back to the affinity mask, then ``os.cpu_count()``.

    Returns:
        ``floor(quota / period)`` (at least 1), or the number of CPUs in the affinity mask.
    """
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _positive(name: str, value: int) -> int:
    """Return ``value`` if it is at least 1.

    Raises:
        ConfigError: ``value`` < 1; the message names the setting ``name``.
    """
    if value < 1:
        raise ConfigError(f"{name} must be at least 1, got {value}")
    return value


def apply_image_preprocess_device(model_options: Mapping[str, Any], platform,
                                  default: str | None = None) -> dict[str, Any]:
    """Resolve the ``image_preprocess_device`` model option and hand it to the model's processors.

    The option, where the camera views are decoded and resized, sits at the top level of the model options of every
    entry point (``Policy(..., image_preprocess_device=...)``, ``xinfer serve --model-options
    '{"image_preprocess_device": ...}'``), not in ``processing``. The processors read it as
    ``processing.image_preprocess_device``, set here when it resolves to ``"gpu"`` (``"cpu"`` is their default).

    Args:
        model_options: the entry point's model options; ``processing``, if given, is a dict.
        platform: the resolved ``Platform``.
        default: the entry point's default when the option is unset (``xinfer serve``: ``"cpu"``), or ``None`` for
            the platform's (``Policy``): ``"gpu"`` where the views can be decoded and resized on the device
            (``device_images``), else ``"cpu"``.

    Returns:
        A copy of ``model_options`` without ``image_preprocess_device``, for ``from_checkpoint`` and
        ``build_processors``.

    Raises:
        ConfigError: ``processing`` sets it; a value other than ``"cpu"`` and ``"gpu"``; ``"gpu"`` on a platform
            without device images.
    """
    options = dict(model_options)
    device = options.pop("image_preprocess_device", None)
    processing = options.get("processing") or {}
    if "image_preprocess_device" in processing:
        raise ConfigError("image_preprocess_device is a top-level model option, not a processing field")
    if device is None:
        device = default
    if device is None:
        device = "gpu" if platform.capabilities.device_images else "cpu"
    elif device not in ("cpu", "gpu"):
        raise ConfigError(f'image_preprocess_device must be "cpu" or "gpu", got {device!r}')
    elif device == "gpu" and not platform.capabilities.device_images:
        if platform.device.partition(":")[0] == "npu":
            raise ConfigError(f"platform {platform.name} cannot decode images on the device (needs CANN DVPP V2)")
        raise ConfigError(f"platform {platform.name} cannot decode images on the device (needs CUDA and torchvision)")
    if device == "gpu":
        options["processing"] = {**processing, "image_preprocess_device": "gpu"}
    return options


def resolve(model, platform, *, model_name: str, quantization: str | None = None,
            explicit: ExplicitSettings | None = None, cpus: int | None = None,
            tuned: TunedTable | None | bool = True) -> ResolvedConfig:
    """Resolve the runtime settings of one engine process for ``model`` on ``platform``.

    ``tuned=True`` looks up the bundled table for (model_name, architecture,
    quantization); pass a ``TunedTable`` to use another, or ``None``/``False``
    for none. Precedence is explicit > tuned table > default. Each setting:

    - ``per_robot_inflight``: explicit, else 1.
    - ``max_running_requests``: explicit, else 32; ``max_queued_requests``: explicit, else 256.
    - ``lanes.<stage>``: explicit, else the tuned value, else 2.
    - ``stream_priorities.<stage>``: explicit, else -1 bounded by the platform's reported range for a final
      stage (one without consumers), else 0; unknown ranges keep default 0, and nonzero requires platform support.
    - ``batch_sizes.<stage>`` (allowed batch sizes, 1 always included): with an explicit ``max_batch``, the
      declared sizes up to it; else every declared size.
    - ``cudagraphs``: explicit, else whether the platform supports graph capture.
    - ``frontends``: explicit, else 2, at most half the CPU quota.
    - ``frontend_workers``: explicit, else 4, with ``frontends × frontend_workers`` at most half the CPU quota.
    - ``shm_buffers``: explicit, else ``ceil(max_running_requests / frontends)``, one per running request (a
      frontend whose buffers are all leased waits).

    Args:
        model: the loaded model; its ``stage_graph`` gives the stages and its ``capabilities()`` the
            batch sizes.
        platform: the resolved platform; ``capabilities.arch`` selects the tuned table and ``capabilities.cudagraphs``
            gates CUDA graphs.
        model_name: registry name of the model, which names the tuned table directory.
        quantization: quantization name for the tuned table (e.g. ``"bf16"``, ``"mxfp8"``); ``None``
            skips the bundled lookup.
        explicit: operator-set values; ``None`` sets nothing.
        cpus: CPUs available to the instance; ``None`` uses ``cpu_quota()``.
        tuned: ``True`` for the bundled table, a ``TunedTable``, or ``None``/``False`` for none.

    Returns:
        A ``ResolvedConfig`` with ``value(name)``, ``settings`` (each with its source) and ``engine_config()``.

    Raises:
        ConfigError: an explicit value is below 1, names an unknown stage, asks for a batch size above the
            model's largest, or enables CUDA graphs on a platform without capture.
        ValueError: the bundled tuned table is malformed.
    """
    explicit = explicit or ExplicitSettings()
    stages = tuple(s.name for s in model.stage_graph)
    capabilities = model.capabilities()
    out: dict[str, Setting] = {}
    for name, spec in model.shape_buckets().items():
        out[f"shape_buckets.{name}"] = Setting(asdict(spec), "model:bucket policy")
    if tuned is True:
        tuned = find_tuned(model_name, platform.capabilities.arch, quantization)
    tuned = tuned or None

    def chosen(name, value, default, source="default"):
        if value is not None:
            return Setting(_positive(name, value), "explicit")
        return Setting(default, source)

    out["per_robot_inflight"] = chosen("per_robot_inflight", explicit.per_robot_inflight, 1)
    out["max_running_requests"] = chosen("max_running_requests", explicit.max_running_requests,
                                         DEFAULT_MAX_RUNNING_REQUESTS)
    out["max_queued_requests"] = chosen("max_queued_requests", explicit.max_queued_requests,
                                        DEFAULT_MAX_QUEUED_REQUESTS)

    unknown = (set(explicit.lanes) | set(explicit.max_batch) | set(explicit.stream_priorities)) - set(stages)
    if unknown:
        raise ConfigError(f"unknown stage(s) {sorted(unknown)}; the model has {list(stages)}")
    for stage in stages:
        if stage in explicit.lanes:
            out[f"lanes.{stage}"] = Setting(_positive(f"lanes.{stage}", explicit.lanes[stage]), "explicit")
        elif tuned is not None and stage in tuned.lanes:
            out[f"lanes.{stage}"] = Setting(tuned.lanes[stage], tuned.source)
        else:
            out[f"lanes.{stage}"] = Setting(DEFAULT_LANES, "default")

    priorities = platform.capabilities.stream_priorities
    bounds = platform.capabilities.stream_priority_range
    if bounds is not None and bounds[0] > bounds[1]:
        # The clamp below only ever yields a value inside a well-formed range, so this guards a platform that
        # reports a reversed one instead of silently picking an out-of-range default.
        raise ConfigError(f"platform {platform.name} declares a reversed stream priority range {bounds}")
    for stage in stages:
        if stage not in explicit.stream_priorities:
            final = priorities and not model.stage_graph.consumers(stage)
            default_priority = (max(bounds[0], min(DEFAULT_FINAL_STAGE_PRIORITY, bounds[1]))
                                if final and bounds is not None else 0)
            out[f"stream_priorities.{stage}"] = Setting(
                default_priority, "default:final stage" if final and default_priority != 0 else "default")
            continue
        priority = explicit.stream_priorities[stage]
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ConfigError(f"stream_priorities.{stage} must be an integer, got {priority!r}")
        if priority != 0 and not priorities:
            raise ConfigError(f"platform {platform.name} does not support nonzero stream priorities")
        if bounds is not None and not bounds[0] <= priority <= bounds[1]:
            raise ConfigError(f"stream_priorities.{stage}={priority} is outside {platform.name}'s range {bounds}")
        out[f"stream_priorities.{stage}"] = Setting(priority, "explicit")

    for stage in stages:
        allowed = sorted(set(capabilities.batch_sizes.get(stage, (1,))) | {1})
        if stage in explicit.max_batch:
            limit = _positive(f"max_batch.{stage}", explicit.max_batch[stage])
            if limit > allowed[-1]:
                raise ConfigError(f"stage {stage} supports batch sizes {allowed}, not {limit}")
            out[f"batch_sizes.{stage}"] = Setting([b for b in allowed if b <= limit], "explicit")
        elif allowed == [1]:
            out[f"batch_sizes.{stage}"] = Setting([1], "model:declares B1 only")
        else:
            out[f"batch_sizes.{stage}"] = Setting(allowed, "model:declared sizes")

    if explicit.cudagraphs is not None:
        if explicit.cudagraphs and not platform.capabilities.cudagraphs:
            raise ConfigError(f"platform {platform.name} does not support CUDA graph capture")
        out["cudagraphs"] = Setting(bool(explicit.cudagraphs), "explicit")
    else:
        supported = bool(platform.capabilities.cudagraphs)
        out["cudagraphs"] = Setting(supported, f"platform:{'supports' if supported else 'lacks'} graph capture")

    cpus = cpus or cpu_quota()
    max_frontends = max(1, cpus // 2)
    out["frontends"] = chosen("frontends", explicit.frontends, min(DEFAULT_FRONTENDS, max_frontends),
                              f"default, ≤ cpus/2 = {max_frontends}")
    max_workers = max(1, max_frontends // out["frontends"].value)
    out["frontend_workers"] = chosen("frontend_workers", explicit.frontend_workers,
                                     min(DEFAULT_FRONTEND_WORKERS, max_workers),
                                     f"default, frontends × workers ≤ cpus/2 = {max_frontends}")
    per_frontend = math.ceil(out["max_running_requests"].value / out["frontends"].value)
    out["shm_buffers"] = chosen("shm_buffers", explicit.shm_buffers, per_frontend,
                                "default:one per running request")
    return ResolvedConfig(out, stages, tuned)

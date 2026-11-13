"""Execution primitives shared by the engine: lanes, uploads and stage outputs.

A lane is one of a stage's parallel executors: a CUDA stream (none on the CPU)
plus, with CUDA graphs enabled, the lane's graph pool and static buffers
(``xinfer.engine.cudagraph``). Stage outputs are buffers the runtime allocates
from the model's ``stage_outputs`` declaration and passes in
``StageBatch.out``. The engine core (``xinfer.engine.core``) schedules
dispatches onto lanes.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch

from xinfer.models.interface import StageBatch, VLAModel
from xinfer.platforms.devices import device_module_for
from xinfer.platforms.interface import Platform


class Lane:
    """One of a stage's parallel executors: an accelerator stream and its identity.

    A lane runs one dispatch at a time; the engine treats it as busy until ``event`` completes.

    Attributes:
        stage: name of the stage the lane serves.
        index: the lane's index within the stage.
        stream: the lane's stream on the platform device, or None on the CPU.
        cudagraphs: the lane's ``LaneCUDAGraphs`` when the engine uses graph capture, else None.
        copy_stream: for a final stage on an accelerator, the stream that copies results to the host (set by the engine,
            never captured); else None.
        event: completion event of the lane's current dispatch, or None (set and cleared by the engine).
    """

    def __init__(self, stage: str, index: int, platform: Platform, *, priority: int = 0):
        """Create the lane; on an accelerator its stream first waits for work on the current stream.

        Args:
            stage: stage name.
            index: lane index within the stage.
            platform: the platform; a stream is created only when there's a torch-style device module
                (``device_module_for`` returned one: CUDA, NPU, ...). CPU/RPU platforms leave ``stream`` None.
            priority: accelerator scheduling hint within the platform's declared range. Default 0.

        Raises:
            ValueError: priority is not an integer, is nonzero on an unsupported platform,
                or the runtime cannot verify or honor the requested value without clamping.
        """
        from xinfer.platforms.devices import device_module_for

        if platform.capabilities.execution_model != "cuda-lanes":
            raise ValueError("lane dispatch requires execution_model='cuda-lanes'; "
                             f"got {platform.capabilities.execution_model!r}")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError(f"stream priority for {stage} must be an integer, got {priority!r}")
        self._device_module = device_module_for(platform.device)
        if priority != 0 and (self._device_module is None or not platform.capabilities.stream_priorities):
            raise ValueError(f"platform {platform.name} does not support nonzero stream priorities")
        bounds = platform.capabilities.stream_priority_range
        if bounds is not None and not bounds[0] <= priority <= bounds[1]:
            raise ValueError(f"stream priority {priority} for {stage} is outside {platform.name}'s range {bounds}")
        self.stage = stage
        self.index = index
        self.stream = (self._device_module.Stream(device=platform.device, priority=priority)
                       if self._device_module is not None else None)
        if self.stream is not None:
            if platform.capabilities.stream_priorities:
                try:
                    actual = self.stream.priority
                except (AttributeError, RuntimeError) as exc:
                    raise ValueError(f"cannot verify stream priority {priority} for {stage} "
                                     f"on {platform.device}") from exc
                if actual != priority:
                    raise ValueError(f"stream priority {priority} for {stage} is unsupported on {platform.device}; "
                                     f"the runtime would clamp it to {actual}")
            # Weights were loaded and converted on the current stream; order the lane after that work.
            self.stream.wait_stream(self._device_module.current_stream(platform.device))
        self.cudagraphs = None  # LaneCUDAGraphs when the engine uses CUDA graphs
        self.copy_stream = None  # final stages on an accelerator: copies results to the host
        self.event = None  # completion event of the lane's current dispatch

    def use_stream(self):
        """Context manager that makes the lane's stream current (a no-op on the CPU)."""
        if self.stream is None:
            return contextlib.nullcontext()
        return self._device_module.stream(self.stream)

    def make_event(self, *, enable_timing: bool = False):
        """An event on the lane's device, not yet recorded; None on the CPU.

        Args:
            enable_timing: whether ``elapsed_time`` may be read from it (timing costs a little on some runtimes).
        """
        if self._device_module is None:
            return None
        return self._device_module.Event(enable_timing=enable_timing)

    def record(self):
        """Record an event on the lane's stream after the work enqueued so far.

        Returns:
            The recorded device event, or None on the CPU, where the work has already run.
        """
        if self.stream is None:
            return None
        event = self._device_module.Event()
        event.record(self.stream)
        return event

    def wait(self, event) -> None:
        """Make the lane's stream wait for ``event`` on the device, without blocking the host.

        Args:
            event: a CUDA event, typically recorded on another stream; None is a no-op, as is the CPU.
        """
        if self.stream is not None and event is not None:
            self.stream.wait_event(event)


def _to_device(value: Any, device: str) -> Any:
    """Copy a tensor to ``device`` without blocking; other values are returned unchanged.

    A CPU tensor that is not pinned is first copied into pinned memory (synchronously, on the host) so that the
    host-to-device copy is asynchronous on the current stream. A tensor already on ``device`` is returned as is.

    Args:
        value: a tensor or a host value.
        device: target device string, e.g. ``"cuda:0"`` or ``"cpu"``.

    Returns:
        The tensor on ``device`` (its copy may still be in flight on the current stream), or ``value``.
    """
    if torch.is_tensor(value):
        if value.device.type == "cpu" and device != "cpu" and not value.is_pinned():
            value = value.pin_memory()
        return value.to(device, non_blocking=True)
    return value


def allocate_outputs(model: VLAModel, stage: str, batch: StageBatch, device: str) -> dict:
    """Allocate the output buffers ``model`` declares for one stage dispatch.

    Args:
        model: the model; its ``stage_outputs(stage, batch)`` gives the shapes and dtypes.
        stage: stage name.
        batch: the dispatch's ``StageBatch`` (inputs set, ``out`` empty).
        device: device of the buffers.

    Returns:
        ``{name: tensor}`` of new tensors with the declared shapes and dtypes; contents are not initialized.
    """
    specs = model.stage_outputs(stage, batch)
    return {name: torch.empty(spec.shape, dtype=spec.dtype, device=device) for name, spec in specs.items()}


def run_stage(model: VLAModel, stage: str, inputs: Mapping[str, Any], device: str, batch_size: int = 1) -> dict:
    """Run one stage eagerly on the current stream and return its outputs (diagnostics, tests).

    Nothing is synchronized or batched here: ``inputs`` must already hold the whole batch (for several requests,
    for example from ``batching.merge``).

    Args:
        model: the model.
        stage: stage name.
        inputs: the stage's inputs (device tensors and host values); copied into a read-only mapping.
        device: device of the output buffers.
        batch_size: number of requests in the batch (``StageBatch.batch_size``).

    Returns:
        A new dict ``{name: tensor}`` with every output the stage declares.
    """
    batch = StageBatch(stage, batch_size, MappingProxyType(dict(inputs)))
    batch.out = allocate_outputs(model, stage, batch, device)
    with torch.inference_mode():
        model.stage_fn(model.stage_graph[stage])(batch)
    return dict(batch.out)

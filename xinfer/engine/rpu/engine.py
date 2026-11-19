"""Single-request stage scheduling for the prepared RPU TP8 runtime.

CUDA lanes cannot implement this backend: TP8 has one device submission in
flight and device accesses require explicit completion. Models prepare a
resident request plan and choose reductions, packing, state updates, and batch
boundaries. This engine executes those batches in StageGraph order. It does
not copy persistent handoffs or automatically warm up/capture a stateful plan.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from xinfer.engine.core import AdmitError, EngineConfig
from xinfer.platforms.rpu.plan import RpuRequestPlan
from xinfer.platforms.rpu import RpuExecution, RpuExecutionContext


def _owned_cpu(value):
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, order="C")
    # CPU Torch outputs remain supported at the public Policy boundary; they
    # never stand in for RPU buffers or cross a resident stage boundary.
    import torch

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("RPU collect must return CPU materialized outputs")
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _owned_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_owned_cpu(item) for item in value)
    if isinstance(value, list):
        return [_owned_cpu(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool, np.generic)):
        return value
    raise TypeError(f"RPU collect returned unsupported output type {type(value).__name__}")


class RpuEngine:
    """B1, one lane and one admitted request, with real completion waits.

    Concurrent submissions are rejected, including while host plan preparation
    or final output materialization runs. A future can be cancelled before the
    worker starts; an executing request drains normally because the public TP
    ABI provides no safe device cancellation. Shutdown drains that request.
    """

    def __init__(self, model, platform, config=None, *, context=None, context_factory=None):
        if platform.capabilities.execution_model != "explicit-completion":
            raise ValueError("RpuEngine requires execution_model='explicit-completion', got "
                             f"{platform.capabilities.execution_model!r} on {platform.name}")

        self.model = model
        self.platform = platform
        self.config = config or EngineConfig(max_running_requests=1, max_queued_requests=1)
        config = self.config
        if config.max_running_requests != 1 or config.max_queued_requests != 1:
            raise ValueError("RPU admission requires max_running_requests=max_queued_requests=1")
        if config.per_robot_inflight != 1:
            raise ValueError("RPU admission requires per_robot_inflight=1")
        if config.cudagraphs is not None:
            raise ValueError("RPU prepared executions do not use CUDA graph capture")
        if any(value != 1 for value in config.lanes.values()) or any(
                value != 1 for value in config.max_batch.values()):
            raise ValueError("RPU supports one lane and batch size 1 per stage")
        if not callable(getattr(model, "create_rpu_request", None)):
            raise TypeError("RPU models must provide create_rpu_request(prepared, context)")
        if context is not None and context_factory is not None:
            raise ValueError("pass either context or context_factory, not both")
        self.context = context if context is not None else (context_factory or RpuExecutionContext)(
            allow_candidate=bool(getattr(model, "rpu_allow_candidate", False)),
            retained_segments=getattr(model, "rpu_retained_segments", 128),
            file_descriptor_budget=getattr(model, "rpu_file_descriptor_budget", None))
        self._owns_context = context is None
        prepare = getattr(model, "prepare_rpu", None)
        if prepare is not None:
            try:
                prepare(self.context)
            except BaseException as error:
                close_model = getattr(model, "close_rpu", None)
                if close_model is not None:
                    try:
                        close_model()
                    except BaseException as cleanup_error:
                        if hasattr(error, "add_note"):
                            error.add_note(f"RPU model preparation cleanup failed: {cleanup_error}")
                if self._owns_context:
                    try:
                        self.context.close()
                    except BaseException as cleanup_error:
                        if hasattr(error, "add_note"):
                            error.add_note(f"RPU preparation cleanup failed: {cleanup_error}")
                raise
        self.order = tuple(stage.name for stage in model.stage_graph)
        self.stats = Counter()
        self._lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._busy = False
        self._stopping = False
        self._closed = False
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xinfer-rpu")

    def submit(self, prepared, robot=None):
        with self._lock:
            if self._stopping or self._busy:
                future = Future()
                future.set_exception(AdmitError("stopped" if self._stopping else "queue_full"))
                return future
            self._busy = True
            future = Future()
            self._worker.submit(self._work, prepared, future)
        return future

    def _work(self, prepared, future):
        result = error = None
        running = future.set_running_or_notify_cancel()
        if running:
            try:
                result = self._run(prepared)
            except BaseException as caught:
                error = caught
        # Release admission before resolving the public future, so an
        # immediate next run cannot observe the previous request as in flight.
        with self._lock:
            self._busy = False
        if running:
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(result)

    def run(self, prepared):
        return self.submit(prepared).result()

    def _run(self, prepared):
        plan = None
        request_error = None
        try:
            if self.context.needs_reset:
                raise RuntimeError("RPU context requires explicit reset before another request")
            # Plan construction failures are released by the model's request
            # rollback hook. A type-validated plan owns its cleanup callbacks.
            raw_plan = self.model.create_rpu_request(prepared, self.context)
            if not isinstance(raw_plan, RpuRequestPlan):
                raise ValueError("RPU request plan must contain exactly the model StageGraph stages")
            plan = raw_plan
            if set(plan.stages) != set(self.order):
                raise ValueError("RPU request plan must contain exactly the model StageGraph stages")
            for stage in self.order:
                executions = plan.stages[stage]
                if not executions:
                    raise ValueError(f"RPU stage {stage} must contain a prepared execution")
                for execution in executions:
                    if not isinstance(execution, RpuExecution) or execution.context is not self.context:
                        raise ValueError("all request executions must belong to the engine context")
                    execution.execute().wait()
                self.stats[f"dispatch_{stage}_b1"] += 1
            result = {key: _owned_cpu(value) for key, value in plan.collect().items()}
            return result
        except BaseException as error:
            request_error = error
            # An error from plan preparation/materialization also invalidates
            # request state. There is no implicit replay of a partly run plan.
            self.context.needs_reset = True
            rollback = getattr(self.model, "rollback_rpu_request", None)
            if plan is None and rollback is not None:
                try:
                    rollback(error)
                except BaseException as cleanup_error:
                    if hasattr(error, "add_note"):
                        error.add_note(f"RPU request rollback failed: {cleanup_error}")
            elif plan is not None and plan.abort is not None:
                try:
                    plan.abort(error)
                except BaseException as cleanup_error:
                    if hasattr(error, "add_note"):
                        error.add_note(f"RPU abort cleanup failed: {cleanup_error}")
            raise
        finally:
            try:
                if plan is not None and plan.release is not None:
                    plan.release()
            except BaseException as cleanup_error:
                self.context.needs_reset = True
                if request_error is None:
                    request_error = cleanup_error
                    raise
                if hasattr(request_error, "add_note"):
                    request_error.add_note(f"RPU request release failed: {cleanup_error}")
            finally:
                outcome = "requests_failed" if request_error is not None else "requests_completed"
                self.stats[outcome] += 1

    def warmup(self, requests):
        # Exactly one execution per supplied request; no eager-then-capture
        # double advance of recurrent/conv/KV state.
        for prepared in requests:
            self.run(prepared)

    def reset(self):
        with self._lock:
            if self._busy or self._stopping:
                raise RuntimeError("RPU reset requires an idle, open engine")
            try:
                self.context.recover()
                reset = getattr(self.model, "reset_rpu", None)
                if reset is not None:
                    reset()
            except BaseException:
                self.context.needs_reset = True
                raise

    def cudagraph_stats(self):
        return {}

    def shutdown(self):
        with self._shutdown_lock:
            if self._closed:
                return
            with self._lock:
                self._stopping = True
            self._worker.shutdown(wait=True)
            close = getattr(self.model, "close_rpu", None)
            error = None
            try:
                if close is not None:
                    close()
            except BaseException as caught:
                error = caught
            if self._owns_context:
                try:
                    self.context.close()
                except BaseException as caught:
                    if error is not None and hasattr(error, "add_note"):
                        error.add_note(f"RPU context cleanup failed: {caught}")
                    error = error or caught
            if error is not None:
                raise error
            self._closed = True

    close = shutdown

"""Engine core: admission, stage queues, lanes, batching and chained dispatch.

One ``Engine`` serves one GPU with all stages in the same process
(docs/design.md §6–§9). Three threads:

- the **engine thread** owns admission (up to ``max_running_requests``), the per-stage FIFO queues, batch
  formation and every stage's CUDA submission (uploads, graph replays, events);
- the **completer** waits for final events and resolves the callers' futures;
- the **device-image thread** decodes and resizes the ``DeviceImage`` inputs of the admitted requests that have
  them, on its own stream, so the blocking JPEG decode call never blocks the engine thread.

Scheduling rules (§6.3):

1. *Zero-wait batching*: a dispatch takes the queued compatible requests that
   are already waiting, up to the stage's batch limit, never waiting for more.
2. *Chained dispatch*: a request enters a stage's queue as soon as its
   upstream stages are *dispatched*; its lane waits for them on the GPU. With
   free lanes a request therefore runs vision→prefill→action without host
   round trips; under load the queues fill and requests are pipelined and
   batched.
3. Results are returned when the final stage's event completes.

The same code serves one robot and a fleet; only the lane counts and batch
limits differ.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import queue
import threading
import time
from collections import Counter, deque
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import torch

from xinfer.engine import batching
from xinfer.engine.device_images import DeviceImageDecoder
from xinfer.engine.executor import Lane, _to_device, allocate_outputs
from xinfer.engine.cudagraph import CUDAGraphConfig, LaneCUDAGraphs, _ReadTracker, capture_batch_sizes, warmup_requests
from xinfer.engine.slots import PoolExhausted, SlotPool
from xinfer.models.interface import DeviceImage, PreparedRequest, StageBatch, VLAModel
from xinfer.observability.annotate import span
from xinfer.observability.stats import RequestStats, StageStats
from xinfer.platforms.interface import Platform

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    """Engine settings (docs/engine.md, "Settings").

    Attributes:
        lanes: lanes (stream, plus a graph pool with CUDA graphs) per stage name; stages not listed get 1.
        max_batch: batch limit per stage name (default 1); capped by ``Capabilities.batch_sizes``.
        max_running_requests: admitted requests in flight (size of the engine's ``SlotPool``).
        max_queued_requests: submitted requests waiting for admission; beyond it ``submit`` fails with
            ``AdmitError("queue_full")``.
        per_robot_inflight: submitted and unfinished requests per robot id; beyond it ``submit`` fails with
            ``AdmitError("per_robot_limit")``. Requests without a robot id are not limited.
        cudagraphs: capture and replay CUDA graphs per lane with these settings; None runs stages eagerly.
        poll_interval_s: longest time the idle engine thread sleeps before polling the lanes' events again.
        stream_priorities: CUDA stream priority per stage (default 0); smaller integers have higher priority.
    """

    lanes: Mapping[str, int] = field(default_factory=dict)  # per stage, default 1
    max_batch: Mapping[str, int] = field(default_factory=dict)  # per stage, default 1
    max_running_requests: int = 8  # admitted requests in flight
    max_queued_requests: int = 256  # submitted requests waiting for admission
    per_robot_inflight: int = 1  # submitted and unfinished requests per robot
    cudagraphs: CUDAGraphConfig | None = None
    poll_interval_s: float = 0.0002
    stream_priorities: Mapping[str, int] = field(default_factory=dict)


class AdmitError(RuntimeError):
    """Raised (through the future) when a request is not admitted.

    ``submit`` sets it on the returned future instead of raising it.

    Attributes:
        code: the reason: ``"queue_full"`` (``max_queued_requests`` reached), ``"per_robot_limit"``
            (``per_robot_inflight`` reached for the robot) or ``"stopped"`` (the engine is shutting down).
    """

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code  # "queue_full", "per_robot_limit", "stopped"


class _Request:
    """One submitted request and its per-stage state, managed by the engine and completion threads after admission.

    Attributes:
        id: sequence number assigned by ``submit``.
        robot: robot id given to ``submit``; None is not subject to the per-robot limit.
        prepared: the request's ``PreparedRequest`` (CPU tensors).
        future: resolves to the final-stage outputs on the CPU, or fails with the request's error.
        handle: the request's ``SlotHandle`` while admitted, else None.
        inputs: what the stages read: ``prepared.tensors`` with tensors copied to the device, each ``DeviceImage``
            replaced by its decoded view, and host values as given. Set at admission; for a request with
            ``DeviceImage`` values, the device-image thread replaces it once the views are decoded. None before
            admission and after the request finishes.
        inputs_ready: CUDA event that completes when ``inputs`` are on the device; a lane waits for it on the GPU
            before running any of the request's stages. Recorded after the copies at admission; for a request with
            ``DeviceImage`` values, replaced by one recorded after the decoding. None on the CPU.
        events: ``{stage: event}`` recorded on the lane after each dispatched stage (None on the CPU).
        outputs: ``{stage: {name: tensor}}``: request-owned copies of the request's row of every handed-off output
            of each dispatched non-final stage.
        dispatched: names of the stages already dispatched.
        submitted_at: ``time.perf_counter()`` at submission.
        admitted_at: ``time.perf_counter()`` at admission, or None.
        ready_at: ``time.perf_counter()`` when the request was queued for its first stages (its inputs ready).
        timing: ``{stage: (began, ended, batch size)}`` of the stage's dispatch, with a stats sink: timing events
            after the dispatch's waits and after the stage (host times on the CPU).
    """

    __slots__ = ("id", "robot", "prepared", "future", "handle", "inputs", "inputs_ready", "events", "outputs",
                 "dispatched", "submitted_at", "admitted_at", "ready_at", "timing")

    def __init__(self, request_id, robot, prepared, future):
        self.id, self.robot, self.prepared, self.future = request_id, robot, prepared, future
        self.handle = self.inputs = self.inputs_ready = None
        self.events: dict[str, Any] = {}
        self.outputs: dict[str, dict] = {}
        self.dispatched: set[str] = set()
        self.submitted_at = time.perf_counter()
        self.admitted_at = None
        self.ready_at = None
        self.timing: dict[str, tuple] = {}


def _timing_event(lane):
    """A timing CUDA event recorded on ``lane``'s stream, or None on the CPU."""
    if lane.stream is None:
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record(lane.stream)
    return event


class Engine:
    """Serves one model on one device: admission, per-stage queues and lanes, batching and chained dispatch.

    See the module docstring and docs/engine.md for the threads and scheduling rules. ``submit`` and ``run`` may
    be called from any thread; everything else on the request path runs on the engine and completion threads.

    Attributes:
        model: the served ``VLAModel``.
        platform: the ``Platform`` the lanes run on.
        config: the ``EngineConfig`` in use.
        stages: the model's ``StageGraph``.
        order: stage names in topological order.
        final_stages: stages no other stage depends on; their outputs are a request's result.
        lanes: ``{stage: [Lane, ...]}`` with ``config.lanes`` lanes per stage (default 1).
        slots: admission ``SlotPool`` with ``config.max_running_requests`` slots.
        stats: ``Counter`` written by the engine thread: ``dispatch_<stage>_b<N>`` counts dispatches per batch size and
            ``padded_dispatch_<stage>`` batches whose ragged inputs had different lengths.
    """

    def __init__(self, model: VLAModel, platform: Platform, config: EngineConfig | None = None,
                 *, start: bool = True):
        """Validate the model's handoffs, derive the batch sizes, create the lanes and start the threads.

        Each stage's allowed batch sizes are its ``Capabilities.batch_sizes`` plus 1, up to its ``max_batch``.
        With CUDA every lane gets its own stream and admission uploads use one separate upload stream.

        Args:
            model: the model to serve.
            platform: the platform to run on.
            config: engine settings; ``EngineConfig()`` when None.
            start: start the engine and completion threads now; with False, call ``start()`` later.

        Raises:
            ValueError: a handoff names an unknown stage, or its producer is not a direct or transitive dependency
                of its consumer.
            RuntimeError: ``config.cudagraphs`` is set but the platform does not support graph capture.
        """
        self.model = model
        self.platform = platform
        self.config = config = config or EngineConfig()
        self.stages = model.stage_graph
        self.order = [stage.name for stage in self.stages]
        self.final_stages = [stage for stage in self.order if not self.stages.consumers(stage)]
        self._fns = {stage.name: model.stage_fn(stage) for stage in self.stages}
        self._span_names = {stage: f"stage::{stage}" for stage in self.order}  # profilers' stage regions
        names = set(self.order)
        unknown_priorities = set(config.stream_priorities) - names
        if unknown_priorities:
            raise ValueError(f"unknown stage(s) in stream_priorities: {sorted(unknown_priorities)}")
        self._handoff_in: dict[str, list] = {stage: [] for stage in names}
        self._handed_off: dict[str, set[str]] = {stage: set() for stage in names}
        self._axes: dict[str, dict[str, int]] = {stage: {} for stage in names}  # stage -> tensor -> batch axis
        for handoff in model.handoffs:
            if handoff.producer not in names or handoff.consumer not in names:
                raise ValueError(f"handoff {handoff.producer}->{handoff.consumer} names an unknown stage")
            if not self._reaches(handoff.producer, handoff.consumer):
                raise ValueError(f"handoff {handoff.producer}->{handoff.consumer} is not along a stage dependency")
            self._handoff_in[handoff.consumer].append(handoff)
            self._handed_off[handoff.producer].update(handoff.tensors)
            for tensor_name, spec in handoff.tensors.items():
                self._axes[handoff.consumer][tensor_name] = spec.batch_axis
                self._axes[handoff.producer][tensor_name] = spec.batch_axis
        self._ragged = {stage: batching.ragged_inputs(model, stage) for stage in names}

        capabilities = model.capabilities()
        self._batch_sizes = {}
        for stage in self.order:
            allowed = sorted(set(capabilities.batch_sizes.get(stage, (1,))) | {1})
            limit = config.max_batch.get(stage, 1)
            self._batch_sizes[stage] = [b for b in allowed if b <= limit]

        if config.cudagraphs is not None and not platform.capabilities.cudagraphs:
            raise RuntimeError(f"platform {platform.name} does not support graph capture")
        self._reads: dict[str, set[str]] = {}
        self.lanes: dict[str, list[Lane]] = {}
        for stage in self.order:
            lanes = []
            for i in range(config.lanes.get(stage, 1)):
                lane = Lane(stage, i, platform, priority=config.stream_priorities.get(stage, 0))
                lane.event = None
                if config.cudagraphs is not None:
                    lane.cudagraphs = LaneCUDAGraphs(model, lane.stream, platform.device, config.cudagraphs,
                                                     self._reads)
                lanes.append(lane)
            self.lanes[stage] = lanes
        if platform.is_cuda:
            for stage in self.final_stages:
                for lane in self.lanes[stage]:
                    lane.copy_stream = torch.cuda.Stream(device=platform.device)
        self._upload = torch.cuda.Stream(device=platform.device) if platform.is_cuda else None

        self.slots = SlotPool(config.max_running_requests)
        self._queues = {stage: deque() for stage in self.order}
        self._waiting: deque[_Request] = deque()
        self._inflight: Counter = Counter()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._completions: queue.Queue = queue.Queue()
        self._request_ids = itertools.count()
        self._stopping = False
        self.stats = Counter()
        # Called on the completer thread with each completed request's RequestStats (xinfer.observability.stats), or
        # None; xinfer-serve sets it after the warmup to record its metrics.
        self.stats_sink = None
        self._engine_thread = threading.Thread(target=self._run_engine_loop, name="xinfer-engine", daemon=True)
        self._completer = threading.Thread(target=self._completion_loop, name="xinfer-completer", daemon=True)
        # DeviceImage inputs are decoded off the engine thread: on one thread (torchvision shares one nvJPEG decoder
        # per device) with its own stream and a decoder created on first use. Requests in it are counted in
        # _device_inflight (engine thread only) and come back through _device_ready.
        self._device_stream = torch.cuda.Stream(device=platform.device) if platform.is_cuda else None
        self._decoder = None
        self._device_requests: queue.Queue = queue.Queue()
        self._device_ready: list = []
        self._device_inflight = 0
        self._device_thread = threading.Thread(target=self._device_images_loop, name="xinfer-device-images",
                                               daemon=True)
        if start:
            self.start()

    def start(self) -> None:
        """Start the engine, completion and device-image threads (done by the constructor by default).

        Call it once, and only for an engine built with ``start=False``.
        """
        self._engine_thread.start()
        self._completer.start()
        self._device_thread.start()

    # ------------------------------------------------------------- public
    def submit(self, prepared: PreparedRequest, robot: Any = None) -> Future:
        """Queue one request; the future resolves to its final outputs on the CPU.

        Thread-safe and non-blocking. A rejected request is reported through the future, which then already holds
        an ``AdmitError``; nothing is raised. Cancelling the future drops the request at its next admission or
        dispatch step.

        Args:
            prepared: the request's CPU inputs; ``prepared.tensors`` is uploaded to the device at admission.
            robot: hashable robot id for the ``per_robot_inflight`` limit; None is not limited.

        Returns:
            A ``Future`` resolving to ``{name: tensor}`` with every output of the final stage for this request: the
            request's row along each output's batch axis, on the CPU (in pinned memory on CUDA). It fails with
            ``AdmitError`` (``"stopped"``, ``"per_robot_limit"`` or ``"queue_full"``) if the request is not
            accepted, or with the exception raised while dispatching it.
        """
        future: Future = Future()
        with self._lock:
            if self._stopping:
                future.set_exception(AdmitError("stopped"))
            elif robot is not None and self._inflight[robot] >= self.config.per_robot_inflight:
                future.set_exception(AdmitError("per_robot_limit"))
            elif len(self._waiting) >= self.config.max_queued_requests:
                future.set_exception(AdmitError("queue_full"))
            else:
                self._inflight[robot] += 1
                self._waiting.append(_Request(next(self._request_ids), robot, prepared, future))
                self._wake.notify()
        return future

    def run(self, prepared: PreparedRequest) -> dict:
        """Submit one request and wait for it.

        Blocks the calling thread. No robot id is passed, so the per-robot limit does not apply.

        Args:
            prepared: the request's CPU inputs.

        Returns:
            The request's final-stage outputs, as described in ``submit``.

        Raises:
            AdmitError: the request was not accepted (``"stopped"`` or ``"queue_full"``).
            Exception: whatever its stage dispatch raised.
        """
        return self.submit(prepared).result()

    def warmup(self, requests) -> list[dict]:
        """Capture the CUDA graphs of representative requests, at every shape bucket, on every lane and batch size.

        Each request is expanded using ``model.shape_buckets()`` and ``model.pad``, one variant per shape requests
        like it can take, so one request of each kind is enough. Each variant runs alone, at batch size 1 on the
        first lane of every stage; ``capture_batch_sizes`` then captures the same inputs on every lane at every
        allowed batch size. Call it before serving traffic: those captures run on the calling thread. Finishing
        the warmup calls ``finish_warmup()`` on every lane's graphs, so shapes not seen yet follow
        ``CUDAGraphConfig.on_miss_after_warmup``. Without CUDA graphs the variants just run. With them, the time,
        the graphs kept per stage, the capture time and the device memory reserved afterwards are logged.

        Args:
            requests: iterable of ``PreparedRequest``.

        Returns:
            Each request's final-stage outputs (see ``submit``), from its first variant, in order.

        Raises:
            Exception: whatever a warmup request raised (see ``run``); the warmup is then not finished.
        """
        started = time.perf_counter()
        graphs = self.config.cudagraphs is not None
        outputs, variants = [], 0
        for prepared in requests:
            results = [self.run(variant) for variant in warmup_requests(self.model, prepared)]
            outputs.append(results[0])
            variants += len(results)
        if graphs:
            capture_batch_sizes(self.lanes, self._batch_sizes, self._merge)
        for lanes in self.lanes.values():
            for lane in lanes:
                if lane.cudagraphs is not None:
                    lane.cudagraphs.finish_warmup()
        if graphs:
            stats = self.cudagraph_stats()
            logger.info("warmup: %d request variants in %.1f s; graphs kept %s, captured in %.1f s; device memory "
                        "reserved %.2f GiB", variants, time.perf_counter() - started,
                        {stage: counters["graphs"] for stage, counters in stats.items()},
                        sum(counters["capture_ms"] for counters in stats.values()) / 1000,
                        torch.cuda.memory_reserved(self.platform.device) / 2 ** 30)
        return outputs

    def cudagraph_stats(self) -> dict:
        """Per-stage CUDA graph counters, summed over the stage's lanes (read without locking: a snapshot).

        Returns:
            ``{stage: {"captures", "captures_after_warmup", "replays", "eager", "evictions", "capture_ms", "graphs"}}``,
            where ``graphs`` is the number of graphs currently kept (see ``LaneCUDAGraphs.stats``); empty without
            CUDA graphs.
        """
        out = {}
        for stage, lanes in self.lanes.items():
            total = Counter()
            for lane in lanes:
                if lane.cudagraphs is not None:
                    total.update(lane.cudagraphs.stats)
                    total["graphs"] += lane.cudagraphs.graph_count()
            if total:
                out[stage] = dict(total)
        return out

    def request_counts(self) -> tuple[int, int]:
        """Requests waiting for admission, and admitted requests whose slots are not free yet."""
        return len(self._waiting), self.slots.in_use()

    def health(self) -> str | None:
        """Why the engine can no longer serve, or None while it can: its engine, completion or device-image thread
        has stopped.

        Meant for a supervisor polling a started engine; after ``shutdown`` the threads are stopped too.
        """
        for thread in (self._engine_thread, self._completer, self._device_thread):
            if not thread.is_alive():
                return f"{thread.name} thread stopped"
        return None

    def shutdown(self) -> None:
        """Stop the engine, completion and device-image threads, then release CUDA graphs and their pools.

        Call it before interpreter teardown: CUDA objects must not outlive the CUDA context.

        New submissions fail with ``AdmitError("stopped")``. The engine thread still admits and dispatches every request
        already accepted and the completer resolves them, so this blocks until they are finished. Then the lanes'
        graphs are dropped, the device is synchronized and unused cached device memory is released. The engine
        cannot be started again.
        """
        with self._lock:
            self._stopping = True
            self._wake.notify()
        if self._engine_thread.is_alive():
            self._engine_thread.join()
        if self._completer.is_alive():
            self._completions.put(None)
            self._completer.join()
        if self._device_thread.is_alive():
            self._device_requests.put(None)
            self._device_thread.join()
        for lanes in self.lanes.values():
            for lane in lanes:
                lane.cudagraphs = None
        if self.platform.is_cuda:
            torch.cuda.synchronize()
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------- engine loop
    def _run_engine_loop(self) -> None:
        """Engine thread body: repeat ``_step`` until shutdown has drained the waiting, decoding and queued requests.

        An exception from ``_step`` is logged and the loop goes on (a failing dispatch has already failed its
        requests' futures). After a pass without progress the thread sleeps on ``_wake`` for up to
        ``poll_interval_s``, also while requests wait for a slot: ``submit`` and finishing requests (which releases
        their slots) wake it, and the timeout lets it poll the busy lanes' events.
        """
        while True:
            try:
                progressed = self._step()
            except Exception:  # noqa: BLE001 - keep serving; the failing batch got the error
                logger.exception("engine step failed")
                progressed = True
            with self._lock:
                if (self._stopping and not self._waiting and not self._device_inflight
                        and not any(self._queues.values())):
                    return
                if not progressed:
                    self._wake.wait(timeout=self.config.poll_interval_s)

    def _step(self) -> bool:
        """One engine step: admit what fits, then submit stage batches from the last stage to the first.

        Serving downstream stages first drains the pipeline before upstream stages add work. Each stage keeps
        dispatching batches while it has queued requests and a free lane.

        Returns:
            True if a request was admitted or a batch was dispatched.
        """
        progressed = self._admit_requests()
        progressed = self._queue_device_ready() or progressed
        for stage in reversed(self.order):  # downstream first drains the pipeline
            stage_queue = self._queues[stage]
            while stage_queue:
                lane = self._free_lane(stage)
                if lane is None:
                    break
                batch = self._select_batch(stage)
                if batch:
                    self._execute_stage_batch(stage, lane, batch)
                    progressed = True
        return progressed

    def _admit_requests(self) -> bool:
        """Admit waiting requests in FIFO order while slots are free (engine thread).

        For each admitted request: allocate a slot, upload ``prepared.tensors`` to the device (with CUDA on the
        upload stream, CPU tensors pinned first; ``DeviceImage`` values stay as they are) and record
        ``request.inputs_ready`` (inside ``span("admit", request=<id>, robot=<robot>)``), then hand the request to the
        device-image thread if it has ``DeviceImage`` values, else append it to the queue of every stage without
        dependencies. A request whose upload raises fails with
        that exception. A cancelled request at the head is finished and dropped. Admission stops at the first request
        that gets no slot, so later requests never overtake it.

        Returns:
            True if at least one request was admitted (or failed at its upload).
        """
        admitted = False
        while True:
            with self._lock:
                if not self._waiting:
                    return admitted
                request = self._waiting[0]
                if request.future.cancelled():
                    self._waiting.popleft()
                    self._finish_request(request, locked=True)
                    continue
                try:
                    request.handle = self.slots.allocate()
                except PoolExhausted:
                    return admitted
                self._waiting.popleft()
            request.admitted_at = time.perf_counter()
            stream = self._upload
            stream_context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            admitted = True
            try:
                with stream_context, span("admit", request=request.id, robot=request.robot):
                    request.inputs = {name: _to_device(value, self.platform.device)
                                      for name, value in request.prepared.tensors.items()}
                    if stream is not None:
                        request.inputs_ready = torch.cuda.Event(enable_timing=self.stats_sink is not None)
                        request.inputs_ready.record(stream)
            except Exception as exc:  # noqa: BLE001 - only this request fails
                self._fail_request(request, exc)
                continue
            if any(isinstance(value, DeviceImage) for value in request.inputs.values()):
                self._device_inflight += 1
                self._device_requests.put(request)
            else:
                self._queue_for_first_stages(request)

    def _queue_for_first_stages(self, request: _Request) -> None:
        """Append ``request`` to the queue of every stage without dependencies (engine thread)."""
        request.ready_at = time.perf_counter()
        for stage in self.order:
            if not self.stages[stage].depends_on:
                self._queues[stage].append(request)

    def _device_images_loop(self) -> None:
        """Device-image thread body: turn each handed-over request's ``DeviceImage`` values into views, in admission
        order.

        The ``DeviceImageDecoder`` is created on first use; if that raises (no CUDA device, no torchvision), the
        request fails and the next one tries again. On CUDA the work runs on the thread's stream after the request's
        upload, and an event recorded after it becomes the request's ``inputs_ready`` event, which the lanes wait for on
        the GPU: no host synchronization. The result, or the exception it raised, goes to ``_device_ready`` and the
        engine thread is woken. The views are held by the request until it finishes, after its final stage, so no
        stage reads freed memory. ``None`` ends the loop.
        """
        stream = self._device_stream
        while True:
            request = self._device_requests.get()
            if request is None:
                return
            inputs = error = event = None
            try:
                if self._decoder is None:
                    self._decoder = DeviceImageDecoder(self.platform.device)
                context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
                with context, torch.inference_mode():
                    if stream is not None and request.inputs_ready is not None:
                        stream.wait_event(request.inputs_ready)
                    inputs = self._decoder(request.inputs)
            except Exception as exc:  # noqa: BLE001 - only this request fails
                error = exc
            if stream is not None:
                try:
                    event = torch.cuda.Event(enable_timing=self.stats_sink is not None)
                    event.record(stream)
                except Exception:  # noqa: BLE001 - the device is in an error state; the request still fails or runs
                    logger.exception("could not record the device images' completion")
                    event = None
            with self._lock:
                self._device_ready.append((request, inputs, event, error))
                self._wake.notify()

    def _queue_device_ready(self) -> bool:
        """Queue the requests whose views are ready for their first stages; fail those whose decoding raised.

        Engine thread. A failed request's tensors are held until the device-image event, after its enqueued work.

        Returns:
            True if any request came back from the device-image thread.
        """
        if not self._device_inflight:
            return False
        with self._lock:
            ready, self._device_ready = self._device_ready, []
        for request, inputs, event, error in ready:
            self._device_inflight -= 1
            if error is not None:
                self._fail_request(request, error, after=event)
                continue
            request.inputs, request.inputs_ready = inputs, event
            self._queue_for_first_stages(request)
        return bool(ready)

    def _free_lane(self, stage: str):
        """Return the first free lane of ``stage`` with its event cleared, or None if every lane is busy.

        A lane is free when it has no dispatch event or its event has completed (polled with ``query()``).
        """
        for lane in self.lanes[stage]:
            if lane.event is None or lane.event.query():
                lane.event = None
                return lane
        return None

    def _batch_axis(self, stage: str, name: str) -> int:
        """Batch axis of tensor ``name`` at ``stage``: handoff ``TensorSpec``, else ``model.batch_axes``, else 0."""
        return self._axes[stage].get(name, self.model.batch_axes.get(name, 0))

    def _stage_inputs(self, stage: str, request: _Request) -> dict:
        """One request's inputs at ``stage``: its device inputs plus the handoff tensors from its producers.

        Args:
            stage: stage name.
            request: an admitted request whose producer stages have been dispatched.

        Returns:
            A new dict: ``request.inputs`` plus, for every handoff into ``stage``, the producer's request-owned output
            tensors (``request.outputs[producer][name]``).

        Raises:
            ValueError: a handoff tensor has the same name as a prepared input.
        """
        view = dict(request.inputs)
        for handoff in self._handoff_in[stage]:
            for name in handoff.tensors:
                if name in view:
                    raise ValueError(f"handoff tensor {name!r} shadows a prepared input")
                view[name] = request.outputs[handoff.producer][name]
        return view

    def _batch_key(self, stage: str, view: Mapping[str, Any]):
        """Compatibility key of one request at ``stage``; requests are batched together only if their keys are equal.

        The key covers exactly the inputs the stage is known to read (learned on its first dispatch): for a tensor
        its shape without the batch axis (and without the sequence axis of a ragged handoff) and its dtype, for a
        host value its ``repr``, and absence for a name the request does not have.

        Args:
            stage: stage name.
            view: the request's inputs at ``stage`` (from ``_stage_inputs``).

        Returns:
            A hashable tuple ordered by input name, or None while the stage's reads are unknown (the request then
            runs alone).
        """
        reads = self._reads.get(stage)
        if reads is None:
            return None
        parts = []
        for name in sorted(reads):
            if name not in view:
                parts.append((name, None))
                continue
            value = view[name]
            if torch.is_tensor(value):
                seq = self._ragged[stage].get(name, (None, None))[1]
                parts.append((name, batching.compat_shape(value, self._batch_axis(stage, name), seq), value.dtype))
            else:
                parts.append((name, repr(value)))
        return tuple(parts)

    def _select_batch(self, stage: str) -> list[_Request]:
        """Select and remove the next batch of ``stage`` from its queue (zero-wait batching, engine thread).

        The queue head is always taken. If the stage can batch and its reads are known, the rest of the queue is
        scanned in order for non-cancelled requests with the head's ``_batch_key``, up to the stage's largest
        allowed batch size; the members are then cut to the largest allowed size that fits and removed from the
        queue. Incompatible requests keep their places, so compatible requests behind them can overtake them. If
        computing the head's key raises, the head fails with that exception; a request whose key raises is left in
        the queue as incompatible and fails when it becomes the head.

        Args:
            stage: stage name; its queue must not be empty.

        Returns:
            The batch's requests, head first; empty if the head was cancelled (it is finished and dropped) or failed.
        """
        stage_queue = self._queues[stage]
        head = stage_queue.popleft()
        if head.future.cancelled():
            self._finish_request(head)
            return []
        batch_sizes = self._batch_sizes[stage]
        if batch_sizes[-1] == 1 or not stage_queue:
            return [head]
        try:
            key = self._batch_key(stage, self._stage_inputs(stage, head))
        except Exception as exc:  # noqa: BLE001 - only this request fails
            self._fail_request(head, exc)
            return []
        if key is None:
            return [head]
        members = [head]
        for request in list(stage_queue):
            if len(members) == batch_sizes[-1]:
                break
            if request.future.cancelled():
                continue
            try:
                compatible = self._batch_key(stage, self._stage_inputs(stage, request)) == key
            except Exception:  # noqa: BLE001 - the request fails when it becomes the head
                compatible = False
            if compatible:
                members.append(request)
        batch_size = max(b for b in batch_sizes if b <= len(members))
        members = members[:batch_size]
        for request in members[1:]:
            stage_queue.remove(request)
        return members

    def _merge(self, stage: str, views: list[dict]) -> dict:
        """Merge a batch's per-request inputs into the stage's ``StageBatch.inputs`` (``batching.merge``).

        Counts ``stats["padded_dispatch_<stage>"]`` when the batch's ragged inputs differ in length.

        Args:
            stage: stage name.
            views: per-request inputs from ``_stage_inputs``, in batch order.

        Returns:
            A mapping: a dict for one request, a lazily merged ``batching.MergedInputs`` for several. Ragged
            handoff tensors are left-padded and come with ``<name>_lengths`` (``int32 [B]``).
        """
        axes = {name: self._batch_axis(stage, name) for name, value in views[0].items() if torch.is_tensor(value)}
        merged = batching.merge(views, axes, self._ragged[stage])
        for name, (_, seq_axis) in self._ragged[stage].items():
            if len(views) > 1 and len({view[name].shape[seq_axis] for view in views if name in view}) > 1:
                self.stats[f"padded_dispatch_{stage}"] += 1
                break
        return merged

    def _execute_stage_batch(self, stage: str, lane: Lane, batch: list[_Request]) -> None:
        """Submit one batch of ``stage`` on ``lane`` and pass its results on (engine thread).

        On CUDA, work is submitted asynchronously; returning does not mean the stage has completed. The lane's
        stream waits for each request's input and upstream stage events, the inputs are merged, and the stage runs
        by graph replay (``lane.cudagraphs``) or eagerly into newly allocated outputs (the eager run also records
        which inputs the stage reads). Outputs may live in lane-owned buffers, so they are copied out before the
        lane's event is recorded: at the final stage each request's row goes to the host (``_copy_to_host``);
        otherwise each request's row of every handed-off output is cloned into ``request.outputs[stage]``.

        The dispatch runs inside ``span("stage::<stage>", batch_size=B, lane=i, requests="<id>,...")``, and the lane's
        waits for the inputs and upstream stages inside ``span("wait")`` (``xinfer.observability.annotate``): the
        shared no-op context unless a profiler sink is configured. The lane
        stays busy until the recorded event completes. A final batch goes to the completer; otherwise each request is
        appended to the queue of every consumer stage whose dependencies are all dispatched.

        Args:
            stage: stage name.
            lane: a free lane of ``stage``.
            batch: admitted requests from ``_select_batch``, in batch order.

        Raises:
            Exception: whatever the dispatch raised on the host; the batch's requests are failed with it
                (``_fail_request``) before it is re-raised. Work already enqueued, including part of the copy to
                the host, may still read the lane's outputs and the requests' tensors, so the lane and the slots are
                released after it completes.
        """
        batch_size = len(batch)
        final = stage in self.final_stages
        timed = self.stats_sink is not None
        try:
            with lane.use_stream(), torch.inference_mode(), span(
                    self._span_names[stage], batch_size=batch_size, lane=lane.index,
                    requests=",".join(str(request.id) for request in batch)):
                with span("wait"):
                    for request in batch:
                        lane.wait(request.inputs_ready)
                        for dependency in self.stages[stage].depends_on:
                            lane.wait(request.events[dependency])
                if timed:
                    began = _timing_event(lane)
                    began_at = time.perf_counter()
                inputs = self._merge(stage, [self._stage_inputs(stage, r) for r in batch])
                if lane.cudagraphs is not None:
                    out = lane.cudagraphs.run(stage, batch_size, inputs)
                else:
                    tracked = _ReadTracker(inputs)
                    stage_batch = StageBatch(stage, batch_size, tracked)
                    stage_batch.out = allocate_outputs(self.model, stage, stage_batch, self.platform.device)
                    self._check_declared(stage, stage_batch.out)
                    self._fns[stage](stage_batch)
                    self._reads.setdefault(stage, set()).update(tracked.names)
                    out = stage_batch.out
                self._check_declared(stage, out)
                if timed:
                    ended = _timing_event(lane)
                    ended_at = time.perf_counter()
                if final:
                    hosts, event = self._copy_to_host(stage, lane, out, batch_size)
                else:  # copy out of the lane's buffers before its next dispatch
                    for i, request in enumerate(batch):
                        request.outputs[stage] = {name: self._row(stage, name, out[name], i, batch_size).clone()
                                              for name in self._handed_off[stage]}
                    event = lane.record()
        except Exception as exc:  # noqa: BLE001
            after = None
            if lane.stream is not None:
                try:
                    if lane.copy_stream is not None:
                        lane.stream.wait_stream(lane.copy_stream)
                    after = lane.event = lane.record()
                except Exception:  # noqa: BLE001 - the device is in an error state; nothing more runs on it
                    logger.exception("could not record the failed dispatch's completion")
            for request in batch:
                self._fail_request(request, exc, after=after)
            raise
        lane.event = event
        self.stats[f"dispatch_{stage}_b{batch_size}"] += 1
        if timed:
            timing = (began_at, ended_at, batch_size) if ended is None else (began, ended, batch_size)
            for request in batch:
                request.timing[stage] = timing
        for request in batch:
            request.events[stage] = event
            request.dispatched.add(stage)
        if final:
            self._completions.put((event, batch, hosts))
            return
        for consumer in self.stages.consumers(stage):
            deps = self.stages[consumer].depends_on
            for request in batch:
                if all(d in request.dispatched for d in deps):
                    self._queues[consumer].append(request)

    def _row(self, stage, name, tensor, i, batch_size):
        """Request ``i``'s row of output ``name`` in a batch of ``batch_size``.

        At batch size 1 this is ``tensor`` itself, else a ``narrow`` view of length 1 along the batch axis of ``name``.
        """
        return tensor if batch_size == 1 else tensor.narrow(self._batch_axis(stage, name), i, 1)

    def _copy_to_host(self, stage, lane, out, batch_size):
        """Copy a final dispatch's rows to the host.

        On CUDA the copies run on the lane's ``copy_stream`` after the lane's work. A pinned buffer records an event
        on every stream it was used on when it is freed, which callers do on their own threads; recorded on a
        stream during a capture, that event would make the next pinned allocation fail.

        Args:
            stage: stage name.
            lane: the lane that ran the dispatch.
            out: the dispatch's outputs ``{name: tensor}``.
            batch_size: number of requests in the batch.

        Returns:
            ``(hosts, event)``: one ``{name: CPU tensor}`` per row, valid once ``event`` completes (None on the CPU).
        """
        if lane.copy_stream is None:
            return [self._to_host(stage, out, i, batch_size) for i in range(batch_size)], lane.record()
        ready = lane.record()
        with torch.cuda.stream(lane.copy_stream):
            lane.copy_stream.wait_event(ready)
            for value in out.values():  # eager outputs must outlive the copies
                value.record_stream(lane.copy_stream)
            hosts = [self._to_host(stage, out, i, batch_size) for i in range(batch_size)]
            event = torch.cuda.Event()
            event.record(lane.copy_stream)
        return hosts, event

    def _to_host(self, stage, out, i, batch_size) -> dict:
        """Copy request ``i``'s row of every output to the host.

        On CUDA each row is copied asynchronously on the current stream into a new pinned buffer, which holds the
        data only after the event recorded next on that stream completes (the completer synchronizes it). On the
        CPU the rows are cloned.

        Args:
            stage: stage name (for the batch axes).
            out: the dispatch's outputs ``{name: tensor}``.
            i: row index in the batch.
            batch_size: number of requests in the batch.

        Returns:
            ``{name: CPU tensor}``.
        """
        host = {}
        for name, value in out.items():
            row = self._row(stage, name, value, i, batch_size)
            if self.platform.is_cuda:
                host_buffer = torch.empty(row.shape, dtype=row.dtype, pin_memory=True)
                host_buffer.copy_(row, non_blocking=True)
                host[name] = host_buffer
            else:
                host[name] = row.clone()
        return host

    def _check_declared(self, stage: str, outputs) -> None:
        """Check that ``outputs`` has every tensor ``stage`` hands off to later stages.

        Raises:
            ValueError: a handed-off tensor is missing.
        """
        missing = self._handed_off[stage] - set(outputs)
        if missing:
            raise ValueError(f"stage {stage!r} does not declare {sorted(missing)}")

    def _reaches(self, upstream: str, stage: str) -> bool:
        """Whether ``upstream`` is a direct or transitive dependency (``Stage.depends_on``) of ``stage``."""
        pending = list(self.stages[stage].depends_on)
        while pending:
            name = pending.pop()
            if name == upstream:
                return True
            pending.extend(self.stages[name].depends_on)
        return False

    # ----------------------------------------------------------- completer
    def _completion_loop(self) -> None:
        """Completer thread body: wait for final-stage batches and resolve their futures.

        Each item is ``(event, batch, hosts)`` from ``_execute_stage_batch``; this thread waits on the event until
        the final stage and its device-to-host copies have finished (there is no event on the CPU). Each request is
        finished (slot and per-robot count released) before its future gets its ``hosts`` entry, so a caller can
        submit the robot's next request as soon as it has the result. Futures already done (cancelled) are left
        alone. If waiting on the event raises, the batch's requests fail with that exception and the loop goes on.
        A ``None`` item, queued by ``shutdown``, ends the loop.
        """
        while True:
            item = self._completions.get()
            if item is None:
                return
            event, batch, hosts = item
            try:
                if event is not None:
                    event.synchronize()
            except Exception as exc:  # noqa: BLE001 - fail this batch, keep completing the others
                for request in batch:
                    self._fail_request(request, exc)
                continue
            for request, host in zip(batch, hosts):
                self._finish_request(request)
                if not request.future.done():
                    request.future.set_result(host)
            sink = self.stats_sink
            if sink is not None:
                for request in batch:
                    stats = self._request_stats(request)
                    if stats is not None:
                        try:
                            sink(stats)
                        except Exception:  # noqa: BLE001 - statistics never stop the completer
                            logger.exception("the stats sink failed")

    def _request_stats(self, request: _Request) -> RequestStats | None:
        """A completed request's ``RequestStats`` (completer).

        A stage waits from the moment it can start (the request's inputs are ready on the device, or its upstream
        stages have finished) to the start of its dispatch, then executes; along the request the waits and execution
        times add up to its inputs being ready to its final stage finishing. The times come from timing events on
        CUDA (all complete once the final stage is) and from host times on the CPU.

        Returns:
            The stats, or None for a request admitted before the sink was set (it has no timing).
        """
        if set(request.timing) != set(self.order):
            return None
        try:
            if request.inputs_ready is None:  # CPU: host times
                times = {stage: (began - request.ready_at, ended - request.ready_at)
                         for stage, (began, ended, _) in request.timing.items()}
            else:
                since = request.inputs_ready.elapsed_time
                times = {stage: (since(began) / 1000, since(ended) / 1000)
                         for stage, (began, ended, _) in request.timing.items()}
        except RuntimeError:  # an event recorded without timing, before the sink was set
            return None
        stages = {}
        for stage, (began, ended) in times.items():
            ready = max((times[upstream][1] for upstream in self.stages[stage].depends_on), default=0.0)
            stages[stage] = StageStats(wait=began - ready, execution=ended - began, batch_size=request.timing[stage][2])
        return RequestStats(admission_wait=request.admitted_at - request.submitted_at, stages=stages)

    def _fail_request(self, request: _Request, exc: BaseException, *, after: Any = None) -> None:
        """Finish ``request``, then fail its future with ``exc`` unless the future is already done.

        Finishing first releases the robot's in-flight count, so the robot can submit again from the future's
        callbacks, as after a result. With ``after``, the slot keeps the request's tensors until that event completes.
        """
        self._finish_request(request, after=after)
        if not request.future.done():
            request.future.set_exception(exc)

    def _finish_request(self, request: _Request, locked: bool = False, after: Any = None) -> None:
        """Release what a request holds once it completed, failed or was dropped (engine or completion thread).

        Releases its slot (if admitted), drops its references to its device inputs and stage outputs, decrements
        its robot's in-flight count and wakes the engine thread. The future is not touched.

        Args:
            request: the request.
            locked: True when the caller already holds ``self._lock``.
            after: event of a failed dispatch's enqueued work; the slot keeps the request's tensors until it completes.
        """
        if request.handle is not None:
            if after is not None:
                self.slots.tensors(request.handle)["failed_dispatch"] = (request.inputs, dict(request.outputs))
            self.slots.release(request.handle, after=after)
            request.handle = None
        request.inputs = None
        request.outputs.clear()
        if locked:
            self._inflight[request.robot] -= 1
            self._wake.notify()
        else:
            with self._lock:
                self._inflight[request.robot] -= 1
                self._wake.notify()

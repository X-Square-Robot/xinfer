"""Prometheus metrics of ``xinfer-serve``, answered at ``GET /metrics`` on the serving port (docs/metrics.md).

The engine process and every frontend process record with ``prometheus_client`` in its multiprocess mode: each
process writes its values to files in the directory named by ``PROMETHEUS_MULTIPROC_DIR``, and ``render``, run by
whichever frontend receives the scrape, reads them all. The variable must be set before ``prometheus_client`` is
first imported (``enable``); frontends spawned afterwards inherit it.

``FrontendMetrics`` records each request a frontend answers. ``EngineMetrics.record`` records the engine's
``RequestStats`` of each completed request (``xinfer-serve`` makes it the engine's ``stats_sink``), and
``EngineMetrics.sample`` reads the engine's queue lengths, health and CUDA graph captures (``xinfer-serve`` calls it
from its supervision loop). The engine itself knows nothing of these metrics.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from xinfer.observability.stats import RequestStats

ENV = "PROMETHEUS_MULTIPROC_DIR"
# Request latencies (seconds): robots expect a response within one control period of about 0.2 s.
LATENCY_BUCKETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0)
# One stage dispatch (seconds).
STAGE_BUCKETS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0)


def available() -> bool:
    """Whether ``prometheus_client`` is installed (``pip install -e '.[serve]'``), without importing it."""
    import importlib.util

    return importlib.util.find_spec("prometheus_client") is not None


def enable(directory) -> None:
    """Record into ``directory`` from this process and the processes it starts afterwards.

    Raises:
        RuntimeError: ``prometheus_client`` was imported before: its values would stay in this process only.
    """
    import sys

    if "prometheus_client" in sys.modules and os.environ.get(ENV) != str(directory):
        raise RuntimeError("prometheus_client was imported before the metrics directory was set")
    os.environ[ENV] = str(directory)


def render() -> tuple[bytes, str]:
    """All processes' metrics in the Prometheus text format, and its content type."""
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return generate_latest(registry), CONTENT_TYPE_LATEST


def mark_dead(pid: int) -> None:
    """Drop a stopped process's live gauges (its counters and histograms stay in the totals)."""
    from prometheus_client import multiprocess

    multiprocess.mark_process_dead(pid)


class FrontendMetrics:
    """What a frontend records per request.

    Args:
        registry: ``prometheus_client`` registry; the default one in ``xinfer-serve``, a separate one in tests.
    """

    def __init__(self, registry=None):
        from prometheus_client import REGISTRY, Counter, Gauge, Histogram

        registry = REGISTRY if registry is None else registry
        self.requests = Counter("xinfer_requests", "Requests answered, by response status.", ["status"],
                                registry=registry)
        self.request_seconds = Histogram(
            "xinfer_request_duration_seconds", "Server time of a successful request: message received to response "
            "built (server_timing.infer_ms).", buckets=LATENCY_BUCKETS, registry=registry)
        self.prepare_seconds = Histogram(
            "xinfer_request_prepare_seconds", "Decoding and the InputProcessor (server_timing.prepare_ms).",
            buckets=LATENCY_BUCKETS, registry=registry)
        self.engine_seconds = Histogram(
            "xinfer_request_engine_seconds",
            "Shared-memory packing and the engine round trip (server_timing.engine_ms).",
            buckets=LATENCY_BUCKETS, registry=registry)
        self.output_seconds = Histogram(
            "xinfer_request_output_seconds", "The OutputProcessor (server_timing.output_ms).",
            buckets=LATENCY_BUCKETS, registry=registry)
        self.connections = Gauge("xinfer_connections", "Open WebSocket connections (robots).",
                                 multiprocess_mode="livesum", registry=registry)

    def answered(self, status: str, timing: Mapping[str, float] | None = None) -> None:
        """Count a response; for a successful one also its ``server_timing`` (milliseconds)."""
        self.requests.labels(status).inc()
        if timing is not None:
            self.request_seconds.observe(timing["infer_ms"] / 1000)
            self.prepare_seconds.observe(timing["prepare_ms"] / 1000)
            self.engine_seconds.observe(timing["engine_ms"] / 1000)
            self.output_seconds.observe(timing["output_ms"] / 1000)


class EngineMetrics:
    """What ``xinfer-serve`` records about its engine: each completed request's ``RequestStats``, and sampled state.

    Args:
        info: labels of ``xinfer_model_info`` (model, quantization, numerics, platform, version).
        registry: ``prometheus_client`` registry; the default one in ``xinfer-serve``, a separate one in tests.
    """

    def __init__(self, info: Mapping[str, str], registry=None):
        from prometheus_client import REGISTRY, Counter, Gauge, Histogram

        registry = REGISTRY if registry is None else registry
        self.admission_seconds = Histogram(
            "xinfer_admission_wait_seconds", "Submission to admission (a free slot) in the engine.",
            buckets=LATENCY_BUCKETS, registry=registry)
        self.stage_wait_seconds = Histogram(
            "xinfer_stage_wait_seconds", "Per request: from when a stage can start (inputs on the device, or its "
            "upstream stages finished) to the start of its dispatch.", ["stage"], buckets=LATENCY_BUCKETS,
            registry=registry)
        self.stage_seconds = Histogram(
            "xinfer_stage_execution_seconds", "Per request: the device time of the stage dispatch that served it.",
            ["stage"], buckets=STAGE_BUCKETS, registry=registry)
        self.dispatches = Counter("xinfer_stage_dispatches", "Stage dispatches of completed requests, by batch size "
                                  "(each request adds 1/batch size).", ["stage", "batch_size"], registry=registry)
        self.waiting = Gauge("xinfer_requests_waiting", "Requests waiting for admission.",
                             multiprocess_mode="livesum", registry=registry)
        self.running = Gauge("xinfer_requests_running", "Admitted requests not yet finished.",
                             multiprocess_mode="livesum", registry=registry)
        self.healthy = Gauge("xinfer_engine_healthy", "1 while the engine's threads run, else 0.",
                             multiprocess_mode="livesum", registry=registry)
        self.late_captures = Counter(
            "xinfer_cudagraph_captures_after_warmup", "CUDA graphs captured while serving: shapes the warmup did "
            "not cover; each capture delays its request.", ["stage"], registry=registry)
        model_info = Gauge("xinfer_model_info", "The served model and build (value 1).", sorted(info),
                           multiprocess_mode="livemax", registry=registry)
        model_info.labels(**info).set(1)
        self._captures: dict | None = None  # the engine's graph captures at the last sample; None before the first

    def record(self, stats: RequestStats) -> None:
        """Record a completed request's admission wait, and each stage's wait, execution and dispatch (the sink).

        A dispatch of ``B`` requests is counted as ``1 / B`` by each of them, so the dispatch counts and the request
        counts change together and their ratio is the mean batch size.
        """
        self.admission_seconds.observe(stats.admission_wait)
        for stage, times in stats.stages.items():
            self.stage_wait_seconds.labels(stage).observe(times.wait)
            self.stage_seconds.labels(stage).observe(times.execution)
            self.dispatches.labels(stage, str(times.batch_size)).inc(1 / times.batch_size)

    def sample(self, engine) -> None:
        """Record ``engine``'s queue lengths and health, and its new CUDA graph captures after warmup.

        The first call sets the captures' baseline and starts their series at 0 (``xinfer-serve`` calls it after the
        warmup).
        """
        waiting, running = engine.request_counts()
        self.waiting.set(waiting)
        self.running.set(running)
        self.healthy.set(1 if engine.health() is None else 0)
        captures = {stage: stats.get("captures_after_warmup", 0) for stage, stats in engine.cudagraph_stats().items()}
        counted = captures if self._captures is None else self._captures  # stages with CUDA graphs only
        for stage, total in captures.items():
            self.late_captures.labels(stage).inc(total - counted.get(stage, 0))
        self._captures = captures

"""Prometheus metrics of ``xinfer-serve``: what the engine and a frontend record, and the multiprocess exposition."""

import os
import subprocess
import sys
import textwrap
import time
from concurrent.futures import wait
from pathlib import Path

import pytest
import torch

pytest.importorskip("prometheus_client")
from prometheus_client import CollectorRegistry  # noqa: E402
from prometheus_client.parser import text_string_to_metric_families  # noqa: E402

from xinfer.engine import Engine, EngineConfig  # noqa: E402
from xinfer.models.interface import (Capabilities, HandoffSpec, OutputSpec, PreparedRequest, Stage,  # noqa: E402
                                     StageGraph, TensorSpec, VLAModel)
from xinfer.observability.metrics import EngineMetrics, FrontendMetrics  # noqa: E402
from xinfer.observability.stats import RequestStats, StageStats  # noqa: E402
from xinfer.platforms import resolve_platform  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INFO = dict(model="two_stage", quantization="bf16", numerics="", platform="cpu", version="test")
TIMING = dict(infer_ms=30.0, prepare_ms=10.0, engine_ms=15.0, output_ms=5.0)


class TwoStage(VLAModel):
    """a: h = 2x;  b (batches up to 4): y = h + 1."""

    stage_graph = StageGraph(Stage("a", "first"), Stage("b", "second", depends_on="a"))
    handoffs = (HandoffSpec("a", "b", {"h": TensorSpec(batch_axis=0)}),)

    def capabilities(self):
        return Capabilities(batch_sizes={"a": (1,), "b": (1, 2, 4)}, num_steps=0)

    def stage_outputs(self, stage, batch):
        n = batch.inputs["x"].shape[-1]
        return {"h" if stage == "a" else "y": OutputSpec((batch.batch_size, n), torch.float32)}

    def first(self, batch):
        batch.out["h"].copy_(batch.inputs["x"] * 2)

    def second(self, batch):
        batch.out["y"].copy_(batch.inputs["h"] + 1)


def _serve(engine, n=6):
    """Submit ``n`` requests, check their results; returns the time from the first submission to the last result."""
    submitted = time.perf_counter()
    futures = [engine.submit(PreparedRequest({"x": torch.full((1, 3), float(i))})) for i in range(n)]
    wait(futures, timeout=30)
    elapsed = time.perf_counter() - submitted
    for i, future in enumerate(futures):
        torch.testing.assert_close(future.result()["y"], torch.full((1, 3), 2.0 * i + 1))
    return elapsed


def test_the_engine_reports_each_request_s_stage_waits_and_executions_to_its_sink():
    reported = []
    engine = Engine(TwoStage(), resolve_platform("cpu"), EngineConfig(max_running_requests=4))
    engine.stats_sink = reported.append
    try:
        elapsed = _serve(engine)
    finally:
        engine.shutdown()
    assert len(reported) == 6
    for stats in reported:
        assert isinstance(stats, RequestStats) and set(stats.stages) == {"a", "b"} and stats.admission_wait >= 0
        assert all(times.wait >= 0 and times.execution >= 0 for times in stats.stages.values())
        assert stats.stages["a"].batch_size == 1 and 1 <= stats.stages["b"].batch_size <= 4
        # along the request: its inputs ready to its final stage finished, within its time in the engine
        assert 0 < sum(times.wait + times.execution for times in stats.stages.values()) <= elapsed


def test_a_failing_sink_does_not_stop_the_completer():
    def sink(stats):
        raise RuntimeError("sink failed")

    engine = Engine(TwoStage(), resolve_platform("cpu"), EngineConfig(max_running_requests=4))
    engine.stats_sink = sink
    try:
        _serve(engine)
        _serve(engine)  # the completer still resolves later requests
    finally:
        engine.shutdown()


class Sampled:
    """What ``EngineMetrics.sample`` reads of an engine."""

    def __init__(self):
        self.graphs, self.counts = {}, (0, 0)

    def request_counts(self):
        return self.counts

    def health(self):
        return None

    def cudagraph_stats(self):
        return self.graphs


def test_engine_metrics_record_request_stats_and_count_each_dispatch_once():
    registry = CollectorRegistry()
    metrics = EngineMetrics(INFO, registry=registry)
    for wait_s in (0.001, 0.003):  # two requests of one dispatch of 2
        metrics.record(RequestStats(admission_wait=0.002, stages={"a": StageStats(wait_s, 0.010, batch_size=2)}))
    metrics.record(RequestStats(admission_wait=0.002, stages={"a": StageStats(0.002, 0.010, batch_size=1)}))
    get = registry.get_sample_value
    assert get("xinfer_admission_wait_seconds_count") == 3
    assert get("xinfer_stage_wait_seconds_sum", {"stage": "a"}) == pytest.approx(0.006)
    assert get("xinfer_stage_execution_seconds_count", {"stage": "a"}) == 3
    assert get("xinfer_stage_dispatches_total", {"stage": "a", "batch_size": "2"}) == pytest.approx(1)
    assert get("xinfer_stage_dispatches_total", {"stage": "a", "batch_size": "1"}) == pytest.approx(1)
    assert get("xinfer_model_info", INFO) == 1


def test_engine_metrics_sample_the_state_and_graph_captures_from_a_baseline():
    registry = CollectorRegistry()
    metrics = EngineMetrics(INFO, registry=registry)
    engine = Sampled()
    engine.graphs = {"a": {"captures_after_warmup": 2}}  # before the baseline: not counted
    metrics.sample(engine)
    engine.graphs, engine.counts = {"a": {"captures_after_warmup": 3}}, (2, 3)
    metrics.sample(engine)
    get = registry.get_sample_value
    assert get("xinfer_cudagraph_captures_after_warmup_total", {"stage": "a"}) == 1
    assert (get("xinfer_requests_waiting"), get("xinfer_requests_running"), get("xinfer_engine_healthy")) == (2, 3, 1)


def test_a_frontend_counts_every_response_and_times_the_successful_ones():
    registry = CollectorRegistry()
    recorded = FrontendMetrics(registry=registry)
    recorded.answered("ok", TIMING)
    recorded.answered("queue_full")
    get = registry.get_sample_value
    assert get("xinfer_requests_total", {"status": "ok"}) == 1
    assert get("xinfer_requests_total", {"status": "queue_full"}) == 1
    assert get("xinfer_request_duration_seconds_count") == 1
    assert get("xinfer_request_duration_seconds_sum") == pytest.approx(0.030)
    assert get("xinfer_request_engine_seconds_sum") == pytest.approx(0.015)


def test_render_sums_the_processes_recording_into_one_directory(tmp_path):
    """Two processes record into the directory; a third renders the sum, as a frontend answers a scrape."""
    script = textwrap.dedent("""
        import sys
        from xinfer.observability import metrics

        metrics.enable(sys.argv[1])
        recorded = metrics.FrontendMetrics()
        for _ in range(int(sys.argv[2])):
            recorded.answered("ok", dict(infer_ms=20.0, prepare_ms=5.0, engine_ms=10.0, output_ms=5.0))
        if len(sys.argv) > 3:
            body, content_type = metrics.render()
            print(content_type)
            sys.stdout.write(body.decode())
    """)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)

    def run(*args):
        return subprocess.run([sys.executable, "-c", script, str(tmp_path), *args], capture_output=True, text=True,
                              check=True, env=env).stdout

    run("2")
    content_type, text = run("3", "render").split("\n", 1)
    assert content_type.startswith("text/plain")
    samples = {(s.name, tuple(sorted(s.labels.items()))): s.value
               for family in text_string_to_metric_families(text) for s in family.samples}
    assert samples[("xinfer_requests_total", (("status", "ok"),))] == 5
    assert samples[("xinfer_request_duration_seconds_count", ())] == 5
    assert samples[("xinfer_request_duration_seconds_sum", ())] == pytest.approx(0.1)

"""xinfer-profile load: segment times from the engine's spans under closed-loop robots (CPU)."""

import csv
import json
import os
from pathlib import Path

import pytest

from xinfer.entrypoints import profile
from xinfer.models.interface import Stage, StageGraph
from xinfer.observability import load_profile


def _added_up(row):
    """Sum of a row's segments that add up to ``total_ms``."""
    return sum(value for name, value in row.items()
               if name.endswith("_ms") and name != "total_ms" and not name.endswith(load_profile.OVERLAPPING))


class _Event:
    """A CUDA timing event stand-in at ``ms`` on a GPU clock."""

    def __init__(self, ms):
        self.ms = ms

    def elapsed_time(self, other):
        return other.ms - self.ms


def test_segments_place_the_gpu_events_on_the_host_clock():
    recorder = load_profile.SpanRecorder(cuda=False)
    recorder.admits = {7: ("0:3", 10.002, 10.003), 8: ("1:0", 10.002, 10.003)}
    recorder.dispatches = [
        dict(stage="a", lane=0, batch_size=1, requests=[7], start=10.004, end=10.0045,
             began=_Event(0), waited=_Event(1), ended=_Event(6)),
        dict(stage="b", lane=1, batch_size=2, requests=[7, 8], start=10.005, end=10.0055,
             began=_Event(0), waited=_Event(5), ended=_Event(9)),  # the lane waits for a on the GPU
        dict(stage="a", lane=1, batch_size=1, requests=[8], start=10.0035, end=10.004,
             began=_Event(0), waited=_Event(0), ended=_Event(4))]
    stages = [(stage.name, stage.depends_on) for stage in StageGraph(Stage("a", "a"), Stage("b", "b", depends_on="a"))]
    rows = load_profile.segments(recorder, {"0:3": (10.000, 10.016), "1:0": (10.001, 10.015)}, stages)
    assert [(row["robot"], row["index"]) for row in rows] == [(0, 3), (1, 0)]
    assert rows[0] == pytest.approx(dict(
        robot=0, index=3, admit_ms=2, inputs_ms=1,
        **{"a.delay_ms": 2, "a.run_ms": 5, "a.batch_size": 1, "a.queue_ms": 1, "a.wait_ms": 1,
           "b.delay_ms": 0, "b.run_ms": 4, "b.batch_size": 2, "b.queue_ms": 0.5, "b.wait_ms": 5},
        complete_ms=2, total_ms=16))  # a done on the GPU at 10.010, b starts then and is done at 10.014
    assert rows[1]["b.delay_ms"] == pytest.approx(2.5)  # its a is done at 10.0075; b starts with request 7's
    assert [_added_up(row) for row in rows] == pytest.approx([row["total_ms"] for row in rows])
    summary = load_profile.summarize(rows)
    assert summary["batch_sizes"] == {"a": {1: 2}, "b": {2: 2}}
    assert set(summary["tail"]["p99"]) == {"admit_ms", "inputs_ms", "a.delay_ms", "a.run_ms", "b.delay_ms",
                                           "b.run_ms", "complete_ms", "total_ms"}


@pytest.mark.parametrize("start,offsets", [("spread", [0.0, 0.05, 0.1, 0.15]), ("together", [0.0] * 4)])
def test_robots_spread_their_first_requests_over_a_period_or_send_them_together(start, offsets):
    from concurrent.futures import Future

    class Engine:  # answers at once
        def submit(self, prepared, robot):
            future = Future()
            future.set_result({})
            return future

    times = load_profile.run_load(Engine(), [None], robots=4, per_robot=1, period_s=0.2, start=start)
    first = min(submitted for submitted, _ in times.values())
    assert [times[f"{r}:0"][0] - first for r in range(4)] == pytest.approx(offsets, abs=0.02)


def test_load_reports_each_request_with_its_stage_segments(tmp_path):
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps([{"x": [1.0, 2.0, 3.0]}, {"x": [4.0, 5.0, 6.0]}]))
    out = tmp_path / "out"
    assert profile.main(["load", "--model", "echo_model:EchoModel", "--checkpoint", "unused", "--requests",
                         str(requests), "--device", "cpu", "--robots", "3", "--requests-per-robot", "6",
                         "--warmup-requests", "2", "--period-ms", "5", "--out", str(out)]) == 0
    rows = list(csv.DictReader(open(out / "requests.csv")))
    assert len(rows) == 18 and {row["act.batch_size"] for row in rows} == {"1"}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["requests"] == 12 and summary["throughput_rps"] > 0
    names = {"admit_ms", "inputs_ms", "act.delay_ms", "act.run_ms", "complete_ms", "total_ms"}
    assert names == set(summary["segments"])
    assert {"act.queue_ms", "act.wait_ms"} <= set(rows[0])
    for row in rows:
        assert sum(float(row[name]) for name in names - {"total_ms"}) == pytest.approx(float(row["total_ms"]))
    assert (out / "summary.md").read_text().startswith("# Requests of echo_model:EchoModel under load")


def test_through_serve_each_request_joins_its_client_time_frontend_and_engine_segments(tmp_path, monkeypatch):
    here = Path(__file__).resolve().parent
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(here.parents[1]), str(here)]))
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps([{"x": [1.0, 2.0, 3.0]}]))
    out = tmp_path / "out"
    assert profile.main(["load", "--serve", "--model", "echo_model:EchoModel", "--checkpoint", "unused",
                         "--requests", str(requests), "--device", "cpu", "--robots", "4", "--requests-per-robot", "5",
                         "--warmup-requests", "1", "--period-ms", "5", "--model-options", '{"scale": 3.0}',
                         "--serve-args", "--frontends 2 --shm-buffer-mib 1", "--out", str(out)]) == 0
    rows = list(csv.DictReader(open(out / "requests.csv")))
    assert len(rows) == 20 and {row["act.batch_size"] for row in rows} == {"1"}
    parts = ("network_ms", "prepare_ms", "handoff_ms", "inputs_ms", "act.delay_ms", "act.run_ms", "output_ms")
    for row in rows:
        assert sum(float(row[name]) for name in parts) == pytest.approx(float(row["total_ms"]))
        assert min(float(row[name]) for name in parts) >= 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["requests"] == 16 and "prepare_ms" in summary["segments"]
    assert (out / "engine" / "engine-requests.json").exists()


def test_a_robot_whose_requests_and_engine_requests_differ_is_reported():
    rows = [dict(robot=0, index=0, sent=0.0, received=0.01, status="ok",
                 server_timing=dict(frontend=0, infer_ms=9.0, prepare_ms=1.0, engine_ms=7.0, output_ms=1.0))]
    with pytest.raises(ValueError, match="robot 0: 1 requests, 0 engine requests"):
        load_profile.serve_segments(rows, {}, [("act", ())])

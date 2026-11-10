"""Per-request segment times of a model under closed-loop robot load (``xinfer-profile load``).

Virtual robots submit prepared requests to an ``Engine`` built as ``xinfer-serve`` builds it, at most one in flight
each: a robot sends its next request ``period`` after the previous one, or when the response arrives if that is
later (the repository's load rule, docs/benchmark.md). ``SpanRecorder``, a span sink
(``xinfer.observability.annotate``), keeps the engine's ``admit``, ``stage::`` and ``wait`` spans with their host
times and, on CUDA, timing events recorded on the lane's stream at their ends; the events are read after the load,
so nothing synchronizes while it runs. A dispatch's first event runs on its free lane at about the dispatch's host
time, which places the GPU times on the host clock.
Along a chain of stages a request's segments add up to its total time, except the
stages' queue and wait times (``OVERLAPPING``), which overlap the upstream stages' run.

In process (``run_load``, ``segments``) there are no frontend processes: the requests are prepared once, before the
load, and the outputs are not processed. Through ``xinfer-serve`` (``xinfer-profile load --serve``,
``serve_segments``) the robots are WebSocket clients sending the requests as robots do; the service, started with
``--profile-dir``, records the same spans in its engine process and writes them at shutdown
(``write_engine_requests``), and each request's client time, ``server_timing`` (frontend processing) and engine
spans are joined by connection.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch


class SpanRecorder:
    """Span sink keeping the engine's ``admit``, ``stage::`` and ``wait`` spans; other spans are ignored.

    Attributes:
        admits: ``{request id: (robot, start, end)}``, host times of the upload at admission.
        dispatches: one dict per stage dispatch: ``stage``, ``lane``, ``batch_size``, ``requests`` (ids), host
            ``start`` and ``end``, and on CUDA the events ``began`` (before the lane's waits), ``waited`` (after
            them) and ``ended`` (after the stage and its handoff copies); None on the CPU.
    """

    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.admits: dict[int, tuple] = {}
        self.dispatches: list[dict] = []
        self._current = threading.local()

    def __call__(self, name: str, args) -> contextlib.AbstractContextManager:
        if name == "admit":
            return self._admit(args)
        if name.startswith("stage::"):
            return self._stage(name.removeprefix("stage::"), args)
        if name == "wait":
            return self._wait()
        return contextlib.nullcontext()

    def _event(self):
        """A timing event on the current stream (the lane's, inside a dispatch), or None on the CPU."""
        if not self.cuda:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @contextlib.contextmanager
    def _admit(self, args):
        start = time.perf_counter()
        yield
        self.admits[args["request"]] = (args["robot"], start, time.perf_counter())

    @contextlib.contextmanager
    def _stage(self, stage, args):
        dispatch = dict(stage=stage, lane=args["lane"], batch_size=args["batch_size"],
                        requests=[int(i) for i in args["requests"].split(",")], start=time.perf_counter(),
                        began=self._event(), waited=None)
        self._current.dispatch = dispatch
        yield
        dispatch.update(ended=self._event(), end=time.perf_counter())
        self.dispatches.append(dispatch)

    @contextlib.contextmanager
    def _wait(self):
        yield
        self._current.dispatch["waited"] = self._event()


def run_load(engine, requests: list, robots: int, per_robot: int, period_s: float, start: str = "spread") -> dict:
    """Run the virtual robots against ``engine`` until each has its ``per_robot`` responses.

    Robot ``r``'s ``i``-th request is ``requests[(r + i) % len(requests)]``, submitted with robot id ``"r:i"``
    (one in flight per robot, so the engine's per-robot limit does not apply). With ``start="spread"`` robot ``r``
    sends its first request ``r / robots`` of a period after the first robot, as independent robots would; with
    ``"together"`` all send it at once, and then stay in step while the responses come within the period.

    Returns:
        ``{"r:i": (submitted, done)}``: host times of the submission and of the result (``time.perf_counter()``).
    """
    times, lock = {}, threading.Lock()
    began = time.perf_counter()

    def robot(r):
        next_at = began + (r * period_s / robots if start == "spread" else 0.0)
        for i in range(per_robot):
            time.sleep(max(next_at - time.perf_counter(), 0.0))
            key, submitted = f"{r}:{i}", time.perf_counter()
            future = engine.submit(requests[(r + i) % len(requests)], robot=key)
            future.add_done_callback(lambda _, key=key, submitted=submitted: _done(key, submitted))
            future.result()
            next_at = max(submitted + period_s, time.perf_counter())

    def _done(key, submitted):
        with lock:
            times[key] = (submitted, time.perf_counter())

    threads = [threading.Thread(target=robot, args=(r,), name=f"robot-{r}") for r in range(robots)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return times


def engine_requests(recorder: SpanRecorder) -> dict:
    """Each recorded request's engine side, once every event has completed (after the load, or the engine's
    shutdown).

    Returns:
        ``{request id: {"robot", "admitted", "uploaded", "stages": {stage: {"start", "end", "wait_ms", "run_ms",
        "done", "batch_size", "lane"}}}}``: host times in ``time.perf_counter()`` seconds (``start`` and ``end`` of
        the dispatch on the host, ``done`` when the stage finished on the GPU), GPU times in ms. On the CPU
        ``wait_ms`` is 0 and ``run_ms`` the host dispatch time.
    """
    requests = {request: dict(robot=robot, admitted=start, uploaded=end, stages={})
                for request, (robot, start, end) in recorder.admits.items()}
    for d in recorder.dispatches:
        if d["began"] is None:
            wait, run, done = 0.0, (d["end"] - d["start"]) * 1000, d["end"]
        else:
            wait, run = d["began"].elapsed_time(d["waited"]), d["waited"].elapsed_time(d["ended"])
            done = d["start"] + d["began"].elapsed_time(d["ended"]) / 1000
        for request in d["requests"]:
            if request in requests:
                requests[request]["stages"][d["stage"]] = dict(start=d["start"], end=d["end"], wait_ms=wait,
                                                               run_ms=run, done=done, batch_size=d["batch_size"],
                                                               lane=d["lane"])
    return requests


def write_engine_requests(recorder: SpanRecorder, stage_graph, directory) -> Path:
    """Write the stages (``[[name, depends_on], ...]``) and ``engine_requests`` to
    ``<directory>/engine-requests.json`` (``xinfer-serve --profile-dir``, at shutdown)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "engine-requests.json"
    stages = [[stage.name, list(stage.depends_on)] for stage in stage_graph]
    path.write_text(json.dumps({"stages": stages, "requests": engine_requests(recorder)}))
    return path


# stage_segments' columns that overlap the upstream stages' run: in the rows, but not summed up or summarized.
OVERLAPPING = (".queue_ms", ".wait_ms")


def stage_segments(request: dict, stages: list) -> dict:
    """One engine request's ``inputs_ms`` (the upload) and per stage ``<stage>.delay_ms`` (from its inputs being ready,
    at the upload's end for a first stage, else when its upstream stages are done on the GPU, to its start on the
    GPU), ``run_ms`` and ``batch_size``. Along a chain of stages these add up to the request's engine time.

    Also per stage, overlapping the upstream run (``OVERLAPPING``): ``<stage>.queue_ms`` (on the host, from the upload's
    or the upstream dispatches' end to its dispatch, waiting for a free lane) and ``wait_ms`` (its lane waiting on the
    GPU for the inputs and upstream stages).

    Args:
        request: an ``engine_requests`` value.
        stages: ``[(name, depends_on), ...]`` in topological order.
    """
    dispatched = request["stages"]
    row = {"inputs_ms": (request["uploaded"] - request["admitted"]) * 1000}
    for name, depends_on in stages:
        d = dispatched[name]
        ready = max((dispatched[dep]["done"] for dep in depends_on), default=request["uploaded"])
        entered = max((dispatched[dep]["end"] for dep in depends_on), default=request["uploaded"])
        row.update({f"{name}.delay_ms": (d["start"] - ready) * 1000 + d["wait_ms"], f"{name}.run_ms": d["run_ms"],
                    f"{name}.batch_size": d["batch_size"], f"{name}.queue_ms": (d["start"] - entered) * 1000,
                    f"{name}.wait_ms": d["wait_ms"]})
    return row


def segments(recorder: SpanRecorder, times: dict, stages: list) -> list[dict]:
    """Each completed request's segment times in ms in process, from the recorded spans and ``run_load``'s times.

    Per request: ``robot``, ``index``, ``admit_ms`` (submission to admission: waiting for a slot), the
    ``stage_segments``, ``complete_ms`` (the final stage done on the GPU to the result: copy to the host and
    completion thread) and ``total_ms`` (submission to result).
    """
    by_robot = {request["robot"]: request for request in engine_requests(recorder).values()}
    rows = []
    for key, (submitted, finished) in times.items():
        request = by_robot[key]
        robot, index = key.split(":")
        rows.append(dict(robot=int(robot), index=int(index), admit_ms=(request["admitted"] - submitted) * 1000,
                         **stage_segments(request, stages),
                         complete_ms=(finished - max(s["done"] for s in request["stages"].values())) * 1000,
                         total_ms=(finished - submitted) * 1000))
    return sorted(rows, key=lambda row: (row["robot"], row["index"]))


def serve_segments(rows: list[dict], engine: dict, stages: list) -> list[dict]:
    """Each request's segment times in ms through ``xinfer-serve``: the clients' rows joined with the engine's.

    A robot's engine robot id is ``(frontend, connection number)``: the frontend from its responses' ``server_timing``,
    the number its rank among the robots on that frontend, which connected in robot order (the clients of
    ``xinfer-profile load --serve``). With one request in flight
    per connection, its engine requests in admission order are its requests in order.

    Per request: ``robot``, ``index``, ``network_ms`` (the client's time minus the service's ``infer_ms``: network
    and message encoding), ``prepare_ms`` and ``output_ms`` (the frontend's input and output processing,
    ``server_timing``), ``handoff_ms`` (``server_timing``'s ``engine_ms`` minus the engine's span from admission to
    the final stage done: shared-memory buffer, pipes, waiting for a slot, copy to the host, completion), the
    ``stage_segments`` and ``total_ms`` (sent to received, on the client).

    Raises:
        ValueError: a robot's request count differs from its connection's engine requests.
    """
    by_robot: dict[int, list] = {}
    for row in sorted(rows, key=lambda row: (row["robot"], row["index"])):
        by_robot.setdefault(row["robot"], []).append(row)
    numbers: dict[int, int] = {}
    connection = {}
    for r, robot_rows in sorted(by_robot.items()):
        frontend = robot_rows[0]["server_timing"]["frontend"]
        connection[r] = (frontend, numbers.get(frontend, 0))
        numbers[frontend] = numbers.get(frontend, 0) + 1
    by_connection: dict[tuple, list] = {}
    for request in sorted(engine.values(), key=lambda request: request["admitted"]):
        if request["robot"] is not None:
            by_connection.setdefault(tuple(request["robot"]), []).append(request)
    out = []
    for r, robot_rows in by_robot.items():
        requests = by_connection.get(connection[r], [])
        if len(requests) != len(robot_rows):
            raise ValueError(f"robot {r}: {len(robot_rows)} requests, {len(requests)} engine requests on its "
                             f"connection {connection[r]}")
        for row, request in zip(robot_rows, requests):
            timing, total = row["server_timing"], (row["received"] - row["sent"]) * 1000
            span = (max(s["done"] for s in request["stages"].values()) - request["admitted"]) * 1000
            out.append(dict(robot=r, index=row["index"], network_ms=total - timing["infer_ms"],
                            prepare_ms=timing["prepare_ms"], handoff_ms=timing["engine_ms"] - span,
                            **stage_segments(request, stages), output_ms=timing["output_ms"], total_ms=total))
    return out


def summarize(rows: list[dict]) -> dict:
    """Percentiles of the segments that add up to ``total_ms``, batch sizes per stage, and where the tail's time goes.

    Returns:
        ``segments`` (``{name: {p50, p95, p99}}``), ``batch_sizes`` (``{stage: {size: requests}}``) and ``tail``:
        for the requests at or above P95 and P99 of ``total_ms``, each segment's mean minus its mean over the median
        band (P45-P55), largest first.
    """
    names = [k for k in rows[0] if k.endswith("_ms") and not k.endswith(OVERLAPPING)]
    values = {name: np.array([row[name] for row in rows]) for name in names}
    summary = {"requests": len(rows),
               "segments": {name: {f"p{q}": float(np.percentile(v, q)) for q in (50, 95, 99)}
                            for name, v in values.items()}}
    batch_sizes = {}
    for row in rows:
        for key, size in row.items():
            if key.endswith(".batch_size"):
                counts = batch_sizes.setdefault(key.removesuffix(".batch_size"), {})
                counts[size] = counts.get(size, 0) + 1
    summary["batch_sizes"] = batch_sizes
    total = values["total_ms"]
    p45, p55 = np.percentile(total, [45, 55])
    band = (total >= p45) & (total <= p55)
    if not band.any():  # few requests: the one nearest the median
        distance = np.abs(total - np.median(total))
        band = distance == distance.min()
    summary["tail"] = {}
    for q in (95, 99):
        tail = total >= np.percentile(total, q)
        excess = {name: float(v[tail].mean() - v[band].mean()) for name, v in values.items()}
        summary["tail"][f"p{q}"] = dict(sorted(excess.items(), key=lambda item: -item[1]))
    return summary

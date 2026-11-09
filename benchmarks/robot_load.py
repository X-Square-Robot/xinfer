"""Load a running WebSocket + msgpack VLA service with virtual robots and report latency per round.

Each robot is a separate process with one connection and at most one request in flight, in closed loop
(the repository's load convention): each robot sends its first request at a random delay within one request
period (180 ms; uniform, seeded by ``--start-seed`` and the round) after a start time shared by the round's
robots, each later one when the robot's action queue calls for it (``virtual_robot.py``), or when the previous
response arrives if that is later. While responses are faster than the period, the robots keep their start phases
for the whole round, so each round measures one random arrangement of the robots' send times, the same one for
the same seed.

Requests are synthetic Wall-OSS observations carrying the RTC fields computed from the previous response, or,
with ``--requests-file``, the given requests in turn without RTC fields (for models without RTC). The
first ``--warmup-requests`` of each robot are excluded from the statistics, not from the error
count. Each round uses new connections, robots and start delays. A response counts as successful when it has no
``error`` and its ``status``, if present, is ``"ok"``; a Wall-OSS response must also carry absolute poses. See
docs/benchmark.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import msgpack
import msgpack_numpy
import numpy as np
from websockets.sync.client import connect

START_MARGIN_S = 2.0  # from launching a round's robots to their shared start time
FIELDS = ("round", "robot", "sequence", "warmup", "ok", "error", "start_s", "end_s", "client_ms", "server_infer_ms",
          "server_timing", "schedule_lag_ms", "payload_ms", "sent_inference_delay", "sent_s_model", "delay_model",
          "elapsed_control_steps", "queue_coverage_ms", "underflow_ms")


def packb(value) -> bytes:
    return msgpack.packb(value, default=msgpack_numpy.encode, use_bin_type=True)


def unpackb(data: bytes):
    return msgpack.unpackb(data, object_hook=msgpack_numpy.decode, raw=False)


def load_requests(paths) -> list[dict]:
    """Requests from JSON (``.json``) or msgpack files that each hold one request map or a list of them."""
    requests = []
    for path in paths:
        data = Path(path).read_bytes()
        value = json.loads(data) if str(path).endswith(".json") else unpackb(data)
        requests += value if isinstance(value, list) else [value]
    return requests


def succeeded(response) -> bool:
    return isinstance(response, dict) and "error" not in response and response.get("status", "ok") == "ok"


def exchange(websocket, request: dict, timeout_s: float) -> tuple[dict, float, float]:
    """Send one request; return the decoded response, the send time and the receive time (perf_counter)."""
    sent = time.perf_counter()
    websocket.send(packb(request))
    raw = websocket.recv(timeout=timeout_s)
    received = time.perf_counter()
    return unpackb(raw), sent, received


def result_fields(response, sent: float, received: float) -> dict:
    timing = response.get("server_timing", {}) if isinstance(response, dict) else {}
    return {"ok": succeeded(response), "error": "" if succeeded(response) else str(response.get("error", response)),
            "start_s": sent, "end_s": received, "client_ms": (received - sent) * 1000,
            "server_infer_ms": timing.get("infer_ms"), "server_timing": json.dumps(timing, sort_keys=True)}


def flow_steps(metadata) -> int:
    """The flow steps an RTC request asks for: the service's (``flow_steps``, or ``rtc.num_flow_steps``), else the
    Wall-OSS default."""
    from virtual_robot import NUM_FLOW_STEPS

    metadata = metadata if isinstance(metadata, dict) else {}
    rtc = metadata.get("rtc") if isinstance(metadata.get("rtc"), dict) else {}
    return int(metadata.get("flow_steps") or rtc.get("num_flow_steps") or NUM_FLOW_STEPS)


def run_robot(task: dict) -> dict:
    """One robot in closed loop; returns its rows and the service's metadata frame."""
    from virtual_robot import Observations, VirtualRobot, actions

    frozen = load_requests(task["requests_file"]) if task["requests_file"] else None
    observations = Observations(f"robot-{task['robot']}")
    robot = VirtualRobot(task["max_inference_delay"])
    rows = []
    with connect(task["url"], max_size=None, compression=None, open_timeout=task["timeout_s"]) as websocket:
        metadata = unpackb(websocket.recv(timeout=task["timeout_s"]))
        steps = flow_steps(metadata)
        time.sleep(max(0.0, task["start_at"] - time.time()))
        start = time.perf_counter()
        for sequence in range(task["requests"]):
            # The first request goes out at once; later ones follow the action queue.
            target = 0.0 if robot.last_response_s is None else max(robot.last_response_s, robot.next_trigger_s or 0.0)
            time.sleep(max(0.0, start + target - time.perf_counter()))
            captured = time.perf_counter()
            fields = robot.rtc_fields(captured - start)
            if frozen is None:
                request = observations.make(fields, steps)
            else:
                request = frozen[(task["robot"] + sequence) % len(frozen)]
            row = {"robot": task["robot"], "sequence": sequence, "warmup": sequence < task["warmup_requests"],
                   "schedule_lag_ms": (captured - start - target) * 1000,
                   "sent_inference_delay": fields["inference_delay"] if fields else None,
                   "sent_s_model": fields["s_model"] if fields else None}
            try:
                response, sent, received = exchange(websocket, request, task["timeout_s"])
            except Exception as exc:  # noqa: BLE001 - without request ids the connection cannot be reused
                rows.append({**row, "ok": False, "error": repr(exc)})
                break
            row.update(result_fields(response, sent, received), payload_ms=(sent - captured) * 1000)
            poses = actions(response) if row["ok"] and frozen is None else None
            if row["ok"] and frozen is None and poses is None:
                row.update(ok=False, error="response has no [T, 14] absolute poses")
            if row["ok"]:
                row.update(robot.on_response(received - sent, received - start, poses))
            else:  # the queue keeps the previous chunk; the next request is due now
                robot.last_response_s = received - start
            rows.append(row)
    return {"rows": rows, "metadata": metadata}


def run_task(task: dict) -> dict:
    """Run one robot in a worker process; a failure to connect becomes an error with no rows."""
    try:
        return run_robot(task)
    except Exception as exc:  # noqa: BLE001
        return {"rows": [], "metadata": None, "error": repr(exc)}


def start_delays(robots: int, seed: int, round_index: int) -> list[float]:
    """Each robot's delay before its first request in a round, in seconds: uniform in one request period, from a
    generator seeded by ``(seed, round_index)``."""
    from virtual_robot import REQUEST_PERIOD_S

    return np.random.default_rng([seed, round_index]).uniform(0.0, REQUEST_PERIOD_S, robots).tolist()


def percentiles(values) -> dict:
    values = [value for value in values if value is not None]
    return {f"p{q}": float(np.percentile(values, q)) for q in (50, 95, 99)} if values else {}


def summarize(rows: list[dict], results: list[dict], expected: int) -> dict:
    """One round's statistics over the steady (non-warmup) rows."""
    steady = [row for row in rows if not row["warmup"]]
    good = [row for row in steady if row["ok"]]
    window = (max(row["end_s"] for row in good) - min(row["start_s"] for row in good)) if good else 0.0
    summary = {"rows": len(rows), "missing": expected - len(rows),
               "errors": sum(not row["ok"] for row in rows) + sum("error" in result for result in results),
               "steady": len(steady), "server_infer_ms": percentiles(row["server_infer_ms"] for row in good),
               "client_ms": percentiles(row["client_ms"] for row in good),
               "throughput_rps": len(good) / window if window else None}
    summary["sent_inference_delay"] = dict(sorted(Counter(row["sent_inference_delay"] for row in steady
                                                          if row.get("sent_inference_delay")).items()))
    summary["underflow_ms"] = sum(row.get("underflow_ms") or 0.0 for row in steady)
    return summary


def describe(index: int, summary: dict) -> str:
    def triple(values: dict) -> str:
        return "/".join(f"{values.get(f'p{q}', float('nan')):.1f}" for q in (50, 95, 99))

    throughput = summary["throughput_rps"] or 0.0
    return (f"round {index}: {summary['rows']} rows, {summary['errors']} errors, {summary['missing']} missing; "
            f"P50/P95/P99 server {triple(summary['server_infer_ms'])} ms, client {triple(summary['client_ms'])} ms; "
            f"{throughput:.1f} req/s; underflow {summary['underflow_ms']:.0f} ms")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="service endpoint, e.g. ws://127.0.0.1:33985")
    parser.add_argument("--robots", type=int, default=1)
    parser.add_argument("--requests-file", nargs="+",
                        help="send these requests (JSON or msgpack files with one request or a list) instead of "
                             "synthetic Wall-OSS observations, without RTC fields")
    parser.add_argument("--requests", type=int, default=80, help="requests per robot and round, warmup included")
    parser.add_argument("--warmup-requests", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--start-seed", type=int, default=0, help="seed of the robots' start delays")
    parser.add_argument("--max-inference-delay", type=int, default=6,
                        help="largest delay estimate in model steps (sent as 3..max+2)")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="connection and response timeout")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args(argv)
    if min(args.robots, args.requests, args.rounds, args.max_inference_delay) < 1:
        parser.error("robots, requests, rounds and max-inference-delay must be positive")
    if not 0 <= args.warmup_requests < args.requests:
        parser.error("require 0 <= warmup-requests < requests")
    if args.out.exists():
        parser.error(f"{args.out} already exists")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    frozen = [Path(path) for path in args.requests_file or []]
    if frozen:  # fail before any connection on an unreadable input
        load_requests(frozen)
    args.out.mkdir(parents=True)
    try:
        revision = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip() or None
    except OSError:  # no git on this machine
        revision = None
    run = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
           "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
           "revision": revision,
           "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in frozen]}
    tasks = [dict(url=args.url, robot=robot, requests=args.requests, warmup_requests=args.warmup_requests,
                  timeout_s=args.timeout_s, max_inference_delay=args.max_inference_delay,
                  requests_file=[str(path) for path in frozen])
             for robot in range(args.robots)]
    rounds = []
    context = multiprocessing.get_context("spawn")
    for index in range(1, args.rounds + 1):
        delays = start_delays(args.robots, args.start_seed, index)
        start_at = time.time() + START_MARGIN_S  # after every robot has connected
        for task, delay in zip(tasks, delays):
            task["start_at"] = start_at + delay
        with context.Pool(args.robots) as pool:
            results = pool.map(run_task, tasks)
        rows = [{"round": index, **row} for result in results for row in result["rows"]]
        with (args.out / f"round_{index}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summary = summarize(rows, results, args.robots * args.requests)
        summary["robot_errors"] = [result["error"] for result in results if "error" in result]
        summary["start_delays_ms"] = [round(delay * 1000, 1) for delay in delays]
        rounds.append({"round": index, **summary})
        run.setdefault("metadata", next((result["metadata"] for result in results if result["metadata"]), None))
        print(describe(index, summary), flush=True)
    (args.out / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps({"rounds": rounds}, indent=2), encoding="utf-8")
    return 0 if all(item["errors"] == item["missing"] == 0 for item in rounds) else 1


if __name__ == "__main__":
    sys.exit(main())

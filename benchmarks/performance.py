"""One row of docs/performance.md: single-request latency and single-GPU capacity of one serving configuration.

Runs on the visible GPU (select it with ``CUDA_VISIBLE_DEVICES``), with nothing else on it:

1. ``request_latency.py`` on the configuration's request set, with its model options; ``Policy`` decodes and resizes
   the views on the device unless they set ``image_preprocess_device``;
2. for each robot count, in increasing order, a newly started ``xinfer-serve`` with the same model options (the views
   processed in the frontends unless they set ``image_preprocess_device``) and the warmup files, then
   ``robot_load.py`` with the repository convention (80 requests per robot, the first 20 dropped, 8 rounds, each with
   random robot start phases): synthetic desktop observations with RTC, or ``--load-requests`` without RTC. It stops
   after the first count that fails.

A count passes when no round has an error or missing row and the median over its rounds of the ``server_infer_ms``
P50 is below 150 ms; the capacity is the largest count that passes with every smaller measured count passing too.
Writes ``OUT/row.json`` (the numbers, the median and worst round of each count, the configuration and the
environment) and prints the table row. See docs/benchmark.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
LIMIT_MS = 150.0
ROUNDS = 8


def round_statistics(rounds: list[dict]) -> dict:
    """The median and the worst over the rounds of each round's server P50 and P95 (``robot_load.py`` summaries)."""
    p50 = [r["server_infer_ms"].get("p50", float("inf")) for r in rounds]
    p95 = [r["server_infer_ms"].get("p95", float("inf")) for r in rounds]
    return {"p50_median": float(median(p50)), "p95_median": float(median(p95)), "p50_worst": max(p50),
            "p95_worst": max(p95)}


def passes(rounds: list[dict]) -> bool:
    """Whether a robot count meets the capacity rule (``robot_load.py`` round summaries)."""
    return (bool(rounds) and all(r["errors"] == r["missing"] == 0 for r in rounds)
            and round_statistics(rounds)["p50_median"] < LIMIT_MS)


def capacity(points: dict[int, list[dict]]) -> tuple[int | None, bool]:
    """``(robots, bounded)``: the largest measured count that passes with every smaller measured count passing, or
    None; ``bounded`` when a larger count was measured and failed."""
    best = None
    for robots in sorted(points):
        if not passes(points[robots]):
            return best, True
        best = robots
    return best, False


def row(result: dict) -> str:
    """The table row: model, configuration, hardware, single-request P50, capacity, the median over its rounds of the
    server P50 and P95, the saturation throughput."""
    robots, bounded = result["capacity"]
    if robots is None:
        cap, p50, p95 = f"< {min(result['load'])}", "—", "—"
    else:
        medians = round_statistics(result["load"][str(robots)])
        cap = f"{robots}" if bounded else f"≥ {robots}"
        p50, p95 = f"{medians['p50_median']:.0f}", f"{medians['p95_median']:.0f}"
    throughput = max((r["throughput_rps"] or 0.0) for rounds in result["load"].values() for r in rounds)
    return (f"| {result['name']} | {result['latency']['total_ms']['p50']:.1f} ms | {cap} | {p50} ms | {p95} ms | "
            f"{throughput:.1f} req/s |")


def environment() -> dict:
    """Where the row was measured (the GPU name comes from ``request_latency.py``; this process stays off the GPU)."""
    import torch

    library = os.environ.get("XINFER_XCOMPUTE_LIBRARY")
    revision = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip() if (HERE.parent / ".git").exists() else None
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(), "revision": revision,
            "torch": torch.__version__, "xcompute_library": library,
            "xcompute_sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest() if library else None}


def gpu_processes() -> str:
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def serve_and_load(args, robots: int, out: Path) -> list[dict]:
    """A new service, ``robot_load.py`` with ``robots`` robots against it, the service stopped; the round summaries."""
    out.mkdir(parents=True)
    port = args.port + robots
    command = [sys.executable, "-m", "xinfer.entrypoints.serve", "--model", args.model, "--checkpoint",
               args.checkpoint, "--model-options", args.model_options, "--port", str(port),
               "--resolved-config", str(out / "resolved.json"), "--warmup", *args.warmup]
    log_path = out / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while "XINFER READY" not in log_path.read_text(errors="replace"):
                if server.poll() is not None:
                    raise RuntimeError(f"xinfer-serve exited before it was ready; see {log_path}")
                time.sleep(1)
            before = {"gpu_processes": gpu_processes(), "loadavg": os.getloadavg()}
            load = [sys.executable, str(HERE / "robot_load.py"), "--url", f"ws://127.0.0.1:{port}", "--robots",
                    str(robots), "--requests", "80", "--warmup-requests", "20", "--rounds", str(ROUNDS), "--out",
                    str(out / "load")]
            if args.load_requests:
                load += ["--requests-file", *args.load_requests]
            subprocess.run(load, check=False)
            (out / "machine.json").write_text(json.dumps(
                {"before": before, "after": {"gpu_processes": gpu_processes(), "loadavg": os.getloadavg()}}))
        finally:
            server.send_signal(signal.SIGINT)
            server.wait(timeout=120)
    summary = out / "load" / "summary.json"
    return json.loads(summary.read_text())["rounds"] if summary.exists() else []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="row label, e.g. 'Wall-OSS · BF16 optimized · RTX 5090'")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-options", default="{}", help="JSON options for xinfer-serve and Policy")
    parser.add_argument("--latency-requests", nargs="+", required=True, help="the single-request request set")
    parser.add_argument("--warmup", nargs="+", required=True, help="xinfer-serve --warmup files")
    parser.add_argument("--load-requests", nargs="+", help="load without RTC from these requests (robot_load.py)")
    parser.add_argument("--robots", type=int, nargs="+", required=True, help="robot counts to measure")
    parser.add_argument("--port", type=int, default=34000)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} already exists")
    args.out.mkdir(parents=True)
    result = {"name": args.name, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "environment": environment(), "latency": {}, "load": {}}
    target = args.out / "latency.json"
    subprocess.run([sys.executable, str(HERE / "request_latency.py"), "--model", args.model, "--checkpoint",
                    args.checkpoint, "--model-options", args.model_options, "--requests-file", *args.latency_requests,
                    "--out", str(target)], check=True)
    latency = json.loads(target.read_text())
    result["latency"] = {k: latency[k] for k in ("total_ms", "prepare_ms", "engine_ms", "output_ms",
                                                 "captures_after_warmup")}
    result["environment"]["gpu"] = latency["device"]
    for robots in sorted(args.robots):
        rounds = serve_and_load(args, robots, args.out / f"robots-{robots}")
        result["load"][str(robots)] = rounds
        if not passes(rounds):
            break
        time.sleep(5)  # let the previous service release the GPU
    result["capacity"] = capacity({int(k): v for k, v in result["load"].items()})
    result["load_statistics"] = {k: round_statistics(v) for k, v in result["load"].items() if v}
    (args.out / "row.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(row(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""One row of docs/performance.md: single-request latency and single-device capacity of one configuration.

Runs on one accelerator, with nothing else on it. Use ``--device npu:N`` for Ascend or the normal CUDA selection:

1. ``request_latency.py`` on the configuration's request set, with its model options; ``Policy`` decodes and resizes
   the views on the device where supported unless they set ``image_preprocess_device``;
2. for each robot count, in increasing order, a newly started ``xinfer-serve`` with the configuration's model options
   and warmup files, then ``robot_load.py`` with the repository convention (80 requests per robot, the first 20
   dropped, 8 rounds, each with random robot start phases): synthetic desktop observations with RTC, or
   ``--load-requests`` without RTC. It stops after the first count that fails.

The capacity is the knee of the latency curve: the largest count before the first one past the knee, a count with an
error or missing row in a round, or whose median over its rounds of the ``server_infer_ms`` P50 is more than 50 %
above that of the previous measured count. The first count is the reference, so the counts should start below the
knee.
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
KNEE = 1.5
ROUNDS = 8


def round_statistics(rounds: list[dict]) -> dict:
    """The median and the worst over the rounds of each round's server P50 and P95 (``robot_load.py`` summaries)."""
    p50 = [r["server_infer_ms"].get("p50", float("inf")) for r in rounds]
    p95 = [r["server_infer_ms"].get("p95", float("inf")) for r in rounds]
    return {"p50_median": float(median(p50)), "p95_median": float(median(p95)), "p50_worst": max(p50),
            "p95_worst": max(p95)}


def past_knee(rounds: list[dict], previous: list[dict] | None) -> bool:
    """Whether a robot count is past the knee (``robot_load.py`` round summaries; ``previous``: those of the previous
    measured count, None for the first)."""
    if not rounds or any(r["errors"] or r["missing"] for r in rounds):
        return True
    return (previous is not None
            and round_statistics(rounds)["p50_median"] > KNEE * round_statistics(previous)["p50_median"])


def capacity(points: dict[int, list[dict]]) -> tuple[int | None, bool]:
    """``(robots, bounded)``: the largest measured count before the first one past the knee, or None; ``bounded``
    when a count past the knee was measured."""
    best, previous = None, None
    for robots in sorted(points):
        if past_knee(points[robots], previous):
            return best, True
        best, previous = robots, points[robots]
    return best, False


def row(result: dict) -> str:
    """The table row: model, configuration, hardware, single-request P50 with its CPU part (the P50s of the input and
    output processing, ``prepare_ms`` + ``output_ms``) and device part (the P50 of ``engine_ms``) in parentheses,
    capacity, the median over its rounds of the server P50 and P95, the saturation throughput."""
    robots, bounded = result["capacity"]
    if robots is None:
        cap, p50, p95 = f"< {min(map(int, result['load']))}", "—", "—"
    else:
        medians = round_statistics(result["load"][str(robots)])
        cap = f"{robots}" if bounded else f"≥ {robots}"
        p50, p95 = f"{medians['p50_median']:.0f}", f"{medians['p95_median']:.0f}"
    throughput = max((r["throughput_rps"] or 0.0 for rounds in result["load"].values() for r in rounds), default=0.0)
    latency = result["latency"]
    cpu = latency["prepare_ms"]["p50"] + latency["output_ms"]["p50"]
    return (f"| {result['name']} | {latency['total_ms']['p50']:.1f} ms ({cpu:.1f} + {latency['engine_ms']['p50']:.1f}) "
            f"| {cap} | {p50} ms | {p95} ms | {throughput:.1f} req/s |")


def environment(npu: bool = False) -> dict:
    """Where the row was measured (device name comes from ``request_latency.py``)."""
    import torch

    if npu:  # the external library the Ascend backend binds
        library = os.environ.get("XINFER_XCOMPUTE_LIBRARY")
    else:  # the kernel library built from this repository
        from xinfer.ops.backends.cuda import _native

        library = _native.LIBRARY_PATH
    revision = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip() if (HERE.parent / ".git").exists() else None
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(), "revision": revision,
              "torch": torch.__version__, "kernel_library": library,
              "kernel_library_sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest() if library else None}
    if npu:
        from importlib.metadata import version

        result["torch_npu"] = version("torch-npu")
    return result


def gpu_processes() -> str:
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def machine_snapshot(npu: bool) -> dict:
    """Record host load and the matching accelerator telemetry."""
    if npu:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10)
        return {"npu_smi": result.stdout.strip(), "loadavg": os.getloadavg()}
    return {"gpu_processes": gpu_processes(), "loadavg": os.getloadavg()}


def serve_and_load(args, robots: int, out: Path) -> list[dict]:
    """A new service, ``robot_load.py`` with ``robots`` robots against it, the service stopped; the round summaries."""
    out.mkdir(parents=True)
    port = args.port + robots
    command = [sys.executable, "-m", "xinfer.entrypoints.serve", "--model", args.model, "--checkpoint",
               args.checkpoint, "--model-options", args.model_options, "--port", str(port),
               "--resolved-config", str(out / "resolved.json"), "--warmup", *args.warmup]
    if args.op_backends:
        command += ["--op-backends", args.op_backends]
    if args.device is not None:
        command += ["--device", args.device]
    log_path = out / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while "XINFER READY" not in log_path.read_text(errors="replace"):
                if server.poll() is not None:
                    raise RuntimeError(f"xinfer-serve exited before it was ready; see {log_path}")
                time.sleep(1)
            before = machine_snapshot(args.npu)
            load = [sys.executable, str(HERE / "robot_load.py"), "--url", f"ws://127.0.0.1:{port}", "--robots",
                    str(robots), "--requests", "80", "--warmup-requests", "20", "--rounds", str(ROUNDS), "--out",
                    str(out / "load")]
            if args.load_requests:
                load += ["--requests-file", *args.load_requests]
            subprocess.run(load, check=False)
            (out / "machine.json").write_text(json.dumps({"before": before, "after": machine_snapshot(args.npu)}))
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
    parser.add_argument("--device", default=None, help="cuda, cuda:N, npu or npu:N (default: automatic CUDA selection)")
    parser.add_argument("--latency-requests", nargs="+", required=True, help="the single-request request set")
    parser.add_argument("--warmup", nargs="+", required=True, help="xinfer-serve --warmup files")
    parser.add_argument("--load-requests", nargs="+", help="load without RTC from these requests (robot_load.py)")
    parser.add_argument("--robots", type=int, nargs="+", required=True, help="robot counts to measure")
    parser.add_argument("--op-backends", default=None,
                        help="JSON operator name to backend pin, as for xinfer-serve --op-backends; also "
                             "applied to the single-request run so both halves use one wiring")
    parser.add_argument("--port", type=int, default=34000)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args(argv)
    device_type = args.device.partition(":")[0] if args.device is not None else "cuda"
    if device_type not in ("cuda", "npu"):
        parser.error("the performance table needs a CUDA or NPU device")
    args.npu = device_type == "npu"
    if args.out.exists():
        parser.error(f"{args.out} already exists")
    args.out.mkdir(parents=True)
    result = {"name": args.name, "capacity_rule": {"median_p50_ratio": KNEE},
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "environment": environment(args.npu), "latency": {}, "load": {}}
    target = args.out / "latency.json"
    command = [sys.executable, str(HERE / "request_latency.py"), "--model", args.model, "--checkpoint",
               args.checkpoint, "--model-options", args.model_options, "--requests-file", *args.latency_requests,
               "--out", str(target)]
    if args.op_backends:
        command += ["--op-backends", args.op_backends]
    if args.device is not None:
        command += ["--device", args.device]
    subprocess.run(command, check=True)
    latency = json.loads(target.read_text())
    result["latency"] = {k: latency[k] for k in ("total_ms", "prepare_ms", "engine_ms", "output_ms",
                                                 "captures_after_warmup")}
    result["environment"]["npu" if args.npu else "gpu"] = latency["device"]
    previous = None
    for robots in sorted(args.robots):
        rounds = serve_and_load(args, robots, args.out / f"robots-{robots}")
        result["load"][str(robots)] = rounds
        if past_knee(rounds, previous):
            break
        previous = rounds
        time.sleep(5)  # let the previous service release the GPU
    result["capacity"] = capacity({int(k): v for k, v in result["load"].items()})
    result["load_statistics"] = {k: round_statistics(v) for k, v in result["load"].items() if v}
    (args.out / "row.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(row(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

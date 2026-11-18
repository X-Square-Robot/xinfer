"""``xinfer-profile``: where a model's time goes (docs/profiling.md).

``xinfer-profile ops``: the GPU time, cost and roofline bound of every operator a model runs.

The model is built under a profiled op table, so every operator call runs in an ``op::`` span, and the engine marks
each stage dispatch with a ``stage::`` span. The command runs every request twice eagerly (warmup; the costs of the
call shapes are recorded), then ``--requests-per-run`` requests eagerly under torch.profiler with CPU and CUDA
activity, and attributes every GPU activity to its operator call and stage. Then the same requests run under CUDA
graph replay, the serving path, for each stage's GPU span. It writes ``ops.md``, ``ops.csv``, ``ops.json``,
``stages.md``, ``trace.json`` (the eager kineto trace), ``trace-graphs.json`` and ``meta.json`` to ``--out``.

``xinfer-profile load``: each request's segment times under closed-loop virtual robots
(``xinfer.observability.load_profile``): in the engine, waiting for a slot, and per stage the queue, the lane's wait
for the inputs and upstream stages on the GPU and the run, with the batch size. In process by default; with
``--serve`` through an ``xinfer-serve`` started for the run, adding the network, the frontend's input and output
processing and the handoff between the frontend and the engine. It writes ``requests.csv``, ``summary.md``,
``summary.json`` and ``meta.json`` (and with ``--serve`` ``serve.log`` and ``engine/``) to ``--out``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

BOUND_COLUMNS = ("bound_us_per_call", "efficiency", "limiter")
OP_COLUMNS = {"op": "Operator", "shape": "Call shape", "calls": "Calls", "us_per_call": "µs/call",
              "total_us": "Total µs", "share": "Share", "gflop_per_call": "GFLOP/call", "mb_per_call": "MB/call",
              "bound_us_per_call": "Bound µs/call", "efficiency": "Efficiency", "limiter": "Limiter", "cost": "Cost"}
DIGITS = {"us_per_call": 2, "bound_us_per_call": 2, "gflop_per_call": 3, "mb_per_call": 3}
STAGE_COLUMNS = {"stage": "Stage", "dispatches": "Dispatches", "host_us": "Host µs", "gpu_span_us": "GPU span µs",
                 "operators_us": "Operators µs", "outside_us": "Outside µs", "overlap_us": "Overlap µs",
                 "idle_us": "Idle µs", "graph_gpu_span_us": "Graph replay GPU span µs"}


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the ``xinfer-profile`` command line (``ops`` and ``load`` subcommands)."""
    parser = argparse.ArgumentParser(prog="xinfer-profile", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("ops", help="the GPU time, cost and bound of every operator")
    load = commands.add_parser("load", help="per-request segment times under closed-loop robot load")
    for sub in (command, load):
        sub.add_argument("--model", required=True, help="registered model name or module:Class")
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--requests", required=True, help="request file (JSON or msgpack), as for --warmup")
        sub.add_argument("--model-options", default="{}", help="JSON options, as for xinfer-serve")
        sub.add_argument("--device", default="cuda")
        sub.add_argument("--out", type=Path, required=True, help="output directory")
    load.add_argument("--robots", type=int, default=8)
    load.add_argument("--requests-per-robot", type=int, default=80)
    load.add_argument("--warmup-requests", type=int, default=20, help="each robot's first requests left out")
    load.add_argument("--period-ms", type=float, default=180.0, help="a robot's shortest time between requests")
    load.add_argument("--start", choices=("spread", "together"), default="spread",
                      help="first requests spread over one period (default) or sent at once")
    load.add_argument("--serve", action="store_true",
                      help="run the robots over WebSocket against an xinfer-serve started for the run")
    load.add_argument("--serve-args", default="", help="more xinfer-serve options, with --serve")
    command.add_argument("--stages", default=None, help="comma-separated stages to report (default: all)")
    command.add_argument("--requests-per-run", type=int, default=None,
                         help="requests profiled, cycling through the file (default: its request count)")
    command.add_argument("--peaks", default=None, help="JSON {pipe: FLOP/s, dram: bytes/s}; default: the table")
    return parser.parse_args(argv)


def _number(value, digits: int = 1) -> str:
    return "" if value is None else value if isinstance(value, str) else f"{value:.{digits}f}"


def _markdown(rows: list[dict], columns: dict) -> list[str]:
    """A Markdown table of ``rows``; shares and efficiencies as percentages."""
    lines = ["| " + " | ".join(columns.values()) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = [f"{100 * row[k]:.1f}%" if k in ("share", "efficiency") and row[k] is not None
                 else f"`{row[k]}`" if k == "shape" and row[k] else _number(row[k], DIGITS.get(k, 1)) for k in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _profile(run, path: Path) -> list[dict]:
    """Run ``run()`` under torch.profiler (CPU and CUDA, every thread) and save the Chrome trace to ``path``."""
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    config = torch.profiler._ExperimentalConfig(profile_all_threads=True)  # the engine thread holds the spans
    with torch.profiler.profile(activities=activities, experimental_config=config) as profiler:
        run()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    return json.loads(path.read_text())["traceEvents"]


def _environment(device: str) -> dict:
    """GPU, clocks and power limit (nvidia-smi), versions and the source the command ran from."""
    import torch

    import xinfer

    props = torch.cuda.get_device_properties(device)
    fields = ("driver_version", "clocks.sm", "clocks.max.sm", "power.limit")
    smi = subprocess.run(["nvidia-smi", f"--id=GPU-{props.uuid}", f"--query-gpu={','.join(fields)}",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    package = Path(xinfer.__file__).parent
    source = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        source.update(path.relative_to(package).as_posix().encode() + b"\0" + path.read_bytes())
    commit = subprocess.run(["git", "-C", str(package), "rev-parse", "HEAD"], capture_output=True, text=True)
    return {"gpu": {"name": props.name, "uuid": str(props.uuid), **dict(zip(fields, smi.split(", ")))},
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "source": {"commit": commit.stdout.strip() or None, "xinfer_sha256": source.hexdigest()}}


def _in_process(args) -> tuple[list[dict], dict, dict]:
    """Run the load in process; returns the rows, each request's ``(start, end)`` and meta fields."""
    from xinfer import ops
    from xinfer.config import apply_image_preprocess_device, resolve
    from xinfer.entrypoints._factory import create_engine
    from xinfer.frontend.protocol import read_requests
    from xinfer.models.registry import get_model_class
    from xinfer.observability import annotate, load_profile
    from xinfer.platforms import resolve_platform

    platform = resolve_platform(args.device)
    if platform.capabilities.execution_model != "cuda-lanes":
        raise SystemExit("xinfer-profile load requires execution_model='cuda-lanes' for admission and stage spans; "
                         f"{platform.name} ({platform.device}) advertises "
                         f"{platform.capabilities.execution_model!r}")
    options = apply_image_preprocess_device(json.loads(args.model_options), platform, default="cpu")
    ops.registry.load_plugins()
    table = ops.registry.build_table(platform)
    with ops.use_ops(table):
        model = get_model_class(args.model).from_checkpoint(args.checkpoint, device=platform.device, ops=table,
                                                            **options)
    prepare = model.input_processor()
    requests = [prepare(request) for request in read_requests(args.requests)]
    quantization = options.get("quantization", "bf16")
    resolved = resolve(model, platform, model_name=args.model,
                       quantization=quantization if isinstance(quantization, str) else None)
    engine = create_engine(platform, model, resolved.engine_config())
    recorder = load_profile.SpanRecorder(platform.capabilities.cuda_interop)
    try:
        engine.warmup(requests)
        annotate.configure(recorder)
        times = load_profile.run_load(engine, requests, args.robots, args.requests_per_robot, args.period_ms / 1000,
                                      args.start)
    finally:
        annotate.configure()
        engine.shutdown()
    missing = args.robots * args.requests_per_robot - len(times)
    if missing:
        raise SystemExit(f"{missing} requests did not complete")
    stages = [(stage.name, stage.depends_on) for stage in model.stage_graph]
    windows = {tuple(map(int, key.split(":"))): span for key, span in times.items()}
    return (load_profile.segments(recorder, times, stages), windows,
            {"configuration": resolved.report(), "ops": table.report()})


def _run_clients(url: str, messages: list, robots: int, per_robot: int, period_s: float, start: str = "spread",
                timeout_s: float = 60.0) -> list[dict]:
    """Closed-loop robots over WebSocket against a running ``xinfer-serve``, sending ``messages`` (packed requests)
    as ``load_profile.run_load`` submits them, one connection each.

    The robots connect one after another, each after its metadata frame, so every frontend numbers its connections
    in robot order (``load_profile.serve_segments`` relies on it).

    Returns:
        One row per request: ``robot``, ``index``, ``sent`` and ``received`` (``time.perf_counter()``), ``status``
        and the response's ``server_timing``.
    """
    from websockets.sync.client import connect

    from xinfer.frontend.protocol import unpackb

    rows, lock = [], threading.Lock()

    def robot(r):
        next_at = began + (r * period_s / robots if start == "spread" else 0.0)
        for i in range(per_robot):
            time.sleep(max(next_at - time.perf_counter(), 0.0))
            sent = time.perf_counter()
            connections[r].send(messages[(r + i) % len(messages)])
            reply = unpackb(connections[r].recv(timeout=timeout_s))
            received = time.perf_counter()
            with lock:
                rows.append(dict(robot=r, index=i, sent=sent, received=received, status=reply.get("status"),
                                 server_timing=reply.get("server_timing")))
            next_at = max(sent + period_s, time.perf_counter())

    with contextlib.ExitStack() as stack:
        connections = []
        for _ in range(robots):
            connections.append(stack.enter_context(connect(url, max_size=None, ping_interval=None)))
            connections[-1].recv()
        began = time.perf_counter()
        threads = [threading.Thread(target=robot, args=(r,), name=f"robot-{r}") for r in range(robots)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    return rows


def _through_serve(args) -> tuple[list[dict], dict, dict]:
    """Start ``xinfer-serve --profile-dir``, run the robots over WebSocket and stop it; returns the rows, each
    request's ``(sent, received)`` and meta fields."""
    from xinfer.frontend.protocol import packb, read_requests
    from xinfer.observability import load_profile

    messages = [packb(request) for request in read_requests(args.requests)]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    engine_dir, log_path = args.out / "engine", args.out / "serve.log"
    command = [sys.executable, "-m", "xinfer.entrypoints.serve", "--model", args.model, "--checkpoint",
               args.checkpoint, "--device", args.device, "--host", "127.0.0.1", "--port", str(port),
               "--model-options", args.model_options, "--warmup", args.requests, "--profile-dir", str(engine_dir),
               *shlex.split(args.serve_args)]
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        while "XINFER READY" not in log_path.read_text():
            if process.poll() is not None:
                raise SystemExit(f"xinfer-serve exited with status {process.returncode}; see {log_path}")
            time.sleep(0.2)
        rows = _run_clients(f"ws://127.0.0.1:{port}", messages, args.robots, args.requests_per_robot,
                                        args.period_ms / 1000, args.start)
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=120)
    failed = sum(row["status"] != "ok" for row in rows)
    if failed or process.returncode:
        raise SystemExit(f"{failed} requests failed, xinfer-serve exited with status {process.returncode}; "
                         f"see {log_path}")
    engine = json.loads((engine_dir / "engine-requests.json").read_text())
    windows = {(row["robot"], row["index"]): (row["sent"], row["received"]) for row in rows}
    return (load_profile.serve_segments(rows, engine["requests"], engine["stages"]), windows,
            {"serve": {"command": command, "log": str(log_path)}})


def _load(args) -> int:
    """Run ``xinfer-profile load`` (see the module docstring)."""
    from xinfer.observability import load_profile

    args.out.mkdir(parents=True, exist_ok=True)
    rows, windows, fields = (_through_serve if args.serve else _in_process)(args)
    steady = [row for row in rows if row["index"] >= args.warmup_requests]
    summary = load_profile.summarize(steady)
    spans = [windows[row["robot"], row["index"]] for row in steady]
    summary["throughput_rps"] = len(spans) / (max(end for _, end in spans) - min(start for start, _ in spans))
    with open(args.out / "requests.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    head = (f"{args.robots} robots, {args.requests_per_robot} requests each (the first {args.warmup_requests} left "
            f"out), one every {args.period_ms:g} ms or on the response, first requests {args.start}"
            f"{', through xinfer-serve' if args.serve else ', in process'}: {summary['requests']} requests, "
            f"{summary['throughput_rps']:.1f} req/s")
    lines = [f"# Requests of {args.model} under load", "", head, "", "| Segment | P50 ms | P95 ms | P99 ms |",
             "|---|---|---|---|"]
    lines += [f"| {name} | {p['p50']:.2f} | {p['p95']:.2f} | {p['p99']:.2f} |"
              for name, p in summary["segments"].items()]
    lines += ["", "Batch sizes: " + "; ".join(f"{stage} {sizes}" for stage, sizes in summary["batch_sizes"].items())]
    for q, excess in summary["tail"].items():
        top = ", ".join(f"{name} {value:+.1f}" for name, value in list(excess.items())[:6])
        lines += ["", f"At or above {q} of total_ms, against the median band (ms): {top}"]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "command": sys.argv,
            "model": args.model, "checkpoint": args.checkpoint, "model_options": json.loads(args.model_options),
            "requests": {"path": args.requests,
                         "sha256": hashlib.sha256(Path(args.requests).read_bytes()).hexdigest()},
            "load": {k: getattr(args, k) for k in ("robots", "requests_per_robot", "warmup_requests", "period_ms",
                                                   "start", "serve")},
            **fields, **(_environment(args.device) if args.device.startswith("cuda") else {})}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("\n".join(lines[:3] + [f"wrote {args.out}"]), flush=True)
    return 0


def main(argv=None) -> int:
    """Run ``xinfer-profile ops`` or ``xinfer-profile load`` (see the module docstring)."""
    args = parse_args(argv)
    if args.command == "load":
        return _load(args)
    from xinfer import ops
    from xinfer.config import ExplicitSettings, apply_image_preprocess_device, resolve
    from xinfer.entrypoints._factory import create_engine
    from xinfer.frontend.protocol import read_requests
    from xinfer.models.registry import get_model_class
    from xinfer.observability import annotate, op_profile
    from xinfer.observability.roofline import PEAKS
    from xinfer.platforms import resolve_platform

    platform = resolve_platform(args.device)
    if not platform.capabilities.cuda_interop:
        raise SystemExit("xinfer-profile ops currently needs torch/Kineto profiling on a CUDA device")

    args.out.mkdir(parents=True, exist_ok=True)
    options = apply_image_preprocess_device(json.loads(args.model_options), platform, default="cpu")
    ops.registry.load_plugins()
    base = ops.registry.build_table(platform)
    recorder = op_profile.CostRecorder()
    table = annotate.profiled_ops(base).wrapped(recorder)
    with ops.use_ops(table):
        model = get_model_class(args.model).from_checkpoint(args.checkpoint, device=platform.device, ops=table,
                                                            **options)
    prepare = model.input_processor()
    requests = [prepare(request) for request in read_requests(args.requests)]
    count = args.requests_per_run or len(requests)
    quantization = options.get("quantization", "bf16")

    def profile(cudagraphs: bool, trace: Path) -> op_profile.Attribution:
        resolved = resolve(model, platform, model_name=args.model,
                           quantization=quantization if isinstance(quantization, str) else None,
                           explicit=ExplicitSettings(cudagraphs=cudagraphs))
        engine = create_engine(platform, model, resolved.engine_config())
        try:
            for request in requests * 2:  # warmup: lazy initialization, autotuning, graph capture; costs if eager
                engine.run(request)
            recorder.active = False
            annotate.configure(annotate.kineto)
            events = _profile(lambda: [engine.run(requests[i % len(requests)]) for i in range(count)], trace)
        finally:
            annotate.configure()
            engine.shutdown()
        op_profile.require_cuda_activity(events)
        return op_profile.attribute(events)

    eager = profile(False, args.out / "trace.json")
    replay = profile(True, args.out / "trace-graphs.json")
    environment = _environment(platform.device)
    if args.peaks:
        peaks, peaks_source = json.loads(Path(args.peaks).read_text()), args.peaks
    else:
        peaks, peaks_source = PEAKS.get(environment["gpu"]["name"]), "table"
    stages = set(args.stages.split(",")) if args.stages else None
    rows = [r for r in op_profile.op_rows(eager, count, recorder.costs, peaks) if not stages or r["stage"] in stages]
    stage_rows = [r for r in op_profile.stage_rows(eager, count, replay) if not stages or r["stage"] in stages]
    covered = op_profile.coverage(eager)
    columns = {k: v for k, v in OP_COLUMNS.items() if peaks or k not in BOUND_COLUMNS}
    summary = (f"{count} request(s) of {args.requests} profiled eagerly on {environment['gpu']['name']}; values per "
               f"request. {covered['in_operators']} of {covered['activities']} GPU activities "
               f"({100 * covered['in_operators_us'] / covered['us']:.1f}% of GPU time) ran inside operator spans, "
               f"{covered['in_stages']} ({100 * covered['in_stages_us'] / covered['us']:.1f}%) inside stage spans.")
    rates = ", ".join(f"{k} {v / 1e9:.0f} GB/s" if k == "dram" else f"{k} {v / 1e12:.1f} TFLOP/s"
                      for k, v in (peaks or {}).items())
    notes = [summary, "", f"Peaks ({peaks_source}): {rates}." if peaks else
             f"No peaks for {environment['gpu']['name']} (pass --peaks): the bound columns are omitted."]
    lines = [f"# Operators of {args.model}", "", *notes]
    for stage in dict.fromkeys(r["stage"] for r in rows):
        lines += ["", f"## {stage}", "", *_markdown([r for r in rows if r["stage"] == stage], columns)]
    (args.out / "ops.md").write_text("\n".join(lines) + "\n")
    with open(args.out / "ops.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["stage"])
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "ops.json").write_text(json.dumps({"coverage": covered, "rows": rows, "stages": stage_rows}, indent=1))
    reconcile = ("Per stage and request: operators + outside - overlap + idle = GPU span (first activity start to "
                 "last activity end of each eager dispatch); the graph replay column is the same span under CUDA "
                 "graph replay, the serving path.")
    (args.out / "stages.md").write_text("\n".join([f"# Stages of {args.model}", "", summary, "", reconcile, "",
                                                   *_markdown(stage_rows, STAGE_COLUMNS)]) + "\n")
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "command": sys.argv,
            "model": args.model, "checkpoint": args.checkpoint, "model_options": options,
            "requests": {"path": args.requests, "profiled": count,
                         "sha256": hashlib.sha256(Path(args.requests).read_bytes()).hexdigest()},
            **environment, "ops": base.report(), "peaks": {"source": peaks_source if peaks else None, "values": peaks}}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("\n".join(notes + [f"wrote {args.out}"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

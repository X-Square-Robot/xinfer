"""``xinfer-profile ops``: the GPU time, cost and roofline bound of every operator a model runs (docs/profiling.md).

The model is built under a profiled op table, so every operator call runs in an ``op::`` span, and the engine marks
each stage dispatch with a ``stage::`` span. The command runs every request twice eagerly (warmup; the costs of the
call shapes are recorded), then ``--requests-per-run`` requests eagerly under torch.profiler with CPU and CUDA
activity, and attributes every GPU activity to its operator call and stage. Then the same requests run under CUDA
graph replay, the serving path, for each stage's GPU span. It writes ``ops.md``, ``ops.csv``, ``ops.json``,
``stages.md``, ``trace.json`` (the eager kineto trace), ``trace-graphs.json`` and ``meta.json`` to ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
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
    """Parse the ``xinfer-profile`` command line (``ops`` subcommand)."""
    parser = argparse.ArgumentParser(prog="xinfer-profile", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    command = parser.add_subparsers(dest="command", required=True).add_parser("ops", help=__doc__.splitlines()[0])
    command.add_argument("--model", required=True, help="registered model name or module:Class")
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--requests", required=True, help="request file (JSON or msgpack), as for --warmup")
    command.add_argument("--model-options", default="{}", help="JSON options, as for xinfer-serve")
    command.add_argument("--stages", default=None, help="comma-separated stages to report (default: all)")
    command.add_argument("--requests-per-run", type=int, default=None,
                         help="requests profiled, cycling through the file (default: its request count)")
    command.add_argument("--peaks", default=None, help="JSON {pipe: FLOP/s, dram: bytes/s}; default: the table")
    command.add_argument("--device", default="cuda")
    command.add_argument("--out", type=Path, required=True, help="output directory")
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


def main(argv=None) -> int:
    """Run ``xinfer-profile ops`` (see the module docstring)."""
    args = parse_args(argv)
    from xinfer import ops
    from xinfer.config import ExplicitSettings, resolve
    from xinfer.engine import Engine
    from xinfer.frontend.protocol import read_requests
    from xinfer.models.registry import get_model_class
    from xinfer.observability import annotate, op_profile
    from xinfer.observability.roofline import PEAKS
    from xinfer.platforms import resolve_platform

    platform = resolve_platform(args.device)
    if not platform.is_cuda:
        raise SystemExit("xinfer-profile ops needs a CUDA device")
    args.out.mkdir(parents=True, exist_ok=True)
    options = json.loads(args.model_options)
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
        engine = Engine(model, platform, resolved.engine_config())
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
            "requests": {"path": args.requests, "count": len(requests), "profiled": count,
                         "sha256": hashlib.sha256(Path(args.requests).read_bytes()).hexdigest()},
            **environment, "ops": base.report(), "peaks": {"source": peaks_source if peaks else None, "values": peaks}}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("\n".join(notes + [f"wrote {args.out}"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

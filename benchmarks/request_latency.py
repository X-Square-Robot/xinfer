"""Single-request latency of a model served in process (``xinfer.Policy``), one request at a time.

Each run takes the next wire request through the three steps of serving it and times each:

- ``prepare_ms``: the model's input processing (decode, resize, tokenize, normalize) on the CPU;
- ``engine_ms``: from the prepared CPU tensors to the final outputs on the CPU (upload, every stage, copy back);
- ``output_ms``: the model's output processing into the response;
- ``total_ms``: the whole request, from the wire request to the response.

No network and no other requests are involved. The requests are first run once for the warmup, which captures
their device graphs where enabled, then ``--warmup-runs`` more runs are discarded. Network latency
with one robot is measured by ``robot_load.py --robots 1``. See docs/benchmark.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def statistics(values: list[float]) -> dict:
    return {**{f"p{q}": float(np.percentile(values, q)) for q in (50, 95, 99)},
            "mean": float(np.mean(values)), "min": min(values), "max": max(values)}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="registered model name or module:Class")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--requests-file", nargs="+", required=True,
                        help="JSON or msgpack files with one wire request or a list, as for xinfer-serve --warmup")
    parser.add_argument("--model-options", default="{}", help="JSON options, as for xinfer-serve")
    parser.add_argument("--device", default=None, help="cuda, cuda:N, npu, npu:N or cpu (default: CUDA when available)")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--op-backends", default=None,
                        help="comma-separated operator=pin items, as for xinfer-serve --op-backends")
    parser.add_argument("--out", type=Path, required=True, help="new JSON result file")
    args = parser.parse_args(argv)
    if args.runs < 1 or args.warmup_runs < 0:
        parser.error("runs must be positive and warmup-runs nonnegative")
    if args.out.exists():
        parser.error(f"{args.out} already exists")
    return args


def _parse_pins(text: str | None) -> dict[str, str] | None:
    """``"op=pin,op=pin"`` to a mapping, the same form ``xinfer-serve --op-backends`` takes."""
    if not text:
        return None
    return {name.strip(): pin.strip() for name, _, pin in (item.partition("=") for item in text.split(","))
            if name.strip()}


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch

    from xinfer import Policy
    from xinfer.frontend.protocol import read_requests
    from xinfer.platforms.devices import device_module_for

    requests = [request for path in args.requests_file for request in read_requests(path)]
    policy = Policy(args.model, args.checkpoint, device=args.device,
                    op_backends=_parse_pins(args.op_backends), **json.loads(args.model_options))
    try:
        policy.warmup(requests)
        warmed = policy.engine.cudagraph_stats()
        samples = {name: [] for name in ("total_ms", "prepare_ms", "engine_ms", "output_ms")}
        for run in range(args.warmup_runs + args.runs):
            started = time.perf_counter()
            prepared = policy.prepare(requests[run % len(requests)])
            prepared_at = time.perf_counter()
            outputs = policy.run_prepared(prepared)
            ran_at = time.perf_counter()
            policy.finish(outputs, prepared)
            finished = time.perf_counter()
            if run >= args.warmup_runs:
                for name, begin, end in (("total_ms", started, finished), ("prepare_ms", started, prepared_at),
                                         ("engine_ms", prepared_at, ran_at), ("output_ms", ran_at, finished)):
                    samples[name].append((end - begin) * 1000)
        captures = {stage: counters["captures"] - warmed.get(stage, {}).get("captures", 0)
                    for stage, counters in policy.engine.cudagraph_stats().items()}
    finally:
        policy.engine.shutdown()
    try:
        revision = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip() or None
    except OSError:  # no git on this machine
        revision = None
    device = policy.platform.device
    module = device_module_for(device)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "revision": revision,
              "inputs": [{"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                         for path in args.requests_file],
              "device": module.get_device_name(device) if module is not None else "cpu",
              "torch": torch.__version__, "policy": policy.describe(), "captures_after_warmup": captures,
              "runs": args.runs, **{name: statistics(values) for name, values in samples.items()},
              "samples": samples}
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"{args.runs} runs, P50/P99 (ms): " + "; ".join(
        f"{name[:-3]} {result[name]['p50']:.2f}/{result[name]['p99']:.2f}" for name in samples), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

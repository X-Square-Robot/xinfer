"""``xinfer-profile ops`` end to end on a tiny two-stage model (GPU)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from xinfer.observability.roofline import PEAKS


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_ops_writes_the_tables(tmp_path):
    requests, out = tmp_path / "requests.json", tmp_path / "out"
    requests.write_text(json.dumps([{"x": [0.5] * 64}]))
    path = os.pathsep.join(filter(None, [str(Path(__file__).parent), os.environ.get("PYTHONPATH")]))
    command = [sys.executable, "-m", "xinfer.entrypoints.profile", "ops", "--model", "profile_model:Tiny",
               "--checkpoint", "unused", "--requests", str(requests), "--requests-per-run", "3", "--out", str(out)]
    run = subprocess.run(command, env={**os.environ, "PYTHONPATH": path}, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert {p.name for p in out.iterdir()} == {"ops.md", "ops.csv", "ops.json", "stages.md", "trace.json",
                                               "trace-graphs.json", "meta.json"}
    result, meta = json.loads((out / "ops.json").read_text()), json.loads((out / "meta.json").read_text())
    rows = {(row["stage"], row["op"]): row for row in result["rows"]}
    linear, norm = rows[("encode", "linear")], rows[("decode", "rms_norm")]
    assert linear["calls"] == 1 and linear["us_per_call"] > 0 and linear["gflop_per_call"] == 2 * 256 * 64 / 1e9
    assert (linear["cost"], norm["cost"]) == ("formula", "generic") and norm["calls"] == 1
    assert (linear["bound_us_per_call"] is not None) == (meta["gpu"]["name"] in PEAKS)
    assert result["coverage"]["in_operators"] >= 2 and meta["requests"]["profiled"] == 3
    for stage in result["stages"]:
        assert stage["dispatches"] == 1 and stage["graph_gpu_span_us"] > 0
        assert stage["operators_us"] + stage["outside_us"] - stage["overlap_us"] + stage["idle_us"] == \
            pytest.approx(stage["gpu_span_us"])

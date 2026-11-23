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


def test_check_distinguishes_accepted_and_declined_calls_on_the_same_backend(tmp_path, monkeypatch):
    from xinfer import ops
    from xinfer.entrypoints.profile import _write_check
    from xinfer.observability.op_profile import ReferenceRecorder
    from xinfer.ops.backends.cuda import flow
    from xinfer.ops.spec import Contract

    x, weight = torch.ones(1, 2, 4), torch.ones(1, 16, 4)
    monkeypatch.setattr(flow, "denoise_input_embed_takes", lambda x, weight: x.size(1) == 1)
    monkeypatch.setattr(flow, "denoise_input_embed", lambda x, weight, **kw: x @ weight[0].T)
    monkeypatch.setenv("XINFER_OPS_STRICT", "1")
    implementation = ops.Implementation(op="flow_input_embed", backend="cuda", function=flow.flow_input_embed,
                                        platforms=("cuda",), contract=Contract(), priority=1, library="kernels")
    base = ops.OpTable("cuda", {"flow_input_embed": implementation})
    replay = ReferenceRecorder()
    table = base.wrapped(replay)
    stages = {"action": ("flow_input_embed",)}
    table.flow_input_embed(x[:, :1], weight)
    assert _write_check(tmp_path, "tiny", "cuda", base, stages, replay) == 0
    table.flow_input_embed(x, weight)
    table.flow_input_embed(x, weight)
    assert _write_check(tmp_path, "tiny", "cuda", base, stages, replay) == 1
    row, = json.loads((tmp_path / "check.json").read_text())["rows"]
    assert row["observed"] and row["backend"] == "cuda" and row["library"] == "kernels"
    shape = "x=float32[1,2,4] weight=float32[1,16,4]"
    assert row["reference_calls"] == [{"shape": shape, "source": "flow_input_embed"}]
    assert "x=float32[1,2,4]" in (tmp_path / "check.md").read_text()
    with pytest.raises(RuntimeError, match="flow_input_embed.*XINFER_OPS_STRICT"):
        base.flow_input_embed(x, weight)
    assert not ReferenceRecorder().references


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("model,expected", [("Tiny", 0), ("ReferenceProjection", 1)])
def test_check_exit_status_and_reports_on_cuda(tmp_path, model, expected):
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps([{"x": [0.5] * 64}]))
    path = os.pathsep.join(filter(None, [str(Path(__file__).parent), os.environ.get("PYTHONPATH")]))
    out = tmp_path / "out"
    run = subprocess.run([sys.executable, "-m", "xinfer.entrypoints.profile", "check",
                          "--model", f"profile_model:{model}", "--checkpoint", "unused",
                          "--requests", str(requests), "--out", str(out)],
                         env={**os.environ, "PYTHONPATH": path, "XINFER_OPS_STRICT": "1"},
                         capture_output=True, text=True)
    assert run.returncode == expected, run.stderr
    report = json.loads((out / "check.json").read_text())
    assert report["ok"] == (expected == 0)
    fallbacks = [r["op"] for r in report["rows"] if r["reference_calls"]]
    assert fallbacks == (["gemm_epilogue"] if expected else [])

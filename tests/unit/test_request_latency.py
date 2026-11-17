"""benchmarks/request_latency.py with the echo model on the CPU."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("msgpack_numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import request_latency  # noqa: E402

from xinfer.frontend.protocol import packb  # noqa: E402


def test_single_request_latency_is_measured_per_run(tmp_path, monkeypatch):
    import xinfer

    create_policy = xinfer.Policy
    passed_options = []

    def policy(*args, **kwargs):
        passed_options.append(kwargs)
        return create_policy(*args, **kwargs)

    monkeypatch.setattr(xinfer, "Policy", policy)
    requests = tmp_path / "requests.msgpack"
    requests.write_bytes(packb([{"x": np.zeros(3, np.float32)}, {"x": np.ones(3, np.float32)}]))
    out = tmp_path / "latency.json"
    options = {"scale": 3.0, "op_backends": {"rms_norm": "reference"}, "reference_ops": True, "cudagraphs": False}
    code = request_latency.main(["--model", "echo_model:EchoModel", "--checkpoint", "unused", "--device", "cpu",
                                 "--requests-file", str(requests), "--runs", "7", "--warmup-runs", "2",
                                 "--model-options", json.dumps(options), "--out", str(out)])
    assert code == 0
    assert passed_options
    assert all(all(kwargs.get(name) == value for name, value in {"device": "cpu", **options}.items())
               for kwargs in passed_options)
    result = json.loads(out.read_text())
    assert result["runs"] == 7 and result["device"] == "cpu" and result["policy"]["platform"] == "cpu"
    samples = result["samples"]
    for name in ("total_ms", "prepare_ms", "engine_ms", "output_ms"):
        assert len(samples[name]) == 7 and 0 < result[name]["min"] <= result[name]["p50"] <= result[name]["max"]
    for total, *steps in zip(samples["total_ms"], samples["prepare_ms"], samples["engine_ms"], samples["output_ms"]):
        assert abs(total - sum(steps)) < 1e-6  # the end-to-end time is the sum of the three steps
    assert result["captures_after_warmup"] == {}  # no CUDA graphs on the CPU


def test_reference_cli_option_reaches_policy_and_keeps_operator_pins(tmp_path, monkeypatch):
    import xinfer

    create_policy = xinfer.Policy
    passed = []

    def policy(*args, **kwargs):
        passed.append(kwargs)
        return create_policy(*args, **kwargs)

    monkeypatch.setattr(xinfer, "Policy", policy)
    requests = tmp_path / "requests.msgpack"
    requests.write_bytes(packb({"x": np.zeros(3, np.float32)}))
    out = tmp_path / "latency.json"
    assert request_latency.main([
        "--model", "echo_model:EchoModel", "--checkpoint", "unused", "--device", "cpu",
        "--requests-file", str(requests), "--runs", "1", "--warmup-runs", "0",
        "--reference-ops", "--op-backends", "rms_norm=reference", "--out", str(out)]) == 0
    assert passed[0]["reference_ops"] is True and passed[0]["op_backends"] == {"rms_norm": "reference"}
    report = json.loads(out.read_text())["policy"]["ops"]
    assert all(row["backend"] == "reference" for row in report)

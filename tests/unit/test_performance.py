"""benchmarks/performance.py: the capacity rule and the table row (CPU, no service)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import performance  # noqa: E402


def _round(p50, p95=None, errors=0, missing=0, throughput=10.0):
    return {"errors": errors, "missing": missing, "throughput_rps": throughput,
            "server_infer_ms": {"p50": p50, "p95": p95 or p50 + 10}}


def test_a_count_is_past_the_knee_when_its_median_round_rises_over_half_or_a_round_has_errors():
    previous = [_round(80), _round(90), _round(100)]  # median 90
    assert not performance.past_knee([_round(100), _round(135), _round(200)], previous)  # median 135: +50 %
    assert performance.past_knee([_round(100), _round(135.1), _round(200)], previous)
    assert not performance.past_knee([_round(300)] * 3, None)  # the first count is the reference
    assert performance.past_knee([_round(90), _round(90, errors=1), _round(90)], previous)
    assert performance.past_knee([_round(90), _round(90, missing=1), _round(90)], None)
    assert performance.past_knee([], previous)


def test_round_statistics_give_the_median_and_the_worst_round():
    stats = performance.round_statistics([_round(90, 112), _round(140, 150), _round(105, 114)])
    assert stats == {"p50_median": 105.0, "p95_median": 114.0, "p50_worst": 140, "p95_worst": 150}


def test_the_capacity_is_the_largest_count_before_the_first_one_past_the_knee():
    def at(p50):
        return [_round(p50)] * 3
    assert performance.capacity({3: at(82), 4: at(115), 5: at(184)}) == (4, True)  # +41 %, then +60 %
    assert performance.capacity({3: at(93), 4: at(92), 5: at(156), 6: at(160)}) == (4, True)
    assert performance.capacity({3: at(80), 4: at(90)}) == (4, False)  # no count past the knee: a lower bound
    assert performance.capacity({3: [_round(80, errors=1)] * 3}) == (None, True)


def test_the_row_reports_the_single_request_parts_the_capacity_point_and_the_saturation_throughput():
    result = {"name": "M · BF16 · GPU",
              "latency": {"total_ms": {"p50": 44.21}, "prepare_ms": {"p50": 2.94}, "engine_ms": {"p50": 41.07},
                          "output_ms": {"p50": 0.18}},
              "load": {"3": [_round(97, 103, throughput=16.7)] * 3,
                       "4": [_round(90, 112, throughput=22.2), _round(140, 150, throughput=22.2),
                             _round(105, 114, throughput=22.2)],
                       "5": [_round(160, throughput=26.4)] * 3},
              "capacity": (4, True)}
    assert performance.row(result) == "| M · BF16 · GPU | 44.2 ms (3.1 + 41.1) | 4 | 105 ms | 114 ms | 26.4 req/s |"
    result["capacity"] = (5, False)
    result["load"]["5"] = [_round(120, throughput=26.4)] * 3
    assert "| ≥ 5 |" in performance.row(result)


def test_no_completed_round_reports_a_capacity_bound_and_zero_throughput():
    result = {"name": "M", "latency": {"total_ms": {"p50": 20}, "prepare_ms": {"p50": 1},
                                       "engine_ms": {"p50": 18}, "output_ms": {"p50": 1}},
              "load": {"2": [], "10": []}, "capacity": (None, True)}
    assert "| < 2 |" in performance.row(result)
    assert "0.0 req/s" in performance.row(result)


def test_npu_device_and_model_options_reach_both_children(tmp_path, monkeypatch):
    commands = {}

    def run(command, **kwargs):
        if command == ["npu-smi", "info"]:
            return SimpleNamespace(stdout="Ascend inventory\n")
        out = Path(command[command.index("--out") + 1])
        if "request_latency.py" in command[1]:
            commands["latency"] = command
            out.write_text(json.dumps({"device": "Ascend", "total_ms": {"p50": 25},
                                       "prepare_ms": {"p50": 2}, "engine_ms": {"p50": 22}, "output_ms": {"p50": 1},
                                       "captures_after_warmup": {}}))
        else:
            out.mkdir()
            (out / "summary.json").write_text(json.dumps({"rounds": [_round(50)] * performance.ROUNDS}))

    class Server:
        def __init__(self, command, stdout, stderr):
            commands["server"] = command
            stdout.write("XINFER READY\n")
            stdout.flush()

        def send_signal(self, signal):
            pass

        def wait(self, timeout):
            pass

    monkeypatch.setattr(performance.subprocess, "run", run)
    monkeypatch.setattr(performance.subprocess, "Popen", Server)
    monkeypatch.setattr(performance, "environment", lambda npu: {"npu_requested": npu})
    monkeypatch.setattr(performance.os, "getloadavg", lambda: (0, 0, 0), raising=False)
    monkeypatch.setattr(performance.time, "sleep", lambda seconds: None)
    out = tmp_path / "result"
    assert performance.main(["--name", "NPU", "--model", "m", "--checkpoint", "ckpt", "--device", "npu:2",
                             "--model-options", '{"image_preprocess_device":"cpu","processing":{"noise_seed":7}}',
                             "--latency-requests", "latency.json", "--warmup", "warmup.json", "--robots", "1",
                             "--out", str(out)]) == 0
    assert set(commands) == {"latency", "server"}
    for command in commands.values():
        assert command[command.index("--device") + 1] == "npu:2"
        options = json.loads(command[command.index("--model-options") + 1])
        assert options["image_preprocess_device"] == "cpu" and options["processing"]["noise_seed"] == 7
    machine = json.loads((out / "robots-1" / "machine.json").read_text())
    assert json.loads((out / "row.json").read_text())["capacity_rule"] == {"median_p50_ratio": 1.5}
    assert machine["before"]["npu_smi"] == "Ascend inventory"


def test_cuda_default_leaves_image_selection_to_policy_capabilities(tmp_path, monkeypatch):
    from xinfer.config import apply_image_preprocess_device

    options = {}

    def run(command, **kwargs):
        options["latency"] = json.loads(command[command.index("--model-options") + 1])
        output = Path(command[command.index("--out") + 1])
        output.write_text(json.dumps({"device": "CUDA", "total_ms": {"p50": 25},
                                      "prepare_ms": {"p50": 2}, "engine_ms": {"p50": 22}, "output_ms": {"p50": 1},
                                      "captures_after_warmup": {}}))

    def serve(args, robots, out):
        options["server"] = json.loads(args.model_options)
        return [_round(151)] * performance.ROUNDS

    monkeypatch.setattr(performance.subprocess, "run", run)
    monkeypatch.setattr(performance, "serve_and_load", serve)
    monkeypatch.setattr(performance, "environment", lambda npu: {})
    assert performance.main(["--name", "CUDA", "--model", "m", "--checkpoint", "ckpt",
                             "--latency-requests", "latency.json", "--warmup", "warmup.json", "--robots", "1",
                             "--out", str(tmp_path / "result")]) == 0
    assert set(options) == {"latency", "server"}
    assert all("image_preprocess_device" not in child for child in options.values())
    # Without nvImageCodec, Policy must retain its supported CPU image path.
    platform = SimpleNamespace(name="cuda-sm120", device="cuda:0",
                               capabilities=SimpleNamespace(device_images=False))
    for child in options.values():
        resolved = apply_image_preprocess_device(child, platform)
        assert resolved.get("processing", {}).get("image_preprocess_device", "cpu") == "cpu"

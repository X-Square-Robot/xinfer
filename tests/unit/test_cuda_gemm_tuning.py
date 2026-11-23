"""Persistence, strict replay, and execution through the generic CUDA GEMM entry point."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
import subprocess
import sys

import pytest
import torch

from xinfer.ops.backends.cuda import gemm_tuning as tuning


def test_concurrent_writers_preserve_the_first_choice_per_shape(tmp_path):
    path = tmp_path / "nested" / "tuning.json"
    with ThreadPoolExecutor(max_workers=4) as pool:
        winners = list(pool.map(lambda i: tuning._store(path, "same", str(i)), range(8)))
        list(pool.map(lambda i: tuning._store(path, str(i), str(i)), range(8)))
    assert len(set(winners)) == 1
    assert tuning._read(path) == {"same": winners[0], **{str(i): str(i) for i in range(8)}}


def test_frozen_replay_matches_algorithm_identity_and_never_times_or_writes(tmp_path, monkeypatch):
    path = tmp_path / "tuning.json"
    mode = ["1"]
    monkeypatch.setattr(tuning, "settings", lambda: (mode[0], path))
    monkeypatch.setattr(tuning, "_indices", {})
    monkeypatch.setattr(tuning, "_hardware", lambda device: ("device", "lib"))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(tuning, "capturing", lambda device: False)
    monkeypatch.setattr(tuning, "_choose", lambda *_: "winner")
    ids, launches = ["first", "winner"], []
    out = torch.zeros(2, 4)

    def run(tensor, index):
        if index == -1:
            return " ".join(ids)
        launches.append(index)

    tuning.execute(out, (None,), "dense_nk", run)
    frozen = path.read_bytes()
    mode[0] = "frozen"
    ids.reverse()
    monkeypatch.setattr(tuning, "_choose", lambda *_: pytest.fail("tried to time a frozen call"))
    tuning.execute(out, (None,), "dense_nk", run)
    assert launches == [1, 0] and path.read_bytes() == frozen
    with pytest.raises(ValueError, match="No matching frozen"):
        tuning.execute(out[:, :2], (None,), "dense_nk", run)
    tuning._indices.clear()
    ids.remove("winner")
    with pytest.raises(ValueError, match="No matching frozen"):
        tuning.execute(out, (None,), "dense_nk", run)
    monkeypatch.setattr(tuning, "capturing", lambda device: True)
    with pytest.raises(ValueError, match="eagerly"):
        tuning.execute(out, (None,), "dense_nk", run)
    mode[0] = "1"
    tuning.execute(out, (None,), "dense_nk", run)
    assert launches[-1] == 0 and path.read_bytes() == frozen


def test_private_trials_require_bitwise_agreement_and_keep_the_input(monkeypatch):
    out = torch.tensor([[23., 42., 7.]])
    before = out.clone()
    monkeypatch.setattr(tuning, "_time", lambda run, scratch, index: (1.0, 0.1, 0.8)[index])

    def run(scratch, index):
        assert scratch.data_ptr() != out.data_ptr()
        scratch.fill_(float(index == 1))

    assert tuning._choose(out, ("heuristic", "inaccurate", "faster"), run) == "faster"
    assert torch.equal(out, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("transposed", [False, True])
def test_generic_bf16_tunes_and_replays_in_a_graph(tmp_path, monkeypatch, transposed):
    from xinfer.ops.backends.cuda import gemm
    from xinfer.ops.backends.cuda._native import BUILD_ARCHS, arch

    device = torch.device("cuda", torch.cuda.current_device())
    if arch(device) not in BUILD_ARCHS:
        pytest.skip("requires a CUDA kernel library built for this device")
    mode, path = ["0"], tmp_path / "record.json"
    monkeypatch.setattr(tuning, "settings", lambda: (mode[0], path))
    monkeypatch.setattr(tuning, "_indices", {})
    generator = torch.Generator(device=device).manual_seed(19)

    def rand(*shape):
        return torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)

    x = rand(33, 128)
    w = rand(128, 96).T if transposed else rand(96, 128)
    bias, residual = rand(96), rand(33, 96)
    assert gemm.kernel_name("gemm", x, w, bias, residual=residual) == "generic"
    baseline = gemm.gemm(x, w, bias, residual=residual)
    mode[0] = "1"
    result = gemm.gemm(x, w, bias, residual=residual)
    assert torch.equal(result, baseline) and path.is_file()
    frozen = path.read_bytes()
    mode[0] = "frozen"
    assert torch.equal(gemm.gemm(x, w, bias, residual=residual), baseline)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gemm.gemm(x, w, bias, residual=residual)  # allocate this lane's workspace
    with torch.cuda.graph(graph, stream=stream):
        replay = gemm.gemm(x, w, bias, residual=residual)
    graph.replay()
    assert torch.equal(replay, baseline) and path.read_bytes() == frozen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generic_mxfp8_tunes_and_replays(tmp_path, monkeypatch):
    from xinfer.ops.backends.cuda import gemm
    from xinfer.ops.backends.cuda._native import BUILD_ARCHS, arch
    from xinfer.ops.reference import mxfp8

    device = torch.device("cuda", torch.cuda.current_device())
    if arch(device) != 120 or 120 not in BUILD_ARCHS:
        pytest.skip("requires an SM120 build and GPU")
    generator = torch.Generator(device=device).manual_seed(19)

    def operand(rows):
        return mxfp8.quantize(torch.randn(rows, 128, device=device, dtype=torch.bfloat16, generator=generator))

    x, w = operand(64), operand(128)
    assert gemm.kernel_name("gemm", x, w) == "generic"
    mode, path = ["0"], tmp_path / "mxfp8.json"
    monkeypatch.setattr(tuning, "settings", lambda: (mode[0], path))
    baseline = gemm.gemm(x, w)
    for mode[0] in ("1", "frozen"):
        assert torch.equal(gemm.gemm(x, w), baseline)
    assert path.is_file()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_frozen_record_survives_a_new_process(tmp_path):
    from xinfer.ops.backends.cuda._native import BUILD_ARCHS, arch

    if arch(torch.device("cuda")) not in BUILD_ARCHS:
        pytest.skip("requires a native CUDA build")
    code = """
import hashlib
import torch
from xinfer.ops.backends.cuda import gemm
generator = torch.Generator(device="cuda").manual_seed(95)
def tensor(*shape):
    return torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
out = gemm.gemm(tensor(33, 128), tensor(96, 128))
print(hashlib.sha256(out.view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
"""
    path = tmp_path / "processes.json"
    def launch(mode):
        env = dict(os.environ, XINFER_GEMM_TUNING=mode, XINFER_GEMM_TUNING_FILE=str(path))
        return subprocess.check_output([sys.executable, "-c", code], env=env, text=True, timeout=120).strip()
    measured = launch("1")
    assert path.is_file() and launch("frozen") == measured

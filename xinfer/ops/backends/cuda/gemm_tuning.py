"""Eager cuBLASLt tuning and read-only replay for the public CUDA GEMM fallback."""

from __future__ import annotations

from functools import cache
import json
import os
from pathlib import Path
from statistics import median
import tempfile
from threading import RLock

import torch

from xinfer.ops.backends.cuda._native import NATIVE, capturing, lib

_indices = {}
_lock = RLock()


@cache
def settings():
    mode = os.environ.get("XINFER_GEMM_TUNING", "0")
    if mode not in ("0", "1", "frozen"):
        raise ValueError("XINFER_GEMM_TUNING must be 0, 1 or frozen")
    name = os.environ.get("XINFER_GEMM_TUNING_FILE")
    if mode == "frozen" and not name:
        raise ValueError("frozen GEMM tuning requires XINFER_GEMM_TUNING_FILE")
    return mode, Path(name).expanduser() if name else None


@cache
def _hardware(device):
    props = torch.cuda.get_device_properties(device)
    return (props.name, props.major, props.minor, props.multi_processor_count, torch.version.cuda,
            int(lib.gemm_cublaslt_version()), NATIVE["version"])


def _alignment(t):
    ptr = t.data_ptr()
    return min(256, ptr & -ptr) if ptr else 256


def _layout(t):
    return None if t is None else (tuple(t.shape), t.stride(), str(t.dtype), _alignment(t))


def _read(path):
    with path.open() as stream:
        record = json.load(stream)
    if not isinstance(record, dict) or not all(isinstance(v, str) for v in record.values()):
        raise ValueError(f"Invalid GEMM tuning record in {path}")
    return record


@cache
def _snapshot(path):
    return _read(path)


def _store(path, key, winner):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = _read(path) if path.exists() else {}
        if key in record:
            return record[key]
        record[key] = winner
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(record, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return winner


def _time(run, output, index):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    events[0].record()
    for event in events[1:]:
        run(output, index)
        event.record()
    events[-1].synchronize()
    return median(a.elapsed_time(b) for a, b in zip(events, events[1:]))


def _choose(out, ids, run):
    if len(ids) == 1:
        return ids[0]
    alignment = _alignment(out)
    offset = alignment // out.element_size() if alignment < 256 else 0
    scratch = out.new_empty(out.numel() + offset)[offset:].view_as(out)
    run(scratch, 0)
    reference = scratch.clone()
    baseline = _time(run, scratch, 0)
    winner, fastest = ids[0], baseline * 0.95
    for index, candidate in enumerate(ids[1:], 1):
        try:
            run(scratch, index)
            if not torch.equal(scratch, reference):
                continue
            elapsed = _time(run, scratch, index)
        except RuntimeError as error:
            if "failed with cuBLAS status 15" not in str(error):  # CUBLAS_STATUS_NOT_SUPPORTED
                raise
            continue
        if elapsed < fastest:
            winner, fastest = candidate, elapsed
    return winner


def execute(out, operands, format, run):
    """Use exact shapes and address alignments. The trial output never aliases a model buffer."""
    mode, path = settings()
    device = out.device
    signature = (format, _layout(out), tuple(_layout(t) for t in operands))
    local_key = (mode, path, device, signature)
    index = _indices.get(local_key)
    if index is None:
        with _lock, torch.cuda.device(device):
            index = _indices.get(local_key)
            if index is None:
                if capturing(device):
                    if mode == "frozen":
                        raise ValueError("warm this GEMM shape eagerly before frozen graph capture")
                    return run(out, 0)
                ids = str(run(out, -1)).split()
                if not ids:  # empty product
                    return run(out, 0)
                key = json.dumps((_hardware(device), signature), separators=(",", ":"))
                if mode == "frozen":
                    winner = _snapshot(path).get(key)
                    if winner not in ids:
                        raise ValueError(f"No matching frozen GEMM algorithm for this call in {path}")
                else:
                    winner = _read(path).get(key) if path and path.exists() else None
                    if winner is None:
                        winner = _choose(out, ids, run)
                        if path:
                            winner = _store(path, key, winner)
                index = ids.index(winner) if winner in ids else 0
                _indices[local_key] = index
    return run(out, index)

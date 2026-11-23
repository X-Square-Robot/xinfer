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
_buckets = {}
_lock = RLock()


@cache
def settings():
    mode = os.environ.get("XINFER_GEMM_TUNING", "1")
    if mode not in ("0", "1", "frozen"):
        raise ValueError("XINFER_GEMM_TUNING must be 0, 1 or frozen")
    name = os.environ.get("XINFER_GEMM_TUNING_FILE")
    if mode == "frozen" and not name:
        raise ValueError("frozen GEMM tuning requires XINFER_GEMM_TUNING_FILE")
    path = Path(name).expanduser() if name else None
    if mode == "1" and (os.environ.get("XINFER_GEMM_TUNING_READONLY") == "1"
                        or (path and path.exists() and path.stat().st_mode & 0o222 == 0)):
        mode = "readonly"
    if mode != "frozen" and lib is not None and not hasattr(lib, "gemm_select_dense"):
        mode = "0"  # Libraries built before candidate enumeration still provide the heuristic entry.
    return mode, path


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


def _aligned(t, rows):
    shape = (rows, *t.shape[1:])
    offset = (_alignment(t) % 256) // t.element_size()
    return t.new_empty((rows * t[0].numel() + offset,))[offset:].view(shape)


def _bucket_key(format, out, operands):
    x = operands[0]
    if x is None:
        return None
    count = (x.shape[0] - 1) // 64
    # The rows vary across a bucket. Preserve strides, base alignment and all weight metadata.
    def rows(t):
        return None if t is None else (t.shape[1:], t.stride(), str(t.dtype), _alignment(t))

    if format == "mxfp8":
        _, scale, weight, w_scale, bias = operands
        rest = (str(scale.dtype), _alignment(scale), _layout(weight), _layout(w_scale), _layout(bias))
    else:
        _, weight, bias, residual = operands
        rest = (_layout(weight), _layout(bias), rows(residual))
    signature = ("bucket64", _hardware(out.device), format, count, rows(out), rows(x), rest)
    return json.dumps(signature, separators=(",", ":"))


def _check_endpoints(out, operands, format, run, winner):
    m = out.shape[0]
    low = 1 + ((m - 1) // 64) * 64
    for rows in (low, low + 63):
        if rows == m:
            continue
        args = list(operands)
        x = args[0] = _aligned(operands[0], rows)
        x.view(torch.uint8).copy_(operands[0][:1].view(torch.uint8).expand_as(x.view(torch.uint8)))
        if format == "mxfp8":
            size = ((rows + 127) // 128) * 128 * ((x.shape[1] // 32 + 3) // 4) * 4
            args[1] = _aligned(operands[1].reshape(-1), size)
            args[1].view(torch.uint8).fill_(127)
        elif args[-1] is not None:
            args[-1] = _aligned(operands[-1], rows)
            args[-1].copy_(operands[-1][:1].expand_as(args[-1]))
        scratch = _aligned(out, rows)
        def trial(target, index):
            return run(target, index, args)
        try:
            ids = str(trial(scratch, -1)).split()
            if winner not in ids:
                return False
            index = ids.index(winner)
            trial(scratch, 0)
            baseline = scratch.clone()
            trial(scratch, index)
            if not torch.equal(scratch, baseline) or _time(trial, scratch, index) > _time(trial, scratch, 0):
                return False
        except RuntimeError as error:
            if "cuBLAS status 15" not in str(error):
                raise
            return False
    return True


def execute(out, operands, format, run):
    """Cache launch indices per exact shape and validate a portable algorithm ID across each 64-row bucket."""
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
                bucket = _bucket_key(format, out, operands)
                cache_key = (path, bucket)
                if mode in ("readonly", "frozen"):
                    record = _snapshot(path) if path and path.exists() else {}
                else:
                    record = _read(path) if path and path.exists() else {}
                winner = record.get(key) or record.get(bucket)
                if mode == "frozen" and winner not in (*ids, "heuristic"):
                    raise ValueError(f"No matching frozen GEMM algorithm for this call in {path}")
                if winner is None and mode == "1":
                    winner = _buckets.get(cache_key) if bucket else None
                    if winner is None:
                        winner = _choose(out, ids, run)
                        if bucket:
                            if winner == ids[0] or not _check_endpoints(out, operands, format, run, winner):
                                winner = "heuristic"
                            _buckets[cache_key] = winner
                        if path:
                            winner = _store(path, bucket or key, winner)
                index = ids.index(winner) if winner in ids else 0
                _indices[local_key] = index
    return run(out, index)

"""Compare and cache a native attention kernel and an optional library during eager warmup."""

from __future__ import annotations

from functools import cache
from importlib import metadata
import json
import logging
import os
from pathlib import Path
from statistics import median
from threading import RLock

import torch

from xinfer.ops.backends.cuda._attention_libraries import layout
from xinfer.ops.backends.cuda._native import NATIVE, capturing
from xinfer.ops.backends.cuda import gemm_tuning

logger = logging.getLogger(__name__)
_lock = RLock()
_winners = {}


def _time(fn, stream):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    events[0].record(stream)
    for event in events[1:]:
        fn()
        event.record(stream)
    events[-1].synchronize()
    return median(a.elapsed_time(b) for a, b in zip(events, events[1:]))


def _compare(device, native, external, library="flex_attention", tolerance=0.01):
    stream = torch.cuda.current_stream(device)
    try:
        expected = external()
    except torch.OutOfMemoryError:
        raise
    except (ImportError, RuntimeError, ValueError, TypeError) as error:
        stream.synchronize()
        logger.info("%s declined attention warmup: %s", library, error)
        return None
    try:
        actual = native()
    except torch.OutOfMemoryError:
        raise
    except (RuntimeError, ValueError) as error:
        stream.synchronize()
        logger.info("The native kernel declined attention warmup: %s", error)
        return library
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return library
    error = torch.linalg.vector_norm(actual.float() - expected.float())
    norm = torch.linalg.vector_norm(expected.float())
    if not bool(torch.isfinite(error) & torch.isfinite(norm) & ((error == 0) | (error < tolerance * norm))):
        return library
    return "kernels" if _time(native, stream) <= _time(external, stream) else library


@cache
def _versions(device, library):
    packages = {"flex_attention": ("triton",), "flash_attn": ("flash-attn",), "fla": ("fla-core", "triton")}
    versions = []
    for name in packages[library]:
        try:
            versions.append(metadata.version(name))
        except metadata.PackageNotFoundError:
            versions.append(None)
    props = torch.cuda.get_device_properties(device)
    return (props.name, props.major, props.minor, props.multi_processor_count, NATIVE["version"],
            torch.__version__, torch.version.cuda, versions)


def choose(op, args, native, external, library="flex_attention"):
    """Return None for static routing. Both candidates read the same inputs and allocate their own output."""
    # Equal strides can still describe slices with different base alignment.
    alignments = tuple(t.data_ptr() % 16 for t in args if torch.is_tensor(t))
    signature = (op, library, layout(args), alignments)
    if signature in _winners:
        return _winners[signature]
    device = args[0].device
    with _lock, torch.cuda.device(device):
        if signature in _winners:
            return _winners[signature]
        if capturing(device):
            return None
        name = os.environ.get("XINFER_GEMM_TUNING_FILE")
        path = Path(name).expanduser() if name else None
        key = None
        if path:
            key = json.dumps(("attention", _versions(device, library), signature), separators=(",", ":"),
                             default=lambda v: v.type if isinstance(v, torch.device) else str(v))
            choice = gemm_tuning._read(path).get(key) if path.exists() else None
            if choice in ("kernels", library):
                _winners[signature] = choice
                return choice
        if (os.environ.get("XINFER_GEMM_TUNING_READONLY") == "1"
                or (path and path.exists() and path.stat().st_mode & 0o222 == 0)):
            return None
        winner = _compare(device, native, external, library, 0.02 if op == "gated_delta_net" else 0.01)
        if winner is not None and path:
            winner = gemm_tuning._store(path, key, winner)
        _winners[signature] = winner or "kernels"
        return _winners[signature]

"""Compare the split-KV kernel and FlexAttention on an eager, unmeasured KV-cache shape."""

from __future__ import annotations

import logging
from statistics import median
from threading import RLock

import torch

from xinfer.ops.backends.cuda._attention_libraries import layout
from xinfer.ops.backends.cuda._native import capturing

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


def _compare(device, native, external):
    stream = torch.cuda.current_stream(device)
    try:
        expected = external()
    except torch.OutOfMemoryError:
        raise
    except (ImportError, RuntimeError, ValueError, TypeError) as error:
        stream.synchronize()
        logger.info("FlexAttention declined the KV-cache warmup: %s", error)
        return None
    try:
        actual = native()
    except torch.OutOfMemoryError:
        raise
    except (RuntimeError, ValueError) as error:
        stream.synchronize()
        logger.info("The native kernel declined the KV-cache warmup: %s", error)
        return "flex_attention"
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return "flex_attention"
    error = torch.linalg.vector_norm(actual.float() - expected.float())
    norm = torch.linalg.vector_norm(expected.float())
    if not bool(torch.isfinite(error) & torch.isfinite(norm) & ((error == 0) | (error < 0.01 * norm))):
        return "flex_attention"
    return "kernels" if _time(native, stream) <= _time(external, stream) else "flex_attention"


def choose(op, args, native, external):
    """Return None for static routing. Both candidates only read the KV caches and allocate their own output."""
    # Equal strides can still describe slices with different base alignment.
    alignments = tuple(t.data_ptr() % 16 for t in args if torch.is_tensor(t))
    signature = (op, layout(args), alignments)
    if signature in _winners:
        return _winners[signature]
    device = args[0].device
    with _lock, torch.cuda.device(device):
        if signature in _winners:
            return _winners[signature]
        if capturing(device):
            return None
        winner = _compare(device, native, external)
        _winners[signature] = winner
        return winner

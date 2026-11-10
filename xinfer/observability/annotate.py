"""Named host regions for profilers, and op tables that mark every operator call (docs/profiling.md).

``span(name, **args)`` marks a region of host code. The region goes to the sinks chosen once by ``configure``,
at startup; with none, the default, ``span`` returns a shared no-op context and nothing is decided per call.

A sink is a callable ``sink(name, args)`` returning a context manager entered around the region, so new sinks
(an in-process recorder) need no change at the call sites. Built in:

- ``kineto``: ``torch.profiler.record_function``, a user annotation in torch profiler (kineto) traces;
- ``nvtx``: an NVTX range for Nsight Systems; needs a CUDA build of torch.

Both name the region ``label(name, args)``, which carries the arguments, so one parser reads either trace.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from collections.abc import Callable, Mapping
from typing import ContextManager

import torch

Sink = Callable[[str, Mapping[str, object]], ContextManager]

_NO_SPAN = contextlib.nullcontext()
_POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _no_span(name: str, args: Mapping[str, object]) -> ContextManager:
    """The sink when annotation is off: the shared no-op context."""
    return _NO_SPAN


_sink: Sink = _no_span


def span(name: str, /, **args) -> ContextManager:
    """A context manager marking the host region it encloses.

    Args:
        name: region name, for example ``"op::linear"``.
        **args: compact values (str, int, float or bool) describing the region, never tensor data.

    Returns:
        The context manager of the configured sinks.
    """
    return _sink(name, args)


class _Spans:
    """Enters several sinks' contexts in order and exits them in reverse."""

    __slots__ = ("contexts",)

    def __init__(self, contexts: list):
        self.contexts = contexts

    def __enter__(self):
        for context in self.contexts:
            context.__enter__()

    def __exit__(self, *exc):
        for context in reversed(self.contexts):
            context.__exit__(*exc)


def _all(sinks: tuple[Sink, ...], name: str, args: Mapping[str, object]) -> ContextManager:
    """The contexts of several sinks, entered together."""
    return _Spans([sink(name, args) for sink in sinks])


def configure(*sinks: Sink) -> None:
    """Choose the sinks of every later ``span``; call at startup. No sinks turns annotation off."""
    global _sink
    _sink = functools.partial(_all, sinks) if len(sinks) > 1 else sinks[0] if sinks else _no_span


def label(name: str, args: Mapping[str, object]) -> str:
    """``name`` followed by `` key=value`` per argument, e.g. ``"op::linear x=bfloat16[8,1024] weight=..."``."""
    return " ".join([name, *(f"{key}={value}" for key, value in args.items())])


def kineto(name: str, args: Mapping[str, object]) -> ContextManager:
    """Sink: a ``torch.profiler.record_function`` named ``label(name, args)``."""
    return torch.profiler.record_function(label(name, args))


class _NvtxRange:
    """An NVTX range pushed on enter and popped on exit."""

    __slots__ = ("message",)

    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        torch.cuda.nvtx.range_push(self.message)

    def __exit__(self, *exc):
        torch.cuda.nvtx.range_pop()


def nvtx(name: str, args: Mapping[str, object]) -> ContextManager:
    """Sink: an NVTX range (``torch.cuda.nvtx``) named ``label(name, args)``; needs a CUDA build of torch."""
    return _NvtxRange(label(name, args))


def _describe(value) -> object:
    """A tensor as ``dtype[d0,d1,...]``, a sequence as ``[a,b,...]`` of its items described, anything else as is."""
    if isinstance(value, torch.Tensor):
        return f"{str(value.dtype).removeprefix('torch.')}[{','.join(map(str, value.shape))}]"
    if isinstance(value, (list, tuple)):
        return f"[{','.join(str(_describe(item)) for item in value)}]"
    return value


def parameter_names(function: Callable) -> tuple[str, ...]:
    """Names of ``function``'s positional parameters, for ``call_args``; empty when it has no Python signature."""
    try:
        return tuple(p.name for p in inspect.signature(function).parameters.values() if p.kind in _POSITIONAL)
    except (TypeError, ValueError):
        return ()


def call_args(params: tuple[str, ...], args: tuple, kwargs: dict) -> dict:
    """Name every argument of a call (by ``params``, else ``arg<i>``) and describe it; None is omitted."""
    named = {(params[i] if i < len(params) else f"arg{i}"): value for i, value in enumerate(args)}
    named.update(kwargs)
    return {key: _describe(value) for key, value in named.items() if value is not None}


def _profiled(implementation) -> Callable:
    """``implementation.function`` called inside ``span("op::<op>", **call arguments)``."""
    function, name = implementation.function, f"op::{implementation.op}"
    params = parameter_names(function)

    @functools.wraps(function)
    def profiled(*args, **kwargs):
        with span(name, **call_args(params, args, kwargs)):
            return function(*args, **kwargs)

    return profiled


def profiled_ops(table):
    """``table`` (an ``xinfer.ops.OpTable``) with every operator called inside a span.

    Each call runs in ``span(f"op::{name}", **args)`` with ``args`` the call's arguments by parameter name: tensors
    as ``dtype[shape]``, host sequences as ``[a,b]``, other values as they are, None omitted. Build a model inside
    ``use_ops(profiled_ops(table))`` to profile it; ``table`` itself is unchanged and binds the selected functions
    directly.
    """
    return table.wrapped(_profiled)

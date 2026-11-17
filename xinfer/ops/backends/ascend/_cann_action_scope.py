"""Reuse immutable attention metadata within one synchronous action forward."""

from contextlib import contextmanager
from contextvars import ContextVar

_metadata = ContextVar("cann_action_metadata", default=None)


@contextmanager
def action_scope():
    """Share derived metadata for one action forward on the current stream.

    Prefix, mask, and rotary inputs must remain unchanged within the scope.
    Each forward, including a new graph capture, starts with an empty cache.
    """
    cache = {}
    token = _metadata.set(cache)
    try:
        yield
    finally:
        cache.clear()
        _metadata.reset(token)


def memo_action_metadata(key, inputs, create):
    cache = _metadata.get()
    if cache is None:
        return create()
    identity = (*key, *(id(tensor) for tensor in inputs))
    if identity not in cache:
        # Keep originals alive so their ids cannot be reused within the scope.
        cache[identity] = (inputs, create())
    return cache[identity][1]

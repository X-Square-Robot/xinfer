"""Operator fixtures shared by the model integration tests."""

from functools import wraps

import pytest

from xinfer import ops
from xinfer.platforms import resolve_platform


@pytest.fixture
def split_vision_ops():
    registry = ops.OpRegistry()
    for name in ops.registry.names():
        registry.define(ops.registry.spec(name))
    reference = registry.spec("qkv_rope_split").reference

    @wraps(reference)
    def split(*args, **kwargs):
        return tuple(t.contiguous() for t in reference(*args, **kwargs))

    registry.add("qkv_rope_split", split, backend="test-vision", platforms=("cpu",),
                 contract=ops.Contract(numerics="bitwise"))
    return registry.build_table(resolve_platform("cpu"), backends={"qkv_rope_split": "test-vision"},
                                reference_only=True)

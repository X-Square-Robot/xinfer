"""Explicit vendor alternatives report the TorchNPU build they call, not the custom kernel hash."""

import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

from xinfer import ops
from xinfer.ops.backends.ascend import _support
from xinfer.ops.registry import OpRegistry


@pytest.mark.parametrize("category", ["cann", "official"])
def test_vendor_registration_version_is_independent_of_the_native_library(monkeypatch, category):
    package = ModuleType("torch_npu")
    package.__version__ = "2.10.test-vendor"

    def npu_linear(*args):
        pytest.fail("registration must not execute a vendor operator")

    def npu_rms_norm(*args):
        pytest.fail("registration must not execute a vendor operator")

    package.npu_linear, package.npu_rms_norm = npu_linear, npu_rms_norm
    registry = OpRegistry()
    for name in ops.registry.names():
        registry.define(ops.registry.spec(name))
    monkeypatch.setattr(importlib.import_module("xinfer.ops.registry"), "registry", registry)
    monkeypatch.setitem(sys.modules, "torch_npu", package)
    monkeypatch.setattr(_support, "VERSION", "xinfer_kernels_ascend.so sha256:unrelated")
    path = Path(__file__).parents[2] / "xinfer/ops/backends/ascend" / f"{category}.py"
    spec = importlib.util.spec_from_file_location(f"_vendor_identity_{category}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    implementations = [entry for name in registry.names() for entry in registry.implementations(name)
                       if entry.backend == "ascend"]
    assert implementations
    assert all(entry.version == "torch_npu 2.10.test-vendor" and entry.priority == -5
               for entry in implementations)
    assert {entry.library for entry in implementations} == (
        {None, "npu_linear", "npu_rms_norm"} if category == "cann" else {"torch_npu"})

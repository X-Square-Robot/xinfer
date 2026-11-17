"""Ascend native and JIT loading, launch errors, and benchmark provenance."""

import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from xinfer import ops
from xinfer.ops.backends.ascend import _native, _support
from xinfer.ops.backends.ascend.src.gemm import g2112_tilelang
from xinfer.ops.registry import OpRegistry

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import performance  # noqa: E402


SOURCE = Path(__file__).parents[2] / "xinfer/ops/backends/ascend/_native.py"


def _load():
    spec = importlib.util.spec_from_file_location("_ascend_loader_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_missing_library_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("XINFER_ASCEND_KERNELS", "1")
    monkeypatch.setenv("XINFER_ASCEND_KERNELS_LIBRARY", str(tmp_path / "missing.so"))
    with pytest.raises(FileNotFoundError, match="XINFER_ASCEND_KERNELS_LIBRARY"):
        _load()


def test_disabled_loader_ignores_explicit_path(monkeypatch, tmp_path):
    monkeypatch.setenv("XINFER_ASCEND_KERNELS", "0")
    monkeypatch.setenv("XINFER_ASCEND_KERNELS_LIBRARY", str(tmp_path / "missing.so"))
    native = _load()
    assert native.lib is None
    assert native.core_library() is None
    assert native.NATIVE["requires"] is None


def test_typed_and_raw_adapters_share_one_library(monkeypatch, tmp_path):
    path = tmp_path / "kernels.so"
    path.write_bytes(b"native-test")
    typed = SimpleNamespace(build_target_archs=lambda: "ascend950dt9582")
    monkeypatch.setenv("XINFER_ASCEND_KERNELS", "1")
    monkeypatch.setenv("XINFER_ASCEND_KERNELS_LIBRARY", str(path))
    monkeypatch.setitem(sys.modules, "tvm_ffi", SimpleNamespace(load_module=lambda filename: typed))
    native = _load()
    loaded = []
    raw = object()
    monkeypatch.setattr(native.ctypes, "CDLL", lambda filename: loaded.append(filename) or raw)
    assert native.core_library() is raw
    assert native.core_library() is raw
    assert loaded == [str(path.resolve())]
    assert native.lib is typed
    assert native.NATIVE["platforms"] == ("ascend950dt9582",)
    assert hashlib.sha256(path.read_bytes()).hexdigest()[:12] in native.NATIVE["version"]


def test_broken_explicit_library_does_not_select_fallback(monkeypatch, tmp_path):
    path = tmp_path / "broken.so"
    path.write_bytes(b"invalid")
    monkeypatch.setenv("XINFER_ASCEND_KERNELS", "1")
    monkeypatch.setenv("XINFER_ASCEND_KERNELS_LIBRARY", str(path))

    def fail(filename):
        raise RuntimeError("invalid native ABI")

    monkeypatch.setitem(sys.modules, "tvm_ffi", SimpleNamespace(load_module=fail))
    with pytest.raises(RuntimeError, match="invalid native ABI"):
        _load()


def test_explicit_library_requires_tvm_ffi(monkeypatch, tmp_path):
    path = tmp_path / "kernels.so"
    path.touch()
    monkeypatch.setenv("XINFER_ASCEND_KERNELS", "1")
    monkeypatch.setenv("XINFER_ASCEND_KERNELS_LIBRARY", str(path))
    monkeypatch.setitem(sys.modules, "tvm_ffi", None)
    with pytest.raises(ModuleNotFoundError, match="tvm_ffi"):
        _load()


@pytest.mark.parametrize("name", ["rope_d64", "qkv_split_vec"])
def test_unbuilt_layout_declines_before_launch(monkeypatch, name):
    module = importlib.import_module(f"xinfer.ops.backends.ascend.{name}")
    monkeypatch.setattr(module, "_CALL", None)
    monkeypatch.setattr(module._native, "core_library", lambda: SimpleNamespace())
    with pytest.raises(ValueError, match="not included in this native build"):
        module._library()


def test_typed_launcher_value_error_is_an_execution_error(monkeypatch):
    def fail(*args):
        raise ValueError("selected launch failed")

    monkeypatch.setattr(_native, "_FFI_FACES", {"selected": fail})
    with pytest.raises(RuntimeError, match="selected launch failed") as error:
        _native.launch_kernel("selected", (), ())
    assert isinstance(error.value.__cause__, ValueError)


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


@pytest.mark.parametrize("available", [False, True])
def test_npu_environment_records_jit_source_and_toolchain_availability(monkeypatch, tmp_path, available):
    def version(name):
        if name == "tilelang":
            if not available:
                raise importlib.metadata.PackageNotFoundError(name)
            return "test-ascend-fork"
        return "test-torch-npu"

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace() if available else None)
    monkeypatch.setattr(_native, "LIBRARY_PATH", None)
    monkeypatch.setattr(performance, "HERE", tmp_path)
    monkeypatch.setenv("BISHENG_HOME", "/compiler")
    monkeypatch.setenv("TILELANG_CACHE_DIR", "/cache")
    identity = performance.environment(True)["tilelang"]
    assert identity["version"] == ("test-ascend-fork" if available else None)
    assert identity["ascend_available"] is available
    from xinfer.ops.backends.ascend import cann_projection

    assert cann_projection.accumulator_route is g2112_tilelang.accumulator_route
    actual_source = Path(inspect.getsourcefile(inspect.unwrap(cann_projection.accumulator_route)))
    assert Path(identity["jit_source"]).resolve() == actual_source.resolve()
    assert identity["jit_source_sha256"] == hashlib.sha256(actual_source.read_bytes()).hexdigest()
    assert identity["bisheng_home"] == "/compiler" and identity["cache_dir"] == "/cache"

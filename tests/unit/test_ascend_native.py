"""Native-loader behavior without device initialization or vendor libraries."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("name", ["rope_d64", "qkv_split_vec"])
def test_unbuilt_layout_declines_before_launch(monkeypatch, name):
    import importlib

    module = importlib.import_module(f"xinfer.ops.backends.ascend.{name}")
    monkeypatch.setattr(module, "_CALL", None)
    monkeypatch.setattr(module._native, "core_library", lambda: SimpleNamespace())
    with pytest.raises(ValueError, match="not included in this native build"):
        module._library()


def test_untuned_qkv_layout_uses_reference():
    import torch

    from xinfer.ops.backends.ascend import gemm
    from xinfer.ops.reference.gemm import qkv_norm_split

    packed = torch.randn(1, 3, 12)
    q_weight, k_weight = torch.randn(2), torch.randn(2)
    expected = qkv_norm_split(packed, q_weight, k_weight, 2, 1, 2, 1e-6)
    actual = gemm.qkv_norm_split(packed, q_weight, k_weight, 2, 1, 2, 1e-6)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)

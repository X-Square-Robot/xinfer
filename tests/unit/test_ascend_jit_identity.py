"""The Ascend JIT facade reuses its shipped source and benchmark identity names that source."""

import hashlib
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from xinfer.ops.backends.ascend import _native, g2112_tilelang
from xinfer.ops.backends.ascend.src.gemm import g2112_tilelang as source

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import performance  # noqa: E402


def test_jit_facade_keeps_the_original_routes_and_globals():
    for name in g2112_tilelang.__all__:
        assert getattr(g2112_tilelang, name) is getattr(source, name)
    active = inspect.unwrap(g2112_tilelang.accumulator_route)
    assert Path(inspect.getsourcefile(active)).resolve() == Path(source.__file__).resolve()


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
    actual_source = Path(inspect.getsourcefile(inspect.unwrap(g2112_tilelang.accumulator_route)))
    assert Path(identity["jit_source"]).resolve() == actual_source.resolve()
    assert identity["jit_source_sha256"] == hashlib.sha256(actual_source.read_bytes()).hexdigest()
    assert identity["bisheng_home"] == "/compiler" and identity["cache_dir"] == "/cache"

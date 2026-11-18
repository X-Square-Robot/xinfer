"""Ascend measurement recipe and kernel provenance checks."""

import hashlib
import importlib.metadata
import importlib.util
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
import performance  # noqa: E402


def _exported(recipe, name):
    for line in recipe.read_text(encoding="utf-8").splitlines():
        fields = shlex.split(line)
        if len(fields) == 2 and fields[0] == "export":
            key, _, value = fields[1].partition("=")
            if key == name:
                return value
    raise AssertionError(f"{name} is missing from {recipe}")


@pytest.mark.parametrize(("module_name", "variable"), [
    ("copykill_r2", "XINFER_COPYKILL_R2"), ("rmsfuse_r4", "XINFER_RMSFUSE_R4"),
])
def test_explicit_recipe_families_match_the_measured_all_selector(monkeypatch, module_name, variable):
    def selected(value):
        monkeypatch.setenv(variable, value)
        spec = importlib.util.spec_from_file_location(
            f"_recipe_{module_name}", ROOT / "xinfer/ops/backends/ascend" / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {family for family in module._FAMILIES if module.enabled(family)}

    explicit = _exported(ROOT / "examples/ascend/wall_oss.env", variable)
    assert explicit != "all"
    assert selected(explicit) == selected("all")


@pytest.mark.parametrize("available", [False, True])
def test_npu_environment_identifies_the_in_tree_aot_library_and_jit_source(monkeypatch, tmp_path, available):
    from xinfer.ops.backends.ascend import _native

    installed_version = importlib.metadata.version
    find_spec = importlib.util.find_spec

    def version(name):
        if name == "tilelang":
            if not available:
                raise importlib.metadata.PackageNotFoundError(name)
            return "test-ascend-fork"
        return "test-torch-npu" if name == "torch-npu" else installed_version(name)

    def compiler_spec(name):
        if name == "tilelang.ascend.torch_exchange":
            return object() if available else None
        return find_spec(name)

    library = tmp_path / "xinfer_kernels_ascend.so"
    library.write_bytes(b"native-library")
    monkeypatch.setattr(_native, "LIBRARY_PATH", str(library))
    monkeypatch.setenv("XINFER_XCOMPUTE_LIBRARY", str(tmp_path / "retired.so"))
    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(importlib.util, "find_spec", compiler_spec)
    monkeypatch.setattr(performance, "HERE", tmp_path)
    monkeypatch.setenv("BISHENG_HOME", "/compiler")
    monkeypatch.setenv("TILELANG_CACHE_DIR", "/cache")

    result = performance.environment(True)
    assert result["kernel_library"] == str(library)
    assert result["kernel_library_sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
    assert result["torch_npu"] == "test-torch-npu"
    identity = result["tilelang"]
    actual_source = ROOT / "xinfer/ops/backends/ascend/src/gemm/g2112_tilelang.py"
    assert identity["version"] == ("test-ascend-fork" if available else None)
    assert identity["ascend_available"] is available
    assert Path(identity["jit_source"]).resolve() == actual_source.resolve()
    assert identity["jit_source_sha256"] == hashlib.sha256(actual_source.read_bytes()).hexdigest()
    assert identity["bisheng_home"] == "/compiler" and identity["cache_dir"] == "/cache"

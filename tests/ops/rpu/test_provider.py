# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""CPU loader/asset admission contracts; fake modules do not qualify RPU execution."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from xinfer.ops.backends import rpu
from xinfer.ops.backends.rpu import _assets, _native

CATALOG = "a" * 64


class NativeModule:
    def __init__(self):
        self.functions = {
            "rpu_abi_version": lambda: 11,
            "compiled_backends": lambda: "rpu",
            "rpu_catalog_sha256": lambda: CATALOG,
            "rpu_runtime_available": lambda: False,
            "rpu_kernel_names": lambda: ["fill_f16"],
        }

    def get_function(self, name):
        if name not in self.functions:
            raise AttributeError(name)
        return self.functions[name]


@pytest.fixture
def native(monkeypatch, tmp_path):
    module = NativeModule()
    library = tmp_path / "custom-build.so"
    library.write_bytes(b"fake native module for loader contract")
    calls = []
    monkeypatch.setenv("XINFER_RPU_LIBRARY", str(library))
    monkeypatch.setattr(_native, "_loaded", None)
    monkeypatch.setattr(_native.lib, "_functions", {})
    monkeypatch.setitem(sys.modules, "tvm_ffi", SimpleNamespace(
        load_module=lambda path: calls.append(path) or module))
    return SimpleNamespace(module=module, library=library, calls=calls)


@pytest.fixture
def assets(monkeypatch, tmp_path):
    directory = tmp_path / "assets"
    directory.mkdir()
    reference = directory / "fill_f16.ref"
    reference.write_bytes(b"external REF fixture, not executable")
    data = {
        "abi_version": 11,
        "catalog_sha256": CATALOG,
        "assets": {"fill_f16": {"sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}},
    }
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(data))
    monkeypatch.setenv("XINFER_RPU_ASSETS", str(directory))
    return SimpleNamespace(directory=directory, reference=reference, manifest=manifest, data=data)


def test_imports_do_not_load_native_or_build_code():
    # The parent xinfer.ops package already imports Torch; isolate the backend's
    # own import contract from that existing parent-package behavior.
    script = """
import importlib.abc
import subprocess
import sys
import xinfer.ops
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('tvm_ffi'):
            raise AssertionError('backend import requested ' + fullname)
sys.meta_path.insert(0, BlockNative())
def unexpected(*args, **kwargs):
    raise AssertionError('backend import started a process')
subprocess.run = subprocess.Popen = unexpected
from xinfer.ops.backends import rpu
from xinfer.ops.backends.rpu import rpu_tp
assert 'xinfer.ops.backends.rpu._native' in sys.modules
assert rpu_tp.RpuTpContext.families()['activation']['function'] == (
    'gelu_tanh', 'gelu_erf', 'sigmoid', 'exp', 'silu', 'softplus')
assert rpu.RpuTpContext is rpu_tp.RpuTpContext
"""
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_loader_reports_actual_artifact_and_caches_native_load(native):
    provider = rpu.load_backend()
    assert provider.backend_factory is rpu.create_backend
    assert provider.provenance == {
        "library": "xinfer_rpu.so",
        "library_sha256": hashlib.sha256(native.library.read_bytes()).hexdigest(),
        "compiled_backends": ["rpu"],
        "abi_version": 11,
        "catalog_sha256": CATALOG,
        "runtime_available": False,
    }
    provider.provenance["compiled_backends"].append("modified")
    assert rpu.load_backend().provenance["compiled_backends"] == ["rpu"]
    assert native.calls == [str(native.library)]


def test_missing_library_is_explicit_without_native_import(monkeypatch, tmp_path):
    monkeypatch.setenv("XINFER_RPU_LIBRARY", str(tmp_path / "missing.so"))
    with pytest.raises(FileNotFoundError, match="Build xinfer_rpu.so explicitly"):
        rpu.load_backend()


@pytest.mark.parametrize("name,value,pattern", [
    ("rpu_abi_version", 10, "ABI mismatch"),
    ("compiled_backends", "cuda;rpu", "independent RPU"),
    ("rpu_catalog_sha256", "invalid", "invalid catalog SHA256"),
])
def test_incompatible_native_metadata_rejected(native, name, value, pattern):
    native.module.functions[name] = lambda: value
    with pytest.raises(RuntimeError, match=pattern):
        rpu.load_backend()


def test_missing_metadata_rejected(native):
    del native.module.functions["rpu_abi_version"]
    with pytest.raises(RuntimeError, match="required metadata export rpu_abi_version"):
        rpu.load_backend()


def test_native_load_failure_explains_sdk_requirement(native, monkeypatch):
    def fail(path):
        raise OSError("SDK dependency unavailable")

    monkeypatch.setattr(sys.modules["tvm_ffi"], "load_module", fail)
    with pytest.raises(RuntimeError, match="external Rhino Launch SDK"):
        rpu.load_backend()


def test_loaded_library_cannot_be_swapped(native, monkeypatch, tmp_path):
    rpu.load_backend()
    other = tmp_path / "other.so"
    other.write_bytes(b"other library")
    monkeypatch.setenv("XINFER_RPU_LIBRARY", str(other))
    with pytest.raises(RuntimeError, match="changed after native loading"):
        rpu.load_backend()


@pytest.mark.parametrize("factory", [rpu.create_backend, rpu.RpuTpContext, rpu.RpuContext])
def test_host_only_context_rejected_before_assets_or_runtime(native, monkeypatch, factory):
    monkeypatch.delenv("XINFER_RPU_ASSETS", raising=False)
    with pytest.raises(RuntimeError, match="host-only.*cannot execute"):
        factory()
    assert not any(name.startswith("rpu_tp_context") for name in native.module.functions)


def test_runtime_context_rejects_missing_assets_before_native_create(native, monkeypatch):
    native.module.functions["rpu_runtime_available"] = lambda: True
    created = []
    native.module.functions["rpu_tp_context_create"] = lambda *_: created.append(True) or 1
    monkeypatch.delenv("XINFER_RPU_ASSETS", raising=False)
    with pytest.raises(FileNotFoundError, match="XINFER_RPU_ASSETS"):
        rpu.create_backend()
    assert not created


def test_runtime_factory_validates_assets_without_implicit_warmup(native, assets):
    native.module.functions["rpu_runtime_available"] = lambda: True
    calls = []
    native.module.functions.update({
        "rpu_tp_context_create_with_pool": lambda *args: calls.append(("create", args)) or 23,
        "rpu_release": lambda handle: calls.append(("release", handle)),
    })
    with rpu.create_backend(allow_candidate=True, retained_segments=6144):
        assert calls == [("create", (True, 6144))]
    assert calls == [("create", (True, 6144)), ("release", 23)]


@pytest.mark.parametrize("factory,export", [
    (rpu.create_backend, "rpu_tp_context_create"),
    (rpu.RpuTpContext, "rpu_tp_context_create"),
    (rpu.RpuContext, "rpu_context_create"),
])
def test_context_records_validated_manifest_identity_as_independent_copies(native, assets, factory, export):
    native.module.functions["rpu_runtime_available"] = lambda: True
    native.module.functions[export] = lambda *_: 23
    native.module.functions["rpu_release"] = lambda _: None
    expected = {
        "manifest_sha256": hashlib.sha256(assets.manifest.read_bytes()).hexdigest(),
    }
    with factory() as context:
        assert context.asset_provenance == expected
        exposed = context.asset_provenance
        exposed["manifest_sha256"] = "b" * 64
        assert context.asset_provenance == expected
        # Later filesystem changes do not rewrite what this context validated.
        assets.manifest.write_text("changed after admission")
        assert context.asset_provenance == expected


@pytest.mark.parametrize("context_type,export", [
    (rpu.RpuTpContext, "rpu_tp_context_create"),
    (rpu.RpuContext, "rpu_context_create"),
])
def test_failed_native_creation_does_not_publish_asset_provenance(native, assets, context_type, export):
    native.module.functions["rpu_runtime_available"] = lambda: True

    def fail(*_):
        raise RuntimeError("native context creation failed")

    native.module.functions[export] = fail
    context = context_type.__new__(context_type)
    with pytest.raises(RuntimeError, match="native context creation failed"):
        context.__init__()
    assert not hasattr(context, "asset_provenance")


def test_valid_asset_manifest_is_bound_to_loaded_native(assets):
    metadata = {"abi_version": 11, "catalog_sha256": CATALOG}
    result = _assets.validate_assets(metadata, {"fill_f16"})
    assert result == {
        "manifest_sha256": hashlib.sha256(assets.manifest.read_bytes()).hexdigest(),
    }


def test_manifest_identity_hashes_the_bytes_that_were_validated(assets, monkeypatch):
    expected = hashlib.sha256(assets.manifest.read_bytes()).hexdigest()
    regular_file = _assets._regular_file

    def change_manifest_during_asset_validation(path):
        regular_file(path)
        if path == assets.reference:
            assets.manifest.write_text("different, unvalidated manifest bytes")

    monkeypatch.setattr(_assets, "_regular_file", change_manifest_during_asset_validation)
    result = _assets.validate_assets({"abi_version": 11, "catalog_sha256": CATALOG}, {"fill_f16"})
    assert result["manifest_sha256"] == expected


@pytest.mark.parametrize("mutation,pattern", [
    (lambda data: data.update(abi_version=10), "manifest ABI"),
    (lambda data: data.update(abi_version=True), "manifest ABI"),
    (lambda data: data.update(catalog_sha256="b" * 64), "catalog SHA256"),
    (lambda data: data.update(assets={}), "nonempty assets"),
    (lambda data: data["assets"]["fill_f16"].update(sha256="bad"), "invalid SHA256"),
    (lambda data: data["assets"]["fill_f16"].update(path="other.ref"), "file paths are fixed"),
    (lambda data: data["assets"].update(unknown={"sha256": "a" * 64}), "outside the native catalog"),
    (lambda data: data["assets"].update({"../escape": {"sha256": "a" * 64}}), "outside the native catalog"),
])
def test_asset_manifest_rejects_mismatch_and_untrusted_paths(assets, mutation, pattern):
    mutation(assets.data)
    assets.manifest.write_text(json.dumps(assets.data))
    with pytest.raises(ValueError, match=pattern):
        _assets.validate_assets({"abi_version": 11, "catalog_sha256": CATALOG}, {"fill_f16"})


@pytest.mark.parametrize("kind,exception,pattern", [
    ("changed", ValueError, "SHA256 mismatch"),
    ("empty", ValueError, "nonempty regular file"),
    ("missing", FileNotFoundError, "does not exist"),
    ("symlink", ValueError, "without symlinks"),
    ("undeclared", ValueError, "absent from the manifest"),
])
def test_asset_bytes_and_file_type_are_checked(assets, tmp_path, kind, exception, pattern):
    if kind == "changed":
        assets.reference.write_bytes(b"modified bytes")
    elif kind == "empty":
        assets.reference.write_bytes(b"")
    elif kind == "missing":
        assets.reference.unlink()
    elif kind == "symlink":
        outside = tmp_path / "outside.ref"
        outside.write_bytes(assets.reference.read_bytes())
        assets.reference.unlink()
        assets.reference.symlink_to(outside)
    else:
        (assets.directory / "unknown.ref").write_bytes(b"unmanifested")
    with pytest.raises(exception, match=pattern):
        _assets.validate_assets({"abi_version": 11, "catalog_sha256": CATALOG}, {"fill_f16"})


def test_manifest_symlink_and_duplicate_keys_are_rejected(assets, tmp_path):
    outside = tmp_path / "manifest.json"
    assets.manifest.rename(outside)
    assets.manifest.symlink_to(outside)
    with pytest.raises(ValueError, match="without symlinks"):
        _assets.validate_assets({"abi_version": 11, "catalog_sha256": CATALOG}, {"fill_f16"})
    assets.manifest.unlink()
    assets.manifest.write_text('{"abi_version":11,"abi_version":11}')
    with pytest.raises(ValueError, match="Duplicate.*abi_version"):
        _assets.validate_assets({"abi_version": 11, "catalog_sha256": CATALOG}, {"fill_f16"})


def test_selector_rejections_need_no_native_library(monkeypatch, tmp_path):
    monkeypatch.setenv("XINFER_RPU_LIBRARY", str(tmp_path / "missing.so"))
    with pytest.raises(TypeError, match="requires selector 'function'"):
        rpu.resolve("activation", {})
    with pytest.raises(ValueError, match="function must be one of"):
        rpu.resolve("activation", {}, function="approximate")
    with pytest.raises(ValueError, match="both as selector and integer"):
        rpu.resolve("activation", {}, integers={"function": 0}, function="gelu_tanh")
    with pytest.raises(TypeError, match="family entry points only"):
        rpu.resolve("gelu_f16", {}, function="gelu_tanh")


def test_native_host_only_resolution_in_subprocess():
    library = os.environ.get("XINFER_RPU_TEST_LIBRARY")
    if not library:
        pytest.skip("set XINFER_RPU_TEST_LIBRARY to an explicitly built host-only xinfer_rpu.so")
    script = """
from types import SimpleNamespace
from xinfer.ops.backends import rpu
provider = rpu.load_backend()
assert not provider.provenance['runtime_available']
t = SimpleNamespace(shape=(16, 16), layout='row_major', distribution='local')
entry, renamed, integers = rpu.resolve('activation', {'x': t, 'out': t}, function='gelu_tanh')
assert entry == 'gelu_f16', (entry, renamed, integers)
assert renamed == {'x': 'src', 'out': 'out'}
assert integers == {'mode': 1}
try:
    rpu.create_backend()
except RuntimeError as error:
    assert 'host-only' in str(error)
else:
    raise AssertionError('host-only library constructed a runtime context')
"""
    environment = {**os.environ, "XINFER_RPU_LIBRARY": str(Path(library).resolve())}
    result = subprocess.run([sys.executable, "-c", script], env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr

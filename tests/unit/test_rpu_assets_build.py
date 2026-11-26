# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""The external compiler boundary and external-asset publication are board-free."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location("build_rpu_assets", Path(__file__).resolve().parents[2]
                                            / "tools/build_rpu_assets.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / "src"
    catalog = root / "common/rpu/catalog.json"
    catalog.parent.mkdir(parents=True)
    (root / "op.rc").write_text("device source fixture\n")
    (root / "op.h").write_text("header fixture\n")
    catalog.write_text(json.dumps({"abi_version": 11, "kernels": [
        {"name": "op", "source": "op.rc", "dependencies": ["op.h"]}]}))
    return root


@pytest.fixture
def compiler(tmp_path):
    path = tmp_path / "site-compiler-wrapper"
    path.write_text("""#!/usr/bin/env python3
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('-O2', action='store_true', required=True)
parser.add_argument('-c', type=Path, required=True)
parser.add_argument('-o', type=Path, required=True)
parser.add_argument('-I', action='append', required=True)
args = parser.parse_args()
args.o.write_bytes(b'compiler-output-fixture:' + args.c.read_bytes())
""")
    path.chmod(0o700)
    return str(path)


def test_external_compilation_publishes_matching_hashes_without_machine_paths(sources, compiler, tmp_path):
    output = tmp_path / "assets"
    manifest = builder.build(sources, output, compiler)
    assert manifest == json.loads((output / "manifest.json").read_text())
    catalog_hash = hashlib.sha256((sources / "common/rpu/catalog.json").read_bytes()).hexdigest()
    assert manifest == {"abi_version": 11,
                        "catalog_sha256": catalog_hash,
                        "assets": {"op": {"sha256": hashlib.sha256((output / "op.ref").read_bytes()).hexdigest()}}}
    receipt = (output / "build.json").read_text()
    assert Path(compiler).name not in receipt
    assert json.loads(receipt)["compiler"] == {"sha256": hashlib.sha256(Path(compiler).read_bytes()).hexdigest()}
    assert str(tmp_path) not in receipt
    assert json.loads(receipt)["sources"] == {
        name: hashlib.sha256((sources / name).read_bytes()).hexdigest() for name in ("op.rc", "op.h")}


def test_failed_compilation_does_not_publish_assets(sources, compiler, tmp_path):
    Path(compiler).write_text("#!/usr/bin/env python3\nraise SystemExit(1)\n")
    output = tmp_path / "assets"
    with pytest.raises(subprocess.CalledProcessError):
        builder.build(sources, output, compiler)
    assert not output.exists()
    assert not list(tmp_path.glob(".xinfer-rpu-assets-*"))


@pytest.mark.parametrize("change", ["outside", "symlink", "disabled", "unknown_selection"])
def test_invalid_catalog_or_selection_is_rejected_before_compilation(sources, compiler, tmp_path, change):
    catalog = sources / "common/rpu/catalog.json"
    data = json.loads(catalog.read_text())
    selected = None
    if change == "outside":
        data["kernels"][0]["source"] = "../op.rc"
    elif change == "symlink":
        (sources / "op.rc").unlink()
        (sources / "op.rc").symlink_to(sources / "op.h")
    elif change == "disabled":
        data["kernels"][0]["execution_status"] = "disabled"
    else:
        selected = ["unknown"]
    catalog.write_text(json.dumps(data))
    output = tmp_path / "assets"
    with pytest.raises(ValueError):
        builder.build(sources, output, compiler, selected)
    assert not output.exists()


def test_nonempty_output_is_never_overwritten(sources, compiler, tmp_path):
    output = tmp_path / "assets"
    output.mkdir()
    existing = output / "existing.ref"
    existing.write_bytes(b"keep existing assets")
    with pytest.raises(ValueError, match="empty directory"):
        builder.build(sources, output, compiler)
    assert existing.read_bytes() == b"keep existing assets"


def test_source_mutation_during_build_is_not_published(sources, compiler, tmp_path, monkeypatch):
    original = builder.subprocess.run

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (sources / "op.h").write_text("concurrent change\n")
        return result

    monkeypatch.setattr(builder.subprocess, "run", mutate)
    output = tmp_path / "assets"
    with pytest.raises(ValueError, match="source changed"):
        builder.build(sources, output, compiler)
    assert not output.exists()


def test_catalog_sources_and_transitive_headers_form_a_complete_build_tree():
    root = builder.SOURCE_ROOT.resolve()
    catalog = json.loads((root / "common/rpu/catalog.json").read_text())
    sources = {entry["source"] for entry in catalog["kernels"]}
    assert sources == {path.relative_to(root).as_posix() for path in root.rglob("*.rc")}

    def headers(path, visited):
        for include in re.findall(r'^#include "([^"]+)"', path.read_text(), re.MULTILINE):
            if include == "__clang_rpu_builtin_vars.h":
                continue  # Public compiler header supplied by hxcc.
            dependency = (path.parent / include).resolve()
            assert dependency.is_relative_to(root) and dependency.is_file(), (path, include)
            if dependency not in visited:
                visited.add(dependency)
                headers(dependency, visited)
        return visited

    for entry in catalog["kernels"]:
        closure = headers(root / entry["source"], set())
        declared = {(root / name).resolve() for name in entry.get("dependencies", [])}
        assert closure <= declared, (entry["name"], closure - declared)

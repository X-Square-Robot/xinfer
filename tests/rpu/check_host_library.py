# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real resolver and metadata ABI without a device runtime."""

import hashlib
import json
from pathlib import Path
import sys

import tvm_ffi

library = tvm_ffi.load_module(sys.argv[1])
catalog = Path(sys.argv[2])
document = json.loads(catalog.read_bytes())
assert library.get_function("rpu_abi_version")() == document["abi_version"]
assert str(library.get_function("rpu_catalog_sha256")()) == hashlib.sha256(catalog.read_bytes()).hexdigest()
assert str(library.get_function("compiled_backends")()) == "rpu"
assert not library.get_function("rpu_runtime_available")()
assert list(library.get_function("rpu_kernel_names")()) == [entry["name"] for entry in document["kernels"]]
for name in ("rpu_context_create", "rpu_tp_context_create", "rpu_tp_prepare", "rpu_tp_execute"):
    try:
        library.get_function(name)
    except AttributeError:
        continue
    raise AssertionError(f"host-only library exposes runtime function {name}")
assert library.get_function("rpu_tp_is_family")("gemm")
assert not library.get_function("rpu_tp_is_family")("unknown")
print("PASS: metadata, catalog fingerprint, family resolver and runtime isolation")

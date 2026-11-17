#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Lower TileLang and compile PIC objects during the CMake build only."""

import argparse
import hashlib
import importlib
import json
import re
import time
from pathlib import Path

from abi import validate_packed_launch_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", default="dav-3510")
    parser.add_argument("--module", required=True)
    parser.add_argument(
        "--kernels", nargs="+", help="Compile only these explicit names"
    )
    parser.add_argument(
        "--symbol-prefix",
        default="xinfer_aot_",
        help="Exported launcher prefix (xinfer_aot_ for the unified branch; the "
        "v22 anchor spelling reproduces the delivery byte for byte -- see the S2 anchor notes)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.symbol_prefix

    # Build dependencies deliberately remain outside the runtime loader.
    import tilelang
    from tilelang.contrib.bisheng import compile_ascend
    from tilelang.env import TILELANG_TEMPLATE_PATH
    from tilelang.jit.adapter.wrapper import TLAscendSourceWrapper

    class AOTSourceWrapper(TLAscendSourceWrapper):
        def _pythonic_expr(self, expression):
            # The upstream host formatter drops IntImm dtypes. A dynamic
            # int64 grid mixed with literal bounds then fails C++ template
            # deduction for min/max. Grid arithmetic is integral; explicitly
            # widen only these host expressions, never the device program.
            expression = super()._pythonic_expr(expression)
            return re.sub(r"\b(min|max)\(", prefix + r"grid_\1(", expression)

    module = importlib.import_module(args.module)
    compile_options = list(getattr(module, "COMPILE_OPTIONS", ()))
    if not all(isinstance(option, str) for option in compile_options):
        raise ValueError("COMPILE_OPTIONS must contain compiler argument strings")
    selected = set(args.kernels) if args.kernels else None
    manifest = {
        "tilelang": tilelang.__version__,
        "arch": args.arch,
        "module": args.module,
        "module_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "kernels": {},
    }
    seen = set()
    for name, program in module.build_programs():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in seen:
            raise ValueError(f"Invalid or repeated kernel name: {name}")
        seen.add(name)
        if selected is not None and name not in selected:
            continue
        started = time.monotonic()
        print(f"AOT lowering {name}", flush=True)
        program = program.with_attr("global_symbol", f"{prefix}{name}_device")
        target = tilelang.tvm.target.Target("ascend")
        with target, tilelang.tvm.transform.PassContext(opt_level=3):
            lowered = tilelang.lower(
                program,
                target=target,
                enable_host_codegen=False,
                enable_device_compile=False,
            )
        launch_layouts = {
            global_var.name_hint: validate_packed_launch_layout(
                [
                    (parameter.name, str(parameter.dtype))
                    for parameter in function.params
                ],
                global_var.name_hint,
            )
            for global_var, function in lowered.device_mod.functions.items()
        }
        wrapper = AOTSourceWrapper(
            tilelang.tvm.IRModule({"main": program}),
            lowered.kernel_source,
            lowered.target,
            device_mod=lowered.device_mod,
            host_mod=lowered.host_mod,
        )
        # Keep the host helpers independent of libstdc++ concepts headers,
        # which this Bisheng release cannot parse in its device pass.
        source = (
            "#include <stdint.h>\n"
            "static inline int64_t " + prefix + "grid_min(int64_t a, int64_t b) "
            "{ return a < b ? a : b; }\n"
            "static inline int64_t " + prefix + "grid_max(int64_t a, int64_t b) "
            "{ return a > b ? a : b; }\n"
        ) + wrapper.lib_code
        for symbol in ("call", "init", "get_last_error"):
            source = re.sub(rf"\b{symbol}\b", f"{prefix}{name}_{symbol}", source)
        source = source.replace(
            "static char error_buf", "static thread_local char error_buf"
        )
        source_path = args.output / f"{name}.asc"
        object_path = args.output / f"{name}.o"
        source_path.write_text(source)
        (args.output / f"{name}.tir").write_text(str(lowered.device_mod))
        print(f"AOT compiling {name}: {source_path}", flush=True)
        binary = compile_ascend(
            source,
            target_format="o",
            npu_arch=args.arch,
            options=["-I" + TILELANG_TEMPLATE_PATH, *compile_options],
            path_target=object_path,
            verbose=True,
        )
        manifest["kernels"][name] = {
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "object_sha256": hashlib.sha256(binary).hexdigest(),
            "build_seconds": time.monotonic() - started,
            "bytes": len(binary),
            "compile_options": compile_options,
            "launcher_abi": re.search(
                rf"int {prefix}{name}_call\(([^)]*)\)", source
            ).group(1),
            "device_launch_layouts": launch_layouts,
        }
        print(f"AOT finished {name}: {len(binary)} object bytes", flush=True)
    # Hand-written AscendC (CCE dialect) kernels: compiled from static
    # sources declared by the module (STATIC_ASCENDC) with the same bisheng
    # options and packed-launch ABI as the TileLang-lowered kernels.
    for name in args.kernels or []:
        if name in seen:
            continue
        static_sources = dict(getattr(module, "STATIC_ASCENDC", {}))
        if name not in static_sources:
            continue
        started = time.monotonic()
        source = Path(static_sources[name]).read_text(encoding="utf-8")
        source_path = args.output / f"{name}.asc"
        object_path = args.output / f"{name}.o"
        source_path.write_text(source)
        print(f"AOT compiling static AscendC {name}: {source_path}", flush=True)
        # build_many.py copies a .tir for every kernel; static sources have
        # none, so record the provenance instead.
        (args.output / f"{name}.tir").write_text(
            f"static AscendC source: {static_sources[name]}" + chr(10), encoding="utf-8"
        )
        binary = compile_ascend(
            source,
            target_format="o",
            npu_arch=args.arch,
            options=["-I" + TILELANG_TEMPLATE_PATH, *compile_options],
            path_target=object_path,
            verbose=True,
        )
        seen.add(name)
        manifest["kernels"][name] = {
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "object_sha256": hashlib.sha256(bytes(binary)).hexdigest(),
            "build_seconds": time.monotonic() - started,
            "bytes": len(binary),
            "compile_options": compile_options,
            "origin": "static AscendC source",
            "launcher_abi": re.search(
                rf"int {prefix}{name}_call\(([^)]*)\)", source
            ).group(1),
        }
        print(f"AOT finished {name}: {len(binary)} object bytes", flush=True)
    if selected is not None and selected != set(manifest["kernels"]):
        raise ValueError(f"Unknown kernel names: {sorted(selected - seen)}")
    if not manifest["kernels"]:
        raise ValueError("The explicit build registry contains no kernels")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

# Third-party notices

XInfer is licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
These notices record upstream attributions and external proprietary boundaries
for the adapted WALL-OSS-1.0-Nano RPU framework sources and external RPU
runtime dependencies.

## RhinoForge WALL-OSS-1.0-Nano adaptation

The WALL-OSS-1.0-Nano checkpoint admission, CPU input processing and action geometry
under `xinfer/models/wall_oss_1_0_nano/rpu/` adapt the publicly available RhinoForge
Wall Qwen3.5 implementation at revision
`2535a27756486bdb139b06764975edbe00004d63`.

The upstream attribution is:

> RhinoForge
>
> Copyright 2026 HUIXI-AI
>
> This product includes software developed by HUIXI-AI.

The upstream components are licensed under Apache-2.0. The original license and
notice are preserved in [licenses/LICENSE.rhinoforge](licenses/LICENSE.rhinoforge)
and [licenses/NOTICE.rhinoforge](licenses/NOTICE.rhinoforge).
The adaptation integrates the checkpoint contracts, banded camera processing and
action conversion with XInfer's prepared requests and resident RPU execution.
The model documentation links to the upstream source files.

## Apache TVM FFI

[Apache TVM FFI](https://github.com/apache/tvm-ffi) is a separately installed
build and runtime dependency, distributed as `apache-tvm-ffi` under the Apache
License, Version 2.0. Its package retains its own license and notices.

## External proprietary components and assets

Real RPU execution requires the external **Rhino Launch SDK**
(`librhino_launch`) provided by **HUIXI-AI**, a compatible device environment,
and matching operator assets. The external **`hxcc`** compiler is needed to
compile device source into those assets. The host-only build requires neither
the SDK nor that compiler.

The SDK, compiler and generated operator binaries are not included in this
repository or its source and binary distributions. Their use and any separate
redistribution are governed by the terms supplied with those components.
Device-source availability does not grant rights to the SDK, compiler, model
weights or other externally supplied assets.

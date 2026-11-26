# Native RPU operators

English | [简体中文](rpu_ops.zh-CN.md)

The RPU operator backend lives in `xinfer/ops/backends/rpu/` and builds its
own `xinfer_rpu.so`. It includes host runtime and TVM-FFI bindings, a catalog
and family resolver, Python adapters, and device source. Its public Python
entry point is `from xinfer.ops.backends import rpu`.

## Distribution and dependencies

| Component | Source distribution | Required use |
|---|---|---|
| Host C++17 runtime, bindings, catalog and Python adapters | Included | Host resolution and device runtime integration |
| Device `.rc` source and its supporting headers | Included | Compile an operator-asset bundle with an externally supplied compiler |
| Generated `.ref` operator assets | Supplied separately | Real operator execution |
| Rhino Launch SDK and `hxcc` | Supplied separately by HUIXI-AI | SDK for runtime builds; compiler for device-source builds |
| Model checkpoints and request data | Supplied separately | Model-level evaluation |

The native library loads operator files from an external asset directory;
operator binaries are not embedded in the library. The host-only build needs
neither the SDK, a compiler for device source, nor a board. Runtime SDK setup,
operator-asset production and device execution are separate steps.
Third-party attribution and component terms are recorded in
[third-party notices](../THIRD_PARTY_NOTICES.md).

The device sources target the R1 Layer0 interface. Compatibility with later
hardware generations requires a separate port and validation.
Native RPU sources are distributed in the source archive. The Python wheel
contains the Python adapters, with native libraries and assets built separately.

## Host-only build

Use a C++17 compiler, CMake 3.18 or newer, and Python with Apache TVM FFI.
Run these commands from the repository root:

```bash
python -m pip install -e ".[kernels]"
cmake -S xinfer/ops/backends/rpu/src -B build/rpu-host \
  -DCMAKE_BUILD_TYPE=Release \
  -DXINFER_RPU_HOST_ONLY=ON -DXINFER_RPU_BUILD_TESTS=ON
cmake --build build/rpu-host --parallel
ctest --test-dir build/rpu-host --output-on-failure
export XINFER_RPU_LIBRARY="$PWD/build/rpu-host/lib/xinfer_rpu.so"
```

CMake uses the selected Python interpreter to locate TVM FFI. If needed, set
`-DPython_EXECUTABLE=/path/to/python` during configuration.

```python
from types import SimpleNamespace

from xinfer.ops.backends import rpu

backend = rpu.load_backend()
print(backend.provenance)
tensor = SimpleNamespace(shape=(16, 16), layout="row_major", distribution="local")
print(rpu.resolve("activation", {"x": tensor, "out": tensor}, function="gelu_tanh"))
```

The resolver example uses metadata, not device allocations, and returns
`('gelu_f16', {'x': 'src', 'out': 'out'}, {'mode': 1})`.

The provenance includes the logical library name and SHA-256, compiled backends,
ABI version, catalog SHA-256 and `runtime_available`. A host-only build reports
`runtime_available=False`: `rpu.resolve(...)` can resolve family calls, but
creating a device context fails explicitly. Host resolution does not execute
an operator or qualify its numerical behavior. A successfully created device
context exposes `asset_provenance`, a copy containing the SHA-256 of the exact manifest
bytes validated before allocation. Installation paths stay local.

## Runtime build

Install a compatible Rhino Launch SDK separately and point `RHINO_SDK_ROOT`
at its root. SDK dynamic libraries must be available to the system loader.

```bash
export RHINO_SDK_ROOT=/path/to/rhino-launch-sdk
cmake -S xinfer/ops/backends/rpu/src -B build/rpu-runtime \
  -DCMAKE_BUILD_TYPE=Release \
  -DXINFER_RPU_HOST_ONLY=OFF -DRHINO_SDK_ROOT="$RHINO_SDK_ROOT" \
  -DXINFER_RPU_BUILD_TESTS=ON
cmake --build build/rpu-runtime --parallel
export XINFER_RPU_LIBRARY="$PWD/build/rpu-runtime/lib/xinfer_rpu.so"
```

`rpu.create_backend(...)` constructs the low-level device context. It requires
the matching device environment and operator assets in addition to the runtime
library. Importing the Python backend does not build the native library or
initialize a device. Set library and asset selection before the first backend
load; missing libraries, incompatible ABI/catalog identity and invalid assets
are errors rather than a fallback to CPU or Torch operators.

## Compile and select operator assets

Configure the externally supplied `hxcc` executable. Use a
new or empty output directory outside the source tree:

```bash
export HXCC=/path/to/hxcc
python tools/build_rpu_assets.py --compiler "$HXCC" --output /path/to/rpu-assets
export XINFER_RPU_ASSETS=/path/to/rpu-assets
```

`--kernels name ...` limits compilation to named catalog entries. The driver
uses the standard `hxcc -O2 -c <source>.rc -o <name>.ref` interface with the
required includes and writes `manifest.json` after compiling the selected
assets. Generated files remain external to the source and binary packages.

The asset directory contains `manifest.json` and same-directory
`<catalog-entry>.ref` files. The manifest has this shape; the build driver
supplies the actual digest strings:

```json
{
  "abi_version": 11,
  "catalog_sha256": "<native catalog SHA-256>",
  "assets": {
    "<catalog-entry>": {"sha256": "<entry.ref SHA-256>"}
  }
}
```

The Python provider checks the ABI and native catalog identity, verifies the
declared file digests, and admits only native catalog entries. Each asset must
be a nonempty regular file in that directory; symbolic links and custom paths
are rejected. The native loader also checks the selected file when loading an
operator. Library, catalog, manifest and operator assets must belong to the
same supported build; a compiler success alone does not establish this match.

## Execution contracts

RPU uses the same `OpRegistry` and `build_table` as CPU, CUDA and Ascend. Its
`prepare_<family>` specifications select callables taking a native context,
resident tensors, integers, scalars and selectors, and returning a native
prepared operation. These signatures differ from Torch tensor-call signatures.
They have no Torch reference or fallback and report
`Contract(numerics="unqualified")`. Unsupported pins and `reference_only`
requests fail explicitly. An empty RPU selection fails when bound; preparing
an unselected operation fails explicitly. Selection does not qualify numerics.

`rpu.bind_ops(table)` returns an operation factory using the selected callables,
including any table wrappers. The matching framework integration injects it
alongside `load_backend().backend_factory`: model plans call `context.prepare`,
which calls the table-selected implementation and then native preparation.
Preparation does not execute or wait; batch execution and completion waits keep
their existing lifecycle. The operator package alone does not supply model or
runtime integration.

`rpu.resolve(op, tensors, integers=None, **selectors)` uses the library's family
resolver. Supported shapes, dtypes, layouts, alias rules and selector values
are defined by the included catalog and the family bindings. Resolve a family
call before preparing its execution; changing numerical behavior requires the
corresponding explicit selector.

The TP interface uses resident buffers and prepared operations. The caller
supplies tensor distribution, operation order and any required collective
steps. A partial projection output remains partial until the caller connects
its reduction. Candidate entries require `allow_candidate=True`; candidate
admission and symbol availability do not imply board acceptance.

## Validation boundary

Host-only CTest and Python tests check contracts, family resolution and loader
behavior without executing device operators. An SDK-stub compile checks host
integration against a test interface, not the real SDK or a board.

The GELU Repeat path uses standard scaled arithmetic. It applies the
1.5957691216 scale in the add stage before the multiply by the input, preserving
the original FP16 rounding order. Both add input scales are initialized to one.

HUIXI-AI reported bitwise agreement with the original implementation using
hxcc 1.1.0 and its R1 simulator for all 65,536 FP16 encodings in the standalone
tanh GELU path, and fused up-projection cases with 33, 225 and 417 rows.
Real-device validation must use the selected library and asset bundle.
Comparisons with Torch also measure approximation error against a different
reference. Device numerical tests are opt-in through `XINFER_RPU_RUN_BOARD=1`.
Run them with an exclusively allocated board and the selected SDK and assets.

Real SDK compilation, device-source compilation and board execution must be
validated separately with the chosen compiler, SDK, asset bundle and device.
Operator numerical alignment, prepared replay, failure recovery and model
precision/performance gates require actual hardware and inputs. This source
migration does not establish numerical or performance acceptance.

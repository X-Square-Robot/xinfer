# Operators

Layers never call a kernel directly. They call an operator by name through the op table that is active while they
are built (`active_ops().rms_norm`). Each operator is defined once with a torch reference. Backends register faster
implementations of it, and the table picks one per operator when the model is built.
[docs/design.md §14](../../docs/design.md#14-operators-and-the-kernel-repository) gives the design;
this page covers the code.

## Layout

```text
xinfer/ops/
  __init__.py      operator definitions (OpSpec list), the process-wide `registry`, warn_slow_fallbacks
  spec.py          OpSpec (name, signature, reference) and Contract (numerics, deterministic, graph_safe)
  registry.py      register, OpRegistry, OpTable, use_ops / active_ops
  reference/       torch implementations; they define the semantics and run on every platform
    __init__.py      basic operators: linear, norms, rotary, attention, ...
    gemm.py, norm.py, attn.py, flow.py  fused BF16 operators
    mxfp8.py         MXFP8 operators
    nvfp4.py         NVFP4 operators
  layout/          storage layouts of the low-precision formats (scale swizzle, weight row orders, rope table)
  backends/        one package per backend (hardware), one module per operator category
    cuda/            attn.py, gemm.py (with gemm_sm120.py, gemm_sm110.py), norm.py, flow.py, position.py
    ascend/          category adapters; _native.py loads its in-tree library, as cuda/_native.py does

```

## Core mechanism

```text
 define (ops/__init__.py)          register (backends/<hw>/<category>.py, at import)
 OpSpec(name, signature, reference)       @register(op, backend="cuda", requires=<library>)
            │                                         │  library missing (None) → not registered
            ▼                                         ▼
 ┌──────────────────────────── registry (process-wide OpRegistry) ───────────────────────────┐
 │  op → [reference (platforms "*", priority -1), cuda:flash_attn, cuda:kernels, ...]       │
 └──────────────────────────────────────────┬────────────────────────────────────────────────┘
                                            │ build_table(platform, backends=pins, reference_only)
                                            ▼
 OpTable: op → the selected implementation → report()
                                            │ use_ops(table)
                                            ▼
 layers: self._rms_norm = active_ops().rms_norm   (bound once, at construction)
```

### Terms

| Term | Meaning | Example |
|---|---|---|
| Operator | A name, a signature and a torch reference that defines the semantics | `rms_norm` |
| Backend | The hardware an implementation runs on; `reference` runs on every platform | `cuda`, `reference` |
| Library | The kernel library an implementation calls; `register(..., requires=lib)` records its name | `flash_attn`, `kernels` |
| Implementation name | `backend:library`, or the backend alone when it calls no library | `cuda:kernels` |
| Platform | The resolved device (`xinfer.platforms`); its name, device type and arch are matched against `platforms` | `cuda-sm120` (`cuda`, `sm120`) |
| Contract | What an implementation guarantees against the reference | `numerics="tolerance"`, `deterministic=False` |

### Registration

A backend module imports the libraries it calls the way it would import any library. It registers each
implementation at module level with `register`, which adds it to the process-wide `registry`:

```python
try:
    import flash_attn
    import flash_attn_2_cuda
except Exception:  # not installed, or no binary for this CUDA: the references run
    flash_attn = flash_attn_2_cuda = None


@register("varlen_attention", backend="cuda", requires=flash_attn)
def varlen_attention(q, k, v, cu_seqlens, *, scale=None):
    ...
```

- `requires=None` (the library did not import) registers nothing, so the operator runs on its next implementation,
  in the end the reference.
- `platforms` defaults to `(backend,)`, which matches every platform of that device type. Pass arch names to narrow
  it: each backend's `NATIVE` arguments contain the architectures its library was built for.
- `version` defaults to `"<library> <library.__version__>"`; it is printed in the selection table.
- `priority` (default 10) ranks implementations; the reference has -1. When two backends tie, the first registered
  wins.
- Contract fields go as keywords (`deterministic=False`); `numerics` defaults to `"tolerance"`.
- `OpRegistry.add` checks that the function takes the same parameters (names and kinds) as the reference, that the
  operator is defined, and that no other function is registered under the same name. Importing a module again
  (`importlib.reload`) replaces its implementations.

`registry.load_plugins()` imports the built-in backend modules (`BUILTIN_MODULES` in `registry.py`), then the modules
named by the `xinfer.ops` entry point group. The process-wide registry calls it before its first `build_table`.

### Selection

`registry.build_table(platform, backends=None, reference_only=False)` builds the `OpTable` once per model:

1. Keep the implementations whose `platforms` match the platform name, device type or arch (`"*"` matches all).
2. A pin (`backends={"rms_norm": "cuda:kernels"}`, or `--op-backends` on the command line) selects the
   highest-priority implementation whose backend, library or full name equals it; a pin that matches nothing is an
   error. `reference_only=True` selects the reference for every operator that is not pinned.
3. Otherwise the highest-priority implementation is selected.

The table binds the selected function itself, so a call costs nothing extra.

### Partial support

An implementation that supports only some calls of an operator (some shapes, some options) runs the others on the
reference itself. Decide once per call key and cache the answer, so the check runs on the first (eager) call of each
key and never under CUDA graph capture (the engine runs each graph key eagerly before capturing it):

```python
@register("prefix_kvcache_attention", backend="cuda", **NATIVE)
def prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, *, flash_mask=None, scale=None,
                             all_keep=False, return_bnsd=False):
    if not _kvcache_listed(q, k_cache, v_cache, prefix_len, new_pos, flash_mask, scale):  # cached per shape key
        return reference.prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, flash_mask=flash_mask,
                                                  scale=scale, all_keep=all_keep, return_bnsd=return_bnsd)
    ...
```

The selection table then names the implementation for every call of the operator, including those it hands to the
reference; say which calls those are in its module docstring.

### Reporting and warnings

- `OpTable.report()` gives one row per operator (`backend`, `library`, `version`, contract). `xinfer-serve`
  logs it at startup and writes it into the resolved configuration.
- `ops.warn_slow_fallbacks(...)` warns, on an accelerator, when operators the model calls run on a fused reference
  (`FUSED_OPS` in `ops/__init__.py`), which composes torch operations in place of a kernel.
- `XINFER_OPS_STRICT=1` turns that warning into an error (CI and benchmarks).
- `XINFER_KERNELS=0`, `XINFER_ASCEND_KERNELS=0` and `XINFER_FLASH_ATTN=0` leave their libraries out;
  `XINFER_<NAME>=0` skips the entry point `<name>`.

## Adding an operator

1. **Write the reference** in `reference/` (the category module, such as `gemm.py`, `norm.py`, `mxfp8.py` or `nvfp4.py`).
   It defines the semantics, including where values are rounded, so write the numerics down in its docstring
   (accumulation dtype, which intermediates are rounded to BF16). Storage formats that callers outside the operator
   rely on (scale layouts, weight row orders) go in `layout/`.

   ```python
   def gated_rms_norm(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
       """``rms_norm(x) · weight · silu(gate)`` with the roundings of the Qwen3.5 Gated DeltaNet output norm.

       The normalization runs in FP32 and is rounded to ``x.dtype``; the weight product stays in that dtype; the
       gate is applied in FP32 and the result rounded once more.
       """
       ...
   ```

2. **Define it** in the `OpSpec` list of `ops/__init__.py`. The signature is documentation (shapes, dtypes, notes);
   the parameters every implementation must take are the reference's.

   ```python
   OpSpec("gated_rms_norm", "gated_rms_norm(x[..., D], gate[..., D], weight[D], eps) -> Tensor",
          reference.gated_rms_norm),
   ```

3. **Call it from a layer** by name, bound at construction:

   ```python
   self._gated_rms_norm = active_ops().gated_rms_norm
   ```

4. **Test the reference** on the CPU (`tests/unit/`): against a plain torch composition or the source model's module,
   including the rounding the docstring promises.

The operator runs everywhere on its reference from here on. Backends add faster implementations without changing the
layer.

## Adding an implementation to an existing backend

1. Put it in the module of its category under `backends/<backend>/` (`cuda/attn.py`, `gemm.py`, `norm.py`, `flow.py`),
   or add a module for a new category and list it in `BUILTIN_MODULES`.
2. Import the library through that backend's `_native.py`. It owns the configured path, build identity and
   architecture checks, and supplies `NATIVE` registration arguments. Missing optional builds register nothing;
   explicit missing or broken libraries fail. Honour the backend's disable switch there.
3. Register it with the same parameters as the reference:

   ```python
   from xinfer.ops.backends.cuda._native import NATIVE
   from xinfer.ops.backends.cuda.norm import rmsnorm
   from xinfer.ops.registry import register


   @register("rms_norm", backend="cuda", **NATIVE)
   def rms_norm(x, weight, eps):
       return rmsnorm(x, weight, float(eps))
   ```

4. Declare its contract truthfully: `numerics="bitwise"` only when it equals the reference bit for bit on the same
   platform, `deterministic=False` when repeated calls may differ (for example split-K or stream-K reductions),
   `graph_safe=False` when it synchronizes with the host or allocates during capture.
5. Run the calls it does not support on the reference, deciding once per call key (see [Partial support](#partial-support)).
6. Test it against the reference on the hardware (`tests/unit/test_fused_ops.py`, `test_attention_ops.py` show the
   pattern), and check that the CPU test suite still passes without the library installed.

## Adding a backend (new hardware)

1. **Platform.** The backend name must be the device type of a platform, so the hardware needs a `Platform` with
   `capabilities.device_type` set to the backend name and `capabilities.arch` to the chip. `xinfer/platforms/detect.py`
   resolves `cuda`, `npu` and `cpu`; NPU platforms register with backend `ascend`. Add new devices there. The `xinfer.platforms` entry point group for external
   platforms is designed but not implemented yet ([docs/design.md §13](../../docs/design.md#13-hardware-platforms)).
2. **Package.** Create `backends/<backend>/` with one module per operator category, the same file names as `cuda/`
   where the operators are the same. Each module imports its libraries and registers with
   `backend="<backend>"`; narrow `platforms` to arch names when a build targets only some chips.
3. **Loading.** In-tree: add the modules to `BUILTIN_MODULES`. Out of tree (a vendor package): name each module in
   the `xinfer.ops` entry point group; importing it registers its implementations.

   ```toml
   [project.entry-points."xinfer.ops"]
   vendor_attn = "vendor_xinfer.attn"
   ```

4. **Native libraries.** Keep loading and build identity in the backend's `_native.py`, and pass its supported
   architectures and fingerprint to registrations. All adapters for that build use the same loaded object.
5. **Check the selection** on the device: `registry.build_table(platform).report()` should list the new backend for
   the operators it implements, and `XINFER_OPS_STRICT=1` should pass for the models it targets.

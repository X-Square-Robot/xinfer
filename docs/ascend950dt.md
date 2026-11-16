# Ascend 950DT

English | [简体中文](ascend950dt.zh-CN.md)

## Runtime

Install matching PyTorch and TorchNPU and source CANN before importing TorchNPU. Select the device with
`--device npu:0` or `Policy(device="npu:0")`. With `ASCEND_RT_VISIBLE_DEVICES`, the index is relative to the
visible set. The platform's backend name is `ascend`; its torch device is `npu:N`.

Wall-OSS uses the existing checkpoint and request formats. The service processes images on the CPU by default.
With PyACL DVPP V2 available, select device images using the top-level model option
`--model-options '{"image_preprocess_device":"gpu"}'`, or `Policy(image_preprocess_device="gpu")`.
`Policy` defaults to device images when supported. Do not put this option inside `processing`.
Unsupported JPEG profiles, including progressive JPEGs, log a warning and use host decoding with device
resize. Record the image path in precision and performance results.

JPEG decoders live in the platform modules. `platforms.devices.jpeg_decoder_for(device)` lazily selects a
decoder exposing `device`, `__call__(jpegs)` and `close()`. The platform owns vendor resources and stream
ordering. The engine preserves the input mapping, uploads raw frames and applies the common resize steps
to both raw and decoded images.

## External xcompute backend

Install a prebuilt Ascend xcompute shared library and the Python package from the same build. The loader in
`xinfer/ops/backends/_xcompute.py` disables rebuild-on-import and serves the Ascend backend's external builds
only: the CUDA backend builds its kernel library from this repository (`cuda/_native.py`,
[kernels.md](kernels.md)) and does not come through here. The Ascend external configuration:

| Variable | Meaning |
|---|---|
| `XINFER_XCOMPUTE_LIBRARY` | optional path to `xcompute.so`; otherwise the package locates its library |
| `XINFER_XCOMPUTE` | `0` disables xcompute implementations |

```bash
export PYTHONPATH=/path/to/xcompute-release/python:$PYTHONPATH
export XINFER_XCOMPUTE_LIBRARY=/path/to/xcompute-release/xcompute.so
```

Supported implementations are selected automatically. A missing library or entry leaves the operator on its
next available implementation. `--op-backends gemm_epilogue=ascend` pins an implementation; a pin without a
matching registration is an error. Partial implementations cache their capability decision per call key and
run unsupported calls on their torch fallback. Native execution errors propagate to the caller. Check the
startup selection table or `Policy.describe()["ops"]` for the selected backend, library and version. Selection
does not prove native coverage for every call shape.

The optional CANN bridge is built by xcompute with `-DBUILD_ASCEND_CANN_GDR=ON -DBUILD_ASCEND_CANN_INFERENCE=ON`.
The external package owns workspaces and FP32 epilogues. The validated environment uses CANN 9.1.1 and
TorchNPU 2.10.0.post4.

### Operator-compatible exports

A paired Ascend xcompute package may export any of the eight operator names in the table below as Python
callables directly in its module namespace. XInfer binds these exports before the existing CANN/legacy routes.
For example, `xcompute.gemm_epilogue` implements XInfer's `gemm_epilogue`, while the existing `xcompute.gemm`
entry keeps its own interface. Packages without these exports keep their existing routes. A package providing
all eight exports needs no CANN bridge imports from XInfer. Partial packages use bridges or torch fallbacks for
the other operators.

Exporting an operator name declares this contract:

- **Signature.** Match the parameters and positional/keyword kinds of the operator's reference in
  [`gemm.py`](../xinfer/ops/reference/gemm.py), [`attn.py`](../xinfer/ops/reference/attn.py) or
  [`norm.py`](../xinfer/ops/reference/norm.py). Parameter names and kinds are checked at load time. Raw FFI
  functions with output/workspace arguments need a Python adapter in xcompute; dynamic module attribute lookup
  is not used to discover operator exports. XInfer explicitly supplies every optional argument using its own
  defaults, so provider defaults do not affect these calls.
- **Numerics.** Preserve the reference's accumulation dtype and BF16 rounding points, including FP32 GEMM
  epilogues and RoPE before conversion, FP32 AdaRMS inputs to GLU, and the BF16 residual sum before FP32 RMSNorm.
  An implementation must pass operator conformance and the model's precision gate; an export alone is not
  numerical validation.
- **Returns and mutation.** Preserve output shape, dtype and tuple order. `prefix_kvcache_attention` returns
  `[B, M, H*D]`; `add_rms_norm` returns `(sum, normalized)`. `qkv_rope_kvcache` updates only the documented K/V
  cache slice in place and returns Q. Other inputs remain unchanged. The linked references define the complete
  shape and ownership contracts.
- **Execution.** Run asynchronously on the current stream and support both eager execution and graph
  capture/replay. Prepare reusable workspaces during the initial eager call. Tensor outputs may use a
  capture-safe allocator; capture must not synchronize the host or make untracked or unsafe allocations.
  Keep buffers alive through queued work and isolate mutable workspace across concurrent streams/lanes.
- **Partial support.** Reject unsupported layouts or options with `ValueError` before launching work or mutating
  inputs. Support decisions must depend on the cached call key: tensor metadata, scalar options and, for varlen
  attention, segment boundaries. XInfer caches the first eager decision and uses the operator's torch fallback
  for rejected keys. Execution failures propagate; a later failure of an admitted key is not retried on torch.

A kernel replacement or a larger supported shape domain that preserves this contract requires only a matching
xcompute package/library release. Bind it through the existing shared loader, restart, and rerun eager/graph
operator checks, lane smoke, precision and latency benchmarks. New operator semantics or signatures require a
separate operator-contract change.

### Wall-OSS operator list

Every operator Wall-OSS calls, except the no-op `prefetch_weights` and `join_prefetch`, selects `ascend:xcompute`
with the default wiring: eleven operator bindings and two operators that call them in turn, and a
per-layout dispatch audit of a full request pass confirmed every call layout admitted (no operator ran on a torch
reference). These are not eleven independent custom kernels: the table records the actual routes and compute
implementations in the measured build. Availability depends on the paired package and binary. Implementations must
support both eager execution and graph capture. Legacy eager-only raw GEMM, GLU and varlen entries are not
registered. Without operator-compatible exports or CANN bridges, these operators use their torch implementations
in both modes.

| Operator | Stage | Route in the measured build | Compute implementation |
|---|---|---|---|
| `gemm_epilogue` | vision, prefill, action | `CannProjection.gemm_epilogue` | CANN Mm + Torch epilogue |
| `gemm_glu_epilogue` | vision, prefill, action | `CannProjection.gemm_glu_epilogue` | CANN Mm + Torch FP32 SiLU-GLU |
| `qkv_rope_epilogue` | vision | `CannProjection.qkv_rope_epilogue` | CANN Mm + Torch bias/RoPE |
| `qkv_rope_kvcache` | prefill, action | `xcompute.qkv_proj_rope` | The unified projection entry with its K/V-cache epilogue; the CANN action bridge is the fallback for builds without it |
| `varlen_attention` | vision | `CannAttention.varlen_attention` | CANN FIA V5 (TND) |
| `causal_attention` | prefill | `CannAttention.causal_attention` | CANN FIA V5 (BNSD) + Torch mask/layout |
| `prefix_kvcache_attention` | action | `CannAttention.prefix_kvcache_attention` | CANN FIA V5 (BNSD) + Torch mask/layout |
| `add_rms_norm` | vision, prefill, action | `xcompute.add_rmsnorm` | Custom AOT kernels; the package entry serves every row count and takes the BF16 weight the vision stores |
| `gemm_add_rms_norm`, `mlp_add_rms_norm` | vision; prefill | the `gemm_epilogue`, `gemm_glu_epilogue` and `add_rms_norm` bindings above, in turn | the same calls as those operators |
| `rms_norm` | vision, prefill, action | `xcompute.rmsnorm` | Custom AOT kernels; the input norms of all three streams |
| `linear` | action | `xcompute.gemm` | The unified GEMM entry (CANN Mm at these shapes); leading dimensions are flattened for its rank-matching operands |
| `flow_velocity_euler` | action | `xcompute.denoise_output_euler` | Custom AOT kernels; the auto table's admitted envelope retried as the named `ascend_tilelang` domain |

Here CANN Mm is ACLNN Matmul, and FIA V5 is CANN FusedInferAttentionScore V5. The custom QKV bias/RoPE/cache
epilogue is restricted to the action shape with 16 tokens, hidden width 1024, 16 query heads, 2 KV heads,
head width 128 and cache append position 768, with compatible contiguous buffers. Other prompt widths and
prefill use Torch/TorchNPU epilogues; the matrix multiplication remains CANN in both cases. The latency
receipts do not record conditional-kernel hit counts or the share of time spent in CANN, custom kernels and
reference operations. An exported alternative entry does not establish that the measured model selected it.

`prefetch_weights` and `join_prefetch` (action) do nothing on the NPU (their references). An xcompute export named `qkv_rope_epilogue` must take the operator's `qk_order` keyword; the standard order serves both values.

Wall-OSS action uses ordinary RMSNorm, biased QKV and unmodulated GLU.
Switching binaries requires rerunning [precision](precision.md) and
[performance](performance.md#ascend-950dt) checks with the same requests and image path.

### Wall-X 2B operator list

Wall-X 2B declares twenty-one distinct operators across its three stages, and with the default wiring every one
of them selects `ascend:xcompute`: a per-layout dispatch audit of a full request pass confirmed all twenty-one
with every call layout admitted, so no operator runs on a torch reference. None of these bindings is a
hand-written kernel in this repository. The table below lists eighteen of the twenty-one because the other
three — `gemm_epilogue`, `gemm_glu_epilogue` and `qkv_rope_epilogue` — are the same entries the Wall-OSS table
above already describes, so repeating their routes would duplicate the routes rather than the operators.

| Operator | Stage | Entry | Notes |
|---|---|---|---|
| `layer_norm_cast` | vision | `xcompute.layernorm_inplace` | ViT layer norm; the entry overwrites its first tensor, so the binding clones — the caller keeps using its input as the residual |
| `segment_attention` | vision | `xcompute.attention_varlen` | Packed segments, no gate; the host segment layout with `max_seqlen`, planned and cached per grid list |
| `rot_pos` | vision | the `rot_pos` / `get_token_counts` library functions | 2-D rotary angles; host-planned grid tensors cached per (grids, merge, device), as the CUDA backend binds them |
| `pos_embed_interp` | vision | the `pos_embed_interp` library function | Bilinear position-table interpolation over the same cached grid tensors |
| `rope_full` | prefill | `xcompute.Rope.full` | Half-split partial rotary (`interleave=False`) |
| `zero_centered_rms_norm` | prefill | `xcompute.Rmsnorm(zero_centered=True)` | Scales by `1 + w`, the checkpoint's raw weight |
| `add_zero_centered_rms_norm` | prefill | `xcompute.AddRmsnorm(zero_centered=True)` | Returns `(sum, normalized)`, the entry's own order |
| `linear` | prefill | `xcompute.gemm` | The unified GEMM entry; leading dimensions are flattened for its rank-matching operands |
| `qkv_norm_split` | prefill | the `qwen35_vla_qkv_norm_replace_forward` library function | Every row a text row; the split is the kernel's outputs, the zero-centered norm its own |
| `gated_causal_attention` | prefill | `xcompute.attention_varlen(causal=True, out_gate=…)`, else the CANN attention bridge's own gated composition | Grouped keys/values are repeated to the query heads, as the reference's helper does; the bridge path serves the head sets the packed entry's white list does not |
| `gated_delta_net` | prefill | `xcompute.CannGatedDeltaNet` | Same parameters in the same order; the entry admits batch 1, 16 K heads and 128-wide heads |
| `qkv_norm_rope_kvcache` | action | `xcompute.qkv_proj_rope` | The unified projection entry with the `QkNorm(zero_centered=True)` and K/V-cache epilogues; this build sizes its workspace per stream |
| `modulated_rms_norm` | action | `xcompute.Adarmsnorm` | Modulation tables are FP32 `[B, 3K]` |
| `gated_residual_modulated_rms_norm` | action | `xcompute.Adarmsnorm(residual=…, branch_gate=…)` | The same entry, which returns `(output, gate, residual_output)` for this form |
| `identity_layer_norms_` | action | `xcompute.double_adarmsnorm_gated_residual_inplace` | Overwrites `residual` and `branch`; the post norm's gate is returned |
| `gated_kvcache_attention` | action | `prefix_kvcache_attention` + the BF16 gate | The reference's two steps, with the cache attention served natively |
| `flow_input_embed` | action | `xcompute.denoise_input_embed` | FP32 activations and a 3-D weight, as the flow head stores them; the auto table's refusal retried as the named `ascend_tilelang` domain |
| `flow_velocity_euler` | action | `xcompute.denoise_output_euler` | `dt` is keyword-only in the entry; the named `ascend_tilelang` domain covers the A-26 action width outside the auto table's measured envelope |

The official-operator and CANN bridge registrations live in `backends/ascend/official.py` and
`backends/ascend/cann.py` and are the only places the backend calls TorchNPU's own operators or the CANN bridge
rather than xcompute entries; both sit below the xcompute ones in priority, so none of them is selected without
an explicit pin.

The four attention operators also have a graph-safe torch composition (`backends/ascend/_torch_attn.py`), so a
layout outside the entries' white list still runs under capture instead of falling back to SDPA, which cannot be
captured. Both halves follow the same rule as Wall-OSS: the selectors are checked before the call, and a layout
the entries reject stays on the partial-support path rather than surfacing as a runtime error.

### CANN official operators (opt-in)

`xinfer/ops/backends/ascend/official.py` and `xinfer/ops/backends/ascend/cann.py` serve Wall-X 2B operators
from TorchNPU's own CANN entries rather than from xcompute. Both modules register below the xcompute entries in
priority, so **none of them is selected by default**: a deployment runs the xcompute entries, and reaches these
only by naming one explicitly.

| Operator | CANN entry | Note |
|---|---|---|
| `linear` | `torch_npu.npu_linear` | Flattens the leading dimensions |
| `qkv_norm_split` | `torch_npu.npu_rms_norm` | The split is views; the zero-centered norm is the official RMSNorm with `1 + w` |
| `gemm_epilogue` | `torch_npu.npu_linear` + FP32 epilogue | The epilogue stays in FP32, where the reference rounds |
| `gemm_glu_epilogue` | `torch_npu.npu_linear` + `silu·mul` | The AdaRMS input norm is the reference's |
| `qkv_norm_rope_kvcache` | `torch_npu.npu_linear` + the reference's norm/rope step | Takes the BF16 weight instead of casting it per call |
| `zero_centered_rms_norm` | `torch_npu.npu_rms_norm` with `1 + w` | The checkpoint stores the raw weight |
| `layer_norm_cast` | `F.layer_norm` | One ACLNN call |

Select one with the pin, for example:

```bash
xinfer-serve --model wallx2b --checkpoint /path --device npu:0 \
    --op-backends gemm_epilogue=ascend:npu_linear
```

The [performance table](performance.md) reports the default xcompute wiring; these pins are the alternative for
deployments that prefer the official operators.

## Graphs

Graph capture is enabled by default when the installed TorchNPU exposes its graph APIs, as on CUDA.
A request's first unseen shape is captured lazily; representative `--warmup` requests can capture the reachable
shape buckets, lanes and batch combinations before serving. Each capture first runs the shape eagerly on the
capture stream, then captures fixed buffers. After warmup, `CUDAGraphConfig.on_miss_after_warmup` applies to
new shapes: `"capture"` (the default) captures once and replays subsequent requests, `"eager"` runs them without
capture, and `"error"` rejects them. `--enforce-eager` or `Policy(cudagraphs=False)` disables graphs.

When no native implementation supports the call, `ascend:torch` attention uses the same FP32 matmul, softmax
and matmul composition in eager execution and capture. This avoids the SDPA side stream that can prevent
capture from closing. CPU/CUDA and explicit reference tables retain the original SDPA reference.
Without native operator pins, `--reference-ops` selects the original references; pair it with `--enforce-eager` on NPU, or use
`Policy(reference_ops=True, cudagraphs=False)`, because the reference attention may not support capture.

## Stream priorities

The engine passes `priority` through the same `Stream(device=..., priority=...)` interface on CUDA and NPU.
It uses the active runtime's stream-priority capability and reported numeric range. When that range is
known, the final stage's preferred priority −1 is bounded by it: a runtime with range 0–7 keeps the final stage
at 0. An unknown range keeps all defaults at 0. Explicit unsupported or out-of-range values are errors; a runtime
advertising priority support that changes a requested priority or cannot report it also fails lane creation.

TorchNPU 2.10.0.post4 and 2.10.0.post7 leave priority range queries and readback unimplemented, and their
managed stream pools create native streams at priority 0. These runtimes therefore advertise no stream-priority
capability and reject nonzero requests. CANN's native priority support does not establish that TorchNPU honors
its constructor argument; see the [post4 stream binding](https://github.com/Ascend/pytorch/blob/5dd8ef3f9b375b5ae4a83538d5785754148c3302/torch_npu/csrc/npu/Stream.cpp#L90-L100),
[post7 stream binding](https://github.com/Ascend/pytorch/blob/3363ea59d49f3e08cfd8d4a1f532b756ccda0682/torch_npu/csrc/npu/Stream.cpp#L103-L113)
and [post7 pool creation](https://github.com/Ascend/pytorch/blob/3363ea59d49f3e08cfd8d4a1f532b756ccda0682/torch_npu/csrc/core/npu/NPUStream.cpp#L254-L268).
The post7 `ExternalStream` wrapper cannot replace a lane stream: its documented restrictions include event
recording/waiting, allocator stream tracking and graph capture; see the [upstream contract](https://github.com/Ascend/pytorch/blob/3363ea59d49f3e08cfd8d4a1f532b756ccda0682/torch_npu/npu/streams.py#L112-L177).

## Validation

Run tests with one visible, exclusively reserved card. Compare fixed prepared inputs and explicit noise
between eager and graph execution, then exercise real requests through the WebSocket service. Check lazy
capture on a cold request without warmup and on a new shape after warmup, followed by replay of that shape. Run the
[precision gate](precision.md) and [benchmarks](benchmark.md), recording source and library identity,
composition, operator selection, image path, checkpoint identity and graph counters. Keep accelerator traces
and experiments outside the inference package. Measure request latency and robot capacity without a profiler.

Select the NPU with `benchmarks/performance.py --device npu:0`; put image settings in its existing
`--model-options`. For isolated backend or eager comparisons, use the existing options of `Policy` or
`xinfer-serve`.

The validated release also passed 24 concurrent requests using eight fixed-input variants in three submission
orders. Outputs were bitwise equal to sequential execution, and both lanes of every stage replayed their graphs.

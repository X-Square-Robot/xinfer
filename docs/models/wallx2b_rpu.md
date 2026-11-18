English | [简体中文](wallx2b_rpu.zh-CN.md)

# Wall-X 2B on RPU TP8

The `wallx2b` registry entry selects the wallx2b FP16 RPU profile when the
device is `rpu` or `rpu:0`. Other devices keep the GPU Wall-X 2B implementation
and its checkpoint format. The RPU profile has no CPU, BF16 or MXFP8 execution
fallback.

This profile targets Rhino RPU (R1) through Rhino Launch SDK. Install the SDK
and a compatible RPU xcompute build separately.

Execution is experimental and requires both explicit candidate admission and
a separately selected AOT xcompute library. Host-side model plans, processing
and lifecycle contracts are implemented; board-level numerical, ABI and
latency acceptance remains required before production use.

## Execution profile

| Property | Contract |
| --- | --- |
| Device | One RPU, physical cores 0–7; only `rpu:0` is accepted |
| Batching | B1, one admitted request, one lane per stage |
| Actions | FP32 physical relative26 `[1,16,26]`; eight denoising steps, without RTC |
| Images | Three canonical cameras with fixed resize, patch packing and M-RoPE rules |
| Execution | SDK-prepared ordered executions with explicit completion waits |
| Intermediates | Native resident buffers; the final output is copied to owned CPU storage |

`torch.device` does not describe this device, and the RPU path does not use a
CUDA graph, Torch stream/event or the generic CUDA `Engine`. Execution uses
the in-process `Policy` interface. The frozen-request format below describes
inputs for inspection, not an additional execution command; this profile does
not add an RPU mode to `xinfer-serve`.

The model/profile supplies the FP16 default used by `Policy` for configuration
and tuned-table lookup. The RPU
configuration adapter supplies the one-request and one-lane defaults; B1 comes
from the model and stream-priority support from the platform. `xinfer-profile load`
is unsupported because this engine does not emit the admission and stage spans
required by load profiling.

## Board-free checkpoint inspection

Install the model-processing dependencies, for example in an environment
provisioned with `python -m pip install -e ".[models]"`. Preflight checks
checkpoint metadata, configuration and asset hashes without loading tensors or
importing xcompute:

```python
from xinfer.models.wallx2b.rpu.config import preflight_checkpoint

manifest = preflight_checkpoint(checkpoint_path)
print(manifest)
```

For frozen-request inspection, provide a non-pickled NPZ with `face_view`,
`left_wrist_view` and `right_wrist_view` as RGB uint8 HWC arrays, `state` as
`[26]`, a Unicode scalar `instruction`, and `initial_noise` as FP32
`[1,16,26]`. Supply the actual noise array; a seed alone does not establish
identical input bytes. The request's views, state and explicit initial noise
are then handed to the same input processor the serving path uses.

Video decoding, task evaluation and ground-truth scoring are outside this
interface.

## Explicit candidate admission

Select an SDK-prepared AOT library explicitly before creating a real execution
context. The selected library must match the already imported xcompute Python
package and must provide the RPU ABI. Acquire the deployment's board lease and
check device occupancy first. Set `XCOMPUTE_LIBRARY_PATH` to the chosen library
(and disable auto-build with `XCOMPUTE_SKIP_AUTO_BUILD=1` when appropriate).

Execution is available through `Policy`. Its context manager closes the engine
and the input processor even when inference raises:

```python
from xinfer import Policy

with Policy("wallx2b", checkpoint, device="rpu:0", quantization="fp16",
            allow_candidate=True) as policy:
    result = policy.infer(observation)
```

Execution never enables candidate capabilities implicitly. A backend that does
not implement a selected family, selector, collective window or fused queue
fails admission instead of silently substituting a different execution plan.

Model plans prepare through xcompute family entry points such as `gemm`,
`rmsnorm`, `qkv_proj`, `gemm_glu` and `attention`. Family selectors pass to the
native resolver unchanged; the host does not choose physical kernel addresses
or maintain an alias table for catalog names.

## Model options

The RPU model accepts only `quantization="fp16"` and requires
`allow_candidate=True`. These options are separate from the public
`image_preprocess_device` setting: RPU input processing stays on the CPU.

| Option | Default | Contract |
| --- | --- | --- |
| `image_workers` | `6` | Size of the input processor's camera pool |
| `audit_inputs` | `False` | Retain additional large input hashes |
| `max_cached_signatures` | `2` | Cached prefix signatures, from one to two |
| `prefill_collective_window` | `1` | Requested AR collective window, from 1 through 15 |
| `action_collective_window` | `1` | Requested M16 collective window, one or four |
| `action_whole_stage` | `False` | Prepare one fused action stage |
| `action_fused_steps` | `False` | Fold the action entry into its stage preparation |
| `vision_whole_stage` | `False` | Prepare one fused vision stage |
| `vision_fused_stage` | `False` | Fuse the vision stage without its whole-stage option |
| `prefill_whole_stage` | `False` | Prepare one fused prefill stage |

Non-default collective and fused modes require their corresponding native
capabilities. Mutually exclusive combinations are rejected before checkpoint
loading. Window settings are retained across normal requests and explicit
reset; they are not silently reduced.

For example:

```python
with Policy("wallx2b", checkpoint, device="rpu:0", quantization="fp16",
            allow_candidate=True, prefill_collective_window=4) as policy:
    result = policy.infer(observation)
```

## Processing and weights

The input processor preserves the checkpoint's two-step image resize, camera
ordering, patch layout, state-token prompt, flow tokens and multimodal RoPE
construction. `audit_inputs=True` records additional input hashes. Close the
processor through `Policy.close()` to drain its camera pool; cancellation is
registered before camera jobs are submitted, and failed camera processing
drains already-submitted work before request teardown.

`preflight_checkpoint` admits the checkpoint's complete configuration, assets
and tensor manifest. `WeightPreparer` then produces host FP16 weight records
for native packing and immutable device allocation. All weights and resident
plans are prepared before the first request; plan construction must not execute
a state-advancing warmup. Ordinary requests bind their explicit inputs and
replay the prepared stages without re-preparing a queue.

Stage outputs stay in native owners. The final normalized action result is
materialized as independent CPU storage and denormalized in the model's
collection callback; the output processor returns that physical result.
Geometry helpers for relative26 and absolute14 are
available for evaluation; ground-truth scoring is not part of policy timing.

## Upstream sources

Checkpoint semantics, the banded image pipeline and evaluation geometry are
adapted from the publicly available RhinoForge Wall Qwen3.5 implementation:

- [Profile](https://github.com/HUIXI-AI/RhinoForge/blob/2535a27756486bdb139b06764975edbe00004d63/python/rpu_backend/adapters/wall_qwen35/profile.py) and [checkpoint admission](https://github.com/HUIXI-AI/RhinoForge/blob/2535a27756486bdb139b06764975edbe00004d63/python/rpu_backend/adapters/wall_qwen35/checkpoint.py).
- [Camera preprocessing](https://github.com/HUIXI-AI/RhinoForge/blob/2535a27756486bdb139b06764975edbe00004d63/python/rpu_backend/adapters/wall_qwen35/preprocessing.py).
- [Evaluation geometry helpers](https://github.com/HUIXI-AI/RhinoForge/blob/2535a27756486bdb139b06764975edbe00004d63/examples/wall_qwen35_openloop.py).

## Failure and lifecycle ownership

The RPU engine admits one request at a time. Cancellation is possible before
its worker starts; after device submission, shutdown and result handling wait
for real completion because the public TP ABI has no nonblocking completion
query or mid-execution cancellation.

A preparation, execution or collection error blocks further requests until an
explicit reset. Rollback releases request bindings and marks plans invalid; only
explicit reset rebuilds the full plans. Model plans do not replay partially executed work. Reset must
be called while the engine is idle and open; it invalidates state generations,
resets native state owners and rebuilds plans while retaining packed weights.
If reset fails, the engine remains blocked. Neither recovery nor a fallback to
another runtime is automatic.

Request outcome statistics include binding cleanup. A request is completed only
after output collection and binding release both succeed. A release failure
counts once as failed and requires reset.

`Policy.close()` drains outstanding engine work, then closes the model's
execution resources and input processor. `RpuEngine.shutdown()` performs the
same drain and model/context cleanup; an externally supplied execution context
keeps its pre-existing resources. Repeated close calls are safe. Cleanup
failure does not leave a failed native release advertised as an available
context. A close requested while another body exception is active annotates
that exception rather than replacing it.

Prefix plans are bounded by `max_cached_signatures`. Request-owned resources
are released after completion, while immutable weights and cached prepared
batches persist until model close. Returned CPU arrays remain valid after
later requests, resets and close.

## Validation and limitations

The `test_rpu_runtime.py` host lifecycle contracts use native fakes and run
without a board or `xcompute.so`. They cover ownership, explicit waits, request
admission, cancellation, failure, reset and release. Other model tests cover
processing, checkpoint manifests, weight transforms and CPU numerical
references; some planner tests require xcompute's host-only `rpu_tp_resolve`
library export and are skipped when that export is unavailable.

```bash
python -m pytest -q -m "not gpu and not assets" \
  tests/models/test_wallx2b_action.py \
  tests/models/test_wallx2b_denoise.py \
  tests/models/test_wallx2b_family_calls.py \
  tests/models/test_wallx2b_identity.py \
  tests/models/test_wallx2b_model.py \
  tests/models/test_wallx2b_position_reference.py \
  tests/models/test_wallx2b_prefill.py \
  tests/models/test_wallx2b_profile.py \
  tests/models/test_wallx2b_rpu_processing.py \
  tests/models/test_wallx2b_static_steps.py \
  tests/models/test_wallx2b_vision.py \
  tests/models/test_wallx2b_weights.py \
  tests/unit/test_rpu_runtime.py
```

These checks do not qualify the board implementation. Complete validation
must use the actual checkpoint, the selected SDK/AOT library and an isolated
board to verify library selection, exported ABI symbols, prepared-operation
replay, output alignment against the reference, failure recovery and repeated
close/reopen behavior. No complete-request numerical or latency acceptance for
this integration is claimed here.

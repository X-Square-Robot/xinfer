# Wall-X 2B design

English | [简体中文](wallx2b_design.zh-CN.md)

Wall-X 2B runs a Qwen3.5 backbone and a flow-matching action expert. A request contains three camera views, the robot state and an instruction. The [model guide](wallx2b.md) describes checkpoint files, inputs and outputs.

## Stage boundaries

| Stage | Computation | Handoff |
|---|---|---|
| Vision | The vision transformer encodes image patches with per-image attention. The merger combines spatial groups of patch embeddings. | Image tokens. |
| Prefill | Image tokens replace the image placeholders in the prompt. The prompt runs through the backbone's DeltaNet and full-attention layers. | Rotated keys and values from the full-attention layers. |
| Action | The flow expert repeatedly predicts a velocity for the action tokens, attending to the stored prompt keys and values. Each Euler step updates the provisional action. | A normalized action chunk, which the output processor converts to robot poses. |

The prompt is independent of the sampled action tokens. The action stage can reuse the same prompt cache across denoising steps. For a real-time chunking (RTC) request, the model holds the supplied prefix fixed during each step. The remaining rows are predicted with that prefix as context.

## MXFP8

BF16 is the default. With the optimized composition, `quantization="mxfp8"` converts selected projection weights once after loading. A block of up to 128 columns in each row shares one scale. Layer normalization and GEMM epilogues can produce the quantized input for the next projection. Intermediate residuals and the output action retain their usual precision.

| Stage | What changes with the MXFP8 preset |
|---|---|
| Vision | The attention QKV and output projections and both MLP projections use MXFP8 operands. The patch embedding and merger stay on the BF16 path. |
| Prefill | The prompt's full-attention and DeltaNet projections and its MLP projections use MXFP8 operands. The prompt KV cache remains BF16. |
| Action | The action expert uses its original computation and consumes the prompt KV cache. |

To enable only prefill, pass `quantization={"groups":{"prefill":"mxfp8"}}` to `Policy`. Use the group `vision` for vision only. The preset requires `numerics="optimized"`. The native path requires SM120 and an xcompute build with MXFP8 GEMM and quantized norm kernels. xcompute selects a specialized or generic block-scaled kernel for each projection. Other devices and older builds use the PyTorch MXFP8 operators. Use `Policy.describe()["ops"]` to check the selected implementations, including `mxfp8_gemm_epilogue` and `mxfp8_gemm_glu_quantize`. The [quantization guide](../quantization.md) describes the E4M3 values and scale layout.

## NVFP4

`quantization="nvfp4"` applies the Wall-X NVFP4 recipe after loading: E2M1 values with one E4M3 scale per 16 columns.
The vision weights take amax block scales (`nvfp4`). The prompt weights take block scales chosen by an MSE search
under a global scale per checkpoint weight, `amax / (448 · 6)` (`nvfp4_mse`); the Q·gate, K and V weights of a
full-attention layer keep their own global scales, so they run as three GEMMs on one quantized input. Every GEMM input
is the BF16 tensor the BF16 composition feeds it, quantized with amax block scales: the norms and the GEMM epilogues
write it directly.

| Stage | What changes with the NVFP4 preset |
|---|---|
| Vision | The attention QKV and output projections, both MLP projections and both merger projections use NVFP4 operands. The patch embedding stays on the BF16 path. |
| Prefill | The DeltaNet `qkv` and output projections, the full-attention Q·gate, K, V and output projections and the MLP projections use NVFP4 operands. The DeltaNet z/b/a projection and the prompt KV cache remain BF16. |
| Action | The action expert uses its original computation and consumes the prompt KV cache. |

The groups also take `nvfp4` and `nvfp4_mse` separately, for example `quantization={"groups":{"prefill":"nvfp4_mse"}}`.
The preset requires `numerics="optimized"`. The native path requires SM120 and an xcompute build with NVFP4 inputs:
xcompute runs the projections with a residual, the prompt projections and the vision fc1 with its quantized output in
its CUTLASS NVFP4 kernels, and the QKV with RoPE and the gate/up projection on its generic block-scaled path, with the
same values. Other devices and builds without NVFP4 use the PyTorch NVFP4 operators. Use `Policy.describe()["ops"]`
to check `nvfp4_gemm_epilogue` and the other `nvfp4_*` operators. The [quantization guide](../quantization.md#nvfp4-format)
describes the E2M1 values, the scales and the operator forms.

## Validation

The [small checkpoint tests](../../tests/models/test_wallx2b_tiny.py) compare the three stages with the BF16 reference using the same inputs for each stage. They check flow and RTC requests and the exact preservation of the pinned actions. The [operator tests](../../tests/unit/test_mxfp8.py) exercise rounding, the prompt padding mask, fused normalization and rotary projections. On a CUDA device with a compatible xcompute build, they also compare the kernels with the PyTorch operators at Wall-X 2B projection sizes.

On an RTX 5090 with xcompute `536ea8f` (which has the MXFP8 GEMM family), MXFP8 measured a single-request P50 of 17.0 ms from serialized request to response in process (`benchmarks/request_latency.py`, GPU image decoding and resizing): 3.4 ms of host input and output processing and 13.6 ms in the engine. BF16 on the same build measured 22.3 ms (3.5 + 18.9). Under robot load (`benchmarks/performance.py`, eight rounds per count, 14–24 robots in steps of two), MXFP8 served 22 robots at a median server P50 / P95 of 76 / 116 ms, and 24 robots took 151 ms. BF16 measured 86 / 103 ms at 14 robots, its capacity on xcompute `7336b0d`, where 15 robots took 170 ms. MXFP8 changes the actions by more than rounding. Assess accuracy on the intended robot task before enabling MXFP8.

For NVFP4 the [operator tests](../../tests/unit/test_nvfp4.py) check the forms against their definitions and, on a GPU with
a compatible xcompute build, xcompute's NVFP4 operators against the PyTorch operators at Wall-X 2B projection sizes;
the small checkpoint tests run the three stages with `quantization="nvfp4"`. On an RTX 5090 with xcompute `3a2bee6`
(the NVFP4 kernels before their merge as `7034bdf`, which gives the same actions bit for bit), the stage graphs took
vision 1.5, prefill 3.1 and action 5.2 ms (9.8 ms; BF16 16.6 ms), and the single-request P50 was 15.2 ms (3.5 + 11.7).
Under robot load NVFP4 served 28 robots at a median server P50 / P95 of 103 / 145 ms; 30 robots took 167 ms. The
saturation was 154.3 req/s (MXFP8 131.2, BF16 81.5 on that GPU). The weights take 3.0 GB after loading (BF16 4.9 GB).
NVFP4 applied to a checkpoint after training changes the actions much more than MXFP8 and does not pass the internal
precision gate on this checkpoint; the recipe is meant for checkpoints trained with its fake quantization. Assess
accuracy on the intended robot task before enabling NVFP4.

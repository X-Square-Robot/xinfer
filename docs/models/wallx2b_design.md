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

## Validation

The [small checkpoint tests](../../tests/models/test_wallx2b_tiny.py) compare the three stages with the BF16 reference using the same inputs for each stage. They check flow and RTC requests and the exact preservation of the pinned actions. The [operator tests](../../tests/unit/test_mxfp8.py) exercise rounding, the prompt padding mask, fused normalization and rotary projections. On a CUDA device with a compatible xcompute build, they also compare the kernels with the PyTorch operators at Wall-X 2B projection sizes.

On an RTX 5090 with xcompute `536ea8f` (which has the MXFP8 GEMM family), MXFP8 measured a single-request P50 of 17.0 ms from serialized request to response in process (`benchmarks/request_latency.py`, GPU image decoding and resizing): 3.4 ms of host input and output processing and 13.6 ms in the engine. BF16 on the same build measured 22.3 ms (3.5 + 18.9). Under robot load (`benchmarks/performance.py`, eight rounds per count, 14–24 robots in steps of two), MXFP8 served 22 robots at a median server P50 / P95 of 76 / 116 ms, and 24 robots took 151 ms. BF16 measured 86 / 103 ms at 14 robots, its capacity on xcompute `7336b0d`, where 15 robots took 170 ms. MXFP8 changes the actions by more than rounding. Assess accuracy on the intended robot task before enabling MXFP8.

# Denoise-head kernels

The two ends of the flow head's denoising step on Wall-X 2B, the CUDA side's `denoise_output/` counterpart:
the step-input embed and the fp32 head pair. Wall-OSS has no denoise head (its flow runs vendor-side).

| Kernel | Model | Shape envelope | n/req | v22 source | Gate evidence |
|---|---|---|---|---|---|
| `denoise_input_embed` | wall2b | flow head input embed (the denoise chain's `[x | extra] @ weight^T` to BF16) | 10 | `src/denoise_output/npu/denoise_input_embed.py` | 2b-elem K-A kernel routing; ELEM composite gate 0.00585 in band |
| `denoise_fp32head_prep` | wall2b | denoise fp32 head preparation (scalar leg) | 10 | `src/denoise_output/npu/denoise_fp32_scalar.py` | 2b-elem; the prep+cube merge (GQA-GLUE k4) was not done, keep two kernels |
| `denoise_fp32head_cube` | wall2b | denoise fp32 head cube | 10 | `src/denoise_output/npu/denoise_head_cube.py` | 2b-elem |

## Model attribution

- **Wall-X 2B** only: 3 x 10 calls per request carry the flow head (busy share ~305 us of the 20.1 ms engine
  window).

## TODO (R1)

- Kernel modules from the v22 sources above.

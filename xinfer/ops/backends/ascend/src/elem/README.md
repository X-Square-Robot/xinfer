# Elementwise kernels

Pure pointwise kernels, the CUDA side's `activation/` + `quant/` slot on this backend. One kernel is in
service: the frame quantizer of the vision-prepare segment.

| Kernel | Model | Shape envelope | n/req | v22 source | Gate evidence |
|---|---|---|---|---|---|
| `quant_u8` | wall2b | frame quantization `uint8(floor(clamp(x + 0.5, 0, 255)))` with the PK4_B32 packing law; mounted at `device_images._resize`, `image_graphs.chain_view`, `npu_images._rgb` | 9 in the engine window, ~30/req full (the DVPP wave overlaps the margin) | `src/soup2/npu/soup2_kernels.py` (K4) | SOUP2: witness over 5 shapes + constructed extremes, 0 flips; gate delta 0.0 |

## Model attribution

- **Wall-X 2B** only, prepare segment. The gain lives outside the engine window: acceptance must measure
  e2e/total, not the window (the audit's finding).

## TODO (R1)

- Kernel module from the v22 source above; the three mount points are engine-side hooks, they stay in the
  engine/platform layer.

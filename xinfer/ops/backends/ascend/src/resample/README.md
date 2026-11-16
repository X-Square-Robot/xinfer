# Resample kernels

Antialiased downsampling of the vision input (the AA-RESAMPLE knife), the Ascend-specific vision-preprocessing
family with no CUDA counterpart in this repository. The up leg was declined by its gate and stays on the vendor
`UpsampleBicubic2dAA`/`Bilinear2d` — only the four down kernels below move in.

| Kernel | Model | Shape envelope | n/req | v22 source | Gate evidence |
|---|---|---|---|---|---|
| `aa_v_720_1280_128_224` | wall2b | AA downsample V low-pass 720x1280 -> 128x224 (25 taps), uint8 in (UNPK_B8) | 1 (down route) | `src/aa_resample/npu/aa_resample_kernels.py` (`aa_v_*`) | in-graph iso 720p 222.0 -> 66.8 us (3.32x) |
| `aa_h_720_1280_128_224` | wall2b | AA downsample H + fused quantization 720x1280 -> 128x224 (vgather2 + quant law) | 1 | `src/aa_resample/npu/aa_resample_kernels.py` (`aa_h_*`) | same |
| `aa_v_480_640_160_224` | wall2b | AA downsample V low-pass 480x640 -> 160x224 (13 taps) | 1 | `src/aa_resample/npu/aa_resample_kernels.py` (`aa_v_*`) | desktop 98.2 -> 51.1 us (1.92x) |
| `aa_h_480_640_160_224` | wall2b | AA downsample H + fused quantization 480x640 -> 160x224 | 1 | `src/aa_resample/npu/aa_resample_kernels.py` (`aa_h_*`) | same |

## Model attribution

- **Wall-X 2B** only, vision segment (D arm; the C arm keeps vendor AA). Family gate: fp64 oracle not worse
  than vendor (flips 5463/86016 vs 10971), D-arm composite gate 0.0259; default on since `07b7447`.

## TODO (R1)

- Kernel modules from the v22 source above; the `aa_probe` probe kernels are retired assets (the real6
  objects are not reproducible — hard rule), they do not move.

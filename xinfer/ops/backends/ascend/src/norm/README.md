# Normalization kernels

RMS and AdaRMS normalizations, the CUDA side's `norm/` counterpart: the fully fused RMS of Wall-X 2B's 46
sites, the action AdaRMS pair, and the decode/prefill RMS fuse family of Wall-OSS. The two one-per-request
remnants ride along (the audit recommends carrying them; the se1 form may absorb them later).

| Kernel | Models | Shape envelope | n/req | v22 source | Gate evidence |
|---|---|---|---|---|---|
| `rmsnorm_full_c2048` | wall2b | fully fused RMS: `s = rne(f32(x) + f32(res))`, 64-lane fp32 vector tree reduction, Newton rsqrt, apply — 46 sites (`add_zero_centered_rms_norm` + `modulated_rms_norm`, `FROM_LAYER=0` default all) | 46 | `src/soupunlock/npu/soupunlock_kernels.py` (K1) | SOUP-UNLOCK: fp64 oracle ulp <= 1, ties vendor; L0 composite gate 0.0342 <= 0.05; e2e -0.185 ms |
| `adarmsnorm_action_double_bf16_f32` | wall2b | action AdaRMS double m16 (v4 recast) | 180 | `src/adanormr2/adanorm_r2.py` (v4) | ADANORM-R2 v4 -0.78 ms; QUINT bisect keeps it fully on |
| `adarmsnorm_action_gated_bf16_f32` | wall2b | action AdaRMS gated m16 (v4 recast) | 130 | `src/adanormr2/adanorm_r2.py` (v4) | same |
| `rms_apply_rstd_h2048` | wall2b | RMS apply rstd h2048 (spectral remnant of the norm chain) | 1 | `src/normalization/npu/rms_apply.py` | lineage in service (SOUPFORM K2 recast predecessor) |
| `rf4_m16` | walloss | decode RMS fuse m16 | 365 | `knives/rmsfuse_r4.py` | RMSFUSE-R4 lineage (bitwise) |
| `rf4_ubv2048_f` | walloss | prefill RMS fuse ubv 2048 (p768/752/736 x 2048) — remnant: the 70 main sites moved to `se1_ubv2048_buf2f`, the vision/prefill entry `rms_norm_cast` stays | 1 | `knives/rmsfuse_r4.py` | remnant x1, in service |
| `se1_ubv1280_buf2f` | walloss | SIMD fused-after add_rms r2112x1280 (vision domain): published arithmetic statement-by-statement order + buf2 double buffer + f32-direct weight | 64 | `src/simdepi/npu/simdepi_r1.py` (`se1_ubv1280_buf2f`) | SIMD-EPILOGUE: 5 shapes sum+norm 0-diff; actions A/B exact 0.0 |
| `se1_ubv2048_buf2f` | walloss | SIMD fused-after add_rms p768/752/736x2048 (prefill domain) | 70 | `src/simdepi/npu/simdepi_r1.py` (`se1_ubv2048_buf2f`) | same |
| `rms_consistent_ubv_h1280` | walloss | vision RMS ubv h1280 — remnant (h2048 sibling x1, 2 remnant points total) | 1 | `src/normalization/npu/rms_consistent.py` | remnant x1, in service |

## Model attribution

- **Wall-X 2B**: `rmsnorm_full_c2048` is the single widest fusion in the model (46 sites, the L0 unlock); the
  AdaRMS pair serves the action stream's double/gated norms; `rms_apply_rstd_h2048` is a 1/req spectral remnant.
- **Wall-OSS**: `rf4_m16` is the highest-dose kernel of the model (365/req); the se1 pair owns vision/prefill
  add_rms; two 1/req remnants remain at the entry casts.

## TODO (R1)

- Kernel modules from the v22 sources above; the normprep/apply_rstd two-step forms and the r5 generator
  family stay out (superseded; the audit's not-shipped list).

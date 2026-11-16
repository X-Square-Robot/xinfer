# Finish kernels

GEMM-fused projection finishes — the family the CUDA side serves inside `gemm/sm120` (fused epilogue GEMMs).
On this backend the GEMM body often stays aclnn while the finish (split, pack, rope store, bias+residual,
GLU) is the kernel; one kernel (`gemm_glu_fp32_finish_split_11008`, `gemm_qkv_vision_rope_finish`) fuses the
GEMM itself.

| Kernel | Models | Shape envelope | n/req | v22 source | Gate evidence |
|---|---|---|---|---|---|
| `glu_split_bf16_c6144` | wall2b | gateup GLU split, bf16 read (vld UNPK_B16 + vcvt); GEMM `[M,2048]x[12288,2048]` with aclnnMatmul bf16 direct out tail (div-type SiLU) | 23 | `src/finishbf16/npu/finishbf16_kernels.py` (gateup) | FINISH-BF16 gateup: composite gate 0.01700; e2e -0.289 ms |
| `zbasplit_bf16_c8224` | wall2b | zba split bf16: `[464,2048]x[8224,2048]` keep{0,1} exact multiply; conv `[P,6144]` / zba `[P,2080]` double buffer (2080 tail UB padded to 2112) | 18 | `src/finishbf16/npu/finishbf16_kernels.py` (zba) | FINISH-BF16 zba: in-stack bitwise flips = 0; gate identical to knife 1 |
| `finish_br_c1024` | wall2b | finish + bias + residual `(1024, 4)` fp32 (SOUPKILL finish family) | 48 | `gap-2b-soup/aot/ops/soupkill_r1.py` (not in the v22 `src/` families; needs family-ization) | SOUPKILL lineage bitwise; the bf16 finres variant stays parked, fp32 in service |
| `vision_qkv_rope_v3p` | wall2b | vision QKV rope packed store: `[868,1024]x[3072,1024]` GEMM tail + `[3, rows, heads, dim]` blocked contiguous write (vropepack) | 24 | `src/soupform/npu/soupform_kernels.py` (K3) | SOUP-FORM witness 0/2666496 flips; zero gate delta |
| `gemm_glu_fp32_finish_split_11008` | walloss | prefill GLU finish, N22016 half-width 11008 split (half-row tiles; 8-slab 1376) | 35 | `src/gemm/npu/gemm_glu_fp32_finish.py` (`split_finish(11008)`) | absorb2/glu-r1; v0.92 M2 0.91 bandwidth saturated (floor) |
| `gemm_qkv_vision_rope_finish` | walloss | vision QKV + rope fused finish | 32 | `src/gemm/npu/gemm_qkv_bf16_wide.py` (L521) | v0.95 vector pipe saturated (floor) |
| `ubfinish_r5_s8_c1280` | walloss | ub finish s8 c1280 (vision/merger tail) | 64 | `knives/ubfinish_r5.py` | v0.61 M2 0.39 half-saturated |

## Model attribution

- **Wall-X 2B**: the bf16 finish pair (gateup/zba) plus the SOUPFORM vropepack and the SOUPKILL finish carry
  the projection tails the SOUP2 pass left behind.
- **Wall-OSS**: the fp32 split finish (11008 is the only split width still served), the vision QKV+rope finish
  and the ub finish carry the prefill/vision tails.

## TODO (R1)

- Kernel modules from the v22 sources above. The `split_finish` generator keeps only the {11008} width
  (2048/6144 forms are superseded by the bf16 pair on wall2b); `soupkill_r1.py` and the vrope/finres/gelu
  bf16 variants follow the audit's not-shipped list.

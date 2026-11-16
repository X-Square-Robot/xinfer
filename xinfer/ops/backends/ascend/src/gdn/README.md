# Gated-delta-rule kernels

The wrapper and prefix layers the CUDA side serves with `gated_delta_rule/`. The chunk recurrence itself stays
on the vendor `aclnnChunkGatedDeltaRule` (18/req on both models, issue-bound cu99; the self-built four-kernel
route was finally judged architecturally over budget at 4.4x, 65 us unreachable) — this family holds the six
kernels that pack, prepare and finish around it on Wall-X 2B. Wall-OSS never built this domain (pure vendor
chunk there).

| Kernel | Model | Shape envelope | n/req | v22 source | Gate evidence |
|---|---|---|---|---|---|
| `dconv4_c6144` | wall2b | GDN depthwise causal conv 1x4, groups 6144: qkv `[W, 6144]`, W in {416, 432, 448, 464}; shifted 4-tap FMA in one pass (absorbs slice + 2 transpose satellites) | 18 | `src/convgemm/npu/conv_gemm.py` | CONV-GEMM: probe2 seq_asc 0 mismatch (P x 3 seeds bitwise); gate 0.0050923228 exact |
| `gdrprep_bf16` | wall2b | GDN gates chain preparation (9 kernels — Sigmoid/Exp/Neg/Slice/Cast/Add/Softplus/Mul — in one) | 18 | `src/gdnv/npu/gdnv_kernels.py` | GDN-VENDOR: C vs B all 10 requests bitwise, delta 0.0; e2e -0.149 ms |
| `gdrapply_bf16` | wall2b | GDN wrapper pack segment (width slice + TND pack family) | 18 | `src/mover2/npu/mover2_kernels.py` | MOVE-R2: action gate 0.0050923228 bitwise; e2e -0.70 ms |
| `gdrtailmix_bf16` | wall2b | GDN wrapper tail segment (output-tail elementwise) | 18 | `src/mover2/npu/mover2_kernels.py` | MOVE-R2, same |
| `qksq_c6144` | wall2b | GDN P1 qk statistics prefix: `x_t [1,T,6144]` strided-window single pass `sq = f32(RNE_bf16(a.a))`; downstream aclnn ReduceSum/Adds/Rsqrt unchanged | 18 | `src/soup2/npu/soup2_kernels.py` (K1) | SOUP2: witness 0 flips; gate delta 0.0 (Q vs B bitwise) |
| `rawsq_r128` | wall2b | GDN P2 rstd prefix: `out[T,hv,dv]` flattened `[7424,128]` single pass `sq = RNE_f32(a.a)` | 18 | `src/soup2/npu/soup2_kernels.py` (K2) | SOUP2, same |

## Model attribution

- **Wall-X 2B** only: all six kernels serve the 18 GDN calls per request (the `c6144`/`c8224` widths are the
  2B GDN projection sizes; conv route dose 288/50-run at capture).

## TODO (R1)

- Kernel modules from the v22 sources above. The `cann-gdn-ffi/gdr_cann_chunk.cc` bridge (the vendor chunk's
  FFI bridge) moves with the routing layer, not with these kernels.
- The gdn-k3 AscendC four-kernel series is dead (final judgment); nothing of it moves in.

# Attention kernels

Decode-attention and prefill-glue kernels: the fused FIA tail of Wall-X 2B, the decode attention pair of
Wall-OSS, the GQA head-order glue, and the g2112 softmax-and-mul vector kernel. The vendor
FusedInferAttentionScore keeps the families this tree does not replace (the audit locked them: varlen/vision
868 families and prefill causal on wall2b at 29/req, all of walloss's 67/req — cu62-94, near saturated).

| Kernel | Models | Shape envelope | n/req | v22 source | Gate evidence |
|---|---|---|---|---|---|
| `fia_fuse_scan_l480` | wall2b | decode FIA split-K scan: Q `(1,8,16,256)`, K/V `(1,2,L,256)`, L in {432, 448, 464, 480}; 28 blocks | 60 | `src/fiafuse/npu/fia_fuse.py` (scan) | witness vs fp64 oracle >= 91.2% not-worse; gate 0.00469; FIA-FIX then default on |
| `fia_fuse_reduce_l480` | wall2b | decode FIA reduce + gated sigmoid tail + M-major pack (pack_pow2 arithmetic) | 60 | `src/fiafuse/npu/fia_fuse.py` (reduce_packed) | same |
| `qkv_split_vec_bhsd_h256` | wall2b | prefill `qkv_norm_split` BHSD direct (pure index permute); 5120/8/2/256 lineage, P in {416, 432, 448, 464}; q/k/v/gate | 6 | `src/gqaglue/npu/qkv_bhsd.py` | GQA-GLUE k1: probe 36/36 bitwise |
| `rope_half_d64_bhsd` | wall2b | RoPE half d64 BHSD (in-kernel angle-doubling copy removes the cat) | 6 | `src/gqaglue/npu/rope_bhsd.py` | GQA-GLUE k3: probe 6/6 bitwise; host 79.4 -> 23.1 us/call |
| `decodeattn_r4_scan_b` | walloss | decode small-M cache attention scan: B1 H16 KV2 M16 D128 L784; Q `[256,128]`, K/V `[1568,128]` bf16, FM `[784,2]` int32 (splitk-v3 static shapes) | 180 | `R/decodeattn-r4/aot/ops/decodeattn_r4.py` (scan) | decodeattn-r4 receipt |
| `decodeattn_r4_reduce_row` | walloss | decode attention reduce (row variant, separate row library) | 180 | `R/decodeattn-r4/aot/ops/decodeattn_r4.py` (reduce) | `XINFER_DECODEATTN_R4_ROW=1` |
| `main_kernel` (SAMU, AIVEC) | walloss | SAMU vector kernel, softmax-and-mul domain of the g2112 family | 32 | `knives/g2112_tilelang.py` (`_samu_simd`) | v0.93, vector pipe saturated (floor) |

## Model attribution

- **Wall-X 2B**: the FIA pair owns decode attention (60/req, replacing the vendor decode family); the GQA glue
  pair owns the prefill head order (6/req) that feeds FIA heads directly.
- **Wall-OSS**: `decodeattn_r4` scan/reduce own decode attention (180/req); SAMU owns the softmax-and-mul
  segment (32/req).

## TODO (R1)

- Kernel modules from the v22 sources above; the Python routing wrappers (FIA route table, `_gqa_causal_bnsd`)
  belong to the glue rewrite, not to this directory.
- `decodeattn_r4` needs family-ization first (its generator lives outside the v22 `src/` families).

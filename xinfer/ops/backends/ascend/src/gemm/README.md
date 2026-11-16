# GEMM kernels

GEMMs with fused epilogues for small decode batches and the model-specific projection finishes, the family the
CUDA side serves with `gemm/sm120/`. The generic path on this backend is aclnn (TorchNPU's matmul operators)
with the epilogue as separate kernels — including the audited direct routes below — so this family holds only
the GEMMs that beat or bind the vendor floor at the served shapes.

| Kernel | Models | Shape envelope | n/req | v22 source | Gate evidence |
|---|---|---|---|---|---|
| `smallm_r3_gateup_glu` | wall2b, walloss | m16 gate/up GEMM + SiLU·mul epilogue, one launch; 2B `[16,1024]x[4096,1024]`, walloss `[16,1024] -> 4096` concat | 240 / 180 | `src/smallm/npu/smallm_r3.py` | GLU-EPILOGUE K-A gate 0.00489 (bitwise level); EPILOGUE-R2 reuse |
| `smallm_r3_down` | wall2b, walloss | m16 down / o_proj GEMM; 2B `[16,2048]x[1024,2048]` (240 down + 60 o_proj), walloss down + o_proj | 300 / 360 | `src/smallm/npu/smallm_r3.py` | EPILOGUE-R2 knife A bitwise, 0.0320437253 gate |
| `gemm_qkv_2b_norm_gate_cache_bnsdq` | wall2b | m16 decode QKV fused: `[16,1024]x[5120,1024]` to dense Q (BNSD) + packed gate + KV append; 5120/8/2/256 lineage | 60 | `src/gemm/npu/qkv_bnsdq.py` | 2b-qkv wave lineage bitwise |
| `bnsd_o_proj_m768` | walloss | o_proj GEMM at M 768 (prefill/vision segments; 32 blocks padded) | 35 | `src/gemm/npu/gemm_smallm_action.py` | smallm-r3 lineage routing |
| `gemm_qkv_cache_bias_rope_m16` | walloss | decode QKV bias + RoPE + KV-cache finish (GEMM body stays vendor, `16,1024 -> 2560`) | 180 | `src/gemm/npu/gemm_qkv_cache.py` | walloss-fw lineage |
| `gemm_qkv_cache_bias_rope_prefill` | walloss | prefill QKV cache finish | 36 | `src/gemm/npu/gemm_qkv_cache.py` | walloss-fw lineage |
| `main_kernel` (GEMM_prefqkv, AICORE) | walloss | prefill QKV GEMM of the g2112 family | 36 | `knives/g2112_tilelang.py` (prefqkv) | v0.79, cu91 |

## Model attribution

- **Wall-X 2B** (decode path, 20.62 ms e2e): `smallm_r3_gateup_glu` and `smallm_r3_down` carry the whole m16
  MLP GEMM surface (240 + 300 of 2,428 launches per request); `gemm_qkv_2b_norm_gate_cache_bnsdq` carries the
  fused decode QKV.
- **Wall-OSS** (38.82 ms e2e): `smallm_r3_down` (down + o_proj, 360) and `smallm_r3_gateup_glu` (180) are the
  only two kernels the models share; the QKV finishes and the g2112 prefill GEMM are walloss-only.

## Route layer (aclnn direct, no kernel face)

The generic path keeps the audited direct routes; R1 ports them into the dispatcher, they are not kernels here:

- `finplain` (SOUP2 K3): plain-finish GEMM as a single aclnnMatmul bf16 instead of Mm fp32 + Cast, over
  `(k, n)` in {(2048,2048), (2048,5120), (1024,1536), (1024,4096), (2048,1024)} with `m` in [1, 8192];
  12/12 production shapes bitwise 0 flips, 29 sites, routed 464/50-run.
- `K1 bf16direct` (GEMM-AUDIT): the trivial-epilogue lane; terminal routed=0 (finplain's block runs first), so
  its logic merges into finplain instead of shipping twice.
- `MX8 down prepack/splitk4` (walloss): aclnnBatchMatMul 4,768x2752; 4,2752x2048 split-K stays prepacked
  (splitk2/4 judged negative).
- `G2112 TORCH_A/C` (walloss): large-GEMM operand views, a Python route.

## TODO (R1)

- Kernel modules from the v22 sources above; the SHAPES envelopes are already declared in `../../gemm_950dt.py`.
- `bnsd_o_proj_m768`, `gemm_qkv_cache_bias_rope_prefill` and the g2112 prefill GEMM need their measured `(N, K)`
  recorded (the audit lists M and dose only); fill their SHAPES entries then.

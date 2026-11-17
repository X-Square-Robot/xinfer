# families.md — kernel inventory of the unified Ascend operator library

The authoritative *what goes in* contract is the UNIFY-PORT INVENTORY (2026-11-16):
the routed>0 kernels of the two served models (Wall-X 2B `xinfer/models/wallx2b`,
Wall-OSS `xinfer/models/wall_oss`). `routed=0` generations are NOT built — the old
combined library carried ~500 cores of which these 44 (+2 day-one carry-overs) are
the whole serving surface.

Organisation: by **computation family** (a coordinator directive of 2026-11-16), not
by model; both models' kernels coexist inside the same family directory. Model
ownership is recorded per kernel below, never in the tree.

## Library map (build-libraries.sh targets; the transition family-library form)

| library | env (`_LIB`) | modules | kernels (models) |
|---|---|---|---|
| `libxinfer_ascend_core.so` | `XINFER_ASCEND_CORE_LIB` | norm/rms_apply, norm/rms_consistent, denoise×3(+helper), gemm/qkv_bnsdq, gemm/gemm_smallm_action, gemm/gemm_qkv_cache, gemm/gemm_qkv_bf16_wide, gemm/gemm_glu_fp32_finish | rms_apply_rstd_h2048 (2b), rms_consistent_ubv_h1280 (wl), rms_consistent_generic_h1024_f32in (wl copykill K2), denoise_input_embed (2b), denoise_fp32head_cube (2b), denoise_fp32head_prep (2b), gemm_qkv_2b_norm_gate_cache_bnsdq (2b), bnsd_o_proj_m768 (wl), smallm_bnsd_o_proj{,_m32,_m64} (BNSD [16, rows in {16,32,64}, 128] o-projection forms, dispatched from `_cann_primitives.mm_pair`; the v22 bridge's SmallmBnsdRoute), gemm_qkv_cache_bias_rope_m16 (wl), gemm_qkv_cache_bias_rope_prefill (wl), gemm_qkv_vision_rope_finish (wl), gemm_glu_fp32_finish_split_11008 (wl) |
| `libsmallm_r3.so` | `XINFER_SMALLM_R3_LIB` | gemm/smallm_r3 (+gemm/silu_and_mul helper) | smallm_r3_gateup_glu (both), smallm_r3_down (both), smallm_r3_qkv (day-one try-on, flag default off — INVENTORY finding 1) |
| `libadanorm_r2.so` | `XINFER_ADANORM_R2_LIB`* | norm/adanorm_r2 | adarmsnorm_action_double_bf16_f32 (2b), adarmsnorm_action_gated_bf16_f32 (2b) |
| `libsoupunlock.so` | `XINFER_SOUPUNLOCK_LIB` | norm/soupunlock_kernels | rmsnorm_full_c2048 (2b) |
| `libsoupform.so` | `XINFER_SOUPFORM_LIB` | norm/soupform_kernels | vision_qkv_rope_v3p (2b) |
| `libsoupkill.so` | `XINFER_SOUPKILL_LIB` | finish/soupkill_r1 | finish_br_c1024 (2b) |
| `libfinishbf16.so` | `XINFER_FINISHBF16_LIB` | finish/finishbf16_kernels | glu_split_bf16_c6144 (2b), zbasplit_bf16_c8224 (2b) |
| `libsimdepi.so`* | `XINFER_SIMDEPI_LIB` | norm/simdepi_r1 | se1_ubv1280_buf2f (wl), se1_ubv2048_buf2f (wl) |
| `librmsfuse_r4.so` | `XINFER_RMSFUSE_R4_LIB` | norm/rmsfuse_r4 | rf4_m16 (wl), rf4_ubv2048_f (wl residual 1/req) |
| `libubfinish_r5.so` | `XINFER_UBFINISH_R5_LIB` | finish/ubfinish_r5 | ubfinish_r5_s8_c1280 (wl), ubfinish_r5_s4_c2048 (registry twin; the v22 ubfinish-r5-registry.json routes c2048 to it) |
| `libdecodeattn_r4.so` | `XINFER_DECODEATTN_R4_LIB` | attn/decodeattn_r4 | decodeattn_r4_scan_b (wl), decodeattn_r4_reduce (wl), decodeattn_r4_reduce_row (wl; the `XINFER_DECODEATTN_R4_ROW=1` final-arm form, INVENTORY wl #11 — token-major twin of reduce, every arithmetic statement byte-identical, only the store address changes) |
| `libfia_fuse.so` | `XINFER_FIA_FUSE_LIB` | attn/fia_fuse | fia_fuse_scan_l{432,448,464,480} + fia_fuse_reduce_l{432,448,464,480} + fia_fuse_reduce_bnsd_l{432,448,464,480} (2b; all four `_BUILD_KEYS` L forms the route serves plus the reduce_bnsd twins the v22 delivery carried) |
| `libgdnv.so` | `XINFER_GDNV_LIB` | gdn/gdnv_kernels | gdrprep_bf16 (2b) |
| `libmove_r2.so` | `XINFER_MOVER2_LIB` | gdn/mover2_kernels | gdrapply_bf16 (2b), gdrtailmix_bf16 (2b) |
| `libconv_gemm.so` | `XINFER_CONVGEMM_LIB` | gdn/conv_gemm | dconv4_c6144 (2b) |
| `libsoup2.so` | `XINFER_SOUP2_LIB` | elem/soup2_kernels | qksq_c6144 (2b), rawsq_r128 (2b), quant_u8 (2b) |
| `libaa_resample.so` | `XINFER_AA_RESAMPLE_LIB` | resample/aa_resample_kernels | aa_v_720_1280_128_224, aa_h_720_1280_128_224, aa_v_480_640_160_224, aa_h_480_640_160_224 (2b, all four) — **asset relink, see the AA adjudication below** |
| `libg2112_tilelang.so`* | `XINFER_G2112_LIB`* (documented; JIT) | gemm/g2112_tilelang | main_kernel (SAMU AIVEC, wl; JIT), main_kernel (GEMM_prefqkv AICORE, wl; JIT) |

Kernels without a library above: **GQA-GLUE bhsd pair** — `qkv_split_vec_bhsd_h256`
and `rope_half_d64_bhsd` live in `libxinfer_ascend_core.so` in the v22 deployment;
they are built by S2 as a separate `libgqa_glue.so` step in the run tree (module
`ops.norm.qkv_bhsd` / `ops.norm.rope_bhsd`) and can be merged into core in a
follow-up relink once the byte-anchor comparison lands (INVENTORY route receipts
are the arbitration source). \* marked env names are new spellings this branch
introduces for families that previously had no dedicated library (adanorm-r2 rode
the combined lib; simdepi/g2112 rode per-knife paths); every v22-era
`XINFER_<FAMILY>_LIB` name is preserved verbatim.

## What is deliberately NOT here (routed=0, INVENTORY §8)

vitfinish_r4 (all generations — final stack has zero face; `XINFER_VITFINISH_R4`
was stale and the mount is removed with it), normprep/apply_rstd (superseded by
rmsnorm_full L0), zbasplit fp32 (superseded by bf16), vrope v3 (superseded by v3p),
qkv_split scalar/vec generations (only BHSD survives), rope d256/non-bhsd (only
bhsd survives), gated_attn_bnsd_pack_pow2 (absorbed by fia_fuse), split_2048/6144
2B usages (only 11008 is served), fc1_gelu/probe_vdiv/gelu_micro (soup2 parked),
gdn-k3 AscendC (dead), FIA varlen replacement (quantitatively rejected), aa_probe,
se1 non-shipping variants, rf5 generation, gap_biasfin, smallm_r3_o_proj
(bnsd_o_proj_m768 serves), gemm_glu_fp32 rowpass/paired (11008 split only).

## Static (non-TileLang) surface — `static/`

## AA-RESAMPLE adjudication (S3, 2026-11-16): the real6 asset relink

The vendored `ops/resample/aa_resample_kernels.py` is the campaign tree's
final (real10-state) source. Regenerating the four down-leg kernels from it
makes the auto-scheduler emit doubled MTE2/MTE3 notifies and waits whose
FIFO flag matching **deadlocks the vector core** (aivec timeout — the
real9 pathology the source header itself documents; reproduced on card 5,
S3 first-run). The **delivered** `libaa_resample-v22.so` was adjudicated by
the campaign's `pf-v22-objcheck.sh` as "the real6 objects are assets; the
relink must reproduce" — a plain `g++ -shared` of the four real6 objects
from `aa-resample/build-real6/aot/aa_resample/`. This branch follows that
adjudication verbatim: `build.sh` renames every symbol containing the old
prefix in those objects (`objcopy --redefine-syms`, the registration is
symbol-address based so the rename is internally consistent) and links
`libaa_resample.so` from them; regeneration from the tree source is
deliberately not attempted. Witness: the renamed relink runs and is
**bitwise-equal to the delivered library** on the 720p v-kernel (card 5
probe, 2026-11-16). `AA_REAL6_DIR` overrides the asset location at build
time; the kernels themselves are ordinary build-time assets exactly like
the bisheng/tilelang toolchain pins.



The C-ABI bridges the glue needs from the core library (aclnn compositions and
finish launchers the python dispatch calls through `_provider`):
`cann_fia_v5.cc` + `cann_fia_vision.cc` (FIA probe pair), `cann_mm_fp32.cc`
(ACLNN Mm pair + typed workspace), `gdr_cann_chunk.cc` (vendor GDN chunk FFI),
`gemm_finish_launchers.cc` (gemm_epilogue_glu / gemm_glu_fp32 deferred /
action_qkv_gemm_norm_rope / gemm_qkv_cache pair), `positions.cc`
(rot_pos/pos_embed_interp/get_token_counts). These are lane-3 ports of the v22
`src/*/npu/*.cc` bridge sources with the exported names mechanically renamed
`xcompute_* -> xinfer_*` and the tvm_ffi registration stripped (ctypes-only
loading). Byte-identity of the compiled objects is not expected (registration
code removed); verification is the bit-level cross-library witness (same inputs
through the v22 combined lib and the new core, compare outputs).

## Model ownership summary

- wall2b-only: 27 kernels (adanorm×2, rmsnorm_full, v3p, finish_br, glu_split_bf16,
  zbasplit_bf16, dconv4, gdrprep, gdrapply, gdrtailmix, qksq, rawsq, quant_u8,
  fia_fuse×2, denoise×3, rms_apply, aa×4, gemm_qkv_2b_bnsdq)
- walloss-only: 13 kernels (se1×2, rf4_m16, rf4_ubv2048_f, rms_consistent,
  ubfinish, decodeattn×2, bnsd_o_proj, gemm_qkv_cache×2, vision_rope_finish,
  split_11008) + g2112 JIT pair
- both: smallm_r3_gateup_glu, smallm_r3_down
- day-one carry-over (default off): smallm_r3_qkv

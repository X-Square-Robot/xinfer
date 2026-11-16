#!/bin/bash
# kernels/build.sh — the unified Ascend operator library, built in-repo without
# xcompute CMake. Recipe = BUILD-SPIKE 2026-11-16 (byte-for-byte contract):
# bisheng 9.3.0 compiles, CANN 9.1.1 links and loads, dav-3510, MAX_JOBS=1,
# the tilelang fork venv, TILELANG_CACHE_DIR isolated per invocation.
#
# Inputs (in this directory):  tools/{generate.py,abi.py}  ops/<family dirs>
# Outputs (under $WORK):       out/<kernel>.{o,asc,tir} manifest.json  lib*.so
#
# Targets: one core library (the always-on class-entry kernels) plus one knife
# library per routed family (XINFER_<FAMILY>_LIB name preserved verbatim).
# `build.sh install` copies the products into lib/ (the loader's default path).
#
# Run from this directory (the aot/ layout contract: generate.py resolves the
# `abi` import and the `--module ops...` packages from its own directory).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HERE/.build}
COMPILER=${BISHENG_HOME:-/usr/local/Ascend/cann-9.3.0}
RUNTIME=${CANN_RUNTIME:-/usr/local/Ascend/cann-9.1.1}
PY=${TILELANG_PYTHON:-/root/tl-fork-store/venv-tl-src/bin/python}
export MAX_JOBS=1

# 1) torch_npu runtime libs must be importable before python starts
#    (tilelang -> torch -> torch_npu backend import crashes otherwise), THEN
#    the device compiler trio points at 9.3.0.
set +u; source "$RUNTIME/set_env.sh" >/dev/null 2>&1 || true; set -u
export BISHENG_HOME=$COMPILER ASCEND_HOME_PATH=$COMPILER ASCEND_TOOLKIT_HOME=$COMPILER
export PATH="$COMPILER/bin:$PATH"
export TILELANG_CACHE_DIR="$WORK/tl-cache"
mkdir -p "$WORK/out" "$WORK/tl-cache"
cd "$HERE/tools"

# 2) The tilelang fork pin (BUILD-SPIKE dependency table). Fail fast on drift:
#    every byte-anchor claim in the README assumes exactly this toolchain.
"$PY" - <<'PYCHECK'
import sys
import tilelang
expected = "0.1.15+cpu.git1e8448fc"
if tilelang.__version__ != expected:
    sys.exit(f"tilelang {tilelang.__version__!r} != pinned {expected!r}; "
             "see kernels/README.md (BUILD-SPIKE 2026-11-16)")
PYCHECK

gen() {  # gen <out-subdir> <module> <kernels...>
    local out="$WORK/out/$1"; shift
    mkdir -p "$out"
    "$PY" generate.py --output "$out" --module "$1" --kernels "${@:2}"
}

link() {  # link <libname> <out-subdir> <objects...>
    local lib="$WORK/$1"; shift
    local dir="$WORK/out/$1"; shift
    local objs=(); local k
    for k in "$@"; do objs+=("$dir/$k.o"); done
    g++ -shared -o "$lib" "${objs[@]}" \
        -L"$RUNTIME/lib64" -lascendcl -lopapi -lruntime -lnnopbase
}

# ---------------------------------------------------------------- core library
# The always-on AOT surface behind the class-style python entries (glue/_core.py):
# reference-composition fallbacks and the 2b/walloss kernels the binding layer
# reaches through the provider. Model-owned knife kernels live in their own
# libraries below; this core carries only what glue/core dispatch launches.
gen core ops.norm.rms_apply          rms_apply_rstd_h2048
gen core ops.norm.rms_consistent     rms_consistent_ubv_h1280 rms_consistent_generic_h1024_f32in
gen core ops.denoise.denoise_input_embed  denoise_input_embed
gen core ops.denoise.denoise_head_cube    denoise_fp32head_cube
gen core ops.denoise.denoise_fp32_scalar  denoise_fp32head_prep
gen core ops.gemm.qkv_bnsdq          gemm_qkv_2b_norm_gate_cache_bnsdq
gen core ops.gemm.gemm_smallm_action bnsd_o_proj_m768_bf16
gen core ops.gemm.gemm_qkv_cache     gemm_qkv_cache_bias_rope_m16 gemm_qkv_cache_bias_rope_prefill
gen core ops.gemm.gemm_qkv_bf16_wide gemm_qkv_vision_rope_finish
gen core ops.gemm.gemm_glu_fp32_finish gemm_glu_fp32_finish_split_11008
link libxinfer_ascend_core.so core \
     rms_apply_rstd_h2048 rms_consistent_ubv_h1280 rms_consistent_generic_h1024_f32in \
     denoise_input_embed denoise_fp32head_cube denoise_fp32head_prep \
     gemm_qkv_2b_norm_gate_cache_bnsdq bnsd_o_proj_m768_bf16 \
     gemm_qkv_cache_bias_rope_m16 gemm_qkv_cache_bias_rope_prefill \
     gemm_qkv_vision_rope_finish gemm_glu_fp32_finish_split_11008

# ---------------------------------------------------------------- knife libraries
# One library per routed family; env names XINFER_<FAMILY>_LIB unchanged.

# SMALLM_R3 (both models; + the day-one try-on qkv kernel, flag default off)
gen smallm_r3 ops.gemm.smallm_r3 smallm_r3_gateup_glu smallm_r3_down smallm_r3_qkv
link libsmallm_r3.so smallm_r3 smallm_r3_gateup_glu smallm_r3_down smallm_r3_qkv

# ADANORM_R2 (wall2b)
gen adanorm_r2 ops.norm.adanorm_r2 adarmsnorm_action_double_bf16_f32 adarmsnorm_action_gated_bf16_f32
link libadanorm_r2.so adanorm_r2 adarmsnorm_action_double_bf16_f32 adarmsnorm_action_gated_bf16_f32

# SOUPUNLOCK (wall2b)
gen soupunlock ops.norm.soupunlock_kernels rmsnorm_full_c2048
link libsoupunlock.so soupunlock rmsnorm_full_c2048

# SOUPFORM (wall2b)
gen soupform ops.norm.soupform_kernels vision_qkv_rope_v3p
link libsoupform.so soupform vision_qkv_rope_v3p

# SOUPKILL (wall2b)
gen soupkill ops.finish.soupkill_r1 finish_br_c1024
link libsoupkill.so soupkill finish_br_c1024

# FINISHBF16 (wall2b)
gen finishbf16 ops.finish.finishbf16_kernels glu_split_bf16_c6144 zbasplit_bf16_c8224
link libfinishbf16.so finishbf16 glu_split_bf16_c6144 zbasplit_bf16_c8224

# SIMDEPI (walloss)
gen simdepi ops.norm.simdepi_r1 se1_ubv1280_buf2f se1_ubv2048_buf2f
link libsimdepi.so simdepi se1_ubv1280_buf2f se1_ubv2048_buf2f

# RMSFUSE_R4 (walloss)
gen rmsfuse_r4 ops.norm.rmsfuse_r4 rf4_m16 rf4_ubv2048_f
link librmsfuse_r4.so rmsfuse_r4 rf4_m16 rf4_ubv2048_f

# UBBFINISH_R5 (walloss)
gen ubfinish_r5 ops.finish.ubfinish_r5 ubfinish_r5_s8_c1280
link libubfinish_r5.so ubfinish_r5 ubfinish_r5_s8_c1280

# DECODEATTN_R4 (walloss)
gen decodeattn_r4 ops.attn.decodeattn_r4 decodeattn_r4_scan_b decodeattn_r4_reduce
link libdecodeattn_r4.so decodeattn_r4 decodeattn_r4_scan_b decodeattn_r4_reduce

# FIA_FUSE (wall2b)
gen fia_fuse ops.attn.fia_fuse fia_fuse_scan_l480 fia_fuse_reduce_l480
link libfia_fuse.so fia_fuse fia_fuse_scan_l480 fia_fuse_reduce_l480

# GDNV / MOVER2 / CONVGEMM (wall2b GDN wrapper stack)
gen gdnv ops.gdn.gdnv_kernels gdrprep_bf16
link libgdnv.so gdnv gdrprep_bf16
gen mover2 ops.gdn.mover2_kernels gdrapply_bf16 gdrtailmix_bf16
link libmove_r2.so mover2 gdrapply_bf16 gdrtailmix_bf16
gen conv_gemm ops.gdn.conv_gemm dconv4_c6144
link libconv_gemm.so conv_gemm dconv4_c6144

# SOUP2 (wall2b)
gen soup2 ops.elem.soup2_kernels qksq_c6144 rawsq_r128 quant_u8
link libsoup2.so soup2 qksq_c6144 rawsq_r128 quant_u8

# AA_RESAMPLE (wall2b)
gen aa_resample ops.resample.aa_resample_kernels \
    aa_v_720_1280_128_224 aa_h_720_1280_128_224 aa_v_480_640_160_224 aa_h_480_640_160_224
link libaa_resample.so aa_resample \
     aa_v_720_1280_128_224 aa_h_720_1280_128_224 aa_v_480_640_160_224 aa_h_480_640_160_224

# ---------------------------------------------------------------- receipts
( cd "$WORK" && find out -name "*.o" | sort | xargs sha256sum > SHA256SUMS.objects
  for l in lib*.so; do sha256sum "$l"; done | tee SHA256SUMS.libs )
echo "BUILD OK: products in $WORK (out/, lib*.so, SHA256SUMS.*)"

case "${1:-}" in
  install)
    mkdir -p "$HERE/lib"
    cp "$WORK"/lib*.so "$HERE/lib/"
    ( cd "$HERE/lib" && sha256sum lib*.so )
    echo "installed into kernels/lib/"
    ;;
esac

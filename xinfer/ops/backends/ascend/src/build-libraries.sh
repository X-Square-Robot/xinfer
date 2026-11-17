#!/bin/bash
# src/build-libraries.sh — the unified Ascend operator libraries (the transition form of
# src/CMakeLists.txt: one core library plus one knife library per routed family,
# built in-repo without xcompute CMake). Recipe = BUILD-SPIKE 2026-11-16
# (byte-for-byte contract): bisheng 9.3.0 compiles, CANN 9.1.1 links and loads,
# dav-3510, MAX_JOBS=1, the tilelang fork venv, TILELANG_CACHE_DIR isolated
# per invocation.
#
# Inputs (in this directory):  tools/{generate.py,abi.py}  <family dirs>
# Outputs (under $WORK):       out/<kernel>.{o,asc,tir} manifest.json  lib*.so
#
# Targets: one core library (the always-on class-entry kernels) plus one knife
# library per routed family (XINFER_<FAMILY>_LIB name preserved verbatim).
# `build-libraries.sh install` copies the products into ../lib/ (the loader's default path).
#
# Run from this directory (the aot/ layout contract: generate.py resolves the
# `abi` import and the `--module <family>.<module>` packages resolve from the
# src/ root while generate.py keeps its script-directory cwd).
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
             "see src/README.md (BUILD-SPIKE 2026-11-16)")
PYCHECK

gen() {  # gen <out-subdir> <module> <kernels...>
    local out="$WORK/out/$1"; shift
    mkdir -p "$out"
    # the ops/ packages sit next to build.sh (one level above tools/), so the
    # module roots resolve from the tree root while generate.py keeps its
    # script-directory cwd (the `from abi import ...` contract). generate.py
    # writes one manifest per invocation; successive calls for the same family
    # (the core library builds from several modules) merge into family.json.
    local manifest="$WORK/out/$1.family.json"
    PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "$PY" generate.py --output "$out"         --module "$1" --kernels "${@:2}"
    "$PY" - "$manifest" "$out/manifest.json" "$1" <<'MERGE'
import json, pathlib, sys
target, source, module = sys.argv[1], sys.argv[2], sys.argv[3]
merged = json.loads(pathlib.Path(target).read_text()) if pathlib.Path(target).exists() else {"kernels": {}}
fresh = json.loads(pathlib.Path(source).read_text())
merged.setdefault("tilelang", fresh.get("tilelang"))
merged.setdefault("arch", fresh.get("arch"))
for name, entry in fresh["kernels"].items():
    entry["module"] = module
    merged["kernels"][name] = entry
pathlib.Path(target).write_text(json.dumps(merged, indent=2) + chr(10))
MERGE
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
gen core norm.rms_apply          rms_apply_rstd_h2048
# The R5 plain-plan kernels (the rms_norm_cast mounts of the 38.82 stack, a22f865): the R5 delivery
# relinked them into the core out-of-band (r5-probes/r5_rebuild.sh); the tree build carries them
# in-band so the recipe reproduces the delivered core library alone.
gen core norm.rms_consistent     rms_consistent_ubv_h1280 rms_consistent_generic_h1024_f32in rms_consistent_ub_h1280 rms_consistent_generic_h1280 rms_consistent_ub_h2048 rms_consistent_generic_h1024
gen core denoise.denoise_output denoise_input_embed denoise_fp32head_cube denoise_fp32head_prep
gen core gemm.qkv_bnsdq          gemm_qkv_2b_norm_gate_cache_bnsdq
gen core gemm.gemm_smallm_action bnsd_o_proj_m768_bf16 smallm_bnsd_o_proj smallm_bnsd_o_proj_m32 smallm_bnsd_o_proj_m64
gen core gemm.gemm_qkv_cache     gemm_qkv_cache_bias_rope_m16 gemm_qkv_cache_bias_rope_prefill
gen core gemm.gemm_qkv_bf16_wide gemm_qkv_vision_rope_finish
gen core gemm.gemm_glu_fp32_finish gemm_glu_fp32_finish_split_11008 gemm_glu_fp32_finish_split_6144
gen core norm.qkv_bhsd qkv_split_vec_bhsd_h256
gen core norm.rope_bhsd rope_half_d64_bhsd
link libxinfer_ascend_core.so core \
     rms_apply_rstd_h2048 rms_consistent_ubv_h1280 rms_consistent_generic_h1024_f32in \
     rms_consistent_ub_h1280 rms_consistent_generic_h1280 rms_consistent_ub_h2048 rms_consistent_generic_h1024 \
     denoise_input_embed denoise_fp32head_cube denoise_fp32head_prep \
     gemm_qkv_2b_norm_gate_cache_bnsdq bnsd_o_proj_m768_bf16 \
     smallm_bnsd_o_proj smallm_bnsd_o_proj_m32 smallm_bnsd_o_proj_m64 \
     gemm_qkv_cache_bias_rope_m16 gemm_qkv_cache_bias_rope_prefill \
     gemm_qkv_vision_rope_finish gemm_glu_fp32_finish_split_11008 \
     gemm_glu_fp32_finish_split_6144 \
     qkv_split_vec_bhsd_h256 rope_half_d64_bhsd

# ---------------------------------------------------------------- knife libraries
# One library per routed family; env names XINFER_<FAMILY>_LIB unchanged.

# SMALLM_R3 (both models; + the day-one try-on qkv kernel, flag default off)
gen smallm_r3 gemm.smallm_r3 smallm_r3_o_proj smallm_r3_gateup_glu smallm_r3_down smallm_r3_qkv
link libsmallm_r3.so smallm_r3 smallm_r3_o_proj smallm_r3_gateup_glu smallm_r3_down smallm_r3_qkv

# ADANORM_R2 (wall2b; the _pinned twins serve the RTC denoise's pinned rows)
gen adanorm_r2 norm.adanorm_r2 adarmsnorm_action_double_bf16_f32 adarmsnorm_action_gated_bf16_f32 adarmsnorm_action_double_bf16_f32_pinned adarmsnorm_action_gated_bf16_f32_pinned
link libadanorm_r2.so adanorm_r2 adarmsnorm_action_double_bf16_f32 adarmsnorm_action_gated_bf16_f32 adarmsnorm_action_double_bf16_f32_pinned adarmsnorm_action_gated_bf16_f32_pinned

# SOUPUNLOCK (wall2b)
gen soupunlock norm.soupunlock_kernels rmsnorm_full_c2048
link libsoupunlock.so soupunlock rmsnorm_full_c2048

# SOUPFORM (wall2b)
gen soupform norm.soupform_kernels normprep_c2048 apply_rstd_c2048 zbasplit_c8224 mulcast_c6144 vision_qkv_rope_v3p
link libsoupform.so soupform normprep_c2048 apply_rstd_c2048 zbasplit_c8224 mulcast_c6144 vision_qkv_rope_v3p

# SOUPKILL (wall2b)
gen soupkill finish.soupkill_r1 finish_br_c1024
link libsoupkill.so soupkill finish_br_c1024

# FINISHBF16 (wall2b)
gen finishbf16 finish.finishbf16_kernels glu_split_bf16_c6144 zbasplit_bf16_c8224
link libfinishbf16.so finishbf16 glu_split_bf16_c6144 zbasplit_bf16_c8224

# SIMDEPI (walloss)
gen simdepi norm.simdepi_r1 se1_ubv1280_buf2f se1_ubv2048_buf2f
link libsimdepi.so simdepi se1_ubv1280_buf2f se1_ubv2048_buf2f

# RMSFUSE_R4 (walloss)
gen rmsfuse_r4 norm.rmsfuse_r4 rf4_m16 rf4_m16_f32in rf4_ubv2048_f
link librmsfuse_r4.so rmsfuse_r4 rf4_m16 rf4_m16_f32in rf4_ubv2048_f

# UBBFINISH_R5 (walloss; s4_c2048 mirrors the v22 registry's second entry)
gen ubfinish_r5 finish.ubfinish_r5 ubfinish_r5_s8_c1280 ubfinish_r5_s4_c2048
link libubfinish_r5.so ubfinish_r5 ubfinish_r5_s8_c1280 ubfinish_r5_s4_c2048

# DECODEATTN_R4 (walloss; reduce_row = the ROW=1 final-arm form, INVENTORY wl #11)
gen decodeattn_r4 attn.decodeattn_r4 decodeattn_r4_scan_b decodeattn_r4_reduce decodeattn_r4_reduce_row
link libdecodeattn_r4.so decodeattn_r4 decodeattn_r4_scan_b decodeattn_r4_reduce decodeattn_r4_reduce_row

# FIA_FUSE (wall2b; all four L forms the route's _BUILD_KEYS serves, plus the
# reduce_bnsd twins the v22 delivery carried)
gen fia_fuse attn.fia_fuse \
    fia_fuse_scan_l432 fia_fuse_scan_l448 fia_fuse_scan_l464 fia_fuse_scan_l480 \
    fia_fuse_reduce_l432 fia_fuse_reduce_l448 fia_fuse_reduce_l464 fia_fuse_reduce_l480 \
    fia_fuse_reduce_bnsd_l432 fia_fuse_reduce_bnsd_l448 fia_fuse_reduce_bnsd_l464 fia_fuse_reduce_bnsd_l480
link libfia_fuse.so fia_fuse \
    fia_fuse_scan_l432 fia_fuse_scan_l448 fia_fuse_scan_l464 fia_fuse_scan_l480 \
    fia_fuse_reduce_l432 fia_fuse_reduce_l448 fia_fuse_reduce_l464 fia_fuse_reduce_l480 \
    fia_fuse_reduce_bnsd_l432 fia_fuse_reduce_bnsd_l448 fia_fuse_reduce_bnsd_l464 fia_fuse_reduce_bnsd_l480

# GDNV / MOVER2 / CONVGEMM (wall2b GDN wrapper stack)
gen gdnv gdn.gdnv_kernels gdrprep_bf16
link libgdnv.so gdnv gdrprep_bf16
gen mover2 gdn.mover2_kernels gdrapply_bf16 gdrtailmix_bf16
link libmove_r2.so mover2 gdrapply_bf16 gdrtailmix_bf16
gen conv_gemm gdn.conv_gemm dconv4_c6144
link libconv_gemm.so conv_gemm dconv4_c6144

# SOUP2 (wall2b)
gen soup2 elem.soup2_kernels qksq_c6144 rawsq_r128 quant_u8
link libsoup2.so soup2 qksq_c6144 rawsq_r128 quant_u8

# AA_RESAMPLE (wall2b) -- the real6 asset objects, renamed and relinked.
# The vendored src/resample/aa_resample_kernels.py is the real10-state
# tree source: regenerating from it makes the auto-scheduler emit doubled
# MTE2/MTE3 notifies whose FIFO flag matching DEADLOCKS the vector core (the
# campaign's real9 pathology -- the source header documents it, and the v22
# objcheck adjudicated "the real6 objects are assets; the relink must
# reproduce"). The shipped libaa_resample-v22.so is a plain g++ -shared of the
# four real6 objects (sha 95bd4874...), so the unified branch links the SAME
# objects with the xinfer_aot_ symbol spelling (objcopy --redefine-syms over
# every symbol containing the old prefix; the device-kernel registration is
# symbol-address based, so the rename is internally consistent). Witness:
# bitwise-equal output vs the delivered library on the 720p v-kernel (R2
# probe, card 5, 2026-11-16). AA_REAL6_DIR overrides the asset location.
aa_relink() {
    local dir="${AA_REAL6_DIR:-/dev/shm/xcompute-parity-v22-20261110/aa-resample/build-real6/aot/aa_resample}"
    local ren="$WORK/out/aa_renamed"; mkdir -p "$ren"
    for k in aa_v_720_1280_128_224 aa_h_720_1280_128_224 aa_v_480_640_160_224 aa_h_480_640_160_224; do
        if [ ! -f "$dir/$k.o" ]; then
            echo "AA asset object missing: $dir/$k.o (set AA_REAL6_DIR)" >&2
            exit 2
        fi
        nm "$dir/$k.o" | awk '{print $3}' | grep 'xcompute_aot_' | sort -u \
            | while read -r s; do printf '%s %s\n' "$s" "${s/xcompute_aot_/xinfer_aot_}"; done \
            > "$ren/$k.map"
        objcopy --redefine-syms="$ren/$k.map" "$dir/$k.o" "$ren/$k.o"
    done
    g++ -shared -o "$WORK/libaa_resample.so" \
        "$ren/aa_v_720_1280_128_224.o" "$ren/aa_h_720_1280_128_224.o" \
        "$ren/aa_v_480_640_160_224.o" "$ren/aa_h_480_640_160_224.o" \
        -L"$RUNTIME/lib64" -lascendcl -lopapi -lruntime -lnnopbase
}
aa_relink

# ---------------------------------------------------------------- static bridges
# The lane-3 C-ABI bridges (ffi.h + cann_fia_v5 + cann_mm_fp32 + gdr_cann_chunk)
# compiled into the core library. torch_npu headers locate the stream helper.
STATIC_SRC_DIR="$HERE/common"
if compgen -G "$STATIC_SRC_DIR/*.cc" > /dev/null; then
    PYV=$(dirname "$(dirname "$PY")")
    TORCH_INC="$PYV/lib/python3.11/site-packages/torch/include"
    TORCH_LIB="$PYV/lib/python3.11/site-packages/torch/lib"
    NPU_LIB_DIR="$PYV/lib/python3.11/site-packages/torch_npu/lib"
    NPU_INC="$PYV/lib/python3.11/site-packages/torch_npu/include"
    PY_INC="$PYV/include"
    mkdir -p "$WORK/out/static"
    # build_info.cc is the single-library form's TVM-FFI self-description (CMakeLists builds it);
    # the family libraries are ctypes-loaded and do not carry it.
    for cc in "$STATIC_SRC_DIR"/*.cc; do
        [ "$(basename "$cc")" = "build_info.cc" ] && continue
        g++ -std=c++17 -O2 -fPIC \
            -I"$STATIC_SRC_DIR" -I"$RUNTIME/include" -I"$RUNTIME/aarch64-linux/include" \
            -I"$TORCH_INC" -I"$TORCH_INC/torch/csrc/api/include" -I"$NPU_INC" -I"$PY_INC" \
            -c "$cc" -o "$WORK/out/static/$(basename "${cc%.cc}").o"
    done
    g++ -shared -o "$WORK/libxinfer_ascend_core.so" \
        "$WORK"/out/static/*.o \
        "$WORK"/out/core/{rms_apply_rstd_h2048,rms_consistent_ubv_h1280,rms_consistent_generic_h1024_f32in,rms_consistent_ub_h1280,rms_consistent_generic_h1280,rms_consistent_ub_h2048,rms_consistent_generic_h1024,denoise_input_embed,denoise_fp32head_cube,denoise_fp32head_prep,gemm_qkv_2b_norm_gate_cache_bnsdq,bnsd_o_proj_m768_bf16,smallm_bnsd_o_proj,smallm_bnsd_o_proj_m32,smallm_bnsd_o_proj_m64,gemm_qkv_cache_bias_rope_m16,gemm_qkv_cache_bias_rope_prefill,gemm_qkv_vision_rope_finish,gemm_glu_fp32_finish_split_11008,gemm_glu_fp32_finish_split_6144,qkv_split_vec_bhsd_h256,rope_half_d64_bhsd}.o \
        -L"$RUNTIME/lib64" -lascendcl -lopapi -lruntime -lnnopbase \
        -L"$TORCH_LIB" -L"$NPU_LIB_DIR" -lc10 -ltorch_cpu -ltorch_npu
fi

# ---------------------------------------------------------------- receipts
( cd "$WORK" && find out -name "*.o" | sort | xargs sha256sum > SHA256SUMS.objects
  for l in lib*.so; do sha256sum "$l"; done | tee SHA256SUMS.libs )
echo "BUILD OK: products in $WORK (out/, lib*.so, SHA256SUMS.*)"

case "${1:-}" in
  install)
    mkdir -p "$HERE/../lib"
    cp "$WORK"/lib*.so "$HERE/../lib/"
    ( cd "$HERE/../lib" && sha256sum lib*.so )
    echo "installed into ascend/lib/"
    ;;
esac

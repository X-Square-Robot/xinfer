# The native kernel library of xinfer's Ascend backend: xinfer_kernels_ascend.so, loaded by ../_native.py.
#
#   pip install -e ".[kernels]"          # installs the TVM-FFI runtime only; pip never compiles
#   cmake -S xinfer/ops/backends/ascend/src -B build/kernels-ascend
#   cmake --build build/kernels-ascend -j1
#
# writes xinfer_kernels_ascend.so next to the Python modules that call it (xinfer/ops/backends/ascend/).
# This runs on the aarch64 build host (openEuler 24.03) with CANN; the full toolchain story, the validated
# version pairing and the environment-order trap are in README.md next to this file.
#
# The transition family libraries (one core library plus one knife library per routed family - the form
# the delivered 20.62/38.82 stacks load) are built by build-libraries.sh in this directory; this CMake
# aggregates the same AOT objects into the one library. R1b retires the family form when the per-kernel
# TVM-FFI glue faces land.
cmake_minimum_required(VERSION 3.24)
project(xinfer_kernels_ascend LANGUAGES CXX)

# The SoC to build for, as the vendor's device name lowercased to alphanumerics: ascend950dt9582 (the string
# torch_npu's device properties and xinfer/platforms/npu.py report as the platform's arch).
# Unset: the SoC of this machine's first NPU, when npu-smi names one; the build host usually has no NPU, so
# pass -DXINFER_ASCEND_SOC=ascend950dt9582 there.
set(XINFER_ASCEND_SOC "" CACHE STRING "Target SoC, e.g. ascend950dt9582")
if(NOT XINFER_ASCEND_SOC)
  execute_process(COMMAND npu-smi info OUTPUT_VARIABLE _smi OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
                  RESULT_VARIABLE _smi_result)
  if(_smi_result EQUAL 0)
    string(REGEX MATCH "[Aa]scend[ \t]*9[0-9][A-Za-z0-9]*" _name "${_smi}")
    string(REGEX REPLACE "[ \t]" "" _name "${_name}")
    string(TOLOWER "${_name}" XINFER_ASCEND_SOC)
  endif()
endif()
set(XINFER_SUPPORTED_SOCS ascend950dt9582)
if(NOT XINFER_ASCEND_SOC IN_LIST XINFER_SUPPORTED_SOCS)
  message(FATAL_ERROR "The Ascend backend has kernels for the SoCs ${XINFER_SUPPORTED_SOCS}; got "
                      "'${XINFER_ASCEND_SOC}'. Pass -DXINFER_ASCEND_SOC=<soc>.")
endif()
# The compile target the TileLang AOT generator passes to bisheng for each SoC (dav-3510 is the 950DT device
# target). Host glue does not use it; the generator invocation does.
set(XINFER_SOC_COMPILE_TARGET_ascend950dt9582 "dav-3510")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(FATAL_ERROR "The Ascend kernel library builds on aarch64 (openEuler 24.03); this is "
                      "${CMAKE_SYSTEM_PROCESSOR}.")
endif()

# The validated toolchain pairing: bisheng 9.3.0 compiles (the TileLang-generated host C and any AscendC
# sources), the CANN 9.1.1 runtime links and loads. Source the 9.1.1 set_env.sh before anything imports
# TorchNPU, then point BISHENG_HOME / ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME at 9.3.0 - the reverse order
# breaks the TorchNPU import (README.md, "the validated pairing").
set(XINFER_CANN_RUNTIME "/usr/local/Ascend/cann-9.1.1" CACHE PATH "CANN runtime root (link and load)")
set(XINFER_BISHENG_HOME "/usr/local/Ascend/cann-9.3.0" CACHE PATH "bisheng device compiler root")
if(NOT EXISTS "${XINFER_BISHENG_HOME}/bin/bisheng")
  message(FATAL_ERROR "bisheng not found at ${XINFER_BISHENG_HOME}/bin/bisheng: pass "
                      "-DXINFER_BISHENG_HOME=<a CANN with bisheng>.")
endif()
foreach(_lib ascendcl opapi runtime nnopbase)
  find_library(XINFER_${_lib} NAMES ${_lib} REQUIRED HINTS "${XINFER_CANN_RUNTIME}/lib64")
endforeach()

find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
                OUTPUT_VARIABLE tvm_ffi_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(tvm_ffi CONFIG REQUIRED)

# The AOT generation step (landed R1a): the generator pair (tools/generate.py + tools/abi.py, the
# build-spike contract) lowers each family's TileLang modules into .o files - device image embedded,
# loaded by aclrtBinaryLoadFromData - with the host launcher symbols renamed to xinfer_aot_<kernel>_.
# One custom command per module, exactly as build-libraries.sh drives generate.py; MAX_JOBS=1 is the
# memory discipline of the shared hosts and TILELANG_CACHE_DIR stays private to this build.
set(XINFER_TILELANG_PYTHON "/root/tl-fork-store/venv-tl-src/bin/python" CACHE FILEPATH
    "Python of the TileLang fork venv that runs the AOT generator")
set(XINFER_AOT_OBJECT_DIR "" CACHE PATH
    "Directory of prebuilt AOT objects to consume instead of generating (e.g. the r5-lib out/ tree; empty = generate)")

# xinfer_aot_module(<group> <module> <kernel>...): declare the custom command for one family module and
# append its objects to _AOT_OBJECTS. <group> is the family-library target of the transition build (the
# build-libraries.sh gen output subdir: core, smallm_r3, soup2, ...), so the generated layout and the
# prebuilt object source's layout agree: XINFER_AOT_OBJECT_DIR/<group>/<k>.o (the r5-lib out/ tree form).
set(_AOT_OBJECTS "")
function(xinfer_aot_module group module)
  set(_kernels "${ARGN}")
  set(_generated "")
  foreach(_k IN LISTS _kernels)
    list(APPEND _generated "${CMAKE_BINARY_DIR}/aot/${group}/${_k}.o")
  endforeach()
  if(XINFER_AOT_OBJECT_DIR)
    foreach(_k IN LISTS _kernels)
      if(EXISTS "${XINFER_AOT_OBJECT_DIR}/${group}/${_k}.o")
        list(APPEND _AOT_OBJECTS "${XINFER_AOT_OBJECT_DIR}/${group}/${_k}.o")
      else()
        message(FATAL_ERROR "XINFER_AOT_OBJECT_DIR does not carry ${group}/${_k}.o")
      endif()
    endforeach()
  else()
    add_custom_command(OUTPUT ${_generated}
      COMMAND /bin/bash ${PROJECT_SOURCE_DIR}/tools/aot-generate.sh ${CMAKE_BINARY_DIR}/aot/${group} ${module} ${_kernels}
      DEPENDS tools/generate.py tools/abi.py tools/aot-generate.sh
      VERBATIM)
    list(APPEND _AOT_OBJECTS ${_generated})
  endif()
  set(_AOT_OBJECTS "${_AOT_OBJECTS}" PARENT_SCOPE)
endfunction()

# The 25 generated module invocations of the family map (src/FAMILIES.md; build-libraries.sh's gen
# calls, same module roots and kernel lists - the byte-anchor recipe).
xinfer_aot_module(core norm.rms_apply rms_apply_rstd_h2048)
xinfer_aot_module(core norm.rms_consistent rms_consistent_ubv_h1280 rms_consistent_generic_h1024_f32in
                  rms_consistent_ub_h1280 rms_consistent_generic_h1280 rms_consistent_ub_h2048 rms_consistent_generic_h1024)
xinfer_aot_module(core denoise.denoise_output denoise_input_embed denoise_fp32head_cube denoise_fp32head_prep)
xinfer_aot_module(core gemm.qkv_bnsdq gemm_qkv_2b_norm_gate_cache_bnsdq)
xinfer_aot_module(core gemm.gemm_smallm_action bnsd_o_proj_m768_bf16 smallm_bnsd_o_proj smallm_bnsd_o_proj_m32 smallm_bnsd_o_proj_m64)
xinfer_aot_module(core gemm.gemm_qkv_cache gemm_qkv_cache_bias_rope_m16 gemm_qkv_cache_bias_rope_prefill)
xinfer_aot_module(core gemm.gemm_qkv_bf16_wide gemm_qkv_vision_rope_finish)
xinfer_aot_module(core gemm.gemm_glu_fp32_finish gemm_glu_fp32_finish_split_11008 gemm_glu_fp32_finish_split_6144)
xinfer_aot_module(core norm.qkv_bhsd qkv_split_vec_bhsd_h256)
xinfer_aot_module(core norm.rope_bhsd rope_half_d64_bhsd)
xinfer_aot_module(smallm_r3 gemm.smallm_r3 smallm_r3_o_proj smallm_r3_gateup_glu smallm_r3_down smallm_r3_qkv)
xinfer_aot_module(adanorm_r2 norm.adanorm_r2 adarmsnorm_action_double_bf16_f32 adarmsnorm_action_gated_bf16_f32 adarmsnorm_action_double_bf16_f32_pinned adarmsnorm_action_gated_bf16_f32_pinned)
xinfer_aot_module(soupunlock norm.soupunlock_kernels rmsnorm_full_c2048)
xinfer_aot_module(soupform norm.soupform_kernels normprep_c2048 apply_rstd_c2048 zbasplit_c8224 mulcast_c6144 vision_qkv_rope_v3p)
xinfer_aot_module(soupkill finish.soupkill_r1 finish_br_c1024)
xinfer_aot_module(finishbf16 finish.finishbf16_kernels glu_split_bf16_c6144 zbasplit_bf16_c8224)
xinfer_aot_module(simdepi norm.simdepi_r1 se1_ubv1280_buf2f se1_ubv2048_buf2f)
xinfer_aot_module(rmsfuse_r4 norm.rmsfuse_r4 rf4_m16 rf4_m16_f32in rf4_ubv2048_f)
xinfer_aot_module(ubfinish_r5 finish.ubfinish_r5 ubfinish_r5_s8_c1280 ubfinish_r5_s4_c2048)
xinfer_aot_module(decodeattn_r4 attn.decodeattn_r4 decodeattn_r4_scan_b decodeattn_r4_reduce decodeattn_r4_reduce_row)
xinfer_aot_module(fia_fuse attn.fia_fuse
  fia_fuse_scan_l432 fia_fuse_scan_l448 fia_fuse_scan_l464 fia_fuse_scan_l480
  fia_fuse_reduce_l432 fia_fuse_reduce_l448 fia_fuse_reduce_l464 fia_fuse_reduce_l480
  fia_fuse_reduce_bnsd_l432 fia_fuse_reduce_bnsd_l448 fia_fuse_reduce_bnsd_l464 fia_fuse_reduce_bnsd_l480)
xinfer_aot_module(gdnv gdn.gdnv_kernels gdrprep_bf16)
xinfer_aot_module(mover2 gdn.mover2_kernels gdrapply_bf16 gdrtailmix_bf16)
xinfer_aot_module(conv_gemm gdn.conv_gemm dconv4_c6144)
xinfer_aot_module(soup2 elem.soup2_kernels qksq_c6144 rawsq_r128 quant_u8)

# The AA-resample real6 objects are assets, not regenerable sources (the auto-scheduler deadlock the
# campaign adjudicated: the real6 objects are the delivered form, renamed and relinked - never
# regenerated; build-libraries.sh carries the full story). They are consumed from the asset directory
# with the xcompute_aot_ -> xinfer_aot_ symbol rename; XINFER_AA_REAL6_DIR overrides the location.
set(XINFER_AA_REAL6_DIR "/dev/shm/xcompute-parity-v22-20261110/aa-resample/build-real6/aot/aa_resample" CACHE PATH
    "The delivered AA-resample real6 asset objects")
set(_AA_KERNELS aa_v_720_1280_128_224 aa_h_720_1280_128_224 aa_v_480_640_160_224 aa_h_480_640_160_224)
set(_AA_OBJECTS "")
foreach(_k IN LISTS _AA_KERNELS)
  if(NOT EXISTS "${XINFER_AA_REAL6_DIR}/${_k}.o")
    message(FATAL_ERROR "AA real6 asset object missing: ${XINFER_AA_REAL6_DIR}/${_k}.o (set XINFER_AA_REAL6_DIR)")
  endif()
  set(_renamed "${CMAKE_BINARY_DIR}/aa_renamed/${_k}.o")
  add_custom_command(OUTPUT "${_renamed}"
    COMMAND /bin/bash ${PROJECT_SOURCE_DIR}/tools/aa-rename.sh "${XINFER_AA_REAL6_DIR}/${_k}.o" "${_renamed}"
    DEPENDS "${XINFER_AA_REAL6_DIR}/${_k}.o" tools/aa-rename.sh
    VERBATIM)
  list(APPEND _AA_OBJECTS "${_renamed}")
endforeach()

# Operators that run on every SoC: the build information and the C-ABI static bridges (the composition
# bridges of the core family library - cann_fia_v5, cann_mm_fp32, gdr_cann_chunk - vendored from the v22
# repo with the tvm_ffi TensorView machinery dropped; they resolve the stream through torch_npu, so the
# TorchNPU headers of the TileLang venv are needed to compile them).
set(XINFER_TORCH_ROOT "" CACHE PATH "The venv site-packages root carrying torch/ and torch_npu/ for the static bridges")
set(XINFER_TORCH_ROOT "" CACHE PATH "The venv site-packages root carrying torch/ and torch_npu/ for the static bridges")
if(NOT XINFER_TORCH_ROOT)
  get_filename_component(_venv "${XINFER_TILELANG_PYTHON}" DIRECTORY)
  get_filename_component(_venv "${_venv}" DIRECTORY)
  set(XINFER_TORCH_ROOT "${_venv}/lib/python3.11/site-packages")
endif()
set(KERNEL_SOURCES
  common/build_info.cc
  common/cann_fia_v5.cc
  common/cann_mm_fp32.cc
  common/gdr_cann_chunk.cc
)
# Kernels of one SoC: the AOT objects of that SoC's compile target, selected by variable name exactly as
# the CUDA side selects KERNEL_SM<arch>_SOURCES - a new SoC needs one new list and one entry in
# XINFER_SUPPORTED_SOCS, nothing else. The objects are the same ones build-libraries.sh links into the
# 17 transition family libraries; the per-kernel numeric contracts stay in the TileLang sources where a
# bitwise result is specified (no bisheng equivalent of the CUDA IEEE flags). The per-family TVM-FFI
# glue faces (the aot_glue.cc files, generated by tools/aot_glue_gen.py from the generator manifests)
# marshal TensorView arguments onto each kernel's packed-launch ABI - the single library's FFI surface,
# the form the backend's unified launch boundary calls; they are SoC-listed because they call the SoC's
# AOT launchers, and the arch OBJECT library compiles them (the _SOC_COMPILABLE filter below).
string(TOUPPER "${XINFER_ASCEND_SOC}" _SOC_UC)
set(KERNEL_${_SOC_UC}_SOURCES
  gemm/aot_glue.cc
  norm/aot_glue.cc
  attn/aot_glue.cc
  finish/aot_glue.cc
  gdn/aot_glue.cc
  denoise/aot_glue.cc
  elem/aot_glue.cc
  resample/aot_glue.cc
  ${_AOT_OBJECTS}
  ${_AA_OBJECTS}
)

# The two-list split is the CUDA shape (KERNEL_SOURCES for every SoC, KERNEL_<SOC>_SOURCES selected by
# variable name - a new SoC needs one new list and one entry in XINFER_SUPPORTED_SOCS, nothing else).
# The CUDA side's arch OBJECT library exists to compile arch-conditional sources; this SoC list carries
# the AOT objects (the generator's or the prebuilt object source's outputs - link inputs, not sources to
# compile), so they feed the SHARED target directly, and the OBJECT library appears when a SoC list gains
# compilable glue (R1b's per-family TVM-FFI .cc files).
set(_SOC_COMPILABLE "")
foreach(_s IN LISTS KERNEL_${_SOC_UC}_SOURCES)
  if(_s MATCHES "\\.(cc|cpp|cxx|c)$")
    list(APPEND _SOC_COMPILABLE ${_s})
  endif()
endforeach()

function(kernels_target_setup target)
  target_compile_definitions(${target} PRIVATE "XINFER_TARGET_SOCS=\"${XINFER_ASCEND_SOC}\"")
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/common
    ${XINFER_CANN_RUNTIME}/include ${XINFER_CANN_RUNTIME}/aarch64-linux/include
    ${XINFER_TORCH_ROOT}/torch/include ${XINFER_TORCH_ROOT}/torch/include/torch/csrc/api/include
    ${XINFER_TORCH_ROOT}/torch_npu/include)
  target_link_libraries(${target} PRIVATE tvm_ffi::header)
endfunction()

set(_ARCH_OBJECTS "")
if(_SOC_COMPILABLE)
  add_library(xinfer_kernels_ascend_arch OBJECT ${_SOC_COMPILABLE})
  kernels_target_setup(xinfer_kernels_ascend_arch)
  set(_ARCH_OBJECTS "$<TARGET_OBJECTS:xinfer_kernels_ascend_arch>")
endif()

add_library(xinfer_kernels_ascend SHARED
  ${KERNEL_SOURCES} ${KERNEL_${_SOC_UC}_SOURCES} ${_ARCH_OBJECTS})
kernels_target_setup(xinfer_kernels_ascend)
target_link_libraries(xinfer_kernels_ascend PRIVATE
  tvm_ffi::shared ${XINFER_ascendcl} ${XINFER_opapi} ${XINFER_runtime} ${XINFER_nnopbase})
set_target_properties(xinfer_kernels_ascend PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/..")

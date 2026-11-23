# Single-library build of the existing Wall-OSS and Wall-X 2B kernels.
# Toolchain and source requirements: README.md.
cmake_minimum_required(VERSION 3.24)
project(xinfer_kernels_ascend LANGUAGES CXX)

set(XINFER_ASCEND_SOC "" CACHE STRING "Target Ascend device family, e.g. ascend950dt")
if(NOT XINFER_ASCEND_SOC)
  execute_process(COMMAND npu-smi info OUTPUT_VARIABLE _smi OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
                  RESULT_VARIABLE _smi_result)
  if(_smi_result EQUAL 0)
    string(REGEX MATCH "[Aa]scend[ \t]*9[0-9][A-Za-z0-9]*" _name "${_smi}")
    string(REGEX REPLACE "[ \t]" "" _name "${_name}")
    string(TOLOWER "${_name}" XINFER_ASCEND_SOC)
  endif()
endif()
if(XINFER_ASCEND_SOC STREQUAL "ascend950dt")
  set(XINFER_ASCEND_SOC "ascend950dt9582")
endif()
set(XINFER_SUPPORTED_SOCS ascend950dt9582)
if(NOT XINFER_ASCEND_SOC IN_LIST XINFER_SUPPORTED_SOCS)
  message(FATAL_ERROR "This build supports Ascend 950DT. Pass -DXINFER_ASCEND_SOC=ascend950dt.")
endif()
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

set(_cann_root "$ENV{ASCEND_TOOLKIT_HOME}")
if(NOT _cann_root)
  set(_cann_root "/usr/local/Ascend/cann")
endif()
set(XINFER_CANN_RUNTIME "${_cann_root}" CACHE PATH "CANN runtime root (link and load)")
set(XINFER_BISHENG_HOME "${XINFER_CANN_RUNTIME}" CACHE PATH "bisheng device compiler root")
if(NOT EXISTS "${XINFER_BISHENG_HOME}/bin/bisheng")
  message(FATAL_ERROR "bisheng not found at ${XINFER_BISHENG_HOME}/bin/bisheng: pass "
                      "-DXINFER_BISHENG_HOME=<a CANN with bisheng>.")
endif()
set(_cann_lib_dir "${XINFER_CANN_RUNTIME}/lib64")
find_library(XINFER_opapi NAMES opapi PATHS "${_cann_lib_dir}" NO_DEFAULT_PATH)
set(_cann_libs ascendcl)
if(XINFER_opapi)
  list(APPEND _cann_libs opapi)
else()
  list(APPEND _cann_libs opapi_nn opapi_transformer)
endif()
list(APPEND _cann_libs runtime nnopbase)
foreach(_lib IN LISTS _cann_libs)
  find_library(XINFER_${_lib} NAMES ${_lib} REQUIRED PATHS "${_cann_lib_dir}" NO_DEFAULT_PATH)
  list(APPEND _cann_link "${XINFER_${_lib}}")
endforeach()
# Newer BiSheng host stubs call the profiling helpers from the Ascend C runtime.
find_library(XINFER_ascendc_runtime NAMES ascendc_runtime PATHS "${_cann_lib_dir}" NO_DEFAULT_PATH)
if(XINFER_ascendc_runtime)
  list(APPEND _cann_link "${XINFER_ascendc_runtime}")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
                OUTPUT_VARIABLE tvm_ffi_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(tvm_ffi CONFIG REQUIRED)

set(XINFER_TILELANG_PYTHON "${Python_EXECUTABLE}" CACHE FILEPATH
    "Python of the TileLang fork venv that runs the AOT generator")

set(_TL_INCLUDE "${PROJECT_SOURCE_DIR}/third_party/tilelang/include")
file(GLOB _TL_HEADERS CONFIGURE_DEPENDS "${_TL_INCLUDE}/tl_templates/ascend/*.h")
file(GLOB_RECURSE _AOT_PYTHON_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/*.py")
set(_AOT_OBJECTS "")
function(xinfer_aot_module group module)
  set(_kernels "${ARGN}")
  set(_generated "")
  foreach(_k IN LISTS _kernels)
    list(APPEND _generated "${CMAKE_BINARY_DIR}/aot/${group}/${_k}.o")
  endforeach()
  add_custom_command(OUTPUT ${_generated}
    COMMAND ${CMAKE_COMMAND} -E env
      "XINFER_CANN_RUNTIME=${XINFER_CANN_RUNTIME}"
      "XINFER_BISHENG_HOME=${XINFER_BISHENG_HOME}"
      "XINFER_TILELANG_PYTHON=${XINFER_TILELANG_PYTHON}"
      /bin/bash ${PROJECT_SOURCE_DIR}/tools/aot-generate.sh ${CMAKE_BINARY_DIR}/aot/${group} ${module} ${_kernels}
    DEPENDS tools/generate.py tools/abi.py tools/aot-generate.sh ${_AOT_PYTHON_SOURCES} ${_TL_HEADERS}
    VERBATIM)
  list(APPEND _AOT_OBJECTS ${_generated})
  set(_AOT_OBJECTS "${_AOT_OBJECTS}" PARENT_SCOPE)
endfunction()

xinfer_aot_module(core norm.rms_apply rms_apply_rstd_h2048)
xinfer_aot_module(core norm.rms_consistent rms_consistent_ubv_h1280 rms_consistent_generic_h1024_f32in
                  rms_consistent_ub_h1280 rms_consistent_generic_h1280 rms_consistent_ub_h2048 rms_consistent_generic_h1024)
xinfer_aot_module(core denoise.denoise_output denoise_input_embed denoise_fp32head_cube denoise_fp32head_prep)
xinfer_aot_module(core gemm.qkv_bnsdq gemm_qkv_2b_norm_gate_cache_bnsdq)
xinfer_aot_module(core gemm.gemm_smallm_action bnsd_o_proj_m768_bf16 smallm_bnsd_o_proj smallm_bnsd_o_proj_m32 smallm_bnsd_o_proj_m64)
xinfer_aot_module(core gemm.gemm_qkv_cache gemm_qkv_cache_bias_rope_m16 gemm_qkv_cache_bias_rope_prefill)
xinfer_aot_module(core gemm.gemm_qkv_bf16_wide gemm_qkv_vision_rope_finish)
xinfer_aot_module(core gemm.gemm_glu_fp32_finish gemm_glu_fp32_finish_split_2048
                  gemm_glu_fp32_finish_split_11008 gemm_glu_fp32_finish_split_6144)
xinfer_aot_module(attn_pack attn.gated_attn_pack_pow2 gated_attn_bnsd_pack_pow2)
xinfer_aot_module(core norm.qkv_bhsd qkv_split_vec_bhsd_h256)
xinfer_aot_module(core norm.rope_bhsd rope_half_d64_bhsd)
xinfer_aot_module(action_gemm_m16 gemm.action_gemm_m16 action_gemm_m16_o_proj action_gemm_m16_gateup_glu action_gemm_m16_down action_gemm_m16_qkv)
xinfer_aot_module(adaptive_rmsnorm norm.adaptive_rmsnorm adarmsnorm_action_double_bf16_f32 adarmsnorm_action_gated_bf16_f32 adarmsnorm_action_double_bf16_f32_pinned adarmsnorm_action_gated_bf16_f32_pinned)
xinfer_aot_module(fused_norm_reduction norm.fused_norm_reduction_kernels rmsnorm_full_c2048)
xinfer_aot_module(projection_norm_epilogue norm.projection_norm_epilogue_kernels normprep_c2048 apply_rstd_c2048 zbasplit_c8224 mulcast_c6144 vision_qkv_rope_packed)
xinfer_aot_module(fp32_accumulator_epilogue finish.fp32_accumulator_epilogue bias_residual_cast_c1024)
xinfer_aot_module(finishbf16 finish.finishbf16_kernels glu_split_bf16_c6144 zbasplit_bf16_c8224)
xinfer_aot_module(buffered_add_rmsnorm norm.buffered_add_rmsnorm add_rmsnorm_h1280_double_buffer_fp32_weight add_rmsnorm_h2048_double_buffer_fp32_weight)
xinfer_aot_module(fused_rmsnorm norm.fused_rmsnorm rmsnorm_m16_h1024 rmsnorm_m16_h1024_fp32_input rmsnorm_unified_buffer_h2048_fp32_weight)
xinfer_aot_module(tiled_bias_epilogue finish.tiled_bias_epilogue bias_add_cast_simd_rows8_c1280 bias_add_cast_simd_rows4_c2048)
xinfer_aot_module(split_kv_decode attn.split_kv_decode split_kv_decode_scan_tile_per_block split_kv_decode_reduce_head_major split_kv_decode_reduce_token_major)
xinfer_aot_module(fia_fuse attn.fia_fuse
  fia_fuse_scan_l432 fia_fuse_scan_l448 fia_fuse_scan_l464 fia_fuse_scan_l480
  fia_fuse_reduce_l432 fia_fuse_reduce_l448 fia_fuse_reduce_l464 fia_fuse_reduce_l480
  fia_fuse_reduce_bnsd_l432 fia_fuse_reduce_bnsd_l448 fia_fuse_reduce_bnsd_l464 fia_fuse_reduce_bnsd_l480)
xinfer_aot_module(gdnv gdn.gdnv_kernels gdrprep_bf16)
xinfer_aot_module(gdn_layout gdn.gdn_layout_kernels gdrapply_bf16 gdrtailmix_bf16)
xinfer_aot_module(conv_gemm gdn.conv_gemm dconv4_c6144)
xinfer_aot_module(square_cast elem.square_cast_kernels qksq_c6144 rawsq_r128)

set(XINFER_TORCH_ROOT "" CACHE PATH "The venv site-packages root carrying torch/ and torch_npu/ for the static bridges")
if(NOT XINFER_TORCH_ROOT)
  execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent.parent)"
    OUTPUT_VARIABLE XINFER_TORCH_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
endif()

set(XINFER_TORCH_LIBS "" CACHE PATH "The torch/torch_npu library root pair (torch/lib, torch_npu/lib)")
if(NOT XINFER_TORCH_LIBS)
  set(XINFER_TORCH_LIBS "${XINFER_TORCH_ROOT}")
endif()
foreach(_lib torch_npu torch_cpu c10)
  find_library(XINFER_${_lib} NAMES lib${_lib}.so REQUIRED
               HINTS "${XINFER_TORCH_LIBS}/torch_npu/lib" "${XINFER_TORCH_LIBS}/torch/lib")
endforeach()
set(KERNEL_SOURCES
  common/build_info.cc
  common/cann_fia_v5.cc
  common/cann_mm_fp32.cc
  common/gdr_cann_chunk.cc
)
string(TOUPPER "${XINFER_ASCEND_SOC}" _SOC_UC)
set(KERNEL_${_SOC_UC}_SOURCES
  gemm/aot_glue.cc
  norm/aot_glue.cc
  attn/aot_glue.cc
  finish/aot_glue.cc
  gdn/aot_glue.cc
  denoise/aot_glue.cc
  elem/aot_glue.cc
  ${_AOT_OBJECTS}
)

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

set(_SOC_LINK_INPUTS "")
foreach(_s IN LISTS KERNEL_${_SOC_UC}_SOURCES)
  if(NOT _s MATCHES "\\.(cc|cpp|cxx|c)$")
    list(APPEND _SOC_LINK_INPUTS ${_s})
  endif()
endforeach()

add_library(xinfer_kernels_ascend SHARED
  ${KERNEL_SOURCES} ${_SOC_LINK_INPUTS} ${_ARCH_OBJECTS})
kernels_target_setup(xinfer_kernels_ascend)
target_link_libraries(xinfer_kernels_ascend PRIVATE
  tvm_ffi::shared ${_cann_link}
  ${XINFER_torch_npu} ${XINFER_torch_cpu} ${XINFER_c10})
set_target_properties(xinfer_kernels_ascend PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/..")

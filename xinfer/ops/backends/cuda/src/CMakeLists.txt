# The native kernel library of xinfer's CUDA backend: xinfer_kernels.so, loaded by ../_native.py.
#
#   cmake -S xinfer/ops/backends/cuda/src -B build/kernels && cmake --build build/kernels -j
#
# writes xinfer_kernels.so next to the Python modules that call it (xinfer/ops/backends/cuda/).
cmake_minimum_required(VERSION 3.24)
project(xinfer_kernels LANGUAGES CXX CUDA)

# The GPU architecture to build for, as the compute capability without the dot: 120 (RTX 50 series) or 110
# (Jetson Thor).
# Unset: the architecture of this machine's first GPU.
set(XINFER_CUDA_ARCH "" CACHE STRING "Target compute capability without the dot, e.g. 120")
if(NOT XINFER_CUDA_ARCH)
  execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                  OUTPUT_VARIABLE _caps OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  string(REGEX MATCH "^[0-9]+\\.[0-9]" _cap "${_caps}")
  string(REPLACE "." "" XINFER_CUDA_ARCH "${_cap}")
endif()
set(XINFER_SUPPORTED_ARCHS 110 120)
if(NOT XINFER_CUDA_ARCH IN_LIST XINFER_SUPPORTED_ARCHS)
  message(FATAL_ERROR "The CUDA backend has kernels for compute capability ${XINFER_SUPPORTED_ARCHS}; got "
                      "'${XINFER_CUDA_ARCH}'. Pass -DXINFER_CUDA_ARCH=<capability>.")
endif()
if(XINFER_CUDA_ARCH STREQUAL "110" AND CMAKE_VERSION VERSION_LESS 3.28)
  message(FATAL_ERROR "Building for compute capability 11.0 (sm_110a) needs CMake 3.28 or later")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
                OUTPUT_VARIABLE tvm_ffi_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(tvm_ffi CONFIG REQUIRED)

get_filename_component(XINFER_ROOT "${PROJECT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(XINFER_CUTLASS_ROOT "${XINFER_ROOT}/third_party/cutlass" CACHE PATH "CUTLASS source tree")
if(NOT EXISTS "${XINFER_CUTLASS_ROOT}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "CUTLASS not found at ${XINFER_CUTLASS_ROOT}: run git submodule update --init, or pass "
                      "-DXINFER_CUTLASS_ROOT=<a CUTLASS checkout>")
endif()

# Operators that run on every architecture.
set(KERNEL_SOURCES
  common/build_info.cc
  gemm/generic.cc
  gemm/epilogue.cu
  norm/rmsnorm.cu
  norm/double_adarmsnorm.cu
  norm/layernorm.cu
  norm/qkv_norm_split.cu
  norm/rmsnorm_quant_mxfp8.cu
  quant/mxfp8_quantization.cu
  quant/nvfp4_quantization.cu
  position/rope.cu
  position/rot_pos.cu
  position/pos_embed_interp.cu
  activation/sigmoid_gate_mul.cu
  attention/attention_kvcache_splitkv.cu
  gated_delta_rule/gated_delta_rule_prefill.cu
  gated_delta_rule/gated_delta_rule_prefill_workspace.cu
  gated_delta_rule/gated_delta_rule_prefill_chunk32.cu
  denoise_output/denoise_input_embed.cu
  denoise_output/denoise_output_euler.cu
  denoise_output/denoise_output_euler_kernel.cu
  prefetch/l2_prefetch.cu
)
# Kernels of one architecture, compiled with its architecture-specific instructions (sm_<arch>a).
set(KERNEL_SM110_SOURCES
  gemm/sm110/bf16_qkv_rope.cu
  gemm/sm110/bf16_gemm_silu_mul.cu
  gemm/sm110/mcast_bf16_sm110.cu
  gemm/sm110/mcast_ffn_sm110.cu
  gemm/sm110/mcast_sm110_ffi.cc
  norm/sm110/bf16_rmsnorm_sm110.cu
)
set(KERNEL_SM120_SOURCES
  gemm/sm120/smallm_bf16_sm120.cu
  gemm/sm120/streamk_bf16_sm120.cu
  gemm/sm120/interleaved_streamk_gelu_bf16_sm120.cu
  gemm/sm120/tma_tile_bf16_sm120.cu
  gemm/sm120/tma_streamk_glu_bf16_sm120.cu
  gemm/sm120/gateup_silu.cu
  gemm/sm120/tile_mxfp8_sm120.cu
  gemm/sm120/nvfp4_cutlass_sm120.cu
  attention/sm120/attention_varlen_lpt_bf16_sm120.cu
)
# Kernels whose results are specified bit for bit against torch keep IEEE arithmetic where the rest of the library
# builds with --use_fast_math: no FMA contraction, exact division and square root, denormals kept.
set_source_files_properties(position/rope.cu position/pos_embed_interp.cu PROPERTIES COMPILE_OPTIONS
  "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-prec-div=true;-prec-sqrt=true;--ftz=false>")
# The same without the FMA part (the kernel's sums are written for the default contraction).
set_source_files_properties(denoise_output/denoise_input_embed.cu PROPERTIES COMPILE_OPTIONS
  "$<$<COMPILE_LANGUAGE:CUDA>:-prec-div=true;-prec-sqrt=true;--ftz=false>")

function(kernels_target_setup target)
  target_compile_definitions(${target} PRIVATE "XINFER_TARGET_ARCHS=\"${XINFER_CUDA_ARCH}\"")
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -ffast-math>)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${XINFER_CUTLASS_ROOT}/include
    ${XINFER_CUTLASS_ROOT}/tools/util/include)
  target_link_libraries(${target} PRIVATE tvm_ffi::header CUDA::cudart)
endfunction()

add_library(xinfer_kernels_arch OBJECT ${KERNEL_SM${XINFER_CUDA_ARCH}_SOURCES})
kernels_target_setup(xinfer_kernels_arch)
set_target_properties(xinfer_kernels_arch PROPERTIES CUDA_ARCHITECTURES "${XINFER_CUDA_ARCH}a")

add_library(xinfer_kernels SHARED ${KERNEL_SOURCES} $<TARGET_OBJECTS:xinfer_kernels_arch>)
kernels_target_setup(xinfer_kernels)
if(XINFER_CUDA_ARCH STREQUAL "110")  # the generic rmsnorm calls the SM110 kernel of BF16 rows
  target_compile_definitions(xinfer_kernels PRIVATE XINFER_WITH_SM110=1)
endif()
target_link_libraries(xinfer_kernels PRIVATE tvm_ffi::shared CUDA::cublasLt)
set_target_properties(xinfer_kernels PROPERTIES
  CUDA_ARCHITECTURES "${XINFER_CUDA_ARCH}"
  PREFIX ""
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/..")

# The native kernel library of xinfer's Ascend backend: xinfer_kernels_ascend.so, loaded by ../_native.py.
#
#   pip install -e ".[kernels]"          # installs the TVM-FFI runtime only; pip never compiles
#   cmake -S xinfer/ops/backends/ascend/src -B build/kernels-ascend
#   cmake --build build/kernels-ascend -j1
#
# writes xinfer_kernels_ascend.so next to the Python modules that call it (xinfer/ops/backends/ascend/).
# This runs on the aarch64 build host (openEuler 24.03) with CANN; the full toolchain story, the validated
# version pairing and the environment-order trap are in README.md next to this file.
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
# target). Host glue does not use it; the generator invocation does (TODO R1, below).
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

# TODO (R1): the AOT generation step. The generator pair (tools/generate.py + tools/abi.py, vendored from the
# v22 knife tree per the build-spike contract) turns each family's TileLang modules into .o files (device image
# embedded, loaded by aclrtBinaryLoadFromData) with host launcher symbols renamed. Wire it as one custom
# command per family, then list its objects in the SoC sources below:
#
#   set(XINFER_TILELANG_PYTHON "/root/tl-fork-store/venv-tl-src/bin/python" CACHE FILEPATH
#       "Python of the TileLang fork venv that runs the AOT generator")
#   add_custom_command(OUTPUT ${_objects}
#     COMMAND ${XINFER_TILELANG_PYTHON} tools/generate.py --output ${CMAKE_BINARY_DIR}/aot
#             --module ops.<family>_kernels --kernels <names>
#     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
#     ENVIRONMENT BISHENG_HOME=${XINFER_BISHENG_HOME} ASCEND_HOME_PATH=${XINFER_BISHENG_HOME}
#                 ASCEND_TOOLKIT_HOME=${XINFER_BISHENG_HOME} MAX_JOBS=1
#                 TILELANG_CACHE_DIR=${CMAKE_BINARY_DIR}/tl-cache)
#
# MAX_JOBS=1 is the memory discipline of the shared hosts; TILELANG_CACHE_DIR stays private to this build.

# Operators that run on every SoC: the build information and the family host glue (thin .cc files exporting
# TVM_FFI_DLL_EXPORT_TYPED_FUNC entries that marshal TensorView arguments and launch the AOT call on the
# current NPU stream). TODO (R1): one glue file per family as its kernels move in.
set(KERNEL_SOURCES
  common/build_info.cc
)
# Kernels of one SoC: the AOT objects of that SoC's compile target and its SoC-specific glue, selected by
# variable name exactly as the CUDA side selects KERNEL_SM<arch>_SOURCES - a new SoC needs one new list and
# one entry in XINFER_SUPPORTED_SOCS, nothing else. TODO (R1): the 44 kernels of the porting inventory, e.g.
#   gemm/smallm_r3_950dt.o
#   gemm/qkv_bnsdq_950dt.o
#   finish/glu_split_bf16_c6144_950dt.o ... (from src/<family>/README.md rosters)
string(TOUPPER "${XINFER_ASCEND_SOC}" _SOC_UC)
set(KERNEL_${_SOC_UC}_SOURCES
)
# TODO (R1): per-kernel numeric contracts. Where a kernel's result is specified bit for bit against torch
# (rope tables, the uint8 quantization law, the keep{0,1} multiplies), its DSL source states the arithmetic
# and the generator passes the options that preserve it; the fast-math decisions of the CUDA side
# (--use_fast_math vs --fmad=false) have no bisheng flag equivalents, so the contract lives in the TileLang
# sources and the family glue asserts it, per family, as the kernels move in.

function(kernels_target_setup target)
  target_compile_definitions(${target} PRIVATE "XINFER_TARGET_SOCS=\"${XINFER_ASCEND_SOC}\"")
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(${target} PRIVATE tvm_ffi::header)
endfunction()

add_library(xinfer_kernels_ascend_arch OBJECT ${KERNEL_${_SOC_UC}_SOURCES})
kernels_target_setup(xinfer_kernels_ascend_arch)

add_library(xinfer_kernels_ascend SHARED ${KERNEL_SOURCES} $<TARGET_OBJECTS:xinfer_kernels_ascend_arch>)
kernels_target_setup(xinfer_kernels_ascend)
target_link_libraries(xinfer_kernels_ascend PRIVATE
  tvm_ffi::shared ${XINFER_ascendcl} ${XINFER_opapi} ${XINFER_runtime} ${XINFER_nnopbase})
set_target_properties(xinfer_kernels_ascend PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/..")

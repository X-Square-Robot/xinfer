# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(xinfer_rpu LANGUAGES CXX)

option(XINFER_RPU_HOST_ONLY "Build resolver and metadata without the Launch SDK" ON)
option(XINFER_RPU_BUILD_TESTS "Build board-free native contract tests" ON)
set(RHINO_SDK_ROOT "$ENV{RHINO_SDK_ROOT}" CACHE PATH "External Rhino Launch SDK root")
set(XINFER_RPU_ASSET_OUTPUT "${CMAKE_BINARY_DIR}/assets" CACHE PATH "External compiled asset output")
set(XINFER_RPU_HXCC "$ENV{HXCC}" CACHE FILEPATH "External hxcc compiler")
if(NOT XINFER_RPU_HXCC)
  set(XINFER_RPU_HXCC hxcc)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter)
if(NOT TARGET tvm_ffi::header OR NOT TARGET tvm_ffi::shared)
execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  RESULT_VARIABLE _ffi_result OUTPUT_VARIABLE _ffi_directory OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _ffi_result EQUAL 0)
  message(FATAL_ERROR "Install apache-tvm-ffi in ${Python_EXECUTABLE} before configuring")
endif()
find_package(tvm_ffi CONFIG REQUIRED PATHS "${_ffi_directory}" NO_DEFAULT_PATH)
endif()
find_package(Threads REQUIRED)

set(XINFER_RPU_NATIVE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
get_filename_component(XINFER_RPU_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(XINFER_RPU_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(XINFER_RPU_CATALOG "${CMAKE_CURRENT_SOURCE_DIR}/common/rpu/catalog.json")
add_custom_command(OUTPUT "${XINFER_RPU_GENERATED}/catalog_metadata.hpp"
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_catalog.py"
    --catalog "${XINFER_RPU_CATALOG}" --output "${XINFER_RPU_GENERATED}/catalog_metadata.hpp"
  DEPENDS "${XINFER_RPU_CATALOG}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_catalog.py"
  VERBATIM)
add_custom_target(xinfer_rpu_catalog DEPENDS "${XINFER_RPU_GENERATED}/catalog_metadata.hpp")

set(XINFER_RPU_LOWERING_SOURCES
  common/rpu/tp_dispatch.cc common/rpu/tp_family.cc
  common/rpu/tp_activation.cc common/rpu/tp_attention.cc common/rpu/tp_comm.cc
  common/rpu/tp_conv.cc common/rpu/tp_denoise_output.cc common/rpu/tp_elementwise.cc
  common/rpu/tp_gated_delta_rule.cc common/rpu/tp_gemm.cc common/rpu/tp_layout.cc
  common/rpu/tp_normalization.cc)
add_library(xinfer_rpu_lowering OBJECT ${XINFER_RPU_LOWERING_SOURCES})
target_include_directories(xinfer_rpu_lowering PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(xinfer_rpu_lowering PUBLIC cxx_std_17)
set_target_properties(xinfer_rpu_lowering PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
  target_compile_options(xinfer_rpu_lowering PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
endif()

set(XINFER_RPU_RUNTIME_SOURCES
  common/rpu/library.cc common/rpu/execution.cc common/rpu/execution_ffi.cc
  common/rpu/tp_ffi.cc gemm/gemm_ffi.cc normalization/layernorm_ffi.cc activation/silu_mul_ffi.cc)
set(XINFER_RPU_FFI_SOURCES common/rpu/metadata_ffi.cc common/rpu/tp_family_ffi.cc)
add_library(xinfer_rpu SHARED ${XINFER_RPU_FFI_SOURCES}
  "${XINFER_RPU_GENERATED}/catalog_metadata.hpp" $<TARGET_OBJECTS:xinfer_rpu_lowering>)
target_include_directories(xinfer_rpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${XINFER_RPU_GENERATED}")
target_link_libraries(xinfer_rpu PRIVATE tvm_ffi::header tvm_ffi::shared Threads::Threads)
target_compile_features(xinfer_rpu PRIVATE cxx_std_17)
set_target_properties(xinfer_rpu PROPERTIES PREFIX "" OUTPUT_NAME xinfer_rpu
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
if(NOT MSVC)
  target_compile_options(xinfer_rpu PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
endif()
if(XINFER_RPU_HOST_ONLY)
  target_compile_definitions(xinfer_rpu PRIVATE XINFER_RPU_RUNTIME=0)
else()
  if(NOT UNIX)
    message(FATAL_ERROR "The Rhino Launch runtime requires a Linux host")
  endif()
  find_path(_rhino_include rhino_launch_program.h PATHS "${RHINO_SDK_ROOT}/include" NO_DEFAULT_PATH)
  find_library(_rhino_library NAMES rhino_launch PATHS "${RHINO_SDK_ROOT}/lib" "${RHINO_SDK_ROOT}/lib64"
    NO_DEFAULT_PATH)
  if(NOT _rhino_include OR NOT _rhino_library)
    message(FATAL_ERROR "RHINO_SDK_ROOT must provide include/rhino_launch_program.h and librhino_launch")
  endif()
  target_sources(xinfer_rpu PRIVATE ${XINFER_RPU_RUNTIME_SOURCES})
  target_include_directories(xinfer_rpu PRIVATE "${_rhino_include}")
  target_link_libraries(xinfer_rpu PRIVATE "${_rhino_library}")
  target_compile_definitions(xinfer_rpu PRIVATE XINFER_RPU_RUNTIME=1)
endif()

# Explicit only: configuring or importing the host library never invokes hxcc.
add_custom_target(rpu_assets
  COMMAND "${Python_EXECUTABLE}" "${XINFER_RPU_REPO_ROOT}/tools/build_rpu_assets.py"
    --compiler "${XINFER_RPU_HXCC}" --output "${XINFER_RPU_ASSET_OUTPUT}"
  WORKING_DIRECTORY "${XINFER_RPU_REPO_ROOT}"
  VERBATIM)
install(TARGETS xinfer_rpu LIBRARY DESTINATION xinfer/ops/backends/rpu)

if(XINFER_RPU_BUILD_TESTS)
  enable_testing()
  add_subdirectory("${XINFER_RPU_REPO_ROOT}/tests/rpu" "${CMAKE_CURRENT_BINARY_DIR}/tests")
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  foreach(_variable IN ITEMS XINFER_RPU_RUNTIME_SOURCES XINFER_RPU_FFI_SOURCES XINFER_RPU_GENERATED)
    set(${_variable} "${${_variable}}" PARENT_SCOPE)
  endforeach()
endif()

cmake_minimum_required(VERSION 3.24)
project(xinfer_host LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Optional host helpers. Build against the Python and Torch used by the service.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
  "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)"
  OUTPUT_VARIABLE _torch_root OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
  "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE _torch_abi OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
# These are host-only C++ extensions. Torch's CMake package also enables CUDA
# for CUDA wheels, which would require nvcc merely to compile pin_memory calls.
foreach(name torch torch_cpu c10 torch_python)
  find_library(_${name}_library ${name} REQUIRED HINTS "${_torch_root}/lib" NO_DEFAULT_PATH)
  list(APPEND _torch_libraries "${_${name}_library}")
endforeach()
find_package(Threads REQUIRED)

foreach(name _wirefast _preparefast _pincpp)
  if(name STREQUAL "_pincpp")
    Python_add_library(${name} MODULE WITH_SOABI ${name}.cpp)
    target_compile_features(${name} PRIVATE cxx_std_17)
    target_include_directories(${name} PRIVATE "${_torch_root}/include" "${_torch_root}/include/torch/csrc/api/include")
    target_compile_definitions(${name} PRIVATE TORCH_EXTENSION_NAME=${name} _GLIBCXX_USE_CXX11_ABI=${_torch_abi})
    target_link_libraries(${name} PRIVATE ${_torch_libraries})
  else()
    Python_add_library(${name} MODULE WITH_SOABI ${name}.c)
  endif()
  set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/..")
endforeach()
target_link_libraries(_preparefast PRIVATE Python::NumPy Threads::Threads)
if(UNIX)
  target_link_libraries(_preparefast PRIVATE m)
endif()

# CANN host calls use the POSIX dynamic loader; CUDA installs do not need them.
option(XINFER_HOST_ASCEND "Build the CANN host-call helper" OFF)
if(XINFER_HOST_ASCEND)
  Python_add_library(_aclfast MODULE WITH_SOABI _aclfast.c)
  target_link_libraries(_aclfast PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(_aclfast PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/..")
endif()

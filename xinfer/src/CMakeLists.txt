cmake_minimum_required(VERSION 3.24)
project(xinfer_host LANGUAGES C CXX)

# Optional host helpers. Build against the Python and Torch used by the service.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
  "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE _torch_prefix OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${_torch_prefix}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python REQUIRED HINTS "${TORCH_INSTALL_PREFIX}/lib")
find_package(Threads REQUIRED)

foreach(name _wirefast _preparefast _pincpp)
  if(name STREQUAL "_pincpp")
    Python_add_library(${name} MODULE WITH_SOABI ${name}.cpp)
    target_compile_features(${name} PRIVATE cxx_std_17)
    separate_arguments(_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(${name} PRIVATE ${_torch_flags})
    target_compile_definitions(${name} PRIVATE TORCH_EXTENSION_NAME=${name})
    target_link_libraries(${name} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
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

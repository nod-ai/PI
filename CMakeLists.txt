cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

if(POLICY CMP0135)
  cmake_policy(SET CMP0116 OLD)
endif()

project(PI LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")

set(TORCH_MLIR_INSTALL_DIR
    ""
    CACHE STRING "Path to TorchMLIR install dir")
include(FetchContent)
if(TORCH_MLIR_INSTALL_DIR STREQUAL "")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
            --torch-mlir-url OUTPUT_VARIABLE TORCH_MLIR_URL)
  FetchContent_Populate(
    MLIR
    URL "${TORCH_MLIR_URL}"
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/torch_mlir_install")
  list(APPEND CMAKE_PREFIX_PATH)
  set(TORCH_MLIR_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/torch_mlir_install")
endif()
message(STATUS "TORCH_MLIR_INSTALL_DIR: ${TORCH_MLIR_INSTALL_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_MLIR_INSTALL_DIR}")
find_package(MLIR REQUIRED CONFIG)

add_subdirectory(cpp_ext)

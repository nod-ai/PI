find_package(Git REQUIRED)

# figure out which tar might be downloaded
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(OS ubuntu)
else()
  set(OS macos)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(AMD64|x86_64)")
  set(ARCH X86)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
  set(ARCH AArch64)
endif()

message(STATUS "os: ${OS} arch: ${ARCH}")

set(TORCH_MLIR_COMMIT
    ""
    CACHE STRING "torch-mlir commit to use.")
# Get submodule torch-mlir hash
if(TORCH_MLIR_COMMIT STREQUAL "")
  message(STATUS "Getting torch-mlir commit hash")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} ls-tree HEAD externals/torch-mlir --object-only
            --abbrev=8
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE TORCH_MLIR_COMMIT
    RESULT_VARIABLE TORCH_MLIR_GIT_SUBMOD_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT TORCH_MLIR_GIT_SUBMOD_RESULT EQUAL "0")
    message(
      FATAL_ERROR "git ls-tree HEAD externals/torch-mlir --object-only failed")
  endif()
endif()
message(STATUS "Using torch-mlir commit ${TORCH_MLIR_COMMIT}")

# set(TORCH_MLIR_COMMIT 6f7d9e83)

# Try to download torch-mlir distro
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/torch-mlir-install.tar.xz)
  message(STATUS "Downloading torch-mlir distro.")
  set(TORCH_MLIR_INSTALL_URL
      "https://github.com/nod-ai/PI/releases/download/torch-mlir-${TORCH_MLIR_COMMIT}/torch-mlir-${TORCH_MLIR_COMMIT}-${OS}-latest-${ARCH}.tar.xz"
  )
  file(
    DOWNLOAD ${TORCH_MLIR_INSTALL_URL}
    ${CMAKE_CURRENT_SOURCE_DIR}/torch-mlir-install.tar.xz
    STATUS TORCH_MLIR_DOWNLOAD_STATUS
    SHOW_PROGRESS
    INACTIVITY_TIMEOUT 10)
  list(GET TORCH_MLIR_DOWNLOAD_STATUS 0 TORCH_MLIR_DOWNLOAD_STATUS_CODE)
  list(GET TORCH_MLIR_DOWNLOAD_STATUS 1 TORCH_MLIR_DOWNLOAD_STATUS_STRING)
  if(NOT TORCH_MLIR_DOWNLOAD_STATUS_CODE EQUAL 0)
    file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/torch-mlir-install.tar.xz)
  endif()
else()
  set(TORCH_MLIR_DOWNLOAD_STATUS_CODE 0)
  set(TORCH_MLIR_DOWNLOAD_STATUS_STRING "No error")
endif()

message(
  STATUS "torch-mlir download status: ${TORCH_MLIR_DOWNLOAD_STATUS_STRING}")

# Unpack download (if successfully downloaded distro) or unshallow module.
if(TORCH_MLIR_DOWNLOAD_STATUS_CODE EQUAL 0)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/torch_mlir_install")
    message(STATUS "Downloaded torch-mlir successfully; untarring...")
    file(ARCHIVE_EXTRACT INPUT
         "${CMAKE_CURRENT_SOURCE_DIR}/torch-mlir-install.tar.xz" DESTINATION
         "${CMAKE_CURRENT_SOURCE_DIR}/torch_mlir_install" VERBOSE)
  endif()
  set(TORCH_MLIR_INSTALL_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/torch_mlir_install/torch_mlir_install")
else()
  message(STATUS "Failed to download torch-mlir because ${TORCH_MLIR_DOWNLOAD_STATUS_STRING}.")
  message(STATUS "Will unshallow submodule and build and install.")

  execute_process(
    COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT)
  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(
      FATAL_ERROR
        "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules"
    )
  endif()

  set(TORCH_MLIR_INSTALL_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/torch_mlir_install/torch_mlir_install")
  set(TORCH_MLIR_MAIN_SRC_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/externals/torch-mlir")
  set(TORCH_MLIR_MAIN_BINARY_DIR "${TORCH_MLIR_INSTALL_DIR}/build")
  file(MAKE_DIRECTORY ${TORCH_MLIR_INSTALL_DIR})
  file(MAKE_DIRECTORY ${TORCH_MLIR_MAIN_BINARY_DIR})

  message(STATUS "TORCH_MLIR_INSTALL_DIR ${TORCH_MLIR_INSTALL_DIR}")
  message(STATUS "TORCH_MLIR_MAIN_SRC_DIR ${TORCH_MLIR_MAIN_SRC_DIR}")
  message(STATUS "TORCH_MLIR_MAIN_BINARY_DIR ${TORCH_MLIR_MAIN_BINARY_DIR}")

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pip install -r
            ${TORCH_MLIR_MAIN_SRC_DIR}/build-requirements.txt
    WORKING_DIRECTORY ${TORCH_MLIR_MAIN_BINARY_DIR}
    RESULT_VARIABLE RESVAR
    OUTPUT_VARIABLE LOG1
    ERROR_VARIABLE LOG1 COMMAND_ECHO STDOUT ECHO_OUTPUT_VARIABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT RESVAR EQUAL 0)
    message(
      FATAL_ERROR "Python requirements install failed because ${RESVAR} ${LOG1}"
    )
  endif()
  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX=${TORCH_MLIR_INSTALL_DIR} -DLLVM_CCACHE_BUILD=ON
      -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PROJECTS=mlir
      -DLLVM_ENABLE_ZSTD=OFF
      -DLLVM_EXTERNAL_PROJECTS=torch-mlir\;torch-mlir-dialects
      -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR=${TORCH_MLIR_MAIN_SRC_DIR}/externals/llvm-external-projects/torch-mlir-dialects
      -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=${TORCH_MLIR_MAIN_SRC_DIR}
      -DLLVM_INCLUDE_UTILS=ON -DLLVM_INSTALL_UTILS=ON -DLLVM_USE_HOST_TOOLS=ON
      -DMLIR_BUILD_MLIR_C_DYLIB=1 -DMLIR_ENABLE_BINDINGS_PYTHON=ON
      -DMLIR_ENABLE_EXECUTION_ENGINE=ON
      -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
      -DTORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS=ON
      -DTORCH_MLIR_ENABLE_LTC=OFF -DTORCH_MLIR_ENABLE_STABLEHLO=OFF
      -S${TORCH_MLIR_MAIN_SRC_DIR}/externals/llvm-project/llvm
      -B${TORCH_MLIR_MAIN_BINARY_DIR} --log-level=VERBOSE
    WORKING_DIRECTORY ${TORCH_MLIR_MAIN_BINARY_DIR}
    RESULT_VARIABLE RESVAR
    OUTPUT_VARIABLE LOG1
    ERROR_VARIABLE LOG1 COMMAND_ECHO STDOUT ECHO_OUTPUT_VARIABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT RESVAR EQUAL 0)
    message(FATAL_ERROR "cmake config failed because ${RESVAR} ${LOG1}")
  endif()
  cmake_host_system_information(RESULT N QUERY NUMBER_OF_PHYSICAL_CORES)
  math(EXPR M "${N} / 2" OUTPUT_FORMAT DECIMAL)
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${TORCH_MLIR_MAIN_BINARY_DIR} --target
            install -j${M} -v
    WORKING_DIRECTORY "${TORCH_MLIR_MAIN_SRC_DIR}"
    RESULT_VARIABLE RESVAR
    OUTPUT_VARIABLE LOG1
    ERROR_VARIABLE LOG1 COMMAND_ECHO STDOUT ECHO_OUTPUT_VARIABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT RESVAR EQUAL 0)
    message(FATAL_ERROR "cmake config build because ${RESVAR} ${LOG1}")
  endif()
endif()

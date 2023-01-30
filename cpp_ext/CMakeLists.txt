if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    MESSAGE("Using Clang (no need to check Torch-MLIR ABI)")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    execute_process(
            COMMAND ${Python3_EXECUTABLE} get_torch_mlir_pybind_v_abi.py
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
            ERROR_VARIABLE error
            OUTPUT_VARIABLE ABI_VERSION
    )
    if(NOT error STREQUAL "")
        message(STATUS "Torch-MLIR ABI Version ERROR: ${error}")
    endif()
    message(STATUS "Torch-MLIR ABI Version: ${ABI_VERSION}")
    add_definitions("-fabi-version=${ABI_VERSION}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    #    MESSAGE("Intel")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    #    MESSAGE("MSVC")
endif()

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

##################################### Bindings path hacks

include(MLIRDetectPythonEnv)
include(AddMLIRPython)
mlir_configure_python_dev_packages()
mlir_detect_pybind11_install()

set(PYTHON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}) # --src-root
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(MLIR_INCLUDE_TESTS 0)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=torch_mlir.")
pybind11_add_module(_pi_mlir MainModule.cpp TorchValues.cpp TorchTypes.cpp)
# for llvm::twine and stuff
target_link_libraries(_pi_mlir PRIVATE LLVMSupport MLIRIR)

INSTALL(TARGETS _pi_mlir LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/pi)
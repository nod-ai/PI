cmake_minimum_required(VERSION 3.13.4)

if (POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
    set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif ()

if (POLICY CMP0075)
    cmake_policy(SET CMP0075 NEW)
endif ()

if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

if (POLICY CMP0116)
    cmake_policy(SET CMP0116 NEW)
endif ()

project(mlir-python-bindings LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

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

set(PYTHON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp) # --src-root
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(MLIR_TABLEGEN_EXE "" CACHE STRING "Path to mlir-tablegen")
message(STATUS "MLIR_TABLEGEN_EXE: ${MLIR_TABLEGEN_EXE}")
set(MLIR_INCLUDE_TESTS 0)
include_directories(cpp/include)


##################################### Extra passes/stuff

add_mlir_library(RefBackend
  cpp/RefBackend.cpp

  LINK_COMPONENTS
    Core

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRTransforms
    MLIRMathTransforms
)

##################################### CMake stuff from source

# CMakeLists RESET


################################################################################
# Structural groupings.
################################################################################

declare_mlir_python_sources(SharkMLIRPythonSources)
declare_mlir_python_sources(SharkMLIRPythonSources.Dialects
  ADD_TO_PARENT SharkMLIRPythonSources)

################################################################################
# Pure python sources and generated code
################################################################################

declare_mlir_python_sources(SharkMLIRPythonSources.Core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  ADD_TO_PARENT SharkMLIRPythonSources
  SOURCES
    _mlir_libs/__init__.py
    ir.py
    passmanager.py
    dialects/_ods_common.py

    # The main _mlir module has submodules: include stubs from each.
    _mlir_libs/_mlir/__init__.pyi
    _mlir_libs/_mlir/ir.pyi
    _mlir_libs/_mlir/passmanager.pyi
)


# TODO(max): this can't be here because it'll get erased

declare_mlir_python_sources(SharkMLIRPythonSources.SharkCompiler
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  ADD_TO_PARENT SharkMLIRPythonSources
  SOURCES_GLOB
    compiler/*.py
)

declare_mlir_python_sources(SharkMLIRPythonSources.ExecutionEngine
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  ADD_TO_PARENT SharkMLIRPythonSources
  SOURCES
    execution_engine.py
    _mlir_libs/_mlirExecutionEngine.pyi
  SOURCES_GLOB
    runtime/*.py
)

declare_mlir_python_sources(SharkMLIRPythonCAPI.HeaderSources
  ROOT_DIR "${MLIR_SOURCE_DIR}/include"
  SOURCES_GLOB "shark-c/*.h"
)

################################################################################
# Dialect bindings
################################################################################

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/AsyncOps.td
  SOURCES_GLOB dialects/async_dialect/*.py
  DIALECT_NAME async_dialect)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/BufferizationOps.td
  SOURCES
    dialects/bufferization.py
    dialects/_bufferization_ops_ext.py
  DIALECT_NAME bufferization)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/BuiltinOps.td
  SOURCES
    dialects/builtin.py
    dialects/_builtin_ops_ext.py
  DIALECT_NAME builtin)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/ComplexOps.td
  SOURCES
    dialects/complex.py
  DIALECT_NAME complex)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/ControlFlowOps.td
  SOURCES
    dialects/cf.py
  DIALECT_NAME cf)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/LLVMOps.td
  SOURCES
    dialects/llvm.py
    dialects/_llvm_ops_ext.py
  DIALECT_NAME llvm)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/FuncOps.td
  SOURCES
    dialects/func.py
    dialects/_func_ops_ext.py
  DIALECT_NAME func)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/GPUOps.td
  SOURCES_GLOB dialects/gpu/*.py
  DIALECT_NAME gpu)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/LinalgOps.td
  SOURCES
    dialects/_linalg_ops_ext.py
  SOURCES_GLOB
    dialects/linalg/*.py
  DIALECT_NAME linalg
  #DEPENDS LinalgOdsGen
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/TransformOps.td
  SOURCES
    dialects/_transform_ops_ext.py
    dialects/transform/__init__.py
  DIALECT_NAME transform)

declare_mlir_dialect_extension_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/SCFLoopTransformOps.td
  SOURCES
    dialects/_loop_transform_ops_ext.py
    dialects/transform/loop.py
  DIALECT_NAME transform
  EXTENSION_NAME loop_transform)

declare_mlir_dialect_extension_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/LinalgStructuredTransformOps.td
  SOURCES
    dialects/_structured_transform_ops_ext.py
    dialects/transform/structured.py
  DIALECT_NAME transform
  EXTENSION_NAME structured_transform)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/MathOps.td
  SOURCES dialects/math.py
  DIALECT_NAME math)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/ArithmeticOps.td
  SOURCES
    dialects/arith.py
    dialects/_arith_ops_ext.py
  DIALECT_NAME arith)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/MemRefOps.td
  SOURCES
    dialects/memref.py
    dialects/_memref_ops_ext.py
  DIALECT_NAME memref)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/MLProgramOps.td
  SOURCES
    dialects/ml_program.py
    dialects/_ml_program_ops_ext.py
  DIALECT_NAME ml_program)

declare_mlir_python_sources(
  SharkMLIRPythonSources.Dialects.quant
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  SOURCES
    dialects/quant.py
    _mlir_libs/_mlir/dialects/quant.pyi)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/PDLOps.td
  SOURCES
    dialects/pdl.py
    dialects/_pdl_ops_ext.py
    _mlir_libs/_mlir/dialects/pdl.pyi
  DIALECT_NAME pdl)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/SCFOps.td
  SOURCES
    dialects/scf.py
    dialects/_scf_ops_ext.py
  DIALECT_NAME scf)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/ShapeOps.td
  SOURCES dialects/shape.py
  DIALECT_NAME shape)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/SparseTensorOps.td
  SOURCES dialects/sparse_tensor.py
  DIALECT_NAME sparse_tensor)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/TensorOps.td
  SOURCES dialects/tensor.py
  DIALECT_NAME tensor)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/TosaOps.td
  SOURCES dialects/tosa.py
  DIALECT_NAME tosa)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
  TD_FILE dialects/VectorOps.td
  SOURCES dialects/vector.py
  DIALECT_NAME vector)

################################################################################
# Python extensions.
# The sources for these are all in lib/Bindings/Python, but since they have to
# be rebuilt for each package and integrate with the source setup here, we
# just reference them here instead of having ordered, cross package target
# dependencies.
################################################################################

#set(PYTHON_SOURCE_DIR "${MLIR_SOURCE_DIR}/lib/Bindings/Python")
declare_mlir_python_extension(SharkMLIRPythonExtension.Core
  MODULE_NAME _mlir
  ADD_TO_PARENT SharkMLIRPythonSources.Core
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    MainModule.cpp
    IRAffine.cpp
    IRAttributes.cpp
    IRCore.cpp
    IRInterfaces.cpp
    IRModule.cpp
    IRTypes.cpp
    PybindUtils.cpp
    Pass.cpp

    # Headers must be included explicitly so they are installed.
    Globals.h
    IRModule.h
    Pass.h
    PybindUtils.h
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIDebug
    MLIRCAPIIR
    MLIRCAPIInterfaces

    # Dialects
    MLIRCAPIFunc
)

# This extension exposes an API to register all dialects, extensions, and passes
# packaged in upstream MLIR and it is used for the upstream "shark" Python
# package. Downstreams will likely want to provide their own and not depend
# on this one, since it links in the world.
# Note that this is not added to any top-level source target for transitive
# inclusion: It must be included explicitly by downstreams if desired. Note that
# this has a very large impact on what gets built/packaged.
declare_mlir_python_extension(SharkMLIRPythonExtension.RegisterEverything
  MODULE_NAME _mlirRegisterEverything
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    RegisterEverything.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
    RefBackend
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIConversion
    MLIRCAPITransforms
    MLIRCAPIRegisterEverything
)

declare_mlir_python_extension(SharkMLIRPythonExtension.Dialects.Linalg.Pybind
  MODULE_NAME _mlirDialectsLinalg
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects.linalg
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectLinalg.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPILinalg
)

declare_mlir_python_extension(SharkMLIRPythonExtension.Dialects.Quant.Pybind
  MODULE_NAME _mlirDialectsQuant
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects.quant
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectQuant.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIQuant
)

declare_mlir_python_extension(SharkMLIRPythonExtension.Dialects.PDL.Pybind
  MODULE_NAME _mlirDialectsPDL
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects.pdl
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectPDL.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIPDL
)

declare_mlir_python_extension(SharkMLIRPythonExtension.Dialects.SparseTensor.Pybind
  MODULE_NAME _mlirDialectsSparseTensor
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects.sparse_tensor
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectSparseTensor.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPISparseTensor
)

declare_mlir_python_extension(SharkMLIRPythonExtension.AsyncDialectPasses
  MODULE_NAME _mlirAsyncPasses
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects.async_dialect
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    AsyncPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIAsync
)

if(MLIR_ENABLE_EXECUTION_ENGINE)
  declare_mlir_python_extension(SharkMLIRPythonExtension.ExecutionEngine
    MODULE_NAME _mlirExecutionEngine
    ADD_TO_PARENT SharkMLIRPythonSources.ExecutionEngine
    ROOT_DIR "${PYTHON_SOURCE_DIR}"
    SOURCES
      ExecutionEngineModule.cpp
    PRIVATE_LINK_LIBS
      LLVMSupport
    EMBED_CAPI_LINK_LIBS
      MLIRCAPIExecutionEngine
  )
endif()

declare_mlir_python_extension(SharkMLIRPythonExtension.GPUDialectPasses
  MODULE_NAME _mlirGPUPasses
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects.gpu
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    GPUPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIGPU
)

declare_mlir_python_extension(SharkMLIRPythonExtension.LinalgPasses
  MODULE_NAME _mlirLinalgPasses
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects.linalg
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    LinalgPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPILinalg
)

declare_mlir_python_extension(SharkMLIRPythonExtension.SparseTensorDialectPasses
  MODULE_NAME _mlirSparseTensorPasses
  ADD_TO_PARENT SharkMLIRPythonSources.Dialects.sparse_tensor
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    SparseTensorPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPISparseTensor
)

# TODO: Figure out how to put this in the test tree.
# This should not be included in the main Python extension. However,
# putting it into SharkMLIRPythonTestSources along with the dialect declaration
# above confuses Python module loader when running under lit.
set(_ADDL_TEST_SOURCES)
if(MLIR_INCLUDE_TESTS)
  set(_ADDL_TEST_SOURCES SharkMLIRPythonTestSources)
  declare_mlir_python_sources(SharkMLIRPythonTestSources)
  declare_mlir_python_sources(SharkMLIRPythonTestSources.Dialects
    ADD_TO_PARENT SharkMLIRPythonTestSources)

  # TODO: this uses a tablegen file from the test directory and should be
  # decoupled from here.
  declare_mlir_python_sources(
    SharkMLIRPythonTestSources.Dialects.PythonTest
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shark"
    ADD_TO_PARENT SharkMLIRPythonTestSources.Dialects
    SOURCES dialects/python_test.py)
  set(LLVM_TARGET_DEFINITIONS
    "${MLIR_MAIN_SRC_DIR}/test/python/python_test_ops.td")
  mlir_tablegen(
    "dialects/_python_test_ops_gen.py"
    -gen-python-op-bindings
    -bind-dialect=python_test)
  add_public_tablegen_target(PythonTestDialectPyIncGen)
  declare_mlir_python_sources(
    SharkMLIRPythonTestSources.Dialects.PythonTest.ops_gen
    ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
    ADD_TO_PARENT SharkMLIRPythonTestSources.Dialects.PythonTest
    SOURCES "dialects/_python_test_ops_gen.py")

  declare_mlir_python_extension(SharkMLIRPythonTestSources.PythonTestExtension
    MODULE_NAME _mlirPythonTest
    ADD_TO_PARENT SharkMLIRPythonTestSources.Dialects
    ROOT_DIR "${MLIR_SOURCE_DIR}/test/python/lib"
    SOURCES
      PythonTestModule.cpp
    PRIVATE_LINK_LIBS
      LLVMSupport
    EMBED_CAPI_LINK_LIBS
      MLIRCAPIPythonTestDialect
  )
endif()

################################################################################
# Common CAPI dependency DSO.
# All python extensions must link through one DSO which exports the CAPI, and
# this must have a globally unique name amongst all embeddors of the python
# library since it will effectively have global scope.
#
# The presence of this aggregate library is part of the long term plan, but its
# use needs to be made more flexible.
#
# TODO: Upgrade to the aggregate utility in https://reviews.llvm.org/D106419
# once ready.
################################################################################

add_mlir_python_common_capi_library(SharkMLIRPythonCAPI
  INSTALL_COMPONENT SharkMLIRPythonModules
  INSTALL_DESTINATION python_packages/mlir_core/shark/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/mlir_core/shark/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_HEADERS
    SharkMLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    SharkMLIRPythonSources
    SharkMLIRPythonExtension.RegisterEverything
    ${_ADDL_TEST_SOURCES}
)

################################################################################
# The fully assembled package of modules.
# This must come last.
################################################################################

add_mlir_python_modules(SharkMLIRPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/mlir_core/shark"
  INSTALL_PREFIX "python_packages/mlir_core/shark"
  DECLARED_SOURCES
    SharkMLIRPythonSources
    SharkMLIRPythonExtension.RegisterEverything
    ${_ADDL_TEST_SOURCES}
  COMMON_CAPI_LINK_LIBS
    SharkMLIRPythonCAPI
)

# CMakeLists END


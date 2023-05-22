include(MLIRDetectPythonEnv)
include(AddMLIRPython)
mlir_configure_python_dev_packages()
mlir_detect_pybind11_install()

set(PI_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")

# THIS NAME IS LOADING BEARING and should be the namespace path to the mlir
# module PLUS THE DOT
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=pi.mlir.")
declare_mlir_python_sources(PiPythonSources)

declare_mlir_python_extension(
  PiPythonSources.Extension
  MODULE_NAME
  _pi_mlir
  ADD_TO_PARENT
  PiPythonSources
  SOURCES
  TorchTypes.cpp
  TorchValues.cpp
  TorchOps.cpp
  TorchTensor.cpp
  PiExtension.cpp)

set(PYBINDINGS_SRC "${MLIR_INSTALL_PREFIX}/src/python/MLIRPythonExtension.Core")
target_include_directories(PiPythonSources.Extension
                           INTERFACE "${PYBINDINGS_SRC}")
# for SetPyError which somehow magically get's linked somewhere...
target_sources(PiPythonSources.Extension
               INTERFACE "${PYBINDINGS_SRC}/PybindUtils.cpp")

set(FILES_TO_PATCH IRModule.h;Globals.h;IRCore.cpp;IRAffine.cpp)

foreach(PATCH ${FILES_TO_PATCH})
  message(STATUS "Patching ${PATCH}")
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/regex_mlir_bindings.py
      ${PYBINDINGS_SRC}/${PATCH}
    RESULT_VARIABLE RESVAR
    OUTPUT_VARIABLE LOG1
    ERROR_VARIABLE LOG1 COMMAND_ECHO STDOUT ECHO_OUTPUT_VARIABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT RESVAR EQUAL 0)
    message(FATAL_ERROR "Failed to patch ${PATCH} because ${RESVAR} ${LOG1}")
  endif()
endforeach()


declare_mlir_python_extension(
  PiPythonSources.SiteInitialize
  MODULE_NAME
  _site_initialize_0
  ADD_TO_PARENT
  PiPythonSources
  SOURCES
  SiteInitialize.cpp)

set(_source_components
    MLIRPythonSources MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything TorchMLIRPythonSources
    TorchMLIRPythonExtensions PiPythonSources)

add_mlir_python_common_capi_library(
  PiAggregateCAPI
  INSTALL_COMPONENT
  PiPythonModules
  # this is relative to CMAKE_INSTALL_PREFIX
  INSTALL_DESTINATION
  mlir/_mlir_libs
  # this is in the build dir
  OUTPUT_DIRECTORY
  "${PI_MLIR_PYTHON_PACKAGES_DIR}/pi/mlir/_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT
  "../../../../.."
  DECLARED_SOURCES
  ${_source_components})

add_mlir_python_modules(
  PiPythonModules
  # this is in the build dir
  ROOT_PREFIX
  "${PI_MLIR_PYTHON_PACKAGES_DIR}/pi/mlir/_mlir"
  INSTALL_PREFIX
  # this is relative to CMAKE_INSTALL_PREFIX
  mlir
  DECLARED_SOURCES
  ${_source_components}
  COMMON_CAPI_LINK_LIBS
  PiAggregateCAPI)

add_llvm_install_targets(install-pi DEPENDS PiPythonModules COMPONENT
                         PiPythonModules)

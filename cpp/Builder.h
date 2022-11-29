//
// Created by mlevental on 11/28/22.
//

#ifndef MLIR_PYTHON_BINDINGS_CPP_BUILDER_H
#define MLIR_PYTHON_BINDINGS_CPP_BUILDER_H

#include <mlir-c/IR.h>
#include <pybind11/pybind11.h>

namespace mlir {
namespace python {

void populateSharkPy(pybind11::module &m);

}
}


#endif//MLIR_PYTHON_BINDINGS_CPP_BUILDER_H

import inspect
from pathlib import Path

import libcst as cst
from libcst import (
    MetadataWrapper,
)
from libcst.metadata import (
    FullRepoManager,
)

from shark.compiler.builders.module import CompilerVisitor
from shark.compiler.providers.type import (
    MyTypeInferenceProvider,
    MLIRTypeProvider,
)
from shark.ir import (
    Context,
    Module,
    Location,
)


def mlir_compile(module, globals=None, use_affine_fors=False):
    file_path = inspect.getfile(module)
    manager = FullRepoManager(
        str(Path(file_path).parent.resolve()), {file_path}, {MyTypeInferenceProvider}
    )
    if globals is None:
        globals = module.__dict__

    with open(file_path) as f:
        file_source = f.read()
    module_ast = cst.parse_module(file_source)

    mlir_context = Context()
    mlir_location_unknown = Location.unknown(context=mlir_context)
    mlir_module = Module.create(loc=mlir_location_unknown)

    wrapper = MetadataWrapper(
        module_ast,
        cache={
            MyTypeInferenceProvider: manager.get_cache_for_path(file_path),
            MLIRTypeProvider: mlir_context,
        },
    )
    visitor = CompilerVisitor(
        mlir_context,
        mlir_module,
        mlir_location_unknown,
        use_affine_fors=use_affine_fors,
        py_global_defs=globals,
    )
    wrapper.visit(visitor)
    return mlir_module

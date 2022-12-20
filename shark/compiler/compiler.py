import inspect
from pathlib import Path

import libcst as cst
from libcst import (
    MetadataWrapper,
)
from libcst.metadata import (
    FullRepoManager,
)

from shark import ir

from shark.compiler.builders.module import CompilerVisitor
from shark.compiler.byte_code_interpreter.execfile import (
    run_python_file,
)
from shark.compiler.providers.type import (
    MyTypeInferenceProvider,
    MLIRTypeProvider,
)
from shark.compiler.pytype_vm.bytecode_compiler import ByteCodeCompiler
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


def callback(
    event,
    opoffset,
    byte_name,
    byte_code,
    line_number,
    int_arg,
    event_arg,
    vm,
):
    print(f"{event=}", f"{byte_name=}")


# TODO(max): this should not be exposed and basically hidden behind "abstract" compile
def mlir_bytecode_pytype_compile(fn):
    fp = inspect.getfile(fn)
    b = ByteCodeCompiler()
    with open(fp) as f:
        b.vm.run_program(f.read(), "", maximum_depth=10)

    print(b.mlir_module)
    # actual = [(op.name, op.line, symbol) for op, symbol, _ in b.ctx.vm.opcode_traces]
    # pprint(actual)


def mlir_bytecode_xpython_compile(fn):
    top_mlir_context = ir.Context()
    mlir_location = ir.Location.unknown(context=top_mlir_context)
    with top_mlir_context, mlir_location:
        mlir_module = ir.Module.create(loc=mlir_location)

    mlir_module = run_python_file(
        inspect.getfile(fn),
        [],
        top_mlir_context,
        mlir_location,
        mlir_module,
    )
    return mlir_module

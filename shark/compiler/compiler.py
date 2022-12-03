from __future__ import annotations

import inspect
import sys

import libcst as cst
from libcst import (
    MetadataWrapper,
)
from libcst.metadata import (
    FullRepoManager,
)

from compiler_visitor import (
    CompilerVisitor,
    ScopeTransformer,
    LiveInLiveOutVisitor,
    LiveInLiveOutProvider,
)
from shark.ir import (
    Context,
    Module,
    Location,
)
from type_visitor import (
    MyTypeInferenceProvider,
    MLIRTypeProvider,
)


def livein_liveout(fn_ast):
    fn_ast = MetadataWrapper(fn_ast).visit(ScopeTransformer())
    livein_liveout_visitor = LiveInLiveOutVisitor()
    MetadataWrapper(fn_ast).visit(livein_liveout_visitor)
    return livein_liveout_visitor.live_ins, livein_liveout_visitor.live_outs


def mlir_compile(fn, globals=None):
    file_path = "test_kernel.py"
    manager = FullRepoManager(".", {file_path}, {MyTypeInferenceProvider})
    if globals is None:
        globals = sys.modules[fn.__module__].__dict__

    fn_source = inspect.getsource(fn)
    fn_ast = cst.parse_module(fn_source)
    live_ins, live_outs = livein_liveout(fn_ast.deep_clone())

    mlir_context = Context()
    mlir_location_unknown = Location.unknown(context=mlir_context)
    mlir_module = Module.create(loc=mlir_location_unknown)

    wrapper = MetadataWrapper(
        fn_ast,
        cache={
            MyTypeInferenceProvider: manager.get_cache_for_path(file_path),
            LiveInLiveOutProvider: (live_ins, live_outs),
            MLIRTypeProvider: mlir_context,
        },
    )
    visitor = CompilerVisitor(
        mlir_context,
        mlir_module,
        mlir_location_unknown,
        py_global_defs=globals,
    )
    wrapper.visit(visitor)
    return mlir_module

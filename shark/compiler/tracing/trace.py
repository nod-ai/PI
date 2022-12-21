import sys
from collections import namedtuple

import ast
import ctypes
import inspect
import os
import pyccolo as pyc
import traceback
import types
from contextlib import contextmanager
from pathlib import Path
from pyccolo import TraceEvent, AstRewriter
from pyccolo.emit_event import _TRACER_STACK
from pytype.pyc.opcodes import dis
from runpy import run_module
from typing import Optional, Union, Tuple, List

from shark import ir
from shark.dialects import func as func_dialect, scf, affine_, arith
from shark.dialects import value_
from shark.dialects._ods_common import get_op_result_or_value
from shark.ir import Type as MLIRType, IntegerType, F64Type


# from uncompyle6 import main


def print_tb():
    for line in traceback.format_stack():
        print(line.strip())


MLIRStackFrame = namedtuple(
    "MLIRStackFrame",
    [
        "block",
        "block_args",
        "context",
        "insertion_point",
        "location",
        "scope_name",
    ],
)


def dis_code(code):
    return dis(
        data=code.co_code,
        python_version=(3, 10),
        co_varnames=code.co_varnames,
        co_names=code.co_names,
        co_consts=code.co_consts,
        co_cellvars=code.co_cellvars,
        co_freevars=code.co_freevars,
        co_lnotab=code.co_lnotab,
        co_firstlineno=code.co_firstlineno,
    )


def infer_mlir_type(py_val) -> MLIRType:
    if isinstance(py_val, int):
        # return IntegerType.get_signed(64)
        return IntegerType.get_signless(64)
    elif isinstance(py_val, float):
        return F64Type.get()
    else:
        raise Exception(f"unsupported val type {type(py_val)} {py_val}")


def make_constant(
    py_cst: Union[int, float, bool],
    index_type: bool = False,
):
    if index_type:
        constant = arith.ConstantOp.create_index(py_cst)
    else:
        constant = arith.ConstantOp(infer_mlir_type(py_cst), py_cst)

    return constant


class MLIRRewriter(AstRewriter):
    def __init__(self, tracers, **kwargs):
        self.mlir_tracer = next(t for t in tracers if isinstance(t, MLIRTracer))
        super(MLIRRewriter, self).__init__(tracers, **kwargs)

    def visit(self, node: ast.AST):
        mod_node = super(MLIRRewriter, self).visit(node)
        for node in ast.walk(mod_node):
            if isinstance(node, ast.For):
                if not isinstance(node.body[-1], ast.Break):
                    break_ = ast.Break(lineno=1, col_offset=1)
                    node.body.append(break_)
        return mod_node


def update_frame_locals(frame, updates: dict):
    # TODO(max): jot this down somewhere
    frame.f_locals.update(updates)
    # https://stackoverflow.com/a/46434344
    # passing c_int(0) will only update
    # passing c_int(1) will remove and update variables as well
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))


class MLIRTracer(pyc.BaseTracer):
    def __init__(
        self,
        mlir_context: ir.Context,
        mlir_location: ir.Location,
        mlir_module: ir.Module,
    ):
        super().__init__()
        self.mlir_context = mlir_context
        self.mlir_location = mlir_location
        self.mlir_module = mlir_module
        self.local_defs = {}

        f64_type = ir.F64Type.get(context=self.mlir_context)
        self.type_mapping = {
            "float": f64_type,
            "float_memref": ir.MemRefType.get((-1,), f64_type, loc=self.mlir_location),
            "int": ir.IntegerType.get_signed(64, context=self.mlir_context),
            "uint": ir.IntegerType.get_unsigned(64, context=self.mlir_context),
        }
        self.pyfunc_to_mlir_func_op = {}

        self.scf_fors = False
        self._mlir_block_stack: List[MLIRStackFrame] = []
        self.enter_mlir_block_scope(
            self.mlir_module.body,
            mlir_context=self.mlir_context or self.mlir_context,
            scope_name="module",
        )

    def enter_mlir_block_scope(
        self,
        block: ir.Block,
        *,
        block_args: Tuple[str] = None,
        mlir_context: Optional[ir.Context] = None,
        insertion_point: Optional[ir.InsertionPoint] = None,
        mlir_location: Optional[ir.Location] = None,
        # for debugging
        scope_name: Optional[str] = None,
    ):
        if mlir_context is None:
            # one past the last op but still inside the block
            mlir_context = ir.Context()
        if insertion_point is None:
            # one past the last op but still inside the block
            insertion_point = ir.InsertionPoint(block)
        if mlir_location is None:
            mlir_location = self.mlir_location
        if block_args is None:
            block_args = ()
        if scope_name is None:
            scope_name = "UNKNOWN_SCOPE"

        mlir_context.__enter__()
        insertion_point.__enter__()
        mlir_location.__enter__()
        self._mlir_block_stack.append(
            MLIRStackFrame(
                block,
                block_args,
                mlir_context,
                insertion_point,
                mlir_location,
                scope_name,
            )
        )
        return block, mlir_location

    def exit_mlir_block_scope(self, scope_name=None):
        if scope_name is None:
            scope_name = "UNKNOWN_SCOPE"
        mlir_stack_frame = self._mlir_block_stack.pop()
        assert (
            mlir_stack_frame.scope_name == scope_name
        ), f"enter and exit in two different scopes {mlir_stack_frame.scope_name} {scope_name}"
        mlir_stack_frame.location.__exit__(None, None, None)
        mlir_stack_frame.insertion_point.__exit__(None, None, None)
        mlir_stack_frame.context.__exit__(None, None, None)
        return mlir_stack_frame.block

    @contextmanager
    def mlir_block_scope(
        self,
        block: ir.Block,
        *,
        block_args: Tuple[str] = None,
        insertion_point: Optional[ir.InsertionPoint] = None,
        mlir_location: Optional[ir.Location] = None,
        scope_name: Optional[str] = None,
    ):
        if scope_name is None:
            scope_name = "UNKNOWN_SCOPE"
        yield self.enter_mlir_block_scope(
            block,
            block_args=block_args,
            insertion_point=insertion_point,
            mlir_location=mlir_location,
            scope_name=scope_name,
        )
        self.exit_mlir_block_scope(scope_name=scope_name)

    def mlir_peek_block_scope(self):
        assert len(self._mlir_block_stack), "no block scope yet"
        return self._mlir_block_stack[-1]

    def get_or_make_mlir_constant(
        self,
        py_cst: Union[int, float, bool],
        name: Optional[str] = None,
        index_type: bool = False,
    ):
        assert isinstance(py_cst, (int, float, bool))
        if name is None:
            name = str(py_cst)
        if (name, index_type) not in self.local_defs:
            self.local_defs[name, index_type] = make_constant(py_cst, index_type)
        return self.local_defs[name, index_type]

    def should_propagate_handler_exception(
        self, evt: TraceEvent, exc: Exception
    ) -> bool:
        return True

    def should_instrument_file(self, filename: str) -> bool:
        return True

    # handlers

    @pyc.before_function_body
    def handle_before_function_body(
        self,
        old_ret,
        node,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **_,
    ):
        # func = types.FunctionType(
        #     frame.f_code,
        #     frame.f_globals,
        #     frame.f_code.co_name,
        #     (), # defaults
        #     closure = tuple(types.CellType(None) for _ in range(len(frame.f_code.co_freevars)))
        # )
        full_arg_spec = inspect.getfullargspec(
            frame.f_back.f_locals[frame.f_code.co_name]
        )
        arg_types = tuple(
            self.type_mapping[ast.unparse(a.annotation)] for a in node.args.args
        )
        func_op = func_dialect.FuncOp(
            name=frame.f_code.co_name,
            type=(
                arg_types,
                (),
            ),
            visibility="private",
            loc=self.mlir_location,
        )
        self.pyfunc_to_mlir_func_op[frame.f_code.co_name] = func_op
        func_op_entry_block = func_op.add_entry_block()

        update_frame_locals(
            frame, {full_arg_spec.args[i]: a for i, a in enumerate(func_op.arguments)}
        )

        self.enter_mlir_block_scope(
            func_op_entry_block, scope_name=frame.f_code.co_name
        )

    @pyc.after_return
    def handle_after_return(
        self,
        old_ret,
        node_id_or_node,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **_,
    ):
        assert frame.f_code.co_name in self.pyfunc_to_mlir_func_op

        func_op = self.pyfunc_to_mlir_func_op[frame.f_code.co_name]
        mlir_return_val = get_op_result_or_value(old_ret)
        func_type = func_op.type
        canonical_func_type = ir.FunctionType.get(
            inputs=func_type.inputs, results=[mlir_return_val.type]
        )
        func_op.attributes["function_type"] = ir.TypeAttr.get(canonical_func_type)
        func_dialect.ReturnOp((mlir_return_val,))
        self.exit_mlir_block_scope(scope_name=frame.f_code.co_name)

        return old_ret

    @pyc.exit_module
    def handle_exit_module(
        self,
        old_ret,
        node_id_or_node,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **kwargs,
    ):
        self.exit_mlir_block_scope(scope_name="module")

    ast_rewriter_cls = MLIRRewriter

    def make_ast_rewriter(self, **kwargs) -> AstRewriter:
        return self.ast_rewriter_cls(_TRACER_STACK, **kwargs)

    @pyc.before_for_loop_body(reentrant=False)
    def handle_before_for_loop_body(
        self,
        old_ret,
        node: ast.For,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **kwargs,
    ):
        range_iter = eval(ast.unparse(node.iter), frame.f_globals, frame.f_locals)
        if not isinstance(range_iter, range):
            raise RuntimeError("Only `range` iterator currently supported")
        for arg in [range_iter.start, range_iter.stop, range_iter.step]:
            assert isinstance(arg, int), f"symbolic range not supported yet {arg}"

        for arg in [range_iter.start, range_iter.stop, range_iter.step]:
            assert isinstance(arg, int), f"symbolic range not supported yet {arg}"

        start, stop, step = range_iter.start, range_iter.stop, range_iter.step
        if self.scf_fors:
            start, stop, step = [
                self.get_or_make_mlir_constant(c, index_type=self.scf_fors)
                for c in [start, stop, step]
            ]
            loop = scf.ForOp(start, stop, step, [], loc=self.mlir_location)
        else:
            loop = affine_.AffineForOp(start, stop, step, loc=self.mlir_location)

        assert isinstance(
            node.target, ast.Name
        ), f"structured for induction vars not supported {node.target}"
        induction_var_name = node.target.id
        update_frame_locals(frame, {induction_var_name: loop.induction_variable})
        self.enter_mlir_block_scope(loop.body)

    #
    @pyc.after_for_loop_iter
    def handle_after_for_loop_iter(
        self,
        old_ret,
        node_id_or_node,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **kwargs,
    ):
        # TODO(max): handle yielded vals
        yielded_vals = []
        if self.scf_fors:
            scf.YieldOp(yielded_vals, loc=self.mlir_location)
        else:
            affine_.AffineYieldOp(yielded_vals, loc=self.mlir_location)

        self.exit_mlir_block_scope()

    @pyc.before_compare
    @pyc.before_binop
    def handle_before_binop(
        self,
        old_ret,
        node,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **kwargs,
    ):
        def eval_op(x, y):
            x, y = map(
                lambda v: self.get_or_make_mlir_constant(v)
                if isinstance(v, (float, int, bool))
                else v,
                (x, y),
            )
            if isinstance(node, ast.Compare):
                op = node.ops[0].__class__.__name__.lower()
            else:
                # ...god damn it
                op = node.op.__class__.__name__.lower().replace("mult", "mul")
            return getattr(value_, op)(x, y)

        return eval_op

    # @pyc.before_subscript_slice
    # @pyc.before_subscript_del
    @pyc.before_subscript_load
    @pyc.before_subscript_store
    def handle_subscr(
        self,
        old_ret,
        node,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **kwargs,
    ):
        this = self

        class _dummy:
            def __getitem__(self, indices):
                target = frame.f_locals[node.value.id]
                indices = tuple(
                    map(
                        lambda v: this.get_or_make_mlir_constant(v, index_type=True)
                        if isinstance(v, (float, int, bool))
                        else v,
                        indices,
                    )
                )
                return target[indices]

            def __setitem__(self, indices, value):
                target = frame.f_locals[node.value.id]
                # TODO(max): handle value
                indices = tuple(
                    map(
                        lambda v: this.get_or_make_mlir_constant(v, index_type=True)
                        if isinstance(v, (float, int, bool))
                        else v,
                        indices,
                    )
                )
                target[indices] = value

        return _dummy()


def get_script_as_module(script: str) -> str:
    # ref: https://nvbn.github.io/2016/08/17/ast-import/
    script_path = Path(script)
    script_dir = script_path.parent.as_posix()
    module_name = os.path.splitext(script_path.name)[0]
    sys.path.insert(0, script_dir)
    return module_name


def trace(script_path, mlir_context, mlir_location, mlir_module) -> ir.Module:
    module_to_run = get_script_as_module(script_path)
    with MLIRTracer(mlir_context, mlir_location, mlir_module):
        run_module(module_to_run)

    return mlir_module


# TODO(max): jot this down somewhere

# @pyc.opcode
# def handle_opcode(
#     self,
#     old_ret,
#     _node_id_or_node,
#     frame: types.FrameType,
#     event,
#     _guard_for_spec,
#     **kwargs,
# ):
#     # print_tb()
#     discodes = dis_code(frame.f_code)
#     byte_index = frame.f_lasti
#     instr = frame.f_code.co_code[byte_index]
#     dis_inst = discodes[byte_index]
#     arg = frame.f_code.co_code[byte_index + 1]
#     print(f"{instr=} {dis_inst} {byte_index=} {arg=}")
#     # Bytecode.from_code(frame.f_code)
#     # print(f"{old_ret=} {event.to_ast()=} {kwargs=}")
#     exit(0)

# @pyc.before_function_body
# def handle_before_function_body(self, old_ret, node_id_or_node, frame: types.FrameType, event, guard_for_spec, **kwargs):
#     # deparsed = main.decompile(None, frame.f_code, out=sys.stdout)
#     # val = frame_stack_read(frame, -1)
#     # cyber_frame = Frame(frame)
#     # bcodes = Bytecode.from_code(frame.f_code)
#     # discodes = dis_code(frame.f_code)
#     # byte_index = frame.f_lasti
#     # instr = frame.f_code.co_code[byte_index]
#     # dis_inst = discodes[byte_index]
#     # arg = frame.f_code.co_code[byte_index + 1]
#     # print(f"{instr=} {dis_inst} {byte_index=} {arg=}")
#
#
#     print(kwargs)
#     print(f"handle_before_function_body")

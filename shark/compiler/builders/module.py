from __future__ import annotations

from ast import literal_eval
from contextlib import contextmanager
from functools import lru_cache

from typing import Optional, Union

from libcst import (
    CSTVisitor,
    Name,
    CSTNode,
    matchers,
    BaseNumber,
    BinaryOperation,
    Assign,
    Expr,
    FunctionDef,
    For,
    Integer,
    Float,
    Module,
)
from libcst.metadata import (
    ParentNodeProvider,
    PositionProvider,
    ExpressionContextProvider,
)

from shark._mlir_libs._mlir.ir import (
    Context,
    Module as MLIRModule,
    Location,
    Block,
    InsertionPoint,
    Value as MLIRValue,
    OpView,
    Operation,
)
from shark.compiler.builders.body import BodyBuilder
from shark.compiler.providers import ReturnFinder
from shark.compiler.providers.scope import MyScopeProvider, LiveInLiveOutProvider
from shark.compiler.providers.type import MLIRTypeProvider, infer_mlir_type
from shark.dialects import arith, func, llvm, scf


class FindName(CSTVisitor):
    _name: Optional[str] = None

    def leave_Name(self, node: Name) -> bool:
        if self._name is not None:
            raise Exception(f"multiple names {node}")
        else:
            self._name = node.value

    def __call__(self, node: CSTNode):
        if isinstance(node, Name):
            return node.value

        node.visit(self)
        assert self._name and isinstance(self._name, str)
        return self._name


class CompilerVisitor(matchers.MatcherDecoratableVisitor):
    METADATA_DEPENDENCIES = (
        MyScopeProvider,
        ParentNodeProvider,
        MLIRTypeProvider,
        PositionProvider,
        ExpressionContextProvider,
        LiveInLiveOutProvider,
    )

    def __init__(
        self,
        mlir_context: Context,
        module: MLIRModule,
        location: Location,
        py_constants=None,
        py_global_defs=None,
        global_scope=None,
    ):
        super().__init__()
        if global_scope is None:
            global_scope = {}
        if py_global_defs is None:
            py_global_defs = {}
        self.mlir_context = mlir_context
        self.module = module
        self._mlir_block_stack = []
        self.location = location
        self.body_builder = BodyBuilder({}, {}, {}, self.location)
        self.py_global_defs = py_global_defs
        self.local_defs = {}
        self.global_uses = {}
        self._live_ranges = None
        self._live_ranges_tree = None

        # TODO(max): this hacky af
        self.current_block = None
        self.current_op = None

    def enter_mlir_block_scope(
        self,
        block: Block,
        *,
        block_args: tuple[str] = None,
        insertion_point: Optional[InsertionPoint] = None,
        location: Optional[Location] = None,
        # for debugging
        scope_name: Optional[str] = None,
    ):
        if insertion_point is None:
            # one past the last op but still inside the block
            insertion_point = InsertionPoint(block)
        if location is None:
            location = self.location
        if block_args is None:
            block_args = ()
        if scope_name is None:
            scope_name = "UNNAMED_SCOPE"

        self.mlir_context.__enter__()
        insertion_point.__enter__()
        location.__enter__()
        self._mlir_block_stack.append(
            (block, block_args, insertion_point, location, scope_name)
        )
        return block, location

    def exit_mlir_block_scope(self, exit_scope_name=None):
        if exit_scope_name is None:
            exit_scope_name = "UNNAMED_SCOPE"
        (
            block,
            block_args,
            insertion_point,
            location,
            enter_scope_name,
        ) = self._mlir_block_stack.pop()
        assert (
            enter_scope_name == exit_scope_name
        ), f"enter and exit in two different scopes {enter_scope_name} {exit_scope_name}"
        location.__exit__(None, None, None)
        insertion_point.__exit__(None, None, None)
        self.mlir_context.__exit__(None, None, None)
        return block

    @contextmanager
    def mlir_block_scope(
        self,
        block: Block,
        *,
        block_args: tuple[str] = None,
        insertion_point: Optional[InsertionPoint] = None,
        location: Optional[Location] = None,
        scope_name: Optional[str] = None,
    ):
        if scope_name is None:
            scope_name = "UNNAMED_SCOPE"
        yield self.enter_mlir_block_scope(
            block,
            block_args=block_args,
            insertion_point=insertion_point,
            location=location,
            scope_name=scope_name,
        )
        self.exit_mlir_block_scope(exit_scope_name=scope_name)

    def peek_block_scope(self):
        assert len(self._mlir_block_stack), "no block scope yet"
        return self._mlir_block_stack[-1][:-1]

    @lru_cache(maxsize=None, typed=True)
    def _get_or_make_mlir_constant(
        self,
        py_cst: Union[int, float, bool],
        name: Optional[str] = None,
        index_type: bool = False,
    ):
        if name is None:
            name = str(py_cst)
        # TODO(max): drop asserts eventually when sure this works
        assert (
            name,
            index_type,
        ) not in self.local_defs, f"duplicate constant {py_cst=} {name=} {index_type=}: type {self.local_defs[name].type}"
        if index_type:
            constant = arith.ConstantOp.create_index(py_cst).result
        else:
            constant = arith.ConstantOp(infer_mlir_type(py_cst), py_cst).result
        self.set_mlir_value((name, index_type), constant)
        return constant

    def _get_mlir_value(self, name) -> MLIRValue:
        # Float and Integer nodes that don't have names...
        if name in self.local_defs:
            ret = self.local_defs[name]
        else:
            raise ValueError(f"{name} is not defined")
        return ret

    def set_mlir_value(
        self, name: Union[str, CSTNode, tuple[str, bool]], value: MLIRValue
    ) -> None:
        self.local_defs[name] = value

    def get_mlir_value(self, node: CSTNode) -> MLIRValue:
        # Float and Integer nodes that don't have names...
        if isinstance(node, BaseNumber):
            return self._get_or_make_mlir_constant(literal_eval(node.value))
        elif isinstance(node, BinaryOperation):
            # values associated with expression nodes (e.g. binary expression) are stored as nodes in
            # local_defs
            return self._get_mlir_value(node)
        else:
            name = FindName()(node)
            return self._get_mlir_value(name)

    # visit_<node> methods ###########################################

    @matchers.visit(matchers.Assign())
    @matchers.visit(matchers.Call())
    @matchers.visit(matchers.BinaryOperation())
    @matchers.visit(matchers.Expr())
    @matchers.visit(matchers.Return())
    def visit_(self, _node):
        return True

    def leave_Assign(self, node: Assign) -> Optional[bool]:
        lhs = [n.target for n in node.targets]
        assert len(lhs) == 1, f"multiple assign targets unsupported {lhs}"
        lhs = lhs[0]
        value = self.get_mlir_value(node.value)
        # if a python name is assigned to another python name eg a = b
        while not isinstance(value, MLIRValue):
            assert isinstance(value, CSTNode) and hasattr(
                value, "name"
            ), f"unknown value {value}"
            value = self.get_mlir_value(value.name)
        self.set_mlir_value(lhs.value, value)

    def visit_BinaryOperation_operator(self, node: BinaryOperation) -> None:
        node.left.visit(self)
        node.right.visit(self)
        left_mlir_value = self.get_mlir_value(node.left)
        right_mlir_value = self.get_mlir_value(node.right)
        assert (
            left_mlir_value.type == right_mlir_value.type
        ), f"op requires the same type for all operands and results {left_mlir_value.type=} {right_mlir_value.type=}"
        mul_val = self.body_builder[node.operator.__class__.__name__](
            left_mlir_value, right_mlir_value
        )
        self.set_mlir_value(node, mul_val)

    def leave_Expr(self, node: Expr) -> Optional[bool]:
        return True

    def visit_FunctionDef(self, node: FunctionDef):
        args = node.params
        arg_types = []
        defaults = [p.default for p in args.params]
        # left to right but starting from the right (since you can't have non-defaults after defaults)
        for i, arg in enumerate(args.params):
            if arg.annotation is not None:
                arg_types.append(self.get_metadata(MLIRTypeProvider, arg.annotation))
            elif defaults[i] is not None:
                arg_types.append(self.get_metadata(MLIRTypeProvider, defaults[i]))
            else:
                raise Exception(f"can't infer type for {repr(arg)}")

        (
            _block,
            _block_args,
            _insertion_point,
            location,
        ) = self.peek_block_scope()
        function_type = self.get_metadata(MLIRTypeProvider, node)
        func_op = func.FuncOp(
            name=node.name.value,
            type=function_type,
            # TODO(max): could (and should) be public
            visibility="private",
            loc=location,
        )
        # func_op.attributes["function_type"] = TypeAttr.get(function_type)
        func_op_entry_block = func_op.add_entry_block()
        for i, func_op_arg in enumerate(func_op.arguments):
            self.set_mlir_value(args.params[i].name.value, func_op_arg)

        with self.mlir_block_scope(func_op_entry_block, scope_name=node.name.value) as (
            _block,
            location,
        ):
            node.body.visit(self)

            function_returns = ReturnFinder()(node)
            # Coerce return values, add ReturnOp and rewrite func type.
            if function_returns is not None:
                assert isinstance(
                    function_returns.value, Name
                ), f"complex returns types not supported yet {function_returns}"
                return_values = self.local_defs[function_returns.value.value]
                if isinstance(return_values, tuple):
                    return_values = list(return_values)
                elif isinstance(return_values, MLIRValue):
                    return_values = [return_values]
                elif isinstance(return_values, OpView):
                    return_values = return_values.operation.results
                elif isinstance(return_values, Operation):
                    return_values = return_values.results
                else:
                    return_values = list(return_values)
            else:
                return_values = []

            func.ReturnOp(return_values)
        return False

    def visit_For(self, node: For):
        if node.iter.func.value != "range":
            raise RuntimeError("Only `range` iterator currently supported")
        for arg in node.iter.args:
            arg.visit(self)
            assert isinstance(
                arg.value, Integer
            ), f"symbolic range not supported yet {arg.value}"

        lb = (
            node.iter.args[0].value.value
            if len(node.iter.args) > 1
            else self._get_or_make_mlir_constant(0, index_type=True)
        )
        ub = (
            node.iter.args[1].value.value
            if len(node.iter.args) > 1
            else node.iter.args[0].value.value
        )
        step = (
            node.iter.args[2].value.value
            if len(node.iter.args) > 2
            else self._get_or_make_mlir_constant(1, index_type=True)
        )

        lb, ub, step = [
            self._get_or_make_mlir_constant(literal_eval(c), index_type=True)
            if isinstance(c, str)
            else c
            for c in [lb, ub, step]
        ]
        assert all(
            [isinstance(x, MLIRValue) for x in [lb, ub, step]]
        ), f"something went wrong converting range for {lb=} {ub=} {step=}"

        _live_ins, live_outs, scope_name = self.get_metadata(
            LiveInLiveOutProvider, node
        )
        live_outs = tuple(live_outs)

        (
            func_body_block,
            _block_args,
            _insertion_point,
            location,
        ) = self.peek_block_scope()

        scope = self.get_metadata(MyScopeProvider, node.body)
        yielded_types = []
        for l in live_outs:
            items = scope._getitem_from_self_or_parent(l)
            assert len(items) == 1, f"multiple parent items {l}"
            item = list(items)[0].node
            yielded_types.append(self.get_metadata(MLIRTypeProvider, item))
        undef_ops = [llvm.UndefOp(yt, loc=location) for yt in yielded_types]
        induction_var = node.target
        assert isinstance(
            induction_var, Name
        ), f"complex for range targets unsupported {induction_var}"

        loop = scf.ForOp(lb, ub, step, undef_ops, loc=location)
        self.set_mlir_value(induction_var, loop.induction_variable)
        # i know these aren't the only block args but i'm abusing the api
        with self.mlir_block_scope(
            loop.body,
            block_args=live_outs,
            scope_name=f"{scope_name}",
        ) as (block, location):
            node.body.visit(self)
            yielded_vals = [self._get_mlir_value(y) for y in live_outs]
            scf.YieldOp(yielded_vals, loc=location)

            for i, yielded_var_name in enumerate(live_outs):
                self.local_defs[yielded_var_name] = block.owner.operation.results[i]
                yielded_type = yielded_types[i]

                with InsertionPoint.at_block_begin(func_body_block), location:
                    cst = arith.ConstantOp(yielded_type, 0.0)
                    undef_ops[i].operation.replace_all_uses_with(cst.operation)

            for undef_op in undef_ops:
                undef_op.operation.erase()

        return False

    @matchers.visit(matchers.Float())
    @matchers.visit(matchers.Integer())
    def visit_python_number(self, node: Float | Integer):
        self._get_or_make_mlir_constant(literal_eval(node.value))

    def visit_Module(self, node: Module):
        with self.mlir_block_scope(
            self.module.body, scope_name=node.__class__.__name__
        ):
            for child in node.body:
                child.visit(self)

        return False

from __future__ import annotations

import builtins
import dataclasses
from ast import literal_eval
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from typing import Callable, Optional, Union

from libcst import (
    BinaryOperation,
    Assign,
    Expr,
    For,
    FunctionDef,
    Name,
    CSTNode,
    matchers,
    CSTVisitor,
    Float,
    Integer,
    BaseNumber,
    Parameters,
    IndentedBlock,
    If,
    Else,
    VisitorMetadataProvider,
    Module,
)
from libcst.metadata import (
    ScopeProvider,
    PositionProvider,
    ParentNodeProvider,
    ExpressionContextProvider,
)

from type_visitor import (
    MLIRTypeProvider,
    ReturnFinder,
    infer_mlir_type,
)


def accept(self, visitor, *args, **kwargs):
    """Visit this node using the given visitor."""
    func = getattr(visitor, "visit_" + self.__class__.__name__.lower())
    return func(self, *args, **kwargs)


from shark.dialects.linalg.opdsl.lang.emitter import (
    _is_integer_type,
    _is_floating_point_type,
    _is_index_type,
    _get_floating_point_width,
    _is_complex_type,
)

from shark.dialects.linalg import FunctionKind, ScalarAssign, ScalarExpression

from shark.dialects import arith, linalg, math, func, scf, llvm

from shark.ir import (
    Type as MLIRType,
    Value as MLIRValue,
    Attribute,
    IntegerAttr,
    IntegerType,
    Context,
    Module as MLIRModule,
    Location,
    InsertionPoint,
    OpView,
    Operation,
    Block,
)


class BodyBuilder:
    """Constructs a structured op body by evaluating assignments."""

    def __init__(
        self,
        type_mapping: dict[str, MLIRType],
        block_arg_mapping: dict[str, MLIRValue],
        fn_attr_mapping: dict[str, str],
        location: Location,
    ):
        self.type_mapping = type_mapping
        self.block_arg_mapping = block_arg_mapping
        self.fn_attr_mapping = fn_attr_mapping
        self.yield_mapping = dict()
        self.location = location

    def assign(self, assignment: ScalarAssign):
        if assignment.arg in self.yield_mapping:
            raise ValueError(
                f"Multiple assignments to the same argument are forbidden: "
                f"{assignment}"
            )
        self.yield_mapping[assignment.arg] = self.expression(assignment.value)

    def expression(self, expr: ScalarExpression) -> MLIRValue:
        if expr.scalar_arg:
            try:
                return self.block_arg_mapping[expr.scalar_arg.arg]
            except KeyError:
                raise ValueError(
                    f"Argument {expr.scalar_arg.arg} is not bound for "
                    f"this structured op."
                )
        elif expr.scalar_const:
            value_attr = Attribute.parse(expr.scalar_const.value)
            return arith.ConstantOp(value_attr.type, value_attr).result
        elif expr.scalar_index:
            dim_attr = IntegerAttr.get(
                IntegerType.get_signless(64), expr.scalar_index.dim
            )
            return linalg.IndexOp(dim_attr).result
        elif expr.scalar_fn:
            kind = expr.scalar_fn.kind.name.lower()
            fn_name = expr.scalar_fn.fn_name
            if expr.scalar_fn.attr_name:
                fn_name, _ = self.fn_attr_mapping[expr.scalar_fn.attr_name]
            fn = self.get_function(f"_{kind}_{fn_name}")
            operand_values = [
                self.expression(operand) for operand in expr.scalar_fn.operands
            ]
            if expr.scalar_fn.kind == FunctionKind.TYPE:
                operand_values = [expr.scalar_fn.type_var.name] + operand_values
            return fn(*operand_values)
        raise NotImplementedError(f"Unimplemented scalar body expression: {expr}")

    def yield_outputs(self, *output_names: str):
        output_values = []
        for n in output_names:
            try:
                output_values.append(self.yield_mapping[n])
            except KeyError:
                raise ValueError(
                    f"Body assignments do not assign all outputs: " f"missing '{n}'"
                )
        linalg.YieldOp(output_values)

    def get_function(self, fn_name: str) -> Callable:
        try:
            fn = getattr(self, f"{fn_name}")
        except AttributeError:
            raise ValueError(f"Function '{fn_name}' is not a known function")
        return fn

    def cast(
        self, type_var_name: str, operand: MLIRValue, is_unsigned_cast: bool = False
    ) -> MLIRValue:
        try:
            to_type = self.type_mapping[type_var_name]
        except KeyError:
            raise ValueError(
                f"Unbound type variable '{type_var_name}' ("
                f"expected one of {self.type_mapping.keys()}"
            )
        if operand.type == to_type:
            return operand
        if _is_integer_type(to_type):
            return self.cast_to_integer(to_type, operand, is_unsigned_cast)
        elif _is_floating_point_type(to_type):
            return self.cast_to_floating_point(to_type, operand, is_unsigned_cast)

    def cast_to_integer(
        self, to_type: MLIRType, operand: MLIRValue, is_unsigned_cast: bool
    ) -> MLIRValue:
        to_width = IntegerType(to_type).width
        operand_type = operand.type
        if _is_floating_point_type(operand_type):
            if is_unsigned_cast:
                return arith.FPToUIOp(to_type, operand, loc=self.location).result
            return arith.FPToSIOp(to_type, operand, loc=self.location).result
        if _is_index_type(operand_type):
            return arith.IndexCastOp(to_type, operand, loc=self.location).result
        # Assume integer.
        from_width = IntegerType(operand_type).width
        if to_width > from_width:
            if is_unsigned_cast:
                return arith.ExtUIOp(to_type, operand, loc=self.location).result
            return arith.ExtSIOp(to_type, operand, loc=self.location).result
        elif to_width < from_width:
            return arith.TruncIOp(to_type, operand, loc=self.location).result
        raise ValueError(
            f"Unable to cast body expression from {operand_type} to " f"{to_type}"
        )

    def cast_to_floating_point(
        self, to_type: MLIRType, operand: MLIRValue, is_unsigned_cast: bool
    ) -> MLIRValue:
        operand_type = operand.type
        if _is_integer_type(operand_type):
            if is_unsigned_cast:
                return arith.UIToFPOp(to_type, operand, loc=self.location).result
            return arith.SIToFPOp(to_type, operand, loc=self.location).result
        # Assume FloatType.
        to_width = _get_floating_point_width(to_type)
        from_width = _get_floating_point_width(operand_type)
        if to_width > from_width:
            return arith.ExtFOp(to_type, operand, loc=self.location).result
        elif to_width < from_width:
            return arith.TruncFOp(to_type, operand, loc=self.location).result
        raise ValueError(
            f"Unable to cast body expression from {operand_type} to " f"{to_type}"
        )

    def type_cast_signed(self, type_var_name: str, operand: MLIRValue) -> MLIRValue:
        return self.cast(type_var_name, operand, False)

    def type_cast_unsigned(self, type_var_name: str, operand: MLIRValue) -> MLIRValue:
        return self.cast(type_var_name, operand, True)

    def unary_exp(self, x: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(x.type):
            return math.ExpOp(x, loc=self.location).result
        raise NotImplementedError("Unsupported 'exp' operand: {x}")

    def unary_log(self, x: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(x.type):
            return math.LogOp(x, loc=self.location).result
        raise NotImplementedError("Unsupported 'log' operand: {x}")

    def unary_abs(self, x: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(x.type):
            return math.AbsFOp(x, loc=self.location).result
        raise NotImplementedError("Unsupported 'abs' operand: {x}")

    def unary_ceil(self, x: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(x.type):
            return math.CeilOp(x, loc=self.location).result
        raise NotImplementedError("Unsupported 'ceil' operand: {x}")

    def unary_floor(self, x: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(x.type):
            return math.FloorOp(x, loc=self.location).result
        raise NotImplementedError("Unsupported 'floor' operand: {x}")

    def unary_negf(self, x: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(x.type):
            return arith.NegFOp(x, loc=self.location).result
        if _is_complex_type(x.type):
            return complex.NegOp(x, loc=self.location).result
        raise NotImplementedError("Unsupported 'negf' operand: {x}")

    def binary_add(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.AddFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            if _is_index_type(lhs.type):
                lhs = self.cast_to_integer(
                    IntegerType.get_signed(64), lhs, is_unsigned_cast=True
                )
            if _is_index_type(rhs.type):
                rhs = self.cast_to_integer(
                    IntegerType.get_signed(64), rhs, is_unsigned_cast=True
                )
            return arith.AddIOp(lhs, rhs, loc=self.location).result
        if _is_complex_type(lhs.type):
            return complex.AddOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'add' operands: {lhs}, {rhs}")

    def binary_sub(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.SubFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            return arith.SubIOp(lhs, rhs, loc=self.location).result
        if _is_complex_type(lhs.type):
            return complex.SubOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'sub' operands: {lhs}, {rhs}")

    def binary_mul(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.MulFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            return arith.MulIOp(lhs, rhs, loc=self.location).result
        if _is_complex_type(lhs.type):
            return complex.MulOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'mul' operands: {lhs}, {rhs}")

    def binary_max_signed(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.MaxFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            return arith.MaxSIOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'max' operands: {lhs}, {rhs}")

    def binary_max_unsigned(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.MaxFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            return arith.MaxUIOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'max_unsigned' operands: {lhs}, {rhs}")

    def binary_min_signed(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.MinFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            return arith.MinSIOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'min' operands: {lhs}, {rhs}")

    def binary_min_unsigned(self, lhs: MLIRValue, rhs: MLIRValue) -> MLIRValue:
        if _is_floating_point_type(lhs.type):
            return arith.MinFOp(lhs, rhs, loc=self.location).result
        if _is_integer_type(lhs.type) or _is_index_type(lhs.type):
            return arith.MinUIOp(lhs, rhs, loc=self.location).result
        raise NotImplementedError("Unsupported 'min_unsigned' operands: {lhs}, {rhs}")

    def __getitem__(self, bin_op: str):
        return {
            "Multiply": self.binary_mul,
            "Add": self.binary_add,
            "Subtract": self.binary_sub,
        }[bin_op]


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


@dataclasses.dataclass
class LiveInsLiveOuts:
    live_ins: set[str]
    live_outs: set[str]
    scope: str

    def __iter__(self):
        yield from [self.live_ins, self.live_outs, self.scope]


class LiveInLiveOutProvider(
    VisitorMetadataProvider[LiveInsLiveOuts], matchers.MatcherDecoratableVisitor
):
    METADATA_DEPENDENCIES = (
        ScopeProvider,
        ParentNodeProvider,
        PositionProvider,
        ExpressionContextProvider,
    )

    new_for_scopes = 0

    @staticmethod
    def gen_cache(*args, **kwargs) -> dict[str, object]:
        pass

    def __init__(self, cache: tuple[dict[str, set[str], dict[str, set[str]]]]) -> None:
        super().__init__()
        self.live_ins, self.live_outs = cache

    def visit_For(self, node: For) -> Optional[bool]:
        self.new_for_scopes += 1
        for_loop_n = f"for_{self.new_for_scopes}"
        self.set_metadata(
            node,
            LiveInsLiveOuts(
                self.live_ins[for_loop_n], self.live_outs[for_loop_n], for_loop_n
            ),
        )
        return True

    # def visit_If(self, node: If) -> Optional[bool]:
    #     scope = self.get_metadata(ScopeProvider, node)
    #     return True
    #
    # def visit_Else(self, node: Else) -> Optional[bool]:
    #     scope = self.get_metadata(ScopeProvider, node)
    #     return True


class CompilerVisitor(matchers.MatcherDecoratableVisitor):
    METADATA_DEPENDENCIES = (
        ScopeProvider,
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
        elif isinstance(node, (BinaryOperation,)):
            # values associated with expression nodes (e.g. binary expression) are stored as nodes in
            # local_defs
            return self._get_mlir_value(node)
        else:
            name = FindName()(node)
            return self._get_mlir_value(name)

    def node_qual_name(self, node: CSTNode):
        qual_name = self.get_metadata(ScopeProvider, node).get_qualified_names_for(node)
        assert len(qual_name) == 1, f"wtfbbq wrong number qualnames {qual_name}"
        qual_name = list(qual_name)[0].name
        return qual_name

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

    # def leave_AnnAssign(self, node: AnnAssign) -> Optional[bool]:
    #     target = node.target.accept(self)
    #     annotation = node.annotation.accept(self)
    #     return f"{target}: {annotation} = {node.value.accept(self)}"

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

        scope = self.get_metadata(ScopeProvider, node)
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


class ScopeTransformer(matchers.MatcherDecoratableTransformer):
    new_for_scopes = 0

    def leave_For(self, original_node: For, updated_node: For) -> CSTNode:
        self.new_for_scopes += 1
        return FunctionDef(
            name=Name(value=f"for_{self.new_for_scopes}"),
            params=Parameters(),
            body=IndentedBlock(body=[updated_node]),
        )


class LiveInLiveOutVisitor(matchers.MatcherDecoratableVisitor):
    METADATA_DEPENDENCIES = (
        ScopeProvider,
        ParentNodeProvider,
        PositionProvider,
        ExpressionContextProvider,
    )

    live_ins = defaultdict(set)
    live_outs = defaultdict(set)
    builtins = set(builtins.__dict__.keys())

    def visit_For(self, node: For) -> Optional[bool]:
        scope = self.get_metadata(ScopeProvider, node)
        assignments = set(scope._assignments.keys()) - {node.target.value}
        accesses = scope.accesses._accesses.keys() - {node.target.value} - self.builtins
        self.live_ins[scope.name] = accesses - assignments
        true_function_scope = self.get_metadata(ScopeProvider, scope.node)
        self.live_outs[scope.name] = self.live_ins[
            true_function_scope.name
        ].intersection(assignments)
        return True

    def visit_If(self, node: If) -> Optional[bool]:
        scope = self.get_metadata(ScopeProvider, node)
        return True

    def visit_Else(self, node: Else) -> Optional[bool]:
        scope = self.get_metadata(ScopeProvider, node)
        return True

    def visit_FunctionDef(self, node: FunctionDef) -> Optional[bool]:
        scope = self.get_metadata(ScopeProvider, node.body)
        assignments = set(scope._assignments.keys())
        accesses = scope.accesses._accesses.keys() - self.builtins
        self.live_ins[scope.name] = accesses - assignments

        return True


# def get_live_range(self, node: Name):
#     if not self._live_ranges:
#         self._live_ranges = defaultdict(dict)
#         function_scope = self.get_metadata(ScopeProvider, node)
#         assert isinstance(
#             function_scope, FunctionScope
#         ), f"other scopes not supported yet {function_scope.__class__.__name__}"
#         for name, assigns in function_scope.assignments._assignments.items():
#             assign_lines = [
#                 self.get_metadata(PositionProvider, a.node) for a in assigns
#             ]
#             assert len(assign_lines) == 1, f"reassignment not supported yet {name}"
#             self._live_ranges[name]["assign_line"] = assign_lines[0].start.line
#             # for access in accesses:
#         for name, accesses in function_scope.accesses._accesses.items():
#             access_lines = [
#                 self.get_metadata(PositionProvider, a.node) for a in accesses
#             ]
#             self._live_ranges[name]["last_access_line"] = max(
#                 a.start.line for a in access_lines
#             )
#
#         live_range_intervals = set()
#         for name, live_range in self._live_ranges.items():
#             if "assign_line" not in live_range:
#                 warnings.warn(f"no assignment line; probably a builtin {name}")
#                 continue
#
#             assign_line = live_range["assign_line"]
#             if "last_access_line" not in live_range:
#                 warnings.warn(f"no access line; probably unused variable {name}")
#                 continue
#
#             last_access_line = live_range["last_access_line"]
#             live_range_interval = Interval(begin=assign_line, end=last_access_line)
#             live_range_intervals.add(live_range_interval)
#
#             self._live_ranges_tree = IntervalTree(live_range_intervals)
#     return self._live_ranges[node.value]

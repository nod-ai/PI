from __future__ import annotations

import inspect
import sys
import warnings
from typing import TYPE_CHECKING, Callable, Sequence, Optional, Union
from types import FunctionType, MethodType

import astroid
from astroid import ClassDef, NodeNG
from astroid.builder import AstroidBuilder


def accept(self, visitor, *args, **kwargs):
    """Visit this node using the given visitor."""
    func = getattr(visitor, "visit_" + self.__class__.__name__.lower())
    return func(self, *args, **kwargs)


NodeNG.accept = accept

from astroid.manager import AstroidManager

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
    Module,
    Location,
    InsertionPoint,
    FunctionType as MLIRFunctionType,
    OpView,
    Operation,
    TypeAttr,
    F64Type,
    IndexType,
)

if TYPE_CHECKING:
    from astroid.nodes import Const
    from astroid.nodes.node_classes import (
        Match,
        MatchAs,
        MatchCase,
        MatchClass,
        MatchMapping,
        MatchOr,
        MatchSequence,
        MatchSingleton,
        MatchStar,
        MatchValue,
        Unknown,
        Assign,
        AugAssign,
    )

# pylint: disable=unused-argument

DOC_NEWLINE = "\0"


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
            "*": self.binary_mul,
            "+": self.binary_add,
            "-": self.binary_sub,
        }[bin_op]


def map_to_mlir_type(thing, context=None) -> MLIRType:
    # catch all that should be factored
    if context is None:
        context = Context()
    with context:
        if isinstance(thing, str):
            return {
                "int": IntegerType.get_signed(64),
                "float": F64Type.get(),
            }[thing]
        else:
            raise Exception(f"unimplemented type mapping for {thing}")


def infer_mlir_type(py_val) -> MLIRType:
    if isinstance(py_val, int):
        return IntegerType.get_signed(64)
    elif isinstance(py_val, float):
        return F64Type.get()
    else:
        raise Exception(f"unsupported val type {type(py_val)} {py_val}")


class CompilerVisitor:
    def __init__(
        self,
        mlir_context,
        module,
        location,
        py_constants=None,
        py_global_scope=None,
        global_scope=None,
    ):
        if global_scope is None:
            global_scope = {}
        if py_global_scope is None:
            py_global_scope = {}
        if py_constants is None:
            py_constants = {}
        self.mlir_context = mlir_context
        self.module = module
        self.current_block = self.module.body
        self.location = location
        self.body_builder = BodyBuilder({}, {}, {}, self.location)
        self.py_constants = py_constants
        self.py_global_scope = py_global_scope
        self.local_scope = {}
        self.global_scope = global_scope
        self.local_defs = {}
        self.global_uses = {}
        self.last_ret_node = None
        self.builtins = {
            "range": range,
            "min": min,
            "max": max,
            "float": float,
            "int": int,
            "print": print,
        }

    def _get_or_make_mlir_constant(
        self,
        py_cst: Union[int, float, bool],
        name: Optional[str] = None,
        index_type: bool = False,
    ):
        with InsertionPoint.at_block_begin(self.current_block), self.location:
            if index_type:
                if (py_cst, index_type) not in self.py_constants:
                    cst = arith.ConstantOp.create_index(py_cst).result
                    self.py_constants[py_cst, index_type] = cst
                cst = self.py_constants[py_cst, index_type]
            else:
                if py_cst not in self.py_constants:
                    cst = arith.ConstantOp(infer_mlir_type(py_cst), py_cst).result
                    self.py_constants[py_cst] = cst
                cst = self.py_constants[py_cst]
        # this is bad/doesn't make sense but otherwise these two branches would be repeated in the above two
        # branches
        if name:
            self.set_mlir_value(name, cst)
        else:
            self.set_mlir_value(f"c{cst.result_number}", cst)
        return cst

    def get_mlir_value(self, name) -> MLIRValue:
        if name in self.local_scope:
            ret = self.local_scope[name]
            if name not in self.local_defs:
                self.global_uses[name] = ret
        # search node.id in global scope
        elif name in self.global_scope:
            ret = self.global_scope[name]
        # search node.id in builtins
        elif name in self.builtins:
            ret = self.builtins[name]
        else:
            raise ValueError(f"{name} is not defined")
        return ret

    def set_mlir_value(self, name: str, value: MLIRValue) -> None:
        """This function is called by visit_Assign() & visit_FuncDef() to create left values (lvalue); ie mapping
        python names to MLIR Values which have their own names (unfortunately there's no way to tell MLIR to rename a variable."""
        self.local_defs[name] = self.local_scope[name] = value

    def __call__(self, node):
        return node.accept(self)

    def _stmt_list(self, stmts):
        for stmt in stmts:
            self.last_ret_node = stmt.accept(self)
            if isinstance(stmt, astroid.Return):
                break
        return stmts and isinstance(stmt, astroid.Return)

    def _precedence(self, node, child, is_left=True):
        val = child.accept(self)
        if isinstance(val, MLIRValue):
            raise Exception("wtfbbq never handled this...")

        return val

    def _should_wrap(self, node, child, is_left):
        node_precedence = node.op_precedence()
        child_precedence = child.op_precedence()

        if node_precedence > child_precedence:
            # 3 * (4 + 5)
            return True

        if (
            node_precedence == child_precedence
            and is_left != node.op_left_associative()
        ):
            # 3 - (4 - 5)
            # (2**3)**4
            return True

        return False

    # visit_<node> methods ###########################################

    def visit_assignattr(self, node):
        return self.visit_attribute(node)

    def visit_assert(self, node):
        if node.fail:
            return f"assert {node.test.accept(self)}, {node.fail.accept(self)}"
        return f"assert {node.test.accept(self)}"

    def visit_assignname(self, node):
        return node

    def visit_assign(self, node):
        lhs = [n.accept(self) for n in node.targets]
        assert len(lhs) == 1, f"multiple assign targets unsupported {lhs}"
        lhs = lhs[0]
        value = node.value.accept(self)
        while not isinstance(value, MLIRValue):
            assert isinstance(value, NodeNG) and hasattr(
                value, "name"
            ), f"unknown value {value}"
            value = self.get_mlir_value(value.name)
        self.set_mlir_value(lhs.name, value)
        return lhs

    def visit_annassign(self, node):
        target = node.target.accept(self)
        annotation = node.annotation.accept(self)
        return f"{target}: {annotation} = {node.value.accept(self)}"

    def visit_binop(self, node):
        left = self._precedence(node, node.left)
        right = self._precedence(node, node.right, is_left=False)
        left_mlir_value = self.local_defs[left.name]
        right_mlir_value = self.local_defs[right.name]
        mul_val = self.body_builder[node.op](left_mlir_value, right_mlir_value)
        return mul_val

    def visit_boolop(self, node):
        values = [f"{self._precedence(node, n)}" for n in node.values]
        return (f" {node.op} ").join(values)

    def visit_break(self, node):
        return "break"

    def visit_call(self, node):
        expr_str = self._precedence(node, node.func)
        args = [arg.accept(self) for arg in node.args]
        if node.keywords:
            keywords = [kwarg.accept(self) for kwarg in node.keywords]
        else:
            keywords = []

        args.extend(keywords)
        return f"{expr_str}({', '.join(args)})"

    def visit_classdef(self, node):
        decorate = node.decorators.accept(self) if node.decorators else ""
        args = [n.accept(self) for n in node.bases]
        if node._metaclass and not node.has_metaclass_hack():
            args.append("metaclass=" + node._metaclass.accept(self))
        args += [n.accept(self) for n in node.keywords]
        args = f"({', '.join(args)})" if args else ""
        return "\n\n{}class {}{}:{}\n{}\n".format(
            decorate, node.name, args, docs, self._stmt_list(node.body)
        )

    def visit_compare(self, node):
        rhs_str = " ".join(
            f"{op} {self._precedence(node, expr, is_left=False)}"
            for op, expr in node.ops
        )
        return f"{self._precedence(node, node.left)} {rhs_str}"

    def visit_comprehension(self, node):
        ifs = "".join(f" if {n.accept(self)}" for n in node.ifs)
        generated = f"for {node.target.accept(self)} in {node.iter.accept(self)}{ifs}"
        return f"{'async ' if node.is_async else ''}{generated}"

    def visit_const(self, node, *args, **kwargs):
        return self._get_or_make_mlir_constant(
            node.value, index_type=kwargs.get("index_type", False)
        )

    def visit_continue(self, node):
        return "continue"

    def visit_delete(self, node):  # XXX check if correct

        return f"del {', '.join(child.accept(self) for child in node.targets)}"

    def visit_delattr(self, node):
        return self.visit_attribute(node)

    def visit_delname(self, node):
        return node.name

    def visit_decorators(self, node):
        return "@%s\n" % "\n@".join(item.accept(self) for item in node.nodes)

    def visit_dict(self, node):
        return "{%s}" % ", ".join(self._visit_dict(node))

    def _visit_dict(self, node):
        for key, value in node.items:
            key = key.accept(self)
            value = value.accept(self)
            if key == "**":
                # It can only be a DictUnpack node.
                yield key + value
            else:
                yield f"{key}: {value}"

    def visit_dictunpack(self, node):
        return "**"

    def visit_dictcomp(self, node):
        return "{{{}: {} {}}}".format(
            node.key.accept(self),
            node.value.accept(self),
            " ".join(n.accept(self) for n in node.generators),
        )

    def visit_expr(self, node):
        return node.value.accept(self)

    def visit_emptynode(self, node):
        return ""

    def visit_excepthandler(self, node):
        if node.type:
            if node.name:
                excs = f"except {node.type.accept(self)} as {node.name.accept(self)}"
            else:
                excs = f"except {node.type.accept(self)}"
        else:
            excs = "except"
        return f"{excs}:\n{self._stmt_list(node.body)}"

    def visit_empty(self, node):
        return ""

    def visit_for(self, for_loop_node):
        if for_loop_node.iter.func.name != "range":
            raise RuntimeError("Only `range` iterator currently supported")

        iter_args = [
            arg.accept(self, index_type=True) for arg in for_loop_node.iter.args
        ]
        lb = (
            iter_args[0]
            if len(iter_args) > 1
            else self._get_or_make_mlir_constant(0, index_type=True)
        )
        ub = iter_args[1] if len(iter_args) > 1 else iter_args[0]
        step = (
            iter_args[2]
            if len(iter_args) > 2
            else self._get_or_make_mlir_constant(1, index_type=True)
        )

        lb, ub, step = [
            self._get_or_make_mlir_constant(c, index_type=True)
            if isinstance(c, (float, bool, int))
            else c
            for c in [lb, ub, step]
        ]

        # TODO(max): this is a hack for liveins
        function_scope = for_loop_node.scope()
        all_dominated_names = set(
            n.name
            for n in function_scope._get_name_nodes()
            if n.lineno > for_loop_node.end_lineno
        )
        loop_carried_vars = set()
        yielded_var_names = set()
        for assign in for_loop_node._get_assign_nodes():
            assert (
                len(assign.targets) == 1
            ), f"multiple assignment not supported {assign}"
            assign_name = assign.targets[0].name
            if assign_name in all_dominated_names:
                yielded_var_names.add(assign_name)

            # If a variable (name) is defined in both its parent & itself, then it's
            # a loop-carried variable. (They must be of the same type)
            _function_scope, assigns = function_scope.lookup(assign_name)
            # you can't have local appear after their first use (you can have globals though...)
            if any([a.lineno < for_loop_node.lineno for a in assigns]):
                loop_carried_vars.add(self.get_mlir_value(assign_name))

        yielded_var_names = list(yielded_var_names)
        # TODO(maX): hack
        throwaway_module = Module.create(loc=self.location)
        with InsertionPoint.at_block_begin(throwaway_module.body):
            self._stmt_list(for_loop_node.body)
        yielded_types = [self.local_defs[y].type for y in yielded_var_names]
        undef_ops = [llvm.UndefOp(y, loc=self.location) for y in yielded_types]
        loop_carried_vars = list(loop_carried_vars)
        induction_var = for_loop_node.target.accept(self)
        loop = scf.ForOp(lb, ub, step, loop_carried_vars + undef_ops, loc=self.location)
        self.set_mlir_value(induction_var, loop.induction_variable)
        with InsertionPoint(loop.body):
            self._stmt_list(for_loop_node.body)
            yielded_vals = [self.get_mlir_value(y) for y in yielded_var_names]
            scf.YieldOp(yielded_vals, loc=self.location)

        for i, yielded_var_name in enumerate(yielded_var_names):
            self.local_defs[yielded_var_name] = loop.results[i]
            yielded_type = yielded_types[i]
            with InsertionPoint.at_block_begin(self.current_block), self.location:
                cst = arith.ConstantOp(yielded_type, 0.0)
                undef_ops[i].operation.replace_all_uses_with(cst.operation)

        for undef_op in undef_ops:
            undef_op.operation.erase()

        return loop

    def visit_importfrom(self, node):
        return "from {} import {}".format(
            "." * (node.level or 0) + node.modname, _import_string(node.names)
        )

    def visit_joinedstr(self, node):
        string = "".join(
            # Use repr on the string literal parts
            # to get proper escapes, e.g. \n, \\, \"
            # But strip the quotes off the ends
            # (they will always be one character: ' or ")
            repr(value.value)[1:-1]
            # Literal braces must be doubled to escape them
            .replace("{", "{{").replace("}", "}}")
            # Each value in values is either a string literal (Const)
            # or a FormattedValue
            if type(value).__name__ == "Const" else value.accept(self)
            for value in node.values
        )

        # Try to find surrounding quotes that don't appear at all in the string.
        # Because the formatted values inside {} can't contain backslash (\)
        # using a triple quote is sometimes necessary
        for quote in ("'", '"', '""', "'''"):
            if quote not in string:
                break

        return "f" + quote + string + quote

    def visit_formattedvalue(self, node):
        result = node.value.accept(self)
        if node.conversion and node.conversion >= 0:
            # e.g. if node.conversion == 114: result += "!r"
            result += "!" + chr(node.conversion)
        if node.format_spec:
            # The format spec is itself a JoinedString, i.e. an f-string
            # We strip the f and quotes of the ends
            result += ":" + node.format_spec.accept(self)[2:-1]
        return "{%s}" % result

    def visit_functiondef(self, node):
        # TODO(max): useful later
        # if node.decorators:
        #     decorate = node.decorators.accept(self)
        args = node.args
        arg_types = []
        defaults = [None] * len(args.args)
        # left to right but starting from the right (since you can't have non-defaults after defaults)
        defaults[-len(args.defaults) :] = args.defaults[:]
        for i, arg in enumerate(args.args):
            if args.annotations[i] is not None:
                annot = args.annotations[i].accept(self)
                if inferred := next(annot.infer()):
                    if (
                        isinstance(inferred, ClassDef)
                        and inferred.pytype() == "builtins.type"
                    ):
                        arg_types.append(
                            map_to_mlir_type(inferred.name, self.mlir_context)
                        )
                    else:
                        raise Exception(f"unhandled annotation {annot}")
                else:
                    raise Exception(f"unhandled annotation {annot}")
            elif defaults[i] is not None:
                default = defaults[i].accept(self)
                arg_types.append(default.type)
            elif inferred := next(arg.infer_lhs()):
                print("wtfbbq inferred type in args")
            else:
                raise Exception(f"can't infer type for {repr(arg)}")
        # these are return annotations
        return_types = []
        if node.returns:
            return_types = [
                map_to_mlir_type(node.returns.accept(self), self.mlir_context)
            ]
        elif all_returns := list(node._get_return_nodes_skip_functions()):
            # TODO(max): check here for results from mlir python bindingins
            # see from_py_func in _func_ops_ext.py
            for ret in all_returns:
                ret = ret.accept(self)
                inferred = next(ret.infer())
                if not inferred:
                    warnings.warn(f"ret {ret} failed inference {inferred}")
                    continue
                pytype = inferred.pytype().replace("builtins.", "")
                return_types.append(map_to_mlir_type(pytype, self.mlir_context))
                # TODO(max): obviously this is wrong
                break

        with self.mlir_context:
            with InsertionPoint(self.module.body):
                implicit_return = len(return_types) == 0
                function_type = MLIRFunctionType.get(
                    inputs=arg_types, results=[] if implicit_return else return_types
                )
                func_op = func.FuncOp(
                    name=node.name,
                    type=function_type,
                    # TODO(maX): could (and should) be public
                    visibility="private",
                    loc=self.location,
                )
                with InsertionPoint(func_op.add_entry_block()):
                    self.current_block = func_op.entry_block
                    for i, arg in enumerate(func_op.entry_block.arguments):
                        self.set_mlir_value(args.args[i].name, arg)
                    has_return = self._stmt_list(node.body)
                    # Coerce return values, add ReturnOp and rewrite func type.
                    if has_return:
                        return_values = self.local_defs[self.last_ret_node.name]
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
                    # Recompute the function type.
                    return_types = [v.type for v in return_values]
                    function_type = MLIRFunctionType.get(
                        inputs=arg_types, results=return_types
                    )
                    func_op.attributes["function_type"] = TypeAttr.get(function_type)

                    func.ReturnOp(return_values, loc=self.location)

        return func_op

    def visit_generatorexp(self, node):
        return "({} {})".format(
            node.elt.accept(self), " ".join(n.accept(self) for n in node.generators)
        )

    def visit_attribute(self, node):
        left = self._precedence(node, node.expr)
        if left.isdigit():
            left = f"({left})"
        return f"{left}.{node.attrname}"

    def visit_global(self, node):
        return f"global {', '.join(node.names)}"

    def visit_if(self, node):
        ifs = [f"if {node.test.accept(self)}:\n{self._stmt_list(node.body)}"]
        if node.has_elif_block():
            ifs.append(f"el{self._stmt_list(node.orelse, indent=False)}")
        elif node.orelse:
            ifs.append(f"else:\n{self._stmt_list(node.orelse)}")
        return "\n".join(ifs)

    def visit_ifexp(self, node):
        return "{} if {} else {}".format(
            self._precedence(node, node.body, is_left=True),
            self._precedence(node, node.test, is_left=True),
            self._precedence(node, node.orelse, is_left=False),
        )

    def visit_import(self, node):
        return f"import {_import_string(node.names)}"

    def visit_keyword(self, node):
        if node.arg is None:
            return f"**{node.value.accept(self)}"
        return f"{node.arg}={node.value.accept(self)}"

    def visit_lambda(self, node):
        args = node.args.accept(self)
        body = node.body.accept(self)
        if args:
            return f"lambda {args}: {body}"

        return f"lambda: {body}"

    def visit_list(self, node):
        return f"[{', '.join(child.accept(self) for child in node.elts)}]"

    def visit_listcomp(self, node):
        return "[{} {}]".format(
            node.elt.accept(self), " ".join(n.accept(self) for n in node.generators)
        )

    def visit_module(self, node):
        # first stop
        return [n.accept(self) for n in node.body]

    def visit_name(self, node):
        return node

    def visit_namedexpr(self, node):
        target = node.target.accept(self)
        value = node.value.accept(self)
        return f"{target} := {value}"

    def visit_nonlocal(self, node):
        return f"nonlocal {', '.join(node.names)}"

    def visit_pass(self, node):
        return "pass"

    def visit_raise(self, node):
        if node.exc:
            if node.cause:
                return f"raise {node.exc.accept(self)} from {node.cause.accept(self)}"
            return f"raise {node.exc.accept(self)}"
        return "raise"

    def visit_return(self, node):
        if node.is_tuple_return() and len(node.value.elts) > 1:
            return [child.accept(self) for child in node.value.elts]

        if node.value:
            return node.value.accept(self)

        return "return"

    def visit_set(self, node):
        return "{%s}" % ", ".join(child.accept(self) for child in node.elts)

    def visit_setcomp(self, node):
        return "{{{} {}}}".format(
            node.elt.accept(self), " ".join(n.accept(self) for n in node.generators)
        )

    def visit_slice(self, node):
        lower = node.lower.accept(self) if node.lower else ""
        upper = node.upper.accept(self) if node.upper else ""
        step = node.step.accept(self) if node.step else ""
        if step:
            return f"{lower}:{upper}:{step}"
        return f"{lower}:{upper}"

    def visit_subscript(self, node):
        idx = node.slice
        if idx.__class__.__name__.lower() == "index":
            idx = idx.value
        idxstr = idx.accept(self)
        if idx.__class__.__name__.lower() == "tuple" and idx.elts:
            # Remove parenthesis in tuple and extended slice.
            # a[(::1, 1:)] is not valid syntax.
            idxstr = idxstr[1:-1]
        return f"{self._precedence(node, node.value)}[{idxstr}]"

    def visit_tryexcept(self, node):
        trys = [f"try:\n{self._stmt_list(node.body)}"]
        for handler in node.handlers:
            trys.append(handler.accept(self))
        if node.orelse:
            trys.append(f"else:\n{self._stmt_list(node.orelse)}")
        return "\n".join(trys)

    def visit_tryfinally(self, node):
        return "try:\n{}\nfinally:\n{}".format(
            self._stmt_list(node.body), self._stmt_list(node.finalbody)
        )

    def visit_tuple(self, node):
        if len(node.elts) == 1:
            return f"({node.elts[0].accept(self)}, )"
        return f"({', '.join(child.accept(self) for child in node.elts)})"

    def visit_unaryop(self, node):
        if node.op == "not":
            operator = "not "
        else:
            operator = node.op
        return f"{operator}{self._precedence(node, node.operand)}"

    def visit_while(self, node):
        whiles = f"while {node.test.accept(self)}:\n{self._stmt_list(node.body)}"
        if node.orelse:
            whiles = f"{whiles}\nelse:\n{self._stmt_list(node.orelse)}"
        return whiles

    def visit_with(self, node):  # 'with' without 'as' is possible

        items = ", ".join(
            f"{expr.accept(self)}" + (v and f" as {v.accept(self)}" or "")
            for expr, v in node.items
        )
        return f"with {items}:\n{self._stmt_list(node.body)}"

    def visit_yield(self, node):
        yi_val = (" " + node.value.accept(self)) if node.value else ""
        expr = "yield" + yi_val
        if node.parent.is_statement:
            return expr

        return f"({expr})"

    def visit_yieldfrom(self, node):
        yi_val = (" " + node.value.accept(self)) if node.value else ""
        expr = "yield from" + yi_val
        if node.parent.is_statement:
            return expr

        return f"({expr})"

    def visit_starred(self, node):
        return "*" + node.value.accept(self)

    def visit_match(self, node: Match) -> str:
        return f"match {node.subject.accept(self)}:\n{self._stmt_list(node.cases)}"

    def visit_matchcase(self, node: MatchCase) -> str:
        guard_str = f" if {node.guard.accept(self)}" if node.guard else ""
        return (
            f"case {node.pattern.accept(self)}{guard_str}:\n"
            f"{self._stmt_list(node.body)}"
        )

    def visit_matchvalue(self, node: MatchValue) -> str:
        return node.value.accept(self)

    def visit_matchsingleton(node: MatchSingleton) -> str:
        return str(node.value)

    def visit_matchsequence(self, node: MatchSequence) -> str:
        if node.patterns is None:
            return "[]"
        return f"[{', '.join(p.accept(self) for p in node.patterns)}]"

    def visit_matchmapping(self, node: MatchMapping) -> str:
        mapping_strings: list[str] = []
        if node.keys and node.patterns:
            mapping_strings.extend(
                f"{key.accept(self)}: {p.accept(self)}"
                for key, p in zip(node.keys, node.patterns)
            )
        if node.rest:
            mapping_strings.append(f"**{node.rest.accept(self)}")
        return f"{'{'}{', '.join(mapping_strings)}{'}'}"

    def visit_matchclass(self, node: MatchClass) -> str:
        if node.cls is None:
            raise Exception(f"{node} does not have a 'cls' node")
        class_strings: list[str] = []
        if node.patterns:
            class_strings.extend(p.accept(self) for p in node.patterns)
        if node.kwd_attrs and node.kwd_patterns:
            for attr, pattern in zip(node.kwd_attrs, node.kwd_patterns):
                class_strings.append(f"{attr}={pattern.accept(self)}")
        return f"{node.cls.accept(self)}({', '.join(class_strings)})"

    def visit_matchstar(self, node: MatchStar) -> str:
        return f"*{node.name.accept(self) if node.name else '_'}"

    def visit_matchas(self, node: MatchAs) -> str:
        # pylint: disable=import-outside-toplevel
        # Prevent circular dependency
        from astroid.nodes.node_classes import MatchClass, MatchMapping, MatchSequence

        if isinstance(node.parent, (MatchSequence, MatchMapping, MatchClass)):
            return node.name.accept(self) if node.name else "_"
        return (
            f"{node.pattern.accept(self) if node.pattern else '_'}"
            f"{f' as {node.name.accept(self)}' if node.name else ''}"
        )

    def visit_matchor(self, node: MatchOr) -> str:
        if node.patterns is None:
            raise Exception(f"{node} does not have pattern nodes")
        return " | ".join(p.accept(self) for p in node.patterns)

    # These aren't for real AST nodes, but for inference objects.

    def visit_frozenset(self, node):
        return node.parent.accept(self)

    def visit_super(self, node):
        return node.parent.accept(self)

    def visit_uninferable(self, node):
        return str(node)

    def visit_property(self, node):
        return node.function.accept(self)

    def visit_evaluatedobject(self, node):
        return node.original.accept(self)

    def visit_unknown(self, node: Unknown) -> str:
        return str(node)


def _import_string(names):
    _names = []
    for name, asname in names:
        if asname is not None:
            _names.append(f"{name} as {asname}")
        else:
            _names.append(name)
    return ", ".join(_names)


def mlir_compile(fn, globals=None):
    manager = AstroidManager()
    if globals is None:
        globals = sys.modules[fn.__module__].__dict__
    fn_source = inspect.getsource(fn)
    fn_ast = manager.ast_from_string(fn_source)
    mlir_context = Context()
    mlir_location_unknown = Location.unknown(context=mlir_context)
    mlir_module = Module.create(loc=mlir_location_unknown)
    compiler_visitor = CompilerVisitor(
        mlir_context, mlir_module, mlir_location_unknown, py_global_scope=globals
    )
    compiler_visitor(fn_ast)
    return compiler_visitor.module

import ast
from enum import Enum
from typing import List, cast

from pyccolo import fast, TraceEvent
from pyccolo.extra_builtins import (
    TRACING_ENABLED,
    make_guard_name,
    EMIT_EVENT,
)
from pyccolo.fast import make_composite_condition, make_test
from pyccolo.stmt_inserter import StatementInserter


def _handle_class_body(
    self,
    node: ast.ClassDef,
    orig_body: List[ast.AST],
) -> List[ast.AST]:
    classdef_copy = cast(
        ast.ClassDef,
        self.orig_to_copy_mapping[id(node)],
    )
    if self.global_guards_enabled:
        classdef_copy = self._global_nonlocal_stripper.visit(classdef_copy)
        class_guard = make_guard_name(classdef_copy)
        self.register_guard(class_guard)
    else:
        class_guard = None
    docstring = []
    if (
        len(orig_body) > 0
        and isinstance(orig_body[0], ast.Expr)
        and isinstance(orig_body[0].value, ast.Str)
    ):
        docstring = [orig_body.pop(0)]
    if len(orig_body) == 0:
        return docstring
    with fast.location_of(classdef_copy):
        if self.global_guards_enabled:
            ret = [
                fast.If(
                    test=make_composite_condition(
                        [
                            make_test(TRACING_ENABLED),
                            make_test(class_guard),
                            self.emit(
                                TraceEvent.before_class_body,
                                node,
                                ret=fast.NameConstant(True),
                            )
                            if self.handler_predicate_by_event[
                                TraceEvent.before_class_body
                            ](classdef_copy)
                            else None,
                        ]
                    ),
                    body=orig_body,
                    orelse=classdef_copy.body
                    if len(docstring) == 0
                    else classdef_copy.body[len(docstring) :],  # noqa: E203
                ),
            ]
        return docstring + ret


def generic_visit(self, node):
    if self.is_tracing_disabled_context(node):
        return node
    for name, field in ast.iter_fields(node):
        if isinstance(field, ast.AST):
            setattr(node, name, self.visit(field))
        elif isinstance(field, list):
            new_field = []
            future_imports = []
            if isinstance(node, ast.Module) and name == "body":
                node_copy = self.get_copy_node(node)
                if self.handler_predicate_by_event[
                    TraceEvent.init_module
                ](node_copy):
                    with fast.location_of(node):
                        new_field.extend(
                            fast.parse(
                                f'{EMIT_EVENT}("{TraceEvent.init_module.name}", '
                                + f"{id(node_copy)})"
                            ).body
                        )
            for inner_node in field:
                if isinstance(inner_node, ast.stmt):
                    if (
                        isinstance(inner_node, ast.ImportFrom)
                        and inner_node.module == "__future__"
                    ):
                        future_imports.append(inner_node)
                    else:
                        new_field.extend(self._handle_stmt(node, name, inner_node))
                elif isinstance(inner_node, ast.AST):
                    new_field.append(self.visit(inner_node))
                else:
                    new_field.append(inner_node)
            new_field = future_imports + new_field
            if name == "body":
                if isinstance(node, ast.Module):
                    new_field = self._handle_module_body(node, new_field)
                elif isinstance(node, (ast.For, ast.While)):
                    new_field = self._handle_loop_body(node, new_field)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    new_field = self._handle_function_body(node, new_field)
                elif isinstance(node, ast.ClassDef):
                    new_field = self._handle_class_body(node, new_field)
            setattr(node, name, new_field)
        else:
            continue
    return node


StatementInserter.generic_visit = generic_visit
StatementInserter._handle_class_body = _handle_class_body

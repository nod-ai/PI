import ast
import inspect
import warnings
from textwrap import dedent
from types import CodeType
from typing import Any


class PiIntFloatBool(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name) and node.func.id in {"int", "bool", "float"}:
            args = ", ".join(ast.unparse(a) for a in node.args)
            new_call = ast.parse(f"__import__('pi').pi_{node.func.id}({args})")
            node = new_call.body[0].value
        return node


def rewrite_ast_callback(frame):
    fun = frame.f_func
    try:
        src = dedent(inspect.getsource(fun))
    except Exception as e:
        warnings.warn(f"couldn't parse {fun.__name__} because {e}")
        return fun.__code__

    src = src.replace("torch.", "pi.")
    tree = ast.parse(src)

    assert isinstance(
        tree.body[0], ast.FunctionDef
    ), f"unexpected ast node {tree.body[0]}"
    old_tree_dump = ast.dump(tree)
    tree = PiIntFloatBool().visit(tree)
    tree = ast.fix_missing_locations(tree)
    tree = ast.increment_lineno(tree, fun.__code__.co_firstlineno - 1)

    if ast.dump(tree) == old_tree_dump:
        return fun.__code__

    module_code_o = compile(tree, fun.__code__.co_filename, "exec")
    f_code_o = next(
        c
        for c in module_code_o.co_consts
        if type(c) is CodeType and c.co_name == fun.__name__
    )
    # f_code_o = f_code_o.replace(
    #     co_argcount=fun.__code__.co_argcount,
    #     co_posonlyargcount=fun.__code__.co_posonlyargcount,
    #     co_kwonlyargcount=fun.__code__.co_kwonlyargcount,
    #     co_nlocals=fun.__code__.co_nlocals,
    #     co_stacksize=fun.__code__.co_stacksize,
    #     co_flags=fun.__code__.co_flags,
    #     co_firstlineno=fun.__code__.co_firstlineno,
    #     # co_code=fun.__code__.co_code,
    #     co_consts=fun.__code__.co_consts,
    #     co_names=fun.__code__.co_names,
    #     co_varnames=fun.__code__.co_varnames,
    #     co_freevars=fun.__code__.co_freevars,
    #     co_cellvars=fun.__code__.co_cellvars,
    #     co_filename=fun.__code__.co_filename,
    #     co_name=fun.__code__.co_name,
    #     co_qualname=fun.__code__.co_qualname,
    #     co_linetable=fun.__code__.co_linetable,
    #     co_exceptiontable=fun.__code__.co_exceptiontable,
    # )
    return f_code_o

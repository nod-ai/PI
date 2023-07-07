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
        return self.generic_visit(node)


def rewrite_ast_callback(frame):
    fun = frame.f_func
    try:
        src = dedent(inspect.getsource(fun))
        # print("prerewrite", src)
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
    if ast.dump(tree) == old_tree_dump:
        return fun.__code__
    # print("post-rewrite", ast.unparse(tree))
    tree = ast.fix_missing_locations(tree)
    tree = ast.increment_lineno(tree, fun.__code__.co_firstlineno - 1)

    module_code_o = compile(tree, fun.__code__.co_filename, "exec")
    f_code_o = next(
        c
        for c in module_code_o.co_consts
        if type(c) is CodeType and c.co_name == fun.__name__
    )
    return f_code_o

import ast
import inspect
import re
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from textwrap import dedent
from typing import List

from torch_mlir.dialects import torch
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.registry import (
    JitOperator,
    Registry,
)
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen import (
    TORCH_TYPE_TO_ODS_TYPE,
)
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen import (
    get_ods_type,
)
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.utils import TextEmitter

UNIQUE_OPS = []


def raw_emit_op(
    operator: JitOperator,
    emitter_td: TextEmitter,
    *,
    traits: List[str],
    has_folder: bool,
    has_canonicalizer: bool,
):
    UNIQUE_OPS.append(operator)


import torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen

torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen.raw_emit_op = (
    raw_emit_op
)

from torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen import emit_ops


def generate_exts():
    registry = Registry.load()

    with open("JitOperatorRegistry.txt", "w") as f:
        for op in registry.by_unique_key.values():
            print(op, file=f)

    with tempfile.NamedTemporaryFile() as f_td:
        emitter_td = TextEmitter(f_td)
        emit_ops(emitter_td, registry)


def get_clean_name(name):
    RESERVED_NAMES = {
        # since this is an aten op
        "pad": "pad__",
        "threshold": "threshold__",
    }

    if name in RESERVED_NAMES:
        return RESERVED_NAMES[name]
    return name


def map_defaults(s, method_sig):
    if s in {"0", "1", "-1"}:
        return s
    if s in {"True", "False"}:
        return s.lower()
    if s in "None":
        return "py::none()"
    if s in {"none", "constant"}:
        return '"{s}"'
    if s == "()":
        return None
    if s == "torch.contiguous_format":
        return "0"
    if match := re.match(r"\[(\d|,|\s)*\]", s):
        tuple = match.group()[1:-1]
        return f"std::vector<int>{{{tuple}}}"
    try:
        float(s)
        return s
    except:
        pass
    raise NotImplementedError(f"{s} for {method_sig}")


def generate_pybind_bindings_for_ops(cpp_ext_dir):
    registry = Registry.load()

    with open("JitOperatorRegistry.txt", "w") as f:
        for op in registry.by_unique_key.values():
            print(op, file=f)

    with tempfile.NamedTemporaryFile() as f_td:
        emitter_td = TextEmitter(f_td)
        emit_ops(emitter_td, registry)

    unimplemented_types = [
        "AnyTorchType",
        "AnyTorchOptionalListOfTorchIntType",
        "AnyTorchListOfOptionalTensorType",
        "Variadic",
    ]
    skips = ["prims::sqrt", "prim::Print", "aten::format", "aten::where.self"]
    skip_return_types = [
        "AnyTorchOptionalScalarType",
        "AnyTorchScalarType",
        "AnyTorchListType",
        "AnyTorchListOfTensorType",
        "AnyTorchOptionalTensorType",
    ]

    ops = []
    for operator in sorted(UNIQUE_OPS, key=lambda o: o.unqualified_name):
        op_name, cpp_class_name = operator.get_mlir_names()

        if operator.is_vararg:
            params = [("res", "Variadic<TorchType>", None)]
        else:
            params = [
                (
                    get_clean_name(arg["name"]),
                    get_ods_type(arg["type"]),
                    arg.get("default_debug"),
                )
                for arg in operator.arguments
            ]

        if operator.is_varret:
            returns = [("res", "Variadic<TorchType>", None)]
        else:
            returns = [
                (ret["name"], get_ods_type(ret["type"]), None)
                for ret in operator.returns
            ]

        unimplemented = False
        for u in unimplemented_types:
            if u in [t for n, t, d in params + returns]:
                unimplemented = True
                warnings.warn(
                    f"not implemented type: {u} for {op_name} {cpp_class_name}"
                )
                break

        for u in skip_return_types:
            if u in [t for n, t, d in returns]:
                warnings.warn(f"not implemented return type {u} for {cpp_class_name}")
                unimplemented = True
                break

        if unimplemented:
            continue

        dup = False
        for u in skips:
            if u in operator.unique_key:
                dup = True
                break
        if dup:
            continue

        init_lines = inspect.getsource(getattr(torch, cpp_class_name).__init__)
        if "InferTypeOpInterface" not in init_lines and returns:
            if len(returns) > 1:
                warnings.warn(f"not implemented multiple returns: {cpp_class_name}")
                continue

        if returns and returns[0][1] == "Variadic<TorchType>":
            warnings.warn(f"not implemented multiple returns: Variadic<TorchType>")
            continue

        ops.append(
            (
                op_name,
                operator.unqualified_name,
                params,
                returns,
                cpp_class_name,
                operator.unique_key,
            )
        )

    with open(f"{cpp_ext_dir}/TorchOps.impls.cpp", "w") as impls_file, open(
        f"{cpp_ext_dir}/TorchOps.inc.h", "w"
    ) as impls_h_file:
        impls_h_emitter = TextEmitter(impls_h_file)
        impls_emitter = TextEmitter(impls_file)
        impls_emitter._INDENT = "    "
        impls_td = lambda *args: impls_emitter.print(*args)
        impls_h_td = lambda *args: impls_h_emitter.print(*args)
        for (
            torch_dialect_op_name,
            unqualified_name,
            params,
            returns,
            cpp_class_name,
            schema,
        ) in ops:
            param_str = ", ".join(
                [
                    f"const Py{typ.replace('Type', 'Value')} &{name}"
                    for name, typ, default_debug in params
                ]
            )

            init_lines = inspect.getsource(getattr(torch, cpp_class_name).__init__)
            if "InferTypeOpInterface" not in init_lines and returns:
                _, typ, _d = returns[0]
                if typ == "AnyTorchTensorType":
                    params = [
                        (
                            "PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())",
                            "",
                            None,
                        )
                    ] + params
                elif typ in {
                    "AnyTorchListOfTorchStringType",
                    "AnyTorchListOfTorchIntType",
                }:
                    params = [
                        (
                            f"Py{typ}(DefaultingPyMlirContext::resolve())",
                            "",
                            None,
                        )
                    ] + params
                else:
                    raise NotImplementedError(f"{typ} return type for {cpp_class_name}")

            impl = dedent(
                f"""
                    // {schema}
                    py::object {unqualified_name}({param_str}) {{
                      return PyGlobals::get().lookupOperationClass("torch.{torch_dialect_op_name}").value()({', '.join([name for name, _typ, _d in params])});
                    }}
                """
            )
            impls_td(impl)
            header = dedent(
                f"""
                    // {schema}
                    py::object {unqualified_name}({param_str});
                """
            )
            impls_h_td(header)

    with open(f"{cpp_ext_dir}/TorchOps.pybinds.cpp", "w") as impls_file:
        impls_emitter = TextEmitter(impls_file)
        impls_emitter._INDENT = "    "
        impls_td = lambda *args: impls_emitter.print(*args)
        for (
            _torch_dialect_op_name,
            unqualified_name,
            params,
            returns,
            cpp_class_name,
            schema,
        ) in ops:
            labels = []
            param_str = []
            tramp_str = []
            for n, t, d in params:
                t = t.replace("Type", "Value")
                if "TorchOptional" in t:
                    param_str.append(f"const Py{t} &{n}")
                    tramp_str.append(f"{n}")
                    labels.append(f'"{n}"_a = py::none()')
                elif d is not None:
                    param_str.append(f"const Py{t} &{n}")
                    tramp_str.append(n)
                    labels.append(f'"{n}"_a = {map_defaults(d, schema)}')
                else:
                    param_str.append(f"const Py{t} &{n}")
                    tramp_str.append(n)
                    labels.append(f'"{n}"_a')

            try:
                last_opt_idx = next(
                    i for i, l in reversed(list(enumerate(labels))) if "py::none()" in l
                )
                if "py::kw_only()" not in labels and last_opt_idx < len(labels) - 1:
                    labels.insert(last_opt_idx + 1, "py::kw_only()")
            except StopIteration:
                pass

            labels_str = ", ".join(labels)
            tramp_str = ", ".join(tramp_str)
            param_str = ", ".join(param_str)

            impl = dedent(
                f"""
                    // {schema}
                    m.def("{unqualified_name}", []({param_str}) {{ return {unqualified_name}({tramp_str}); }}, {labels_str});
                """
            )

            impls_td(impl)

    return ops


class TensorMethodVisitor(ast.NodeVisitor):
    skip = [
        "run_on_actual_value",
        "__subclasscheck__",
        "__instancecheck__",
        "_is_pi_tensor",
        "__class__",
        "type",
        "value",
        "__init__",
        "where",
    ]

    def __init__(self, ops):
        self.ops = ops
        self.binds_file = open(f"{cpp_ext_dir}/TorchTensor.pybinds.cpp", "w")
        self.binds_emitter = TextEmitter(self.binds_file)
        self.binds_emitter._INDENT = "    "
        self.binds_td = lambda *args: self.binds_emitter.print(*args)

        self.binds_tramps_file = open(
            f"{cpp_ext_dir}/TorchTensor.pybinds_tramps.cpp", "w"
        )
        self.binds_tramps_emitter = TextEmitter(self.binds_tramps_file)
        self.binds_tramps_emitter._INDENT = "    "
        self.binds_tramps_td = lambda *args: self.binds_tramps_emitter.print(*args)
        self.visited = set()

    def emit_not_implemented(self, method_sig, op_name):
        impl = dedent(
            f"""
                // {method_sig}
                c.def("{op_name}", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) {{ throw NotImplementedError("{op_name} with signature {method_sig}"); }});
            """
        )
        self.binds_td(impl)

    # TODO(max): default args
    def visit_FunctionDef(self, node: ast.FunctionDef):
        method_sig = (
            ast.unparse(node)
            .replace("def", "")
            .replace("\n", "")
            .replace("...", "")
            .replace(":", "")
            .strip()
        )

        op_name = node.name
        if op_name in self.skip:
            if method_sig not in self.visited:
                self.visited.add(method_sig)
            return
        if op_name not in self.ops and op_name.replace("__", "") not in self.ops:
            if method_sig not in self.visited:
                self.visited.add(method_sig)
                self.emit_not_implemented(method_sig, op_name)
            return

        posonlyargs = {k.arg for k in node.args.posonlyargs}
        assert len(posonlyargs) == 0
        kwonlyargs = {
            k.arg: node.args.kw_defaults[i] for i, k in enumerate(node.args.kwonlyargs)
        }

        defaults = node.args.defaults
        arg_names = [a.arg for a in node.args.args if a not in kwonlyargs]
        arg_types = [a.annotation for a in node.args.args if a not in kwonlyargs]
        if node.args.vararg:
            arg_names.append(f"*{node.args.vararg.arg}")
        if node.args.kwonlyargs:
            for kwarg in node.args.kwonlyargs:
                arg_names.append(kwarg.arg)
                arg_types.append(kwarg.annotation)

        def try_to_find_op(op_name):

            for i, (params, returns, cpp_class_name, schema) in enumerate(
                self.ops.get(op_name, [])
            ):
                params_dict = dict([(n, t) for n, t, _d in params])
                if (
                    set(arg_names) == set(params_dict.keys())
                    and "self" in params_dict
                    and params_dict["self"] == "AnyTorchTensorType"
                ):
                    return params, returns, cpp_class_name, schema, op_name

        op = try_to_find_op(op_name)
        if op is None and op_name.startswith("__") and op_name.endswith("__"):
            op = try_to_find_op(op_name.replace("__", ""))

        if op is None:
            warnings.warn(f"found no matching overload for {op_name=}")
            return

        params, returns, cpp_class_name, schema, overload_op_name = op
        if schema in self.visited:
            return

        self.visited.add(schema)

        params_dict = dict([(n, t) for n, t, _d in params])
        if arg_names != [p[0] for p in params] or overload_op_name != op_name:
            tramp_param_str = ", ".join(
                [
                    f"const mlir::torch::Py{params_dict[name].replace('Type', 'Value')} &{name}"
                    for name in arg_names
                ]
            )
            impl = dedent(
                f"""
                        // {method_sig}
                        // {schema}
                        py::object {op_name}({tramp_param_str}) {{
                          return mlir::torch::{overload_op_name}({', '.join([p[0] for p in params])});
                        }}
                    """
            )
            self.binds_tramps_td(impl)

        assert params[0][0] == "self", f"{params} for {method_sig=}"
        labels = []
        param_str = []
        tramp_str = []
        for i, (n, t, d) in enumerate(params):
            t = t.replace("Type", "Value")
            if "TorchOptional" in t:
                labels.append(f'"{n}"_a = py::none()')
                param_str.append(f"const Py{t} &{n}")
                tramp_str.append(f"{n}")
                continue
            elif n in kwonlyargs:
                if "py::kw_only()" not in labels:
                    labels.append("py::kw_only()")
                assert (
                    d is not None
                ), f"default_debug is None for {n=} {t=} {method_sig=}"
                defau = map_defaults(d, method_sig)
                if defau is not None:
                    labels.append(f'"{n}"_a = {defau}')
                    param_str.append(f"const Py{t} &{n}")
                    tramp_str.append(f"{n}")
                    continue
            elif len(params) - len(defaults) <= i < len(params):
                assert (
                    d is not None
                ), f"default_debug is None for {n=} {t=} {method_sig=}"
                defau = map_defaults(d, method_sig)
                if defau is not None:
                    labels.append(f'"{n}"_a = {defau}')
                    param_str.append(f"const Py{t} &{n}")
                    tramp_str.append(f"{n}")
                    continue

            param_str.append(f"const Py{t} &{n}")
            tramp_str.append(n)
            labels.append(f'"{n}"_a')

        try:
            last_opt_idx = next(
                i for i, l in reversed(list(enumerate(labels))) if "py::none()" in l
            )
            if "py::kw_only()" not in labels and last_opt_idx < len(labels) - 1:
                labels.insert(last_opt_idx + 1, "py::kw_only()")
        except StopIteration:
            pass

        # apparently you can't label the self arg for a class method???
        labels_str = ", ".join(labels[1:])
        if labels_str:
            labels_str = f", {labels_str}"
        tramp_str = ", ".join(tramp_str)
        param_str = ", ".join(param_str)

        impl = dedent(
            f"""
                    // {method_sig}
                    // {schema}
                    c.def("{op_name}", []({param_str}) {{ return {op_name}({tramp_str}); }}{labels_str});
                """
        )

        self.binds_td(impl)


class FindTensorClass(ast.NodeVisitor):
    def __init__(self, ops):
        self.ops = ops
        self.tens_visitor = TensorMethodVisitor(self.ops)

    def visit_ClassDef(self, node: ast.ClassDef):
        if node.name == "_TensorBase":
            self.tens_visitor.visit(node)


def generate_tensor_bindings(ops):
    with open("torch/_C/__init__.pyi") as f:
        tree = ast.parse(f.read())

    ops_dict = defaultdict(list)
    for (
        _torch_dialect_op_name,
        unqualified_name,
        params,
        returns,
        cpp_class_name,
        schema,
    ) in ops:
        ops_dict[unqualified_name].append((params, returns, cpp_class_name, schema))
    f = FindTensorClass(ops_dict)
    f.visit(tree)
    f.tens_visitor.binds_file.close()
    f.tens_visitor.binds_tramps_file.close()


if __name__ == "__main__":
    cpp_ext_dir = str((Path(__file__).parent.parent.parent / "cpp_ext").absolute())
    ops = generate_pybind_bindings_for_ops(cpp_ext_dir)
    generate_tensor_bindings(ops)

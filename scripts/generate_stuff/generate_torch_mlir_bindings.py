import ast
import inspect
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
        "pad": "pad__"
    }

    if name in RESERVED_NAMES:
        return RESERVED_NAMES[name]
    return name


def generate_pybind_bindings_for_ops(cpp_ext_dir):
    registry = Registry.load()

    with open("JitOperatorRegistry.txt", "w") as f:
        for op in registry.by_unique_key.values():
            print(op, file=f)

    with tempfile.NamedTemporaryFile() as f_td:
        emitter_td = TextEmitter(f_td)
        emit_ops(emitter_td, registry)

    unimplemented_types = [
        "AnyTorchOptionalScalarType",
        "AnyTorchType",
        "AnyTorchOptionalListOfTorchIntType",
        "AnyTorchListOfOptionalTensorType",
        "Variadic",
    ]
    skips = ["prims::sqrt", "prim::Print", "aten::format"]
    skip_return_types = [
        "AnyTorchScalarType",
        "AnyTorchListType",
        "AnyTorchListOfTensorType",
        "AnyTorchOptionalTensorType",
    ]

    ops = []
    for operator in sorted(UNIQUE_OPS, key=lambda o: o.unqualified_name):
        op_name, cpp_class_name = operator.get_mlir_names()

        if operator.is_vararg:
            params = [("res", "Variadic<TorchType>")]
        else:
            params = [
                (get_clean_name(arg["name"]), get_ods_type(arg["type"]))
                for arg in operator.arguments
            ]

        if operator.is_varret:
            returns = [("res", "Variadic<TorchType>")]
        else:
            returns = [
                (ret["name"], get_ods_type(ret["type"])) for ret in operator.returns
            ]

        unimplemented = False
        for u in unimplemented_types:
            if u in [t for n, t in params + returns]:
                unimplemented = True
                warnings.warn(
                    f"not implemented type: {u} for {op_name} {cpp_class_name}"
                )
                break

        for u in skip_return_types:
            if u in [t for n, t in returns]:
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
                    for name, typ in params
                ]
            )

            init_lines = inspect.getsource(getattr(torch, cpp_class_name).__init__)
            if "InferTypeOpInterface" not in init_lines and returns:
                _, typ = returns[0]
                if typ == "AnyTorchTensorType":
                    params = [
                        (
                            "PyAnyTorchTensorType::getWithLeastStaticInformation(DefaultingPyMlirContext::resolve())",
                            "",
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
                        )
                    ] + params
                else:
                    raise NotImplementedError(f"{typ} return type for {cpp_class_name}")

            impl = dedent(
                f"""
                    // {schema}
                    py::object {unqualified_name}({param_str}) {{
                      return PyGlobals::get().lookupOperationClass("torch.{torch_dialect_op_name}").value()({', '.join([name for name, _typ in params])});
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
            if any("AnyTorchOptional" in t for n, t in params):
                param_str = ", ".join(
                    [
                        f"const Py{typ.replace('Type', 'Value').replace('AnyTorchOptional', 'DefaultingTorchOptional')} &{name}"
                        for name, typ in params
                    ]
                )
                tramp_str = ", ".join(
                    [f"{n}.get()" if "AnyTorchOptional" in t else n for n, t in params]
                )
                labels_str = ", ".join(
                    [
                        f'"{n}"_a = py::none()'
                        if "AnyTorchOptional" in t
                        else f'"{n}"_a'
                        for n, t in params
                    ]
                )

                impl = dedent(
                    f"""
                        // {schema}
                        m.def("{unqualified_name}", []({param_str}) {{ return {unqualified_name}({tramp_str}); }}, {labels_str});
                    """
                )
            else:
                param_str = ", ".join(
                    [
                        f"const Py{typ.replace('Type', 'Value')} &"
                        for name, typ in params
                    ]
                )
                labels_str = ", ".join([f'"{n}"_a' for n, t in params])
                if labels_str:
                    labels_str = f", {labels_str}"
                impl = dedent(
                    f"""
                            // {schema}
                            m.def("{unqualified_name}", py::overload_cast<{param_str}>(&{unqualified_name}){labels_str});
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
        kwonlyargs = {k.arg for k in node.args.kwonlyargs}
        arg_names = [a.arg for a in node.args.args if a not in kwonlyargs]
        arg_types = [a.annotation for a in node.args.args if a not in kwonlyargs]
        if node.args.vararg:
            arg_names.append(f"*{node.args.vararg.arg}")
        # TODO(max): handle kwonlyargs and defaults for them
        if node.args.kwonlyargs:
            for kwarg in node.args.kwonlyargs:
                arg_names.append(kwarg.arg)
                arg_types.append(kwarg.annotation)

        def try_to_find_op(op_name):

            for i, (params, returns, cpp_class_name, schema) in enumerate(
                self.ops.get(op_name, [])
            ):
                params_dict = dict(params)
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

        params_dict = dict(params)
        if arg_names != [p[0] for p in params]:
            # different orders...
            tramp_param_str = ", ".join(
                [
                    f"const mlir::torch::Py{params_dict[name].replace('Type', 'Value')} &{name}"
                    for name in arg_names
                ]
            )
            impl = dedent(
                f"""
                        // {schema}
                        py::object {op_name}({tramp_param_str}) {{
                          return mlir::torch::{overload_op_name}({', '.join([p[0] for p in params])});
                        }}
                    """
            )
            self.binds_tramps_td(impl)

        if any("AnyTorchOptional" in t for n, t in params):
            param_str = ", ".join(
                [
                    f"const Py{typ.replace('Type', 'Value').replace('AnyTorchOptional', 'DefaultingTorchOptional')} &{name}"
                    for name, typ in params
                ]
            )
            tramp_str = ", ".join(
                [f"{n}.get()" if "AnyTorchOptional" in t else n for n, t in params]
            )
            assert params[0][0] == "self"
            # apparently you can't label the self arg for a class method???
            labels = [
                f'"{n}"_a = py::none()' if "AnyTorchOptional" in t else f'"{n}"_a'
                for n, t in params
            ]
            last_opt_idx = next(
                i for i, l in reversed(list(enumerate(labels))) if "py::none()" in l
            )
            if last_opt_idx < len(labels) - 1:
                labels.insert(last_opt_idx + 1, "py::kw_only()")

            labels_str = ", ".join(labels[1:])

            impl = dedent(
                f"""
                        // {schema}
                        c.def("{op_name}", []({param_str}) {{ return {op_name}({tramp_str}); }}, {labels_str});
                    """
            )

        else:
            param_str = ", ".join(
                [
                    f"const Py{params_dict[name].replace('Type', 'Value')}&"
                    for name in arg_names
                ]
            )
            labels = [f'"{n}"_a' for n, t in params]
            labels_str = ", ".join(labels[1:])
            if labels_str:
                labels_str = f", {labels_str}"

            if kwonlyargs:
                warnings.warn(f"{op_name=} has kwonly args: {kwonlyargs=}")
            impl = dedent(
                f"""
                    // {method_sig}
                    // {schema}
                    c.def("{op_name}", py::overload_cast<{param_str}>(&{overload_op_name}){labels_str});
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

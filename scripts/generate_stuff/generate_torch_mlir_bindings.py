import inspect
import tempfile
import warnings
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
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.utils import TextEmitter


from torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen import (
    get_ods_type,
)

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


def generate_pybind_bindings(cpp_ext_dir):
    registry = Registry.load()
    with tempfile.NamedTemporaryFile() as f_td:
        emitter_td = TextEmitter(f_td)
        emit_ops(emitter_td, registry)

    unimplemented_types = [
        "AnyTorchListValue",
        "AnyTorchListOfTensorValue",
        "AnyTorchOptionalScalarValue",
        "AnyTorchOptionalTensorValue",
        "AnyTorchValue",
        "AnyTorchScalarValue",
        "AnyTorchOptionalListOfTorchIntValue",
        "AnyTorchListOfOptionalTensorValue",
        #
        "AnyTorchListType",
        "AnyTorchListOfTensorType",
        "AnyTorchOptionalScalarType",
        "AnyTorchOptionalTensorType",
        "AnyTorchType",
        "AnyTorchScalarType",
        "AnyTorchOptionalListOfTorchIntType",
        "AnyTorchListOfOptionalTensorType",
        "Variadic",
    ]
    skips = ["prims::sqrt", "prim::Print", "aten::format"]

    ops = []
    for operator in sorted(UNIQUE_OPS, key=lambda o: o.unqualified_name):
        op_name, cpp_class_name = operator.get_mlir_names()

        if operator.is_vararg:
            params = [("res", "Variadic<TorchType>")]
        else:
            params = [
                (arg["name"], get_ods_type(arg["type"])) for arg in operator.arguments
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
                warnings.warn(f"not implemented type: {u}")
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
                operator.unqualified_name,
                params,
                returns,
                cpp_class_name,
                operator.unique_key,
            )
        )

    with open(f"{cpp_ext_dir}/TorchOps.impls.cpp", "w") as stubs_td:
        stubs_emitter_td = TextEmitter(stubs_td)
        stubs_emitter_td._INDENT = "    "
        stub_td = lambda *args: stubs_emitter_td.print(*args)
        for (
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
                elif typ == "AnyTorchListOfTorchStringType":
                    params = [
                        (
                            "PyAnyTorchListOfTorchStringType(DefaultingPyMlirContext::resolve())",
                            "",
                        )
                    ] + params
                elif typ == "AnyTorchListOfTorchIntType":
                    params = [
                        (
                            "PyAnyTorchListOfTorchIntType(DefaultingPyMlirContext::resolve())",
                            "",
                        )
                    ] + params
                else:
                    raise NotImplementedError(typ)

            impl = dedent(
                f"""
                    // {schema}
                    py::object {unqualified_name}({param_str}) {{
                      auto torch = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects")).attr("torch");
                      return torch.attr("{cpp_class_name}")({', '.join([name for name, _typ in params])});
                    }}
                """
            )
            stub_td(impl)

    with open(f"{cpp_ext_dir}/TorchOps.pybinds.cpp", "w") as stubs_td:
        stubs_emitter_td = TextEmitter(stubs_td)
        stubs_emitter_td._INDENT = "    "
        stub_td = lambda *args: stubs_emitter_td.print(*args)
        for (
            unqualified_name,
            params,
            returns,
            cpp_class_name,
            schema,
        ) in ops:
            param_str = ", ".join(
                [f"const Py{typ.replace('Type', 'Value')} &" for name, typ in params]
            )

            impl = dedent(
                f"""
                    // {schema}
                    m.def("{unqualified_name}", py::overload_cast<{param_str}>(&{unqualified_name}));
                """
            )
            stub_td(impl)


if __name__ == "__main__":
    cpp_ext_dir = str((Path(__file__).parent.parent.parent / "cpp_ext").absolute())
    generate_pybind_bindings(cpp_ext_dir)

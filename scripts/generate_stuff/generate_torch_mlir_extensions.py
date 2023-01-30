import argparse
import inspect
import re
import tempfile
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import List, Any, Union
import black
import torch_ods_type_predicates

from torch_mlir.dialects.torch.importer.jit_ir.build_tools.registry import (
    JitOperator,
    Registry,
    _get_default_value,
)
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.utils import TextEmitter
from torch_mlir.dialects import torch as torch_dialect


ALL_OPS = []
DEBUG = False
BLACKLIST = {
    "QuantizedLinearOp",
    "PrimsSqrtOp",
    "AtenAddTOp",
    "ScalarImplicit",
}
OP_NAME_SUBS = {"linalg_vector_norm": "vector_norm"}
UNIQUE_OPS = []


def py_reserved_keywords(k):
    subs = {
        "from": "from_",
        "self": "self_",
        "list": "list_",
    }
    return subs.get(k, k)


from torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen import (
    get_ods_type,
)


@dataclass
class PyTorchType:
    name: str
    optional_el: bool
    list: bool
    optional: bool
    original: str

    def ods_type(self):
        return get_ods_type(self.original)


def get_full_pytorch_type(arg):
    pytorch_type = arg["type"]
    if pytorch_type == "Tensor?[]":
        return PyTorchType("Tensor", True, False, False, pytorch_type)
    assert "?[]" not in pytorch_type
    full_type = re.match(
        r"(?P<name>\w+)(?P<list>\[\])?(?P<optional>\?)?", pytorch_type
    ).groupdict()
    return PyTorchType(
        full_type["name"],
        False,
        full_type["list"] is not None,
        full_type["optional"] is not None,
        pytorch_type,
    )


def pretty_print_type_hint(hint: Any) -> str:
    if isinstance(hint, type) and not str(type(hint)) == "<class 'typing.GenericMeta'>":
        hint_str = hint.__name__
    else:
        hint_str = str(hint)
    return (
        hint_str.replace("typing.", "")
        .replace("NoneType", "None")
        .replace("torch_ods_type_predicates.", "")
    )


def _pytype_to_fn_pytype_common(arg) -> str:
    full_pytorch_type = get_full_pytorch_type(arg)
    ods_type_annot = getattr(torch_ods_type_predicates, full_pytorch_type.ods_type())
    pretty_type_annot = pretty_print_type_hint(ods_type_annot)
    if match := re.findall(r"(dtype|layout|memory_format)", arg["name"]):
        match = match[0]
        pretty_type_annot = pretty_type_annot.replace("int", f"int, pi_{match}")
    pretty_type_annot = (
        pretty_type_annot.replace("~T,", "")
        .replace("int", "builtins.int")
        .replace("str", "builtins.str")
        .replace("bool", "builtins.bool")
        .replace("float", "builtins.float")
    )
    return pretty_type_annot


def get_wrapper_function_signature(op):
    def parameter_decl_builder(arg) -> str:
        pytype = _pytype_to_fn_pytype_common(arg)
        default = _get_default_value(arg)
        parameter_name = py_reserved_keywords(arg["name"])
        if parameter_name == "out":
            assert "alias_info" in arg
            assert default == ""
            default = "= None"
        return f"{parameter_name}: {pytype}{default}"

    def ret_decl_builder(arg) -> str:
        ret_type = _pytype_to_fn_pytype_common(arg)
        return ret_type

    def_name = OP_NAME_SUBS.get(op.unqualified_name, op.unqualified_name)
    parameter_decls = list(map(parameter_decl_builder, op.arguments))
    ret_decls = list(map(ret_decl_builder, op.returns))
    parameters = ", ".join(parameter_decls)
    result = ", ".join(ret_decls)
    if len(ret_decls) >= 2:
        result = f"Tuple[{result}]"

    if result == "":
        result = None

    return f"def {def_name}({parameters}) -> {result}:"


def emit_torch_wrappers(
    operators,
    all,
    stubs_emitter_td: TextEmitter,
):
    counts = Counter(all)
    stub_td = lambda *args: stubs_emitter_td.print(*args)
    for operator in operators:
        if operator.overload_name:
            stub_td(f"# overload {operator.overload_name}")
        if counts[operator.unqualified_name] > 1:
            stub_td("@register_dispatch")

        _op_name, cpp_class_name = operator.get_mlir_names()
        stub_td(get_wrapper_function_signature(operator))
        with stubs_emitter_td.indent():
            stub_td("assert check_argument_types()")
            args = {}
            for arg in operator.arguments:
                arg_name = py_reserved_keywords(arg["name"])
                arg_type = _pytype_to_fn_pytype_common(arg)
                pytorch_type = arg["type"]
                pytype = arg["pytype"]
                args[arg_name] = arg_type

                if pytorch_type.startswith("Scalar"):
                    stub_td(
                        f"if isinstance({arg_name}, (builtins.int, builtins.float)):"
                    )
                    with stubs_emitter_td.indent():
                        stub_td(
                            f"{arg_name} = torch_dialect.ConstantNumberOp({arg_name}).result"
                        )

                if "device" in arg_name:
                    stub_td(f"if isinstance({arg_name}, builtins.str):")
                    with stubs_emitter_td.indent():
                        stub_td(
                            f"{arg_name} = torch_dialect.ConstantStrOp({arg_name}).result"
                        )

                if match := re.findall(r"(dtype|layout|memory_format)", arg_name):
                    match = match[0]
                    stub_td(f"if isinstance({arg_name}, pi_{match}):")
                    with stubs_emitter_td.indent():
                        stub_td(f"{arg_name} = {arg_name}.value")

                if match := re.findall(r"^(bool|int|float|str)(\[\])?", pytorch_type):
                    match = match[0]
                    el_type, is_list = match
                    if is_list:
                        # torch_dialect.PrimListConstructOp(TorchListOfTorchBool(), [t, tt]).result
                        stub_td(
                            f"if isinstance({arg_name}, (builtins.list, builtins.tuple)) and builtins.len({arg_name}):"
                        )
                        with stubs_emitter_td.indent():
                            stub_td(f"{arg_name} = builtins.list({arg_name})")
                            stub_td(f"for i, a in enumerate({arg_name}):")
                            with stubs_emitter_td.indent():
                                stub_td(
                                    f"if not isinstance(a, builtins.{el_type}): assert isinstance(a, Torch_Value), f'wrong type: {{a}}; should be {el_type}'"
                                )
                                stub_td(f"else:")
                                with stubs_emitter_td.indent():
                                    stub_td(
                                        f"{arg_name}[i] = torch_dialect.Constant{el_type.capitalize()}Op(a).result"
                                    )

                            if el_type == "str":
                                el_type = "string"

                            stub_td(
                                f"ls_type = Torch_List.of(Torch_{el_type.capitalize()}Type())"
                            )
                            stub_td(
                                f"{arg_name} = torch_dialect.PrimListConstructOp(ls_type, {arg_name}).result"
                            )
                    else:
                        stub_td(f"if isinstance({arg_name}, builtins.{el_type}):")
                        with stubs_emitter_td.indent():
                            stub_td(
                                f"{arg_name} = torch_dialect.Constant{el_type.capitalize()}Op({arg_name}).result"
                            )

                if pytype.startswith("Optional"):
                    stub_td(f"if {arg_name} is None:")
                    with stubs_emitter_td.indent():
                        stub_td(f"{arg_name} = torch_dialect.ConstantNoneOp().result")

                if pytorch_type == "Tensor[]":
                    stub_td(
                        f"if isinstance({arg_name}, (builtins.list, builtins.tuple)) and builtins.len({arg_name}):"
                    )
                    with stubs_emitter_td.indent():
                        stub_td(
                            f"assert builtins.all([isinstance(a, Tensor) for a in {arg_name}])"
                        )
                        stub_td(f"ls_type = Torch_List.of(Torch_NonValueTensorType())")
                        stub_td(
                            f"{arg_name} = torch_dialect.PrimListConstructOp(ls_type, {arg_name}).result"
                        )

                if pytorch_type == "Tensor?[]":
                    stub_td(f"{arg_name} = builtins.list({arg_name})")
                    stub_td(f"for i, a in enumerate({arg_name}):")
                    with stubs_emitter_td.indent():
                        stub_td(f"if a is not None: assert isinstance(a, Tensor)")
                        stub_td(f"else:")
                        with stubs_emitter_td.indent():
                            stub_td(
                                f"{arg_name}[i] = torch_dialect.ConstantNoneOp().result"
                            )

                    stub_td(
                        f"{arg_name} = torch_dialect.PrimListConstructOp(TorchListOfOptionalTensorType(), {arg_name}).result"
                    )

            init_lines = inspect.getsource(
                getattr(torch_dialect, cpp_class_name).__init__
            )
            if not "InferTypeOpInterface" in init_lines and operator.returns:
                res_type_names = []
                for i, r in enumerate(operator.returns):
                    if r["pytype"] == "List[t]":
                        res_type = "Torch_List.of(Torch_AnyType())"
                    elif r["pytype"] == "List[str]":
                        res_type = "Torch_List.of(Torch_StringType())"
                    elif r["pytype"] == "List[int]":
                        res_type = "Torch_List.of(Torch_IntType())"
                    elif r["pytype"] in {"Any", "t"}:
                        res_type = "Torch_AnyType()"
                    else:
                        res_type = _pytype_to_fn_pytype_common(r)
                        if res_type == "Tensor":
                            res_type = "Torch_NonValueTensorType()"
                        elif res_type == "TorchNumber":
                            res_type = "Torch_NumberType()"
                        elif "Union" in res_type or "Sequence" in res_type:
                            raise Exception(res_type, r["pytype"])
                    res_type_name = f"result{i}_type"
                    res_type_names.append(res_type_name)
                    stub_td(f"{res_type_name} = {res_type}")
                call_str = f'torch_dialect.{cpp_class_name}({", ".join(res_type_names)}, {", ".join([f"{k}" for k, _v in args.items()])})'
            else:
                call_str = f'torch_dialect.{cpp_class_name}({", ".join([f"{k}" for k, _v in args.items()])})'

            if len(operator.returns) == 0:
                ret = call_str
            elif len(operator.returns) == 1:
                if operator.returns[0]["pytype"] == "Tensor":
                    ret = f"return Tensor({call_str})"
                else:
                    ret = f"return Torch_Value({call_str}.result)"
            else:
                stub_td(f"op_results = get_op_results_or_values({call_str})")
                ret = f"return tuple([Tensor(o) for o in op_results])"
            stub_td(f"{ret}")
            stub_td("\n")


def generate_torch_wrappers(torch_ops_ext_dir: Path):
    torch_wrappers_fp = torch_ops_ext_dir / "_torch_wrappers.py"
    with open(torch_wrappers_fp, "w") as stubs_td:
        stubs_emitter_td = TextEmitter(stubs_td)
        stubs_emitter_td._INDENT = "    "
        stubs_emitter_td.print(
            dedent(
                f"""\
            import builtins
            from typing import List, Optional, Any, Dict, Tuple, Union, Sequence

            from ._tensor import Tensor
            from .types_ import (
                dtype as pi_dtype,
                layout as pi_layout,
                memory_format as pi_memory_format,
                Torch_Value,
                Torch_List,
                Torch_Dict,
            )
            # noinspection PyUnresolvedReferences
            from ._pi_mlir import (
                TorchListOfTorchBoolType,
                TorchListOfTorchFloatType,
                TorchListOfTorchIntType,
                TorchListOfTorchStringType,
                TorchListOfNonValueTensorType,
                
                TorchListOfOptionalTensorType,
                
                TorchOptionalBoolType,
                TorchOptionalFloatType,
                TorchOptionalIntType,
                TorchOptionalStringType,
                TorchOptionalDeviceType,
                TorchOptionalGeneratorType,
                TorchOptionalNonValueTensorType,
                
                Torch_AnyType,
                Torch_BoolType,
                Torch_DeviceType,
                Torch_FloatType,
                Torch_IntType,
                Torch_NumberType,
                Torch_StringType,
                Torch_NonValueTensorType,
                Torch_GeneratorType
            )
            from .dispatcher import register_dispatch
            from torch_mlir.dialects import torch as torch_dialect
            from torch_mlir.dialects._ods_common import (
                get_op_results_or_values,
            )
            from typeguard import check_argument_types
            
            TorchNumber = Union[Torch_Value[Torch_IntType], Torch_Value[Torch_FloatType], int, float]
            """
            )
        )

        TORCH_WRAPPERS = sorted(UNIQUE_OPS, key=lambda t: t.unqualified_name)
        TORCH_WRAPPERS = [
            t for t in TORCH_WRAPPERS if t.unqualified_name not in BLACKLIST
        ]
        ALL = [t for t in ALL_OPS if t not in BLACKLIST]
        emit_torch_wrappers(TORCH_WRAPPERS, ALL, stubs_emitter_td)
        stubs_emitter_td.print("\n\n")
        all = [
            f'"{OP_NAME_SUBS.get(t.unqualified_name, t.unqualified_name)}"'
            for t in TORCH_WRAPPERS
        ]
        stubs_emitter_td.print(f"__all__ = [{', '.join(sorted(set(all)))}]")

    black.format_file_in_place(
        torch_wrappers_fp, fast=False, mode=black.Mode(), write_back=black.WriteBack.YES
    )


def raw_emit_op(
    operator: JitOperator,
    emitter_td: TextEmitter,
    *,
    traits: List[str],
    has_folder: bool,
    has_canonicalizer: bool,
):
    op_name, cpp_class_name = operator.get_mlir_names()
    if cpp_class_name in BLACKLIST:
        return
    if operator.unqualified_name == "torch.quantized.linear":
        return

    if operator.is_vararg:
        warnings.warn(f"{cpp_class_name} is vararg; skipping")
        return

    if operator.is_varret:
        warnings.warn(f"{cpp_class_name} is vararg; skipping")
        return

    UNIQUE_OPS.append(operator)
    ALL_OPS.append(operator.unqualified_name)


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


def main(args):
    generate_exts()
    generate_torch_wrappers(args.wrapper_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PI wrappers")
    parser.add_argument(
        "--wrapper_dir",
        help="directory where PyTorch wrappers should be placed",
        default="../../pi",
        type=Path,
    )
    args = parser.parse_args()
    main(args)

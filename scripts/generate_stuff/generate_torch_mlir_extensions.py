import warnings
from textwrap import dedent
from typing import List, Callable

from torch_mlir.dialects.torch.importer.jit_ir.build_tools.registry import (
    JitOperator,
    Registry,
    _pytype_to_decomposition_fn_pytype,
    _get_default_value,
    _rename_python_keyword_parameter_name,
)
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.utils import TextEmitter
from torchgen.api.python import signature_from_schema, FunctionSchema


# from scripts.generate_stuff.generate_pytorch_wrappers import (
#     tensor_method_signatures_dict,
#     function_signatures_dict,
# )


ALL = []


def _get_function_signature(
    self,
    function_kind: str,
    parameter_decl_builder: Callable[["SIG_ATTR_TYPE"], str],
    ret_decl_builder: Callable[["SIG_ATTR_TYPE"], str],
) -> str:
    mlir_op_name, _ = self.get_mlir_names()
    # Replace `.` with a valid Python identifier character.
    # `〇` vaguely looks like `.`.
    def_name = "_".join(mlir_op_name.split(".")).replace("aten_", "")
    parameter_decls = list(map(parameter_decl_builder, self.arguments))
    ret_decls = list(map(ret_decl_builder, self.returns))
    parameters = ", ".join(parameter_decls)
    result = ", ".join(ret_decls)
    if len(ret_decls) >= 2:
        result = f"Tuple[{result}]"

    if len(ret_decls) == 0:
        result = "None"

    ALL.append(def_name)
    return f"def {def_name}({parameters}) -> {result}:"


JitOperator._get_function_signature = _get_function_signature


TORCH_TYPE_TO_ODS_TYPE = {
    "Tensor": "AnyTorchTensorType",
    "Tensor?": "AnyTorchOptionalTensorType",
    "Tensor?[]": "AnyTorchListOfOptionalTensorType",
    "Tensor[]": "AnyTorchListOfTensorType",
    "Scalar": "AnyTorchScalarType",
    "Scalar?": "AnyTorchOptionalScalarType",
    "int": "Torch_IntType",
    "int[]": "AnyTorchListOfTorchIntType",
    "int?": "AnyTorchOptionalIntType",
    "int[]?": "AnyTorchOptionalListOfTorchIntType",
    "bool": "Torch_BoolType",
    "bool[]": "AnyTorchListOfTorchBoolType",
    "bool?": "AnyTorchOptionalBoolType",
    "float": "Torch_FloatType",
    "float?": "AnyTorchOptionalFloatType",
    "float[]": "AnyTorchListOfTorchFloatType",
    "float[]?": "AnyTorchOptionalListOfTorchFloatType",
    "t[]": "AnyTorchListType",
    "t": "AnyTorchType",
    "t1": "AnyTorchType",
    "t2": "AnyTorchType",
    "Any": "AnyTorchType",
    "Device": "Torch_DeviceType",
    "Device?": "AnyTorchOptionalDeviceType",
    "Generator": "Torch_GeneratorType",
    "Generator?": "AnyTorchOptionalGeneratorType",
    "str": "Torch_StringType",
    "str?": "AnyTorchOptionalStringType",
    "str[]": "AnyTorchListOfTorchStringType",
    "Dict": "Torch_DictType",
    "__torch__.torch.classes.quantized.LinearPackedParamsBase": "Torch_LinearParamsType",
}


# pyt_type is reversed
def convert_type(pyt_type: str):
    if pyt_type.endswith("?"):
        nested, interior = convert_type(pyt_type[:-1])
        return f"Optional[{nested}]", interior
    elif pyt_type.endswith("[]"):
        nested, interior = convert_type(pyt_type[:-2])
        return f"List[{nested}]", interior
    else:
        subs = {
            "Scalar": "Number",
            "t2": "Tensor",
            "t1": "Tensor",
            "t": "Tensor",
            "Dict(str, t)": "Dict[str, Tensor]",
            "device": "Device",
        }
        interior = subs.get(pyt_type, pyt_type)
        return interior, interior


def convert_type_to_op(arg_name, pyt_type, p_td, emitter_td):
    _, interior = convert_type(pyt_type)

    if pyt_type.endswith("?"):
        p_td(f"if {arg_name} is not None:")
        with emitter_td.indent():
            convert_type_to_op(arg_name, pyt_type[:-1], p_td, emitter_td)
        p_td(f"else:")
        with emitter_td.indent():
            p_td(f"{arg_name} = torch_dialect.ConstantNoneOp()")
        p_td("")
    elif pyt_type.endswith("[]"):
        if interior == "Tensor":
            pass
            # p_td(f"{arg_name} = get_op_results_or_values({arg_name})")
        else:
            op = convert_type_to_op(None, pyt_type[:-2], p_td, emitter_td)
            p_td(f"{arg_name} = list(map({op}, {arg_name}))")
        p_td(f"{arg_name} = torch_dialect.PrimListConstructOp({arg_name})")
    else:
        if interior in {"int", "bool", "float", "Number", "str"}:
            op = f"torch_dialect.Constant{interior.capitalize()}Op"
            if arg_name is not None:
                p_td(f"{arg_name} = {op}({arg_name})")
            else:
                return op
        else:
            if arg_name is not None:
                p_td(
                    f"assert is_mlir_value({arg_name}), f'`{arg_name}` should be a Value but is {{type({arg_name}).__module__}}.{{type({arg_name}).__name__}}'"
                )


EXISTING = {
    "ConstantFloatOp",
    "ConstantIntOp",
    "ConstantStrOp",
    "ConstantBoolOp",
    "PrimListConstructOp",
}


def py_reserved_keywords(k):
    subs = {
        "from": "from_",
        "self": "self_",
        "list": "list_",
    }
    return subs.get(k, k)


def get_wrapper_function_signature(operator):
    """Gets the Python function signature for this op's decomposition function.

    While this is technically debug-only output, it is useful to copy-paste
    it from the debug dump into the shape library definitions, as many
    ops have extra default arguments and stuff that are tedious to write out
    right.
    """

    def parameter_decl_builder(arg: "SIG_ATTR_TYPE") -> str:
        pytype = convert_type(arg["type"])[0]
        default = _get_default_value(arg)
        if arg["name"] == "out":
            default = "= None"
        parameter_name = py_reserved_keywords(
            _rename_python_keyword_parameter_name(arg["name"])
        )
        return f"{parameter_name}: {pytype}{default}"

    def ret_decl_builder(arg: "SIG_ATTR_TYPE") -> str:
        ret = convert_type(arg["type"])[0]
        if not ret:
            ret = "None"
        return ret

    return operator._get_function_signature(
        "", parameter_decl_builder, ret_decl_builder
    )


def raw_emit_op(
    operator: JitOperator,
    emitter_td: TextEmitter,
    *,
    traits: List[str],
    has_folder: bool,
    has_canonicalizer: bool,
):
    p_td = lambda *args: emitter_td.print(*args)
    stub_td = lambda *args: stubs_emitter_td.print(*args)
    op_name, cpp_class_name = operator.get_mlir_names()
    if cpp_class_name in {"QuantizedLinearOp"} | EXISTING:
        return

    # Generate unique result names for ops with nameless results
    multiple_results = len(operator.returns) > 1

    if any(
        [
            arg["type"].endswith("?[]")
            or arg["type"].endswith("[]?")
            or "Dict" in arg["type"]
            or "Device" in arg["type"]
            or "Generator" in arg["type"]
            for arg in operator.arguments
        ]
    ):
        print(f"{cpp_class_name} has weird args")
        return

    if operator.is_vararg:
        print(f"{cpp_class_name} is vararg")
        return
    else:
        args = {
            py_reserved_keywords(arg["name"]): convert_type(arg["type"])[0]
            for arg in operator.arguments
        }
        for k, v in args.items():
            args[k] = v.replace("Tensor", "Value").replace("Number", '"Number"')

    def generic_result_name(i):
        return "result" + (str(i) if multiple_results else "")

    if operator.is_varret:
        print(f"{cpp_class_name} is vararg")
        return
    else:
        ret_names = [
            f'{ret["name"] or generic_result_name(e)}'
            for e, ret in enumerate(operator.returns)
        ]

    if any([ret["type"] == "Device" for ret in operator.returns]):
        print(f"{cpp_class_name} returns device")
        return

    p_td(f"class {cpp_class_name}:")
    ret_type_names = []
    arg_names = []
    with emitter_td.indent():
        args_str = ", ".join([f"{k}: {v}" for k, v in args.items()])
        if args_str:
            args_str = f" {args_str},"
        else:
            args_str = ""
        p_td(f"def __init__(self,{args_str} *, loc=None, ip=None):")
        with emitter_td.indent():
            if any(
                [
                    convert_type(arg["type"])[1] != "Tensor"
                    or "?" in arg["type"]
                    or "[]" in arg["type"]
                    for arg in operator.arguments
                ]
            ):
                p_td(f"from torch_mlir.dialects import torch as torch_dialect\n\n")

            for arg in operator.arguments:
                arg_name = py_reserved_keywords(arg["name"])
                arg_names.append(arg_name)
                arg_type = arg["type"]
                p_td(f"if not is_mlir_value({arg_name}):")
                with emitter_td.indent():
                    convert_type_to_op(arg_name, arg_type, p_td, emitter_td)
                p_td(f"else:")
                not_none_arg_type = arg["type"].replace("?", "")
                with emitter_td.indent():
                    p_td(f"{arg_name} = get_op_result_or_value({arg_name})")
                    if not_none_arg_type in {"Tensor", "t"}:
                        p_td(
                            f"""assert str({arg_name}.type).startswith("!torch.vtensor"), f'`{arg_name}` should be a torch.vtensor but is {{type({arg_name}).__module__}}.{{type({arg_name}).__name__}}'"""
                        )
                    elif not_none_arg_type == "int":
                        p_td(
                            f"""assert str({arg_name}.type) == '!torch.int', f'`{arg_name}` should be a !torch.int but is {{type({arg_name}).__module__}}.{{type({arg_name}).__name__}}'"""
                        )
                    elif not_none_arg_type == "str":
                        p_td(
                            f"""assert str({arg_name}.type) == '!torch.str', f'`{arg_name}` should be a !torch.str but is {{type({arg_name}).__module__}}.{{type({arg_name}).__name__}}'"""
                        )
                    elif not_none_arg_type == "float":
                        p_td(
                            f"""assert str({arg_name}.type) == '!torch.float', f'`{arg_name}` should be a !torch.float but is {{type({arg_name}).__module__}}.{{type({arg_name}).__name__}}'"""
                        )
                    elif not_none_arg_type == "bool":
                        p_td(
                            f"""assert str({arg_name}.type) == '!torch.bool', f'`{arg_name}` should be a !torch.bool but is {{type({arg_name}).__module__}}.{{type({arg_name}).__name__}}'"""
                        )
                    elif not_none_arg_type == "Scalar":
                        p_td(
                            f"""assert str({arg_name}.type) in {{'!torch.float', '!torch.int'}}, f'`{arg_name}` should be a !torch.number but is {{type({arg_name}).__module__}}.{{type({arg_name}).__name__}}'"""
                        )
                    elif not_none_arg_type == "Any":
                        p_td(
                            f"""assert str({arg_name}.type) == '!torch.Any', f'`{arg_name}` should be a !torch.Any but is {{type({arg_name}).__module__}}.{{type({arg_name}).__name__}}'"""
                        )
                    elif not_none_arg_type == "int[]":
                        p_td(
                            f"""assert str({arg_name}.type) == '!torch.list<int>', f'`{arg_name}` should be a !torch.list<int> but is {{type({arg_name}).__module__}}.{{type({arg_name}).__name__}}'"""
                        )
                    elif not_none_arg_type in {"t[]", "Tensor[]"}:
                        p_td(
                            f"""assert str({arg_name}.type) == '!torch.list<Tensor>', f'`{arg_name}` should be a !torch.list<Tensor> but is {{type({arg_name}).__module__}}.{{type({arg_name}).__name__}}'"""
                        )
                    else:
                        print(
                            f"{cpp_class_name} weird arg {arg_name} type {arg['type']}"
                        )
                        p_td(f"# should be {arg['type']}")
                        p_td(f"pass")

                    p_td("\n")

            for e, ret in enumerate(operator.returns):
                name = f'{ret["name"] or generic_result_name(e)}'
                if ret["type"] in {"Tensor", "t"}:
                    p_td(f"""{name}_type = Type.parse("!torch.vtensor")""")
                elif ret["type"] == "int":
                    continue
                    # p_td(f"""{name}_type = Type.parse("!torch.int")""")
                elif ret["type"] == "str":
                    continue
                    # p_td(f"""{name}_type = Type.parse("!torch.str")""")
                elif ret["type"] == "float":
                    continue
                    # p_td(f"""{name}_type = Type.parse("!torch.float")""")
                elif ret["type"] == "bool":
                    continue
                    # p_td(f"""{name}_type = Type.parse("!torch.bool")""")
                elif ret["type"] == "Scalar":
                    continue
                    # p_td(f"""{name}_type = Type.parse("!torch.number")""")
                elif ret["type"] == "Any":
                    continue
                    # p_td(f"""{name}_type = Type.parse("!torch.Any")""")
                elif ret["type"] == "int[]":
                    p_td(f"""{name}_type = Type.parse("!torch.list<int>")""")
                elif ret["type"] == "t[]":
                    p_td(f"""{name}_type = Type.parse("!torch.list<Tensor>")""")
                else:
                    raise Exception(
                        f"{cpp_class_name} weird ret {name} type {ret['type']}"
                    )
                ret_type_names.append(f"{name}_type")

            if ret_type_names:
                ret_type_names = f"{', '.join(ret_type_names)}, "
            else:
                ret_type_names = ""

            if arg_names:
                arg_names = f"{', '.join(arg_names)}, "
            else:
                arg_names = ""

            p_td(
                f"super({cpp_class_name}, self).__init__({ret_type_names}{arg_names}loc=loc, ip=ip)"
            )
            p_td("\n")
        p_td("\n")

    stub_td(get_wrapper_function_signature(operator))
    with stubs_emitter_td.indent():
        for arg in operator.arguments:
            arg_name = py_reserved_keywords(arg["name"])
            if arg_name == "dtype":
                stub_td("if dtype is not None and isinstance(dtype, Enum):")
                with stubs_emitter_td.indent():
                    stub_td("dtype = dtype.value")
            if arg["type"] == "Tensor":
                stub_td(
                    f"assert isinstance({arg_name}, Tensor), f'`{arg_name}` should be a {{Tensor.__module__}}.{{Tensor.__name__}} but is {{type({arg_name}).__module__}}.{{type({arg_name}).__name__}}'"
                )
                stub_td(f"{arg_name} = {arg_name}.value")
            elif arg["type"] == "Tensor?":
                stub_td(f"if {arg_name} is not None:")
                with stubs_emitter_td.indent():
                    stub_td(
                        f"assert isinstance({arg_name}, Tensor), f'`{arg_name}` should be a {{Tensor.__module__}}.{{Tensor.__name__}} but is {{type({arg_name}).__module__}}.{{type({arg_name}).__name__}}'"
                    )
                    stub_td(f"{arg_name} = {arg_name}.value")
            elif arg["type"] == "Tensor[]":
                stub_td(
                    f"assert builtins.all(isinstance(t, Tensor) for t in {arg_name})"
                )
                stub_td(f"{arg_name} = [t.value for t in {arg_name}]")

        call_str = f'torch_dialect.{cpp_class_name}({", ".join([f"{k}" for k, _v in args.items()])})'
        if len(operator.returns) == 0:
            ret = call_str
        elif len(operator.returns) == 1:
            ret = f"return Tensor({call_str})"
        else:
            stub_td(f"op_results = get_op_results_or_values({call_str})")
            ret = f"return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])"
        stub_td(f"{ret}")
        stub_td("\n")


import torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen

torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen.raw_emit_op = (
    raw_emit_op
)

from torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen import emit_ops

# native_yaml_path = "native_functions.yaml"
# tags_yaml_path = "tags.yaml"
# deprecated_yaml_path = "deprecated.yaml"
#
# native_functions = parse_native_yaml(native_yaml_path, tags_yaml_path).native_functions
# native_functions = list(filter(should_generate_py_binding, native_functions))
#
# function_signatures = load_signatures(
#     native_functions, deprecated_yaml_path, method=False, pyi=True
# )
# sig_groups = get_py_torch_functions(function_signatures)
#
#
# tensor_method_signatures = load_signatures(
#     native_functions,
#     deprecated_yaml_path,
#     method=True,
#     skip_deprecated=True,
#     pyi=True,
# )
# tensor_method_sig_groups = get_py_torch_functions(tensor_method_signatures, method=True)

_torch_ops_ext_fp = "../../shark/dialects/_torch_ops_ext.py"
_torch_wrappers_fp = "../../shark/dialects/_torch_wrappers.py"

registry = Registry.load()
with open(_torch_ops_ext_fp, "w") as f_td:
    emitter_td = TextEmitter(f_td)
    emitter_td._INDENT = "    "
    with open(_torch_wrappers_fp, "w") as stubs_td:
        stubs_emitter_td = TextEmitter(stubs_td)
        stubs_emitter_td._INDENT = "    "
        stubs_emitter_td.print(
            dedent(
                f"""\
        from enum import Enum
        import builtins
        
        from .._tensor import Tensor
        from ..types_ import Number, is_a_torch_tensor
        from typing import List, Optional, Any, Tuple

        from torch_mlir.dialects import torch as torch_dialect
        from torch_mlir.dialects._ods_common import (
            get_default_loc_context,
            get_op_result_or_value,
            get_op_results_or_values,
        )
        
        """
            )
        )

        emitter_td.print(
            dedent(
                f"""\
        try:
            # from shark import Tensor, Number
            from torch_mlir.ir import *
            from torch_mlir.dialects._ods_common import (
                get_default_loc_context,
                get_op_result_or_value,
                get_op_results_or_values,
            )
            from ._torch_ops_ext_custom import *
        except ImportError as e:
            raise RuntimeError("Error loading imports from extension module") from e

        from typing import List, Optional, Any


        """
            )
        )
        emit_ops(emitter_td, registry)

        assert len(ALL) == len(set(ALL)), "duplicate ALL"
        ALL = [f"'{a}'" for a in ALL]
        stubs_emitter_td.print("\n\n")
        stubs_emitter_td.print(f"__all__ = [{', '.join(ALL)}]")

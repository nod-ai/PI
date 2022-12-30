import collections
from typing import List

from torchgen.api.python import PythonSignatureGroup
from torchgen.gen import parse_native_yaml
from torchgen.model import SelfArgument, TensorOptionsArguments, NativeFunction

from scripts.generate_stuff.gen_pyi import (
    get_py_torch_functions,
    blocklist,
    binary_ops,
    comparison_ops,
    symmetric_comparison_ops,
    unary_ops,
    to_py_type_ops,
)
from scripts.generate_stuff.gen_python_functions import (
    should_generate_py_binding,
    load_signatures,
)


def generate_type_hints(sig_group: PythonSignatureGroup) -> List[str]:
    type_hints: List[str] = []

    # Some deprecated ops that are on the blocklist are still included in pyi
    if sig_group.signature.name in blocklist and not sig_group.signature.deprecated:
        return type_hints

    # deprecated signatures have separate entries for their functional and out variants
    # (as opposed to the native ops, which fuse the two into a single signature).
    # generate the functional variant here, if an out variant exists.
    if sig_group.signature.deprecated and sig_group.outplace is not None:
        type_hint = sig_group.signature.signature_str_pyi(skip_outputs=True)
        type_hints.append(type_hint)

    # PythonSignatureGroups that have both a functional + out variant get a single signature, with an optional out argument
    # Generates the out variant if one exists. Otherwise, generate the functional variant
    type_hint = sig_group.signature.signature_str_pyi(
        skip_outputs=sig_group.outplace is None
    )
    type_hints.append(type_hint)

    # Some operators also additionally have a vararg variant of their signature
    type_hint_vararg = sig_group.signature.signature_str_pyi_vararg(
        skip_outputs=sig_group.outplace is None
    )
    if type_hint_vararg:
        type_hints.append(type_hint_vararg)

    return type_hints


def sig_for_ops(opname: str) -> List[str]:
    """sig_for_ops(opname : str) -> List[str]

    Returns signatures for operator special functions (__add__ etc.)"""

    # we have to do this by hand, because they are hand-bound in Python

    assert opname.endswith("__") and opname.startswith("__"), "Unexpected op {}".format(
        opname
    )

    name = opname[2:-2]
    if name in binary_ops:
        return ["def {}(self, other: Any) -> Tensor: ...".format(opname)]
    elif name in comparison_ops:
        sig = "def {}(self, other: Any) -> Tensor: ...".format(opname)
        if name in symmetric_comparison_ops:
            # unsafe override https://github.com/python/mypy/issues/5704
            sig += "  # type: ignore[override]"
        return [sig]
    elif name in unary_ops:
        return ["def {}(self) -> Tensor: ...".format(opname)]
    elif name in to_py_type_ops:
        if name in {"bool", "float", "complex"}:
            tname = name
        elif name == "nonzero":
            tname = "bool"
        else:
            tname = "int"
        if tname in {"float", "int", "bool", "complex"}:
            tname = "builtins." + tname
        return ["def {}(self) -> {}: ...".format(opname, tname)]
    else:
        raise Exception("unknown op", opname)


native_yaml_path = "native_functions.yaml"
tags_yaml_path = "tags.yaml"
deprecated_yaml_path = "deprecated.yaml"

native_functions = parse_native_yaml(native_yaml_path, tags_yaml_path).native_functions
native_functions = list(filter(should_generate_py_binding, native_functions))

function_signatures = load_signatures(
    native_functions, deprecated_yaml_path, method=False, pyi=True
)
function_sig_groups = get_py_torch_functions(function_signatures)

tensor_method_signatures = load_signatures(
    native_functions,
    deprecated_yaml_path,
    method=True,
    skip_deprecated=True,
    pyi=True,
)
tensor_method_sig_groups = get_py_torch_functions(tensor_method_signatures, method=True)


# def create_unique_key(sig_group: PythonSignatureGroup) -> str:
def create_unique_key(base: NativeFunction) -> str:
    func = base.func

    # is_vararg = sig_group.signature.signature_str_pyi_vararg(
    #     skip_outputs=sig_group.outplace is None
    # )
    # if is_vararg is not None:
    #     print(is_vararg)
    is_vararg = False
    is_varret = False

    overload = "" if not func.name.overload_name else f".{func.name.overload_name}"
    if is_vararg:
        arg_str = "..."
    else:
        arg_str = []
        for arg in func.arguments.all:
            if isinstance(arg, SelfArgument):
                arg_str.append(str(arg.argument.type))
            elif isinstance(arg, TensorOptionsArguments):
                arg_str.append(str(arg.dtype.type))
                arg_str.append(str(arg.layout.type))
                arg_str.append(str(arg.device.type))
                arg_str.append(str(arg.pin_memory.type))
            else:
                typ_str = str(arg.type)
                if typ := arg.type.is_list_like():
                    if typ.size is not None:
                        typ_str = typ_str.replace(str(typ.size), "")

                arg_str.append(typ_str)

        arg_str = ", ".join(arg_str)

    if is_varret:
        ret_str = "..."
    else:
        ret_str = ", ".join(str(ret.type) for ret in func.returns)
    if "." in str(func.name):
        unqualified_name, _overload = str(func.name).split(".")
    else:
        unqualified_name = str(func.name)
    return f"{base.namespace}::{unqualified_name}{overload} : ({arg_str}) -> ({ret_str})"


function_signatures_dict = {
    create_unique_key(group.base): group
    for group in sorted(function_sig_groups, key=lambda g: g.signature.name)
}
for uniq in function_signatures_dict:
    print(uniq)

tensor_method_signatures_dict = {
    create_unique_key(group.base): group
    for group in sorted(tensor_method_sig_groups, key=lambda g: g.signature.name)
}
for uniq in tensor_method_signatures_dict:
    print(uniq)

print()
import ast
import re
import warnings
from collections import defaultdict
from pathlib import Path
from textwrap import dedent, indent

import orjson
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.registry import (
    Registry,
)
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.utils import TextEmitter


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


RESERVED_NAMES = {
    # since this is an aten op
    "pad": "pad__",
    "threshold": "threshold__",
}


def get_clean_name(name):
    if name in RESERVED_NAMES:
        return RESERVED_NAMES[name]
    return name


UNIMPLEMENTED_TYPES = {
    "AnyTorchType",
    "anonymous_430",
}
SKIP_OPS = {"Torch_PrimsSqrtOp", "Torch_AtenChunkOp"}
SKIP_TENSOR_BINDS = {
    "@overload view(self, dtype: _dtype) -> Tensor",
    "@overload view(self, size: Sequence[Union[_int, SymInt]]) -> Tensor",
    "@overload view(self, *size: _int) -> Tensor",
    "__truediv__(self, other: Any) -> Tensor",
    "__rtruediv__(self, other: Any) -> Tensor",
    "chunk(self, chunks: _int, dim: _int=0) -> List[Tensor]",
    "__getitem__(self, indices: Union[None, _int, slice, Tensor, List, Tuple]) -> Tensor",
}

TORCH_OPS_IMPL_CPP = "TorchOps.impls.cpp"
TORCH_OPS_INC_H = "TorchOps.inc.h"
TORCH_OPS_PYBINDS_CPP = "TorchOps.pybinds.cpp"
TORCH_TENSOR_PYBINDS_CPP = "TorchTensor.pybinds.cpp"
TORCH_TENSOR_PYBINDS_TRAMPS_CPP = "TorchTensor.pybinds_tramps.cpp"


def get_schema_from_ods(ods):
    schema = ods["summary"].replace("Generated op for", "").replace("`", "").strip()
    return schema


def get_params_from_ods_args(ods):
    args = ods["arguments"]["args"]
    params = [
        (
            get_clean_name(name),
            def_["def"],
        )
        for def_, name in args
    ]
    return params


def get_result_type_from_ods(ods):
    results = ods["results"]["args"]

    result_type_args = []
    result_types = []
    for result in results:
        result_type = result[0]["def"]
        result_types.append(result_type)
        if "Any" in result_type:
            if result_type == "AnyTorchTensorType":
                result_type_arg = "PyAnyTorchTensorType::getWithLeastStaticInformation(loc->getContext().get())"
            elif result_type in {
                "AnyTorchListOfTorchStringType",
                "AnyTorchListOfTorchIntType",
            }:
                result_type_arg = f"Py{result_type}(loc->getContext().get())"
            elif result_type == "AnyTorchListType":
                assert len(ods["arguments"]["args"])
                first_arg_name = ods["arguments"]["args"][0][1]
                contained_type_first_arg = f"torchMlirTorchListTypeGetContainedType(mlirValueGetType({first_arg_name}))"
                result_type_arg = f"Py{result_type}({contained_type_first_arg}, loc->getContext().get())"
            elif result_type == "AnyTorchListOfTensorType":
                contained_type_first_arg = "PyAnyTorchTensorType::getWithLeastStaticInformation(loc->getContext().get())"
                result_type_arg = f"Py{result_type}({contained_type_first_arg}, loc->getContext().get())"
            else:
                warnings.warn(f"Unimplemented return type {result_type}")
                return
            result_type_args.append(result_type_arg)
        else:
            # is known since it's not an Any
            continue

    if len(result_type_args):
        result_type_arg = f"{', '.join(result_type_args)}"
    else:
        result_type_arg = ""

    if len(result_types) == 0:
        result_type = "void"
        cast = ""
    elif len(result_types) == 1:
        result_type = f"Py{result_types[0].replace('Type', 'Value')}"
        cast = f".cast<{result_type}>()"
    else:
        result_types = [f"Py{r.replace('Type', 'Value')}" for r in result_types]
        result_type = f"std::tuple<{', '.join(result_types)}>"
        cast = f".cast<{result_type}>()"

    return result_type, result_type_arg, cast


def get_defaults_from_jit_op(op):
    defaults = {
        a["name"]: a["default_debug"] for a in op.arguments if "default_debug" in a
    }
    return defaults


def generate_torch_ops_impls(cpp_ext_dir, torch_mlir_ods_json):
    registry = Registry.load()

    with open("JitOperatorRegistry.txt", "w") as f:
        for jit_op in registry.by_unique_key.values():
            print(jit_op, file=f)

    jitops_odses = []
    with open(f"{cpp_ext_dir}/{TORCH_OPS_IMPL_CPP}", "w") as impls_file, open(
        f"{cpp_ext_dir}/{TORCH_OPS_INC_H}", "w"
    ) as impls_h_file:
        impls_h_emitter = TextEmitter(impls_h_file)
        impls_emitter = TextEmitter(impls_file)
        impls_emitter._INDENT = "    "
        impls_td = lambda *args: impls_emitter.print(*args)
        impls_h_td = lambda *args: impls_h_emitter.print(*args)

        ods_ops = {
            k: v
            for k, v in torch_mlir_ods_json.items()
            if isinstance(v, dict)
            and v.get("opDialect", {}).get("def") == "Torch_Dialect"
        }

        for op_name, ods in ods_ops.items():
            if op_name in SKIP_OPS:
                warnings.warn(f"Skipping op {op_name}")
                continue
            schema = get_schema_from_ods(ods)
            if schema not in registry.by_unique_key:
                warnings.warn(f"Unimplemented non-torch-jit op {op_name}")
                continue

            jit_op = registry.by_unique_key[schema]
            n_returns = len(jit_op.returns)
            op_name, cpp_class_name = jit_op.get_mlir_names()
            params = get_params_from_ods_args(ods)
            result_type_result_type_arg_cast = get_result_type_from_ods(ods)
            if result_type_result_type_arg_cast is None:
                warnings.warn(f"Unimplemented return type for {schema=}")
                continue

            (
                result_type,
                result_type_arg,
                cast,
            ) = result_type_result_type_arg_cast
            has_unimplemented_type = UNIMPLEMENTED_TYPES.intersection(
                {type for _name, type in params}
            )
            if has_unimplemented_type:
                warnings.warn(
                    f"Unimplemented param type {has_unimplemented_type} for {schema=}"
                )
                continue

            api_param_str = ", ".join(
                [
                    f"const Py{type.replace('Type', 'Value')} &{name}"
                    for name, type in params
                ]
                + ["PyLocation *loc", "PyInsertionPoint *ip"]
            )
            joined_params = ", ".join([name for name, _type in params])

            op_name = f"torch.{op_name}"
            impls_td(
                dedent(
                    f"""\
                    // {schema}
                    {result_type} {jit_op.unqualified_name}({api_param_str}) {{
                      std::string operationName = "{op_name}";
                    """
                )
            )

            if result_type_arg == "" and result_type != "void":
                infer_return_types = dedent(
                    f"""\
                    auto _returnTypes = inferReturnTypes(operationName, {{{joined_params}}}, loc->getContext().get(), loc); 
                    """
                )
            else:
                infer_return_types = dedent(
                    f"""\
                    std::vector<PyType> _returnTypes = {{{result_type_arg}}}; 
                    """
                )
            impls_td(indent(infer_return_types, "  "))

            impl = dedent(
                f"""\
                      std::vector<std::reference_wrapper<const PyType>> returnTypes; 
                      for (const auto& returnType : _returnTypes) 
                        returnTypes.push_back(returnType);
                      PyOperationRef opRef = createOperation(operationName,
                                returnTypes,
                                {{{joined_params}}}, 
                                /*attributes=*/{{}}, 
                                loc, 
                                ip);
                      MlirOperation operation = opRef->get();
                """
            )
            impls_td(indent(impl, "  "))

            if n_returns == 0:
                impls_td(indent("// no result", "  "))
            elif n_returns == 1:
                impls_td(
                    indent(
                        dedent(
                            f"""\
                            return {{opRef, mlirOperationGetResult(operation, 0)}};
                            """
                        ),
                        "  ",
                    )
                )
            else:
                impls_td(
                    indent(
                        dedent(
                            f"""\
                            return {result_type}({', '.join(['{opRef, mlirOperationGetResult(operation, ' + str(i) + ')}' for i in range(n_returns)])});
                            """
                        ),
                        "  ",
                    )
                )
            impls_td("}\n")

            header = dedent(
                f"""
                    // {schema}
                    {result_type} {jit_op.unqualified_name}({api_param_str});
                """
            )
            impls_h_td(header)

            jitops_odses.append((jit_op, ods))

    return jitops_odses


def generate_torch_ops_pybinds(jitops_odses, cpp_ext_dir):
    with open(f"{cpp_ext_dir}/{TORCH_OPS_PYBINDS_CPP}", "w") as impls_file:
        impls_emitter = TextEmitter(impls_file)
        impls_emitter._INDENT = "    "
        impls_td = lambda *args: impls_emitter.print(*args)
        for jit_op, ods in jitops_odses:
            labels = []
            param_str = []
            tramp_str = []
            schema = get_schema_from_ods(ods)
            params = get_params_from_ods_args(ods)
            defaults = get_defaults_from_jit_op(jit_op)

            for name, type in params:
                type = type.replace("Type", "Value")
                if "TorchOptional" in type:
                    param_str.append(f"const Py{type} &{name}")
                    tramp_str.append(f"{name}")
                    labels.append(f'"{name}"_a = py::none()')
                elif name in defaults:
                    default = defaults[name]
                    param_str.append(f"const Py{type} &{name}")
                    tramp_str.append(name)
                    labels.append(f'"{name}"_a = {map_defaults(default, schema)}')
                else:
                    param_str.append(f"const Py{type} &{name}")
                    tramp_str.append(name)
                    labels.append(f'"{name}"_a')

            param_str.extend(
                ["DefaultingPyLocation &loc", "const DefaultingPyInsertionPoint &ip"]
            )
            labels.extend(
                ["py::kw_only()", '"loc"_a = py::none()', '"ip"_a = py::none()']
            )
            tramp_str.extend(["loc.get()", "ip.get()"])

            labels_str = ", ".join(labels)
            tramp_str = ", ".join(tramp_str)
            param_str = ", ".join(param_str)

            result_type, *_ = get_result_type_from_ods(ods)
            impl = dedent(
                f"""
                    // {schema}
                    m.def("{jit_op.unqualified_name}", []({param_str}) -> {result_type} {{ return {jit_op.unqualified_name}({tramp_str}); }}, {labels_str});
                """
            )

            impls_td(impl)


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

    def __init__(self, ops, cpp_ext_dir):
        self.ops = ops
        self.binds_file = open(f"{cpp_ext_dir}/{TORCH_TENSOR_PYBINDS_CPP}", "w")
        self.binds_emitter = TextEmitter(self.binds_file)
        self.binds_emitter._INDENT = "    "
        self.binds_td = lambda *args: self.binds_emitter.print(*args)

        self.binds_tramps_file = open(
            f"{cpp_ext_dir}/{TORCH_TENSOR_PYBINDS_TRAMPS_CPP}", "w"
        )
        self.binds_tramps_emitter = TextEmitter(self.binds_tramps_file)
        self.binds_tramps_emitter._INDENT = "    "
        self.binds_tramps_td = lambda *args: self.binds_tramps_emitter.print(*args)
        self.visited = set()

    def emit_not_implemented(self, method_sig, op_name):
        impl = dedent(
            f"""
                // {method_sig}
                c.def("{op_name}", [](PyAnyTorchTensorValue& self, py::args args, py::kwargs kwargs) {{ throw NotImplementedError("NotImplementedError: {op_name} with signature {method_sig}"); }});
            """
        )
        self.binds_td(impl)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        method_sig = (
            ast.unparse(node)
            .replace("def", "")
            .replace("\n", "")
            .replace("...", "")
            .strip()
        )
        if method_sig[-1] == ":":
            method_sig = method_sig[:-1]

        if method_sig in SKIP_TENSOR_BINDS:
            return

        if "truediv" in method_sig:
            print(method_sig)

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

        # defaults = node.args.defaults
        arg_names = [a.arg for a in node.args.args if a not in kwonlyargs]
        arg_types = [a.annotation for a in node.args.args if a not in kwonlyargs]
        if node.args.vararg:
            arg_names.append(f"*{node.args.vararg.arg}")
        if node.args.kwonlyargs:
            for kwarg in node.args.kwonlyargs:
                arg_names.append(kwarg.arg)
                arg_types.append(kwarg.annotation)

        def try_to_find_op(op_name):
            matching_ops = []
            for i, (jit_op, ods) in enumerate(self.ops.get(op_name, [])):
                params_dict = dict(get_params_from_ods_args(ods))
                if (
                    set(arg_names) == set(params_dict.keys())
                    and "self" in params_dict
                    and params_dict["self"] == "AnyTorchTensorType"
                ):
                    matching_ops.append((jit_op, ods))
            return matching_ops

        matching_jitop_ods = try_to_find_op(op_name)
        overload_op_name = None
        if (
            len(matching_jitop_ods) == 0
            and op_name.startswith("__")
            and op_name.endswith("__")
        ):
            overload_op_name = op_name.replace("__", "")
            matching_jitop_ods = try_to_find_op(overload_op_name)

        if len(matching_jitop_ods) == 0:
            warnings.warn(
                f"found no matching overload for {op_name=} with {method_sig=}"
            )
            return

        for jit_op, ods in matching_jitop_ods:
            schema = get_schema_from_ods(ods)
            params = get_params_from_ods_args(ods)
            defaults = get_defaults_from_jit_op(jit_op)
            if schema in self.visited:
                return

            result_type, *_ = get_result_type_from_ods(ods)

            self.visited.add(schema)

            if result_type:
                return_ = "return "
            else:
                return_ = ""

            params_dict = dict(params)
            if arg_names != [p[0] for p in params] or (
                overload_op_name is not None and overload_op_name != op_name
            ):
                tramp_param_str = ", ".join(
                    [
                        f"const Py{params_dict[name].replace('Type', 'Value')} &{name}"
                        for name in arg_names
                    ]
                    + ["PyLocation *loc", "PyInsertionPoint *ip"]
                )
                impl = dedent(
                    f"""
                            // {method_sig}
                            // {schema}
                            {result_type} {op_name}({tramp_param_str}) {{
                              {return_}{overload_op_name}({', '.join([p[0] for p in params] + ["loc", "ip"])});
                            }}
                        """
                )
                self.binds_tramps_td(impl)

            assert params[0][0] == "self", f"{params} for {method_sig=}"
            labels = []
            param_str = []
            tramp_str = []
            for i, (name, type) in enumerate(params):
                type = type.replace("Type", "Value")
                if "TorchOptional" in type:
                    labels.append(f'"{name}"_a = py::none()')
                    param_str.append(f"const Py{type} &{name}")
                    tramp_str.append(f"{name}")
                elif name in kwonlyargs:
                    if "py::kw_only()" not in labels:
                        labels.append("py::kw_only()")
                    assert (
                        name in defaults
                    ), f"default_debug is None for {name=} {type=} {method_sig=}"
                    defau = map_defaults(defaults[name], method_sig)
                    if defau is not None:
                        labels.append(f'"{name}"_a = {defau}')
                        param_str.append(f"const Py{type} &{name}")
                        tramp_str.append(f"{name}")
                elif len(params) - len(defaults) <= i < len(params):
                    assert (
                        name in defaults
                    ), f"default_debug is None for {name=} {type=} {method_sig=}"
                    defau = map_defaults(defaults[name], method_sig)
                    if defau is not None:
                        labels.append(f'"{name}"_a = {defau}')
                        param_str.append(f"const Py{type} &{name}")
                        tramp_str.append(f"{name}")
                else:
                    param_str.append(f"const Py{type} &{name}")
                    tramp_str.append(name)
                    labels.append(f'"{name}"_a')

            # apparently you can't label the self arg for a class method???
            param_str.extend(
                ["DefaultingPyLocation &loc", "const DefaultingPyInsertionPoint &ip"]
            )
            if "py::kw_only()" not in labels:
                labels.append("py::kw_only()")
            labels.extend(['"loc"_a = py::none()', '"ip"_a = py::none()'])
            tramp_str.extend(["loc.get()", "ip.get()"])
            labels_str = ", ".join(labels[1:])
            tramp_str = ", ".join(tramp_str)
            param_str = ", ".join(param_str)

            impl = dedent(
                f"""
                        // {method_sig}
                        // {schema}
                        c.def("{op_name}", []({param_str}) -> {result_type} {{ return {op_name}({tramp_str}); }}, {labels_str});
                    """
            )

            self.binds_td(impl)


class FindTensorClass(ast.NodeVisitor):
    def __init__(self, ops, cpp_ext_dir):
        self.tens_visitor = TensorMethodVisitor(ops, cpp_ext_dir)

    def visit_ClassDef(self, node: ast.ClassDef):
        if node.name == "_TensorBase":
            self.tens_visitor.visit(node)


def generate_tensor_bindings(ops, cpp_ext_dir):
    with open("__init__.pyi") as f:
        tree = ast.parse(f.read())

    ops_dict = defaultdict(list)
    for jit_op, ods in jitops_odses:
        ops_dict[jit_op.unqualified_name].append((jit_op, ods))
    f = FindTensorClass(ops_dict, cpp_ext_dir)
    f.visit(tree)
    f.tens_visitor.binds_file.close()
    f.tens_visitor.binds_tramps_file.close()


if __name__ == "__main__":
    cpp_ext_dir = str((Path(__file__).parent.parent.parent / "cpp_ext").absolute())
    with open("torch.json") as f:
        ods_parsed_json = orjson.loads(f.read())
    jitops_odses = generate_torch_ops_impls(cpp_ext_dir, ods_parsed_json)
    generate_torch_ops_pybinds(jitops_odses, cpp_ext_dir)
    generate_tensor_bindings(jitops_odses, cpp_ext_dir)

from __future__ import annotations

import json
import re
import subprocess
import warnings
from pathlib import Path
from typing import (
    Optional,
    Union,
    List,
    Mapping,
    Dict,
)

import libcst as cst
from libcst import (
    Assign,
    Call,
    FunctionDef,
    Name,
    Return,
    Annotation,
    Param,
    Float,
    Tuple,
    AssignTarget,
    VisitorMetadataProvider,
    CSTNode,
)
from libcst import matchers
from libcst._position import CodeRange, CodePosition
from libcst.metadata import (
    ScopeProvider,
    ParentNodeProvider,
    PositionProvider,
    ProviderT,
    ExpressionContextProvider,
)
from libcst.metadata.type_inference_provider import (
    run_command,
    _process_pyre_data,
    PyreData,
)


def accept(self, visitor, *args, **kwargs):
    """Visit this node using the given visitor."""
    func = getattr(visitor, "visit_" + self.__class__.__name__.lower())
    return func(self, *args, **kwargs)


from shark.ir import (
    Type as MLIRType,
    Attribute,
    IntegerType,
    Context,
    Location,
    F64Type,
)

# pylint: disable=unused-argument

DOC_NEWLINE = "\0"


# types are uniqed by context (see mlir/IR/Types.h::Type):
# They wrap a pointer to the storage object owned by MLIRContext.
# ...
#   bool operator==(Type other) const { return impl == other.impl; }
#   bool operator!=(Type other) const { return !(*this == other); }
#   protected:
#       ImplType *impl{nullptr};
def map_type_str_to_mlir_type(
    thing: Union[str, Tuple[str]], context: Context, location: Location = None
) -> Union[MLIRType, Tuple[MLIRType]]:
    # catch all that should be factored
    if location is None:
        location = Location.unknown(context=context)
    with context, location:
        if isinstance(thing, tuple):
            return tuple(map(lambda t: map_type_str_to_mlir_type(t, context), thing))
        else:
            return {
                "int": IntegerType.get_signed(64),
                "float": F64Type.get(),
            }[thing]


def infer_mlir_type(py_val) -> MLIRType:
    if isinstance(py_val, int):
        # return IntegerType.get_signed(64)
        return IntegerType.get_signless(64)
    elif isinstance(py_val, float):
        return F64Type.get()
    else:
        raise Exception(f"unsupported val type {type(py_val)} {py_val}")


def map_pyre_type_str_to_mlir_type_str(type_info):
    pyre_typing_map = {"float": "float", "int": "int", "range": "int"}

    if type := re.findall(r"typing.Type\[(\w+)\]", type_info):
        type, *_ = type
        return pyre_typing_map[type]
    elif type_info in pyre_typing_map:
        return pyre_typing_map[type_info]
    # pyre does a really weird thing for binary ops...
    elif type := re.findall(
        r"BoundMethod\[typing\.Callable\((\w+)\.__mul__\)\[\[Named\(self, (\w+)\), (\w+)\], (\w+)\], (\w+)\]",
        type_info,
    ):
        assert len(set(type[0])) == 1
        return pyre_typing_map[type[0][0]]
    elif type_info.startswith("typing.Callable"):
        # TODO(max): sometimes return types has stuff but i've only seen typing.Any
        input_types, _return_types = re.findall(r"\[\[(.*)\],(.*)\]", type_info)[0]
        input_types = tuple(
            map(
                lambda r: r[1].strip(),
                re.findall(r"Named\((\w+), (\w+)(.*?)\)", input_types),
            )
        )
        return input_types
    else:
        raise Exception(f"unknown type {type_info}")


def map_libcst_type(node):
    libcst_typing_map = {"Float": "float"}
    return libcst_typing_map[node.__class__.__name__]


class MyTypeInferenceProvider(VisitorMetadataProvider[str]):
    METADATA_DEPENDENCIES = (PositionProvider,)

    @staticmethod
    # pyre-fixme[40]: Static method `gen_cache` cannot override a non-static method
    #  defined in `cst.metadata.base_provider.BaseMetadataProvider`.
    def gen_cache(
        root_path: Path, paths: List[str], timeout: Optional[int]
    ) -> Mapping[str, object]:
        params = ",".join(f"path='{root_path / path}'" for path in paths)
        cmd_args = ["pyre", "--noninteractive", "query", f"types({params})"]
        try:
            stdout, stderr, return_code = run_command(cmd_args, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise exc

        if return_code != 0:
            raise Exception(f"stderr:\n {stderr}\nstdout:\n {stdout}")
        try:
            resp = json.loads(stdout)["response"]
        except Exception as e:
            raise Exception(f"{e}\n\nstderr:\n {stderr}\nstdout:\n {stdout}")
        # CST strips leading lines with hashes in them but pyre does not
        processed_paths = {
            path: _process_pyre_data(data) for path, data in zip(paths, resp)
        }
        for path, types in processed_paths.items():
            full_path = root_path / path
            with open(full_path) as f:
                num_hash_lines = 0
                for l in f.readlines():
                    if l.startswith("#"):
                        num_hash_lines += 1
                    else:
                        break
            for type in types["types"]:
                type["location"]["start"]["line"] -= num_hash_lines
                type["location"]["stop"]["line"] -= num_hash_lines

        return processed_paths

    def __init__(self, cache: Mapping[ProviderT, PyreData]) -> None:
        super().__init__(cache)
        lookup: Dict[CodeRange, str] = {}
        cache_types = cache[type(self)].get("types", [])
        for item in cache_types:
            location = item["location"]
            start = location["start"]
            end = location["stop"]
            lookup[
                CodeRange(
                    start=CodePosition(start["line"], start["column"]),
                    end=CodePosition(end["line"], end["column"]),
                )
            ] = item["annotation"]
        self.lookup: Dict[CodeRange, str] = lookup

    def _parse_metadata(self, node: cst.CSTNode) -> None:
        range = self.get_metadata(PositionProvider, node)
        if range in self.lookup:
            self.set_metadata(node, self.lookup.pop(range))

    def visit_Name(self, node: Name):
        self._parse_metadata(node)

    def visit_Attribute(self, node: Attribute):
        self._parse_metadata(node)

    def visit_Call(self, node: Call):
        self._parse_metadata(node)

    def visit_FunctionDef(self, node: FunctionDef):
        self._parse_metadata(node.name)


class ReturnFinder(matchers.MatcherDecoratableVisitor):
    _returns = []
    func_node = None

    @matchers.visit(matchers.FunctionDef())
    def visit_(self, node):
        self.func_node = node
        return True

    @matchers.visit(matchers.Return())
    def visit_(self, node):
        self._returns.append(node)
        return False

    def __call__(self, node: CSTNode):
        node.visit(self)
        if len(self._returns) == 0:
            warnings.warn(f"no return for {self.func_node.name}")
            return None

        function_return = self._returns.pop()
        if len(self._returns) > 0:
            raise Exception(f"multiple return sites unsupported {self.func_node.name}")
        return function_return


class MLIRTypeProvider(
    VisitorMetadataProvider[MLIRType], matchers.MatcherDecoratableVisitor
):
    METADATA_DEPENDENCIES = (
        MyTypeInferenceProvider,
        ScopeProvider,
        ParentNodeProvider,
        ExpressionContextProvider,
    )

    def gen_cache(self, *args, **kwargs):
        pass

    def __init__(self, cache: Context):
        super().__init__()
        self.context = cache

    @matchers.visit(matchers.AssignTarget())
    @matchers.visit(matchers.Assign())
    @matchers.visit(matchers.Annotation())
    @matchers.visit(matchers.FunctionDef())
    @matchers.visit(matchers.Param())
    @matchers.visit(matchers.Return())
    def visit_(self, _node):
        return True

    def leave_AssignTarget(self, node: AssignTarget):
        typ = self.get_metadata(type(self), node.target)
        self.set_metadata(node, typ)

    def leave_Assign(self, node: Assign) -> None:
        assert (
            len(node.targets) == 1
        ), f"multiple assign targets unsupported {node.targets}"
        lhs = node.targets[0]
        value_mlir_type = self.get_metadata(type(self), lhs.target, None)
        assert value_mlir_type, f"no type found for {lhs.target}"
        self.set_metadata(lhs, value_mlir_type)

    def leave_Annotation(self, node: Annotation):
        typ = self.get_metadata(type(self), node.annotation)
        self.set_metadata(node, typ)

    def leave_Param(self, node: Param):
        # there's a bug in pyre where a param with a default value is inferred to be any
        # even though the default determines the type
        # so check that first and set the name of both the param and the name
        typ = None
        if node.default:
            typ = self.get_metadata(type(self), node.default)
        if node.annotation and typ is None:
            typ = self.get_metadata(type(self), node.annotation)
        if node.default and typ is None:
            typ = self.get_metadata(type(self), node.default)

        # try strong type inference
        if typ is None:
            typ = self.get_metadata(type(self), node.name)

        assert typ
        self.set_metadata(node, typ)
        if self.get_metadata(type(self), node.name, None) is None:
            self.set_metadata(node.name, typ)

    def leave_Return(self, node: Return):
        if isinstance(node.value, Name):
            if return_type := self.get_metadata(type(self), node.value, None):
                return_type = return_type
            else:
                scope = self.get_metadata(ScopeProvider, node.value)
                last_assign_node = list(scope.assignments[node.value.value])[-1].node
                return_type = self.get_metadata(type(self), last_assign_node)
                assert return_type

            self.set_metadata(node.value, return_type)
            self.set_metadata(node, return_type)
        else:
            raise Exception(f"unsupported expr return {node.value}")

    def leave_FunctionDef(self, node: FunctionDef):
        if arg_types := self.get_metadata(type(self), node.name):
            arg_types = arg_types
        else:
            arg_types = [self.get_metadata(type(self), p) for p in node.params.params]

        if node.returns:
            return_types = [self.get_metadata(type(self), node.returns)]
        else:
            function_returns = ReturnFinder()(node)
            if function_returns is not None:
                function_return_type = self.get_metadata(
                    type(self), function_returns, None
                )
                assert function_return_type, f"return type not found {node.name}"
                return_types = [function_return_type]
            else:
                raise Exception(f"no return for {node.name}")

                # (
                #     [self.standard_types["i64"]] * len(arg_names),
                #     [self.standard_types["i64"]],
                # ),
        # with Context(), Location.unknown():
        #     function_type = MLIRFunctionType.get(inputs=arg_types, results=return_types)
        self.set_metadata(node, (arg_types, return_types))

    def visit_Name(self, node: Name):
        if node.value in {"float", "int", "bool"}:
            type_info = node.value
        else:
            # use strong type inference
            try:
                type_info: str = self.get_metadata(MyTypeInferenceProvider, node)
            except KeyError:
                type_info = None

        if type_info is None:
            warnings.warn(f"couldn't get typeinfo for {node}")
            return

        if type_info == "typing.Any":
            # type narrowing?
            # maybe we already have a type for this?
            if existing_type := self.get_metadata(type(self), node, None):
                self.set_metadata(node, existing_type)
                return
            else:
                warnings.warn(f"unrecognized {type_info=} type for node {node.value}")
                return

        mlir_type_str: str = map_pyre_type_str_to_mlir_type_str(type_info)
        mlir_type = map_type_str_to_mlir_type(mlir_type_str, self.context)
        if existing_type := self.get_metadata(type(self), node, None):
            assert str(mlir_type) == str(
                existing_type
            ), f"mismatched types {existing_type=} {mlir_type=} for node {node.value}"
            return
        self.set_metadata(node, mlir_type)

    def visit_Float(self, node: Float):
        self.set_metadata(node, map_type_str_to_mlir_type("float", self.context))

    def visit_Integer(self, node: Float):
        self.set_metadata(node, map_type_str_to_mlir_type("int", self.context))

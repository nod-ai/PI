import sys
from collections import namedtuple
from contextlib import contextmanager
from typing import Optional

from pytype import context, load_pytd, config
from pytype.abstract import abstract_utils
from pytype.pytd import pytd
from pytype.tracer_vm import CallTracer

from shark import ir
from shark.dialects import func

MLIRStackFrame = namedtuple(
    "MLIRStackFrame",
    [
        "block",
        "block_args",
        "context",
        "insertion_point",
        "location",
        "scope_name",
    ],
)


class MyCallTracer(CallTracer):
    def __init__(self, *args, **kwargs):
        assert "top_context" in kwargs
        assert "module" in kwargs
        assert "location" in kwargs
        self.mlir_context = kwargs.pop("context", None)
        self.top_mlir_context = kwargs.pop("top_context")
        self.mlir_module = kwargs.pop("module")
        self.mlir_location = kwargs.pop("location")
        super().__init__(*args, **kwargs)

        self.f64_type = ir.F64Type.get(context=self.top_mlir_context)
        self.f64_memref_type = ir.MemRefType.get(
            (-1,), self.f64_type, loc=self.mlir_location
        )

        # TODO(max): not correct - should be a function of the call or module or something like that
        self._mlir_block_stack: list[MLIRStackFrame] = []
        self.enter_mlir_block_scope(
            self.mlir_module.body,
            mlir_context=self.mlir_context or self.top_mlir_context,
            scope_name="module",
        )
        self.fn_lines = {}

    def enter_mlir_block_scope(
        self,
        block: ir.Block,
        *,
        block_args: tuple[str] = None,
        mlir_context: Optional[ir.Context] = None,
        insertion_point: Optional[ir.InsertionPoint] = None,
        mlir_location: Optional[ir.Location] = None,
        # for debugging
        scope_name: Optional[str] = None,
    ):
        if mlir_context is None:
            # one past the last op but still inside the block
            mlir_context = ir.Context()
        if insertion_point is None:
            # one past the last op but still inside the block
            insertion_point = ir.InsertionPoint(block)
        if mlir_location is None:
            mlir_location = self.mlir_location
        if block_args is None:
            block_args = ()
        if scope_name is None:
            scope_name = "UNNAMED_SCOPE"

        mlir_context.__enter__()
        insertion_point.__enter__()
        mlir_location.__enter__()
        self._mlir_block_stack.append(
            MLIRStackFrame(
                block,
                block_args,
                mlir_context,
                insertion_point,
                mlir_location,
                scope_name,
            )
        )
        return block, mlir_location

    def exit_mlir_block_scope(self, scope_name=None):
        if scope_name is None:
            scope_name = "UNNAMED_SCOPE"
        mlir_stack_frame = self._mlir_block_stack.pop()
        assert (
            mlir_stack_frame.scope_name == scope_name
        ), f"enter and exit in two different scopes {mlir_stack_frame.scope_name} {scope_name}"
        mlir_stack_frame.location.__exit__(None, None, None)
        mlir_stack_frame.insertion_point.__exit__(None, None, None)
        mlir_stack_frame.context.__exit__(None, None, None)
        return mlir_stack_frame.block

    @contextmanager
    def mlir_block_scope(
        self,
        block: ir.Block,
        *,
        block_args: tuple[str] = None,
        insertion_point: Optional[ir.InsertionPoint] = None,
        mlir_location: Optional[ir.Location] = None,
        scope_name: Optional[str] = None,
    ):
        if scope_name is None:
            scope_name = "UNNAMED_SCOPE"
        yield self.enter_mlir_block_scope(
            block,
            block_args=block_args,
            insertion_point=insertion_point,
            mlir_location=mlir_location,
            scope_name=scope_name,
        )
        self.exit_mlir_block_scope(scope_name=scope_name)

    def peek_block_scope(self):
        assert len(self._mlir_block_stack), "no block scope yet"
        return self._mlir_block_stack[-1]

    def byte_FOR_ITER(self, state, op):
        state = super().byte_FOR_ITER(state, op)
        iter_next_val_type = state.data_stack[-1].Data(None)
        assert (
            len(iter_next_val_type) == 1
        ), f"unrefined return val from iter not supported"
        store_location = op.next.pretty_arg
        # no need to even do anything special here since pytype naturally doesn't want to
        # execute the loop multiple times
        return state

    def byte_CALL_FUNCTION(self, state, op):
        num_args = op.arg
        funcv = state.peek(num_args + 1)
        filtered_data = funcv.FilteredData(state.node)
        assert len(filtered_data) == 1
        filtered_data = filtered_data[0]

        # TODO(max): 'builtins.builtins.print' != 'builtins.print'
        # assert filtered_data.full_name == filtered_data.name

        func_name = filtered_data.name
        print(f"call function {func_name} @ {op.line}")
        state = super().byte_CALL_FUNCTION(state, op)
        return state

    def byte_LOAD_ATTR(self, state, op):
        name = self.frame.f_code.co_names[op.arg]
        print(f"load attr {name} @ {op.line}")
        state = super().byte_LOAD_ATTR(state, op)
        return state

    def byte_LOAD_METHOD(self, state, op):
        # original_co_code = self.frame.f_code.original_co_code
        method_name = self.frame.f_code.co_names[op.arg]
        print(f"load method {method_name} @ {op.line}")
        state = super().byte_LOAD_METHOD(state, op)
        # see the impl in super
        self.loaded_method = (state.peek(1), method_name)
        return state

    def byte_CALL_METHOD(self, state, op):
        num_args = op.arg
        loaded_method = state.peek(num_args + 1)
        assert self.loaded_method[0] == loaded_method
        print(f"call method {self.loaded_method[1]} @ {op.line}")
        state = super().byte_CALL_METHOD(state, op)
        # TODO(max): num args? and then values should be in frame?
        return state

    def byte_MAKE_FUNCTION(self, state, op):
        # NB: this is a [-n] and -0 gets from the front
        func_name_var = state.peek(1)
        func_name = abstract_utils.get_atomic_python_constant(func_name_var)
        print(f"make function {func_name} @ {op.line}")

        func_op = func.FuncOp(
            name=func_name,
            type=((self.f64_type,) * 3, (self.f64_memref_type,)),
            visibility="public",
            loc=self.mlir_location,
        )
        func_op_entry_block = func_op.add_entry_block()
        self.enter_mlir_block_scope(func_op_entry_block, scope_name=func_name)

        state = super().byte_MAKE_FUNCTION(state, op)
        return state

    def byte_RETURN_VALUE(self, state, op):
        implicit_return = (
            op.name == "RETURN_VALUE" and op.line not in self._director.return_lines
        )
        print(f"return value {implicit_return=} @ {op.line}")
        if not implicit_return:
            print()

        if hasattr(self.frame, "func") and hasattr(self.frame.func, "data"):
            self.exit_mlir_block_scope(scope_name=self.frame.func.data.full_name)
        else:
            mlir_stack_frame = self.peek_block_scope()
            assert mlir_stack_frame.context == self.top_mlir_context
        self.trace_opcode(op, None, None)
        state = super().byte_RETURN_VALUE(state, op)
        return state

    def trace_opcode(self, op, symbol, val):
        super().trace_opcode(op, symbol, val)


def _MatchLoaderConfig(options, loader):
    """Match the |options| with the configuration of |loader|."""
    if not loader:
        return False
    assert isinstance(loader, load_pytd.Loader)
    if options.use_pickled_files != isinstance(loader, load_pytd.PickledPyiLoader):
        return False
    return options == loader.options


class ByteCodeCompiler:
    _loader: load_pytd.Loader
    python_version: tuple[int, int] = sys.version_info[:2]

    @property
    def loader(self):
        if not _MatchLoaderConfig(self.options, self._loader):
            # Create a new loader only if the configuration in the current options
            # does not match the configuration in the current loader.
            self._loader = load_pytd.create_loader(self.options)
        return self._loader

    @classmethod
    def set_up_class(cls):
        # We use class-wide loader to avoid creating a new loader for every test
        # method if not required.
        cls._loader = None

        def t(name):  # pylint: disable=invalid-name
            return pytd.ClassType("builtins." + name)

        cls.bool = t("bool")
        cls.dict = t("dict")
        cls.float = t("float")
        cls.complex = t("complex")
        cls.int = t("int")
        cls.list = t("list")
        cls.none_type = t("NoneType")
        cls.object = t("object")
        cls.set = t("set")
        cls.frozenset = t("frozenset")
        cls.str = t("str")
        cls.bytearray = t("bytearray")
        cls.tuple = t("tuple")
        cls.unicode = t("unicode")
        cls.generator = t("generator")
        cls.function = pytd.ClassType("typing.Callable")
        cls.anything = pytd.AnythingType()
        cls.nothing = pytd.NothingType()
        cls.module = t("module")
        cls.file = t("file")

        # The various union types use pytd_utils.CanonicalOrdering()'s ordering:
        cls.intorstr = pytd.UnionType((cls.int, cls.str))
        cls.strorunicode = pytd.UnionType((cls.str, cls.unicode))
        cls.intorfloat = pytd.UnionType((cls.float, cls.int))
        cls.intorfloatorstr = pytd.UnionType((cls.float, cls.int, cls.str))
        cls.complexorstr = pytd.UnionType((cls.complex, cls.str))
        cls.intorfloatorcomplex = pytd.UnionType((cls.int, cls.float, cls.complex))
        cls.int_tuple = pytd.GenericType(cls.tuple, (cls.int,))
        cls.nothing_tuple = pytd.TupleType(cls.tuple, ())
        cls.intorfloat_tuple = pytd.GenericType(cls.tuple, (cls.intorfloat,))
        cls.int_set = pytd.GenericType(cls.set, (cls.int,))
        cls.intorfloat_set = pytd.GenericType(cls.set, (cls.intorfloat,))
        cls.unknown_frozenset = pytd.GenericType(cls.frozenset, (cls.anything,))
        cls.float_frozenset = pytd.GenericType(cls.frozenset, (cls.float,))
        cls.empty_frozenset = pytd.GenericType(cls.frozenset, (cls.nothing,))
        cls.int_list = pytd.GenericType(cls.list, (cls.int,))
        cls.str_list = pytd.GenericType(cls.list, (cls.str,))
        cls.intorfloat_list = pytd.GenericType(cls.list, (cls.intorfloat,))
        cls.intorstr_list = pytd.GenericType(cls.list, (cls.intorstr,))
        cls.anything_list = pytd.GenericType(cls.list, (cls.anything,))
        cls.nothing_list = pytd.GenericType(cls.list, (cls.nothing,))
        cls.int_int_dict = pytd.GenericType(cls.dict, (cls.int, cls.int))
        cls.int_str_dict = pytd.GenericType(cls.dict, (cls.int, cls.str))
        cls.str_int_dict = pytd.GenericType(cls.dict, (cls.str, cls.int))
        cls.nothing_nothing_dict = pytd.GenericType(
            cls.dict, (cls.nothing, cls.nothing)
        )
        cls.make_tuple = lambda self, *args: pytd.TupleType(cls.tuple, tuple(args))

    def setUp(self):
        self.set_up_class()
        self.options = config.Options.create(
            python_version=self.python_version,
            always_use_return_annotations=True,
            enable_cached_property=True,
            overriding_default_value_checks=True,
            overriding_parameter_count_checks=True,
            overriding_parameter_name_checks=True,
            overriding_return_type_checks=True,
            strict_parameter_checks=True,
            strict_primitive_comparisons=True,
            use_enum_overlay=True,
        )

    def __init__(
        self,
        mlir_context: ir.Context = None,
        mlir_location: ir.Location = None,
        mlir_module: ir.Module = None,
    ):
        self.setUp()
        self.ctx = context.Context(options=self.options, loader=self.loader)
        if mlir_context is None:
            self.top_mlir_context = ir.Context()
        else:
            self.top_mlir_context = mlir_context

        if mlir_location is None:
            self.mlir_location = ir.Location.unknown(context=self.top_mlir_context)
        else:
            self.mlir_location = mlir_location

        if mlir_module is None:
            with self.top_mlir_context, self.mlir_location:
                self.mlir_module = ir.Module.create(loc=self.mlir_location)
        else:
            self.mlir_module = mlir_module
        self.ctx.vm = MyCallTracer(
            self.ctx,
            module=self.mlir_module,
            top_context=self.top_mlir_context,
            location=self.mlir_location,
        )

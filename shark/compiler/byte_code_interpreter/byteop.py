import inspect
import logging
import operator
import sys
import types
from typing import Any, Callable

from shark.compiler.byte_code_interpreter.errors import PyVMError
from shark.compiler.byte_code_interpreter.pyobj import (
    Function,
    traceback_from_frame,
    Generator,
    COMPREHENSION_FN_NAMES,
)
from shark.compiler.byte_code_interpreter.vmtrace import (
    PyVMEVENT_RETURN,
    PyVMEVENT_YIELD,
)
from shark.dialects import func as func_dialect, scf, affine_
from shark import ir
from shark.dialects._ods_common import get_op_result_or_value

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def identity(x):
    return x


FSTRING_CONVERSION_MAP = {0: identity, 1: str, 2: repr}


MAKE_FUNCTION_SLOT_NAMES = ("closure", "annotations", "kwdefaults", "defaults")
MAKE_FUNCTION_SLOTS = len(MAKE_FUNCTION_SLOT_NAMES)


def fmt_binary_op(vm, arg=None, repr=repr):
    """returns a string of the repr() for each of the the first two
    elements of evaluation stack

    """
    return " (%s, %s)" % (repr(vm.peek(2)), repr(vm.top()))


def fmt_ternary_op(vm, arg=None, repr=repr):
    """returns string of the repr() for each of the first three
    elements of evaluation stack
    """
    return " (%s, %s, %s)" % (repr(vm.peek(3)), repr(vm.peek(2)), repr(vm.top()))


def fmt_unary_op(vm, arg=None, repr=repr):
    """returns string of the repr() for the first element of
    the evaluation stack
    """
    # We need to check the length because sometimes in a return event
    # (as opposed to a
    # a RETURN_VALUE callback can* the value has been popped, and if the
    # return valuse was the only one on the stack, it will be empty here.
    if len(vm.frame.stack):
        return " (%s)" % (repr(vm.top()),)
    else:
        raise PyVMError("Empty stack in unary op")


class ByteOp:
    def __init__(self, vm):
        self.vm = vm
        # This is used in `vm.format_instruction()` to pick out stack elements
        # to better show operand(s) of opcode.
        self.stack_fmt = {}
        # Set this lazily in "convert_method_native_func
        self.method_func_access = None
        self.create_op_handlers()

    def create_op_handlers(self):
        unary_operators = {
            "POSITIVE": operator.pos,
            "NEGATIVE": operator.neg,
            "NOT": operator.not_,
            "CONVERT": repr,
            "INVERT": operator.invert,
        }
        for op_suf, impl in unary_operators.items():
            op_name = "UNARY_" + op_suf
            self.stack_fmt[op_name] = fmt_unary_op

            def unary_operator(impl=impl):
                # TODO(max): fix - use correct way to bind classmethod
                x = self.vm.pop()
                if not isinstance(x, ir.Type):
                    if isinstance(x, (float, int)):
                        x = self.vm.get_or_make_mlir_constant(x)
                self.vm.push(impl(x))

            setattr(self, op_name, unary_operator)

        binary_and_inplace_operators = {
            "ADD": self.vm.body_builder.binary_add,
            "AND": operator.and_,
            "DIVIDE": getattr(operator, "div", lambda x, y: x / y),
            "FLOOR_DIVIDE": operator.floordiv,
            "LSHIFT": operator.lshift,
            "MATRIX_MULTIPLY": operator.matmul,
            "MODULO": operator.mod,
            "MULTIPLY": self.vm.body_builder.binary_mul,
            "OR": operator.or_,
            "POWER": pow,
            "RSHIFT": operator.rshift,
            # TODO(max): this is a hack but hey :shrug:
            # "SUBSCR": self.vm.body_builder.memref_load,
            "SUBTRACT": operator.sub,
            "TRUE_DIVIDE": operator.truediv,
            "XOR": operator.xor,
        }
        for op_suf, impl in binary_and_inplace_operators.items():
            inplace_op_name = "INPLACE_" + op_suf
            binary_op_name = "BINARY_" + op_suf
            self.stack_fmt[inplace_op_name] = fmt_binary_op
            self.stack_fmt[binary_op_name] = fmt_binary_op

            def _operator(impl=impl):
                # TODO(max): fix - use correct way to bind classmethod
                x, y = self.vm.popn(2)
                if not isinstance(x, ir.Type):
                    if isinstance(x, (float, int)):
                        x = self.vm.get_or_make_mlir_constant(x)
                if not isinstance(y, ir.Type):
                    if isinstance(y, (float, int)):
                        y = self.vm.get_or_make_mlir_constant(y)
                # TODO(max): store/load context for subscr?
                # maybe that's only AST?
                # TODO(max): technically this isn't the correct semantics for
                # inplace ops (which should push x) but that's not allowed anyway
                self.vm.push(impl(x, y))

            setattr(self, inplace_op_name, _operator)
            setattr(self, binary_op_name, _operator)

        self.stack_fmt["BINARY_SUBSCR"] = fmt_binary_op

        self.compare_operators = [
            operator.lt,  # <
            operator.le,  # <=
            operator.eq,  # ==
            operator.ne,  # !=
            self.vm.body_builder.compare_gt,  # >
            operator.ge,  # >=
            lambda x, y: x in y,
            lambda x, y: x not in y,
            lambda x, y: x is y,
            lambda x, y: x is not y,
            lambda x, y: issubclass(x, BaseException)
            and issubclass(x, y),  # exception-match
        ]

    def slice_operator(self, op):
        start = 0
        end = None  # we will take this to mean end
        op, count = op[:-2], int(op[-1])
        if count == 1:
            start = self.vm.pop()
        elif count == 2:
            end = self.vm.pop()
        elif count == 3:
            end = self.vm.pop()
            start = self.vm.pop()
        l = self.vm.pop()
        if end is None:
            end = len(l)
        if op.startswith("STORE_"):
            l[start:end] = self.vm.pop()
        elif op.startswith("DELETE_"):
            del l[start:end]
        else:
            self.vm.push(l[start:end])

    def build_container(self, count, container_fn):
        # TODO(max): handle other container types?
        elts = self.vm.popn(count)
        self.vm.push(container_fn(elts))

    def call_function_with_args_resolved(self, func, pos_args, named_args):
        assert (
            not named_args
        ), f"named_args not supported for {func.func_name} @ {func.func_code.co_firstlineno}"
        frame = self.vm.frame
        if hasattr(func, "im_func"):
            # Methods get self as an implicit first parameter.
            if func.im_self is not None:
                pos_args.insert(0, func.im_self)
            # The first parameter must be the correct type.
            if not isinstance(pos_args[0], func.im_class):
                raise TypeError(
                    "unbound method %s() must be called with %s instance "
                    "as first argument (got %s instance instead)"
                    % (
                        func.im_func.func_name,
                        func.im_class.__name__,
                        type(pos_args[0]).__name__,
                    )
                )
            func = func.im_func

        # FIXME: put this in a separate routine.
        if inspect.isbuiltin(func):
            log.debug("handling built-in function %s" % func.__name__)
            if func == globals:
                # Use the frame's globals(), not the interpreter's
                self.vm.push(frame.f_globals)
                return
            elif func == locals:
                # Use the frame's locals(), not the interpreter's
                self.vm.push(frame.f_globals)
                return
            elif func == compile:
                # Set dont_inherit parameter.
                # FIXME: we should set other flags too based on the interpreted environment?
                if len(pos_args) < 5 and "dont_inherit" not in named_args:
                    named_args["dont_inherit"] = True
                    pass
            # In Python 3.0 or greater, "exec()" is a builtin.  In
            # Python 2.7 it was an opcode EXEC_STMT and is not a
            # built-in function.
            #
            # FIXME: a better test would be nice. There can be
            # other builtin "exec"s. Tk has a built-in "eval". See 3.6.10
            # test_tcl.py.
            # If we drop the requirement of supporting 2.7 we can do the simpler
            # and more reliable:
            #   func == exec
            elif func.__name__ == "exec":
                if not 1 <= len(pos_args) <= 3:
                    raise PyVMError(
                        "exec() builtin should have 1..3 positional arguments; got %d"
                        % (len(pos_args))
                    )
                n = len(pos_args)
                assert 1 <= n <= 3

                # Note that in contrast to `eval()` handled below, if
                # the `locals` parameter is not provided, the
                # `globals` parameter value (whether provided or
                # default value) is used for the `locals`
                # parameter. So we shouldn't use the frame's `locals`.
                if len(pos_args) == 1:
                    pos_args.append(self.vm.frame.f_globals)

                source = pos_args[0]
                if isinstance(source, str) or isinstance(source, bytes):
                    try:
                        pos_args[0] = compile(
                            source, "<string>", mode="exec", dont_inherit=True
                        )
                    except (TypeError, SyntaxError, ValueError):
                        raise
                self.vm.push(self.vm.run_code(*pos_args, toplevel=False))
                return
            elif func == eval:
                if not 1 <= len(pos_args) <= 3:
                    raise PyVMError(
                        "eval() builtin should have 1..3 positional arguments; got %d"
                        % (len(pos_args))
                    )
                assert 1 <= len(pos_args) <= 3
                # Use the frame's globals(), not the interpreter's
                n = len(pos_args)
                if n < 2:
                    pos_args.append(self.vm.frame.f_globals)
                # Likewise for locals()
                if n < 3:
                    pos_args.append(self.vm.frame.f_locals)
                assert len(pos_args) == 3

                source = pos_args[0]
                if isinstance(source, str):
                    try:
                        pos_args[0] = compile(
                            source, "<string>", mode="eval", dont_inherit=True
                        )
                    except (TypeError, SyntaxError, ValueError):
                        raise
                self.vm.push(self.vm.run_code(*pos_args, toplevel=False))
                return
        elif func == type and len(pos_args) == 3:
            # Set __module__
            assert not named_args
            namespace = pos_args[2]
            namespace["__module__"] = namespace.get(
                "__name__", self.vm.frame.f_globals["__name__"]
            )

        if inspect.isfunction(func):
            # Try to convert to an interpreter function so we can interpret it.
            func = self.vm.fn2native[func]

        if inspect.isfunction(func):
            log.debug("calling native function %s" % func.__name__)

        if func in self.vm.fn2native_mlir:
            func_op = self.vm.fn2native_mlir[func]
            func_op_entry_block = func_op.add_entry_block()
            with self.vm.mlir_block_scope(func_op_entry_block):
                retval = func(*list(func_op.arguments), **named_args)
                mlir_return_val = get_op_result_or_value(retval)
                func_type = func_op.type
                canonical_func_type = ir.FunctionType.get(
                    inputs=func_type.inputs, results=[mlir_return_val.type]
                )
                func_op.attributes["function_type"] = ir.TypeAttr.get(
                    canonical_func_type
                )
                func_dialect.ReturnOp((mlir_return_val,))
        else:
            retval = func(*pos_args, **named_args)

        self.vm.push(retval)

    def call_function(self, argc: int, var_args, keyword_args=None) -> Any:
        if keyword_args is None:
            keyword_args = {}
        named_args = {}
        len_kw, len_pos = divmod(argc, 256)
        for i in range(len_kw):
            key, val = self.vm.popn(2)
            named_args[key] = val
        named_args.update(keyword_args)
        pos_args = self.vm.popn(len_pos)
        pos_args.extend(var_args)

        func = self.vm.pop()
        return self.call_function_with_args_resolved(func, pos_args, named_args)

    def convert_native_to_function(self, frame, func: Callable) -> Callable:
        assert inspect.isfunction(func) or isinstance(func, Function)
        slots = {"kwdefaults": {}, "annotations": {}}
        if self.vm.version >= (3, 0):
            slots["globs"] = frame.f_globals
            arg2attr = {
                "code": "__code__",
                "name": "__name__",
                "argdefs": "__defaults__",
                "kwdefaults": "__kwdefaults__",
                "annotations": "__annotations__",
                "closure": "__closure__",
                # FIXME: add __qualname__, __doc__
                # and __module__
            }
        else:
            slots["kwdefaults"] = {}
            slots["annotations"] = {}
            arg2attr = {
                "code": "func_code",
                "name": "__name__",
                "argdefs": "func_defaults",
                "globs": "func_globals",
                "annotations": "doesn't exist",
                "closure": "func_closure",
                # FIXME: add __doc__
                # and __module__
            }

        for argname, attribute in arg2attr.items():
            if hasattr(func, attribute):
                slots[argname] = getattr(func, attribute)

        closure = getattr(func, arg2attr["closure"])
        if not closure:
            # FIXME: we don't know how to convert functions with closures yet.
            native_func = func

            func = Function(
                slots["name"],
                slots["code"],
                slots["globs"],
                slots["argdefs"],
                slots["closure"],
                self.vm,
                slots["kwdefaults"],
                slots["annotations"],
            )
            self.vm.fn2native[native_func] = func
        return func

    def do_raise(self, exc, cause):
        if exc is None:  # reraise
            exc_type, val, tb = self.vm.last_exception
            if exc_type is None:
                return "exception"  # error
            else:
                return "reraise"

        elif type(exc) == type:
            # As in `raise ValueError`
            exc_type = exc
            val = exc()  # Make an instance.
        elif isinstance(exc, BaseException):
            # As in `raise ValueError('foo')`
            exc_type = type(exc)
            val = exc
        else:
            return "exception"  # error

        # If you reach this point, you're guaranteed that
        # val is a valid exception instance and exc_type is its class.
        # Now do a similar thing for the cause, if present.
        if cause:
            if type(cause) == type:
                cause = cause()
            elif not isinstance(cause, BaseException):
                return "exception"  # error

            val.__cause__ = cause

        self.vm.last_exception = exc_type, val, val.__traceback__
        return "exception"

    def lookup_name(self, name):
        """Returns the value in the current frame associated for name"""
        frame = self.vm.frame
        if name in frame.f_locals:
            val = frame.f_locals[name]
        elif name in frame.f_globals:
            val = frame.f_globals[name]
        elif name in frame.f_builtins:
            val = frame.f_builtins[name]
        else:
            raise NameError("name '%s' is not defined" % name)
        return val

    def print_item(self, item, to=None):
        if to is None:
            to = sys.stdout

        # Python 2ish has file.softspace whereas
        # Python 3ish doesn't. Here is the doc on softspace:

        # Boolean that indicates whether a space character needs to be
        # printed before another value when using the print
        # statement. Classes that are trying to simulate a file object
        # should also have a writable softspace attribute, which
        # should be initialized to zero. This will be automatic for
        # most classes implemented in Python (care may be needed for
        # objects that override attribute access); types implemented
        # in C will have to provide a writable softspace attribute.

        # Note This attribute is not used to control the print
        # statement, but to allow the implementation of print to keep
        # track of its internal state.
        if hasattr(to, "softspace") and to.softspace:
            print(" ", end="", file=to)
            to.softspace = 0
        print(item, end="", file=to)

        if hasattr(to, "softspace"):
            if isinstance(item, str):
                if (not item) or (not item[-1].isspace()) or (item[-1] == " "):
                    to.softspace = 1
            else:
                to.softspace = 1

    def print_newline(self, to=None):
        if to is None:
            to = sys.stdout
        print("", file=to)
        if hasattr(to, "softspace"):
            to.softspace = 0

    def BEFORE_ASYNC_WITH(self):
        raise PyVMError("BEFORE_ASYNC_WITH not implemented yet")

    def BEGIN_FINALLY(self):
        """Pushes NULL onto the stack for using it in END_FINALLY, POP_FINALLY, WITH_CLEANUP_START and WITH_CLEANUP_FINISH. Starts the finally block."""
        self.vm.push(None)

    def BINARY_SUBSCR(self):
        x, y = self.vm.popn(2)
        assert all(not isinstance(v, slice) for v in y), "slicing not supported yet"
        # TODO(max): store/load context for subscr?
        # maybe that's only AST?
        # TODO(max): technically this isn't the correct semantics for
        # inplace ops (which should push x) but that's not allowed anyway
        self.vm.push(self.vm.body_builder.memref_load(x, y))

    def BRKPT(self):
        """Psuedo opcode: breakpoint. We added this. TODO: call callback, then run
        instruction that should have gotten run.
        """
        vm = self.vm
        frame = vm.frame
        last_i = frame.f_lasti
        orig_opcode = frame.brkpt[last_i]
        orig_opname = vm.opc.opname[orig_opcode]
        log.info("Breakpoint at offset %d instruction %s" % (last_i, orig_opname))
        (
            byte_name,
            byte_code,
            int_arg,
            arguments,
            opoffset,
            line_number,
        ) = vm.parse_byte_and_args(orig_opcode, replay=True)

        if vm.callback:
            result = vm.callback(
                "breakpoint", last_i, byte_name, byte_code, line_number, None, [], vm
            )

            # FIXME: DRY with vmtrace code
            if result:
                if result == "finish":
                    frame.f_trace = None
                    frame.event_flags = PyVMEVENT_RETURN | PyVMEVENT_YIELD
                elif result == "return":
                    # Immediate return with value
                    return self.vm.return_value
                elif result == "skip":
                    # Don't run instruction
                    return result

        if log.isEnabledFor(logging.INFO):
            vm.log(byte_name, int_arg, arguments, opoffset, line_number)
        return vm.dispatch(byte_name, int_arg, arguments, opoffset, line_number)

    def BUILD_CONST_KEY_MAP(self, count):
        """
        The version of BUILD_MAP specialized for constant keys. count
        values are consumed from the stack. The top element on the
        stack contains a tuple of keys.
        """
        keys = self.vm.pop()
        values = self.vm.popn(count)
        kvs = dict(zip(keys, values))
        self.vm.push(kvs)

    def BUILD_LIST(self, count: int):
        """Works as BUILD_TUPLE, but creates a list."""
        elts = self.vm.popn(count)
        self.vm.push(elts)

    def BUILD_MAP(self, count):
        """
        Pushes a new dictionary object onto the stack. Pops 2 * count
        items so that the dictionary holds count entries: {..., TOS3:
        TOS2, TOS1: TOS}.

        Changed in version 3.5: The dictionary is created from stack
        items instead of creating an empty dictionary pre-sized to
        hold count items.
        """
        kvs = self.vm.popn(count * 2)
        self.vm.push(dict(kvs[i : i + 2] for i in range(0, len(kvs), 2)))

    def BUILD_MAP_UNPACK_WITH_CALL(self, oparg):
        """
        This is similar to BUILD_MAP_UNPACK, but is used for f(**x, **y,
        **z) call syntax. The lowest byte of oparg is the count of
        mappings, the relative position of the corresponding callable
        f is encoded in the second byte of oparg.
        """
        # In 3.5 fn_pos may be always 1 which meant the stack
        # entry after the mappings. In 3.6 this function-position
        # encoding was dropped. But we'll follow the spec.
        fn_pos, count = divmod(oparg, 256)
        fn_pos -= 1

        elts = self.vm.popn(count)
        if elts:
            kwargs = {k: v for m in elts for k, v in m.items()}
        else:
            kwargs = None
        func = self.vm.pop(fn_pos)

        # Put everything in the right order for CALL_FUNCTION_KW
        self.vm.push(func)
        if kwargs:
            self.vm.push(kwargs)

    def BUILD_SET(self, count):
        """Works as BUILD_TUPLE, but creates a set. New in version 2.7"""
        elts = self.vm.popn(count)
        self.vm.push(set(elts))

    def BUILD_SLICE(self, count):
        """
        Pushes a slice object on the stack. argc must be 2 or 3. If it is
        2, slice(TOS1, TOS) is pushed; if it is 3, slice(TOS2, TOS1,
        TOS) is pushed. See the slice() built-in function for more
        information.
        """
        if count == 2:
            x, y = self.vm.popn(2)
            self.vm.push(slice(x, y))
        elif count == 3:
            x, y, z = self.vm.popn(3)
            self.vm.push(slice(x, y, z))
        else:  # pragma: no cover
            raise PyVMError("Strange BUILD_SLICE count: %r" % count)

    def BUILD_STRING(self, count):
        """
        The version of BUILD_MAP specialized for constant keys. count
        values are consumed from the stack. The top element on the
        stack contains a tuple of keys.
        """
        assert isinstance(count, int) and count >= 0
        values = self.vm.popn(count)
        self.vm.push("".join(values))

    def BUILD_TUPLE(self, count: int):
        """Creates a tuple consuming count items from the stack, and pushes
        the resulting tuple onto the stack.
        """
        self.build_container(count, tuple)

    def BUILD_TUPLE_UNPACK_WITH_CALL(self, count):
        """
        This is similar to BUILD_TUPLE_UNPACK, but is used for f(*x, *y,*z)
        call syntax. The stack item at position count + 1 should be the
        corresponding callable f.
        """
        assert isinstance(count, int) and count >= 0
        parameter_tuples = self.vm.popn(count)
        parameters = [
            parameter for sublist in parameter_tuples for parameter in sublist
        ]
        self.vm.push(parameters)

    def CALL_FUNCTION(self, argc: int):
        """
        Calls a callable object.
        The low byte of argc indicates the number of positional
        arguments, the high byte the number of keyword arguments.

        The stack contains keyword arguments on top (if any), then the
        positional arguments below that (if any), then the callable
        object to call below that.

        Each keyword argument is represented with two values on the
        stack: the argument's name, and its value, with the argument's
        value above the name on the stack. The positional arguments
        are pushed in the order that they are passed in to the
        callable object, with the right-most positional argument on
        top. CALL_FUNCTION pops all arguments and the callable object
        off the stack, calls the callable object with those arguments,
        and pushes the return value returned by the callable object.
        """
        # TODO(max): load vars/inputs args here
        try:
            return self.call_function(argc, var_args=[], keyword_args={})
        except TypeError as exc:
            tb = self.vm.last_traceback = traceback_from_frame(self.vm.frame)
            self.vm.last_exception = (TypeError, exc.args, tb)
            return "exception"

    def CALL_FUNCTION_EX(self, flags):
        """
        Calls a callable object with variable set of positional and
        keyword arguments. If the lowest bit of flags is set, the top
        of the stack contains a mapping object containing additional
        keyword arguments. Below that is an iterable object containing
        positional arguments and a callable object to
        call. BUILD_MAP_UNPACK_WITH_CALL and
        BUILD_TUPLE_UNPACK_WITH_CALL can be used for merging multiple
        mapping objects and iterables containing arguments. Before the
        callable is called, the mapping object and iterable object are
        each  unpacked  and their contents passed in as keyword and
        positional arguments respectively. CALL_FUNCTION_EX pops all
        arguments and the callable object off the stack, calls the
        callable object with those arguments, and pushes the return
        value returned by the callable object.
        """
        assert isinstance(flags, int)
        namedargs = self.vm.pop() if flags & 1 else {}
        posargs = self.vm.pop()
        func = self.vm.pop()
        self.call_function_with_args_resolved(func, posargs, namedargs)

    def call_function_kw(self, argc: int):
        namedargs = {}
        namedargs_tup = self.vm.pop()
        for name in reversed(namedargs_tup):
            namedargs[name] = self.vm.pop()

        lenPos = argc - len(namedargs_tup)
        posargs = self.vm.popn(lenPos)
        func = self.vm.pop()
        self.call_function_with_args_resolved(func, posargs, namedargs)

    def CALL_FUNCTION_KW(self, argc: int):
        """
        Calls a callable object with positional (if any) and keyword
        arguments.

        argc indicates the total number of positional and
        keyword arguments. The top element on the stack contains a tuple
        of keyword argument names. Below that are keyword arguments in
        the order corresponding to the tuple. Below that are positional
        arguments, with the right-most parameter on top. Below the
        arguments is a callable object to call. CALL_FUNCTION_KW pops
        all arguments and the callable object off the stack, calls the
        callable object with those arguments, and pushes the return
        value returned by the callable object.

        Changed in version 3.6: Keyword arguments are packed in a tuple
        instead of a dictionary, argc indicates the total number of
        arguments.
        """
        return self.call_function_kw(argc)

    def CALL_FUNCTION_VAR(self, argc: int):
        """Calls a callable object, similarly to `CALL_FUNCTION_VAR` and
        `CALL_FUNCTION_KW`. *argc* represents the number of keyword
        and positional arguments, identically to `CALL_FUNCTION`. The
        top of the stack contains a mapping object, as per
        `CALL_FUNCTION_KW`. Below that are keyword arguments (if any),
        stored identically to `CALL_FUNCTION`. Below that is an iterable
        object containing additional positional arguments. Below that
        are positional arguments (if any) and a callable object,
        identically to `CALL_FUNCTION`a. Before the callable is called,
        the mapping object and iterable object are each "unpacked" and
        their contents passed in as keyword and positional arguments
        respectively, identically to `CALL_FUNCTION_VAR` and
        `CALL_FUNCTION_KW`. The mapping object and iterable object are
        both ignored when computing the value of argc.

        Changed in version 3.5: In all Python versions 3.4, the
        iterable object (var_args) was above the keyword arguments
        (keyword_args); in 3.5 the iterable object was moved below the
        keyword arguments.

        """
        keyword_args = {}
        len_kw, len_pos = divmod(argc, 256)
        for i in range(len_kw):
            key, val = self.vm.popn(2)
            keyword_args[key] = val
        var_args = self.vm.pop()
        pos_args = self.vm.popn(len_pos)
        pos_args.extend(var_args)
        func = self.vm.pop()
        self.call_function_with_args_resolved(
            func, pos_args=pos_args, named_args=keyword_args
        )

    def CALL_METHOD(self, count):
        """Calls a method. argc is the number of positional
        arguments. Keyword arguments are not supported. This opcode is
        designed to be used with LOAD_METHOD. Positional arguments are
        on top of the stack. Below them, the two items described in
        LOAD_METHOD are on the stack (either self and an unbound
        method object or NULL and an arbitrary callable). All of them
        are popped and the return value is pushed.

        rocky: In our setting, before "self" we have an additional
        item which is the status of the LOAD_METHOD. There is no way
        in Python to represent a value outside of a Python value which
        you can do in C, and is in effect what NULL is.
        """
        posargs = self.vm.popn(count)
        is_success = self.vm.pop()
        if is_success:
            func = self.vm.pop()
            self.call_function_with_args_resolved(func, posargs, {})
        else:
            # FIXME: do something else
            raise PyVMError("CALL_METHOD malfunctioned")

    def COMPARE_OP(self, opname):
        """Performs a Boolean operation. The operation name can be found in cmp_op[opname]."""
        x, y = self.vm.popn(2)
        if not isinstance(x, ir.Type):
            if isinstance(x, (float, int)):
                x = self.vm.get_or_make_mlir_constant(x)
        if not isinstance(y, ir.Type):
            if isinstance(y, (float, int)):
                y = self.vm.get_or_make_mlir_constant(y)
        self.vm.push(self.compare_operators[opname](x, y))

    def CONTAINS_OP(self, invert: int):
        """Performs in comparison, or not in if invert is 1."""
        TOS1, TOS = self.vm.popn(2)
        if invert:
            self.vm.push(TOS1 not in TOS)
        else:
            self.vm.push(TOS1 in TOS)
        return

    def COPY_DICT_WITHOUT_KEYS(self):
        """TOS is a tuple of mapping keys, and TOS1 is the match
        subject. Replace TOS with a dict formed from the items of TOS1, but
        without any of the keys in TOS."""
        # FIXME
        raise PyVMError("MATCH_COPY_DICT_WITHOUT_KEYS not implemented")

    def DELETE_ATTR(self, name):
        """Implements del TOS.name, using namei as index into co_names."""
        obj = self.vm.pop()
        delattr(obj, name)

    def DELETE_FAST(self, var_num):
        """Deletes local co_varnames[var_num]."""
        del self.vm.frame.f_locals[var_num]

    def DELETE_GLOBAL(self, name):
        """Implements del name, where name in global."""
        del self.vm.frame.f_globals[name]

    def DELETE_NAME(self, name):
        """Implements del name, where name is the index into co_names attribute of the code object."""
        del self.vm.frame.f_locals[name]

    def DELETE_SUBSCR(self):
        """Implements del TOS1[TOS]."""
        obj, subscr = self.vm.popn(2)
        del obj[subscr]

    def DICT_MERGE(self, i):
        """Like DICT_UPDATE but raises an exception for duplicate keys."""
        TOS = self.vm.pop()
        assert isinstance(TOS, dict)
        destination = self.vm.peek(i)
        assert isinstance(destination, dict)
        dups = set(destination.keys()) & set(TOS.keys())
        if bool(dups):
            raise RuntimeError("Duplicate keys '%s' in DICT_MERGE" % dups)
        destination.update(TOS)

    def DICT_UPDATE(self, i):
        """Calls dict.update(TOS1[-i], TOS). Used to build dicts."""
        TOS = self.vm.pop()
        assert isinstance(TOS, dict)
        destination = self.vm.peek(i)
        assert isinstance(destination, dict)
        destination.update(TOS)

    def DUP_TOP(self):
        """Duplicates the reference on top of the stack."""
        self.vm.push(self.vm.top())

    def DUP_TOP_TWO(self):
        """Duplicates the reference on top of the stack."""
        a, b = self.vm.popn(2)
        self.vm.push(a, b, a, b)

    def END_ASYNC_FOR(self):
        """Terminates an `async for1 loop. Handles an exception raised when
        awaiting a next item. If TOS is StopAsyncIteration pop 7 values from
        the stack and restore the exception state using the second three of
        them. Otherwise re-raise the exception using the three values from the
        stack. An exception handler block is removed from the block stack."""

        raise PyVMError("END_ASYNC_FOR not implemented yet")

    def END_FINALLY(self):
        """Terminates a finally clause. The interpreter recalls whether the
        exception has to be re-raised or execution has to be continued
        depending on the value of TOS.

        * If TOS is NULL (pushed by BEGIN_FINALLY) continue from the next instruction.
          TOS is popped.

        * If TOS is an integer (pushed by CALL_FINALLY), sets the bytecode counter to TOS.
          TOS is popped.

        * If TOS is an exception type (pushed when an exception has
          been raised) 6 values are popped from the stack, the first
          three popped values are used to re-raise the exception and
          the last three popped values are used to restore the
          exception state. An exception handler block is removed from
          the block stack.
        """
        v = self.vm.pop()
        if v is None:
            why = None
        elif isinstance(v, int):
            self.vm.jump(v)
            why = "return"
        elif issubclass(v, BaseException):
            # from trepan.api import debug; debug()
            exctype = v
            val = self.vm.pop()
            tb = self.vm.pop()
            self.vm.last_exception = (exctype, val, tb)

            raise PyVMError("END_FINALLY not finished yet")
            # FIXME: pop 3 more values
            # why = "reraise"
        else:  # pragma: no cover
            raise PyVMError("Confused END_FINALLY")
        return why

    def FORMAT_VALUE(self, flags):
        """Used for implementing formatted literal strings (f-strings). Pops
        an optional fmt_spec from the stack, then a required value. flags is
        interpreted as follows:

        * (flags & 0x03) == 0x00: value is formatted as-is.
        * (flags & 0x03) == 0x01: call str() on value before formatting it.
        * (flags & 0x03) == 0x02: call repr() on value before formatting it.
        * (flags & 0x03) == 0x03: call ascii() on value before formatting it.
        * (flags & 0x04) == 0x04: pop fmt_spec from the stack and use it, else use an empty fmt_spec.

        Formatting is performed using PyObject_Format(). The result is
        pushed on the stack.
        """
        assert isinstance(flags, int)
        if flags & 0x04 == 0x04:
            format_spec = self.vm.pop()
        else:
            format_spec = ""

        value = self.vm.pop()
        attr_flags = flags & 0x03
        if attr_flags:
            value = FSTRING_CONVERSION_MAP.get(attr_flags, identity)(value)

        result = format(value, format_spec)
        self.vm.push(result)

    def FOR_ITER(self, jump_offset):
        """
        TOS is an iterator. Call its next() method. If this yields a new
        value, push it on the stack (leaving the iterator below
        it). If the iterator indicates it is exhausted TOS is popped,
        and the bytecode counter is incremented by delta.

        Note: jump = delta + f.f_lasti set in parse_byte_and_args()
        """

        # TODO(max): here is where you would short-circuit evaluation of several iterations
        # by removing exception catch and just doing that
        # TODO(max): this is handled automatically right now since in GET_ITER we pass a list_iter with only
        # the induction variable

        iterobj = self.vm.top()
        try:
            v = next(iterobj)
            self.vm.push(v)
        except StopIteration:
            self.vm.pop()
            # TODO(max): handle yielded vals
            yielded_vals = []
            if self.vm.scf_fors:
                scf.YieldOp(yielded_vals, loc=self.vm.mlir_location)
            else:
                affine_.AffineYieldOp(yielded_vals, loc=self.vm.mlir_location)
            self.vm.exit_mlir_block_scope()
            self.vm.jump(jump_offset)

    def GEN_START(self, kind):
        """Pops TOS. If TOS was not None, raises an exception. The kind
        operand corresponds to the type of generator or coroutine and
        determines the error message. The legal kinds are 0 for
        generator, 1 for coroutine, and 2 for async generator.
        """
        generator = self.vm.pop()
        # if generator is None:
        #     raise self.vm.PyVMError("GEN_START TOS is None")
        # FIXME
        assert kind in (0, 1, None)

    def GET_AITER(self):
        """
        Implements TOS = get_awaitable(TOS.__aiter__()). See GET_AWAITABLE
        for details about get_awaitable
        """
        # raise self.vm.PyVMError("GET_AITER not implemented yet")
        anext_fn = getattr(self.vm.pop(), "__aiter__")
        return self.call_function(anext_fn, [])

    def GET_ANEXT(self):
        """
        Implements PUSH(get_awaitable(TOS.__anext__())). See GET_AWAITABLE
        for details about get_awaitable
        """
        # raise self.vm.PyVMError("GET_ANEXT not implemented yet")
        anext_fn = getattr(self.vm.pop(), "__anext__")
        return self.call_function(anext_fn, [])

    def GET_AWAITABLE(self):
        """
        Implements TOS = get_awaitable(TOS), where get_awaitable(o)
        returns o if o is a coroutine object or a generator object
        with the CO_ITERABLE_COROUTINE flag, or resolves
        o.__await__.
        """
        raise PyVMError("GET_AWAITABLE not implemented yet")

    def GET_ITER(self):
        """Implements TOS = iter(TOS)."""
        range_iter = self.vm.peek(1)

        if not isinstance(range_iter, range):
            raise RuntimeError("Only `range` iterator currently supported")
        for arg in [range_iter.start, range_iter.stop, range_iter.step]:
            assert isinstance(arg, int), f"symbolic range not supported yet {arg}"

        start, stop, step = range_iter.start, range_iter.stop, range_iter.step
        if self.vm.scf_fors:
            start, stop, step = [
                self.vm.get_or_make_mlir_constant(c, index_type=self.vm.scf_fors)
                for c in [start, stop, step]
            ]
            loop = scf.ForOp(start, stop, step, [], loc=self.vm.mlir_location)
        else:
            loop = affine_.AffineForOp(start, stop, step, loc=self.vm.mlir_location)

        self.vm.enter_mlir_block_scope(loop.body)
        _tos = self.vm.pop()
        # TODO(max): this is perfect but maybe too perfect
        self.vm.push(iter([loop.induction_variable]))

    def GET_LEN(self):
        """Push len(TOS) onto the stack."""
        self.vm.push(len(self.vm.pop()))

    def GET_YIELD_FROM_ITER(self):
        """
        If TOS is a generator iterator or coroutine object it is left as
        is. Otherwise, implements TOS = iter(TOS).
        """
        TOS = self.vm.top()
        if inspect.isgeneratorfunction(TOS) or inspect.iscoroutinefunction(TOS):
            return
        TOS = self.vm.pop()
        self.vm.push(iter(TOS))

    def IMPORT_FROM(self, name):
        """
        Loads the attribute co_names[namei] from the module found in TOS.
        The resulting object is pushed onto the stack, to be
        subsequently stored by a STORE_FAST instruction.

        Note: name = co_names[namei] set in parse_byte_and_args()
        """
        mod = self.vm.top()
        if not hasattr(mod, name):
            if not hasattr(mod, "__file__"):
                # Builtins don't have a __file__ attribute
                value = ImportError(f"cannot import name '{name}' from '{mod.__name__}")
            else:
                value = ImportError(
                    f"cannot import name '{name}' from '{mod.__name__} ({mod.__file__}"
                )

            self.vm.last_exception = (ImportError, value, None)
            return "exception"

        self.vm.push(getattr(mod, name))

    def IMPORT_NAME(self, name):
        """
        Imports the module co_names[namei]. TOS and TOS1 are popped and
        provide the fromlist and level arguments of __import__().  The
        module object is pushed onto the stack.  The current namespace
        is not affected: for a proper import statement, a subsequent
        STORE_FAST instruction modifies the namespace.

        Note: name = co_names[namei] set in parse_byte_and_args()
        """
        level, fromlist = self.vm.popn(2)
        frame = self.vm.frame

        # Should we replace import "name" with a compatabliity version?
        # if importlib is not None:
        #     module_spec = importlib.util.find_spec(name)
        #     module = importlib.util.module_from_spec(module_spec)

        #     load_module = (
        #         module_spec.loader.exec_module
        #         if hasattr(module_spec.loader, "exec_module")
        #         else module_spec.loader.load_module
        #     )
        #     load_module(module)

        # elif PYTHON_VERSION_TRIPLE >= (3, 0):
        #     # This should make a *copy* of the module so we keep interpreter and
        #     # interpreted programs separate.
        #     # See below for how we handle "sys" import
        #     # FIXME: should split on ".". Doesn't work for, say, os.path
        #     if level < 0:
        #         level = 0
        #     module = importlib.__import__(
        #         name, frame.f_globals, frame.f_locals, fromlist, level
        #     )
        # else:
        #     module = __import__(name, frame.f_globals, frame.f_locals, fromlist, level)

        # INVESTIGATE: the above doesn't work for things like "import os.path as osp"
        # The module it finds ins os.posixpath which doesn't have a "path" attribute
        # while the below finds "os" which does have a "path" attribute.
        #
        assert level >= -1, f"Invalid Level number {level} on IMPORT_NAME"
        module = None
        if level == -1:
            # In Python 2.6 added the level parameter and it was -1 by default until but not including 3.0.
            # -1 means try relative imports before absolute imports.
            # FIXME: give warning that we can't handle absolute import. Or fix up code to handle possible absolute import.
            raise Exception("can't do absolute import")
            # level = 0

        if module is None:
            module = __import__(name, frame.f_globals, frame.f_locals, fromlist, level)

        self.vm.push(module)

    def IMPORT_STAR(self):
        """Loads all symbols not starting with '_' directly from the module
        TOS to the local namespace. The module is popped after loading all
        names. This opcode implements from module import *.
        """
        # TODO: this doesn't use __all__ properly.
        mod = self.vm.pop()
        for attr in dir(mod):
            if attr[0] != "_":
                self.vm.frame.f_locals[attr] = getattr(mod, attr)

    def IS_OP(self, invert: int):
        """Performs is comparison, or is not if invert is 1."""
        TOS1, TOS = self.vm.popn(2)
        if invert:
            self.vm.push(TOS1 is not TOS)
        else:
            self.vm.push(TOS1 is TOS)
        pass

    def JUMP_ABSOLUTE(self, target):
        """Set bytecode counter to target."""
        self.vm.jump(target)

    def JUMP_FORWARD(self, delta):
        """Increments bytecode counter by delta."""
        self.vm.jump(delta)

    def JUMP_IF_FALSE_OR_POP(self, target):
        """
        If TOS is false, sets the bytecode counter to target and leaves TOS
        on the stack. Otherwise (TOS is true), TOS is popped.
        """
        val = self.vm.top()
        if not val:
            self.vm.jump(target)
        else:
            self.vm.pop()

    def JUMP_IF_NOT_EXC_MATCH(self, target: int):
        """Tests whether the second value on the stack is an exception
        matching TOS, and jumps if it is not.  Pops two values from
        the stack.
        """
        TOS1, TOS = self.vm.popn(2)
        # FIXME: not sure what operation should be used to test not "matches".
        if not issubclass(TOS1, TOS):
            self.vm.jump(target)
        return

    def JUMP_IF_TRUE_OR_POP(self, target):
        """
        If TOS is true, sets the bytecode counter to target and leaves TOS
        on the stack. Otherwise (TOS is false), TOS is popped.
        """
        val = self.vm.top()
        if val:
            self.vm.jump(target)
        else:
            self.vm.pop()

    def LIST_APPEND(self, count):
        """Calls list.append(TOS[-i], TOS). Used to implement list comprehensions.
        While the appended value is popped off, the list object remains on the stack
        so that it is available for further iterations of the loop.
        """
        val = self.vm.pop()
        the_list = self.vm.peek(count)
        the_list.append(val)

    def LIST_EXTEND(self, i):
        """Calls list.extend(TOS1[-i], TOS). Used to build lists."""
        TOS = self.vm.pop()
        destination = self.vm.peek(i)
        assert isinstance(destination, list)
        destination.extend(TOS)

    def LIST_TO_TUPLE(self):
        """
        Pops a list from the stack and pushes a tuple containing the same values.
        """
        self.vm.push(tuple(self.vm.pop()))

    def LOAD_ASSERTION_ERROR(self):
        """
        Pushes AssertionError onto the stack. Used by the `assert` statement.
        """
        self.vm.push(AssertionError)

    def LOAD_ATTR(self, name):
        """Replaces TOS with getattr(TOS, co_names[namei]).

        Note: name = co_names[namei] set in parse_byte_and_args()
        """
        obj = self.vm.pop()
        val = getattr(obj, name)
        self.vm.push(val)

    def LOAD_BUILD_CLASS(self):
        """Pushes builtins.__build_class__() onto the stack. It is later called by CALL_FUNCTION to construct a class."""
        self.vm.push(__build_class__)

    def LOAD_CLASSDEREF(self, count):
        """
        Much like LOAD_DEREF but first checks the locals dictionary before
        consulting the cell. This is used for loading free variables in class
        bodies.
        """
        self.vm.push(self.vm.frame.cells[count].get())

    def LOAD_CLOSURE(self, i):
        """
        Pushes a reference to the cell contained in slot i of the cell and
        free variable storage. The name of the variable is co_cellvars[i] if i is less
        than the length of co_cellvars. Otherwise it is co_freevars[i -len(co_cellvars)].
        """
        self.vm.push(self.vm.frame.cells[i])

    def LOAD_CONST(self, const):
        """Pushes co_consts[consti] onto the stack."""
        self.vm.push(const)

    def LOAD_DEREF(self, name):
        """
        Loads the cell contained in slot i of the cell and free variable
        storage. Pushes a reference to the object the cell contains on the
        stack.
        """
        self.vm.push(self.vm.frame.cells[name].get())

    def LOAD_FAST(self, name):
        """
        Pushes a reference to the local co_varnames[var_num] onto the stack.
        """
        if name in self.vm.frame.f_locals:
            val = self.vm.frame.f_locals[name]
        else:
            raise UnboundLocalError(
                "local variable '%s' referenced before assignment" % name
            )
        self.vm.push(val)

    def LOAD_GLOBAL(self, name):
        """
        Loads the global named co_names[namei] onto the stack.

        Note: name = co_names[namei] set in parse_byte_and_args()
        """
        f = self.vm.frame
        if name in f.f_globals:
            val = f.f_globals[name]
        elif name in f.f_builtins:
            val = f.f_builtins[name]
        else:
            raise NameError("global name '%s' is not defined" % name)
        self.vm.push(val)

    def LOAD_LOCALS(self):
        """
        Pushes a reference to the locals of the current scope on the
        stack. This is used in the code for a class definition: After the
        class body is evaluated, the locals are passed to the class
        definition."""
        self.vm.push(self.vm.frame.f_locals)

    def LOAD_METHOD(self, name):
        """Loads a method named co_names[namei] from the TOS object. TOS is
        popped. This bytecode distinguishes two cases: if TOS has a
        method with the correct name, the bytecode pushes the unbound
        method and TOS. TOS will be used as the first argument (self)
        by CALL_METHOD when calling the unbound method. Otherwise,
        NULL and the object return by the attribute lookup are pushed.

        rocky: In our implementation in Python we don't have NULL: all
        stack entries have *some* value. So instead we'll push another
        item: the status. Also, instead of pushing the unbound method
        and self, we will pass the bound method, since that is what we
        have here. So TOS (self) is not pushed back onto the stack.
        """
        TOS = self.vm.pop()
        if hasattr(TOS, name):
            # FIXME: check that gettr(TO, name) is a method
            self.vm.push(getattr(TOS, name))
            self.vm.push("LOAD_METHOD lookup success")
        else:
            self.vm.push("fill in attribute method lookup")
            self.vm.push(None)

    def LOAD_NAME(self, name):
        """Pushes the value associated with co_names[namei] onto the stack."""
        # Running this opcode can raise a NameError.
        #
        # FIXME: Better would be to separate NameErrors caused by
        # interpreting bytecode versus NameErrors that are caused as a result of bugs
        # in the interpreter.
        self.vm.push(self.lookup_name(name))
        # try:
        #     self.lookup_name(name)
        # except NameError:
        #     self.vm.last_traceback = traceback_from_frame(self.vm.frame)
        #     tb  = traceback_from_frame(self.vm.frame)
        #     self.vm.last_exception = (NameError, NameError("name '%s' is not defined" % name), tb)
        #     return "exception"
        # else:
        #     self.vm.push(self.lookup_name(name))

    def MAKE_CLOSURE(self, argc: int):
        """
        Creates a new function object, sets its func_closure slot, and
        pushes it on the stack. TOS is the code associated with the
        function. If the code object has N free variables, the next N
        items on the stack are the cells for these variables. The
        function also has argc default parameters, where are found
        before the cells.
        """
        name = self.vm.pop()
        closure, code = self.vm.popn(2)
        defaults = self.vm.popn(argc)
        globs = self.vm.frame.f_globals
        fn = Function(name, code, globs, defaults, closure, self.vm)
        self.vm.push(fn)

    def MAKE_FUNCTION(self, argc: int):
        """
        Pushes a new function object on the stack. From bottom to top,
        the consumed stack must consist of values if the argument
        carries a specified flag value

        * 0x01 a tuple of default values for positional-only and positional-or-keyword parameters in positional order
        * 0x02 a dictionary of the default values for the keyword-only parameters
               the key is the parameter name and the value is the default value
        * 0x04 a tuple of strings containing parameters  annotations
        * 0x08 a tuple containing cells for free variables, making a closure
          the code associated with the function (at TOS1)
        * the qualified name of the function (at TOS)

        Changed from version 3.6: Flag value 0x04 is a tuple of strings instead of dictionary
        """
        # TODO(max): create vars

        qualname = self.vm.pop()
        name = qualname.split(".")[-1]
        code = self.vm.pop()

        slot = {
            "defaults": tuple(),
            "kwdefaults": {},
            "annotations": tuple(),
            "closure": tuple(),
        }
        assert 0 <= argc < (1 << MAKE_FUNCTION_SLOTS)
        have_param = list(
            reversed([True if 1 << i & argc else False for i in range(4)])
        )
        for i in range(MAKE_FUNCTION_SLOTS):
            if have_param[i]:
                slot[MAKE_FUNCTION_SLOT_NAMES[i]] = self.vm.pop()

        globs = self.vm.frame.f_globals
        if not inspect.iscode(code) and hasattr(code, "to_native"):
            code = code.to_native()

        # Convert annotations tuple into dictionary
        annotations = {}
        annotations_tup = slot["annotations"]
        for i in range(0, len(annotations_tup), 2):
            annotations[annotations_tup[i]] = annotations_tup[i + 1]

        assert len(annotations) == code.co_argcount, f"missing annotations for {name}"

        fn_vm = Function(
            name=name,
            qualname=qualname,
            code=code,
            globs=globs,
            argdefs=slot["defaults"],
            closure=slot["closure"],
            vm=self.vm,
            kwdefaults=slot["kwdefaults"],
            annotations=annotations,
        )

        if argc == 0 and code.co_name in COMPREHENSION_FN_NAMES:
            fn_vm.has_dot_zero = True

        if fn_vm._func:
            self.vm.fn2native[fn_vm] = fn_vm._func
        else:
            raise Exception("no fn_vm._func")

        func_op = func_dialect.FuncOp(
            name=name,
            type=(
                tuple(
                    [
                        self.vm.body_builder.type_mapping[t.__name__]
                        for _a, t in annotations.items()
                    ]
                ),
                (),
            ),
            visibility="private",
            loc=self.vm.mlir_location,
        )
        self.vm.fn2native_mlir[fn_vm] = func_op
        self.vm.push(fn_vm)

    def MAP_ADD(self, count):
        """Calls dict.setitem(TOS1[-count], TOS1, TOS). Used to implement dict
        comprehensions.

        For all of the SET_ADD, LIST_APPEND and MAP_ADD instructions,
        while the added value or key/value pair is popped off, the
        container object remains on the stack so that it is available
        for further iterations of the loop.
        """
        # FIXME: the below seems fishy.
        key, val = self.vm.popn(2)
        the_map = self.vm.peek(count)
        the_map[key] = val

    def MATCH_CLASS(self):
        """TOS is a tuple of keyword attribute names, TOS1 is the class being
        matched against, and TOS2 is the match subject. count is the number of
        positional sub-patterns.

        Pop TOS. If TOS2 is an instance of TOS1 and has the positional
        and keyword attributes required by count and TOS, set TOS to
        True and TOS1 to a tuple of extracted attributes. Otherwise,
        set TOS to False.
        """
        # FIXME
        raise PyVMError("MATCH_CLASS not implemented")

    def MATCH_KEYS(self):
        """TOS is a tuple of mapping keys, and TOS1 is the match subject. If
        TOS1 contains all of the keys in TOS, push a tuple containing
        the corresponding values, followed by True. Otherwise, push
        None, followed by False.
        """
        # FIXME
        raise PyVMError("MATCH_KEYS not implemented")

    def MATCH_MAPPING(self):
        """If TOS is an instance of collections.abc.Mapping (or, more
        technically: if it has the Py_TPFLAGS_MAPPING flag set in its
        tp_flags), push True onto the stack. Otherwise, push False.
        """
        # FIXME
        raise PyVMError("MATCH_MAPPING not implemented")

    def MATCH_SEQUENCE(self):
        """If TOS is an instance of collections.abc.Sequence and is not an
        instance of str/bytes/bytearray (or, more technically: if it
        has the Py_TPFLAGS_SEQUENCE flag set in its tp_flags), push
        True onto the stack. Otherwise, push False.
        """
        # FIXME
        raise PyVMError("MATCH_SEQUENCE not implemented")

    def NOP(self):
        "Do nothing code. Used as a placeholder by the bytecode optimizer."
        pass

    def POP_BLOCK(self):
        """
        Removes one block from the block stack. Per frame, there is a
        stack of blocks, denoting nested loops, try statements, and
        such."""
        self.vm.pop_block()

    def POP_EXCEPT(self):
        """
        Removes one block from the block stack. The popped block must be an
        exception handler block, as implicitly created when entering an except
        handler. In addition to popping extraneous values from the frame
        stack, the last three popped values are used to restore the exception
        state."""
        block = self.vm.pop_block()
        if block.type != "except-handler":
            raise PyVMError("popped block is not an except handler; is %s" % block)
        self.vm.unwind_block(block)

    def POP_JUMP_IF_FALSE(self, target):
        """If TOS is false, sets the bytecode counter to target. TOS is popped."""
        # TODO(max): hack this so it falls through toe the next
        val = self.vm.pop()
        if not val:
            self.vm.jump(target)

    def POP_JUMP_IF_TRUE(self, target):
        """If TOS is true, sets the bytecode counter to target. TOS is popped."""
        # TODO(max): hack this so it falls through toe the next
        val = self.vm.pop()
        if val:
            self.vm.jump(target)

    def POP_TOP(self):
        "Removes the top-of-stack (TOS) item."
        self.vm.pop()

    def RAISE_VARARGS(self, argc: int):
        """
        Raises an exception. argc indicates the number of arguments to the
        raise statement, ranging from 0 to 3. The handler will find
        the traceback as TOS2, the parameter as TOS1, and the
        exception as TOS.
        """
        cause = exc = None
        if argc == 2:
            cause = self.vm.pop()
            exc = self.vm.pop()
        elif argc == 1:
            exc = self.vm.pop()
        return self.do_raise(exc, cause)

    def RERAISE(self):
        raise RuntimeError("RERAISE not implemented yet")
        pass

    def RETURN_VALUE(self):
        """Returns with TOS to the caller of the function."""
        self.vm.return_value = self.vm.pop()
        if self.vm.frame.generator:
            self.vm.frame.generator.finished = True

        return "return"

    def ROT_FOUR(self):
        "Lifts second, third and forth stack item one position up, moves top down to position four."
        a, b, c, d = self.vm.popn(4)
        self.vm.push(d, a, b, c)

    def ROT_N(self, count: int):
        """
        Lift the top count stack items one position up, and move TOS down to position count.
        """
        # FIXME
        raise PyVMError("ROT_N not implemented")

    def ROT_THREE(self):
        "Lifts second and third stack item one position up, moves top down to position three."
        a, b, c = self.vm.popn(3)
        self.vm.push(c, a, b)

    def ROT_TWO(self):
        "Swaps the two top-most stack items."
        a, b = self.vm.popn(2)
        self.vm.push(b, a)

    def SETUP_ANNOTATIONS(self):
        """
        Checks whether __annotations__ is defined in locals(), if not it
        is set up to an empty dict. This opcode is only emitted if a
        class or module body contains variable annotations
        statically.
        """
        if "__annotations__" not in self.vm.frame.f_locals:
            self.vm.frame.f_locals["__annotations__"] = {}

    def SETUP_ASYNC_WITH(self):
        """Creates a new frame object."""
        raise PyVMError("SETUP_ASYNC_WITH not implemented yet")

    def SETUP_FINALLY(self, jump_offset):
        """
        Pushes a try block from a try-except clause onto the block
        stack. delta points to the finally block.

        Note: jump = delta + f.f_lasti set in parse_byte_and_args()
        """
        self.vm.push_block("finally", jump_offset)

    def convert_method_native_func(self, frame, method):
        """If a method's function is a native functions, converted it to the
        corresponding PyVM Method so that we can interpret it.
        """
        if not self.method_func_access:
            for func_attr in ("__func__", "im_func"):
                if hasattr(method, func_attr):
                    # Save attribute access name, so we don't
                    # have to compute this again.
                    self.method_func_access = func_attr
                    break
                pass
            else:
                raise PyVMError(
                    "Can't find method function attribute; tried '__func__' and '_im_func'"
                )
            pass

        try:
            func = getattr(method, self.method_func_access)
        except:
            func = method

        if inspect.isfunction(func):
            func = self.convert_native_to_function(self.vm.frame, func)
            method = types.MethodType(func, method.__self__)
        return method

    def SETUP_WITH(self, delta):
        """
        This opcode performs several operations before a with block
        starts. First, it loads __exit__() from the context manager
        and pushes it onto the stack for later use by
        WITH_CLEANUP. Then, __enter__() is called, and a finally block
        pointing to delta is pushed. Finally, the result of calling
        the enter method is pushed onto the stack. The next opcode
        will either ignore it (POP_TOP), or store it in (a)
        variable(s) (STORE_FAST, STORE_NAME, or UNPACK_SEQUENCE).
        """
        context_manager = self.vm.pop()

        # Make sure __enter__ and __exit__ functions in context_manager are
        # converted to our Function type so we can interpret them.
        # Note though that built-in functions can't be traced.
        if not inspect.isbuiltin(context_manager.__exit__):
            try:
                exit_method = self.convert_method_native_func(
                    self.vm.frame, context_manager.__exit__
                )
            except:
                exit_method = context_manager.__exit__
        else:
            exit_method = context_manager.__exit__
        self.vm.push(exit_method)
        if not inspect.isbuiltin(context_manager.__enter__):
            self.convert_method_native_func(self.vm.frame, context_manager.__enter__)
        finally_block = context_manager.__enter__()
        self.vm.push_block("finally", delta)
        self.vm.push(finally_block)

    def SET_ADD(self, count):
        """Calls set.add(TOS1[-count], TOS). Used to implement set
        comprehensions.
        """
        val = self.vm.pop()
        the_set = self.vm.peek(count)
        the_set.add(val)

    def SET_UPDATE(self, i):
        """Calls set.update(TOS1[-i], TOS). Used to build sets."""
        TOS = self.vm.pop()
        destination = self.vm.peek(i)
        assert isinstance(destination, set)
        destination.update(TOS)

    def STORE_ATTR(self, name):
        """Implements TOS.name = TOS1, where namei is the index of name in co_names."""
        val, obj = self.vm.popn(2)
        setattr(obj, name, val)

    def STORE_DEREF(self, name):
        """
        Stores TOS into the cell contained in slot i of the cell and free variable storage.
        """
        self.vm.frame.cells[name].set(self.vm.pop())

    def STORE_FAST(self, var_num):
        """Stores TOS into the local co_varnames[var_num]."""
        self.vm.frame.f_locals[var_num] = self.vm.pop()

    def STORE_GLOBAL(self, name):
        "Works as STORE_NAME, but stores the name as a global."
        f = self.vm.frame
        f.f_globals[name] = self.vm.pop()

    def STORE_NAME(self, name):
        """Implements name = TOS. namei is the index of name in the attribute
        co_names of the code object. The compiler tries to use STORE_LOCAL or
        STORE_GLOBAL if possible."""
        self.vm.frame.f_locals[name] = self.vm.pop()

    def STORE_SUBSCR(self):
        """Implements TOS1[TOS] = TOS2."""
        val, obj, subscr = self.vm.popn(3)
        if not all(isinstance(s, ir.Type) for s in subscr):
            subscr = list(subscr)
            for i, s in enumerate(subscr):
                assert isinstance(s, int)
                subscr[i] = self.vm.get_or_make_mlir_constant(s, index_type=True)
        self.vm.body_builder.memref_store(obj, val, subscr)
        # TODO(max): this is an opportunity to wrap and keep these refs
        # obj[subscr] = val

    def UNPACK_SEQUENCE(self, count):
        """Unpacks TOS into count individual values, which are put onto the
        stack right-to-left.
        """
        seq = self.vm.pop()
        for x in reversed(seq):
            self.vm.push(x)

    def WITH_CLEANUP(self):
        """Cleans up the stack when a "with" statement block exits. On top of
        the stack are 1 3 values indicating how/why the finally clause
        was entered:

        * TOP = None
        * (TOP, SECOND) = (WHY_{RETURN,CONTINUE}), retval
        * TOP = WHY_*; no retval below it
        * (TOP, SECOND, THIRD) = exc_info()

        Under them is EXIT, the context manager s __exit__() bound method.

        In the last case, EXIT(TOP, SECOND, THIRD) is called,
        otherwise EXIT(None, None, None).

        EXIT is removed from the stack, leaving the values above it in
        the same order. In addition, if the stack represents an
        exception, and the function call returns a  true  value, this
        information is  zapped , to prevent END_FINALLY from
        re-raising the exception. (But non-local gotos should still be
        resumed.)

        All of the following opcodes expect arguments. An argument is
        two bytes, with the more significant byte last.
        """
        # The code here does some weird stack manipulation: the __exit__ function
        # is buried in the stack, and where depends on what's on top of it.
        # Pull out the exit function, and leave the rest in place.
        # In Python 3.x this is fixed up so that the __exit__ funciton os TOS
        v = w = None
        u = self.vm.top()
        if u is None:
            exit_func = self.vm.pop(1)
        elif isinstance(u, str):
            if u in ("return", "continue"):
                exit_func = self.vm.pop(2)
            else:
                exit_func = self.vm.pop(1)
            u = None
        elif issubclass(u, BaseException):
            w, v, u = self.vm.popn(3)
            exit_func = self.vm.pop()
            self.vm.push(w, v, u)
        else:  # pragma: no cover
            raise PyVMError("Confused WITH_CLEANUP")
        exit_ret = exit_func(u, v, w)
        err = (u is not None) and bool(exit_ret)
        if err:
            # An error occurred, and was suppressed
            self.vm.popn(3)
            self.vm.push(None)

    def WITH_EXCEPT_START(self):
        # FIXME
        raise RuntimeError("WITH_EXCEPT_START not implemented yet")

    def YIELD_FROM(self):
        """
        Pops TOS and delegates to it as a subiterator from a generator.
        """
        u = self.vm.pop()
        x = self.vm.top()

        try:
            if not isinstance(x, Generator) or u is None:
                # Call next on iterators.
                retval = next(x)
            else:
                retval = x.send(u)
            self.vm.return_value = retval
        except StopIteration as e:
            self.vm.pop()
            self.vm.push(e.value)
        else:
            # FIXME: The code has the effect of rerunning the last instruction.
            # I'm not sure if or why it is correct.
            if self.vm.version >= (3, 6):
                self.vm.jump(self.vm.frame.f_lasti - 2)
            else:
                self.vm.jump(self.vm.frame.f_lasti - 1)
            return "yield"

    def YIELD_VALUE(self):
        """
        Pops TOS and yields it from a generator.
        """
        self.vm.return_value = self.vm.pop()
        return "yield"

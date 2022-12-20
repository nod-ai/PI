"""A pure-Python Python bytecode interpreter."""
# Based on:
# pyvm2 by Paul Swartz (z3p), from http://www.twistedmatrix.com/users/z3p/
import linecache
import logging
import reprlib
import sys
from collections import namedtuple, defaultdict
from contextlib import contextmanager
from typing import Optional, Union

import six
from xdis import (
    code2num,
    CO_NEWLOCALS,
    op_has_argument,
    next_offset,
)
from xdis.op_imports import get_opcode_module

from shark.compiler.byte_code_interpreter.body import BodyBuilder
from shark.compiler.byte_code_interpreter.errors import PyVMError, PyVMUncaughtException
from shark.compiler.byte_code_interpreter.pyobj import (
    Frame,
    Block,
    Traceback,
    traceback_from_frame,
)
from shark import ir

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

LINE_NUMBER_WIDTH = 4
LINE_NUMBER_WIDTH_FMT = "L. %%-%dd@" % LINE_NUMBER_WIDTH
LINE_NUMBER_SPACES = " " * (LINE_NUMBER_WIDTH + len("L. ")) + "@"

# Create a repr that won't overflow.
repr_obj = reprlib.Repr()
repr_obj.maxother = 120
repper = repr_obj.repr


def format_instruction(
    frame,
    opc,
    bytecode_name,
    int_arg,
    arguments,
    offset,
    line_number,
    extra_debug,
    vm=None,
):
    """Formats an instruction. What's a little different here is that in
    contast to Python's `dis`, or a colorized version of that, used in
    `trepan3k` we may have access to the frame eval stack and therefore
    can show operands in a nicer way.

    But we also make use of xdis' nicer argument formatting as well. These appear
    for example in MAKE_FUNCTION, and CALL_FUNCTION.
    """
    code = frame.f_code if frame else None
    byte_code = opc.opmap.get(bytecode_name, 0)

    if vm and bytecode_name in vm.byteop.stack_fmt:
        stack_args = vm.byteop.stack_fmt[bytecode_name](vm, int_arg, repr)
    else:
        stack_args = ""

    if hasattr(opc, "opcode_arg_fmt") and bytecode_name in opc.opcode_arg_fmt:
        argrepr = f"""[{opc.opcode_arg_fmt[bytecode_name](int_arg)}] {int_arg}"""
    elif int_arg is None:
        argrepr = ""
    elif byte_code in opc.COMPARE_OPS:
        argrepr = opc.cmp_op[int_arg]
    elif isinstance(arguments, list) and arguments:
        argrepr = arguments[0]
    else:
        argrepr = arguments

    line_str = (
        LINE_NUMBER_SPACES
        if line_number is None
        else LINE_NUMBER_WIDTH_FMT % line_number
    )
    mess = "%s%3d: %s%s %s" % (line_str, offset, bytecode_name, stack_args, argrepr)
    if extra_debug and frame:
        mess += " %s in %s:%s" % (code.co_name, code.co_filename, frame.f_lineno)
    return mess


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


class PyVM(object):
    def __init__(
        self,
        mlir_context: ir.Context,
        mlir_location: ir.Location,
        mlir_module: ir.Module,
        vmtest_testing=True,
        format_instruction_func=format_instruction,
    ):

        self.local_defs = {}
        self.mlir_context = mlir_context
        self.mlir_location = mlir_location
        self.mlir_module = mlir_module

        f64_type = ir.F64Type.get(context=self.mlir_context)
        self.body_builder = BodyBuilder(
            type_mapping={
                "float": f64_type,
                "float_memref": ir.MemRefType.get((-1,), f64_type, loc=self.mlir_location),
                "int": ir.IntegerType.get_signed(64, context=self.mlir_context),
                "uint": ir.IntegerType.get_unsigned(64, context=self.mlir_context),
            },
            block_arg_mapping={},
            fn_attr_mapping={},
            context=self.mlir_context,
            location=self.mlir_location,
        )

        self.scf_fors = False

        # TODO(max): not correct - should be a function of the call or module or something like that
        self._mlir_block_stack: list[MLIRStackFrame] = []
        self.enter_mlir_block_scope(
            self.mlir_module.body,
            mlir_context=self.mlir_context or self.mlir_context,
            scope_name="module",
        )

        # The call stack of frames.
        self.frames = []
        # The current frame.
        self.frame = None
        self.return_value = None
        self.last_exception = None
        self.last_traceback_limit = None
        self.last_traceback = None
        self.format_instruction = format_instruction_func

        # FIXME: until we figure out how to fix up test/vmtest.el
        # This changes how we report a VMRuntime error.
        self.vmtest_testing = vmtest_testing

        # Like sys.exc_info() tuple
        self.last_exception = None

        # Sometimes we need a native function (e.g. for method lookup), but
        # most of the time we want a VM function defined in pyobj.
        # This maps between the two.
        self.fn2native = {}
        self.fn2native_mlir = {}

        self.in_exception_processing = False

        # This is somewhat hokey:
        # Give byteop routines a way to raise an error, without having
        # to import this file. We import from from byteops.
        # Alternatively, VMError could be
        # pulled out of this file

        self.opc = get_opcode_module()
        # self.byteop = get_byteop(self, python_version, is_pypy)
        from shark.compiler.byte_code_interpreter.byteop import ByteOp

        self.byteop = ByteOp(self)

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
            scope_name = "UNKNOWN_SCOPE"

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
            scope_name = "UNKNOWN_SCOPE"
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
            scope_name = "UNKNOWN_SCOPE"
        yield self.enter_mlir_block_scope(
            block,
            block_args=block_args,
            insertion_point=insertion_point,
            mlir_location=mlir_location,
            scope_name=scope_name,
        )
        self.exit_mlir_block_scope(scope_name=scope_name)

    def mlir_peek_block_scope(self):
        assert len(self._mlir_block_stack), "no block scope yet"
        return self._mlir_block_stack[-1]

    def get_or_make_mlir_constant(
        self,
        py_cst: Union[int, float, bool],
        name: Optional[str] = None,
        index_type: bool = False,
    ):
        if name is None:
            name = str(py_cst)
        if (name, index_type) not in self.local_defs:
            self.local_defs[name, index_type] = self.body_builder.constant(
                py_cst, index_type
            )
        return self.local_defs[name, index_type]

    ##############################################
    # Frame operations. First the frame stack....
    ##############################################
    def access(self, i=0):
        """return object at position i.
        Default to the top of the stack, but `i` can be a count from the top
        instead.
        """
        return self.frame.stack[-1 - i]

    def peek(self, n):
        if n <= 0:
            raise PyVMError("Peek value must be greater than 0")
        try:
            return self.frame.stack[-n]
        except:
            return 0

    def pop(self, i=0):
        """Pop a value from the stack.

        Default to the top of the stack, but `i` can be a count from the top
        instead.

        """
        return self.frame.stack.pop(-1 - i)

    def popn(self, n):
        """Pop a number of values from the value stack.

        A list of `n` values is returned, the deepest value first.

        """
        if n:
            ret = self.frame.stack[-n:]
            self.frame.stack[-n:] = []
            return ret
        else:
            return []

    def push(self, *vals):
        """Push values onto the value stack."""
        self.frame.stack.extend(vals)

    def top(self):
        """Return the value at the top of the stack, with no changes."""
        return self.frame.stack[-1]

    # end of frame stack operations
    # onto frame block operations..

    def pop_block(self):
        return self.frame.block_stack.pop()

    def push_block(self, type, handler=None, level=None):
        if level is None:
            level = len(self.frame.stack)
        self.frame.block_stack.append(Block(type, handler, level))

    def top_block(self):
        return self.frame.block_stack[-1]

    def jump(self, jump):
        """Move the bytecode pointer to `jump`, so it will execute next,
        However we subtract one from the offset, because fetching the
        next instruction adds one before fetching.
        """
        # The previous pyvm2.py code *always* had self.frame.f_lasti
        # represent the *next* instruction rather than the *last* or
        # current instruction currently under execution. While this
        # was easier to code, consisitent and worked, IT DID NOT
        # REPRESENT PYTHON's semantics. It became unbearable when I
        # added a debugger for x-python that relies on
        # self.frame.f_last_i being correct.
        self.frame.f_lasti = jump
        self.frame.fallthrough = False

    def make_frame(
        self, code, callargs=None, f_globals=None, f_locals=None, closure=None
    ):
        # The callargs default is safe because we never modify the dict.
        # pylint: disable=dangerous-default-value

        if callargs is None:
            callargs = {}
        log.debug(
            "make_frame: code=%r, callargs=%s, f_globals=%r, f_locals=%r",
            code,
            repper(callargs),
            (type(f_globals), id(f_globals)),
            (type(f_locals), id(f_locals)),
        )
        if f_globals is not None:
            f_globals = f_globals
            if f_locals is None:
                f_locals = f_globals
        elif self.frames:
            f_globals = self.frame.f_globals
            if f_locals is None:
                f_locals = {}
        else:
            # TODO(ampere): __name__, __doc__, __package__ below are not correct
            f_globals = f_locals = {
                "__builtins__": __builtins__,
                "__name__": "__main__",
                "__doc__": None,
                "__package__": None,
            }

        # Implement NEWLOCALS flag. See Objects/frameobject.c in CPython.
        if code.co_flags & CO_NEWLOCALS:
            f_locals = {"__locals__": {}}

        f_locals.update(callargs)
        frame = Frame(
            f_code=code,
            f_globals=f_globals,
            f_locals=f_locals,
            f_back=self.frame,
            closure=closure,
        )

        # THINK ABOUT: should this go into making the frame?
        frame.linestarts = dict(self.opc.findlinestarts(code, dup_lines=True))

        log.debug("%r", frame)
        return frame

    def push_frame(self, frame):
        self.frames.append(frame)
        self.frame = frame

    def pop_frame(self):
        self.frames.pop()
        if self.frames:
            self.frame = self.frames[-1]
        else:
            self.frame = None

    def print_frames(self):
        """Print the call stack for debugging. Note that the
        format exactly the same as in traceback.print_tb()
        """
        for f in self.frames:
            filename = f.f_code.co_filename
            lineno = f.line_number()
            print('  File "%s", line %d, in %s' % (filename, lineno, f.f_code.co_name))
            linecache.checkcache(filename)
            line = linecache.getline(filename, lineno, f.f_globals)
            if line:
                print("    " + line.strip())

    def resume_frame(self, frame):
        frame.f_back = self.frame
        log.debug("resume_frame: %r", frame)

        # Make sure we advance to the next instruction after where we left off.
        if frame.f_lasti == -1:
            # We are just starting out. Set offset to the first
            # instruction, and signal that we should not increment
            # this before fetching next instruction.
            frame.fallthrough = False
            frame.f_lasti = 0
        else:
            frame.fallthrough = True

        val = self.eval_frame(frame)
        frame.f_back = None
        return val

    ##############################################
    # End Frame operations.
    ##############################################

    # This is the main entry point
    def run_code(self, code, f_globals=None, f_locals=None, toplevel=True):
        """run code using f_globals and f_locals in our VM"""
        self.top_level_frame = self.make_frame(
            code, f_globals=f_globals, f_locals=f_locals
        )
        try:
            val = self.eval_frame(self.top_level_frame)
        except Exception:
            # Until we get test/vmtest.py under control:
            if self.vmtest_testing:
                raise
            if self.last_traceback:
                self.last_traceback.print_tb()
                print("%s" % self.last_exception[0].__name__, end="")
                le1 = self.last_exception[1]
                tail = ""
                if le1:
                    tail = "\n".join(le1.args)
                print(tail)
            raise

        # Frame ran to normal completion... check some invariants
        if toplevel:
            if self.frames:  # pragma: no cover
                raise PyVMError("Frames left over!")
            if self.frame and self.frame.stack:  # pragma: no cover
                raise PyVMError("Data left on stack! %r" % self.frame.stack)

        return val

    def unwind_block(self, block):
        if block.type == "except-handler":
            offset = 3
        else:
            offset = 0

        while len(self.frame.stack) > block.level + offset:
            self.pop()

        if block.type == "except-handler":
            tb, value, exctype = self.popn(3)
            self.last_exception = exctype, value, tb

    def parse_byte_and_args(self, byte_code, replay=False):
        """Parse 1 - 3 bytes of bytecode into
        an instruction and optionally arguments.

        Argument replay is used to handle breakpoints.
        """

        f = self.frame
        f_code = f.f_code
        co_code = f_code.co_code
        extended_arg = 0

        # Note: There is never more than one argument.
        # The list size is used to indicate whether an argument
        # exists or not.
        # FIMXE: remove and use int_arg as a indicator of whether
        # the argument exists.
        arguments = []
        int_arg = None

        while True:
            if f.fallthrough:
                if not replay:
                    f.f_lasti = next_offset(byte_code, self.opc, f.f_lasti)
            else:
                # Jump instructions must set this False.
                f.fallthrough = True
            offset = f.f_lasti
            line_number = self.frame.linestarts.get(offset, None)
            if line_number is not None:
                f.f_lineno = line_number
            if not replay:
                byte_code = co_code[offset]
            bytecode_name = self.opc.opname[byte_code]
            arg_offset = offset + 1

            if op_has_argument(byte_code, self.opc):
                int_arg = code2num(co_code, arg_offset) | extended_arg
                # Note: Python 3.6.0a1 is 2, for 3.6.a3 and beyond we have 1
                arg_offset += 1
                if byte_code == self.opc.EXTENDED_ARG:
                    extended_arg = int_arg << 8
                    continue
                else:
                    # TODO(max): how the fuck does intellij infer unused???
                    extended_arg = 0

                if byte_code in self.opc.CONST_OPS:
                    arg = f_code.co_consts[int_arg]
                elif byte_code in self.opc.FREE_OPS:
                    if int_arg < len(f_code.co_cellvars):
                        arg = f_code.co_cellvars[int_arg]
                    else:
                        var_idx = int_arg - len(f.f_code.co_cellvars)
                        arg = f_code.co_freevars[var_idx]
                elif byte_code in self.opc.NAME_OPS:
                    arg = f_code.co_names[int_arg]
                elif byte_code in self.opc.JREL_OPS:
                    # Many relative jumps are conditional,
                    # so setting f.fallthrough is wrong.
                    int_arg += int_arg
                    arg = arg_offset + int_arg
                elif byte_code in self.opc.JABS_OPS:
                    # We probably could set fallthough, since many (all?)
                    # of these are unconditional, but we'll make the jump do
                    # the work of setting.
                    int_arg += int_arg
                    arg = int_arg
                elif byte_code in self.opc.LOCAL_OPS:
                    arg = f_code.co_varnames[int_arg]
                else:
                    arg = int_arg
                arguments = [arg]
            break

        return bytecode_name, byte_code, int_arg, arguments, offset, line_number

    def log(self, bytecode_name, int_arg, arguments, offset, line_number):
        """Log arguments, block stack, and data stack for each opcode."""
        op = self.format_instruction(
            self.frame,
            self.opc,
            bytecode_name,
            int_arg,
            arguments,
            offset,
            line_number,
            log.isEnabledFor(logging.DEBUG),
            vm=self,
        )
        indent = "    " * (len(self.frames) - 1)
        stack_rep = repper(self.frame.stack)
        block_stack_rep = repper(self.frame.block_stack)

        log.debug("  %sframe.stack: %s" % (indent, stack_rep))
        log.debug("  %sblocks     : %s" % (indent, block_stack_rep))
        log.info("%s%s" % (indent, op))

    def dispatch(self, bytecode_name, int_arg, arguments, offset, line_number):
        """Dispatch by bytecode_name to the corresponding methods.
        Exceptions are caught and set on the virtual machine."""

        why = None
        self.in_exception_processing = False
        try:
            # dispatch
            bytecode_fn = getattr(self.byteop, bytecode_name)
            if bytecode_fn is None:  # pragma: no cover
                raise PyVMError(
                    "Unknown bytecode type: %s\n\t%s"
                    % (
                        self.format_instruction(
                            self.frame,
                            self.opc,
                            bytecode_name,
                            int_arg,
                            arguments,
                            offset,
                            line_number,
                            False,
                        ),
                        bytecode_name,
                    )
                )
            why = bytecode_fn(*arguments)

        except Exception as e:
            # Deal with exceptions encountered while executing the op.
            self.last_exception = sys.exc_info()

            # FIXME: dry code
            if not self.in_exception_processing:
                if self.last_exception[0] != SystemExit:
                    log.info(
                        (
                            "exception in the execution of "
                            "instruction:\n\t%s"
                            % self.format_instruction(
                                self.frame,
                                self.opc,
                                bytecode_name,
                                int_arg,
                                arguments,
                                offset,
                                line_number,
                                False,
                            )
                        )
                    )
                if not self.last_traceback:
                    self.last_traceback = traceback_from_frame(self.frame)
                self.in_exception_processing = True

            why = "exception"

        return why

    def manage_block_stack(self, why):
        """Manage a frame's block stack.
        Manipulate the block stack and data stack for looping,
        exception handling, or returning."""
        assert why != "yield"

        block = self.frame.block_stack[-1]
        if block.type == "loop" and why == "continue":
            self.jump(self.return_value)
            why = None
            return why

        if not (block.type == "except-handler" and why == "silenced"):
            self.pop_block()
            self.unwind_block(block)

        if block.type == "loop" and why == "break":
            why = None
            self.jump(block.handler)
            return why

        if why == "exception" and block.type in ["setup-except", "finally"]:
            self.push_block("except-handler")
            exctype, value, tb = self.last_exception
            self.push(tb, value, exctype)
            # PyErr_Normalize_Exception goes here
            self.push(tb, value, exctype)
            why = None
            self.jump(block.handler)
            return why

        elif block.type == "finally":
            if why in ("return", "continue"):
                self.push(self.return_value)
            self.push(why)

            why = None
            self.jump(block.handler)
            return why
        elif block.type == "except-handler" and why == "silenced":
            # 3.5+ WITH_CLEANUP_FINISH
            # Nothing needs to be done here.
            return None
        elif why == "return":
            # 3.8+ END_FINALLY
            pass

        return why

    # Interpreter main loop
    # This is analogous to CPython's _PyEval_EvalFramDefault() (in 3.x newer Python)
    # or eval_frame() in older 2.x code.
    def eval_frame(self, frame):
        """Run a frame until it returns (somehow).

        Exceptions are raised, the return value is returned.

        """
        self.f_code = frame.f_code
        if frame.f_lasti == -1:
            # We were started new, not yielded back from.
            frame.f_lasti = 0
            # Don't increment before fetching next instruction.
            frame.fallthrough = False
            byte_code = None
        else:
            byte_code = self.f_code.co_code[frame.f_lasti]
            # byte_code == opcode["YIELD_VALUE"]?

        self.push_frame(frame)
        while True:
            (
                bytecode_name,
                byte_code,
                int_arg,
                arguments,
                offset,
                line_number,
            ) = self.parse_byte_and_args(byte_code)
            if log.isEnabledFor(logging.INFO):
                self.log(bytecode_name, int_arg, arguments, offset, line_number)

            # When unwinding the block stack, we need to keep track of why we
            # are doing it.
            why = self.dispatch(bytecode_name, int_arg, arguments, offset, line_number)
            if why == "exception":
                # TODO: ceval calls PyTraceBack_Here, not sure what that does.

                # Deal with exceptions encountered while executing the op.
                if not self.in_exception_processing:
                    # FIXME: DRY code
                    if self.last_exception[0] != SystemExit:
                        log.info(
                            (
                                "exception in the execution of "
                                "instruction:\n\t%s"
                                % self.format_instruction(
                                    frame,
                                    self.opc,
                                    bytecode_name,
                                    int_arg,
                                    arguments,
                                    offset,
                                    line_number,
                                    False,
                                )
                            )
                        )
                    if self.last_traceback is None:
                        self.last_traceback = traceback_from_frame(frame)
                    self.in_exception_processing = True

            elif why == "reraise":
                why = "exception"

            if why != "yield":
                while why and frame.block_stack:
                    # Deal with any block management we need to do.
                    why = self.manage_block_stack(why)

            if why:
                break

        # TODO: handle generator exception state

        self.pop_frame()

        if why == "exception":
            last_exception = self.last_exception
            if last_exception and last_exception[0]:
                if isinstance(last_exception[2], Traceback):
                    if not self.frame:
                        if isinstance(last_exception, tuple):
                            self.last_exception = PyVMUncaughtException.from_tuple(
                                last_exception
                            )
                        raise self.last_exception
                else:
                    six.reraise(*self.last_exception)
            else:
                raise PyVMError("Borked exception recording")
            # if self.exception and .... ?
            # log.error("Haven't finished traceback handling, nulling traceback information for now")
            # six.reraise(self.last_exception[0], None)

        self.in_exception_processing = False
        return self.return_value

    ## Operators

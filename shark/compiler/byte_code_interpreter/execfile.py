"""Execute files of Python code."""

import os
import sys
import tokenize

# To silence the "import imp" DeprecationWarning below
import warnings

warnings.filterwarnings("ignore")
# TODO(max): swap with importlib (or importlab?)
import imp

from shark import ir
from shark.compiler.byte_code_interpreter.vm import PyVM
from shark.compiler.byte_code_interpreter.errors import (
    PyVMUncaughtException,
    NoSourceError,
)
from shark.compiler.byte_code_interpreter.vmtrace import PyVMTraced, format_instruction


def exec_code_object(
    code,
    env,
    mlir_context: ir.Context,
    mlir_location: ir.Location,
    mlir_module: ir.Module,
    callback=None,
    format_instruction=format_instruction,
):
    if callback:
        vm = PyVMTraced(
            callback,
            mlir_context,
            mlir_location,
            mlir_module,
            format_instruction_func=format_instruction,
        )
        try:
            vm.run_code(code, f_globals=env)
        except PyVMUncaughtException:
            vm.last_exception = event_arg = (
                vm.last_exception[0],
                vm.last_exception[1],
                vm.last_traceback,
            )
            callback("fatal", 0, "fatalOpcode", 0, -1, event_arg, [], vm)
    else:
        vm = PyVM(
            mlir_context,
            mlir_location,
            mlir_module,
            format_instruction_func=format_instruction,
        )
        try:
            vm.run_code(code, f_globals=env)
        except PyVMUncaughtException:
            pass

    return vm


BUILTINS = sys.modules["builtins"]


def rsplit1(s, sep):
    """The same as s.rsplit(sep, 1), but works in 2.3"""
    parts = s.split(sep)
    return sep.join(parts[:-1]), parts[-1]


def run_python_module(
    modulename,
    args,
    mlir_context: ir.Context,
    mlir_location: ir.Location,
    mlir_module: ir.Module,
):
    """Run a python module, as though with ``python -m name args...``.

    `modulename` is the name of the module, possibly a dot-separated name.
    `args` is the argument array to present as sys.argv, including the first
    element naming the module being executed.

    """
    openfile = None
    glo, loc = globals(), locals()
    try:
        try:
            # Search for the module - inside its parent package, if any - using
            # standard import mechanics.
            if "." in modulename:
                packagename, name = rsplit1(modulename, ".")
                package = __import__(packagename, glo, loc, ["__path__"])
                searchpath = package.__path__
            else:
                packagename, name = None, modulename
                searchpath = None  # "top-level search" in imp.find_module()
            openfile, pathname, _ = imp.find_module(name, searchpath)

            # Complain if this is a magic non-file module.
            if openfile is None and pathname is None:
                raise NoSourceError("module does not live in a file: %r" % modulename)

            # If `modulename` is actually a package, not a mere module, then we
            # pretend to be Python 2.7 and try running its __main__.py script.
            if openfile is None:
                packagename = modulename
                name = "__main__"
                package = __import__(packagename, glo, loc, ["__path__"])
                searchpath = package.__path__
                openfile, pathname, _ = imp.find_module(name, searchpath)
        except ImportError:
            _, err, _ = sys.exc_info()
            raise NoSourceError(str(err))
    finally:
        if openfile:
            openfile.close()

    # Finally, hand the file off to run_python_file for execution.
    args[0] = pathname
    run_python_file(
        pathname, args, mlir_context, mlir_location, mlir_module, package=packagename
    )


def run_python_file(
    filename,
    args,
    mlir_context: ir.Context,
    mlir_location: ir.Location,
    mlir_module: ir.Module,
    package=None,
    callback=None,
    format_instruction=format_instruction,
):
    """Run a python file as if it were the main program on the command line.

    `filename` is the path to the file to execute, it need not be a .py file.
    `args` is the argument array to present as sys.argv, including the first
    element naming the file being executed.  `package` is the name of the
    enclosing package, if any.

    If `callback` is not None, it is a function which is called back as the
    execution progresses. This can be used for example in a debugger, or
    for custom tracing or statistics gathering.
    """
    # Create a module to serve as __main__
    old_main_mod = sys.modules["__main__"]
    main_mod = imp.new_module("__main__")
    sys.modules["__main__"] = main_mod
    main_mod.__file__ = filename
    if package:
        main_mod.__package__ = package
    main_mod.__builtins__ = BUILTINS

    # set sys.argv and the first path element properly.
    old_argv = sys.argv
    old_path0 = sys.path[0]

    # note: the type of args is na tuple; we want type(sys.argv) == list
    sys.argv = [filename] + list(args)

    if package:
        sys.path[0] = ""
    else:
        sys.path[0] = os.path.abspath(os.path.dirname(filename))

    try:
        # Open the source or bytecode file.
        source_file = tokenize.open(filename)
        source = source_file.read()
        source_file.close()

        # We have the source.  `compile` still needs the last line to be clean,
        # so make sure it is, then compile a code object from it.
        if not source or source[-1] != "\n":
            source += "\n"
        code = compile(source, filename, "exec")

        # Execute the source file.
        vm = exec_code_object(
            code,
            main_mod.__dict__,
            mlir_context,
            mlir_location,
            mlir_module,
            callback,
            format_instruction=format_instruction,
        )
    finally:
        # Restore the old __main__
        sys.modules["__main__"] = old_main_mod

        # Restore the old argv and path
        sys.argv = old_argv
        sys.path[0] = old_path0

    return vm.mlir_module

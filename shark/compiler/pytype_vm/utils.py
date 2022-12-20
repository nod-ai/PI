import logging

from pytype.abstract import abstract_utils
from pytype.abstract import abstract
from pytype.vm_utils import _check_defaults

log = logging.getLogger(__name__)


def make_function(
    name, node, code, globs, defaults, kw_defaults, closure, annotations, opcode, ctx
):
    """Create a function or closure given the arguments."""
    if closure:
        closure = tuple(c for c in abstract_utils.get_atomic_python_constant(closure))
        log.info("closure: %r", closure)
    if not name:
        name = abstract_utils.get_atomic_python_constant(code).co_name
    if not name:
        name = "<lambda>"
    val = abstract.InterpreterFunction.make(
        name,
        def_opcode=opcode,
        code=abstract_utils.get_atomic_python_constant(code),
        f_locals=ctx.vm.frame.f_locals,
        f_globals=globs,
        defaults=defaults,
        kw_defaults=kw_defaults,
        closure=closure,
        annotations=annotations,
        ctx=ctx,
    )
    var = ctx.program.NewVariable()
    var.AddBinding(val, code.bindings, node)
    _check_defaults(node, val, ctx)
    if val.signature.annotations:
        ctx.vm.functions_type_params_check.append((val, ctx.vm.frame.current_opcode))
    return var

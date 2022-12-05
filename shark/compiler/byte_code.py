from textwrap import dedent

from bytecode import Bytecode, Instr


def print_insts(insts):
    for b in insts:
        insts.append(b)
        print(b)
        if isinstance(b.arg, int):
            arg_s = b.arg

        elif isinstance(b.arg, str):
            arg_s = f"'{b.arg}'"
        else:
            arg_s = str(b.arg)
        print(f"Instr('{b.name}', {arg_s}),")


def strip_return(insts: list[Instr], strip_return_value=False):
    assert insts[-1].name == "RETURN_VALUE"
    if strip_return_value:
        assert insts[-2].name.startswith("LOAD"), f"unexpected instruction {insts[-2]}"
    else:
        assert (
            insts[-2].name == "LOAD_CONST" and insts[-2].arg is None
        ), f"return value being used {insts[-2]}"
    insts.pop()
    insts.pop()
    if insts[-1].name == "PRINT_EXPR":
        # no clue but it comes when you compile a single call
        insts.pop()
    return insts


def replace_with_fast(insts: list[Instr], replacements=None):
    if replacements is None:
        replacements = set()
    for i, inst in enumerate(insts):
        if inst.name in {"STORE_NAME", "STORE_GLOBAL"}:
            insts[i] = Instr("STORE_FAST", inst.arg)
            replacements.add(inst.arg)
        elif inst.name in {"LOAD_NAME", "LOAD_GLOBAL"}:
            assert inst.arg in replacements
            insts[i] = Instr("LOAD_FAST", inst.arg)


def mlir_compile(fn):
    prolog = compile(
        dedent(
            """\
    import shark
    import shark.ir
    from shark.dialects import linalg
    from shark.ir import F32Type, InsertionPoint, Context, Location, Module
    ctx = Context()
    location = Location.unknown(context=ctx)
    ctx.__enter__()
    location.__enter__()
    module = Module.create()
    ip = InsertionPoint(module.body)
    ip.__enter__()
    """
        ),
        "",
        "exec",
    )
    prolog_insts = Bytecode.from_code(prolog)
    replacements = set()
    prolog_insts = strip_return(prolog_insts)
    replace_with_fast(prolog_insts, replacements)

    epilogue = compile(
        dedent(
            """\
    ip.__exit__(None, None, None)
    location.__exit__(None, None, None)
    ctx.__exit__(None, None, None)
    """
        ),
        "",
        "exec",
    )
    epilog_insts = Bytecode.from_code(epilogue)
    epilog_insts = strip_return(epilog_insts)
    replace_with_fast(epilog_insts, replacements)

    bytecode = Bytecode.from_code(fn.__code__)
    insts = list(bytecode)
    replace_with_fast(insts, replacements)
    assert insts[-1].name == "RETURN_VALUE"
    return_value = None
    if insts[-2].name.startswith("LOAD"):
        return_value = insts[-2]
    insts = strip_return(insts, strip_return_value=True)
    if return_value is not None:
        epilog_insts.append(return_value)
    if return_value is not None:
        epilog_insts.extend([Instr("LOAD_FAST", "module"), Instr("BUILD_TUPLE", 2)])
    epilog_insts.append(Instr("RETURN_VALUE"))
    bytecode.clear()
    bytecode.extend(prolog_insts + insts + epilog_insts)

    fn.__code__ = bytecode.to_code()
    return fn

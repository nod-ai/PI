import re

from shark._mlir_libs._mlir.ir import Type, Context, InsertionPoint, Location

from shark.dialects import memref
from compiler_utils import add_dummy_value, traverse_op_region_block_iterators


def promote_alloc(module):
    def parse_memref_type_str(s):
        shape = list(map(int, re.findall(r"(\d)+x", s)))
        typ = re.findall(r"x([a-z].*)>", s)[0]
        typ = Type.parse(typ, Context())
        operand_segment_sizes = []
        for s in shape:
            segment = []
            for i in range(s):
                segment.append(add_dummy_value())
            operand_segment_sizes.append(segment)

        return operand_segment_sizes

    memrefs_to_erase = set()

    def handler(mem_alloc):
        res_type = mem_alloc.results.types[0]
        # operand_segment_sizes = parse_memref_type_str(str(res_type))
        entry_block = module.body.operations[0].regions[0].blocks[0]
        with InsertionPoint.at_block_begin(entry_block), Location.unknown():
            op = memref.AllocaOp(res_type, [], [])
            # print(op)
            mem_alloc.operation.replace_all_uses_with(op.operation)
            memrefs_to_erase.add(mem_alloc)

    with Context(), Location.unknown():
        traverse_op_region_block_iterators(
            module.operation,
            lambda op: handler(op) if op.operation.name == "memref.alloc" else None,
        )

        for memref_to_erase in memrefs_to_erase:
            memref_to_erase.operation.erase()

        traverse_op_region_block_iterators(
            module.operation,
            lambda op: op.operation.erase() or Exception("done")
            if op.operation.name == "memref.dealloc"
            else None,
        )


def affine_store_load_forwarding(module):
    def handler(affine_for):
        stores_to_forward = set()
        loads_to_delete = set()
        ops = list(affine_for.operation.regions[0].blocks[0].operations)
        for i, op in reversed(list(enumerate(ops))):
            if op.operation.name == "affine.load":
                loads_to_delete.add(op)
                buffer, *idxs = op.operands
                j = i - 1
                while j >= 0:
                    other_op = ops[j]
                    if other_op.operation.name == "affine.store":
                        val, other_buffer, *other_idxs = other_op.operands
                        if idxs == other_idxs:
                            stores_to_forward.add((val, other_op, op))
                            break
                    j -= 1
        if stores_to_forward:
            for i, (val, store, load) in enumerate(stores_to_forward):
                load.operation.replace_all_uses_with(val.owner)
                store.operation.erase()
                load.operation.erase()

    with Context(), Location.unknown():
        traverse_op_region_block_iterators(
            module.operation,
            lambda op: handler(op) if op.operation.name == "affine.for" else None,
        )


def unrolling_pipeline(unroll_factor):
    return [
        # f"func.func(affine-loop-unroll{{unroll-full unroll-full-threshold={unroll_factor}}})",
        f"func.func(affine-loop-unroll{{unroll-factor={unroll_factor} unroll-up-to-factor=1}})"
        if unroll_factor < 100
        else f"func.func(affine-loop-unroll{{unroll-full unroll-full-threshold={unroll_factor}}})",
        # f"func.func(affine-loop-unroll{{unroll-factor={unroll_factor} unroll-up-to-factor=0}})",
        # f"func.func(affine-loop-unroll-jam{{unroll-jam-factor={unroll_factor}}})",
    ]

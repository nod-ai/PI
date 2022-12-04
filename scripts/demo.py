import shark.dialects.memref as memref
from shark._mlir_libs._mlir.ir import (
    InsertionPoint,
    IndexType,
    Location,
    F32Type,
    MemRefType,
)
from shark.passmanager import PassManager
from shark.dialects import func, scf, arith, linalg
from shark.ir import Context


def icst(x):
    return arith.ConstantOp.create_index(x)


def fcst(x):
    return arith.ConstantOp(f32, x)


with Context() as ctx, Location.unknown():
    f32 = F32Type.get()
    index_type = IndexType.get()

    @func.FuncOp.from_py_func(MemRefType.get((12, 12), f32))
    def simple_loop(mem):
        lb = icst(0)
        ub = icst(42)
        step = icst(2)
        tens = linalg.InitTensorOp([3, 4], f32)
        loop1 = scf.ForOp(lb, ub, step, [])
        with InsertionPoint(loop1.body):
            loop2 = scf.ForOp(lb, ub, step, [])
            with InsertionPoint(loop2.body):
                loop3 = scf.ForOp(lb, ub, step, [])
                with InsertionPoint(loop3.body):
                    f = fcst(1.14)
                    memref.StoreOp(
                        f, mem, [loop2.induction_variable, loop3.induction_variable]
                    )
                    scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])
        return

    # fn = simple_loop()
    simple_loop.func_op.print()

    pass_man = PassManager.parse("async-to-async-runtime")

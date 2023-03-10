import pi
from pi.testing.framework import TestUtils
from pi.testing.registry import register_test_case
from pi.utils.annotations import export, annotate_args


class AddIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.int64, True), ([], pi.int64, True)])
    def forward(self, lhs, rhs):
        return lhs + rhs


@register_test_case(module_factory=lambda: AddIntModule())
def AddIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


class AtenIntBoolOpConstFalseModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None])
    def forward(self):
        return pi.ops.aten.Int(False)


@register_test_case(module_factory=lambda: AtenIntBoolOpConstFalseModule())
def AtenIntBoolOpConstFalseModule_basic(module, tu: TestUtils):
    module.forward()


class AtenIntBoolOpConstTrueModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None])
    def forward(self):
        return pi.ops.aten.Int(True)


@register_test_case(module_factory=lambda: AtenIntBoolOpConstTrueModule())
def AtenIntBoolOpConstTrueModule_basic(module, tu: TestUtils):
    module.forward()


class AtenIntBoolOpModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.bool, True)])
    def forward(self, x):
        return pi.ops.aten.Int(x)


@register_test_case(module_factory=lambda: AtenIntBoolOpModule())
def AtenIntBoolOpModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=0, high=2).bool())


class AtenIntTensorByteDtypeModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.uint8, True)])
    def forward(self, val):
        return val


@register_test_case(module_factory=lambda: AtenIntTensorByteDtypeModule())
def AtenIntTensorByteDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100).to(dtype=pi.uint8))


class AtenIntTensorCharDtypeModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.int8, True)])
    def forward(self, val):
        return val


@register_test_case(module_factory=lambda: AtenIntTensorCharDtypeModule())
def AtenIntTensorCharDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100).to(dtype=pi.int8))


class BoolFloatConstantModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None])
    def forward(self):
        return pi.ops.aten.Bool(5.0)


@register_test_case(module_factory=lambda: BoolFloatConstantModule())
def BoolFloatConstantModule_basic(module, tu: TestUtils):
    module.forward()


class BoolFloatFalseModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.float64, True)])
    def forward(self, a):
        sub = a - a
        return pi.ops.aten.Bool(sub)


@register_test_case(module_factory=lambda: BoolFloatFalseModule())
def BoolFloatFalseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(low=0.5).double())


class BoolFloatTrueModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.float64, True)])
    def forward(self, a):
        return pi.ops.aten.Bool(a)


@register_test_case(module_factory=lambda: BoolFloatTrueModule())
def BoolFloatTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(low=0.5).double())


class BoolIntConstantModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None])
    def forward(self):
        return pi.ops.aten.Bool(5)


@register_test_case(module_factory=lambda: BoolIntConstantModule())
def BoolIntConstantModule_basic(module, tu: TestUtils):
    module.forward()


class BoolIntFalseModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.int64, True)])
    def forward(self, a):
        sub = a - a
        return pi.ops.aten.Bool(sub)


@register_test_case(module_factory=lambda: BoolIntFalseModule())
def BoolIntFalseModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=1, high=100))


class BoolIntTrueModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.int64, True)])
    def forward(self, a):
        return pi.ops.aten.Bool(a)


@register_test_case(module_factory=lambda: BoolIntTrueModule())
def BoolIntTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=1, high=100))


class CeilFloatModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.float64, True), ([], pi.float64, True)])
    def forward(self, lhs, rhs):
        sub = lhs - rhs
        return pi.ops.aten.ceil(sub)


# @register_test_case(module_factory=lambda: CeilFloatModule())
# def CeilFloatModule_basic(module, tu: TestUtils):
#     module.forward(pi.rand(()).double(), pi.rand(()).double())


class DivFloatModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.float64, True), ([], pi.float64, True)])
    def forward(self, lhs, rhs):
        return lhs / rhs


# @register_test_case(module_factory=lambda: DivFloatModule())
# def DivFloatModule_basic(module, tu: TestUtils):
#     module.forward(pi.rand(()).double(), pi.rand(()).double())


class DivIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.int64, True), ([], pi.int64, True)])
    def forward(self, lhs, rhs):
        return pi.ops.aten.div(lhs, rhs)


@register_test_case(module_factory=lambda: DivIntModule())
def DivIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-10, high=10), tu.randint(low=3, high=10))


class EqIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.int64, True), ([], pi.int64, True)])
    def forward(self, lhs, rhs):
        return lhs == rhs


@register_test_case(module_factory=lambda: EqIntModule())
def EqIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


class GeFloatIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.float64, True), ([], pi.int64, True)])
    def forward(self, lhs, rhs):
        return lhs >= rhs


# @register_test_case(module_factory=lambda: GeFloatIntModule())
# def GeFloatIntModule_basic(module, tu: TestUtils):
#     module.forward(pi.randn(()).double(), tu.randint(low=-100, high=100))


class GeFloatModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.float64, True), ([], pi.float64, True)])
    def forward(self, lhs, rhs):
        return lhs >= rhs


# @register_test_case(module_factory=lambda: GeFloatModule())
# def GeFloatModule_basic(module, tu: TestUtils):
#     module.forward(pi.randn(()).double(), pi.randn(()).double())


class GtFloatIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.float64, True), ([], pi.int64, True)])
    def forward(self, lhs, rhs):
        return lhs > rhs


class GeIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([], pi.int64, True),
            ([], pi.int64, True),
        ]
    )
    def forward(self, lhs, rhs):
        return pi.ops.aten.ge(lhs, rhs)


@register_test_case(module_factory=lambda: GeIntModule())
def GeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


# @register_test_case(module_factory=lambda: GtFloatIntModule())
# def GtFloatIntModule_basic(module, tu: TestUtils):
#     module.forward(pi.randn(()).double(), tu.randint(low=-100, high=100))


class GtIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.int64, True), ([], pi.int64, True)])
    def forward(self, lhs, rhs):
        return lhs > rhs


@register_test_case(module_factory=lambda: GtIntModule())
def GtIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


class MulIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.int64, True), ([], pi.int64, True)])
    def forward(self, lhs, rhs):
        return lhs * rhs


@register_test_case(module_factory=lambda: MulIntModule())
def MulIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


class NeFloatIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.float64, True), ([], pi.int64, True)])
    def forward(self, lhs, rhs):
        return lhs != rhs


# @register_test_case(module_factory=lambda: NeFloatIntModule())
# def NeFloatIntModule_basic(module, tu: TestUtils):
#     module.forward(pi.randn(()).double(), tu.randint(low=-100, high=100))


class NeIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.int64, True), ([], pi.int64, True)])
    def forward(self, lhs, rhs):
        return lhs != rhs


@register_test_case(module_factory=lambda: NeIntModule())
def NeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


class ScalarImplicitFloatModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.float64, True)])
    def forward(self, x):
        return pi.ops.aten.ScalarImplicit(x)


@register_test_case(module_factory=lambda: ScalarImplicitFloatModule())
def ScalarImplicitFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand().double())


class ScalarImplicitIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.int64, True)])
    def forward(self, x):
        return pi.ops.aten.ScalarImplicit(x)


@register_test_case(module_factory=lambda: ScalarImplicitIntModule())
def ScalarImplicitIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100))


class SqrtIntConstantModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None])
    def forward(self):
        return pi.ops.aten.sqrt(5)


@register_test_case(module_factory=lambda: SqrtIntConstantModule())
def SqrtIntConstantModule_basic(module, tu: TestUtils):
    module.forward()


class SqrtIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.int64, True)])
    def forward(self, a):
        return pi.ops.aten.sqrt(a)


@register_test_case(module_factory=lambda: SqrtIntModule())
def SqrtIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(high=10))


class SubFloatModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.float64, True), ([], pi.float64, True)])
    def forward(self, lhs, rhs):
        return lhs - rhs


# @register_test_case(module_factory=lambda: SubFloatModule())
# def SubFloatModule_basic(module, tu: TestUtils):
#     module.forward(pi.rand(()).double(), pi.rand(()).double())


class SubIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.int64, True), ([], pi.int64, True)])
    def forward(self, lhs, rhs):
        return lhs - rhs


@register_test_case(module_factory=lambda: SubIntModule())
def SubIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


class TensorToBool(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], pi.bool, True)])
    def forward(self, x):
        return x


class TensorToBoolZeroRank(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.bool, True)])
    def forward(self, x):
        return x


@register_test_case(module_factory=lambda: TensorToBoolZeroRank())
def TensorToBoolZeroRank_basic(module, tu: TestUtils):
    module.forward(pi.tensor(1, dtype=pi.bool))


@register_test_case(module_factory=lambda: TensorToBool())
def TensorToBool_basic(module, tu: TestUtils):
    module.forward(pi.tensor([[1]], dtype=pi.bool))


class TensorToFloat(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], pi.float64, True)])
    def forward(self, x):
        return x


class TensorToFloatZeroRank(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.float64, True)])
    def forward(self, x):
        return x


# @register_test_case(module_factory=lambda: TensorToFloatZeroRank())
# def TensorToFloatZeroRank_basic(module, tu: TestUtils):
#     module.forward(pi.rand((), dtype=pi.float64))


@register_test_case(module_factory=lambda: TensorToFloat())
def TensorToFloat_basic(module, tu: TestUtils):
    module.forward(pi.rand((1, 1), dtype=pi.float64))


class TensorToInt(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], pi.int64, True)])
    def forward(self, x):
        return x


class TensorToIntZeroRank(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], pi.int64, True)])
    def forward(self, x):
        return x


@register_test_case(module_factory=lambda: TensorToIntZeroRank())
def TensorToIntZeroRank_basic(module, tu: TestUtils):
    module.forward(tu.randint(high=10))


@register_test_case(module_factory=lambda: TensorToInt())
def TensorToInt_basic(module, tu: TestUtils):
    module.forward(tu.randint(1, 1, high=10))


class UnsafeViewCollapseDynamicWithAtenSizeIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1, -1], pi.float32, True),
            ([], pi.int64, True),
            ([], pi.int64, True),
        ]
    )
    def forward(self, a, b, c):
        return pi.ops.aten._unsafe_view(a, [a.size(0), b, c, a.size(3), 384])


# @register_test_case(
#     module_factory=lambda: UnsafeViewCollapseDynamicWithAtenSizeIntModule()
# )
# def UnsafeViewCollapseDynamicWithAtenSizeIntModule_basic(module, tu: TestUtils):
#     module.forward(tu.rand(2, 3, 5, 4, 12, 32), pi.tensor(3), pi.tensor(5))


class ViewCollapseDynamicWithAtenSizeIntModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1, -1], pi.float32, True),
            ([], pi.int64, True),
            ([], pi.int64, True),
        ]
    )
    def forward(self, a, b, c):
        return a.view(a.size(0), b, c, a.size(3), 384)


# @register_test_case(module_factory=lambda: ViewCollapseDynamicWithAtenSizeIntModule())
# def ViewCollapseDynamicWithAtenSizeIntModule_basic(module, tu: TestUtils):
#     module.forward(tu.rand(2, 3, 5, 4, 12, 32), pi.tensor(3), pi.tensor(5))


class AtenSubFloatModule(pi.nn.Module):
    def __init__(self):
        super().__init__()
        self.value1 = 1.0
        self.value2 = 2.0

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return pi.sub(self.value1, self.value2)


@register_test_case(module_factory=lambda: AtenSubFloatModule())
def AtenSubFloatModule_basic(module, tu: TestUtils):
    module.forward()

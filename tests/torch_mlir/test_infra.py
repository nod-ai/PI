import pi
from pi.utils import annotate_args, export, mlir_mod_ctx

pi.nn.Module.train = lambda *args, **kwargs: None
from infra.util import TestUtils, register_test_case, PIConfig, GLOBAL_TEST_REGISTRY


class ElementwiseAddModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], pi.float32, True),
            ([], pi.float32, True),
        ]
    )
    def forward(self, a, b):
        return a + b


@register_test_case(module_factory=lambda: ElementwiseAddModule())
def ElementwiseAddModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand())


class Add_Module(pi.nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor = pi.ones(2, 3)

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], pi.float32, True),
        ]
    )
    def forward(self, x):
        return pi.ops.aten.add_(x, self.tensor)


@register_test_case(module_factory=lambda: Add_Module())
def Add_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3))


class AtenIntTensorCharDtypeModule(pi.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([], pi.int8, True),
        ]
    )
    def forward(self, val):
        return int(val)


@register_test_case(module_factory=lambda: AtenIntTensorCharDtypeModule())
def AtenIntTensorCharDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100).to(dtype=pi.int8))


class Test:
    def test_place_holders(self):
        tu = TestUtils()
        a, b = tu.rand(4), tu.rand()
        a, b = a.to_nonvalue_tensor_type(), b.to_nonvalue_tensor_type()

    def test_cases(self):
        tests = sorted(GLOBAL_TEST_REGISTRY.values(), key=lambda t: t.unique_name)
        assert tests, "failed to load tests"
        pi_config = PIConfig()

        with mlir_mod_ctx():
            for test in tests:
                pi_mlir_module = pi_config.compile(test)
                print(pi_mlir_module)

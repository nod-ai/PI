import logging
import unittest

# noinspection PyUnresolvedReferences
from pi._pi_mlir import (
    TorchListOfNonValueTensorType as TorchListOfTensor,
    TorchListOfTorchBoolType as TorchListOfTorchBool,
    TorchListOfTorchIntType as TorchListOfTorchInt,
    TorchListOfTorchFloatType as TorchListOfTorchFloat,
    TorchListOfTorchStringType as TorchListOfTorchString,
)
from pi._pi_mlir import (
    Torch_FloatType,
    Torch_IntType,
    Torch_BoolType,
)
from torch_mlir.dialects import torch as torch_dialect
from typeguard import check_argument_types, check_return_type, TypeCheckError, CallMemo

from pi._torch_wrappers import any as torch_any, Float, NumToTensor, randint
from pi.dispatcher.function import NotFoundLookupError
from pi.mlir.utils import cm
from pi.types_ import Torch_Value, Torch_List

FORMAT = "%(asctime)s, %(levelname)-8s [%(filename)s:%(module)s:%(funcName)s:%(lineno)d] %(message)s"
formatter = logging.Formatter(FORMAT)
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


def FloatCheckArgReturn(
    a: Torch_Value[Torch_FloatType],
) -> Torch_Value[Torch_FloatType]:
    assert check_argument_types()
    retval = Torch_Value(torch_dialect.AtenFloatScalarOp(a).result)
    assert check_return_type(retval)
    return retval


def FloatCheckFail(a: Torch_Value[Torch_FloatType]) -> Torch_Value[Torch_IntType]:
    assert check_argument_types()
    retval = Torch_Value(torch_dialect.AtenFloatScalarOp(a).result)
    assert check_return_type(retval)
    return retval


class TestTypeChecking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._connection = cm()

    def run(self, result=None):
        with cm() as module:
            self.module = module
            super(TestTypeChecking, self).run(result)

    def test_smoke(self):
        t = Torch_Value(torch_dialect.ConstantIntOp(1).result)
        # print(t)
        tt = Torch_Value(torch_dialect.ConstantIntOp(2).result)
        # print(tt)
        typ = Torch_List.of(Torch_IntType())
        l = Torch_Value(torch_dialect.PrimListConstructOp(typ, [t, tt]).result)
        # print(l)

    def test_type_checking(self):
        t = Torch_Value(torch_dialect.ConstantFloatOp(1.0).result)
        f = Float(t)
        f = FloatCheckArgReturn(t)

    def test_lists(self):
        t = Torch_Value(torch_dialect.ConstantIntOp(1).result)
        # print(t)
        tt = Torch_Value(torch_dialect.ConstantIntOp(2).result)
        l = torch_any([t, tt])

        typ = Torch_List.of(Torch_IntType())
        l = Torch_Value(torch_dialect.PrimListConstructOp(typ, [t, tt]).result)
        with self.assertRaises(
            NotFoundLookupError,
            # r"Not correct type param \(<class 'pi\._mlir\.TorchListOfTorchBoolType'>\): TorchListOfTorchInt\(\!torch\.list<int>\)",
        ):
            torch_any(l)

        ll = torch_any([True, True])

        t = Torch_Value(torch_dialect.ConstantBoolOp(True).result)
        tt = Torch_Value(torch_dialect.ConstantBoolOp(True).result)
        ll = torch_any([t, tt])

        typ = Torch_List.of(Torch_BoolType())
        l = Torch_List(torch_dialect.PrimListConstructOp(typ, [t, tt]).result)
        ll = torch_any(l)

    def test_rand(self):
        r = randint(1, 10, [1, 2, 3])

    def test_tensor(self):
        t = Torch_Value(torch_dialect.ConstantIntOp(1).result)
        n = NumToTensor(t)

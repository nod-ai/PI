import ast
from textwrap import dedent

import numpy as np
from pyframe_eval import Dynamo

import pi
from pi import nn
from pi.mlir import Torch_IntValue
from pi.mlir.compile import pipile
from pi.utils import TensorPlaceholder, mlir_mod_ctx, tensor_op
from pi.utils.ast_rewrite import PiIntFloatBool, rewrite_ast_callback
from util import check_correct


class MyConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3)

    def forward(self, x):
        y = self.conv(x)
        return y


class MyIntModule(nn.Module):
    def forward(self, x):
        y = int(x)
        return y


class TestDynamo:
    def test_rewriter(self):
        tree = ast.parse(
            dedent(
                """\
            int(bob)
        """
            )
        )
        tree = PiIntFloatBool().visit(tree)
        check_correct("__import__('pi').pi_int(bob)", ast.unparse(tree))

        tree = ast.parse(
            dedent(
                """\
            float(bob)
        """
            )
        )
        tree = PiIntFloatBool().visit(tree)
        check_correct("__import__('pi').pi_float(bob)", ast.unparse(tree))

        tree = ast.parse(
            dedent(
                """\
            bool(bob)
        """
            )
        )
        tree = PiIntFloatBool().visit(tree)
        check_correct("__import__('pi').pi_bool(bob)", ast.unparse(tree))

        tree = ast.parse(
            dedent(
                """\
            int(bob, 1)
        """
            )
        )
        tree = PiIntFloatBool().visit(tree)
        check_correct("__import__('pi').pi_int(bob, 1)", ast.unparse(tree))

    def test_dynamo(self):
        def foo():
            x = Torch_IntValue(5)
            y = int(x)
            return y

        with mlir_mod_ctx() as module, Dynamo(rewrite_ast_callback):
            z = foo()

        assert z == 5
        check_correct(
            dedent(
                """\
        module {
          %int5 = torch.constant.int 5
        }
        """
            ),
            module,
        )

    def test_dynamo_with_module(self):
        test_module = MyConv2d()

        with mlir_mod_ctx() as module:
            t = tensor_op(np.random.randn(1, 3, 32, 32))
            with Dynamo(rewrite_ast_callback):
                z = test_module(t)
            check_correct(
                dedent(
                    """\
            module {
              %0 = torch.tensor.literal(dense<""> : tensor<1x3x32x32xf64>) : !torch.tensor<[1,3,32,32],f64>
              %1 = torch.tensor.literal(dense<5.4055683432028091E-31> : tensor<1xf64>) : !torch.tensor<[1],f64>
              %2 = torch.tensor.literal(dense<[]> : tensor<1x3x3x3xf64>) : !torch.tensor<[1,3,3,3],f64>
              %int1 = torch.constant.int 1
              %int1_0 = torch.constant.int 1
              %3 = torch.prim.ListConstruct %int1, %int1_0 : (!torch.int, !torch.int) -> !torch.list<int>
              %int0 = torch.constant.int 0
              %int0_1 = torch.constant.int 0
              %4 = torch.prim.ListConstruct %int0, %int0_1 : (!torch.int, !torch.int) -> !torch.list<int>
              %int1_2 = torch.constant.int 1
              %int1_3 = torch.constant.int 1
              %5 = torch.prim.ListConstruct %int1_2, %int1_3 : (!torch.int, !torch.int) -> !torch.list<int>
              %int1_4 = torch.constant.int 1
              %6 = torch.aten.conv2d %0, %2, %1, %3, %4, %5, %int1_4 : !torch.tensor<[1,3,32,32],f64>, !torch.tensor<[1,3,3,3],f64>, !torch.tensor<[1],f64>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.tensor
            }
            """
                ),
                module,
            )

    def test_dynamo_with_pipile(self):
        test_module = MyConv2d()
        t = TensorPlaceholder((1, 3, 32, 32), dtype_=pi.float64)
        module = pipile(test_module, example_args=(t,))
        check_correct(
            dedent(
                """\
            module attributes {pi.module_name = "MyConv2d"} {
              func.func @forward(%arg0: !torch.vtensor<[1,3,32,32],f64>) -> !torch.vtensor<[1,1,30,30],f64> {
                %false = torch.constant.bool false
                %0 = torch.vtensor.literal(dense<[...]> : tensor<1x3x3x3xf64>) : !torch.vtensor<[1,3,3,3],f64>
                %int0 = torch.constant.int 0
                %int1 = torch.constant.int 1
                %1 = torch.vtensor.literal(dense<5.4055683432028091E-31> : tensor<1xf64>) : !torch.vtensor<[1],f64>
                %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
                %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
                %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
                %5 = torch.aten.convolution %arg0, %0, %1, %2, %3, %2, %false, %4, %int1 : !torch.vtensor<[1,3,32,32],f64>, !torch.vtensor<[1,3,3,3],f64>, !torch.vtensor<[1],f64>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,30,30],f64>
                return %5 : !torch.vtensor<[1,1,30,30],f64>
              }
            }
        """
            ),
            module,
        )

    def test_dynamo_with_module_ints(self):
        test_module = MyIntModule()
        t = TensorPlaceholder((1,), dtype_=pi.int64)
        module = pipile(test_module, example_args=(t,))
        check_correct(
            dedent(
                """\
        module attributes {pi.module_name = "MyIntModule"} {
          func.func @forward(%arg0: !torch.vtensor<[1],si64>) -> !torch.int {
            %0 = torch.aten.Int.Tensor %arg0 : !torch.vtensor<[1],si64> -> !torch.int
            return %0 : !torch.int
          }
        }
        """
            ),
            module,
        )

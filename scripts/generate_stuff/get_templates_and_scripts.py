import json
import os
from pathlib import Path
from textwrap import dedent

import requests


def get_tools_autograd():
    os.makedirs("tools/autograd", exist_ok=True)
    j = requests.get(
        "https://api.github.com/repos/pytorch/pytorch/contents/tools/autograd?ref=main"
    ).json()
    for record in j:
        path = record["path"]
        if record["type"] == "file":
            with open(path, "w") as f:
                file = requests.get(
                    f"https://raw.githubusercontent.com/pytorch/pytorch/main/{path}"
                ).text
                if "gen_python_functions" in path:
                    file = file.replace(
                        "from torchgen.yaml_utils import YamlLoader",
                        "from .torchgen.yaml_utils import YamlLoader",
                    )

                f.write(file)
        elif record["type"] == "dir":
            os.makedirs(path, exist_ok=True)

    os.makedirs("tools/autograd/torchgen", exist_ok=True)
    with open("tools/autograd/torchgen/yaml_utils.py", "w") as f:
        f.write(
            requests.get(
                f"https://raw.githubusercontent.com/pytorch/pytorch/main/torchgen/yaml_utils.py"
            ).text
        )

    with open("gen_pyi.py", "w") as f:
        f.write(
            requests.get(
                f"https://raw.githubusercontent.com/pytorch/pytorch/main/tools/pyi/gen_pyi.py"
            ).text
        )


def get_templates():
    templates = [
        "aten/src/ATen/native/native_functions.yaml",
        "aten/src/ATen/native/tags.yaml",
        "tools/autograd/deprecated.yaml",
        "torch/nn/functional.pyi.in",
        "torch/_C/_nn.pyi.in",
        "torch/_C/__init__.pyi.in",
        "torch/_C/_VariableFunctions.pyi.in",
        "torch/_C/return_types.pyi.in",
    ]

    for template in templates:
        path = Path(template)
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w") as f:
            f.write(
                requests.get(
                    f"https://raw.githubusercontent.com/pytorch/pytorch/main/{path}"
                ).text
            )

    with open("aten/src/ATen/native/native_functions.yaml", "r") as f:
        f = f.read()

    f = f.replace(
        dedent(
            """
        - func: resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)
          use_const_ref_for_mutable_tensors: True
          variants: function, method
          dispatch:
            CompositeExplicitAutograd: resize_as_
          autogen: resize_as, resize_as.out
          tags: inplace_view
        """
        ),
        dedent(
            """
        - func: resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)
          use_const_ref_for_mutable_tensors: True
          variants: function, method
          dispatch:
            CompositeExplicitAutograd: resize_as_
          autogen: resize_as, resize_as.out
        # tags: inplace_view
        """
        ),
    ).replace(
        dedent(
            """
            - func: _assert_async.msg(Tensor self, str assert_msg) -> ()
              dispatch:
                CPU: _assert_async_msg_cpu
                CUDA: _assert_async_msg_cuda
            """
        ),
        dedent(
            """
            # - func: _assert_async.msg(Tensor self, str assert_msg) -> ()
            #   dispatch:
            #     CPU: _assert_async_msg_cpu
            #     CUDA: _assert_async_msg_cuda
            """
        ),
    )

    with open("aten/src/ATen/native/native_functions.yaml", "w") as g:
        g.write(f)


get_tools_autograd()
get_templates()

import inspect
import re

import numpy as np
import torch

from pi.lazy_importer.run_lazy_imports import do_package_imports, do_hand_imports
from pi.lazy_importer import lazy_imports


#


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    total_dims = 1
    for dim in shape:
        total_dims *= dim
    values = []
    for _ in range(total_dims):
        values.append(np.random.random() * scale)
    return torch.tensor(data=values, dtype=torch.float).view(shape).contiguous()


def run(
    CTor,
    down_block_types=("CrossAttnDownBlock2D", "ResnetDownsampleBlock2D"),
    up_block_types=("UpBlock2D", "ResnetUpsampleBlock2D"),
):
    unet = CTor(
        **{
            "block_out_channels": (32, 64),
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 2,
            "sample_size": 32,
        }
    )
    unet.eval()
    batch_size = 4
    num_channels = 4
    sizes = (32, 32)

    noise = floats_tensor((batch_size, num_channels) + sizes)
    time_step = torch.tensor([10])
    encoder_hidden_states = floats_tensor((batch_size, 4, 32))
    output = unet(noise, time_step, encoder_hidden_states)


def make_linearized():
    def filter(ret):
        try:
            MODULE_TARGET = lambda x: re.match(
                r"(huggingface|torch|diffusers)", inspect.getmodule(x).__package__
            )
            return MODULE_TARGET(ret)
        except:
            return None

    lazy_imports.MODULE_TARGET = filter

    def _inner():

        from diffusers import UNet2DConditionModel

        run(
            UNet2DConditionModel,
            down_block_types=("CrossAttnDownBlock2D", "ResnetDownsampleBlock2D"),
            up_block_types=("UpBlock2D", "ResnetUpsampleBlock2D"),
        )
        run(
            UNet2DConditionModel,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "ResnetUpsampleBlock2D"),
        )

    prefix = "from pi.models.unet.prologue import CONFIG_NAME, LORA_WEIGHT_NAME"
    name = "unet_linearized"
    do_package_imports(_inner, prefix, name)


def run_linearized():
    from pi.models.unet import linearized

    run(linearized.UNet2DConditionModel)


if __name__ == "__main__":
    make_linearized()

import torch
import numpy as np

from pi.models.unet import UNet2DConditionModel
import torch_mlir

unet = UNet2DConditionModel(
    **{
        "block_out_channels": (32, 64),
        "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
        "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
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


def floats_tensor(shape, scale=1.0, rng=None, name=None):

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(np.random.random() * scale)

    return torch.tensor(data=values, dtype=torch.float).view(shape).contiguous()


noise = floats_tensor((batch_size, num_channels) + sizes)
time_step = torch.tensor([10])
encoder_hidden_states = floats_tensor((batch_size, 4, 32))

output = unet(noise, time_step, encoder_hidden_states)
print(output)

traced = torch.jit.trace(unet, (noise, time_step, encoder_hidden_states), strict=False)
frozen = torch.jit.freeze(traced)
print(frozen.graph)


module = torch_mlir.compile(
    frozen,
    (noise, time_step, encoder_hidden_states),
    use_tracing=True,
    output_type=torch_mlir.OutputType.RAW,
)
with open("unet.mlir", "w") as f:
    f.write(str(module))

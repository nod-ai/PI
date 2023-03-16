import os

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "diffusion_pytorch_model.bin"
FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"
SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
DIFFUSERS_DYNAMIC_MODULE_NAME = "diffusers_modules"
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "").upper() in ENV_VARS_TRUE_VALUES

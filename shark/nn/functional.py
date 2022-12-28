r"""Functional interface"""
from typing import Callable, List, Optional, Tuple, Union
import math
import warnings

import shark

from torch_mlir.dialects import torch as torch_dialect

from shark.types_ import dtype, size

__all__ = ["conv2d", "pad", "softmax"]

Tensor = shark.Tensor


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, size] = 1,
    padding: Union[int, size] = 0,
    dilation: Union[int, size] = 1,
    groups: int = 1,
) -> Tensor:
    return Tensor(
        torch_dialect.AtenConv2dOp(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )
    )


def pad(input, pad, mode="constant", value=None) -> Tensor:
    return Tensor(
        torch_dialect.AtenPadOp(
            input,
            pad,
            mode,
            value,
        )
    )


def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.size(
                0
            ), "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(
                0
            ), "Padding_idx must be within num_embeddings"
            padding_idx = weight.size(0) + padding_idx
    else:
        padding_idx = -1
    if max_norm is not None:
        raise NotImplementedError
    return shark.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)


def embedding_bag(
    input: Tensor,
    weight: Tensor,
    offsets: Optional[Tensor] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2,
    scale_grad_by_freq: bool = False,
    mode: str = "mean",
    sparse: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: Optional[int] = None,
) -> Tensor:
    # Check for backward compatibility.
    # Used to be embedding_bag(weight, input, ...)
    # Now is     embedding_bag(input, weight, ...)
    if weight.dtype == shark.long and input.is_floating_point():
        warnings.warn(
            "Argument order of nn.functional.embedding_bag was changed. "
            "Usage `embedding_bag(weight, input, ...)` is deprecated, "
            "and should now be `embedding_bag(input, weight, ...)`."
        )
        weight, input = input, weight

    if per_sample_weights is not None and input.size() != per_sample_weights.size():
        raise ValueError(
            "embedding_bag: If per_sample_weights ({}) is not None, "
            "then it must have the same shape as the input ({})".format(
                per_sample_weights.shape, input.shape
            )
        )

    if input.dim() == 2:
        if offsets is not None:
            type_str = "<unknown>"
            type_str = str(type(offsets))
            raise ValueError(
                "if input is 2D, then offsets has to be None"
                ", as input is treated is a mini-batch of"
                " fixed length sequences. However, found "
                "offsets of type {}".format(type_str)
            )
        offsets = shark.arange(0, input.numel(), input.size(1), dtype=input.dtype)

        input = input.reshape(-1)
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.reshape(-1)
    elif input.dim() == 1:
        if offsets is None:
            raise ValueError("offsets has to be a 1D Tensor but got None")
        if offsets.dim() != 1:
            raise ValueError("offsets has to be a 1D Tensor")
    else:
        raise ValueError(
            "input has to be 1D or 2D Tensor,"
            " but got Tensor of dimension {}".format(input.dim())
        )
    if mode == "sum":
        mode_enum = 0
    elif mode == "mean":
        mode_enum = 1
    elif mode == "max":
        mode_enum = 2

        if scale_grad_by_freq:
            raise ValueError(
                "max mode does not support scaling the gradient by the frequency"
            )

        if sparse:
            raise ValueError("max mode does not support sparse weights")

    else:
        raise ValueError("mode has to be one of sum, mean or max")

    if max_norm is not None:
        raise NotImplementedError
        # XXX: equivalent to
        # with torch.no_grad():
        #   torch.nembedding_renorm_
        # remove once script supports set_grad_enabled
        # _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)

    if per_sample_weights is not None and mode != "sum":
        raise NotImplementedError(
            "embedding_bag: per_sample_weights was not None. "
            "per_sample_weights is only supported for mode='sum' "
            "(got mode='{}'). Please open a feature request on GitHub.".format(mode)
        )
    ret, _, _, _ = shark.embedding_bag(
        weight,
        input,
        offsets,
        scale_grad_by_freq,
        mode_enum,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )
    return ret


def _get_softmax_dim(name: str, ndim: int, stacklevel: int) -> int:
    warnings.warn(
        "Implicit dimension choice for {} has been deprecated. "
        "Change the call to include dim=X as an argument.".format(name),
        stacklevel=stacklevel,
    )
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


def softmax(
    input: Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[dtype] = None,
) -> Tensor:
    if dim is None:
        dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret

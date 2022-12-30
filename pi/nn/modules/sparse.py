from typing import Optional, Union

from pi import Tensor
from .module import Module
from .. import functional as F
from ..parameter import UninitializedParameter
from .. import init

__all__ = ["Embedding", "EmbeddingBag"]


class Embedding(Module):
    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Union[Tensor, UninitializedParameter]
    freeze: bool
    sparse: bool

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Union[Tensor, UninitializedParameter]] = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"dtype": dtype}
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = UninitializedParameter(
                (num_embeddings, embedding_dim),
                **factory_kwargs,
            )
            # self.reset_parameters()
        else:
            raise NotImplementedError
            # assert list(_weight.shape) == [
            #     num_embeddings,
            #     embedding_dim,
            # ], "Shape of weight does not match num_embeddings and embedding_dim"
            # self.weight = (_weight, requires_grad=not _freeze)

        self.sparse = sparse

    def reset_parameters(self) -> None:
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)

    # @classmethod
    # def from_pretrained(
    #     cls,
    #     embeddings,
    #     freeze=True,
    #     padding_idx=None,
    #     max_norm=None,
    #     norm_type=2.0,
    #     scale_grad_by_freq=False,
    #     sparse=False,
    # ):
    #
    #     assert (
    #         embeddings.dim() == 2
    #     ), "Embeddings parameter is expected to be 2-dimensional"
    #     rows, cols = embeddings.shape
    #     embedding = cls(
    #         num_embeddings=rows,
    #         embedding_dim=cols,
    #         _weight=embeddings,
    #         _freeze=freeze,
    #         padding_idx=padding_idx,
    #         max_norm=max_norm,
    #         norm_type=norm_type,
    #         scale_grad_by_freq=scale_grad_by_freq,
    #         sparse=sparse,
    #     )
    #     return embedding


class EmbeddingBag(Module):

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "mode",
        "sparse",
        "include_last_offset",
        "padding_idx",
    ]

    num_embeddings: int
    embedding_dim: int
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    mode: str
    sparse: bool
    include_last_offset: bool
    padding_idx: Optional[int]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        include_last_offset: bool = False,
        padding_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(EmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        if _weight is None:
            self.weight = UninitializedParameter(
                (num_embeddings, embedding_dim), **factory_kwargs
            )
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = _weight
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            self.weight[self.padding_idx].fill_(0)

    def forward(
        self,
        input: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        return F.embedding_bag(
            input,
            self.weight,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights,
            self.include_last_offset,
            self.padding_idx,
        )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        s += ", mode={mode}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(
        cls,
        embeddings: Tensor,
        freeze: bool = True,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        include_last_offset: bool = False,
        padding_idx: Optional[int] = None,
    ) -> "EmbeddingBag":

        assert (
            embeddings.dim() == 2
        ), "Embeddings parameter is expected to be 2-dimensional"
        rows, cols = embeddings.shape
        embeddingbag = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            mode=mode,
            sparse=sparse,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        )
        embeddingbag.weight.requires_grad = not freeze
        return embeddingbag

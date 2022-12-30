import warnings

from .distance import PairwiseDistance
from .module import Module
from .. import functional as F
from .. import _reduction as _Reduction

from pi import Tensor
from typing import Callable, Optional

__all__ = [
    "L1Loss",
    "NLLLoss",
    "NLLLoss2d",
    "PoissonNLLLoss",
    "GaussianNLLLoss",
    "KLDivLoss",
    "MSELoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "HingeEmbeddingLoss",
    "MultiLabelMarginLoss",
    "SmoothL1Loss",
    "HuberLoss",
    "SoftMarginLoss",
    "CrossEntropyLoss",
    "MultiLabelSoftMarginLoss",
    "CosineEmbeddingLoss",
    "MarginRankingLoss",
    "MultiMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
    "CTCLoss",
]


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.weight: Optional[Tensor]


class L1Loss(_Loss):

    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(L1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)


class NLLLoss(_WeightedLoss):

    __constants__ = ["ignore_index", "reduction"]
    ignore_index: int

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(NLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class NLLLoss2d(NLLLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        warnings.warn(
            "NLLLoss2d has been deprecated. "
            "Please use NLLLoss instead as a drop-in replacement and see "
            "https://pyshark.org/docs/master/nn.html#shark.nn.NLLLoss for more details."
        )
        super(NLLLoss2d, self).__init__(
            weight, size_average, ignore_index, reduce, reduction
        )


class PoissonNLLLoss(_Loss):

    __constants__ = ["log_input", "full", "eps", "reduction"]
    log_input: bool
    full: bool
    eps: float

    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        size_average=None,
        eps: float = 1e-8,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(PoissonNLLLoss, self).__init__(size_average, reduce, reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps

    def forward(self, log_input: Tensor, target: Tensor) -> Tensor:
        return F.poisson_nll_loss(
            log_input,
            target,
            log_input=self.log_input,
            full=self.full,
            eps=self.eps,
            reduction=self.reduction,
        )


class GaussianNLLLoss(_Loss):

    __constants__ = ["full", "eps", "reduction"]
    full: bool
    eps: float

    def __init__(
        self, *, full: bool = False, eps: float = 1e-6, reduction: str = "mean"
    ) -> None:
        super(GaussianNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        return F.gaussian_nll_loss(
            input, target, var, full=self.full, eps=self.eps, reduction=self.reduction
        )


class KLDivLoss(_Loss):

    __constants__ = ["reduction"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> None:
        super(KLDivLoss, self).__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(
            input, target, reduction=self.reduction, log_target=self.log_target
        )


class MSELoss(_Loss):

    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)


class BCELoss(_WeightedLoss):

    __constants__ = ["reduction"]

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(
            input, target, weight=self.weight, reduction=self.reduction
        )


class BCEWithLogitsLoss(_Loss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: Optional[Tensor] = None,
    ) -> None:
        super(BCEWithLogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)
        self.weight: Optional[Tensor]
        self.pos_weight: Optional[Tensor]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            input,
            target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )


class HingeEmbeddingLoss(_Loss):

    __constants__ = ["margin", "reduction"]
    margin: float

    def __init__(
        self,
        margin: float = 1.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(HingeEmbeddingLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.hinge_embedding_loss(
            input, target, margin=self.margin, reduction=self.reduction
        )


class MultiLabelMarginLoss(_Loss):

    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(MultiLabelMarginLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.multilabel_margin_loss(input, target, reduction=self.reduction)


class SmoothL1Loss(_Loss):

    __constants__ = ["reduction"]

    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean", beta: float = 1.0
    ) -> None:
        super(SmoothL1Loss, self).__init__(size_average, reduce, reduction)
        self.beta = beta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)


class HuberLoss(_Loss):

    __constants__ = ["reduction", "delta"]

    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        super().__init__(reduction=reduction)
        self.delta = delta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.huber_loss(input, target, reduction=self.reduction, delta=self.delta)


class SoftMarginLoss(_Loss):

    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(SoftMarginLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.soft_margin_loss(input, target, reduction=self.reduction)


class CrossEntropyLoss(_WeightedLoss):

    __constants__ = ["ignore_index", "reduction", "label_smoothing"]
    ignore_index: int
    label_smoothing: float

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class MultiLabelSoftMarginLoss(_WeightedLoss):

    __constants__ = ["reduction"]

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(MultiLabelSoftMarginLoss, self).__init__(
            weight, size_average, reduce, reduction
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.multilabel_soft_margin_loss(
            input, target, weight=self.weight, reduction=self.reduction
        )


class CosineEmbeddingLoss(_Loss):

    __constants__ = ["margin", "reduction"]
    margin: float

    def __init__(
        self,
        margin: float = 0.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(CosineEmbeddingLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return F.cosine_embedding_loss(
            input1, input2, target, margin=self.margin, reduction=self.reduction
        )


class MarginRankingLoss(_Loss):

    __constants__ = ["margin", "reduction"]
    margin: float

    def __init__(
        self,
        margin: float = 0.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(MarginRankingLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return F.margin_ranking_loss(
            input1, input2, target, margin=self.margin, reduction=self.reduction
        )


class MultiMarginLoss(_WeightedLoss):

    __constants__ = ["p", "margin", "reduction"]
    margin: float
    p: int

    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(MultiMarginLoss, self).__init__(weight, size_average, reduce, reduction)
        if p != 1 and p != 2:
            raise ValueError("only p == 1 and p == 2 supported")
        assert weight is None or weight.dim() == 1
        self.p = p
        self.margin = margin

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.multi_margin_loss(
            input,
            target,
            p=self.p,
            margin=self.margin,
            weight=self.weight,
            reduction=self.reduction,
        )


class TripletMarginLoss(_Loss):

    __constants__ = ["margin", "p", "eps", "swap", "reduction"]
    margin: float
    p: float
    eps: float
    swap: bool

    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-6,
        swap: bool = False,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ):
        super(TripletMarginLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return F.triplet_margin_loss(
            anchor,
            positive,
            negative,
            margin=self.margin,
            p=self.p,
            eps=self.eps,
            swap=self.swap,
            reduction=self.reduction,
        )


class TripletMarginWithDistanceLoss(_Loss):

    __constants__ = ["margin", "swap", "reduction"]
    margin: float
    swap: bool

    def __init__(
        self,
        *,
        distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: str = "mean",
    ):
        super(TripletMarginWithDistanceLoss, self).__init__(
            size_average=None, reduce=None, reduction=reduction
        )
        self.distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = (
            distance_function if distance_function is not None else PairwiseDistance()
        )
        self.margin = margin
        self.swap = swap

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return F.triplet_margin_with_distance_loss(
            anchor,
            positive,
            negative,
            distance_function=self.distance_function,
            margin=self.margin,
            swap=self.swap,
            reduction=self.reduction,
        )


class CTCLoss(_Loss):

    __constants__ = ["blank", "reduction"]
    blank: int
    zero_infinity: bool

    def __init__(
        self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = False
    ):
        super(CTCLoss, self).__init__(reduction=reduction)
        self.blank = blank
        self.zero_infinity = zero_infinity

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        return F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            self.blank,
            self.reduction,
            self.zero_infinity,
        )


# TODO: L1HingeEmbeddingCriterion
# TODO: MSECriterion weight
# TODO: ClassSimplexCriterion

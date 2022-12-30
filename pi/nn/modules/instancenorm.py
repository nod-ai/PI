from pi import Tensor

from .batchnorm import _NormBase
from .. import functional as F

__all__ = [
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    # "LazyInstanceNorm1d",
    # "LazyInstanceNorm2d",
    # "LazyInstanceNorm3d",
]


class _InstanceNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(_InstanceNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _get_no_batch_dim(self):
        raise NotImplementedError

    def _handle_no_batch_input(self, input):
        return self._apply_instance_norm(input.unsqueeze(0)).squeeze(0)

    def _apply_instance_norm(self, input):
        return F.instance_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        # at version 1: removed running_mean and running_var when
        # track_running_stats=False (default)
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            for name in ("running_mean", "running_var"):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    "Unexpected running stats buffer(s) {names} for {klass} "
                    "with track_running_stats=False. If state_dict is a "
                    "checkpoint saved before 0.4.0, this may be expected "
                    "because {klass} does not track running stats by default "
                    "since 0.4.0. Please remove these keys from state_dict. If "
                    "the running stats are actually needed, instead set "
                    "track_running_stats=True in {klass} to enable them. See "
                    "the documentation of {klass} for details.".format(
                        names=" and ".join(
                            '"{}"'.format(k) for k in running_stats_keys
                        ),
                        klass=self.__class__.__name__,
                    )
                )
                for key in running_stats_keys:
                    state_dict.pop(key)

        super(_InstanceNorm, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        if input.dim() == self._get_no_batch_dim():
            return self._handle_no_batch_input(input)

        return self._apply_instance_norm(input)


class InstanceNorm1d(_InstanceNorm):
    def _get_no_batch_dim(self):
        return 2

    def _check_input_dim(self, input):
        if input.dim() not in (2, 3):
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )


# class LazyInstanceNorm1d(_LazyNormBase, _InstanceNorm):
#
#     cls_to_become = InstanceNorm1d  # type: ignore[assignment]
#
#     def _get_no_batch_dim(self):
#         return 2
#
#     def _check_input_dim(self, input):
#         if input.dim() not in (2, 3):
#             raise ValueError(
#                 "expected 2D or 3D input (got {}D input)".format(input.dim())
#             )


class InstanceNorm2d(_InstanceNorm):
    def _get_no_batch_dim(self):
        return 3

    def _check_input_dim(self, input):
        if input.dim() not in (3, 4):
            raise ValueError(
                "expected 3D or 4D input (got {}D input)".format(input.dim())
            )


# class LazyInstanceNorm2d(_LazyNormBase, _InstanceNorm):
#
#     cls_to_become = InstanceNorm2d  # type: ignore[assignment]
#
#     def _get_no_batch_dim(self):
#         return 3
#
#     def _check_input_dim(self, input):
#         if input.dim() not in (3, 4):
#             raise ValueError(
#                 "expected 3D or 4D input (got {}D input)".format(input.dim())
#             )


class InstanceNorm3d(_InstanceNorm):
    def _get_no_batch_dim(self):
        return 4

    def _check_input_dim(self, input):
        if input.dim() not in (4, 5):
            raise ValueError(
                "expected 4D or 5D input (got {}D input)".format(input.dim())
            )


# class LazyInstanceNorm3d(_LazyNormBase, _InstanceNorm):
#
#     cls_to_become = InstanceNorm3d  # type: ignore[assignment]
#
#     def _get_no_batch_dim(self):
#         return 4
#
#     def _check_input_dim(self, input):
#         if input.dim() not in (4, 5):
#             raise ValueError(
#                 "expected 4D or 5D input (got {}D input)".format(input.dim())
#             )

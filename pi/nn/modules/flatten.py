from typing import Tuple, Union

from .module import Module
from ..types import Size

from pi import Tensor

__all__ = ["Flatten", "Unflatten"]


class Flatten(Module):
    __constants__ = ["start_dim", "end_dim"]
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        return input.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return "start_dim={}, end_dim={}".format(self.start_dim, self.end_dim)


class Unflatten(Module):
    NamedShape = Tuple[Tuple[str, int]]

    __constants__ = ["dim", "unflattened_size"]
    dim: Union[int, str]
    unflattened_size: Union[Size, NamedShape]

    def __init__(
        self, dim: Union[int, str], unflattened_size: Union[Size, NamedShape]
    ) -> None:
        super(Unflatten, self).__init__()

        if isinstance(dim, int):
            self._require_tuple_int(unflattened_size)
        elif isinstance(dim, str):
            self._require_tuple_tuple(unflattened_size)
        else:
            raise TypeError("invalid argument type for dim parameter")

        self.dim = dim
        self.unflattened_size = unflattened_size

    def _require_tuple_tuple(self, input):
        if isinstance(input, tuple):
            for idx, elem in enumerate(input):
                if not isinstance(elem, tuple):
                    raise TypeError(
                        "unflattened_size must be tuple of tuples, "
                        + "but found element of type {} at pos {}".format(
                            type(elem).__name__, idx
                        )
                    )
            return
        raise TypeError(
            "unflattened_size must be a tuple of tuples, "
            + "but found type {}".format(type(input).__name__)
        )

    def _require_tuple_int(self, input):
        if isinstance(input, (tuple, list)):
            for idx, elem in enumerate(input):
                if not isinstance(elem, int):
                    raise TypeError(
                        "unflattened_size must be tuple of ints, "
                        + "but found element of type {} at pos {}".format(
                            type(elem).__name__, idx
                        )
                    )
            return
        raise TypeError(
            "unflattened_size must be a tuple of ints, but found type {}".format(
                type(input).__name__
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        return input.unflatten(self.dim, self.unflattened_size)

    def extra_repr(self) -> str:
        return "dim={}, unflattened_size={}".format(self.dim, self.unflattened_size)

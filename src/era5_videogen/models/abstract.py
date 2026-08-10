from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
from omegaconf import ListConfig

_Shape2D = Union[int, tuple[int, int], list[int], ListConfig]
_Shape3D = Union[int, tuple[int, int, int], list[int], ListConfig]

class AbstractNetwork(torch.nn.Module, ABC):
    """All networks should inherit this."""

    def __init__(
        self,
        img_resolution: tuple[int, int],
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("subclass must implement this.")


def get_activation(activation_f: str) -> nn.Module:
    activations = [
        nn.Tanh,
        nn.ReLU,
        nn.LeakyReLU,
        nn.SiLU,
        nn.SELU,
        # ...
    ]
    names = [str(o.__name__).lower() for o in activations]
    try:
        return activations[names.index(str(activation_f).lower())]
    except:
        raise NotImplementedError(f"{activation_f=} is not yet implemented.")
    
def pair(t):
    return (t, t) if isinstance(t, int) else t


def _shape_tuple(value, ndim: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * ndim
    if isinstance(value, ListConfig):
        value = list(value)
    if isinstance(value, list):
        value = tuple(value)
    if not isinstance(value, tuple):
        raise TypeError(f"Invalid type {type(value)}")
    if len(value) != ndim or any(not isinstance(x, int) for x in value):
        raise TypeError(f"Expected a tuple of {ndim} ints, got {value!r}")
    return value


@dataclass
class Shape2D:
    _shape: _Shape2D

    def __post_init__(self) -> None:
        self.shape = _shape_tuple(self._shape, ndim=2)
        assert (  # [!] self.shape must be (int, int)
            isinstance(self.shape, tuple)
            and isinstance(self.shape[0], int)
            and isinstance(self.shape[1], int)
        )

        self.height = self.shape[0]
        self.width = self.shape[1]


@dataclass
class Shape3D:
    _shape: _Shape3D

    def __post_init__(self) -> None:
        self.shape = _shape_tuple(self._shape, ndim=3)
        assert (
            isinstance(self.shape, tuple)
            and isinstance(self.shape[0], int)
            and isinstance(self.shape[1], int)
            and isinstance(self.shape[2], int)
        )

        self.depth = self.shape[0]
        self.height = self.shape[1]
        self.width = self.shape[2]

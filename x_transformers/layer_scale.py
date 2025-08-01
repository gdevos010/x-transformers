# post branch operator
import torch

from einops import rearrange
from torch import Tensor, nn
from torch.nn import Module

from x_transformers.utils import default


class LayerScale(Module):
    def __init__(self, fn: Module, dim, init_value=0.0, unit_offset=False) -> None:
        super().__init__()
        self.unit_offset = unit_offset

        self.fn = fn
        self.gamma = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.gamma, init_value - float(unit_offset))

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)

        gamma = self.gamma + float(self.unit_offset)

        if isinstance(out, Tensor):
            return out * gamma

        out, *rest = out
        return out * gamma, *rest


class AdaptiveLayerScale(Module):
    def __init__(
        self, fn: Module, dim, dim_condition=None, init_bias_value=-2.0
    ) -> None:
        super().__init__()
        self.fn = fn

        dim_condition = default(dim_condition, dim)
        self.to_gamma = nn.Linear(dim_condition, dim)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def forward(self, x, *, condition, **kwargs):
        if condition.ndim == 2:
            condition = rearrange(condition, "b d -> b 1 d")

        out = self.fn(x, **kwargs)
        gamma = self.to_gamma(condition).sigmoid()

        if isinstance(out, Tensor):
            return out * gamma

        out, *rest = out
        return out * gamma, *rest

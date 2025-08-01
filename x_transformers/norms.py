import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn, tensor
from torch.nn import Module

from x_transformers.utils import LinearNoBias, default

# norms


class Scale(Module):
    def __init__(self, value, fn) -> None:
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        scale_fn = lambda t: t * self.value

        if not isinstance(out, tuple):
            return scale_fn(out)

        return (scale_fn(out[0]), *out[1:])


class LayerNorm(Module):
    def __init__(self, dim, unit_offset=False):
        """bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less"""
        super().__init__()
        self.unit_offset = unit_offset

        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma = nn.Parameter(torch.ones(dim))
        nn.init.constant_(self.gamma, 1.0 - float(unit_offset))

    def forward(self, x):
        normed = self.ln(x)
        gamma = self.gamma + float(self.unit_offset)
        return normed * gamma


class AdaptiveLayerNorm(Module):
    def __init__(self, dim, dim_condition=None):
        super().__init__()
        dim_condition = default(dim_condition, dim)

        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_gamma = LinearNoBias(dim_condition, dim)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, "b d -> b 1 d")

        normed = self.ln(x)
        gamma = self.to_gamma(condition)
        return normed * (gamma + 1.0)


class ScaleNorm(Module):
    def __init__(self, dim, unit_offset=False):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim**0.5

        self.g = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.g, 1.0 - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim=-1) * self.scale * gamma


class RMSNorm(Module):
    def __init__(self, dim, unit_offset=False):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim**0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1.0 - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim=-1) * self.scale * gamma


class AdaptiveRMSNorm(Module):
    def __init__(self, dim, dim_condition=None):
        super().__init__()
        self.scale = dim**0.5
        dim_condition = default(dim_condition, dim)

        self.to_gamma = LinearNoBias(dim_condition, dim)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, "b d -> b 1 d")

        normed = F.normalize(x, dim=-1)
        gamma = self.to_gamma(condition)
        return normed * self.scale * (gamma + 1.0)


class SimpleRMSNorm(Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.scale = dim**0.5

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale


class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = SimpleRMSNorm(dim)
        self.gamma = nn.Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.0)


class DynamicTanh(Module):
    """https://arxiv.org/abs/2503.10622"""

    def __init__(self, dim, init_alpha=1.0, gamma=1.0, beta=0.0, unit_offset=False):
        super().__init__()
        self.pre_tanh_scale = nn.Parameter(tensor(init_alpha))

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

        self.pre_tanh_scale_offset = init_alpha if unit_offset else 0.0
        self.gamma_offset = float(unit_offset)

        nn.init.constant_(self.pre_tanh_scale, 0 if unit_offset else init_alpha)
        nn.init.constant_(self.gamma, 1.0 - float(unit_offset))

    def forward(self, x):
        pre_tanh_scale = self.pre_tanh_scale + self.pre_tanh_scale_offset
        gamma = self.gamma + self.gamma_offset
        return (x * pre_tanh_scale).tanh() * gamma + self.beta

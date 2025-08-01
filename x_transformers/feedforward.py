from collections.abc import Callable
from copy import deepcopy

import torch

from torch import Tensor, nn
from torch.nn import Module

from x_transformers.activations import ReluSquared
from x_transformers.utils import Sequential, default, exists, init_zero_

# feedforward


class GLU(Module):
    def __init__(self, dim_in, dim_out, activation: Callable, mult_bias=False) -> None:
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate) * self.mult_bias


class FeedForward(Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        glu=False,
        glu_mult_bias=False,
        swish=False,
        relu_squared=False,
        custom_activation=None,
        post_act_ln=False,
        dropout=0.0,
        sublayer_dropout=0.0,
        no_bias=False,
        zero_init_output=False,
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if exists(custom_activation):
            activation = deepcopy(custom_activation)
        elif relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            proj_in = GLU(dim, inner_dim, activation, mult_bias=glu_mult_bias)
        else:
            proj_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias=not no_bias), activation
            )

        proj_out = nn.Linear(inner_dim, dim_out, bias=not no_bias)

        self.ff = Sequential(
            proj_in,
            LayerNorm(inner_dim) if post_act_ln else None,
            nn.Dropout(dropout),
            proj_out,
            nn.Dropout(sublayer_dropout) if sublayer_dropout > 0.0 else None,
        )

        # init last linear layer to 0

        if zero_init_output:
            init_zero_(proj_out)

    def forward(self, x, deep_embed=None) -> Tensor:
        out = self.ff(x)

        if exists(deep_embed):
            out = out * deep_embed

        return out

from torch.nn import Module
from torch import nn
from torch import Tensor
from x_transformers.utils import exists, l2norm, pad_at_dim, LinearNoBias
from einops import rearrange
from torch import cat
import math
from torch.utils._pytree import tree_flatten, tree_unflatten
import torch


# embedding


class TokenEmbedding(Module):
    def __init__(self, dim, num_tokens, l2norm_embed=False) -> None:
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x.long())
        return l2norm(token_emb) if self.l2norm_embed else token_emb

    def init_(self) -> None:
        if self.l2norm_embed:
            nn.init.normal_(self.emb.weight, std=1e-5)
            return
        nn.init.kaiming_normal_(self.emb.weight)


# residual and residual gates


class Residual(Module):
    def __init__(
        self, dim, scale_residual=False, scale_residual_constant=1.0, **kwargs
    ):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def prepare(self, residual):
        return residual, residual, dict()

    def forward(self, x, residual, **kwargs):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual


class GRUGating(Module):
    def __init__(self, dim, scale_residual=False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None

    def prepare(self, residual):
        return residual, residual, dict()

    def forward(self, x, residual, **kwargs):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, "b n d -> (b n) d"), rearrange(residual, "b n d -> (b n) d")
        )

        return gated_output.reshape_as(x)


# token shifting


def shift(t, amount, mask=None):
    if amount == 0:
        return t

    amount = min(amount, t.shape[1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.0)

    return pad_at_dim(t, (amount, -amount), dim=-2, value=0.0)


class ShiftTokens(Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get("mask", None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = [
            shift(*args, mask=mask) for args in zip(segments_to_shift, shifts)
        ]
        x = cat((*segments_to_shift, *rest), dim=-1)
        return self.fn(x, **kwargs)


class FoldAxially(Module):
    def __init__(self, axial_dim, fn: Module):
        super().__init__()
        self.fn = fn
        self.axial_dim = axial_dim  # will fold the sequence as rearrange("b (n axial_dim) ... -> (b axial_dim) n ...")

    def forward(self, x, *args, **kwargs):
        if self.axial_dim == 1:
            return self.fn(x, *args, **kwargs)

        seq_len, axial_dim = x.shape[1], self.axial_dim

        next_multiple = math.ceil(seq_len / axial_dim) * axial_dim
        x = pad_at_dim(x, (0, next_multiple - seq_len), dim=1)

        x = rearrange(
            x, "b (n axial_dim) ... -> (b axial_dim) n ...", axial_dim=axial_dim
        )

        out = self.fn(x, *args, **kwargs)

        (out, *rest_out), tree_spec = tree_flatten(out)

        out = rearrange(
            out, "(b axial_dim) n ... -> b (n axial_dim) ...", axial_dim=axial_dim
        )

        out = out[:, :seq_len]
        out = tree_unflatten((out, *rest_out), tree_spec)

        return out


# skip connection combining


class ConcatCombine(Module):
    def __init__(self, dim, prev_layer_ind):
        super().__init__()
        self.prev_layer_ind = prev_layer_ind
        self.combine = LinearNoBias(dim * 2, dim)

    def forward(self, x, prev_layers: list[Tensor]):
        skip = prev_layers[self.prev_layer_ind]
        concatted_skip = cat((skip, x), dim=-1)
        return self.combine(concatted_skip)

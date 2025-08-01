# positional embeddings

from __future__ import annotations

import math

import einx
import torch
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, arange, cat, einsum, nn, stack
from torch._tensor import Tensor
from torch.amp import autocast
from torch.nn import Module, ModuleList

from x_transformers.norms import (
    LayerNorm,
)
from x_transformers.utils import (
    Sequential,
    default,
    divisible_by,
    exists,
    l2norm,
    pad_at_dim,
)


class AbsolutePositionalEmbedding(Module):
    def __init__(self, dim, max_seq_len, l2norm_embed=False):
        super().__init__()
        self.scale = dim**-0.5 if not l2norm_embed else 1.0
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None, seq_start_pos=None, offset=0):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, (
            f"you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}"
        )

        if not exists(pos):
            pos = arange(seq_len, device=device) + offset

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min=0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb


class ScaledSinusoidalEmbedding(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert divisible_by(dim, 2)
        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5)

        half_dim = dim // 2
        freq_seq = arange(half_dim).float() / half_dim
        inv_freq = theta**-freq_seq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None, seq_start_pos=None, offset=0):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = arange(seq_len, device=device) + offset

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum("i, j -> i j", pos, self.inv_freq)
        emb = cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale


class RelativePositionBias(Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, causal=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        q_pos = arange(j - i, j, dtype=torch.long, device=device)
        k_pos = arange(j, dtype=torch.long, device=device)
        rel_pos = einx.subtract("j, i -> i j", k_pos, q_pos)
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> h i j")
        return bias * self.scale


class CoPE(Module):
    """Appendix B of https://arxiv.org/abs/2405.18719"""

    def __init__(
        self,
        dim,
        heads,
        max_pos,
        soft_onehot=False,
        talking_heads=False,
        soft_onehot_temp=5e-2,
    ):
        super().__init__()
        self.max_pos = max_pos
        self.pos_emb = nn.Parameter(torch.zeros(max_pos, dim))

        self.talking_heads = (
            nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else None
        )
        self.soft_onehot = soft_onehot
        self.soft_onehot_temp = soft_onehot_temp

        if not soft_onehot:
            return

        self.register_buffer("positions", arange(max_pos))

    def forward(self, query, attn_logits):
        if exists(self.talking_heads):
            i, j = attn_logits.shape[-2:]
            causal_mask = attn_logits.new_ones(i, j).triu_(j - i + 1).bool()

            attn_logits = self.talking_heads(attn_logits)

            attn_logits = attn_logits.masked_fill(
                causal_mask, -torch.finfo(attn_logits.dtype).max
            )

        # compute positions

        gates = attn_logits.sigmoid()

        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.max_pos - 1)

        logits_int = einsum("b h n d, p d -> b h n p", query, self.pos_emb)

        if self.soft_onehot:
            diff_pos = einx.subtract("i, j -> i j", pos, self.positions).abs()
            soft_onehot_pos = F.softmax(-diff_pos / self.soft_onehot_temp, dim=-1)
            cope_pos_emb = einsum(
                "b h i j p, b h i p -> b h i j", soft_onehot_pos, logits_int
            )
        else:
            # interpolate from integer positions
            pos_ceil = pos.ceil().long()
            pos_floor = pos.floor().long()
            logits_ceil = logits_int.gather(-1, pos_ceil)
            logits_floor = logits_int.gather(-1, pos_floor)

            w = pos - pos_floor
            cope_pos_emb = logits_ceil * w + logits_floor * (1 - w)

        return cope_pos_emb


class DynamicPositionBias(Module):
    def __init__(self, dim, *, heads, depth, log_distance=False, norm=False):
        super().__init__()
        assert depth >= 1, (
            "depth for dynamic position bias MLP must be greater or equal to 1"
        )
        self.log_distance = log_distance

        self.mlp = ModuleList([])

        self.mlp.append(
            Sequential(nn.Linear(1, dim), LayerNorm(dim) if norm else None, nn.SiLU())
        )

        for _ in range(depth - 1):
            self.mlp.append(
                Sequential(
                    nn.Linear(dim, dim), nn.LayerNorm(dim) if norm else None, nn.SiLU()
                )
            )

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = arange(j - i, j, device=device)
        context_arange = arange(j, device=device)
        indices = einx.subtract("i, j -> i j", seq_arange, context_arange)
        indices += j - 1

        # input to continuous positions MLP
        pos = arange(-j + 1, j, device=device).float()
        pos = rearrange(pos, "... -> ... 1")

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(
                pos.abs() + 1
            )  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases
        bias = pos[indices]
        bias = rearrange(bias, "i j h -> h i j")
        return bias


class AlibiPositionalBias(Module):
    def __init__(
        self, heads, total_heads=None, slopes: list[int] | None = None, **kwargs
    ):
        super().__init__()
        self.heads = heads
        self.total_heads = default(total_heads, heads)

        slopes = Tensor(default(slopes, self._get_slopes(heads)))
        slopes = rearrange(slopes, "h -> h 1 1")

        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    @property
    def device(self):
        return next(self.buffers()).device

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : heads - closest_power_of_2
            ]
        )

    def forward_custom_pos(self, pos_i: Tensor, pos_j: Tensor | None = None):
        h, device = self.total_heads, self.device

        pos_j = default(pos_j, pos_i)
        bias = -einx.subtract("... j, ... i -> ... i j", pos_j, pos_i).abs()

        if bias.ndim == 3:
            bias = rearrange(bias, "b i j -> b 1 i j")

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=-3)

        return bias

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        seq_arange = arange(j - i, j, device=device)
        context_arange = arange(j, device=device)
        bias = -einx.subtract("j, i -> 1 i j", context_arange, seq_arange).abs()

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=-3)

        self.register_buffer("bias", bias, persistent=False)
        return self.bias


class DataDependentAlibi(Module):
    """https://openreview.net/forum?id=q2Lnyegkr8"""

    def __init__(
        self,
        dim,
        heads,
        causal=True,
        bias_init=5.0,
        post_log_scale=1.0,
    ):
        super().__init__()

        self.causal = causal

        linear = nn.Linear(dim, heads * (1 if causal else 2))

        self.to_forget_gates = nn.Sequential(
            linear, Rearrange("b n h -> b h n"), nn.LogSigmoid()
        )

        nn.init.constant_(linear.bias, bias_init)
        self.post_log_scale = post_log_scale

    def forward(self, x):
        bidirectional = not self.causal

        forget_gates = self.to_forget_gates(x) * self.post_log_scale

        forget_gates = forget_gates.cumsum(dim=-1)

        if bidirectional:
            forget_gates, forget_gates_reversed = forget_gates.chunk(2, dim=1)

        forget_gates = einx.subtract(
            "b h i, b h j -> b h i j", forget_gates, forget_gates
        )

        if bidirectional:
            forget_gates_reversed = einx.subtract(
                "b h j, b h i -> b h i j", forget_gates_reversed, forget_gates_reversed
            )
            forget_gates = forget_gates.tril() + forget_gates_reversed.triu()

        return forget_gates


class PerRowDataDependentAlibi(Module):
    """same as data dependent alibi from forgetting transformer, but the forgetting gates are also derived by a queries and keys with a small head dimension"""

    def __init__(self, dim, heads, causal=True, dim_head=8, post_log_scale=1.0):
        super().__init__()
        assert causal, "bidirectional not supported yet"

        self.scale = dim_head**-0.5

        linear = nn.Linear(dim, heads * dim_head * 2, bias=False)

        self.to_forget_gates = nn.Sequential(
            linear, Rearrange("b n (qk h d) -> qk b h n d", qk=2, d=dim_head)
        )

        self.post_log_scale = post_log_scale

    def forward(self, x):
        q, k = self.to_forget_gates(x)
        forget_gates = einsum("... i d, ... j d -> ... i j", q, k) * self.scale

        forget_gates = F.logsigmoid(forget_gates) * self.post_log_scale

        # mask out upper triangle + diagonal

        n = x.shape[-2]
        causal_mask = torch.ones((n, n), dtype=torch.bool, device=x.device).triu()

        forget_gates = forget_gates.masked_fill(causal_mask, 0.0)

        # reverse cumsum

        forget_gates = forget_gates.flip(dims=(-1,))
        forget_gates = forget_gates.cumsum(dim=-1)
        forget_gates = forget_gates.flip(dims=(-1,))

        return forget_gates


class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        use_xpos=False,
        scale_base=512,
        interpolation_factor=1.0,
        base=10000,
        base_rescale_factor=1.0,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer("scale", None)
            return

        scale = (arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer("scale", scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = arange(seq_len, device=device)
        return self.forward(t)

    @autocast("cuda", enabled=False)
    def forward(self, t, offset=0):
        max_pos = t.max() + 1

        if t.ndim == 1:
            t = rearrange(t, "n -> 1 n")

        freqs = (
            torch.einsum("b i , j -> b i j", t.type_as(self.inv_freq), self.inv_freq)
            / self.interpolation_factor
        )
        freqs = stack((freqs, freqs), dim=-1)
        freqs = rearrange(freqs, "... d r -> ... (d r)")

        if not exists(self.scale):
            return freqs, 1.0

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "... n -> ... n 1")
        scale = stack((scale, scale), dim=-1)
        scale = rearrange(scale, "... d r -> ... (d r)")

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = cat((t, t_unrotated), dim=-1)

    return out.type(orig_dtype)

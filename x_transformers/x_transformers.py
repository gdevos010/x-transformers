from __future__ import annotations

import math

from contextlib import nullcontext
from random import randrange

import torch
import torch.nn.functional as F

from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from torch import Tensor, arange, cat, einsum, nn
from torch._tensor import Tensor
from torch.amp import autocast
from torch.nn import Module, ModuleDict, ModuleList

from x_transformers.attention import Attention
from x_transformers.attention_layers import AttentionLayers
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from x_transformers.components import (
    TokenEmbedding,
)
from x_transformers.layer_intermediates import LayerIntermediates
from x_transformers.norms import (
    LayerNorm,
)
from x_transformers.postional_embeddings import (
    AbsolutePositionalEmbedding,
    ScaledSinusoidalEmbedding,
)
from x_transformers.utils import (
    LinearNoBias,
    Sequential,
    always,
    at_most_one_of,
    cast_tuple,
    default,
    divisible_by,
    exists,
    first,
    groupby_prefix_and_trim,
    log,
    masked_mean,
    max_neg_value,
    pad_at_dim,
    pick_and_pop,
)

# einstein notation

# b - batch
# n - sequence
# d - feature dimension
# h - attention heads
# i, j - sequence (source, target)


# entropy


def calc_entropy(t: Tensor, is_prob=False) -> Tensor:
    prob = t.softmax(dim=-1) if not is_prob else t
    return -(prob * log(prob)).sum(dim=-1)


# auxiliary loss helpers


def calc_z_loss(pre_softmax_attns: list[Tensor], mask=None, weight=1.0):
    # the same loss applied to the mixture of experts router logits in https://arxiv.org/abs/2202.08906
    # in the paper, in a tiny footnote, they mention using it on attention logits with stabilizing effects
    # also used in PaLM as one of the measures

    lse = 0.0

    for attn in pre_softmax_attns:
        lse = lse + attn.logsumexp(dim=-1)

    loss = torch.square(lse)
    loss = reduce(loss, "b h n -> b n", "sum")

    if not exists(mask):
        return loss.mean() * weight

    loss = loss[mask].sum() / mask.sum().clamp(min=1e-5)
    return loss * weight


# structured dropout, more effective than traditional attention dropouts


def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device=device)

    if exists(mask):
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1.0 - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices

    batch_indices = arange(b, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim=-1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = arange(num_keep, device=device) < rearrange(
            seq_keep_counts, "b -> b 1"
        )

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on encoder"
        super().__init__(causal=False, **kwargs)


class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on decoder"
        super().__init__(causal=True, **kwargs)


class PrefixDecoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on decoder"
        super().__init__(causal=False, **kwargs)

    def forward(self, x, *args, attn_mask=None, prefix_attn_len=None, **kwargs):
        b, n, device = x.shape[0], x.shape[1], x.device
        causal_mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)

        forwarded_mask = ~causal_mask

        if exists(prefix_attn_len):
            if isinstance(prefix_attn_len, int):
                prefix_attn_len = torch.full((b,), prefix_attn_len, device=device)

            prefix_mask = arange(n, device=device) < rearrange(
                prefix_attn_len, "b -> b 1 1 1"
            )
            forwarded_mask = forwarded_mask | prefix_mask

        if exists(attn_mask):
            forwarded_mask = forwarded_mask & attn_mask

        return super().forward(x, *args, attn_mask=forwarded_mask, **kwargs)


class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs) -> None:
        super().__init__(cross_attend=True, only_cross=True, **kwargs)


class AttentionPool(Module):
    def __init__(
        self,
        dim,
        num_pooled_tokens=1,
        dim_context=None,
        add_residual=False,
        depth=1,
        heads=8,
        dim_head=64,
        use_transformer_blocks=None,
        squeeze_output=None,
        attn_kwargs: dict = dict(),
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        squeeze_output = default(squeeze_output, False)
        assert not (squeeze_output and num_pooled_tokens > 1)

        use_transformer_blocks = default(use_transformer_blocks, depth > 1)
        assert use_transformer_blocks or depth == 1

        self.queries = nn.Parameter(torch.randn(num_pooled_tokens, dim) * 1e-2)

        if use_transformer_blocks:
            assert not add_residual, (
                "residual already in effect when doing a full cross attention based transformer for pooling"
            )
            attn_kwargs = {f"attn_{k}": v for k, v in attn_kwargs.items()}

            self.pooler = CrossAttender(
                dim=dim,
                cross_attn_dim_context=dim_context,
                depth=depth,
                heads=heads,
                attn_dim_head=dim_head,
            )
        else:
            self.pooler = Attention(
                dim=dim,
                dim_context=dim_context,
                heads=heads,
                dim_head=dim_head,
                **attn_kwargs,
            )

        self.add_residual = add_residual
        self.squeeze_output = squeeze_output

    def forward(self, context, mask=None):
        batch = context.shape[0]

        queries = repeat(self.queries, "n d -> b n d", b=batch)

        pooled = self.pooler(queries, context, context_mask=mask)

        if self.add_residual:
            pooled = pooled + queries

        if self.squeeze_output:
            pooled = rearrange(pooled, "b 1 d -> b d")

        return pooled


class ViTransformerWrapper(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        attn_layers: Encoder,
        channels=3,
        num_classes=None,
        post_emb_norm=False,
        num_register_tokens=0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert divisible_by(image_size, patch_size), (
            "image dimensions must be divisible by the patch size"
        )
        dim = attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size**2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        has_register_tokens = num_register_tokens > 0
        self.has_register_tokens = has_register_tokens

        if has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        self.patch_to_embedding = nn.Sequential(
            LayerNorm(patch_dim), nn.Linear(patch_dim, dim), LayerNorm(dim)
        )

        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers

        self.mlp_head = (
            nn.Linear(dim, num_classes) if exists(num_classes) else nn.Identity()
        )

    def forward(self, img, return_embeddings=False, return_logits_and_embeddings=False):
        b, p = img.shape[0], self.patch_size

        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)
        n = x.shape[1]

        x = x + self.pos_embedding[:, :n]

        x = self.post_emb_norm(x)
        x = self.dropout(x)

        if self.has_register_tokens:
            r = repeat(self.register_tokens, "n d -> b n d", b=b)
            x, ps = pack((x, r), "b * d")

        embed = self.attn_layers(x)

        if self.has_register_tokens:
            embed, _ = unpack(embed, ps, "b * d")

        assert at_most_one_of(return_embeddings, return_logits_and_embeddings)

        if not exists(self.mlp_head) or return_embeddings:
            return embed

        pooled = embed.mean(dim=-2)
        logits = self.mlp_head(pooled)

        if not return_logits_and_embeddings:
            return logits

        return logits, embed


class TransformerWrapper(Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers: AttentionLayers,
        embed_num_tokens: dict[str, int] = dict(),
        emb_dim=None,
        max_mem_len=0,
        shift_mem_down=0,
        emb_dropout=0.0,
        post_emb_norm=False,
        num_memory_tokens=None,
        memory_tokens_interspersed_every=None,
        tie_embedding=False,
        logits_dim=None,
        return_only_embed=False,
        num_output_heads=1,
        use_abs_pos_emb=True,
        scaled_sinu_pos_emb=False,
        l2norm_embed=False,
        recycling=False,  # from Jumper et al. - Alphafold2
        train_max_recycle_steps=4,  # saw a benefit for language modeling up to 3 recycling steps, so let's default this to 4
        emb_frac_gradient=1.0,  # GLM-130B and Cogview successfully used this, set at 0.1
        attn_z_loss_weight=1e-4,
        average_pool_embed=False,
        use_cls_token=False,
        num_cls_tokens=1,
        attn_pool=False,
        num_pooled_tokens=1,
        attn_pool_depth=1,
        dim_pooled_tokens=None,
        squeeze_out_last_dim=False,
        token_emb: TokenEmbedding | None = None,
        mixture_of_softmax=False,
        mixture_of_softmax_k=4,
        sigsoftmax_logits=False,
        ff_deep_embed=False,
        to_logits: Module | None = None,
        add_continuous_pred_head=False,
    ):
        super().__init__()

        dim = attn_layers.dim
        depth = attn_layers.depth

        emb_dim = default(emb_dim, dim)
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens
        self.num_cls_tokens = num_cls_tokens

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.l2norm_embed = l2norm_embed

        if not exists(token_emb):
            token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed=l2norm_embed)

        self.token_emb = token_emb

        no_abs_pos_emb = max_seq_len == 0 or not (
            use_abs_pos_emb and not attn_layers.disable_abs_pos_emb
        )

        if no_abs_pos_emb:
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(emb_dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(
                emb_dim, max_seq_len, l2norm_embed=l2norm_embed
            )

        # additional embeddings - say type embedding from BERT

        self.embeds = None

        if len(embed_num_tokens) > 0:
            self.embeds = ModuleDict(
                {
                    f"{name}_embed": nn.Embedding(num_tokens, emb_dim)
                    for name, num_tokens in embed_num_tokens.items()
                }
            )

        # deep embed

        # credit goes to Braden Koszarsky for first devising value embeddings in nanogpt-speedrun project
        # then Bo Peng for coming up with this alternate design in feedforward for RWKV 8
        # improvements were clearest to me (on my toy setup) with multiplying on output of feedforward, will try with attention at future date

        self.ff_deep_embed = None
        if ff_deep_embed:
            self.ff_deep_embed = nn.Parameter(torch.ones(num_tokens, depth, dim))

        # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

        self.emb_frac_gradient = emb_frac_gradient

        self.post_emb_norm = LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers

        self.init_()

        assert num_output_heads > 0

        assert at_most_one_of(average_pool_embed, use_cls_token)

        # maybe recycling

        self.recycling = recycling
        self.recycled_proj = LinearNoBias(dim, dim) if recycling else None

        self.train_max_recycle_steps = train_max_recycle_steps

        # either cls token or attn pool, but not both

        assert not (use_cls_token and attn_pool)

        # classic cls token from the bert days

        self.cls_token = None

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(num_cls_tokens, dim))
            nn.init.normal_(self.cls_token, std=0.02)

        # attn pool

        self.attn_pool = None

        if attn_pool:
            self.attn_pool = AttentionPool(
                dim=default(dim_pooled_tokens, dim),
                dim_context=dim,
                num_pooled_tokens=num_pooled_tokens,
                depth=attn_pool_depth,
                heads=self.attn_layers.attn_heads,
                dim_head=self.attn_layers.attn_dim_head,
            )

        # whether to average pool the embed (`global average pool`)

        self.average_pool_embed = average_pool_embed

        # output type

        self.output_is_log_prob = mixture_of_softmax

        self.to_mixture = None
        self.combine_mixture = None

        if mixture_of_softmax:
            assert num_output_heads == 1

            self.to_mixture = Sequential(
                LinearNoBias(dim, dim * mixture_of_softmax_k),
                Rearrange("... (k d) -> ... k d", k=mixture_of_softmax_k),
            )

            self.combine_mixture = LinearNoBias(dim, mixture_of_softmax_k)

        # sig softmax

        self.sigsoftmax_logits = sigsoftmax_logits

        # output head, usually to logits of num_tokens

        logits_dim = default(logits_dim, num_tokens)

        self.has_multiple_heads = num_output_heads > 1

        if return_only_embed:
            self.to_logits = None
        elif tie_embedding:
            assert isinstance(token_emb, TokenEmbedding), (
                "can only tie embedding if using `TokenEmbedding`"
            )
            self.to_logits = lambda t: t @ self.token_emb.emb.weight.t()
        elif num_output_heads > 1:
            self.to_logits = ModuleList(
                [LinearNoBias(dim, logits_dim) for _ in range(num_output_heads)]
            )
        else:
            self.to_logits = (
                LinearNoBias(dim, logits_dim) if not exists(to_logits) else to_logits
            )

        # add a head that predicts the embedding of the next step

        self.add_continuous_pred_head = add_continuous_pred_head

        if add_continuous_pred_head:
            self.to_next_embed_pred = nn.Sequential(
                LinearNoBias(dim, dim), nn.SiLU(), LinearNoBias(dim, dim)
            )

        # memory tokens (like [cls]) from Memory Transformers paper

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

        # squeeze out last dimension if possible

        self.squeeze_out_last_dim = squeeze_out_last_dim

        # whether can do cached kv decoding

        self.can_cache_kv = (
            self.num_memory_tokens == 0
            and not recycling
            and self.attn_layers.can_cache_kv
        )
        self.can_cache_kv_outside_max_seq_len = no_abs_pos_emb

    def init_(self):
        if hasattr(self.token_emb, "init_"):
            self.token_emb.init_()

        if self.l2norm_embed:
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)

    def forward(
        self,
        x,
        return_embeddings=False,
        return_logits_and_embeddings=False,
        return_intermediates=False,
        return_embeddings_and_intermediates=False,
        return_logit_entropies=False,
        return_next_embed_pred=False,
        mask=None,
        return_mems=False,
        return_attn=False,
        mems=None,
        mem_masks=None,
        recycle_steps=None,
        pos=None,
        prepend_embeds=None,
        prepend_mask=None,
        embed_ids: dict[str, Tensor] = dict(),
        sum_embeds=None,
        return_attn_z_loss=False,
        attn_z_loss_weight=1e-4,
        seq_start_pos=None,
        cache: LayerIntermediates | None = None,
        input_not_include_cache=False,
        token_emb_kwargs=dict(),
        to_logits_kwargs=dict(),
        **kwargs,
    ):
        # if sequence is None, auto create an empty one if `prepend_embeds` was supplied

        if not exists(x):
            assert exists(prepend_embeds)
            x = prepend_embeds.new_empty((prepend_embeds.shape[0], 0), dtype=torch.long)

        # shapes and variables

        (
            b,
            n,
            device,
            token_ids,
            num_mems,
            has_memory_tokens,
            emb_frac_gradient,
            orig_mask,
        ) = (
            x.shape[0],
            x.shape[1],
            x.device,
            x,
            self.num_memory_tokens,
            self.num_memory_tokens > 0,
            self.emb_frac_gradient,
            mask,
        )

        return_hiddens = (
            return_mems
            | return_attn
            | return_intermediates
            | return_attn_z_loss
            | return_embeddings_and_intermediates
        )
        return_embeddings = (
            return_embeddings
            | (not exists(self.to_logits))
            | return_embeddings_and_intermediates
        )

        # take care of position embedding offsets in the presence of cache and sequence is less than cache length (not full sequence)

        seq_pos_offset = 0

        if exists(cache) and input_not_include_cache:
            seq_pos_offset = cache.cache_length

        # absolute positional embedding

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = (
            self.pos_emb(x, pos=pos, seq_start_pos=seq_start_pos, offset=seq_pos_offset)
            if not external_pos_emb
            else pos
        )
        x = self.token_emb(x, **token_emb_kwargs) + pos_emb

        # add additional embeddings

        assert not (exists(self.embeds) ^ (len(embed_ids) > 0)), (
            "`embed_num_tokens` must be defined on `TransformerWrapper`"
        )

        if exists(self.embeds):
            assert len(embed_ids) == len(self.embeds)

            for name, embed_id in embed_ids.items():
                embed_key = f"{name}_embed"

                assert embed_key in self.embeds
                embed = self.embeds[embed_key](embed_id)

                x = x + embed

        # for summing embeddings passed externally - needs this for self-conditioning in non-autoregressive training

        if exists(sum_embeds):
            x = x + sum_embeds

        # post embedding norm, purportedly leads to greater stabilization

        x = self.post_emb_norm(x)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], (
                "prepended embeddings need to have same dimensions as text model dimensions"
            )

            x = cat((prepend_embeds, x), dim=-2)

            if exists(prepend_mask) or exists(mask):
                mask = default(
                    mask, lambda: torch.ones((b, n), device=device, dtype=torch.bool)
                )
                prepend_mask = default(
                    prepend_mask,
                    lambda: torch.ones(
                        (b, prepend_seq), device=device, dtype=torch.bool
                    ),
                )

                mask = cat((prepend_mask, mask), dim=-1)

        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # init embed

        init_embed = x

        # embedding dropout

        x = self.emb_dropout(x)

        x = self.project_emb(x)

        # maybe deep embeds

        deep_embed_and_ids = None

        if exists(self.ff_deep_embed):
            deep_embed_and_ids = (self.ff_deep_embed, token_ids)

        # maybe cls token

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, "... -> b ...", b=b)
            x, cls_packed_shape = pack([cls_tokens, x], "b * d")

            if exists(mask):
                mask = F.pad(mask, (self.num_cls_tokens, 0), value=True)

        # maybe memory / register tokens

        if has_memory_tokens:
            mem_seq = x.shape[-2]
            mem_every = self.memory_tokens_interspersed_every

            if exists(mem_every):
                assert mem_every > 0
                assert isinstance(self.attn_layers, Decoder), "only for decoder"
                next_seq_len = math.ceil(n / mem_every) * mem_every

                x = pad_at_dim(x, (0, next_seq_len - n), dim=-2, value=0.0)
                x = rearrange(x, "b (n m) d -> (b n) m d", m=mem_every)

            mem = repeat(self.memory_tokens, "n d -> b n d", b=x.shape[0])
            x, mem_packed_shape = pack((mem, x), "b * d")

            # auto-handle masking after appending memory tokens
            if not exists(mem_every) and exists(mask):
                mask = pad_at_dim(mask, (num_mems, 0), dim=-1, value=True)

            if exists(mem_every):
                x = rearrange(x, "(b n) m d -> b (n m) d", b=b)

        # handle maybe shifting of memories

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[: self.shift_mem_down], mems[self.shift_mem_down :]
            mems = [*mems_r, *mems_l]

        # attn layers kwargs

        kwargs = dict(
            **kwargs,
            seq_pos_offset=seq_pos_offset,
            seq_start_pos=seq_start_pos,
            input_not_include_cache=input_not_include_cache,
        )

        # attention layers

        if not self.recycling:
            assert not exists(recycle_steps) or recycle_steps == 1, (
                "you did not train with recycling"
            )

            # regular

            attended, intermediates = self.attn_layers(
                x,
                mask=mask,
                mems=mems,
                mem_masks=mem_masks,
                cache=cache,
                deep_embeds_and_ids=deep_embed_and_ids,
                return_hiddens=True,
                **kwargs,
            )

        else:
            # recycling

            recycle_steps = default(
                recycle_steps,
                (randrange(self.train_max_recycle_steps) + 1)
                if self.training
                else None,
            )
            assert exists(recycle_steps) and recycle_steps > 0, (
                "`recycle_steps` must be provided on forward if recycling is turned on and not training"
            )

            for i in range(recycle_steps):
                first_step = i == 0
                last_step = i == (recycle_steps - 1)

                context = nullcontext if last_step else torch.no_grad

                with context():
                    maybe_recycled = (
                        self.recycled_proj(attended.detach()) if not first_step else 0.0
                    )

                    attended, intermediates = self.attn_layers(
                        x + maybe_recycled,
                        mask=mask,
                        mems=mems,
                        mem_masks=mem_masks,
                        cache=cache,
                        return_hiddens=True,
                        **kwargs,
                    )

        x = attended

        # handle memories post-attention

        if has_memory_tokens:
            if exists(mem_every):
                x = rearrange(x, "b (n m) d -> (b n) m d", m=(mem_every + num_mems))

            mem, x = unpack(x, mem_packed_shape, "b * d")

            intermediates.memory_tokens = mem

            if exists(mem_every):
                x = rearrange(x, "(b n) m d -> b (n m) d", b=b)

            x = x[:, :mem_seq]

        # store last layer hiddens, for access in case of cls token or attention pooling

        intermediates.last_layer_hiddens = x

        # global average pool

        if self.average_pool_embed:
            x = masked_mean(x, mask=orig_mask, dim=1)

        # cls token(s)

        if exists(self.cls_token):
            x, last_layer_hiddens = unpack(x, cls_packed_shape, "b * d")

            intermediates.last_layer_hiddens = last_layer_hiddens

            if x.shape[1] == 1:
                x = rearrange(
                    x, "b 1 d -> b d"
                )  # Remove sequence dimension if num_cls_tokens=1 to keep previous behavior

        # attention pool

        is_encoder = not self.attn_layers.causal
        return_pooled_tokens = exists(self.attn_pool) and is_encoder

        if (
            exists(self.attn_pool)
            and (
                return_intermediates or is_encoder
            )  # in a new paper, they use attention pooling on decoder - so we'll default to returning pooled tokens if encoder, but for decoder, they must set `return_intermediates`
        ):
            attn_pooled_tokens = self.attn_pool(x, mask=mask)

            intermediates.attn_pooled_tokens = attn_pooled_tokens

        # handle expansion to mixture if needed (for mixture of softmax)

        combine_mixture = None

        if exists(self.to_mixture):
            combine_mixture = self.combine_mixture(x).softmax(dim=-1)
            x = self.to_mixture(x)

        # projecting to logits

        if not return_embeddings:
            if self.has_multiple_heads:
                logits = tuple(fn(x, **to_logits_kwargs) for fn in self.to_logits)
            else:
                logits = self.to_logits(x, **to_logits_kwargs)

        # maybe sig softmax

        if self.sigsoftmax_logits:
            logits = logits + logits.sigmoid().log()

        # handle maybe combine mixture

        if exists(combine_mixture):
            with autocast("cuda", enabled=False):
                prob = logits.softmax(dim=-1)
                mos = einsum("... k d, ... k -> ... d", prob, combine_mixture)
                logits = log(mos)

        # maybe squeeze out last dimension of logits

        if self.squeeze_out_last_dim:
            logits = tuple(
                (rearrange(t, "... 1 -> ...") if t.shape[-1] == 1 else t)
                for t in cast_tuple(logits)
            )

            if not self.has_multiple_heads:
                logits = first(logits)

        # different returns

        if return_logits_and_embeddings:
            out = (logits, x)
        elif return_embeddings_and_intermediates:
            out = (x, intermediates)
        elif return_embeddings:
            out = x
        elif return_pooled_tokens:
            intermediates.logits = logits
            out = attn_pooled_tokens
        else:
            out = logits

        # maybe next embed pred

        if return_next_embed_pred:
            assert self.add_continuous_pred_head
            next_embed_out = self.to_next_embed_pred(x)

            out = (out, (next_embed_out, init_embed))

        # logit entropies

        if return_logit_entropies:
            intermediates.logit_entropies = calc_entropy(logits)
            return_intermediates = True

        # aux loss

        if return_attn_z_loss:
            pre_softmax_attns = [
                t.pre_softmax_attn for t in intermediates.attn_intermediates
            ]
            intermediates.attn_z_loss = calc_z_loss(
                pre_softmax_attns, weight=attn_z_loss_weight
            )
            return_intermediates = True

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = (
                [cat(pair, dim=-2) for pair in zip(mems, hiddens, strict=False)]
                if exists(mems)
                else hiddens
            )
            new_mems = [t[..., -self.max_mem_len :, :].detach() for t in new_mems]

            if not return_intermediates:
                return out, new_mems

            intermediates.mems = new_mems

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = [t.post_softmax_attn for t in intermediates.attn_intermediates]
            return out, attn_maps

        return out


class XTransformer(Module):
    def __init__(
        self,
        *,
        dim,
        tie_token_emb=False,
        ignore_index=-100,
        pad_value=0,
        cross_attn_tokens_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        enc_kwargs, kwargs = groupby_prefix_and_trim("enc_", kwargs)
        dec_kwargs, kwargs = groupby_prefix_and_trim("dec_", kwargs)

        assert "dim" not in enc_kwargs and "dim" not in dec_kwargs, (
            "dimension of either encoder or decoder must be set with `dim` keyword"
        )
        enc_transformer_kwargs = pick_and_pop(["num_tokens", "max_seq_len"], enc_kwargs)
        enc_transformer_kwargs["emb_dropout"] = enc_kwargs.pop("emb_dropout", 0)
        enc_transformer_kwargs["num_memory_tokens"] = enc_kwargs.pop(
            "num_memory_tokens", None
        )
        enc_transformer_kwargs["scaled_sinu_pos_emb"] = enc_kwargs.pop(
            "scaled_sinu_pos_emb", False
        )
        enc_transformer_kwargs["use_abs_pos_emb"] = enc_kwargs.pop(
            "use_abs_pos_emb", True
        )

        dec_transformer_kwargs = pick_and_pop(["num_tokens", "max_seq_len"], dec_kwargs)
        dec_transformer_kwargs["emb_dropout"] = dec_kwargs.pop("emb_dropout", 0)
        dec_transformer_kwargs["scaled_sinu_pos_emb"] = dec_kwargs.pop(
            "scaled_sinu_pos_emb", False
        )
        dec_transformer_kwargs["use_abs_pos_emb"] = dec_kwargs.pop(
            "use_abs_pos_emb", True
        )

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout  # how many tokens from the encoder to dropout when cross attending from decoder - seen in a couple papers, including Perceiver AR - this will also be very effective regularization when cross attending to very long memories

        self.encoder = TransformerWrapper(
            **enc_transformer_kwargs,
            return_only_embed=True,
            attn_layers=Encoder(dim=dim, **enc_kwargs),
        )

        self.decoder = TransformerWrapper(
            **dec_transformer_kwargs,
            attn_layers=Decoder(dim=dim, cross_attend=True, **dec_kwargs),
        )

        if tie_token_emb:
            self.decoder.token_emb = self.encoder.token_emb

        self.decoder = AutoregressiveWrapper(
            self.decoder, ignore_index=ignore_index, pad_value=pad_value
        )

    @torch.no_grad()
    def generate(
        self, seq_in, seq_out_start, seq_len, mask=None, attn_mask=None, **kwargs
    ):
        encodings = self.encoder(
            seq_in, mask=mask, attn_mask=attn_mask, return_embeddings=True
        )
        return self.decoder.generate(
            seq_out_start, seq_len, context=encodings, context_mask=mask, **kwargs
        )

    def forward(self, src, tgt, mask=None, attn_mask=None, src_prepend_embeds=None):
        enc = self.encoder(
            src,
            mask=mask,
            attn_mask=attn_mask,
            prepend_embeds=src_prepend_embeds,
            return_embeddings=True,
        )

        if exists(src_prepend_embeds) and exists(mask):
            mask = pad_at_dim(
                mask, (src_prepend_embeds.shape[-2], 0), dim=-1, value=True
            )

        if self.training and self.cross_attn_tokens_dropout > 0:
            enc, mask = dropout_seq(enc, mask, self.cross_attn_tokens_dropout)

        out = self.decoder(tgt, context=enc, context_mask=mask)
        return out

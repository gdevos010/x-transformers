# attention. it is all we need
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import partial

import einx
import torch
import torch.nn.functional as F

from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from torch import arange, cat, nn
from torch.nn import Module, ModuleList
from torch.utils._pytree import tree_flatten

from x_transformers.attend import Attend, Intermediates
from x_transformers.components import (
    FoldAxially,
)
from x_transformers.norms import (
    MultiheadRMSNorm,
)
from x_transformers.postional_embeddings import (
    AlibiPositionalBias,
    CoPE,
    DataDependentAlibi,
    PerRowDataDependentAlibi,
    apply_rotary_pos_emb,
)
from x_transformers.utils import (
    DEFAULT_DIM_HEAD,
    LinearNoBias,
    always,
    default,
    divisible_by,
    exists,
    first,
    init_zero_,
    l2norm,
    log,
    max_neg_value,
    maybe,
    or_reduce,
    pad_at_dim,
    softclamp,
)


class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head=DEFAULT_DIM_HEAD,
        dim_context=None,
        heads=8,
        causal=False,
        flash=False,
        pre_talking_heads=False,
        post_talking_heads=False,
        pre_scale_post_talking_heads=False,
        head_scale=False,
        sparse_topk=None,
        sparse_topk_straight_through=False,
        num_mem_kv=0,
        dropout=0.0,
        sublayer_dropout=0.0,
        on_attn=False,
        gate_value_heads=False,
        swiglu_values=False,
        gate_values=False,
        zero_init_output=False,
        hard=False,
        max_attend_past=None,
        qk_norm=False,
        qk_norm_groups=1,
        qk_norm_scale=10,
        qk_norm_dim_scale=False,
        value_rmsnorm=False,  # used in alphagenome and bytedance's GR3 for further stability
        l2_distance=False,
        sigmoid=False,
        selective=False,
        custom_attn_fn: Callable | None = None,
        hybrid_module: Module | None = None,
        hybrid_mask_kwarg: str | None = None,
        hybrid_fold_axial_dim: int | None = None,
        hybrid_learned_mix=False,
        one_kv_head=False,
        kv_heads=None,
        value_dim_head=None,
        dim_out=None,
        add_zero_kv=False,  # same as add_zero_attn in pytorch
        rotate_num_heads=None,
        data_dependent_alibi=False,
        data_dependent_alibi_per_row=False,
        data_dependent_alibi_per_row_dim_head=8,
        data_dependent_alibi_kwargs: dict = dict(),
        use_cope=False,
        cope_max_pos=16,
        cope_soft_onehot_pos=False,
        cope_talking_heads=False,
        softclamp_logits=False,
        logit_softclamp_value=50.0,
        learned_value_residual_mix=False,
        laser=False,  # https://arxiv.org/abs/2411.03493v1
        laser_softclamp_value=15.0,
        qkv_receive_diff_residuals=False,
        use_latent_q=False,
        dim_latent_q=None,
        use_latent_kv=False,
        dim_latent_kv=None,
        latent_rope_subheads=None,
        onnxable=False,
        attend_sdp_kwargs: dict = dict(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ),
    ):
        super().__init__()
        dim_kv = default(dim_context, dim)

        self.scale = dim_head**-0.5

        self.heads = heads
        self.causal = causal
        self.max_attend_past = max_attend_past

        assert not (exists(kv_heads) and one_kv_head), (
            "either attn_one_kv_head is set to True (in which case kv_heads is set to 1), or attn_kv_heads is set, but not both"
        )

        value_dim_head = default(value_dim_head, dim_head)
        kv_heads = default(kv_heads, heads)

        kv_heads = 1 if one_kv_head else kv_heads
        assert divisible_by(heads, kv_heads)

        self.kv_heads = kv_heads

        q_dim = dim_head * heads
        k_dim = dim_head * kv_heads
        v_dim = value_dim_head * kv_heads
        out_dim = value_dim_head * heads

        # determine input dimensions to qkv based on whether intermediate latent q and kv are being used
        # for eventually supporting multi-latent attention (MLA)

        self.to_latent_q = None
        self.to_latent_kv = None
        self.to_rotateable_k = None  # for their "decoupled rope", subheads of keys that comes directly from base sequence (does not go through latents)

        dim_q_input = dim
        dim_kv_input = dim_kv

        if use_latent_q:
            assert exists(dim_latent_q)
            self.to_latent_q = LinearNoBias(dim, dim_latent_q)
            dim_q_input = dim_latent_q

        if use_latent_kv:
            assert exists(dim_latent_kv)
            self.to_latent_kv = LinearNoBias(dim, dim_latent_kv)
            dim_kv_input = dim_latent_kv

        if exists(latent_rope_subheads):
            assert not exists(rotate_num_heads), (
                "`rotate_num_heads` cannot be set when multi-latent attention is being used"
            )
            rotate_num_heads = latent_rope_subheads

            k_dim = dim_head * (kv_heads - latent_rope_subheads)

            self.to_rotateable_k = LinearNoBias(dim, dim_head * latent_rope_subheads)
            self.split_rotateable_k_heads = Rearrange(
                "b n (h d) -> b h n d", h=latent_rope_subheads
            )

        self.use_latent_q = use_latent_q
        self.use_latent_kv = use_latent_kv

        # query key projection

        self.to_q = LinearNoBias(dim_q_input, q_dim)
        self.to_k = LinearNoBias(dim_kv_input, k_dim)
        self.to_v = LinearNoBias(dim_kv_input, v_dim)

        # split and merge of attention heads

        self.split_q_heads = Rearrange("b n (h d) -> b h n d", h=heads)
        self.split_k_heads = Rearrange("b n (h d) -> b h n d", d=dim_head)
        self.split_v_heads = Rearrange("b n (h d) -> b h n d", d=value_dim_head)

        self.merge_heads = Rearrange("b h n d -> b n (h d)")

        # whether qkv receives different residual stream combinations from hyper connections or lime

        self.qkv_receive_diff_residuals = qkv_receive_diff_residuals

        # enhancing gradients to attention through exponentiated values

        self.laser = laser
        self.laser_softclamp_value = laser_softclamp_value

        # add GLU gating for aggregated values, from alphafold2

        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, out_dim)
            self.to_v_gate_activation = F.silu if swiglu_values else F.sigmoid
            nn.init.constant_(self.to_v_gate.weight, 0)
            nn.init.constant_(self.to_v_gate.bias, 10)

        # add per head gating of the output values, from 'Attend to nothing' paper

        self.to_v_head_gate = None
        if gate_value_heads:
            self.to_v_head_gate = nn.Linear(dim, heads)
            nn.init.constant_(self.to_v_head_gate.weight, 0)
            nn.init.constant_(self.to_v_head_gate.bias, 10)

        # cosine sim attention

        self.qk_norm = qk_norm
        self.qk_norm_groups = qk_norm_groups
        self.qk_norm_scale = qk_norm_scale

        # whether to use the rmsnorm (equivalent to cosine sim attention when scale is equal to 1) - https://arxiv.org/abs/2302.05442

        self.qk_norm_dim_scale = qk_norm_dim_scale

        self.qk_norm_q_scale = self.qk_norm_k_scale = 1
        if qk_norm and qk_norm_dim_scale:
            self.qk_norm_q_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            self.qk_norm_k_scale = nn.Parameter(torch.ones(kv_heads, 1, dim_head))

        assert (not qk_norm) or divisible_by(dim_head, qk_norm_groups), (
            "dimension per attention head must be divisible by the qk norm groups"
        )
        assert not (qk_norm and (dim_head // qk_norm_groups) <= 2), (
            "the group dimension may be too small (2 was too small in my tests, but 4 still works, surprisingly)"
        )

        # value rms norm

        self.value_rmsnorm = (
            MultiheadRMSNorm(dim_head, heads=heads) if value_rmsnorm else None
        )

        # contextual positional encoding
        # https://arxiv.org/html/2405.18719v2

        cope = None

        if use_cope:
            assert causal, "CoPE was designed for causal attention"
            assert not flash, "CoPE is not flash attention compatible"

            cope = CoPE(
                dim=dim_head,
                heads=heads,
                max_pos=cope_max_pos,
                talking_heads=cope_talking_heads,
                soft_onehot=cope_soft_onehot_pos,
            )

        # data dependent alibi
        # https://openreview.net/forum?id=q2Lnyegkr8

        self.data_dependent_alibi = None

        if data_dependent_alibi:
            dda_klass = (
                DataDependentAlibi
                if not data_dependent_alibi_per_row
                else PerRowDataDependentAlibi
            )
            dda_kwargs = dict(dim=dim, heads=heads, causal=causal)

            if data_dependent_alibi_per_row:
                dda_kwargs.update(dim_head=data_dependent_alibi_per_row_dim_head)

            self.data_dependent_alibi = dda_klass(
                **dda_kwargs, **data_dependent_alibi_kwargs
            )

        # attend class - includes core attention algorithm + talking heads

        self.attend = Attend(
            heads=heads,
            causal=causal,
            pre_talking_heads=pre_talking_heads,
            post_talking_heads=post_talking_heads,
            pre_scale_post_talking_heads=pre_scale_post_talking_heads,
            dropout=dropout,
            sparse_topk=sparse_topk,
            sparse_topk_straight_through=sparse_topk_straight_through,
            hard=hard,
            qk_norm=qk_norm,
            scale=qk_norm_scale if qk_norm else self.scale,
            l2_distance=l2_distance,
            sigmoid=sigmoid,
            selective=selective,
            custom_attn_fn=custom_attn_fn,
            add_zero_kv=add_zero_kv,
            flash=flash,
            softclamp_logits=softclamp_logits,
            logit_softclamp_value=logit_softclamp_value,
            cope=cope,
            onnxable=onnxable,
            sdp_kwargs=attend_sdp_kwargs,
        )

        # head scaling

        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, heads, 1, 1))

        # explicit topk sparse attention

        self.sparse_topk = sparse_topk

        # add memory key / values

        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(kv_heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(kv_heads, num_mem_kv, dim_head))

        # maybe learned value residual mixer per token

        self.to_value_residual_mix = (
            nn.Sequential(
                nn.Linear(dim, heads), nn.Sigmoid(), Rearrange("b n h -> b h n 1")
            )
            if learned_value_residual_mix
            else always(0.5)
        )

        # attention on attention

        self.attn_on_attn = on_attn

        # hybrid module, in same vein as hymba https://www.arxiv.org/abs/2411.13676

        hybrid_mix = None
        hybrid_norms = None
        hybrid_module = maybe(deepcopy)(hybrid_module)

        if exists(hybrid_module) and exists(hybrid_fold_axial_dim):
            hybrid_module = FoldAxially(
                axial_dim=hybrid_fold_axial_dim, fn=hybrid_module
            )
            hybrid_mix = LinearNoBias(dim, heads) if hybrid_learned_mix else None

            hybrid_norms = ModuleList(
                [
                    MultiheadRMSNorm(dim_head, heads=heads),
                    MultiheadRMSNorm(dim_head, heads=heads),
                ]
            )

        self.hybrid_module = hybrid_module
        self.hybrid_norms = hybrid_norms
        self.hybrid_mix = hybrid_mix
        self.hybrid_mask_kwarg = hybrid_mask_kwarg  # for bidirectional, can forward `mask` into the hybrid module and let it handle variable lengths

        # output dimension by default same as input, but can be overridden

        dim_out = default(dim_out, dim)
        self.to_out = (
            nn.Sequential(LinearNoBias(out_dim, dim_out * 2), nn.GLU())
            if on_attn
            else LinearNoBias(out_dim, dim_out)
        )

        # sublayer dropout

        self.sublayer_dropout = (
            nn.Dropout(sublayer_dropout) if sublayer_dropout > 0.0 else None
        )

        # the number of attention heads to rotate, for decoupled rope in multi-latent attention

        rotate_num_heads = default(rotate_num_heads, heads)

        assert 0 < rotate_num_heads <= heads
        is_partial_rotate_heads = rotate_num_heads < heads
        assert not (is_partial_rotate_heads and kv_heads < heads), (
            "grouped query attention not compatible with partial rotate heads (decoupled rope for multi-latent attention), yet"
        )

        self.rotate_num_heads = rotate_num_heads

        # whether parent can kv cache

        self.can_cache_kv = not selective

        # init output projection 0

        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        rel_pos=None,
        attn_bias=None,
        rotary_pos_emb=None,
        context_rotary_pos_emb=None,
        pos=None,  # for custom alibi positions
        prev_attn=None,
        mem=None,
        mem_mask=None,
        return_intermediates=False,
        cache: Intermediates | None = None,
        value_residual=None,
    ):
        (
            b,
            n,
            h,
            kv_h,
            head_scale,
            num_mem_kv,
            device,
            has_context,
            qkv_receive_diff_residuals,
            is_multi_latent_attn,
        ) = (
            x.shape[0],
            x.shape[1],
            self.heads,
            self.kv_heads,
            self.head_scale,
            self.num_mem_kv,
            x.device,
            exists(context),
            self.qkv_receive_diff_residuals,
            self.use_latent_kv,
        )

        # an interesting possibility with hyper connections
        # having queries, keys, values be routed from different layers

        assert not (qkv_receive_diff_residuals and has_context), (
            "qkv receiving different sequences can only be used for self attention"
        )

        if qkv_receive_diff_residuals:
            assert x.ndim == 4 and x.shape[0] == 3

            q_input, k_input, v_input = x
        else:
            kv_input = default(context, x)
            q_input, k_input, v_input = x, kv_input, kv_input

        if exists(mem):
            k_input, mem_packed_shape = pack([mem, k_input], "b * d")
            v_input, _ = pack([mem, v_input], "b * d")

        # multi-latent attention logic
        # https://arxiv.org/abs/2405.04434 - Deepseek-AI team

        k_sub_heads = None  # the rotateable subheads of keys derived from base sequence

        if self.use_latent_q:
            q_input = self.to_latent_q(q_input)

        if is_multi_latent_attn:
            assert not qkv_receive_diff_residuals
            needs_k_sub_heads = exists(self.to_rotateable_k)

            latent_kv_input = self.to_latent_kv(k_input)

            if needs_k_sub_heads:
                rotateable_k = self.to_rotateable_k(k_input)
                k_sub_heads = self.split_rotateable_k_heads(rotateable_k)

            if exists(cache):
                cached_latent_kv, maybe_cached_k_sub_heads = cache.cached_kv
                latent_kv_input = cat((cached_latent_kv, latent_kv_input), dim=-2)

                if exists(maybe_cached_k_sub_heads):
                    k_sub_heads = cat((maybe_cached_k_sub_heads, k_sub_heads), dim=-2)

            if return_intermediates:
                cached_kv = (latent_kv_input, k_sub_heads)

            k_input = v_input = latent_kv_input

        # query, key, value projection

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q = self.split_q_heads(q)
        k = self.split_k_heads(k)
        v = self.split_v_heads(v)

        # take care of decoupled rope from multi-latent attention

        if exists(k_sub_heads):
            k = cat((k, k_sub_heads), dim=1)

        # if previous values passed in for residual, either invoke resformer

        orig_values = v

        # https://arxiv.org/abs/2410.17897v1

        if exists(value_residual):
            value_residual_mix = self.to_value_residual_mix(q_input)
            v = value_residual.lerp(v, value_residual_mix)

        # qk normalization

        if self.qk_norm:
            qk_l2norm = partial(l2norm, groups=self.qk_norm_groups)
            q, k = map(qk_l2norm, (q, k))
            scale = self.qk_norm_scale

            q = q * self.qk_norm_q_scale
            k = k * self.qk_norm_k_scale

        # maybe value rmsnorm

        v = maybe(self.value_rmsnorm)(v)

        # take care of caching

        if not is_multi_latent_attn:
            if exists(cache):
                ck, cv = cache.cached_kv

                if exists(mem):
                    mk, k = unpack(k, mem_packed_shape, "b h * d")
                    mv, v = unpack(v, mem_packed_shape, "b h * d")

                k = cat((ck, k), dim=-2)
                v = cat((cv, v), dim=-2)

                if exists(mem):
                    k = cat((mk, k), dim=-2)
                    v = cat((mv, v), dim=-2)

            if return_intermediates:
                mem_len = mem.shape[-2] if exists(mem) else 0
                cached_kv = (k[..., mem_len:, :], v[..., mem_len:, :])

        if exists(rotary_pos_emb):
            rotate_num_heads = self.rotate_num_heads
            partial_rotate_heads = rotate_num_heads < h

            freqs, xpos_scale = rotary_pos_emb
            q_xpos_scale, k_xpos_scale = (
                (xpos_scale, xpos_scale**-1.0) if exists(xpos_scale) else (1.0, 1.0)
            )

            if partial_rotate_heads:
                q_rest, q = q[:, :-rotate_num_heads], q[:, -rotate_num_heads:]
                k_rest, k = k[:, :-rotate_num_heads], k[:, -rotate_num_heads:]

            q = apply_rotary_pos_emb(q, freqs, q_xpos_scale)

            if has_context:
                # override with `context_rotary_pos_emb` if provided

                freqs, xpos_scale = context_rotary_pos_emb
                _, k_xpos_scale = (
                    (xpos_scale, xpos_scale**-1.0) if exists(xpos_scale) else (1.0, 1.0)
                )

            k = apply_rotary_pos_emb(k, freqs, k_xpos_scale)

            if partial_rotate_heads:
                q = cat((q_rest, q), dim=1)
                k = cat((k_rest, k), dim=1)

        input_mask = context_mask

        if not exists(input_mask) and not has_context:
            input_mask = mask

            if (exists(input_mask) or exists(mem_mask)) and exists(mem):
                seq_len, mem_len = n, mem.shape[-2]

                if not exists(mem_mask):
                    input_mask = pad_at_dim(
                        input_mask, (mem_len, 0), dim=-1, value=True
                    )
                elif not exists(input_mask):
                    input_mask = pad_at_dim(mem_mask, (0, seq_len), dim=-1, value=True)
                else:
                    input_mask = cat((mem_mask, input_mask), dim=-1)

        # i, j determined for relative positional bias, excluding memory key / values

        i, j = tuple(t.shape[-2] for t in (q, k))

        # maybe append memory key / values

        if num_mem_kv > 0:
            mem_k, mem_v = tuple(
                repeat(t, "h n d -> b h n d", b=b) for t in (self.mem_k, self.mem_v)
            )

            if self.qk_norm:
                mem_k = l2norm(mem_k)
                mem_k = mem_k * self.qk_norm_k_scale

            k = cat((mem_k, k), dim=-2)
            v = cat((mem_v, v), dim=-2)

            if exists(input_mask):
                input_mask = pad_at_dim(
                    input_mask, (self.num_mem_kv, 0), dim=-1, value=True
                )

        # determine masking

        mask_value = max_neg_value(q)
        masks = []
        final_attn_mask = None

        if exists(input_mask):
            input_mask = rearrange(input_mask, "b j -> b 1 1 j")
            masks.append(~input_mask)

        if exists(attn_mask):
            assert 2 <= attn_mask.ndim <= 4, (
                "attention mask must have greater than 2 dimensions but less than or equal to 4"
            )
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, "i j -> 1 1 i j")
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, "h i j -> 1 h i j")
            masks.append(~attn_mask)

        if exists(self.max_attend_past):
            range_q = arange(j - i, j, device=device)
            range_k = arange(j, device=device)
            dist = einx.subtract("i, j -> 1 1 i j", range_q, range_k)
            max_attend_past_mask = dist > self.max_attend_past
            max_attend_past_mask = pad_at_dim(
                max_attend_past_mask, (num_mem_kv, 0), value=False, dim=-1
            )  # handle memory key / values
            masks.append(max_attend_past_mask)

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        # prepare relative positional bias, if needed

        if exists(rel_pos):
            assert not exists(attn_bias)

            if exists(pos):
                assert isinstance(rel_pos, AlibiPositionalBias), (
                    "only alibi allowed for custom positions at the moment"
                )
                # allow for custom positions to be passed in
                attn_bias = rel_pos.forward_custom_pos(pos)
            else:
                attn_bias = rel_pos(i, j)

            attn_bias = pad_at_dim(
                attn_bias, (num_mem_kv, 0)
            )  # handle memory key / values

        # prepare data dependent alibi from forgetting transformers paper, if needed

        if exists(self.data_dependent_alibi):
            attn_bias = self.data_dependent_alibi(x)

            attn_bias = pad_at_dim(attn_bias, (num_mem_kv, 0))

        if self.laser:
            v = softclamp(v, self.laser_softclamp_value)
            v = v.exp()

        # attention is all we need

        out, intermediates = self.attend(
            q, k, v, mask=final_attn_mask, attn_bias=attn_bias, prev_attn=prev_attn
        )

        # laser

        if self.laser:
            out = log(out)

        # store the values for resformer

        intermediates.values = orig_values

        # normformer scaling of heads

        if head_scale:
            out = out * self.head_scale_params

        # per head gating, from https://arxiv.org/abs/2306.12929

        if exists(self.to_v_head_gate):
            head_gate = self.to_v_head_gate(x)
            out = einx.multiply("b n h, b h n d ->b h n d", head_gate.sigmoid(), out)

        # if exists hybrid module, must do a normalization

        # hybrid module

        if exists(self.hybrid_module):
            # hybrid input

            hybrid_forward_kwargs = dict()

            if not self.causal and exists(self.hybrid_mask_kwarg):
                hybrid_forward_kwargs = {self.hybrid_mask_kwarg: mask}

            # handle maybe hybrid cache

            hybrid_forward_args = ()

            if exists(cache) and exists(cache.hybrid_hidden):
                hybrid_hiddens = cache.hybrid_hidden
                hybrid_forward_args = (hybrid_hiddens,)

            # hybrid forward

            hybrid_outputs = self.hybrid_module(
                x, *hybrid_forward_args, **hybrid_forward_kwargs
            )

            # handle hybrid out

            (hybrid_out, *rest_hybrid_outs), _ = tree_flatten(hybrid_outputs)

            # handle variable hybrid output and multi rmsnorm before summing to main attention output (also normed)

            if hybrid_out.ndim == 3:
                hybrid_out = rearrange(hybrid_out, "b n (h d) -> b h n d", h=h)

            if len(rest_hybrid_outs) > 0:
                hybrid_hidden = first(rest_hybrid_outs)
                intermediates.hybrid_hidden = hybrid_hidden

            out_norm, hybrid_out_norm = self.hybrid_norms

            out = out_norm(out)
            hybrid_out = hybrid_out_norm(hybrid_out)

            if exists(self.hybrid_mix):
                mix = self.hybrid_mix(x)
                mix = rearrange(mix, "b n h -> b h n 1")
                out = out.lerp(hybrid_out, mix.sigmoid())
            else:
                out = 0.5 * (out + hybrid_out)

        # merge heads

        out = self.merge_heads(out)

        # alphafold2 styled gating of the values

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            out = out * self.to_v_gate_activation(gates)

        # combine the heads

        out = self.to_out(out)

        # maybe sublayer dropout

        out = maybe(self.sublayer_dropout)(out)

        if exists(mask) and not exists(cache):
            out = einx.where("b n, b n d, -> b n d", mask, out, 0.0)

        if not return_intermediates:
            return out

        intermediates.cached_kv = cached_kv

        return out, intermediates

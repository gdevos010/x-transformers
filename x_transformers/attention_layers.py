from __future__ import annotations

from functools import partial

import einx
import torch

from einops import rearrange, reduce
from loguru import logger
from torch import Tensor, arange, nn
from torch.nn import Module, ModuleList

from x_transformers.attention import Attention
from x_transformers.components import (
    ConcatCombine,
    GRUGating,
    Residual,
    ShiftTokens,
)
from x_transformers.feedforward import FeedForward
from x_transformers.hyper_connections import HyperConnection
from x_transformers.layer_intermediates import LayerIntermediates
from x_transformers.layer_scale import AdaptiveLayerScale, LayerScale
from x_transformers.lime import DynamicLIMe
from x_transformers.norms import (
    AdaptiveLayerNorm,
    AdaptiveRMSNorm,
    DynamicTanh,
    LayerNorm,
    RMSNorm,
    Scale,
    ScaleNorm,
    SimpleRMSNorm,
)
from x_transformers.postional_embeddings import (
    AlibiPositionalBias,
    DynamicPositionBias,
    RelativePositionBias,
    RotaryEmbedding,
)
from x_transformers.utils import (
    DEFAULT_DIM_HEAD,
    LinearNoBias,
    at_most_one_of,
    cast_tuple,
    default,
    divisible_by,
    equals,
    exists,
    first,
    groupby_prefix_and_trim,
    maybe,
    not_equals,
    softclamp,
)


class AttentionLayers(Module):
    def __init__(
        self,
        dim,
        depth=None,
        heads=8,
        causal=False,
        cross_attend=False,
        only_cross=False,
        use_scalenorm=False,
        use_rmsnorm=False,
        use_dynamic_tanh=False,
        dynamic_tanh_init_alpha=1.0,
        use_simple_rmsnorm=False,
        use_adaptive_layernorm=False,
        use_adaptive_rmsnorm=False,
        use_adaptive_layerscale=False,  # paired with use_adaptive_layernorm for ada-ln-zero from DiT paper
        norm_add_unit_offset=True,
        dim_condition=None,
        adaptive_condition_mlp=False,
        adaptive_condition_mlp_expansion=4,
        alibi_pos_bias=False,
        alibi_num_heads=None,
        rel_pos_bias=False,
        rel_pos_num_buckets=32,
        rel_pos_max_distance=128,
        dynamic_pos_bias=False,
        dynamic_pos_bias_log_distance=False,
        dynamic_pos_bias_mlp_depth=2,
        dynamic_pos_bias_norm=False,
        rotary_pos_emb=False,
        rotary_emb_dim=None,
        rotary_xpos=False,
        rotary_interpolation_factor=1.0,
        rotary_xpos_scale_base=512,
        rotary_base_rescale_factor=1.0,
        rotate_num_heads=None,
        weight_tie_layers=False,
        custom_layers: tuple[str, ...] | None = None,
        layers_execute_order: tuple[int, ...] | None = None,
        sandwich_coef=None,
        par_ratio=None,
        residual_attn=False,
        cross_residual_attn=False,
        macaron=False,
        pre_norm=True,
        pre_norm_has_final_norm=True,
        gate_residual=False,
        scale_residual=False,
        scale_residual_constant=1.0,
        shift_tokens=0,
        sandwich_norm=False,
        softclamp_output=False,
        softclamp_output_value=30.0,
        zero_init_branch_output=False,
        layer_dropout=0.0,
        cross_attn_tokens_dropout=0.0,
        disable_abs_pos_emb=None,
        use_layerscale=False,
        layerscale_init_value=0.0,
        unet_skips=False,
        integrate_layers=False,
        layer_integrate_use_softmax=True,
        num_residual_streams=1,
        qkv_receive_diff_residuals=False,
        reinject_input=False,  # seen first in DEQ paper https://arxiv.org/abs/1909.01377, but later used in a number of papers trying to achieve depthwise generalization https://arxiv.org/abs/2410.03020v1
        learned_reinject_input_gate=False,
        add_value_residual=False,  # resformer from Zhou et al - https://arxiv.org/abs/2410.17897v1 - further corroboration by https://arxiv.org/abs/2412.15113 (faster emergence of ICL) - looks like this setting may becoming a necessity for every transformer soon
        learned_value_residual_mix=True,  # seeing big improvements when the value residual mix value is learned per token - credit goes to @faresobeid for taking the first step with learned scalar mix, then @Blinkdl for taking it a step further with data dependent. here we will use per token learned
        rel_pos_kwargs: dict = dict(),
        residual_fn_kwargs: dict = dict(),
        verbose=True,
        **kwargs,
    ):
        super().__init__()
        rotary_pos_emb = rotary_pos_emb or rotary_xpos

        ff_kwargs, kwargs = groupby_prefix_and_trim("ff_", kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim("attn_", kwargs)
        cross_attn_kwargs, kwargs = groupby_prefix_and_trim("cross_attn_", kwargs)

        dim_head = attn_kwargs.get("dim_head", DEFAULT_DIM_HEAD)
        data_dependent_alibi = attn_kwargs.get("data_dependent_alibi", False)

        assert len(kwargs) == 0, f"unrecognized kwargs passed in {kwargs.keys()}"

        self.dim = dim
        self.causal = causal
        self.layers = ModuleList([])

        self.attn_heads = heads
        self.attn_dim_head = dim_head

        # routing related
        # 1. greater than one residual stream, proposed in Hyper-Connections paper https://arxiv.org/abs/2409.19606
        # 2. integrating more than one past layer, from LIMe paper https://arxiv.org/abs/2502.09245

        qkv_receive_diff_residuals |= integrate_layers  # qkv always receives different views if integrating layers

        # hyper connections

        assert num_residual_streams > 0
        has_hyper_connections = num_residual_streams > 1

        self.num_residual_streams = num_residual_streams
        self.stream_emb = (
            nn.Parameter(torch.zeros(num_residual_streams, dim))
            if num_residual_streams > 1
            else None
        )

        assert not (has_hyper_connections and gate_residual)

        hyper_conn_produce_diff_views = (
            qkv_receive_diff_residuals and not integrate_layers
        )

        # LIMe

        hiddens_counter = 0
        self.layer_integrators = ModuleList([])

        assert not (
            qkv_receive_diff_residuals
            and not (hyper_conn_produce_diff_views or integrate_layers)
        )

        # positions related

        self.disable_abs_pos_emb = default(
            disable_abs_pos_emb, (rel_pos_bias or rotary_pos_emb)
        )

        rotary_emb_dim = default(rotary_emb_dim, dim_head // 2)

        assert rotary_emb_dim <= dim_head, (
            f"rotary emb dim {rotary_emb_dim} must be less than or equal to attention head dimension {dim_head}"
        )

        if verbose and rotary_emb_dim < 32:
            logger.warning(
                "when training language model, rotary embedding dimension should be at least 32"
            )

        assert not (rotary_xpos and not causal), (
            "rotary xpos is not compatible with bidirectional attention"
        )
        self.rotary_pos_emb = (
            RotaryEmbedding(
                rotary_emb_dim,
                use_xpos=rotary_xpos,
                scale_base=rotary_xpos_scale_base,
                interpolation_factor=rotary_interpolation_factor,
                base_rescale_factor=rotary_base_rescale_factor,
            )
            if rotary_pos_emb
            else None
        )

        assert at_most_one_of(alibi_pos_bias, rel_pos_bias, data_dependent_alibi), (
            "you can only choose one of Alibi positional bias, data dependent Alibi (forgetting transformers), dynamic tanh, or T5 relative positional bias"
        )
        assert rel_pos_num_buckets <= rel_pos_max_distance, (
            "number of relative position buckets must be less than the relative position max distance"
        )

        # relative positional bias

        flash_attn = attn_kwargs.get("flash", False)
        assert at_most_one_of(rel_pos_bias, dynamic_pos_bias, alibi_pos_bias), (
            "you can only choose up to one of t5, alibi, or dynamic positional bias"
        )

        self.rel_pos = None

        if rel_pos_bias:
            assert not flash_attn, (
                "flash attention not compatible with t5 relative positional bias"
            )
            self.rel_pos = RelativePositionBias(
                scale=dim_head**0.5,
                causal=causal,
                heads=heads,
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
                **rel_pos_kwargs,
            )
        elif dynamic_pos_bias:
            assert not flash_attn, (
                "flash attention not compatible with dynamic positional bias"
            )
            self.rel_pos = DynamicPositionBias(
                dim=dim // 4,
                heads=heads,
                log_distance=dynamic_pos_bias_log_distance,
                depth=dynamic_pos_bias_mlp_depth,
                norm=dynamic_pos_bias_norm,
                **rel_pos_kwargs,
            )
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, (
                "number of ALiBi heads must be less than the total number of heads"
            )
            self.rel_pos = AlibiPositionalBias(
                heads=alibi_num_heads, total_heads=heads, **rel_pos_kwargs
            )

        assert not (not pre_norm and sandwich_norm), (
            "sandwich norm cannot be used when not using prenorm"
        )

        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        assert not (flash_attn and (residual_attn or cross_residual_attn)), (
            "flash attention is not compatible with residual attention"
        )

        self.cross_attend = cross_attend

        # determine norm

        assert at_most_one_of(
            use_scalenorm,
            use_rmsnorm,
            use_dynamic_tanh,
            use_simple_rmsnorm,
            use_adaptive_layernorm,
            use_adaptive_rmsnorm,
        ), (
            "you can only use either scalenorm, rmsnorm, adaptive layernorm, adaptive rmsnorm, or simple rmsnorm"
        )

        norm_need_condition = False
        dim_condition = default(dim_condition, dim)
        dim_condition_mult = 1

        if adaptive_condition_mlp:
            dim_condition_mult = adaptive_condition_mlp_expansion

        if use_scalenorm:
            norm_class = ScaleNorm
        elif use_rmsnorm:
            norm_class = RMSNorm
        elif use_simple_rmsnorm:
            norm_class = SimpleRMSNorm
        elif use_dynamic_tanh:
            assert pre_norm, "dynamic tanh norm only tested for pre-norm"
            norm_class = partial(DynamicTanh, init_alpha=dynamic_tanh_init_alpha)
        elif use_adaptive_layernorm:
            norm_need_condition = True
            norm_class = partial(
                AdaptiveLayerNorm, dim_condition=dim_condition * dim_condition_mult
            )
        elif use_adaptive_rmsnorm:
            norm_need_condition = True
            norm_class = partial(
                AdaptiveRMSNorm, dim_condition=dim_condition * dim_condition_mult
            )
        else:
            norm_class = LayerNorm

        norm_fn = partial(norm_class, dim)

        if not norm_need_condition and norm_add_unit_offset:
            # researcher Ohad Rubin shares in a blog post by adding an offset to gammas, they can be subjected to weight decay safely
            norm_fn = partial(norm_fn, unit_offset=True)

        self.norm_need_condition = norm_need_condition
        self.dim_condition = dim_condition

        # determine default block layer type order

        if cross_attend and not only_cross:
            default_block = ("a", "c", "f")
        elif cross_attend and only_cross:
            default_block = ("c", "f")
        else:
            default_block = ("a", "f")

        if macaron:
            default_block = ("f",) + default_block

        # determine post branch wrapper

        assert at_most_one_of(use_layerscale, use_adaptive_layerscale)

        post_branch_fn = None
        post_branch_fn_needs_condition = False

        if use_layerscale:
            post_branch_fn = partial(
                LayerScale, dim=dim, init_value=layerscale_init_value
            )
        elif use_adaptive_layerscale:
            post_branch_fn = partial(
                AdaptiveLayerScale,
                dim=dim,
                dim_condition=dim_condition * dim_condition_mult,
            )
            post_branch_fn_needs_condition = True

        self.post_branch_fn_needs_condition = post_branch_fn_needs_condition

        if (
            exists(post_branch_fn)
            and not post_branch_fn_needs_condition
            and norm_add_unit_offset
        ):
            post_branch_fn = partial(post_branch_fn, unit_offset=True)

        # setup mlp for conditioning

        self.need_condition = norm_need_condition or post_branch_fn_needs_condition

        self.adaptive_mlp = nn.Identity()

        if self.need_condition and adaptive_condition_mlp:
            self.adaptive_mlp = nn.Sequential(
                LinearNoBias(dim_condition, dim_condition * dim_condition_mult),
                nn.SiLU(),
            )

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, "zero_init_output": True}
            ff_kwargs = {**ff_kwargs, "zero_init_output": True}

        # setup weight tying, which is a special case of `layer_execute_order`

        assert not (
            exists(layers_execute_order) and exists(custom_layers) and exists(depth)
        ), (
            "depth should not be passed in if using custom layers and custom layer execution order"
        )

        assert not (
            weight_tie_layers
            and any([*map(exists, (custom_layers, par_ratio, sandwich_coef))])
        )

        if weight_tie_layers:
            assert exists(depth), (
                "depth must be passed in with `weight_tie_layers` = True"
            )
            assert not exists(layers_execute_order)
            layers_execute_order = tuple(range(len(default_block))) * depth
            depth = 1

        # calculate layer block order

        len_default_block = 1

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, "par ratio out of range"
            default_block = tuple(filter(not_equals("f"), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = (
                par_depth * 2 // 3
            )  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, (
                "default block is too large for par_ratio"
            )
            par_block = default_block + ("f",) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ("f",) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, (
                "sandwich coefficient should be less than the depth"
            )
            layer_types = (
                ("a",) * sandwich_coef
                + default_block * (depth - sandwich_coef)
                + ("f",) * sandwich_coef
            )
        else:
            assert exists(depth), "`depth` must be passed in for `Decoder` or `Encoder`"
            layer_types = default_block * depth
            len_default_block = len(default_block)

        self.layer_types = layer_types
        self.layers_execute_order = default(
            layers_execute_order, tuple(range(len(layer_types)))
        )

        assert all([i < len(self.layer_types) for i in self.layers_execute_order])

        self.num_attn_layers = len(list(filter(equals("a"), layer_types)))

        # set the depth

        depth = default(depth, len(self.layers_execute_order))
        self.depth = depth

        # stochastic depth

        self.layer_dropouts = cast_tuple(layer_dropout, len(layer_types))

        # structured dropout for cross attending

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # optional soft clamping just before the final norm
        # used in gemma 2

        self.softclamp_output = softclamp_output
        self.softclamp_output_value = softclamp_output_value

        # whether it has post norm

        self.final_norm = (
            norm_fn() if pre_norm and pre_norm_has_final_norm else nn.Identity()
        )

        # whether unet or not

        self.unet_skips = unet_skips
        num_skips = self.depth // len_default_block

        assert not (unet_skips and num_skips == 0), (
            "must have depth of at least 2 for unet skip connections"
        )

        skip_indices = [i * len_default_block for i in range(num_skips)]

        self.skip_combines = ModuleList([])

        # whether there is reinjection of input at every layer

        self.reinject_input = reinject_input
        self.reinject_input_proj = (
            nn.Linear(dim, dim, bias=False) if reinject_input else None
        )
        self.learned_reinject_input_gate = (
            nn.Linear(dim, 1, bias=False) if learned_reinject_input_gate else None
        )

        # add the value from the first self attention block to all latter projected self attention values as a residual

        self.add_value_residual = add_value_residual

        is_first_self_attn = True
        is_first_cross_attn = True
        learned_value_residual_mix &= add_value_residual

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(
            zip(self.layer_types, shift_tokens, strict=False)
        ):
            # `ind` is the index of each module - attention, feedforward, cross attention
            # but `block_ind` refers to the typical enumeration of a transformer block (attn + ff + [optional] cross attn)

            block_begin = divisible_by(ind, len_default_block)
            block_ind = ind // len_default_block

            is_last_layer = ind == (len(self.layer_types) - 1)

            # attention, cross attention, feedforward

            layer_qkv_receives_diff_view = (
                layer_type == "a"
                and qkv_receive_diff_residuals
                and not (is_first_self_attn and integrate_layers)
            )

            if layer_type == "a":
                self_attn_learned_value_residual = (
                    learned_value_residual_mix and not is_first_self_attn
                )

                layer = Attention(
                    dim,
                    heads=heads,
                    causal=causal,
                    qkv_receive_diff_residuals=layer_qkv_receives_diff_view,
                    learned_value_residual_mix=self_attn_learned_value_residual,
                    rotate_num_heads=rotate_num_heads,
                    **attn_kwargs,
                )
                is_first_self_attn = False

            elif layer_type == "c":
                layer = Attention(
                    dim, heads=heads, **{**attn_kwargs, **cross_attn_kwargs}
                )
                is_first_cross_attn = False

            elif layer_type == "f":
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)

            else:
                raise Exception(f"invalid layer type {layer_type}")

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if exists(post_branch_fn):
                layer = post_branch_fn(layer)

            layer_integrate = None

            if integrate_layers:
                num_layer_hiddens = ind + 1
                layer_integrate_num_view = 3 if layer_qkv_receives_diff_view else 1

                layer_integrate = DynamicLIMe(
                    dim,
                    num_layer_hiddens,
                    num_views=layer_integrate_num_view,
                    use_softmax=layer_integrate_use_softmax,
                )

            if has_hyper_connections:
                residual_fn = partial(
                    HyperConnection, num_residual_streams=num_residual_streams
                )

                if layer_type == "a" and hyper_conn_produce_diff_views:
                    residual_fn = partial(residual_fn, num_input_views=3)

            elif gate_residual:
                residual_fn = GRUGating
            else:
                residual_fn = Residual

            residual = residual_fn(
                dim,
                layer_index=ind,
                scale_residual=scale_residual,
                scale_residual_constant=scale_residual_constant,
                **residual_fn_kwargs,
            )

            # handle unet skip connection

            skip_combine = None
            is_latter_half = block_begin and block_ind >= (self.depth / 2)

            if self.unet_skips and is_latter_half:
                skip_combine = ConcatCombine(dim, skip_indices.pop())

            # all normalizations of the layer

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm else None
            post_main_norm = norm_fn() if not pre_norm else None

            norms = ModuleList([pre_branch_norm, post_branch_norm, post_main_norm])

            self.skip_combines.append(skip_combine)

            self.layer_integrators.append(layer_integrate)

            self.layers.append(ModuleList([norms, layer, residual]))

        # determine whether can cache kv

        self.can_cache_kv = all(
            [
                module.can_cache_kv
                for module in self.modules()
                if isinstance(module, Attention)
            ]
        )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        self_attn_kv_mask=None,
        mems=None,
        mem_masks=None,
        seq_start_pos: Tensor | None = None,
        seq_pos_offset: int = 0,
        cache: LayerIntermediates | None = None,
        input_not_include_cache=False,
        cache_age=1,
        return_hiddens=False,
        rotary_pos_emb=None,
        pos=None,
        context_pos=None,
        attn_bias=None,
        deep_embeds_and_ids: tuple[nn.Parameter, Tensor] | None = None,
        condition=None,
        in_attn_cond=None,  # https://arxiv.org/abs/2105.04090
        layers_execute_order: tuple[int, ...] | None = None,
    ):
        assert not (self.cross_attend ^ exists(context)), (
            "context must be passed in if cross_attend is set to True"
        )
        assert not (exists(condition) ^ self.need_condition), (
            "condition needs to be passed in if using adaptive layernorm or vice versa"
        )

        # handle condition

        if exists(condition):
            assert condition.shape[-1] == self.dim_condition, (
                f"expected condition dimension of {self.dim_condition} but received {condition.shape[-1]}"
            )

            assert condition.ndim in {2, 3}

            if condition.ndim == 2:
                condition = rearrange(condition, "b d -> b 1 d")

            condition = self.adaptive_mlp(condition)

        # setup maybe layernorm kwarg

        norm_kwargs = dict()

        if self.norm_need_condition:
            norm_kwargs.update(condition=condition)

        # maybe post branch fn conditioning (DiT paper's ada-ln-zero)

        block_forward_kwargs = dict()

        if self.post_branch_fn_needs_condition:
            block_forward_kwargs.update(condition=condition)

        # initialize accums

        hiddens = []
        layer_hiddens = []
        intermediates = []

        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers
        mem_masks = (
            mem_masks.copy() if exists(mem_masks) else [None] * self.num_attn_layers
        )

        # handle left padded sequences

        if exists(seq_start_pos):
            seq_arange = arange(x.shape[-2], device=x.device, dtype=torch.long)
            left_pad_mask = seq_arange >= seq_start_pos[..., None]

            if exists(self_attn_kv_mask):
                self_attn_kv_mask = self_attn_kv_mask & left_pad_mask
            else:
                self_attn_kv_mask = left_pad_mask

        # rotary positions

        cross_attn_rotary_pos_emb = dict()

        if exists(self.rotary_pos_emb):
            if not exists(rotary_pos_emb):
                maybe_mem = first(
                    mems, None
                )  # todo - handle edge case where different layers get different memory lengths. don't think this will ever come up but who knows
                mem_len = maybe_mem.shape[1] if exists(maybe_mem) else 0

                if not exists(pos):
                    pos = (
                        arange(x.shape[1] + mem_len + seq_pos_offset, device=x.device)
                        - mem_len
                    )

                rotary_pos_emb = self.rotary_pos_emb(pos)

            # allow for rotary positions for context if provided

            if exists(context_pos):
                assert self.cross_attend
                context_rotary_pos_emb = self.rotary_pos_emb(context_pos)

                cross_attn_rotary_pos_emb.update(
                    rotary_pos_emb=rotary_pos_emb,
                    context_rotary_pos_emb=context_rotary_pos_emb,
                )

        # assume cached key / values

        prev_cache_length = 0

        attn_cache = []

        if exists(cache):
            assert self.causal and not exists(attn_mask)

            prev_cache_length = cache.cache_length

            if exists(context):
                context = context[:, :0]

            if cache_age > 0:
                x = x[:, -cache_age:]  # for spec decoding, may be greater than 1

                if exists(deep_embeds_and_ids):
                    deep_embeds, token_ids = deep_embeds_and_ids
                    token_ids = token_ids[:, -cache_age:]
                    deep_embeds_and_ids = (deep_embeds, token_ids)

            attn_cache = cache.attn_intermediates

        next_cache_length = x.shape[1]

        iter_attn_cache = iter(attn_cache)

        # handle deep embeds if needed

        deep_embeds = []

        if exists(deep_embeds_and_ids):
            deep_embeds, token_ids = deep_embeds_and_ids
            deep_embeds_across_depth = deep_embeds[token_ids]
            deep_embeds = rearrange(deep_embeds_across_depth, "b n l d -> l b n d")

        deep_embeds_iter = iter(deep_embeds)

        # setup multistreams if needed

        streams = self.num_residual_streams
        is_multistream = streams > 1

        if is_multistream:
            x = einx.add("b n d, s d -> (b s) n d", x, self.stream_emb)

        # get layers to be executed

        layer_variables = (
            self.layer_types,
            self.skip_combines,
            self.layers,
            self.layer_dropouts,
            self.layer_integrators,
        )

        # able to override the layers execution order on forward, for trying to depth extrapolate

        layers_execute_order = default(layers_execute_order, self.layers_execute_order)
        layer_variables = tuple(
            tuple(layer_variable[i] for i in layers_execute_order)
            for layer_variable in layer_variables
        )

        # derived input for reinjection if needed

        inp_inject = None

        if self.reinject_input:
            assert not exists(in_attn_cond)
            inp_inject = self.reinject_input_proj(x)

        elif exists(in_attn_cond):
            # handle in-attention conditioning, which serves the same purpose of having the network learn the residual
            inp_inject = (
                in_attn_cond
                if in_attn_cond.ndim == 3
                else rearrange(in_attn_cond, "b d -> b 1 d")
            )

        if exists(inp_inject) and exists(self.learned_reinject_input_gate):
            inp_inject_gate = self.learned_reinject_input_gate(x).sigmoid()
            inp_inject = inp_inject * inp_inject_gate

        # store all hiddens for skips

        skip_hiddens = []

        # for value residuals

        first_self_attn_inter = None
        first_cross_attn_inter = None

        # go through the attention and feedforward layers

        for ind, (
            layer_type,
            skip_combine,
            (norm, block, residual_fn),
            layer_dropout,
            layer_integrator,
        ) in enumerate(zip(*layer_variables, strict=False)):
            is_last = ind == (len(self.layers) - 1)

            # handle skip connections

            skip_hiddens.append(x)

            if exists(skip_combine):
                x = skip_combine(x, skip_hiddens)

            # layer dropout

            if self.training and layer_dropout > 0.0 and random() < layer_dropout:
                continue

            if layer_type == "a":
                if return_hiddens:
                    hiddens.append(x)

                layer_mem = mems.pop(0) if mems else None
                layer_mem_mask = mem_masks.pop(0) if mem_masks else None

            if layer_type == "c":
                if self.training and self.cross_attn_tokens_dropout > 0.0:
                    context, context_mask = dropout_seq(
                        context, context_mask, self.cross_attn_tokens_dropout
                    )

            x, inner_residual, residual_kwargs = residual_fn.prepare(x)

            layer_hiddens.append(x)

            if exists(layer_integrator):
                x = layer_integrator(x, layer_hiddens)

            pre_norm, post_branch_norm, post_main_norm = norm

            if self.need_condition:
                pre_norm = maybe(partial)(pre_norm, **norm_kwargs)
                post_branch_norm = maybe(partial)(post_branch_norm, **norm_kwargs)
                post_main_norm = maybe(partial)(post_main_norm, **norm_kwargs)

            if exists(inp_inject):
                x = x + inp_inject

            if exists(pre_norm):
                x = pre_norm(x)

                if layer_type == "a" and exists(layer_mem):
                    layer_mem = pre_norm(layer_mem)

            block = partial(block, **block_forward_kwargs)

            # handle maybe value residuals

            maybe_self_attn_value_residual = None
            maybe_cross_attn_value_residual = None

            if self.add_value_residual:
                if exists(first_self_attn_inter):
                    maybe_self_attn_value_residual = first_self_attn_inter.values

                if exists(first_cross_attn_inter):
                    maybe_cross_attn_value_residual = first_cross_attn_inter.values

            # forward depending on layer type

            if layer_type == "a":
                out, inter = block(
                    x,
                    mask=mask,
                    context_mask=self_attn_kv_mask,
                    attn_mask=attn_mask,
                    rel_pos=self.rel_pos,
                    pos=pos,
                    rotary_pos_emb=rotary_pos_emb,
                    prev_attn=prev_attn,
                    cache=next(iter_attn_cache, None),
                    mem=layer_mem,
                    mem_mask=layer_mem_mask,
                    attn_bias=attn_bias,
                    value_residual=maybe_self_attn_value_residual,
                    return_intermediates=True,
                )
            elif layer_type == "c":
                out, inter = block(
                    x,
                    context=context,
                    mask=mask,
                    context_mask=context_mask,
                    prev_attn=prev_cross_attn,
                    cache=next(iter_attn_cache, None),
                    value_residual=maybe_cross_attn_value_residual,
                    **cross_attn_rotary_pos_emb,
                    return_intermediates=True,
                )
            elif layer_type == "f":
                out = block(x, deep_embed=next(deep_embeds_iter, None))

            # store first self or cross attention intermediate for value residual

            if not exists(first_self_attn_inter) and layer_type == "a":
                first_self_attn_inter = inter

            if not exists(first_cross_attn_inter) and layer_type == "c":
                first_cross_attn_inter = inter

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, inner_residual, **residual_kwargs)

            if layer_type in ("a", "c") and return_hiddens:
                inter.layer_type = layer_type
                intermediates.append(inter)

            if layer_type == "a" and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == "c" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            layer_hiddens.append(x)

        if self.softclamp_output:
            x = softclamp(x, self.softclamp_output_value)

        final_norm = self.final_norm

        if self.need_condition:
            final_norm = maybe(partial)(final_norm, **norm_kwargs)

        # take care of multistreams if needed, use sum for now

        if is_multistream:
            x = reduce(x, "(b s) n d -> b n d", "sum", s=streams)

        x = final_norm(x)

        if not return_hiddens:
            return x

        intermediates = LayerIntermediates(
            hiddens=hiddens,
            last_hidden=x,
            attn_intermediates=intermediates,
            layer_hiddens=layer_hiddens,
            cache_length=next_cache_length + prev_cache_length,
        )

        return x, intermediates

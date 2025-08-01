from dataclasses import dataclass

from torch import Tensor

from x_transformers.attend import Intermediates


@dataclass
class LayerIntermediates:
    hiddens: list[Tensor] | None = (
        None  # all hiddens, before the final norm (in pre-norm architecture)
    )
    last_hidden: Tensor | None = (
        None  # very last hidden after all attention layers, after the final norm
    )
    attn_intermediates: list[Intermediates] | None = None
    layer_hiddens: list[Tensor] | None = None
    attn_z_loss: Tensor | None = None
    mems: Tensor | None = None
    last_layer_hiddens: Tensor | None = None
    attn_pooled_tokens: Tensor | None = None
    memory_tokens: Tensor | None = None
    logit_entropies: Tensor | None = None
    logits: Tensor | None = None
    cache_length: int = 0

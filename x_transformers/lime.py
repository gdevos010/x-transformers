# LIMe - layer integrated memory (dynamic version)
from torch.functional import Tensor


from einops.layers.torch import Rearrange
from einops import rearrange
from x_transformers.utils import (
    Sequential,
)
from torch import nn
from torch.nn import Module
from x_transformers.norms import (
    RMSNorm,
)
from torch import einsum, stack, is_tensor


class DynamicLIMe(Module):
    def __init__(self, dim, num_layers, num_views=1, norm=True, use_softmax=True):
        super().__init__()
        self.num_layers = num_layers
        self.multiple_views = num_views > 1

        self.to_weights = Sequential(
            RMSNorm(dim) if norm else None,
            nn.Linear(dim, num_views * num_layers),
            Rearrange("... (views layers) -> views ... layers", views=num_views),
            nn.Softmax(dim=-1) if use_softmax else nn.ReLU(),
        )

    def forward(self, x, hiddens) -> Tensor:
        if not is_tensor(hiddens):
            hiddens = stack(hiddens)

        assert hiddens.shape[0] == self.num_layers, (
            f"expected hiddens to have {self.num_layers} layers but received {tuple(hiddens.shape)} instead (first dimension must be layers)"
        )

        weights = self.to_weights(x)

        out = einsum("l b n d, v b n l -> v b n d", hiddens, weights)

        if self.multiple_views:
            return out

        return rearrange(out, "1 ... -> ...")

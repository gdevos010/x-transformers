# hyper connections
import torch

from einops import rearrange
from torch import cat, einsum, nn
from torch.nn import Module


class HyperConnection(Module):
    def __init__(
        self,
        dim,
        *,
        layer_index,
        num_residual_streams,
        num_input_views=1,
        tanh=True,
        **kwargs,
    ) -> None:
        """https://arxiv.org/abs/2409.19606
        Appendix J - Algorithm 2, Dynamic only
        """
        super().__init__()

        self.act = nn.Tanh() if tanh else nn.Identity()

        self.norm = nn.LayerNorm(dim, bias=False)

        self.num_residual_streams = num_residual_streams
        self.layer_index = layer_index

        self.static_beta = nn.Parameter(torch.ones(num_residual_streams))

        init_alpha0 = torch.zeros((num_residual_streams, num_input_views))
        init_alpha0[layer_index % num_residual_streams, :] = 1.0

        self.static_alpha = nn.Parameter(
            cat([init_alpha0, torch.eye(num_residual_streams)], dim=1)
        )

        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(dim, num_residual_streams + num_input_views)
        )
        self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.num_input_views = num_input_views

        self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
        self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)

    def prepare(self, residuals):
        residuals = rearrange(
            residuals, "(b s) n d -> b n s d", s=self.num_residual_streams
        )

        normed = self.norm(residuals)

        wc_weight = self.act(normed @ self.dynamic_alpha_fn)
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale
        alpha = dynamic_alpha + self.static_alpha

        dc_weight = self.act(normed @ self.dynamic_beta_fn)
        dynamic_beta = dc_weight * self.dynamic_beta_scale
        beta = dynamic_beta + self.static_beta

        # width connection

        mix_h = einsum("... s t, ... s d -> ... t d", alpha, residuals)

        views = self.num_input_views

        if views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., :views, :], mix_h[..., views:, :]
            branch_input = rearrange(branch_input, "... v d -> v ... d")

        return branch_input, residuals, dict(beta=beta)

    def forward(self, x, residuals, *, beta):
        residuals = einsum("b n d, b n s -> b n s d", x, beta) + residuals
        return rearrange(residuals, "b n s d -> (b s) n d")

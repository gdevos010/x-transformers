import torch.nn.functional as F

from torch._tensor import Tensor
from torch.nn import Module

# activations


class ReluSquared(Module):
    def forward(self, x) -> Tensor:
        return F.relu(x) ** 2

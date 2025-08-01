from torch._tensor import Tensor


from torch.nn import Module
import torch.nn.functional as F


# activations


class ReluSquared(Module):
    def forward(self, x) -> Tensor:
        return F.relu(x) ** 2

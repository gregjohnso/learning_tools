import torch
import torch.nn as nn


class LLMLossWrapper(nn.Module):
    """
    Wrapper for loss functions that expect a flattened view of the input and target.

    We do this because the LLM/tokenized outputs expect outputs in dimension order of [B, <Dimensions (time/spatial)>, C],
    whereas the rest of the ML universe expects [B, C, <Dimensions (time/spatial)>].
    """

    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, y_hat, y) -> torch.Tensor:
        return self.loss_fn(y_hat.view(-1, y_hat.size(-1)), y.view(-1))

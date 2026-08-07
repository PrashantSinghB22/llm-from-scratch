import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """
    One head of masked self-attention.
    """

    def __init__(self, n_embed, head_size, block_size):
        super().__init__()

        # Linear projections
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # Causal mask
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        # Create Q, K and V
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1)

        # Scale
        wei = wei / math.sqrt(k.size(-1))

        # Apply causal mask
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # Convert scores to probabilities
        wei = F.softmax(wei, dim=-1)

        # Weighted sum of values
        out = wei @ v

        return out
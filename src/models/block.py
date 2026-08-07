import torch.nn as nn

from models.attention import MultiHeadAttention
from models.feedforward import FeedForward


class Block(nn.Module):

    def __init__(
        self,
        n_embed,
        num_heads,
        block_size
    ):
        super().__init__()

        head_size = n_embed // num_heads

        self.sa = MultiHeadAttention(
            num_heads,
            head_size,
            n_embed,
            block_size
        )

        self.ffwd = FeedForward(
            n_embed
        )

        self.ln1 = nn.LayerNorm(
            n_embed
        )

        self.ln2 = nn.LayerNorm(
            n_embed
        )

    def forward(self, x):

        x = x + self.sa(
            self.ln1(x)
        )

        x = x + self.ffwd(
            self.ln2(x)
        )

        return x
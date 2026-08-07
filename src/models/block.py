
"""
block.py

Implementation of a Transformer Block.

A Transformer block is the fundamental building block of GPT-style
language models. It combines:

- Multi-Head Self-Attention
- Feed-Forward Network
- Layer Normalization
- Residual Connections

Multiple Transformer blocks are stacked together to build
the complete GPT architecture.
"""

import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention
from src.models.feedforward import FeedForward


class Block(nn.Module):
    """
    A single Transformer block.

    This implementation follows the Pre-LayerNorm architecture:

        LayerNorm
            ↓
        Multi-Head Attention
            ↓
        Residual Connection
            ↓
        LayerNorm
            ↓
        Feed-Forward Network
            ↓
        Residual Connection

    The input and output shapes remain identical, allowing
    multiple blocks to be stacked sequentially.
    """

    def __init__(
        self,
        n_embed: int,
        num_heads: int,
        block_size: int,
    ) -> None:
        """
        Initialize the Transformer block.

        Args:
            n_embed:
                Embedding dimension.

            num_heads:
                Number of attention heads.

            block_size:
                Maximum context length.
        """
        super().__init__()

        # Size of each attention head.
        head_size = n_embed // num_heads

        # Multi-head masked self-attention.
        self.sa = MultiHeadAttention(
            num_heads,
            head_size,
            n_embed,
            block_size,
        )

        # Position-wise feed-forward network.
        self.ffwd = FeedForward(
            n_embed,
        )

        # Layer normalization before attention.
        self.ln1 = nn.LayerNorm(
            n_embed,
        )

        # Layer normalization before feed-forward.
        self.ln2 = nn.LayerNorm(
            n_embed,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute a forward pass through the Transformer block.

        Args:
            x:
                Input tensor of shape
                (batch_size, sequence_length, embedding_dimension).

        Returns:
            Tensor with the same shape as the input.
        """

        # Self-attention with residual connection.
        x = x + self.sa(
            self.ln1(x)
        )

        # Feed-forward network with residual connection.
        x = x + self.ffwd(
            self.ln2(x)
        )

        return x


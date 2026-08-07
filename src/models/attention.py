
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """
    A single masked self-attention head.

    Each token is projected into Query (Q), Key (K), and Value (V)
    representations. Attention scores are computed using the scaled
    dot-product attention mechanism and a causal mask is applied so
    tokens cannot attend to future positions.
    """

    def __init__(
        self,
        n_embed: int,
        head_size: int,
        block_size: int,
    ) -> None:
        """
        Initialize a single attention head.

        Args:
            n_embed: Size of the input embedding.
            head_size: Output dimension for this attention head.
            block_size: Maximum sequence length supported.
        """
        super().__init__()

        # Linear projections for Query, Key, and Value.
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # Lower triangular matrix used for causal masking.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute masked self-attention.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Tensor of shape (B, T, head_size).
        """
        B, T, C = x.shape

        # Compute Query, Key, and Value matrices.
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Compute scaled attention scores.
        wei = q @ k.transpose(-2, -1)
        wei = wei / math.sqrt(k.size(-1))

        # Prevent attending to future tokens.
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # Convert scores into probabilities.
        wei = F.softmax(wei, dim=-1)

        # Compute weighted sum of Value vectors.
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head masked self-attention.

    Multiple attention heads learn different relationships within
    the same input sequence. Their outputs are concatenated and
    projected back to the original embedding dimension.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        n_embed: int,
        block_size: int,
    ) -> None:
        """
        Initialize the multi-head attention module.

        Args:
            num_heads: Number of attention heads.
            head_size: Dimension of each attention head.
            n_embed: Embedding dimension.
            block_size: Maximum context length.
        """
        super().__init__()

        self.heads = nn.ModuleList(
            [
                Head(
                    n_embed,
                    head_size,
                    block_size,
                )
                for _ in range(num_heads)
            ]
        )

        # Project concatenated head outputs back into embedding space.
        self.proj = nn.Linear(
            num_heads * head_size,
            n_embed,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all attention heads in parallel.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Tensor of shape (B, T, C).
        """
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1,
        )

        out = self.proj(out)

        return out


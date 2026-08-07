
"""
feedforward.py

Implementation of the position-wise Feed-Forward Network (FFN)
used inside each Transformer block.

After the attention mechanism gathers contextual information,
the feed-forward network processes each token independently to
learn more complex feature representations.

Architecture:

Input
  │
  ▼
Linear (n_embed → 4 × n_embed)
  │
  ▼
ReLU
  │
  ▼
Linear (4 × n_embed → n_embed)
  │
  ▼
Output
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Every token is passed independently through the same
    two-layer neural network.

    The hidden layer is expanded to four times the embedding
    dimension before being projected back to the original size.
    """

    def __init__(self, n_embed: int) -> None:
        """
        Initialize the feed-forward network.

        Args:
            n_embed: Embedding dimension of the Transformer.
        """
        super().__init__()

        self.net = nn.Sequential(

            # Expand feature dimension.
            nn.Linear(
                n_embed,
                4 * n_embed,
            ),

            # Introduce non-linearity.
            nn.ReLU(),

            # Project back to embedding dimension.
            nn.Linear(
                4 * n_embed,
                n_embed,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the feed-forward network.

        Args:
            x:
                Input tensor of shape
                (batch_size, sequence_length, embedding_dimension).

        Returns:
            Tensor with the same shape as the input.
        """
        return self.net(x)

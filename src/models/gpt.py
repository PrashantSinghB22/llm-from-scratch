"""
gpt.py

Implementation of a GPT-style language model.

The model consists of:

- Token Embeddings
- Positional Embeddings
- Stacked Transformer Blocks
- Final Layer Normalization
- Linear Language Modeling Head

During training the model predicts the next token in a sequence
using Cross Entropy Loss.

During inference the model generates text autoregressively by
sampling one token at a time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.block import Block


class GPTLanguageModel(nn.Module):
    """
    GPT-style autoregressive language model.

    This model predicts the next token given all previous tokens
    in the current context window.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        block_size: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        """
        Initialize the GPT model.

        Args:
            vocab_size:
                Number of unique tokens.

            n_embed:
                Embedding dimension.

            block_size:
                Maximum context length.

            num_heads:
                Number of attention heads.

            num_layers:
                Number of Transformer blocks.
        """
        super().__init__()

        self.block_size = block_size

        # Learnable token embeddings.
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embed,
        )

        # Learnable positional embeddings.
        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embed,
        )

        # Stack multiple Transformer blocks.
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embed,
                    num_heads,
                    block_size,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer normalization.
        self.ln_f = nn.LayerNorm(
            n_embed,
        )

        # Project hidden states to vocabulary logits.
        self.lm_head = nn.Linear(
            n_embed,
            vocab_size,
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        """
        Forward pass through the GPT model.

        Args:
            idx:
                Input token IDs with shape (B, T).

            targets:
                Target token IDs used during training.

        Returns:
            logits:
                Vocabulary predictions.

            loss:
                Cross entropy loss if targets are provided,
                otherwise None.
        """

        B, T = idx.shape

        # Convert token IDs into embedding vectors.
        tok_emb = self.token_embedding_table(idx)

        # Create positional embeddings.
        pos_emb = self.position_embedding_table(
            torch.arange(
                T,
                device=idx.device,
            )
        )

        # Combine token and positional information.
        x = tok_emb + pos_emb

        # Pass through Transformer blocks.
        x = self.blocks(x)

        # Normalize final hidden states.
        x = self.ln_f(x)

        # Predict vocabulary logits.
        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:

            B, T, C = logits.shape

            # Flatten for Cross Entropy Loss.
            logits = logits.view(
                B * T,
                C,
            )

            targets = targets.view(
                B * T,
            )

            loss = F.cross_entropy(
                logits,
                targets,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        At every iteration:

        1. Keep only the latest context window.
        2. Predict the next token.
        3. Sample from the probability distribution.
        4. Append the sampled token.
        5. Repeat.

        Args:
            idx:
                Initial prompt.

            max_new_tokens:
                Number of tokens to generate.

        Returns:
            Generated token sequence.
        """

        for _ in range(max_new_tokens):

            # Keep only the most recent context.
            idx_cond = idx[:, -self.block_size:]

            # Predict next-token logits.
            logits, _ = self(idx_cond)

            # Use only the final time step.
            logits = logits[:, -1, :]

            # Convert logits to probabilities.
            probs = torch.softmax(
                logits,
                dim=-1,
            )

            # Sample the next token.
            idx_next = torch.multinomial(
                probs,
                num_samples=1,
            )

            # Append the generated token.
            idx = torch.cat(
                (idx, idx_next),
                dim=1,
            )

        return idx


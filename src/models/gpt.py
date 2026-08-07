import torch
import torch.nn as nn

class GPTLanguageModel(nn.Module):

  def __init__(
      self,
      vocab_size,
      n_embed,
      block_size,
      num_heads,
      num_layers
  ):
      super().__init__()

      self.token_embedding_table = nn.Embedding(
          vocab_size,
          n_embed
      )

      self.position_embedding_table = nn.Embedding(
          block_size,
          n_embed
      )

      self.blocks = nn.Sequential(
          *[
              Block(
                  n_embed,
                  num_heads,
                  block_size
              )
              for _ in range(num_layers)
          ]
      )

      self.ln_f = nn.LayerNorm(
          n_embed
      )

      self.lm_head = nn.Linear(
          n_embed,
          vocab_size
      )

  def forward(self, idx, targets=None):

    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx)

    pos_emb = self.position_embedding_table(
        torch.arange(T, device=idx.device)
    )

    x = tok_emb + pos_emb

    x = self.blocks(x)

    x = self.ln_f(x)

    logits = self.lm_head(x)

    if targets is None:
        loss = None
    else:

        B, T, C = logits.shape

        logits = logits.view(B*T, C)

        targets = targets.view(B*T)

        loss = F.cross_entropy(
            logits,
            targets
        )

    return logits, loss
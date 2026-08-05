import torch
import torch.nn as nn

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()

    self.token_embedding_table = nn.Embedding(
      num_embeddings=vocab_size,
      embedding_dim=vocab_size
    )

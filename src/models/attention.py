import torch
import torch.nn as nn


class Head(nn.Module):

  def __init__(self, n_embed, head_size):
    super().__init__()

    self.key = nn.Linear(
        n_embed,
        head_size,
        bias=False
    )

    self.query = nn.Linear(
        n_embed,
        head_size,
        bias=False
    )

    self.value = nn.Linear(
        n_embed,
        head_size,
        bias=False
    )

  def forward(self, x):
    k = self.key(x)

    q = self.query(x)

    v = self.value(x)


def create_dataset(encoded_text, block_size):
  inputs = []
  targets = []

  for i in range(len(encoded_text) - block_size):
    x = encoded_text[i: i + block_size]
    y = encoded_text[i + 1: i + block_size + 1]

    inputs.append(x)
    targets.append(y)

  return inputs, targets

import torch

def get_batch(data, block_size, batch_size):

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i+block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i+1:i+block_size+1]
        for i in ix
    ])

    return x, y
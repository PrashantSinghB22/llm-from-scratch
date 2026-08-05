def create_dataset(encoded_text, block_size):
  inputs = []
  targets = []

  for i in range(len(encoded_text) - block_size):
    x = encoded_text[i: i + block_size]
    y = encoded_text[i + 1: i + block_size + 1]

    inputs.append(x)
    targets.append(y)

  return inputs, targets
import torch

from training.dataset import get_batch

data = torch.tensor(
    [5,2,8,1,9,4,7,6]
)

x, y = get_batch(
    data,
    block_size=4,
    batch_size=2
)

print("Inputs")
print(x)

print()

print("Targets")
print(y)

print()

print(x.shape)

print(y.shape)
import torch

from models.bigram import BigramLanguageModel

vocab_size = 65

model = BigramLanguageModel(vocab_size)

idx = torch.tensor([2, 10, 25])

logits = model(idx)

print("Input Shape :", idx.shape)
print("Output Shape:", logits.shape)
print(logits)
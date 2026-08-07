"""
train.py

Train a GPT-style language model from scratch.

This script performs the complete training pipeline:

1. Load the dataset.
2. Build the character-level vocabulary.
3. Encode the text into integer token IDs.
4. Create training and validation datasets.
5. Generate mini-batches.
6. Train the GPT model.
7. Save checkpoints.
8. Generate sample text.

This script is intentionally kept simple for educational purposes.
"""

import os
from pathlib import Path

import torch

from src.models.gpt import GPTLanguageModel

# ============================================================
# Hyperparameters
# ============================================================

batch_size = 64
block_size = 128

max_iters = 5000
eval_interval = 500
eval_iters = 200

learning_rate = 3e-4

n_embed = 128
num_heads = 4
num_layers = 4

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

# ============================================================
# Load Dataset
# ============================================================

DATA_PATH = Path("data") / "wizard_of_oz.txt"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Build the character vocabulary.
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Character-to-index and index-to-character mappings.
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode and decode helper functions.
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Convert the entire dataset into token IDs.
data = torch.tensor(
    encode(text),
    dtype=torch.long,
)

# 90% training / 10% validation split.
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

# ============================================================
# Batch Loader
# ============================================================


def get_batch(split):
    """
    Generate a random mini-batch.

    Args:
        split:
            Either "train" or "val".

    Returns:
        Input and target tensors moved to the selected device.
    """

    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,),
    )

    x = torch.stack(
        [
            data[i:i + block_size]
            for i in ix
        ]
    )

    y = torch.stack(
        [
            data[i + 1:i + block_size + 1]
            for i in ix
        ]
    )

    return x.to(device), y.to(device)


# ============================================================
# Loss Estimation
# ============================================================

@torch.no_grad()
def estimate_loss():
    """
    Estimate average training and validation loss.

    The model is temporarily switched to evaluation mode before
    computing the losses and then restored to training mode.
    """

    out = {}

    model.eval()

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            X, Y = get_batch(split)

            _, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


# ============================================================
# Create Model
# ============================================================

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embed=n_embed,
    block_size=block_size,
    num_heads=num_heads,
    num_layers=num_layers,
)

model = model.to(device)

print(
    f"\nModel Parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}\n"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)

# ============================================================
# Training Loop
# ============================================================

os.makedirs(
    "checkpoints",
    exist_ok=True,
)

for iteration in range(max_iters):

    if iteration % eval_interval == 0:

        losses = estimate_loss()

        print(
            f"Step {iteration:5d} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Val Loss: {losses['val']:.4f}"
        )

        # Save an intermediate checkpoint.
        torch.save(
            model.state_dict(),
            "checkpoints/gpt.pt",
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

# ============================================================
# Save Final Model
# ============================================================

torch.save(
    model.state_dict(),
    "checkpoints/gpt_final.pt",
)

print("\nTraining Finished!\n")

# ============================================================
# Generate Sample Text
# ============================================================

context = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device,
)

generated = model.generate(
    context,
    max_new_tokens=500,
)

print("\nGenerated Text:\n")

print(
    decode(
        generated[0].tolist()
    )
)


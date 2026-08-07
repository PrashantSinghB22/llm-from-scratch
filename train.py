import os
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

from pathlib import Path

DATA_PATH = Path("data") / "wizard_of_oz.txt"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

# ============================================================
# Batch Loader
# ============================================================

def get_batch(split):

    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)

# ============================================================
# Loss Estimation
# ============================================================

@torch.no_grad()
def estimate_loss():

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

print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)

# ============================================================
# Training Loop
# ============================================================

os.makedirs("checkpoints", exist_ok=True)

for iteration in range(max_iters):

    if iteration % eval_interval == 0:

        losses = estimate_loss()

        print(
            f"Step {iteration:5d} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Val Loss: {losses['val']:.4f}"
        )

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
# Final Save
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

print(decode(generated[0].tolist()))
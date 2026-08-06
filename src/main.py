model = BigramLanguageModel(vocab_size)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)

for step in range(1000):

    optimizer.zero_grad()

    logits = model(inputs)

    # loss = ...

    # loss.backward()

    # optimizer.step()
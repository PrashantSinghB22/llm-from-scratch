optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)

for step in range(5000):

    x, y = get_batch(
        data,
        block_size,
        batch_size
    )

    logits, loss = model(
        x,
        y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 500 == 0:
        print(step, loss.item())
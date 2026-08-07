# Training

## Objective

The model is trained to predict the next token in a sequence.

Example:

Input:

```text
The cat sat
```

Target:

```text
on
```

The prediction error is measured using Cross Entropy Loss.

---

## Optimization

The project uses the AdamW optimizer.

During each training iteration:

1. Sample a mini-batch.
2. Perform a forward pass.
3. Compute the loss.
4. Compute gradients using backpropagation.
5. Update model parameters.

This process is repeated for many iterations until the model learns meaningful language patterns.

---

## Text Generation

After training, the model generates text autoregressively.

Each newly generated token becomes part of the input for predicting the next token.

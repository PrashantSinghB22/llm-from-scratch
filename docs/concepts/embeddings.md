# Embeddings

## What are Embeddings?

Embeddings convert token IDs into dense vectors of real numbers.

Instead of representing tokens as integers, the model learns a vector for every token.

Example:

```text
Token ID

15

↓

Embedding

[0.21, -0.44, 1.08, ...]
```

These vectors are learned during training.

---

## Why are Embeddings Important?

Integers have no semantic meaning.

For example:

```text
Dog = 5
Cat = 6
Tree = 20
```

These numbers are merely identifiers.

Embeddings allow the model to learn relationships between tokens in a continuous vector space.

---

## Positional Embeddings

Transformers process tokens in parallel.

Without positional information, the sentences

"The cat chased the dog."

and

"The dog chased the cat."

would appear identical.

Positional embeddings solve this problem by encoding token order.

# Multi-Head Attention

## Why Multiple Heads?

A single attention head can only learn one type of relationship at a time.

Multiple heads allow the model to focus on different aspects of the input simultaneously.

Examples include:

* Grammar
* Long-range dependencies
* Punctuation
* Sentence structure

---

## Workflow

Each head performs self-attention independently.

The outputs are concatenated and projected back into the embedding dimension.

This enables the model to capture richer contextual information than a single attention head.

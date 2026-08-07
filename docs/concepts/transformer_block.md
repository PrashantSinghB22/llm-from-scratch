# Transformer Block

The Transformer Block is the fundamental building block of GPT.

Each block contains:

1. Layer Normalization
2. Multi-Head Self-Attention
3. Residual Connection
4. Layer Normalization
5. Feed-Forward Network
6. Residual Connection

Architecture:

```text
Input
 │
 ▼
LayerNorm
 │
 ▼
Multi-Head Attention
 │
 ▼
Residual Add
 │
 ▼
LayerNorm
 │
 ▼
Feed Forward
 │
 ▼
Residual Add
 │
 ▼
Output
```

Transformer blocks can be stacked because they preserve the input tensor shape.

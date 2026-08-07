# Self-Attention

## Motivation

Language understanding depends on context.

Consider the sentence:

```text
The animal didn't cross the street because it was tired.
```

The meaning of "it" depends on earlier words.

Self-attention allows every token to decide which previous tokens are important.

---

## Query, Key and Value

Every token is projected into three vectors:

* Query (Q)
* Key (K)
* Value (V)

The attention score is computed by comparing queries with keys.

The resulting weights are applied to the value vectors.

---

## Causal Masking

GPT predicts text one token at a time.

During training, future tokens must remain hidden.

A causal mask ensures that each token can only attend to itself and previous tokens.

This prevents information leakage during training.

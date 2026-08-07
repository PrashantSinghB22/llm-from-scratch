# Tokenization

## What is Tokenization?

Tokenization is the process of converting raw text into smaller pieces called **tokens**.

A language model cannot understand text directly. It only understands numbers, so every token must first be converted into an integer.

Example:

```text
Hello

↓

['H', 'e', 'l', 'l', 'o']

↓

[12, 4, 19, 19, 7]
```

In this project, a **character-level tokenizer** is used.

Each unique character in the dataset is assigned a unique integer ID.

---

## Why is Tokenization Necessary?

Neural networks operate on numerical data.

Tokenization provides a mapping between human-readable text and numerical representations that can later be converted into embeddings.

---

## Advantages

* Simple to implement
* Easy to understand
* Small vocabulary
* Ideal for learning

---

## Limitations

* Long sequences
* Slow generation
* Does not capture word-level semantics

Modern LLMs typically use subword tokenization techniques such as Byte Pair Encoding (BPE) or SentencePiece.

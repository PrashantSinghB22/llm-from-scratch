# 🧠 LLM From Scratch (PyTorch)

A GPT-style Language Model implemented from scratch in **PyTorch** to understand how Large Language Models (LLMs) work internally.

> This project was built as a learning journey to understand every major component of a GPT-style Transformer architecture, starting from tokenization and ending with autoregressive text generation.

---

# 📖 Overview

Large Language Models like GPT are built from several fundamental ideas:

* Tokenization
* Embeddings
* Self-Attention
* Multi-Head Attention
* Feed-Forward Networks
* Residual Connections
* Layer Normalization
* Transformer Blocks
* Autoregressive Training

Instead of relying on high-level libraries, this project implements these components manually using PyTorch to understand how they work together.

The goal of this repository is **education**, not production performance.

---

# ✨ Features

* Character-level tokenizer
* Token embedding layer
* Positional embedding layer
* Masked self-attention
* Multi-head self-attention
* Feed-forward network
* Layer normalization
* Residual connections
* Transformer blocks
* GPT-style language model
* Cross-entropy training
* Autoregressive text generation

---

# 🏗️ Project Structure

```text
llm-from-scratch/

├── data/
│   └── wizard_of_oz.txt
│
├── docs/
│   ├── concepts/
│   ├── diagrams/
│   ├── screenshots/
│   └── results.md
│
├── notebooks/
│
├── src/
│   └── models/
│       ├── attention.py
│       ├── block.py
│       ├── feedforward.py
│       └── gpt.py
│
├── train.py
├── requirements.txt
├── README.md
├── LICENSE
└── CHANGELOG.md
```

---

# 🧩 Model Architecture

The model follows a simplified GPT architecture.

```text
Input Text
      │
      ▼
Character Tokenizer
      │
      ▼
Token IDs
      │
      ▼
Token Embeddings
      │
      ▼
Positional Embeddings
      │
      ▼
Transformer Blocks
      │
      ▼
LayerNorm
      │
      ▼
Linear Head
      │
      ▼
Vocabulary Logits
      │
      ▼
Next Token Prediction
```

---

# 🧠 Components Implemented

## 1. Character Tokenizer

Converts raw text into integer token IDs.

Example:

```
"hello"

↓

[7, 4, 11, 11, 14]
```

---

## 2. Token Embeddings

Each token ID is mapped to a dense vector representation that the model can learn during training.

---

## 3. Positional Embeddings

Since Transformers process tokens in parallel, positional embeddings provide information about token order.

---

## 4. Self-Attention

Allows every token to decide which previous tokens are important for understanding the current token.

The implementation uses causal masking so tokens cannot attend to future tokens during training.

---

## 5. Multi-Head Attention

Multiple attention heads learn different relationships within the same sequence.

---

## 6. Feed-Forward Network

A small neural network applied independently to every token after attention.

---

## 7. Transformer Block

Each Transformer block contains:

* LayerNorm
* Multi-Head Attention
* Residual Connection
* LayerNorm
* Feed-Forward Network
* Residual Connection

Multiple blocks are stacked to build the complete GPT model.

---

# 📚 Dataset

The model is trained on:

* **Dataset:** Wizard of Oz (character-level)
* **Tokenizer:** Character-based

The repository can easily be adapted to use other plain-text datasets.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>

cd llm-from-scratch
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

Start training using:

```bash
python train.py
```

During training the script periodically reports:

* Training Loss
* Validation Loss

Model checkpoints can be saved for later inference.

---

# ✍️ Text Generation

After training, the model generates text by repeatedly predicting the next token and feeding it back into itself.

This process is called **autoregressive generation**.

---

# 📈 Results

Example metrics and generated samples can be found in:

```
docs/results.md
```

---

# 📚 Concepts Covered

This project helped me understand:

* Tokenization
* Embeddings
* Context Windows
* Self-Attention
* Multi-Head Attention
* Layer Normalization
* Residual Connections
* Transformer Blocks
* GPT Architecture
* Cross Entropy Loss
* AdamW Optimizer
* Autoregressive Text Generation

---

# 🚀 Future Improvements

Planned improvements for Version 2:

* Byte Pair Encoding (BPE)
* Dropout
* GELU activation
* Learning rate scheduling
* Mixed precision training
* Better sampling (temperature, top-k, top-p)
* Configuration system
* Unit tests
* Attention visualization

---

# 📚 References

* Attention Is All You Need (Vaswani et al.)
* PyTorch Documentation
* GPT Architecture Papers

---

# 👨‍💻 Author

Built as part of a personal learning journey to understand Large Language Models from first principles using PyTorch.

---

# ⭐ Acknowledgements

This project was created for educational purposes and focuses on understanding how GPT-style language models are implemented rather than reproducing production-scale systems.

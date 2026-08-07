# Changelog

All notable changes to this project will be documented in this file.

The format loosely follows **Keep a Changelog**, and this project uses semantic versioning.

---

## [1.0.0] - 2026-08-07

### 🎉 Initial Release

This is the first complete implementation of a GPT-style language model built from scratch using PyTorch.

### Added

* Character-level tokenizer
* Vocabulary creation from raw text
* Encode and decode functions
* Dataset preprocessing
* Train/validation split
* Batch generation
* Token embedding layer
* Positional embedding layer
* Single-head self-attention
* Multi-head self-attention
* Causal masking
* Feed-forward neural network
* Layer normalization
* Residual connections
* Transformer block
* GPT language model
* Cross-entropy loss
* AdamW optimizer
* Autoregressive text generation
* Model checkpoint saving
* Educational Jupyter notebooks
* Comprehensive README
* MIT License

### Fixed

* Corrected dataset loading path using the `data/` directory.
* Standardized imports across the `src` package.
* Added missing PyTorch functional API imports.
* Verified the complete training pipeline.

### Documentation

* Added project overview.
* Documented repository structure.
* Added architecture overview.
* Prepared repository for GitHub release.

---

## Future Releases

Planned improvements for Version 2 include:

* Byte Pair Encoding (BPE)
* GELU activation
* Dropout
* Learning rate scheduler
* Improved sampling (temperature, top-k, top-p)
* Configuration system
* Unit tests
* Attention visualization
* Mixed precision training
* Better modular architecture

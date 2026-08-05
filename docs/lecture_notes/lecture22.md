# Lecture 22 - Bigram Language Model

## Key Ideas

- A Bigram model predicts the next token using only the current token.
- It can be represented as a trainable lookup table.
- Each row corresponds to one current token.
- Each column corresponds to one possible next token.
- The output of the lookup table is logits.
- Softmax converts logits into probabilities.
- Cross Entropy Loss measures prediction quality.
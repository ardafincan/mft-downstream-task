# Training & Setup Report

## Overview

The training process (`train.py`, `embedding_trainer.py`) implements a knowledge distillation setup where student models (initialized with either MFT or Tabi tokenizers) learn to mimic the embedding space of a teacher model using Cosine Embedding Loss.

**Key Files:** `train.py`, `embedding_trainer.py`, `TRAINING_DETAILS.md`.

## Experimental Setup

### Models Compared

| Model Name           | Backbone         | Tokenizer    |
| :------------------- | :--------------- | :----------- |
| **mft-random-init**  | (Random Weights) | MFT (Hybrid) |
| **tabi-random-init** | (Random Weights) | Tabi (BPE)   |

_Note: `random-init` baselines serve as a sanity check to ensure the distillation process is actually learning better representations than chance/architecture priors._

### Training Strategy: Two-Phase

To ensure stability, especially for custom initialized models:

1.  **Warmup Phase:** 100 steps, linear warmup. Stabilizes gradients.
2.  **Full Training Phase:** 1 Epoch over the entire corpus (`alibayram/cosmos-corpus-encoded`).

### Hyperparameters

- **Loss:** `CosineEmbeddingLoss` (Focuses on vector direction/semantic similarity).
- **Batch Size:** 256.
- **Learning Rate:** 5e-5.
- **Precision:** bfloat16 (BF16).
- **Hardware:** NVIDIA H100 80GB.

## Implementation Details

- **Tokenizer Bypass:** Since `sentence-transformers` expects standard HuggingFace tokenizers, we patched `Transformer.py` to accept pre-computed `input_ids`. This allows training models with custom python-based tokenizers (MFT) without rewriting the entire trainer loop.
- **Pooling:** Mean pooling of token embeddings (excluding padding).

## Paper Integration Points

- **Experimental Setup Section:**
  - List hyperparameters (BS=256, LR=5e-5, Loss=Cosine).
  - Describe the Student-Teacher distillation framework.
  - Mention the "Random Init" baseline as a control variable.
  - Briefly describe the "Tokenizer Bypass" if discussing reproducibility (or in Appendix).

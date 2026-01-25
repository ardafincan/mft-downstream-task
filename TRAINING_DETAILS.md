# Academic Experiment Protocol: MFT Embedding Distillation

**Date:** January 25, 2026 17:00  
**Experiment:** Downstream Task - Embedding Distillation  
**Compute Node:** Verda-H100

## 1. Abstract

This experiment evaluates the efficacy of a specialized **Morphological Tokenizer (MFT)** versus a standard **Subword Tokenizer (Tabi/BERT)** in a knowledge distillation setup. We aim to determine if linguistically aware tokenization improves semantic embedding quality for agglutinative languages (Turkish) when distilled from a high-quality teacher model.

## 2. Infrastructure Specification

| Component              | Specification                                  |
| :--------------------- | :--------------------------------------------- |
| **GPU Accelerator**    | NVIDIA H100 80GB HBM3 (PCIe)                   |
| **VRAM Utilization**   | ~79GB (Peak during training at batch size 256) |
| **Compute Capability** | 9.0 (Hopper Architecture)                      |
| **CUDA Version**       | 13.0                                           |
| **Driver Version**     | 580.95.05                                      |
| **Precision**          | BFloat16 (BF16) on `torch>=2.0.0`              |

## 3. Dataset Configuration

**Source:** `alibayram/cosmos-corpus-encoded`

The dataset underwent rigorous pre-processing to eliminate runtime tokenization overhead and ensure sequence length compliance.

- **Sequence Limit:** `MAX_SEQ_LENGTH = 2048` tokens.
- **Filtering Protocol:** Samples where **either** MFT or Tabi encoded length exceeded 2048 were discarded.
- **Columns:**
  - `mft_input_ids`: Pre-computed integer sequences (MFT Vocab).
  - `tabi_input_ids`: Pre-computed integer sequences (TabiBERT Vocab).
  - `teacher_embedding_final`: Float vectors from the Teacher model.

## 4. Model Architectures

We evaluate 6 distinct student configurations (3 Architecture pairings $\times$ 2 Tokenizers):

### Group A: MFT Tokenizer (Custom Morphological)

_Utilizes a hybrid vocabulary of Roots (`kokler.json`), Suffixes (`ekler.json`), and BPE tokens._

1.  **`mft-embeddinggemma`**: embeddinggemma-300m backbone.
2.  **`mft-embeddingmagibu`**: embeddingmagibu-200m backbone.
3.  **`mft-random-init`**: Randomly initialized baseline.

### Group B: Tabi Tokenizer (Standard Subword)

_Utilizes a standard 32k vocabulary from `alibayram/TabiBERT-tokenizer-32k`._

1. **`tabi-embeddinggemma`**: embeddinggemma-300m backbone.
2. **`tabi-embeddingmagibu`**: embeddingmagibu-200m backbone.
3. **`tabi-random-init`**: Randomly initialized baseline.

## 5. Methodological Framework

### 5.1 Distillation Objective

We utilize **Cosine Embedding Loss** to maximize the semantic alignment between student and teacher vectors.

$$ \mathcal{L}_{cosine} = 1 - \cos(v_{student}, v\_{teacher}) $$

Where vectors are $\ell_2$-normalized. This focuses purely on semantic directionality rather than magnitude.

### 5.2 Model Pipeline Structure

Each student model pipeline consists of two primary modules:

1.  **Transformer**: The base encoder (Gemma/BERT) outputs contextualized token embeddings $H \in \mathbb{R}^{L \times d}$ where $L$ is sequence length and $d$ is hidden dimension.
2.  **Pooling**: We employ **Mean Pooling** to derive a fixed-size sentence representation $u \in \mathbb{R}^d$:
    $$ u = \frac{1}{L} \sum\_{i=1}^{L} H*i $$
    \_Note: Padding tokens are excluded from the mean calculation via the attention mask.*

### 5.3 Novel Tokenization Bypass Strategy

To enable training of architectures with incompatible custom tokenizers (e.g., Gemma + MFT) using the `sentence-transformers` library, we implemented a **Tokenizer Bypass**:

1.  **Offline Encoding:** `input_ids` are pre-generated during dataset preparation.
2.  **Runtime Patch:** The `Transformer.py` module in `sentence-transformers` was patched to make tokenizer initialization optional (`try-except` block). This allows the training loop to load the model architecture without crashing on tokenizer validation errors, consuming the pre-encoded inputs directly and respecting the `trust_remote_code=True` flag required for custom MFT modeling code.

## 6. Training Hyperparameters

| Parameter                  | Configuration       | Justification / Detail                                |
| :------------------------- | :------------------ | :---------------------------------------------------- |
| **Batch Size**             | 256                 | Optimized for H100 80GB saturation (~80GB VRAM).      |
| **Epochs**                 | 1                   | Single-pass distillation sufficient for large corpus. |
| **Learning Rate**          | 5e-5                | Standard Transformer fine-tuning rate.                |
| **Optimizer**              | AdamW               | `torch.optim.AdamW` (betas=(0.9, 0.999), eps=1e-8).   |
| **Scheduler**              | Linear              | Linear decay with constant warmup (`ratio=0.01`).     |
| **Loss Function**          | CosineEmbeddingLoss | $1 - \cos(u, v)$.                                     |
| **Max Grad Norm**          | 1.0                 | Gradient clipping for stability.                      |
| **Random Seed**            | Random              | System entropy (unfixed).                             |
| **Context Length**         | 2048                | Defined in `prepare_dataset.py`.                      |
| **BF16**                   | True                | Enabled for training stability and memory efficiency. |
| **Gradient Checkpointing** | True                | Enabled to fit Batch Size 256.                        |

## 7. Stability & Monitoring

- **Memory Management:** Explicit `gc.collect()` and `torch.cuda.empty_cache()` are triggered between model training sessions to prevent fragmentation-induced OOM.
- **Logging:**
  - **Step 1-20:** Logged every step (Detailed initial convergence tracking).
  - **Step 20+:** Logged every 20 steps.
  - **Checkpointing:** Every 200 steps.

## 8. Reproducibility

To reproduce this experiment:

1.  **Dependencies**:
    ```text
    torch>=2.0.0
    transformers>=4.38.0
    sentence-transformers (Local Build)
    ```
2.  **Execution**:
    ```bash
    python train.py
    ```
3.  **Note**: Ensure `sentence_transformers/models/Transformer.py` contains the custom patch for robust tokenizer loading.

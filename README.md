# Turkish Tokenizer & Morphologically-Aware Embeddings

A morphologically-aware Turkish tokenizer designed for NLP tasks, featuring:

- **Root + Suffix decomposition** based on Turkish morphology
- **Vowel harmony** aware decoding
- **Integration with SentenceTransformers** for semantic embeddings
- **32K vocabulary** optimized for Turkish text

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using with SentenceTransformers](#using-with-sentencetransformers)
- [Vocabulary Structure](#vocabulary-structure)
- [Files Description](#files-description)
- [API Reference](#api-reference)
- [Training Your Own Model](#training-your-own-model)
- [License](#license)

## Overview

Traditional BPE tokenizers split Turkish words arbitrarily, losing morphological information. This tokenizer uses a linguistically-informed approach:

```
Traditional BPE: "evlerinden" → ["evl", "er", "ind", "en"]
This Tokenizer:  "evlerinden" → ["ev", "ler", "in", "den"]
                                  (root) (plural) (possessive) (ablative)
```

This preserves the morphological structure, enabling better downstream NLP performance for Turkish.

## Features

### 🔤 Morphologically-Aware Tokenization

- Decomposes words into **roots** (kökler) and **suffixes** (ekler)
- Handles Turkish-specific morphophonemic changes (consonant softening, vowel harmony)
- Supports BPE fallback for unknown words and foreign text

### 🔄 Intelligent Decoding

- Applies **vowel harmony** rules during decoding
- Handles consonant mutations (k→ğ, p→b, t→d, ç→c)
- Correctly reconstructs Turkish text from token IDs

### 🤖 SentenceTransformers Integration

- Modified `sentence_transformers` library to support custom tokenizers
- Create semantic embeddings with morphologically-aware tokenization

## Installation

### Prerequisites

```bash
pip install torch transformers sentence-transformers python-dotenv
```

### Clone the Repository

```bash
git clone <repository-url>
cd tr-tokenizer-train
```

### Environment Setup

Create a `.env` file with your Hugging Face token:

```
HF_TOKEN=your_huggingface_token_here
```

## Quick Start

### Basic Tokenization

```python
import turkish_tokenizer as tt

# Initialize tokenizer
tokenizer = tt.TurkishTokenizer()

# Tokenize text
text = "Türkiye'nin en güzel şehirlerinden biridir"
tokens = tokenizer.tokenize(text)
print(tokens)  # [' türkiye', 'nin', ' en', ' güzel', ' şehir', 'ler', 'in', 'den', ' bir', 'i', 'dir']

# Encode to IDs
ids = tokenizer.encode(text)
print(ids)  # [token_id_1, token_id_2, ...]

# Decode back to text
decoded = tokenizer.decode(ids)
print(decoded)  # "Türkiye'nin en güzel şehirlerinden biridir"
```

### Callable Interface

```python
# The tokenizer is callable and returns a dict compatible with transformers
result = tokenizer("Merhaba dünya")
print(result)
# {'input_ids': [0, 4103, 2608], 'attention_mask': [1, 1, 1]}
```

## Using with SentenceTransformers

> ⚠️ **Important**: To use this tokenizer with SentenceTransformers, you must use the **modified `sentence_transformers` library** included in this repository. The standard library does not support custom tokenizers.

### Why Modified SentenceTransformers?

The model [`alibayram/mft-downstream-task-embeddinggemma`](https://huggingface.co/alibayram/mft-downstream-task-embeddinggemma) was trained with this custom Turkish tokenizer. Since the tokenizer is not a standard Hugging Face tokenizer (no `tokenizer.json` file), you need to:

1. Use the modified `sentence_transformers` library from this repository
2. Pass the `TurkishTokenizer` instance via the `custom_tokenizer` parameter

### Installation

```bash
# Clone this repository - it includes the modified sentence_transformers
git clone <repository-url>
cd tr-tokenizer-train

# The sentence_transformers folder contains the modified library
# It will be used automatically when you run scripts from this directory
```

### Usage Example

```python
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import turkish_tokenizer as tt

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# Initialize the Turkish tokenizer
tokenizer = tt.TurkishTokenizer()

# Load the model with custom tokenizer
model = SentenceTransformer(
    "alibayram/mft-downstream-task-embeddinggemma",
    custom_tokenizer=tokenizer,
    token=HF_TOKEN
)

# Encode sentences
sentences = [
    "Bu mutlu bir insan",
    "Bu mutlu bir köpek",
    "Bugün güneşli bir gün"
]

embeddings = model.encode(sentences)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 768)

# Calculate similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
```

### What Was Modified?

The following changes were made to enable custom tokenizer support:

1. **`sentence_transformers/SentenceTransformer.py`**:
   - Added `custom_tokenizer` parameter to `__init__`
   - After loading the model, replaces the Transformer module's tokenizer with the custom one

2. **`sentence_transformers/models/Transformer.py`**:
   - Added `custom_tokenizer` parameter to `__init__`
   - Modified `tokenize()` method to handle custom tokenizers that don't support all HuggingFace kwargs (like `padding`, `truncation`, `return_tensors`)

## Vocabulary Structure

The tokenizer uses a 32,768 token vocabulary divided into three categories:

### 1. Roots (kökler) - IDs 0-19,999

- Special tokens (0-99): `<uppercase>`, `<unknown>`, `<pad>`, `<eos>`, etc.
- Word roots with morphophonemic variants
- Example: `" ev"` (house), `" git"/"gid"` (go)

### 2. Suffixes (ekler) - IDs 20,000-20,071

Turkish grammatical suffixes with all vowel harmony variants:

- `lar/ler` (plural)
- `da/de/ta/te` (locative)
- `dan/den/tan/ten` (ablative)
- `lık/lik/luk/lük` (noun forming)
- And many more...

### 3. BPE Tokens - IDs 20,072-32,767

Fallback BPE tokens for:

- Foreign words
- Unknown words
- Subword units not in the root vocabulary

## Files Description

| File                           | Description                                           |
| ------------------------------ | ----------------------------------------------------- |
| `turkish_tokenizer.py`         | Main tokenizer class with encode/decode methods       |
| `turkish_decoder.py`           | Morphology-aware decoder with vowel harmony rules     |
| `kokler.json`                  | Root words dictionary (~20K tokens)                   |
| `ekler.json`                   | Turkish suffixes (~72 suffix groups)                  |
| `bpe_tokenler.json`            | Fallback BPE tokens (~12K tokens)                     |
| `mft_embeddinggemma_cloner.py` | Script to clone EmbeddingGemma with Turkish tokenizer |
| `test_custom_tokenizer.py`     | Test script for SentenceTransformer integration       |
| `sentence_transformers/`       | Modified library with custom tokenizer support        |

## API Reference

### TurkishTokenizer

```python
class TurkishTokenizer:
    vocab_size: int  # 32768
    pad_token: str   # "<pad>"
    eos_token: str   # "<eos>"
    pad_token_id: int
    eos_token_id: int

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""

    def tokenize(self, text: str) -> List[str]:
        """Convert text to token strings."""

    def get_vocab(self) -> Dict[str, int]:
        """Get the complete vocabulary dictionary."""

    def __call__(self, text: str) -> Dict[str, List[int]]:
        """Callable interface returning input_ids and attention_mask."""
```

## Training Your Own Model

### Clone an Existing Model with Turkish Tokenizer

Use `mft_embeddinggemma_cloner.py` to create a model with the Turkish tokenizer:

```bash
python mft_embeddinggemma_cloner.py
```

This script:

1. Loads the original EmbeddingGemma model
2. Maps tokens from the original vocabulary to Turkish vocabulary
3. Initializes embeddings using mean pooling of source tokens
4. Saves and uploads to Hugging Face Hub

### How Embedding Initialization Works

For each token in the Turkish vocabulary:

1. If it exists in the source vocabulary → use the original embedding
2. If not → tokenize with source tokenizer and average the embeddings

```python
# Example: "merhaba" not in source vocab
# Tokenized by source: ["mer", "ha", "ba"]
# New embedding = mean(embed["mer"], embed["ha"], embed["ba"])
```

## Vowel Harmony in Decoding

The decoder applies Turkish vowel harmony rules:

| Previous Vowel | Suffix Variants          |
| -------------- | ------------------------ |
| a, ı           | lar, da, dan, dı, lık... |
| e, i           | ler, de, den, di, lik... |
| o, u           | lar, da, dan, du, luk... |
| ö, ü           | ler, de, den, dü, lük... |

### Consonant Softening

Hard consonants (p, ç, t, k) soften before vowel-initial suffixes:

- kitap → kitab-ı (book-ACC)
- ağaç → ağac-a (tree-DAT)
- kurt → kurd-u (wolf-ACC)
- çocuk → çocuğ-un (child-GEN)

## Examples

### Similarity Search

```python
# Find similar Turkish sentences
query = "Yapay zeka gelecekte önemli olacak"
documents = [
    "Makine öğrenmesi geleceğin teknolojisidir",
    "Bugün hava çok güzel",
    "Türkiye'de turizm sektörü büyüyor"
]

query_embedding = model.encode(query)
doc_embeddings = model.encode(documents)

similarities = model.similarity(query_embedding, doc_embeddings)
# Returns highest similarity for the AI/ML related sentence
```

## License

This project is open source. See LICENSE file for details.

## Acknowledgments

- Based on Google's [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m)
- Uses [SentenceTransformers](https://www.sbert.net/) library (modified version)
- Turkish morphology rules based on Turkish language grammar

---

**Note**: For questions or issues, please open a GitHub issue.

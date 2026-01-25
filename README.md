# MFT vs BPE Tokenizer Comparison for Turkish Embeddings

This repository compares **Morphologically-aware Turkish tokenization (MFT)** with **standard BPE tokenization** for sentence embedding models.

## 🎯 Project Goal

Investigate whether morphologically-aware tokenization improves Turkish sentence embeddings compared to standard BPE tokenization. We evaluate this by:

1. Creating 6 different base models (4 cloned + 2 random initialization)
2. Training each on the same data
3. Evaluating on Turkish STS benchmark at each checkpoint

## 📊 Experiment Design

### Tokenizers Compared

| Tokenizer    | Type          | Vocab Size | Description                                       |
| ------------ | ------------- | ---------- | ------------------------------------------------- |
| **MFT**      | Morphological | 32K        | Root + suffix decomposition with vowel harmony    |
| **TabiBERT** | BPE           | 32K        | Standard BPE trained on Turkish (pruned from 52K) |

### 6 Model Variants

All models use the same architecture with **131M parameters** (131,420,928 params with 32K vocab).

| #   | Model                | HuggingFace ID                                   | Tokenizer | Initialization              | Parameters |
| --- | -------------------- | ------------------------------------------------ | --------- | --------------------------- | ---------- |
| 1   | mft-embeddinggemma   | `alibayram/mft-downstream-task-embeddinggemma`   | MFT       | Cloned from EmbeddingGemma  | 131M       |
| 2   | mft-embeddingmagibu  | `alibayram/mft-downstream-task-embeddingmagibu`  | MFT       | Cloned from EmbeddingMagibu | 131M       |
| 3   | tabi-embeddinggemma  | `alibayram/tabi-downstream-task-embeddinggemma`  | TabiBERT  | Cloned from EmbeddingGemma  | 131M       |
| 4   | tabi-embeddingmagibu | `alibayram/tabi-downstream-task-embeddingmagibu` | TabiBERT  | Cloned from EmbeddingMagibu | 131M       |
| 5   | mft-random-init      | `alibayram/mft-random-init`                      | MFT       | Random Xavier init          | 131M       |
| 6   | tabi-random-init     | `alibayram/tabi-random-init`                     | TabiBERT  | Random Xavier init          | 131M       |

> **Note:** Both EmbeddingGemma (originally 262K vocab → 300M params) and EmbeddingMagibu (originally 131K vocab → 200M params) are resized to 32K vocab, resulting in identical parameter counts. The extra ~5M params come from Dense projection layers in the SentenceTransformer.

## 🔤 MFT Tokenizer

Traditional BPE tokenizers split Turkish words arbitrarily, losing morphological information:

```
BPE (Theoretical): "evlerinden" → ["evl", "er", "ind", "en"]
BPE (TabiBERT): "evlerinden" → ["ev", "lerinden"]
MFT Tokenizer:  "evlerinden" → ["ev", "ler", "in", "den"]
                                 (root) (plural) (poss.) (ablative)

# another example: "Yapay zeka ve makine öğrenmesi"
BPE (Theoretical) → ["yapay", "zeka", "ve", "makine", "öğrenmesi"]
BPE (TabiBERT) → ['Yap', 'ay', ' zek', 'a ve ', 'makine ', 'öğren', 'mesi']
MFT Tokenizer → ['<uppercase>', ' yapay', ' zeka', ' ve', ' makine', ' öğren', 'me', 'si']
```

MFT preserves morphological structure:

- **Roots** (kökler): Word stems with consonant softening variants
- **Suffixes** (ekler): 72 suffix groups with vowel harmony
- **BPE fallback**: For unknown words and foreign text

## 📈 Evaluation

Evaluation uses the Turkish STS benchmark (`figenfikri/stsb_tr`):

```bash
# Evaluate a single model
python evaluate_sts_tr.py --model "alibayram/mft-downstream-task-embeddinggemma"

# Compare multiple models
python evaluate_sts_tr.py --model "model1" "model2" "model3"
```

Results are saved to `sts_benchmark_results.json`.

## ⚠️ Important: Using MFT Models

Models with MFT tokenizer **do not include tokenizer.json** because the tokenizer is morphology-based (not BPE). You must use the modified `sentence_transformers` library included in this repo:

```python
from sentence_transformers import SentenceTransformer
import turkish_tokenizer as tt

# Initialize custom tokenizer
tokenizer = tt.TurkishTokenizer()

# Load model with custom tokenizer
model = SentenceTransformer(
    "alibayram/mft-downstream-task-embeddinggemma",
    custom_tokenizer=tokenizer
)

# Encode sentences
embeddings = model.encode(["Merhaba dünya", "Türkiye güzel bir ülke"])
```

## 📁 Repository Structure

```
tr-tokenizer-train/
├── turkish_tokenizer.py      # MFT tokenizer implementation
├── turkish_decoder.py        # Vowel harmony aware decoder
├── kokler.json               # Turkish roots (~20K)
├── ekler.json                # Turkish suffixes (72 groups)
├── bpe_tokenler.json         # BPE fallback tokens
├── evaluate_sts_tr.py        # STS benchmark evaluation
│
├── mft_embeddinggemma_cloner.py      # Cloner scripts
├── mft_embeddingmagibu_cloner.py
├── tabi_embeddinggemma_cloner.py
├── tabi_embeddingmagibu_cloner.py
├── random_init.py                    # Random init (both tokenizers, seed=42)
│
└── sentence_transformers/    # Modified library with custom_tokenizer support
```

## 🚀 Quick Start

### Setup

```bash
git clone <repository-url>
cd tr-tokenizer-train
pip install torch transformers sentence-transformers python-dotenv datasets scipy

# Create .env with HuggingFace token
echo "HF_TOKEN=your_token_here" > .env
```

### Generate Base Models

```bash
# Cloned embeddings (mean pooling from source model)
python mft_embeddinggemma_cloner.py
python mft_embeddingmagibu_cloner.py
python tabi_embeddinggemma_cloner.py
python tabi_embeddingmagibu_cloner.py

# Random initialization baselines (same seed=42 for both)
python random_init.py
```

### Evaluate

```bash
python evaluate_sts_tr.py -m "alibayram/mft-downstream-task-embeddinggemma" "alibayram/mft-downstream-task-embeddingmagibu" "alibayram/tabi-downstream-task-embeddinggemma" "alibayram/tabi-downstream-task-embeddingmagibu" "alibayram/mft-random-init" "alibayram/tabi-random-init"
```

## 📊 Results

### Baseline Results (Before Training)

Evaluation on Turkish STS benchmark (`figenfikri/stsb_tr`) using pre-training embeddings:

| Model                                            | Tokenizer | Init Source      | Pearson    | Spearman   |
| ------------------------------------------------ | --------- | ---------------- | ---------- | ---------- |
| `alibayram/mft-downstream-task-embeddingmagibu`  | **MFT**   | EmbeddingMagibu  | **0.6107** | **0.5965** |
| `alibayram/mft-random-init`                      | **MFT**   | Random (seed=42) | 0.4709     | 0.4596     |
| `alibayram/mft-downstream-task-embeddinggemma`   | **MFT**   | EmbeddingGemma   | 0.4497     | 0.4382     |
| `alibayram/tabi-downstream-task-embeddingmagibu` | TabiBERT  | EmbeddingMagibu  | 0.4246     | 0.4172     |
| `alibayram/tabi-random-init`                     | TabiBERT  | Random (seed=42) | 0.4053     | 0.3860     |
| `alibayram/tabi-downstream-task-embeddinggemma`  | TabiBERT  | EmbeddingGemma   | 0.3830     | 0.3722     |

#### Key Observations

1. **MFT consistently outperforms BPE** across all initialization strategies
2. **Best baseline**: MFT + EmbeddingMagibu cloning achieves 0.61 Pearson (vs 0.42 for TabiBERT)
3. **EmbeddingMagibu > EmbeddingGemma** for both tokenizers as clone source

> 💡 **Insight**: The morphologically-aware vocabulary design of MFT provides a strong inductive bias that improves semantic similarity even before any fine-tuning.

### Post-Training Results

Results will be tracked after training in `sts_benchmark_results.json`. We compare:

- **Pearson correlation** with human similarity scores
- **Spearman correlation** (rank-based)
- Performance across training checkpoints and epochs

## 🔧 Technical Details

### Embedding Initialization Strategies

1. **Cloned (Mean Pooling)**: For each target token, find matching source tokens and average their embeddings
2. **Random (Xavier, Seeded)**: Initialize all parameters randomly using Xavier uniform initialization with a fixed seed (42) for reproducibility. The `random_init.py` script creates the model **once** and pushes to both repos—only the tokenizer files differ:
   - `alibayram/mft-random-init`: No tokenizer files (uses custom `TurkishTokenizer` at inference)
   - `alibayram/tabi-random-init`: Includes TabiBERT tokenizer files

### Why TabiBERT was pruned to 32K?

TabiBERT originally has ~52K tokens. We pruned to 32K to match MFT vocabulary size, ensuring fair comparison (same embedding matrix capacity).

### Modified SentenceTransformers

The standard library doesn't support custom tokenizers. Our modifications:

- Added `custom_tokenizer` parameter to `SentenceTransformer.__init__`
- Modified `Transformer.tokenize()` to handle non-HuggingFace tokenizers
- Tokenizer is replaced after model loading (works with both cloned and new models)

## 📚 References

- [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m) - Base embedding model
- [TabiBERT](https://huggingface.co/boun-tabilab/TabiBERT) - Turkish BPE tokenizer source
- [Turkish STS Benchmark](https://huggingface.co/datasets/figenfikri/stsb_tr) - Evaluation dataset
- [SentenceTransformers](https://www.sbert.net/) - Embedding framework

## License

This project is open source. See LICENSE file for details.

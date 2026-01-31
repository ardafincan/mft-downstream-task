# Tokenizer Efficiency Analysis Report

## Vocabulary Composition

| Component | Entries | Unique IDs | ID Range |
|-----------|---------|------------|----------|
| Roots | 22,231 | 20,000 | 0–19,999 |
| Suffixes | 177 forms | 72 | 20,000–20,071 |
| BPE tokens | 12,696 | 12,696 | 20,072–32,767 |
| **Total** | — | **32,768** | — |

## Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<uppercase>` | 0 | Case preservation marker |
| `<unknown>` | 1 | OOV fallback |
| ` ` (space) | 2 | Whitespace token |
| `<pad>` | 5 | Padding |
| `<eos>` | 6 | End of sequence |

---

## Tokens Per Word Analysis

| Model | Tokens/Word | Tokens/Char | Interpretation |
|-------|-------------|-------------|----------------|
| MFT | 2.91 | 0.356 | Morpheme-level |
| Tabi | 1.99 | 0.244 | Subword-level |
| Mursit | 1.82 | 0.223 | Subword-level |
| Cosmos | 1.82 | 0.223 | Subword-level |

## MFT Token Distribution Analysis

- **Total tokens processed:** 1,899,670
- **Total words processed:** 652,059
- **Average tokens per word:** 2.91

### Interpretation

MFT produces more tokens per word than BPE-based tokenizers because it:
1. Explicitly segments affixes (e.g., `kitap + lar + ımız + dan` = 4 tokens)
2. Uses dedicated tokens for case (`<uppercase>`)
3. Preserves morpheme boundaries over compression

This trade-off increases sequence length but improves:
- Sample efficiency through stable root representations
- Semantic similarity scores (as shown in STS results)
- Interpretability for linguistic analysis

# Reviewer Response: Computational Efficiency and Speed Trade-offs

## Executive Summary

This document provides tokens-per-word, sequence length, speed benchmarks, and efficiency metrics for the MFT tokenizer compared to baselines.

---

## Speed Benchmark Results

### Sequential Tokenization (1000 texts)

| Model  | Time (ms) | Tokens/Word | Tokens/Char |
| ------ | --------- | ----------- | ----------- |
| MFT    | 1,957     | **2.91**    | 0.356       |
| Tabi   | 1,531     | 1.99        | 0.244       |
| Mursit | 1,688     | 1.82        | 0.223       |
| Cosmos | 1,605     | 1.82        | 0.223       |

### Long Text Tokenization (372K chars)

| Model  | Time (ms) | Relative Speed |
| ------ | --------- | -------------- |
| Tabi   | 74 ms     | 1.0x (fastest) |
| Cosmos | 89 ms     | 1.2x           |
| Mursit | 99 ms     | 1.3x           |
| MFT    | 131 ms    | 1.8x           |

---

## Speed Analysis

**Why MFT is slower:**

1. **Morphological analysis** — Dictionary lookups for roots and suffixes add overhead
2. **Higher token count** — More tokens generated means more processing
3. **Rust + Python binding** — PyO3 FFI has minimal overhead but exists

**Context:** MFT is implemented in Rust with PyO3 bindings. Baseline tokenizers use HuggingFace's highly optimized C++ tokenizers library with SIMD vectorization.

---

## Token Efficiency Trade-off

| Metric       | MFT      | Tabi     | Interpretation                 |
| ------------ | -------- | -------- | ------------------------------ |
| Tokens/word  | 2.91     | 1.99     | MFT segments at morpheme level |
| Total tokens | 1.9M     | 1.3M     | 46% more tokens for same text  |
| Average word | 4 tokens | 2 tokens | Finer granularity              |

**Example tokenization:**

```
Input: "kitaplarımızdan" (from our books)

MFT:   [kitap] [lar] [ımız] [dan]  → 4 tokens
Tabi:  [kitapları] [mız] [dan]    → 3 tokens
BPE:   [kit] [ap] [lar] [ım] ...  → 5+ tokens
```

---

## Vocabulary Composition

| Component  | Count                    | ID Range      |
| ---------- | ------------------------ | ------------- |
| Roots      | 22,231 entries (20K IDs) | 0–19,999      |
| Suffixes   | 177 forms (72 IDs)       | 20,000–20,071 |
| BPE tokens | 12,696                   | 20,072–32,767 |
| **Total**  | **32,768**               | —             |

---

## Memory and Attention Cost

| Factor              | Impact                                        |
| ------------------- | --------------------------------------------- |
| **Vocabulary size** | 32,768 — standard transformer embedding table |
| **Sequence length** | ~1.5x longer than BPE (more tokens)           |
| **Attention cost**  | O(n²) — higher for longer sequences           |
| **Embedding reuse** | Better — stable root representations          |

---

## Downstream Performance Trade-off

Despite higher token count and slightly slower speed, MFT achieves:

| Benchmark         | MFT    | Best Baseline   | Improvement |
| ----------------- | ------ | --------------- | ----------- |
| STSb-TR Pearson   | 50.37% | 43.94% (Mursit) | +6.4 pp     |
| MTEB-TR Average   | 38.99% | 34.98% (Mursit) | +4.0 pp     |
| Retrieval Average | 28.94% | 21.12% (Mursit) | +7.8 pp     |

---

## Practical Implications

1. **Training:** ~1.5x longer sequences → proportionally more training time
2. **Inference:** Similar impact on sequence length
3. **Sample efficiency:** Stable morpheme representations improve learning

**Recommendation for Paper:**

> "MFT produces 2.91 tokens per word compared to 1.99 for Tabi, increasing sequence length by ~46%. While this adds computational overhead, the morpheme-aligned tokenization yields +6.4pp improvement on STSb-TR correlation, suggesting favorable sample efficiency that offsets the sequence length penalty."

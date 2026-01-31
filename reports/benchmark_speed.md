# Tokenizer Speed Benchmark Report

## Summary

Performance comparison of MFT Turkish tokenizer against baseline tokenizers.

---

## Sequential Tokenization (Many Short Texts)

| Model  | Time (ms) | Speedup vs MFT | Tokens    | Tok/Word | Tok/Char |
| ------ | --------- | -------------- | --------- | -------- | -------- |
| MFT    | 1935.20   | Baseline       | 1,899,670 | 2.91     | 0.356    |
| Tabi   | 1544.04   | 0.80x faster   | 1,298,725 | 1.99     | 0.244    |
| Mursit | 1654.60   | 0.86x faster   | 1,187,418 | 1.82     | 0.223    |
| Cosmos | 1620.09   | 0.84x faster   | 1,186,834 | 1.82     | 0.223    |

## Long Text Tokenization (Concatenated Corpus)

| Model  | Time (ms) | Speedup vs MFT |
| ------ | --------- | -------------- |
| Tabi   | 74.22     | 0.57x faster   |
| Cosmos | 93.63     | 0.72x faster   |
| Mursit | 103.20    | 0.79x faster   |
| MFT    | 130.73    | Baseline       |

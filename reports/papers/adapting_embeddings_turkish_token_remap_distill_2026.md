# Adapting Pretrained Embedding Models to Turkish via Token Remapping and Distillation (Bayram, 2026) - Notes

Source: `papers/Adapting Pretrained Embedding Models to Turkish via Token Remapping and Distillation.pdf`

## What it is

1. Train a Turkish SentencePiece/BPE tokenizer (large vocab).
2. Clone a teacher embedding model by mapping new tokens to teacher-token sequences and initializing embeddings by composition.
3. Train with offline teacher-embedding alignment (cosine objective) to avoid online teacher inference.

## Key points relevant to our paper

- Strong motivation that tokenizer choice and vocabulary adaptation matter for Turkish embedding quality.
- Reports STS and TR-MTEB style evaluations and emphasizes practical, reproducible tooling for Turkish embedding models.

## How it relates to our work

- Supports the downstream-evaluation framing: morphology/segmentation choices should be validated with real tasks (STS/retrieval benchmarks), not only token purity.
- Provides a complementary perspective: tokenizer adaptation via token remapping vs our morphology-first tokenization pipeline.

## Paper insertion points

- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/related_work.tex`: cite as Turkish embedding adaptation work motivating downstream STS/MTEB evaluation.

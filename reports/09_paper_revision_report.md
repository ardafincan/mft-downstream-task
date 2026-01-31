# Paper Revision Report (Jan 29, 2026)

This report summarizes the changes made to the paper in `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/` after reviewing `reviewers.txt`, the `reports/` experiment notes, and the repository codebase.

## Goals from the request

- Add substantially more detail about the downstream experiments and their results (without removing existing information).
- Keep the paper’s writing style consistent and avoid `itemize`.
- Improve figure/table placement so pages remain readable.
- Ensure the paper addresses the major reviewer concern: missing downstream task performance.

## Key paper changes

### 1) Expanded and clarified the downstream experimental protocol

File: `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex`

Added a detailed description of the controlled downstream setup:

- **Architecture control:** all four models share the same encoder backbone (`google/embeddinggemma-300m`) and the same vocabulary size (32,768).
- **Initialization control:** all models are random-initialized with a fixed seed (42).
- **Training control:** all models are trained with the same embedding-distillation objective and hyperparameters; only tokenizer / `input_ids` differs.
- **Data control:** training uses `alibayram/cosmos-corpus-0-05-with-embeddings` and the unified encoded dataset `alibayram/cosmos-corpus-encoded`, which stores token streams for all tokenizers.
- **Length control:** strict max sequence length 2048; any sample exceeding the limit for _any_ tokenizer is dropped to prevent truncation-based unfairness.
- **Reproducibility note:** documented the “tokenizer bypass” approach used to train the custom Python tokenizer stream with the same trainer stack.

New table:

- `Table~\ref{tab:distillation_setup}`: summarizes architecture, dataset IDs, sequence length filtering, hyperparameters, and hardware.

### 2) Added more detailed STS reporting (numbers + efficiency + statistical framing)

File: `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex`

Additions:

- Explicitly stated **test split sample size** (`n=1379`) and **train split sample size** (`n=5749`) from `sts_benchmark_results.json`.
- Reported **Pearson + Spearman** for all four models with consistent units.
- Added a **95% CI statement** (Fisher transform) for Pearson to support “statistically robust” claims at the given `n`.
- Added a table containing **end-to-end encoding time** reported by the evaluation script.

New table:

- `Table~\ref{tab:sts_results}`: STS train/test Pearson+Spearman and timing (batch size 32).

### 3) Version-tracking robustness: corrected and quantified

File: `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex`

Fixes/additions:

- **Corrected an incorrect claim** that a revision reached “76.10% Pearson” (not supported by the tracked `*-random-init` version-eval JSONs in this repo).
- Added a robustness table based on the tracked HuggingFace commit history evaluation outputs:
  - `version_eval_alibayram_mft_random_init.json`
  - `version_eval_alibayram_tabi_random_init.json`
  - `version_eval_alibayram_cosmosGPT2_random_init.json`
  - `version_eval_alibayram_newmindaiMursit_random_init.json`

### 4) MTEB-TR: added category summary + average figure

File: `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex`

Additions:

- Added an **average-score figure** using the existing artifact:
  - `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/figures/mteb_average_scores.png`
- Added **category-average table** using the grouped results in `MTEB_BENCHMARK_RESULTS.md`.

New figure/table:

- `Figure~\ref{fig:mteb_average}`: MTEB average over 26 tasks.
- `Table~\ref{tab:mteb_category_averages}`: category-level averages for BitextMining/Classification/Clustering/Other/PairClassification/Retrieval/STS.

### 5) Removed `itemize` from the paper (style constraint)

File: `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex`

The previous category analysis used an `itemize` block; it was rewritten into narrative paragraphs so the paper has no `itemize` environments.

### 6) TurBLiMP interpretation fixed to match the code

File: `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex`

The previous wording described TurBLiMP as “accuracy distinguishing grammatical/ungrammatical sentences”. In this repo, `evaluate_turblimp.py` computes **average cosine similarity** between good/bad pairs (embedding-based proxy). The paper now:

- Defines the metric as cosine similarity between minimal pair embeddings.
- Notes each category uses 1,000 pairs in our run.
- Explicitly warns it should not be interpreted as direct grammaticality classification accuracy.

## Figure/table placement adjustments

File: `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex`

To prevent page overflow after adding multiple new tables/figures:

- Several large floats were changed from forced placement (`[H]`) to top-of-page placement (`[t]`).
- The version-history figures were slightly reduced in width (`0.9\linewidth` → `0.85\linewidth`).
- The STS chart float was changed from `[h]` to `[t]` for more consistent placement.

## Validation

- Verified the paper builds via `latexmk -pdf main.tex` under `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/`.
- The build still emits some existing PDF destination-duplication warnings from hyperref, but compilation succeeds and the new content is included.

## Mapping back to `reviewers.txt` (high-level)

- **Major concern (missing downstream performance):** addressed by significantly expanding the downstream evaluation protocol and adding STS/MTEB/TurBLiMP details, robustness, and additional tables/figures.
- **Experimental transparency / reproducibility:** added dataset IDs, architecture/seed control, sequence-length filtering logic, and trainer/stack notes.
- **Evaluation validity:** corrected the TurBLiMP metric description to match the actual computation.

## Files changed

- Modified: `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex`
- Added: `reports/09_paper_revision_report.md`

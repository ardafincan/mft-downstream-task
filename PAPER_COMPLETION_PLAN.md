# Paper Completion Plan: *Tokens with Meaning* + Experiments Integration

## 0) Goal (what “done” means)

1. **Complete and polish the paper** in `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/` so it is internally consistent, reproducible, and includes the repo’s experimental evidence.
2. **Read and cite every paper in `papers/`** and mention each in the paper text (at least once, in an appropriate section).
3. **Incorporate tests/experiments from this repo** (distillation setup + STS + MTEB + version tracking) into the paper with clear methodology, results tables/figures, and limitations.
4. **Create small Markdown “reports” per folder/file first**, then convert the key points into LaTeX edits (paper sections, tables, figures, citations).

Deliverable: updated LaTeX sources + updated `tokenizer.bib` + any new `*.tex` tables/figures needed, and the paper compiles cleanly.

---

## 1) Scope & important clarifications (updated with your decisions)

### 1.0 Confirmed decisions (from you)
- **Downstream distillation experiments will be part of the main Results** (not Appendix-only), with the Appendix used only for long version-history tables if needed.
- **Tokenizer comparison is the objective**; we will not center narrative on architecture effects (Gemma vs Magibu), beyond minimal reporting as a control.
- If it improves clarity/presentation, it’s OK to **re-run report-generation scripts** and/or add a small helper script for cleaner charts/tables.

### 1.1 “Read all files detailly”
This repo includes large vendored codebases (notably `sentence_transformers/` and `mteb-tr/`). A literal line-by-line “deep reading” of those would be extremely time-consuming and mostly irrelevant to the paper.

Proposed interpretation (recommended):
- **Project-owned code** (top-level scripts + tokenizer implementation + results artifacts) gets deep reading and detailed reporting.
- **Vendored libraries** get **targeted reading**: we document (a) what they are, (b) how we use them, and (c) any modifications we made (e.g., tokenizer bypass in `sentence_transformers/models/Transformer.py`).

Assumption (unless you object): **targeted reading for vendored libraries** + deep reading for project-owned code and produced artifacts.

### 1.2 Re-running experiments vs. using existing artifacts
Some scripts depend on HuggingFace datasets/models and require network/HF tokens (e.g., `prepare_dataset.py`, `train.py`, `evaluate_sts_tr.py` for remote models).

Proposed approach:
- Use existing artifacts already in-repo (`*_BENCHMARK_RESULTS.md`, `*.json`, `results/**`, `*.png`, `*.pdf`) as the “source of truth”.
- Optionally re-run selected scripts only if you want fresh reproduction checks.

Assumption (per your note): **only re-run if it improves tables/figures**, otherwise integrate existing artifacts.

### 1.3 Narrative goals for the downstream section (your “most important part”)

#### 1.3.1 Core claims we will support using repo artifacts
1. **STS:** MFT-distilled models achieve materially higher Pearson/Spearman on `figenfikri/stsb_tr` than Tabi-distilled models (source: `STS_BENCHMARK_RESULTS.md`, `sts_benchmark_results.json`).
2. **MTEB-TR:** Even if Tabi gets isolated wins on some tasks/aggregates, the tokenizer-first story prioritizes **semantic similarity + retrieval relevance**, where MFT is stronger overall in our evidence (source: `MTEB_BENCHMARK_RESULTS.md`, `results/**`).
3. **Random-init sanity check:** With random initialization, **Tabi does not catch MFT** on STS and does not surpass it on overall MTEB averages; this supports that improvements are not just noise (source: the random-init rows in both STS/MTEB reports).

#### 1.3.2 How we will handle “BPE (Tabi) is slower to train/adapt”
The repo currently provides strong **quality** evidence, but not an obvious direct **training-time** comparison between tokenizers.

Plan (safe + defensible):
- Prefer the empirically supported statement: **“Under the same downstream distillation budget, Tabi does not catch MFT on STS/retrieval-focused quality.”**
- If you want an explicit *efficiency* angle, we’ll do one of:
  - (A) add a small local-only microbenchmark (tokenization/encoding throughput on a fixed text sample) and report it as *tokenization throughput*, not training speed; or
  - (B) keep “adaptation” phrasing as *sample efficiency / need for language-specific priors*, backed by citations rather than wall-clock claims.

---

## 2) Inventory (what we will touch and why)

### 2.1 Paper sources (primary)
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/main.tex` (main entry)
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/abstract.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/introduction.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/methodology.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/related_work.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/future_work.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/conclusion.tex`
- `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/tokenizer.bib`

### 2.2 “Downstream” experiment artifacts to integrate
- Protocol/writeups: `TRAINING_DETAILS.md`, `STS_BENCHMARK_RESULTS.md`, `MTEB_BENCHMARK_RESULTS.md`, `VERSION_BENCHMARK_RESULTS.md`
- Raw/structured results: `sts_benchmark_results.json`, `sts_benchmark_results_p.json`, `results/**`
- Figures: `sts_benchmark_chart*.png`, `mteb_average_scores.png`, `version_history_*.png`

### 2.3 Key experiment code (to document)
- Dataset: `prepare_dataset.py`, `upload_dataset.py`
- Training: `train.py`, `embedding_trainer.py`
- Evaluation/reporting: `evaluate_sts_tr.py`, `generate_sts_tables.py`, `generate_mteb_report.py`
- Baselines/setup: `random_init.py`, `setup_remote.sh`, `requirements.txt`
- Tokenizer + dictionaries: `turkish_tokenizer.py`, `turkish_decoder.py`, `kokler.json`, `ekler.json`, `bpe_tokenler.json`
- Integration test: `test_custom_tokenizer.py`

### 2.4 Papers that must be cited + mentioned (from `papers/`)
- `papers/Asgari et al. - 2025 - MorphBPE ... .pdf`
- `papers/Beken Fikri et al. - 2021 - Semantic Similarity Based Evaluation ... .pdf`
- `papers/Rahimov - 2025 - miLLi Model ... .pdf`
- `papers/Türker et al. - 2026 - TabiBERT ... .pdf`
- `papers/papers.bib` (BibTeX entries we can reuse/merge)

### 2.5 Vendored/auxiliary code (targeted documentation)
- `sentence_transformers/` (we will document the tokenizer bypass + how it enables offline `input_ids` training)
- `mteb-tr/` (we will document how we use it for MTEB-style evaluation + which tasks/metrics are reported)

---

## 3) Reporting workflow (Markdown first, then LaTeX)

I will create a `reports/` folder and produce short, focused Markdown notes before editing the paper. Proposed structure:

- `reports/00_repo_overview.md`
- `reports/01_tokenizer_implementation.md` (tokenizer pipeline + dictionaries + decoding)
- `reports/02_dataset_preparation.md` (`prepare_dataset.py`, dataset schema, filtering, sequence limits)
- `reports/03_training_distillation.md` (`train.py`, `embedding_trainer.py`, hyperparams, hardware)
- `reports/04_sts_evaluation.md` (`evaluate_sts_tr.py`, charts, JSON format)
- `reports/05_mteb_evaluation.md` (`generate_mteb_report.py`, `results/**`, aggregate metrics)
- `reports/06_version_tracking.md` (`VERSION_BENCHMARK_RESULTS.md`, what “revision” means, charts)
- `reports/07_vendored_dependencies.md` (what we rely on inside `sentence_transformers/` + `mteb-tr/`, and what we changed)
- `reports/papers/`:
  - `reports/papers/morphbpe_2025.md`
  - `reports/papers/semantic_similarity_summarization_2021.md`
  - `reports/papers/milli_2025.md`
  - `reports/papers/tabibert_2026.md`

Each report will follow the same mini-template:
- **What this file/folder is**
- **Key parameters/assumptions**
- **Outputs produced**
- **How it supports paper claims**
- **Exact LaTeX section(s) to update**

---

## 4) Paper integration plan (what will be added/changed)

### 4.1 Baseline pass: ensure the current paper is coherent
1. Read `main.tex` and each section `*.tex` end-to-end.
2. Identify gaps/inconsistencies:
   - Missing citations / uncited claims
   - Missing dataset/benchmark definitions
   - Terminology drift (MFT vs hybrid tokenizer naming)
   - Figures/tables referenced but not introduced (or vice versa)
3. Create a short “paper gap list” report: `reports/00_repo_overview.md`.

### 4.2 Literature: integrate and mention each `papers/` PDF
For each PDF in `papers/`:
1. Extract: problem, method, datasets, metrics, key findings, and how it relates to our approach.
2. Add/merge BibTeX entry into `Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/tokenizer.bib`.
3. Add at least one explicit mention and citation in the paper text:
   - Likely `related_work.tex` for all four
   - Possibly `results_and_analysis.tex` (if directly comparable)
   - Possibly `methodology.tex` (if method inspiration/contrast)

Planned placement (initial proposal, will confirm after reading each PDF):
- MorphBPE (morphology-aware + BPE hybrid): `related_work.tex` + “hybrid tokenization” framing.
- miLLi (local linguistic insights for robust tokenization): `related_work.tex` + “linguistic priors” discussion.
- TabiBERT (Turkish foundation model + benchmark): `related_work.tex` + benchmark context; also cite where we describe Tabi tokenizer baseline.
- Semantic similarity for summarization evaluation (2021): cite when motivating correlation-based evaluation (ties cleanly to STS-like correlation reporting in our downstream experiments).

### 4.3 Experiments: add a new “Downstream Distillation” evaluation section
The current paper focuses on tokenization quality and TR-MMLU-style evaluation. This repo adds a second axis: *tokenizer choice → embedding distillation quality → downstream retrieval/STS performance*.

Plan:
1. Add a subsection in `methodology.tex`:
   - Distillation objective (cosine loss)
   - Teacher embedding column and offline encoding rationale
   - Tokenizer bypass details (why needed; what code change enables it)
   - Training hyperparameters and compute (from `TRAINING_DETAILS.md`)
2. Add a subsection in `results_and_analysis.tex`:
   - STS results summary (from `STS_BENCHMARK_RESULTS.md`)
   - MTEB results summary (from `MTEB_BENCHMARK_RESULTS.md`)
   - Version history analysis (from `VERSION_BENCHMARK_RESULTS.md`) as:
     - a short robustness/reproducibility note in the main text, and
     - an Appendix section only if the detailed history tables are too long
3. Add figures/tables:
   - Convert key Markdown tables into LaTeX `table` environments (or `\input{}` generated `*.tex` tables).
   - Include existing charts `sts_benchmark_chart_test.png`, `mteb_average_scores.png`, `version_history_pearson.png`, etc.
4. Add a limitations paragraph:
   - dependency on teacher model choice
   - remaining confounds (backbone differences are reported but not the focus)
   - what the random-init baseline demonstrates

### 4.4 Implementation details: tokenizer algorithm + dictionaries
In `methodology.tex` (or a dedicated implementation subsection):
- Tie the *code-level pipeline* (`turkish_tokenizer.py` + dictionaries) to the conceptual framework:
  - root finding / longest-match heuristic
  - affix segmentation + equivalence classes
  - phonological normalization and reverse decoding
  - BPE fallback behavior
  - casing/whitespace special tokens

Where helpful, add one compact algorithm/pseudocode block or flow figure (only if the paper benefits and space allows).

---

## 5) Concrete execution steps (what I will do once you approve)

### Phase A — Reporting (Markdown reports)
1. Create `reports/` and write `reports/00_repo_overview.md`.
2. Write the 6–8 focused reports listed in section 3.
3. For each report, include “paper insertion points” (exact `*.tex` file + subsection title).

### Phase B — Paper edits (LaTeX integration)
1. Update `related_work.tex` to include and discuss all `papers/` items.
2. Update `methodology.tex` with a “Downstream Distillation Setup” subsection:
   - dataset, offline encoding, bypass patch, training details
3. Update `results_and_analysis.tex` with tokenizer-first framing:
   - **STS:** headline table + chart, explicit “MFT vs Tabi” delta, and random-init sanity check.
   - **MTEB-TR:** overall average + category averages + short category takeaways (do not over-focus on backbone differences).
   - **Version benchmark:** 3–5 sentence robustness summary; move detailed version history table(s) to Appendix if needed.
4. Update `tokenizer.bib`:
   - merge entries from `papers/papers.bib`
   - ensure every new citation key is referenced at least once
5. Ensure `main.tex` compiles cleanly and references resolve.

### Phase C — Reproducibility & polish
1. Confirm all reported numbers match their source artifacts (`*.md` / `*.json` / `results/**`).
2. Add a short “Reproducibility” paragraph (commands + artifact locations).
3. Run a final LaTeX build check; fix broken references/figures.

Optional (only if it improves presentation):
- Re-run `generate_sts_tables.py` / `generate_mteb_report.py` to regenerate cleaner tables.
- Add a small helper script to emit LaTeX tables directly from `sts_benchmark_results.json` and `results/**` so paper tables stay consistent with raw results.

---

## 6) Acceptance checklist (quick review criteria)

- All four `papers/*.pdf` are (a) cited in BibTeX and (b) explicitly mentioned in the paper text.
- Downstream experiments are described with enough detail to reproduce (inputs, code entrypoints, hyperparams, metrics).
- Figures/tables compile and are referenced in text.
- No dangling citations; bibliography compiles.
- Claims in the paper are supported by either:
  - cited literature, or
  - results artifacts in this repo.

---

## 7) Quick confirmations (so I implement the exact narrative you want)

1. OK to phrase the key claim as: **“Under the same downstream distillation budget, Tabi does not catch MFT on STS/retrieval-focused quality”** (and keep any “slower to train” claim either benchmarked separately or softened/cited)?
2. For MTEB, should the headline metric be **overall average** (simple) or a **retrieval+STS subset average** (more aligned with your objective)?
3. Should detailed version history go to the **Appendix by default** (recommended), with only a short robustness summary in the main Results?

# Final Revision Report: "Tokens with Meaning"

## Overview

This document details the revisions made to the paper _"Tokens with Meaning: A Hybrid Tokenization Approach for NLP"_ in response to reviewer feedback and new experimental data.

## 1. Structural Refactoring

### Methodology Section (`methodology.tex`)

- **Restructured into Subsections:**
  - **Dictionary Construction:** Detailed the root (22k) and affix (230) inventories, including the phonological normalization strategy (mapping variants like _kitap_/_kitabı_ to single IDs).
  - **Input Normalization:** documented usage of `<uppercase>` tokens and CamelCase splitting.
  - **Encoding Algorithm:** Added `Algorithm 1` (pseudocode) to formally define the greedy longest-prefix match + BPE fallback pipeline.
  - **Decoding Algorithm:** Explained the reconstruction logic (vowel harmony, consonant assimilation).
- **Glossing:** Updated linguistic examples to use standard formatting (Segmented line, Morpheme Gloss line, Free Translation line).
- **Efficiency:** Removed unsupported performance claims; focused on method description.

### Introduction (`introduction.tex`)

- Ensured strict separation of content: moved specific quantitative result summaries to the Results section, keeping the Intro focused on motivation, problem definition, and contributions.

## 2. Experimental Results Integration (`results_and_analysis.tex`)

### Expanded Downstream Evaluation

- **MTEB-TR Analysis:**
  - Integrated the full 26-task MTEB benchmark results.
  - Added a detailed breakdown of performance trade-offs:
    - **MFT Wins:** Dominates in **Semantic Textual Similarity (STS)** (+16.1%) and **Retrieval**, validating the "meaning-first" hypothesis.
    - **Tabi Wins:** Stronger in generic **Classification** and **BitextMining**, attributed to its massive 1T token pre-training (vs. limited training for our random-init baseline).
  - **Conclusion:** MFT provides a better structural prior for semantic tasks, while massive scale (Tabi) aids surface pattern recognition.

### STS Benchmark

- Confirmed MFT's significant lead in zero-shot STS (Pearson: 50.37% vs 33-43% for baselines).
- Included Version History plots to demonstrate robustness across experiment iterations.

### Bibliography (`tokenizer.bib`)

- Verified integration of references from `papers/` (TabiBERT, MorphBPE, etc.).
- Ensured citation styles in `related_work.tex` are consistent and do not treat citations as nouns where inappropriate.

## 3. Reviewer Checklist Compliance

- [x] **Missing downstream task performance:** Added STS and MTEB sections.
- [x] **Intro contains results:** Quantitative claims moved.
- [x] **Methodology needs subsections + algorithm:** Done.
- [x] **Linguistic examples need glossing:** Applied `\glossex` formatting.
- [x] **Citation style:** Checked `related_work.tex`.
- [x] **Data transparency:** Sources for corpora and dictionaries described in Methodology.

## Compilation Status

- The paper compiles successfully with `pdflatex`.
- Cross-references (Algorithms, Figures, Tables) are resolved.

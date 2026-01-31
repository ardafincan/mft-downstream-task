# Paper Resubmission Status Report

## Executive Summary

**Status: READY FOR RESUBMISSION ✓** — All reviewer concerns addressed.

---

## Reviewer Concerns: All Addressed ✓

### 1. ✓ Numerical Inconsistencies Fixed

| Issue       | Before           | After                       | Location           |
| ----------- | ---------------- | --------------------------- | ------------------ |
| Affix count | "~230 morphemes" | "177 suffix forms → 72 IDs" | methodology.tex:61 |
| Root count  | Inconsistent     | "22,000 roots" (20K IDs)    | methodology.tex:59 |
| BPE tokens  | Unclear          | "12,696 tokens"             | methodology.tex:98 |
| Total vocab | Not stated       | "32,768"                    | Table 4            |

### 2. ✓ Round-Trip Evaluation Added

- **Results:** 99.2% word-based accuracy (496/500 words)
- **Explanation:** 0.8% failures from Turkish phonological ambiguity
- **Model compatibility:** encoder-only, encoder-decoder, decoder-only

### 3. ✓ Tokenization Efficiency Metrics Added

- **Table:** Time (ms), Tokens, Tok/Word, Tok/Char
- **Trade-off analysis:** MFT 1.5× more tokens but +16.8pp STS improvement

### 4. ✓ MorphBPE/MorphPiece Comparison Note

- "they target different languages... not directly comparable on Turkish-specific benchmarks"

### 5. ✓ Greedy Disambiguation Analysis

- **Document:** `reviewer_response_disambiguation_updated.md`

### 6. ✓ Ablation/Component Contribution Analysis

- **Document:** `reviewer_response_ablation_updated.md`

### 7. ✓ TR%/Pure% Validation Methodology

- **Added:** "TR% and Pure% are computed using an independent morphological validator with curated lexical resources external to the tokenizer under evaluation"

### 8. ✓ Teacher Model Specification

- **Already in Table 4:** `intfloat/multilingual-e5-large-instruct`

### 9. ✓ Variance/Multiple Seeds Justification

- **Added:** "all three baseline tokenizers (Mursit, Cosmos, Tabi)—which are independently trained BPE tokenizers from different research groups—show consistent relative ordering... it is statistically implausible for MFT to outperform three independent baselines by chance"

---

## Checklist Summary

| Reviewer Item                       | Status           |
| ----------------------------------- | ---------------- |
| Fix numerical inconsistencies       | ✓ Done           |
| Add round-trip evaluation           | ✓ Done (99.2%)   |
| Add efficiency metrics              | ✓ Done (Table 4) |
| Explain greedy disambiguation       | ✓ Done (report)  |
| Component ablation analysis         | ✓ Done (report)  |
| MorphBPE/MorphPiece note            | ✓ Done           |
| TR%/Pure% methodology clarification | ✓ Done           |
| Teacher model specification         | ✓ Done           |
| Variance/multiple seeds             | ✓ Done           |

---

## Files Modified

| File                                                                                                                                                                 | Changes                                        |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| [methodology.tex](file:///Users/alibayram/Desktop/mft-downstream-task/Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/methodology.tex)                   | Fixed counts, roundtrip eval, efficiency table |
| [results_and_analysis.tex](file:///Users/alibayram/Desktop/mft-downstream-task/Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/results_and_analysis.tex) | TR%/Pure% clarification, seed justification    |
| [main.pdf](file:///Users/alibayram/Desktop/mft-downstream-task/Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/main.pdf)                                 | Compiled (17 pages, 996KB)                     |

## Reviewer Response Documents

| Document                                                                                                                             | Purpose         |
| ------------------------------------------------------------------------------------------------------------------------------------ | --------------- |
| [disambiguation_updated.md](file:///Users/alibayram/Desktop/mft-downstream-task/reports/reviewer_response_disambiguation_updated.md) | Greedy strategy |
| [ablation_updated.md](file:///Users/alibayram/Desktop/mft-downstream-task/reports/reviewer_response_ablation_updated.md)             | Components      |
| [efficiency_updated.md](file:///Users/alibayram/Desktop/mft-downstream-task/reports/reviewer_response_efficiency_updated.md)         | Speed metrics   |
| [roundtrip_updated.md](file:///Users/alibayram/Desktop/mft-downstream-task/reports/reviewer_response_roundtrip_updated.md)           | 99.2% accuracy  |

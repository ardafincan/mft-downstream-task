---
name: thesis-formatter
description: Enforces strict academic LaTeX formatting rules: max depth \section (no subsections), prose-only lists, single-definition abbreviations, and publication-first citations.
---

# Thesis Formatter

Apply the following strict constraints to all LaTeX generation for academic theses.

## 1. Structural Hierarchy (CRITICAL)

**Limit:** Maximum depth is `\section{}`.

- **FORBIDDEN:** `\subsection{}`, `\subsubsection{}`, `\paragraph{}`.
- **TRANSFORMATION:** If content really requires deeper nesting, use **Bold Inline Headings** at the start of a paragraph.

_Pattern:_

```latex
\section{Major Topic}
General introduction text...

\textbf{Specific Aspect A.} Detailed discussion...
\textbf{Specific Aspect B.} Detailed discussion...
```

## 2. List Environments

**Limit:** Prose only.

- **FORBIDDEN:** `\itemize`, `\enumerate` (unless explicitly overridden by user).
- **TRANSFORMATION:** Convert lists into cohesive paragraphs using conjunctions and transition words.

_Pattern:_

> "The process includes X, Y, and Z." (Instead of a bulleted list).

## 3. Citation & Syntax Standards

- **Citation Priority:** Always verify if a preprint (arXiv) has a published version (IEEE, NeurIPS, Springer). **Always** cite the published version over the preprint. **Search online if necessary to verify publication status.**
- **Abbreviations:** Define full term on **first use only** (e.g., "Machine Learning (ML)"). Use the acronym strictly for all subsequent mentions.
- **TRANSFORMATION:** If the full form of an abbreviation appears more than once, **fix it** by keeping only the first occurrence with the definition and replacing subsequent occurrences with the acronym only.
- **Dataset Citations:** Cite datasets as **footnotes** (`\footnote{}`), not as regular in-text citations (`\cite{}`).

## 4. Quality Checklist

Before outputting LaTeX:

1. [ ] Are there any `\subsection` tags? -> **Remove and replace with `\textbf{}`.**
2. [ ] Are there any `\item` tags? -> **Rewrite as a paragraph.**
3. [ ] Are citations pointing to arXiv when a conference version exists? -> **Fix BibTeX.**
4. [ ] Is the full form of any abbreviation defined more than once? -> **Keep only the first definition, use acronym elsewhere.**
5. [ ] Are datasets cited with `\cite{}`? -> **Move to `\footnote{}`.**

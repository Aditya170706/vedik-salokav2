# Retrieval Benchmark — Results

- Collection: **48 documents** (fixed sample48 pool)
- Queries: **31** (29 with at least one relevant passage; the remaining are negative/answerless and excluded from R/P/MRR)
- Judgment status: AI-generated candidates — **not yet reviewer-confirmed**
- Vector index: `D:\vedic_shlokaphase4\vedic_shloka_system\benchmarks\retrieval\vector_index_sample`

> **Coverage limitation.** The semantic index in this run contains only the 48 collection documents, so every method is compared on that identical fixed pool. Absolute scores are therefore not representative of retrieval over the full ~252k-document corpus, where keyword retrieval at least has full coverage. See `benchmarks/README.md`.

## 1. Method comparison (weights are semantic / keyword / graph)

| Variant | Weights (s/k/g) | R@1 | P@1 | R@3 | P@3 | R@5 | P@5 | R@10 | P@10 | MRR | MAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| semantic_only | 1.00/0.00/0.00 | 0.406 | 0.483 | 0.482 | 0.207 | 0.632 | 0.193 | 0.714 | 0.134 | 0.581 | 0.541 |
| keyword_only | 0.00/1.00/0.00 | 0.660 | 0.793 | 0.774 | 0.356 | 0.800 | 0.248 | 0.890 | 0.169 | 0.835 | 0.802 |
| graph_only | 0.00/0.00/1.00 | 0.162 | 0.241 | 0.170 | 0.103 | 0.187 | 0.069 | 0.470 | 0.083 | 0.306 | 0.279 |
| hybrid_full | 0.45/0.35/0.20 | 0.729 | 0.862 | 0.778 | 0.368 | 0.809 | 0.262 | 0.872 | 0.159 | 0.877 | 0.845 |
| hybrid_no_graph | 0.45/0.35/0.00 | 0.660 | 0.793 | 0.774 | 0.356 | 0.800 | 0.248 | 0.890 | 0.169 | 0.835 | 0.802 |
| hybrid_graph_0.10 | 0.45/0.35/0.10 | 0.694 | 0.828 | 0.778 | 0.368 | 0.812 | 0.262 | 0.837 | 0.155 | 0.860 | 0.827 |
| hybrid_graph_0.40 | 0.45/0.35/0.40 | 0.654 | 0.759 | 0.780 | 0.368 | 0.806 | 0.255 | 0.834 | 0.152 | 0.815 | 0.798 |
| hybrid_no_semantic | 0.00/0.35/0.20 | 0.619 | 0.724 | 0.668 | 0.322 | 0.685 | 0.214 | 0.812 | 0.134 | 0.751 | 0.737 |
| hybrid_no_keyword | 0.45/0.00/0.20 | 0.453 | 0.586 | 0.601 | 0.276 | 0.676 | 0.214 | 0.731 | 0.128 | 0.677 | 0.606 |

## 2. Knowledge-graph contribution (ablation)

Difference of the production hybrid against the same hybrid with the graph signal removed (`hybrid_full` − `hybrid_no_graph`).

| Metric | @1 | @3 | @5 | @10 |
| --- | --- | --- | --- | --- |
| ΔRecall | 0.069 | 0.003 | 0.010 | -0.018 |
| ΔPrecision | 0.069 | 0.011 | 0.014 | -0.010 |

ΔMRR (full − no-graph) = **+0.0427**
Paired randomisation test on per-query MRR (n=29 queries, 100,000 permutations, seed=1729): **p = 0.0628** (two-sided). Treat p > 0.05 as 'not distinguishable from noise' at this sample size.

## 3. Per-query breakdown (hybrid_full)

| Query | Category | n_rel | R@5 | P@5 | RR |
| --- | --- | --- | --- | --- | --- |
| Q01 | lexical | 4 | 1.000 | 0.800 | 1.000 |
| Q02 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q03 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q04 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q05 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q06 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q07 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q08 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q09 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q10 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q11 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q12 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q13 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q14 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q15 | lexical | 1 | 1.000 | 0.200 | 1.000 |
| Q16 | cross-lingual | 1 | 1.000 | 0.200 | 1.000 |
| Q17 | cross-lingual | 1 | 0.000 | 0.000 | 0.125 |
| Q18 | cross-lingual | 1 | 1.000 | 0.200 | 1.000 |
| Q19 | cross-lingual | 1 | 1.000 | 0.200 | 1.000 |
| Q20 | cross-lingual | 1 | 0.000 | 0.000 | 0.071 |
| Q21 | conceptual | 10 | 0.500 | 1.000 | 1.000 |
| Q22 | conceptual | 1 | 1.000 | 0.200 | 1.000 |
| Q23 | conceptual | 1 | 1.000 | 0.200 | 1.000 |
| Q24 | conceptual | 1 | 1.000 | 0.200 | 1.000 |
| Q25 | conceptual | 2 | 1.000 | 0.400 | 1.000 |
| Q26 | conceptual | 5 | 0.000 | 0.000 | 0.042 |
| Q27 | conceptual | 6 | 0.167 | 0.200 | 1.000 |
| Q28 | keyword-field | 9 | 0.556 | 1.000 | 1.000 |
| Q29 | conceptual | 4 | 0.250 | 0.200 | 0.200 |
| Q30 | negative | 0 | — | — | — |
| Q31 | negative | 0 | — | — | — |

## 3b. Per-category aggregation

MRR / R@5 per query category for every variant. Categories are defined in `queries.json` (lexical, cross-lingual, conceptual, keyword-field, negative).

| Variant | lexical: MRR / R@5 | cross-lingual: MRR / R@5 | conceptual: MRR / R@5 | keyword-field: MRR / R@5 | negative: MRR / R@5 |
| --- | --- | --- | --- | --- | --- |
| semantic_only | 0.603 / 0.750 | 0.396 / 0.600 | 0.664 / 0.454 | 0.500 / 0.444 | — |
| keyword_only | 1.000 / 1.000 | 0.396 / 0.600 | 0.778 / 0.579 | 1.000 / 0.556 | — |
| graph_only | 0.140 / 0.067 | 0.110 / 0.000 | 0.654 / 0.525 | 1.000 / 0.222 | — |
| hybrid_full | 1.000 / 1.000 | 0.639 / 0.600 | 0.780 / 0.615 | 1.000 / 0.556 | — |
| hybrid_no_graph | 1.000 / 1.000 | 0.396 / 0.600 | 0.778 / 0.579 | 1.000 / 0.556 | — |
| hybrid_graph_0.10 | 1.000 / 1.000 | 0.529 / 0.600 | 0.788 / 0.623 | 1.000 / 0.556 | — |
| hybrid_graph_0.40 | 1.000 / 1.000 | 0.402 / 0.600 | 0.703 / 0.602 | 1.000 / 0.556 | — |
| hybrid_no_semantic | 1.000 / 1.000 | 0.110 / 0.000 | 0.654 / 0.537 | 1.000 / 0.556 | — |
| hybrid_no_keyword | 0.646 / 0.750 | 0.639 / 0.600 | 0.718 / 0.615 | 1.000 / 0.444 | — |

## 4. Negative-query behaviour (hybrid_full)

Answerless queries: a well-behaved system should return low-confidence results. This reports the top score it assigns.

| Query | Query text | Top score | Top-1 id |
| --- | --- | --- | --- |
| Q30 | why is the sky blue | 0.357 | SHLOKA_002268 |
| Q31 | how to perform the soma sacrifice | 0.329 | SHLOKA_250531 |

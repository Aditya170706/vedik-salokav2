# Research Benchmarks

Reproducible evaluation harnesses for the Vedic Shloka Intelligence system.
They implement checklist items **3.1–3.5** (retrieval + mathematical-correctness
measurement) and produce the verified result tables needed for the manuscript.

```
benchmarks/
├── common.py                     # collection loading, IR metrics, writers
├── retrieval/
│   ├── queries.json              # 31 queries + relevance judgements
│   ├── run_retrieval_eval.py     # keyword / semantic / hybrid + graph ablation
│   │                             #   + per-category aggregation + paired
│   │                             #   randomisation significance test
│   ├── vector_index_sample/      # benchmark-local 48-doc index (derived)
│   └── search_index_sample/      # benchmark-local 48-doc keyword index (derived)
├── formula/
│   ├── items.json                # 21 formula items with provenance
│   └── run_formula_eval.py       # 5-stage math-correctness harness
├── render_manuscript_tables.py   # renders reports/manuscript.md tables from JSON
└── results/
    ├── retrieval_metrics.{json,md}
    └── formula_metrics.{json,md}
```

## Running

```bash
python benchmarks/retrieval/run_retrieval_eval.py          # ~2 min (loads embedding model once)
python benchmarks/retrieval/run_retrieval_eval.py --limit 5  # smoke test
python benchmarks/formula/run_formula_eval.py               # ~1 s
python benchmarks/render_manuscript_tables.py               # regenerate manuscript
python benchmarks/render_manuscript_tables.py --check       # verify freshness (CI)
python -m pytest tests/test_benchmarks.py -q                # 28 fast tests
```

The retrieval harness never mutates the shipped artifacts under `database/`;
it writes derived indexes under `benchmarks/retrieval/`.

## Evaluation collection (why 48 documents)

Semantic retrieval can only return documents present in the vector index, and
the shipped index contains **the 48 shlokas of the five sample sources**.
Scoring keyword (full ~252k-doc corpus) against semantic (48 docs) would be
meaningless, so every method is evaluated on the **identical 48-document pool**.
Keyword retrieval runs over a pool-local `LocalSearchIndex`; hybrid retrieval
reuses the production `HybridRetrievalEngine` with only its keyword backend
swapped for the pool-local one.

**This caps all absolute scores.** They are not representative of retrieval over
the full corpus, where keyword retrieval at least has full coverage.

## Headline results

### Retrieval (`results/retrieval_metrics.md`)

| Variant | R@1 | R@5 | MRR | MAP |
| --- | --- | --- | --- | --- |
| semantic_only | 0.406 | 0.632 | 0.581 | 0.541 |
| keyword_only | 0.660 | 0.800 | 0.835 | 0.802 |
| graph_only | 0.162 | 0.187 | 0.306 | 0.279 |
| **hybrid_full** | **0.729** | **0.809** | **0.877** | **0.845** |
| hybrid_no_graph | 0.660 | 0.800 | 0.835 | 0.802 |

- **The graph signal helps top-1 but not deep recall — and not significantly.**
  Removing it (`hybrid_full − hybrid_no_graph`) changes ΔMRR **+0.043**,
  ΔR@1 **+0.069**, but ΔR@10 **−0.018**. A paired randomisation test on
  per-query MRR (n=29, 100k permutations) gives **p = 0.063** — the effect
  does not reach significance at this sample size. Raising the graph weight
  to 0.40 *reduces* MRR to 0.815, so 0.20 is near-optimal for this pool.
- **Cross-lingual queries are where the channels diverge.** Per-category MRR
  (hybrid_full): lexical 1.000, conceptual 0.780, cross-lingual 0.639. The
  cross-lingual figure is carried by the semantic channel alone; keyword
  fusion dilutes it (keyword_only cross-lingual MRR 0.396).
- **The graph signal is a document-richness prior, not a relevance signal.**
  `_graph_relevance` scores a shloka by `len(relations)/10`, and the local graph
  gives every shloka 4 structural edges plus one per keyword — so the score
  measures keyword count. `graph_only` (MRR 0.306) is far below keyword.
- **Keyword beats dense retrieval on this pool** because most queries are
  lexical and the pool is tiny; the cross-lingual queries (Q17, Q20) and the
  bound-compound query (Q29 "वर्ग") are where semantic wins.
- **Negative queries** (Q30, Q31) still return a full ranked list with no way
  to abstain — the system cannot say "no answer".

### Mathematical correctness (`results/formula_metrics.md`)

| Stage | Scored | Correct | Rate |
| --- | --- | --- | --- |
| 1. Passage detection (accuracy) | 21 | 12 | 0.571 |
| 2. Formula retrieval (any known formula) | 15 | 10 | 0.667 |
| 3. Formula extraction (exact key) | 9 | 1 | 0.111 |
| 4a. Symbolic parse valid | 13 | 13 | 1.000 |
| 4b. Symbolic equivalence | 13 | 1 | 0.077 |
| 5. Interpretation name match | 15 | 1 | 0.067 |

The stages are measured **independently** (extraction is not gated on
detection) so one failure does not mask another. Interpretation is an
automatic proxy; final scoring is reviewer-judged.

## Bugs and gaps surfaced

1. **Stale vector index (data-consistency bug).** `database/vector_index/`
   stores 48 vectors whose IDs (`SHLOKA_000001`…) have **zero overlap** with the
   corpus (`SHLOKA_002266`…). Ranking works (text is stored) but returned IDs
   cannot be joined to the corpus. `benchmarks/common.vector_index_id_overlap`
   reports it; the benchmark uses a fresh index from the same 48 docs.
2. **Detection misses English-language mathematics.** `MATH_KEYWORDS_LATIN`
   contains only transliterations (`varga`, `kshetra`…), so every English math
   passage (F02, F04, F05, F07, F08) is missed → detection recall 0.50.
3. **Substring matching mis-routes formulas.** `FormulaExtractor._HINT_MAP`
   uses `kw in text`; `"भुज"` (side) is a substring of `"त्रिभुज"` (triangle),
   so triangle passages return `pythagorean`. Combined with dict-order
   precedence (`क्षेत्रफल` before `वृत्त`), circle passages return
   `triangle_area`. This drives extraction accuracy down to 0.111.
4. **Detection false positive from `क्षेत्र`.** Gita 1.1 `धर्मक्षेत्रे`
   (dharma-field) is flagged mathematical.
5. **O(E)-per-document graph lookup.** The local KG backend's
   `find_related_shlokas` scans all ~1.07M edges per call, so hybrid retrieval
   with the graph signal costs ~5 s **per query** on this 48-doc pool alone.
   The harness precomputes relation counts in one pass (identical scores).
6. **No abstention.** Negative queries always return top-k results.

## Limitations

- **48-document pool; 31 queries; 21 formula items.** Small; treat rates as
  indicative, not precise. No significance testing is possible.
- **Judgements are AI-generated candidates.** Every query and formula item sets
  `needs_reviewer: true`; a Sanskrit-reading reviewer must confirm relevance
  and the interpretation labels before publication.
- **Formula benchmark is text-only** (no KG `shloka_id`), isolating the
  keyword/registry path. The KG-lookup path is not benchmarked here.
- **Synthetic vs external sources.** Project-corpus passages are synthetic
  samples; the externally cited verses (Baudhāyana, Āryabhaṭīya) are
  `verified_secondary` and their Devanagari should be checked against a critical
  edition (`provenance.notes` records what needs checking).
- Detection is tested text-only; in production the Phase-2 domain prediction is
  also available and would recover some missed items.

## Provenance and reviewer workflow

Each formula item records `source.citation`, `source.license` and a
`provenance` block with `level` ∈ {`verified_secondary`, `project_corpus`,
`needs_review`}, `checked_by`, `needs_reviewer` and `notes`. The intended
workflow is: run the harness → a reviewer assigns `reviewer_score` and flips
`needs_reviewer` to false → re-run to regenerate the tables.

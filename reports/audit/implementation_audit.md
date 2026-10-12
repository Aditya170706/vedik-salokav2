# Implementation Audit — File-by-File

Scope: every Phase 1/2/3/4 component a research claim depends on. Each entry
lists (a) the claim the manuscript makes about the component, (b) what the code
actually does, (c) bugs or missing pieces found, with evidence. Line numbers are
indicative, not exact; files were read on 2026-10-10.

Severity: **B** = bug (wrong behaviour), **M** = methodological issue affecting
a claim, **G** = missing component / gap.

---

## 1. Corpus and data (Phase 1)

| File | Verdict |
| --- | --- |
| `scripts/module1_ingestion.py` – `module5_dataset.py` | Not re-audited line-by-line here; their outputs were verified by inspection of `data/annotated/` |

**Corpus inventory (verified by loading every annotated file):**

| File | Documents |
| --- | --- |
| `itihasa_ramayana_mahabharata.json` | 216,771 |
| `puranas_dataset.json` | 19,594 |
| `rigveda.json` | 10,477 |
| `atharvaveda.json` | 2,265 |
| `bhagavad_gita_18_chapters.json` | 1,402 |
| `yajurveda.json` | 1,453 |
| `samaveda.json` | 216 |
| 5 sample files (48 docs) | 48 |
| **Total** | **252,226** |

- **(M) Synthetic majority.** The two dominant collections
  (Itihāsa 216k, Purāṇas 19k — 94 % of the corpus) were produced/augmented by
  `scripts/expand_dataset.py` and `scripts/build_journal_dataset.py`. Domain
  labels on these documents are templated, not scholarly. Any claim of
  corpus-scale accuracy inherits this. *Manuscript must say the 252k corpus is
  a synthetic scale-test corpus and confine evaluation claims to the 48-doc
  pool + 21 formula items.*
- **(G) No train/test de-duplication audit.** With template expansion, near
  duplicates across train/test inflate Phase-2 accuracy (see §2).

## 2. Phase-2 classification (report files under `reports/`)

Verified numbers (`reports/*/eval_results.json`, test_size = 5,312 each):

| Task | Accuracy | Macro-F1 |
| --- | --- | --- |
| Veda | 0.9945 | 0.9787 |
| Domain | 0.8368 | **0.3146** |
| Branch | 0.8665 | **0.3185** |

- **(B→M) Accuracy hides class collapse.** Domain/branch accuracy ≈ 0.84–0.87
  but macro-F1 ≈ 0.31–0.32: the head classes carry accuracy while rare classes
  (including *Mathematics*) are near-zero recall. The veda report shows the
  same pattern structurally (`Other`: support 0, f1 0.0).
- **(M) The math detector consumes these labels** (`MathContentDetector`
  Rules 1–3), so detection quality is bounded by *Mathematics* recall, which
  the reports show to be weak. The formula benchmark deliberately runs
  detection text-only, which is the honest measurement, but the manuscript
  must not claim Phase-2 labels rescue detection.
- **(G) Leakage risk unquantified.** No de-dup check between train and test
  CSVs exists (`data/training/*.csv`), despite template-expanded data.

## 3. `database/` artifacts (consistency)

- **(B) Stale vector index — confirmed.** `database/vector_index/metadata.json`
  holds 48 vectors with IDs `SHLOKA_000001…SHLOKA_000048`; corpus IDs are
  `SHLOKA_000001`-free (`SHLOKA_002266…`). Overlap = 0/48, verified by
  `benchmarks.common.vector_index_id_overlap`. Ranking output is text-correct
  but IDs cannot be joined to the corpus, so any downstream KG lookup on a
  semantic result silently returns nothing (`LocalBackend` matches on
  `Shloka:<id>`). `p3_pipeline.py` only builds the index when
  `embeddings.npy` is *absent*, so the stale index is self-perpetuating.
- **(B) Semantic index covers 48 / 252,226 documents = 0.019 % of the
  corpus.** `build_vector_index()` was evidently run only on the five sample
  files (or aborted). Production semantic retrieval is a 48-document island.
  This is the single largest gap between the system's description ("hybrid
  retrieval over the corpus") and its implementation.
- **(G) Vector index is not sharded/ANN.** `SemanticSearchEngine.search` is an
  O(N·D) dense matmul + Python sort. Fine for 48 docs; untested at 252k (≈390 MB
  float32 — would fit, but the sort is the bottleneck). Not a bug today, but no
  evidence exists that scale claims hold.
- KG and search index are consistent with the corpus:
  `database/knowledge_graph/graph.json` = 252,267 nodes / 1,067,200 edges
  (BELONGS_TO, HAS_DOMAIN, HAS_BRANCH, FROM_SOURCE: 252,226 each; HAS_KEYWORD:
  54,294; HAS_FORMULA: 4,002 → only **1.6 %** of documents carry a formula
  edge).

## 4. `retrieval/hybrid_engine.py`

Claim: *final_score = 0.45·semantic + 0.35·keyword + 0.20·graph fuses three
complementary signals.*

- **(B) The graph signal is not a relevance signal.**
  `_graph_relevance = min(len(relations)/10, 1)`. Every shloka node has exactly
  four structural edges (BELONGS_TO, HAS_DOMAIN, HAS_BRANCH, FROM_SOURCE) plus
  one HAS_KEYWORD per keyword. So graph_score ≈ (4 + n_keywords)/10 — a
  document-keyword-count prior. Verified empirically: `graph_only` reaches MRR
  0.306 vs keyword 0.835; the ablation (ΔMRR +0.043 for graph on/off) is
  therefore a *keyword-prior* effect, not a relational one. The benchmark README
  documents this; the manuscript must describe it as such.
- **(M) Unnormalised score fusion.** keyword_score ∈ [0,1] is overlap/|query|,
  semantic_score is cosine (≈[−0.1, 1] for this model), graph ∈ [0, 1]. Weight
  magnitudes are not calibrated against each other; the 0.45/0.35/0.20 split is
  a heuristic tuned on a 31-query set (graph weight sweep 0.10/0.20/0.40 is in
  the benchmark; overfitting risk acknowledged in README).
- **(M) Candidate-set truncation.** Each retriever contributes only top-2k
  candidates; fusion happens after truncation, so a document ranked 2k+1 by one
  signal cannot be rescued by another.
- Minor: `__init__` eagerly constructs both sub-engines even if their weight is
  0. `doc` metadata comes from whichever retriever first inserted the id (keyword
  docs carry `shloka_text`, semantic docs carry `text`; `_format_retrieved`
  papers over this with a fallback).

## 5. `retrieval/semantic_search.py`

- Index builder and searcher are otherwise sound (unit-normalised embeddings,
  dot-product cosine, metadata stored alongside).
- **(B)** See stale-index issue §3; also `build_vector_index()` loads *all*
  annotated files via `glob("*.json")` yet the shipped index has 48 docs — the
  artifact on disk was not produced by the current code path (reproducibility
  gap: no build log, and `index_config.json` in the shipped dir predates the
  corpus renumbering).

## 6. `retrieval/symbolic_search.py`

- Token-overlap scoring (`_score`) is transparent and reproducible; token
  regex splits on danda — correct for Devanagari.
- **(M) Score ignores term rarity** (raw overlap count): common terms
  (e.g. "सूत्र") match as strongly as discriminative ones; no IDF. Also no
  transliteration normalisation, so `वर्ग` and `varga` never match each other —
  a key reason the cross-lingual queries Q17/Q20 fail under hybrid (keyword
  contribution 0) and only the semantic channel can save them.
- **(G) Elasticsearch path is dead code in practice** — `_try_es` probes
  localhost:9200 on every init and silently falls back. No test covers the ES
  branch.

## 7. `graph/graph_queries.py` + `scripts/module6_knowledge_graph.py`

- **(B) O(E)-per-document lookups.** `_LocalBackend.find_related_shlokas` and
  `find_formulas` scan the full 1,067,200-edge list per call. Hybrid retrieval
  evaluates ~2·top_k candidates per query ⇒ ~5 s/query even on the 48-doc pool.
  The benchmark precomputes counts in one pass (`CachedGraphRelevance`) to work
  around this; production has no such index. A `from → edges` adjacency map
  would fix it in 6 lines.
- **(B) `_LocalBackend.find_by_veda`** returns shloka ids from `BELONGS_TO`
  edges but its Neo4j counterpart returns node records — the two backends have
  incompatible return types behind one façade.
- Schema is flat (veda/domain/branch/source/keyword/formula only). There are no
  *relational* edges between concepts, no temporal/author nodes: the "knowledge
  graph" is a classification metadata store. Claims should say "typed metadata
  graph" — the ablation in the benchmark shows the graph adds no relational
  value today.

## 8. `reasoning/math_engine.py`

- **(B) Substring keyword routing.** `FormulaExtractor._HINT_MAP` tests
  `kw in text`; "भुज" (side) ⊂ "त्रिभुज" (triangle) ⇒ triangle passages return
  *pythagorean*; "क्षेत्रफल" precedes "वृत्त" in dict order ⇒ circle passages
  return *triangle_area*. This single defect drives stage-3 extraction accuracy
  to 0.111 (benchmark F06/F09/F11/F12). Word-boundary matching for Devanagari
  (no spaces between compounds, so: longest-match-first + compound-aware
  segmentation) is the fix.
- **(B) English mathematics invisible to detection.**
  `MATH_KEYWORDS_LATIN` contains transliterations only (`varga`, `kshetra`…);
  every English math item (F02, F04, F05, F07, F08) is missed → detection
  recall 0.500.
- **(B) `क्षेत्र` false positive.** Gita 1.1 `धर्मक्षेत्रे` (dharma-field) triggers
  the Devanagari keyword → N05 is the single FP.
- **(M) Detector logic**: any single trigger ⇒ flag (score ≥ 0.60 threshold is
  vacuous because every rule sets score ≥ 0.60); Rule 6 (any digit) alone flags
  a passage. No abstention, no calibration.
- **(M) `FormulaGenerator` emits generic templates.** 4 of its 5 branches output
  fixed expressions (`a+b`, `a*b`, …) keyed off loose regexes; SymPy-validity
  (1.000) is therefore trivially true and carries no information — the
  benchmark correctly separates "parse valid" (4a) from "equivalent" (4b).
- Registry (`KNOWN_FORMULAS`, `phase3_config.py`) has 7 formulas; the corpus
  carries 4,002 formula edges, but the annotation `formula` field is a free
  string that is normalised by `lower().replace(" ", "_")` before lookup —
  anything outside the 7 keys degrades to `knowledge_graph_raw` with the raw
  string as its own expression/LaTeX (self-validating, no check).

## 9. `reasoning/reasoning_engine.py`

- **(M) Confidence is uncalibrated.** `ConfidenceEstimator` mixes classifier
  confidence, hybrid top-1 score and relation-count density with fixed weights;
  no calibration data exists. The API surfaces it as `confidence`, which
  invites over-reading. No abstention path is exposed despite the trace
  appending "low confidence" text.
- **(M) Formula lookup on the *retrieved top-1* doc**: `reason()` passes
  `top_id` (top retrieved neighbour) to `FormulaExtractor.extract`, not a
  self-id; with the stale-index bug, `top_id` from the semantic channel is
  never in the KG, silently disabling the KG path in production.

## 10. API / cache / metrics (`api/`, `cache/`)

- Sound engineering: lifespan loading, typed schemas with input validation,
  request audit log, Redis→LRU fallback, thread-safe metrics.
- Minor: CORS `allow_origins=["*"]`; `p95` uses nearest-rank on ≤1000 samples;
  cache key lowercases input (harmless for Devanagari).

## 11. Benchmarks (`benchmarks/`) — the evidence base

- Harnesses are reproducible: re-running both today reproduced the shipped
  tables exactly (`tests/test_benchmarks.py`, 20/20 pass).
- **(M) Judgements are AI-generated** (`judged_by: "ai_candidate"`,
  `needs_reviewer: true` on every query and item). This is disclosed
  everywhere; publication requires a Sanskrit-reading reviewer.
- **(M) 48-document pool caps absolute scores**; keyword runs over a
  pool-local index to keep the comparison fair. Methodologically correct but
  must be stated in the manuscript.
- **(G) No significance testing possible** at n=31 queries/21 items; the
  ΔMRR +0.043 graph effect is within noise for this n (a bootstrap or randomisation
  test would formalise this; with 29 scored queries a paired randomisation test
  is computable and should be added before publication).
- **(G) KG-formula path unbenchmarked**: formula items run text-only
  (`kg_engine=None`), so the `HAS_FORMULA` edge path (4,002 edges) has no
  measured accuracy.

## Summary of what the evidence will and will not support

**Supported by measurements:** hybrid > keyword-only on a fixed 48-doc pool
(ΔMRR +0.043, driven by a keyword-richness prior rather than relational graph
reasoning); symbolic parse validity of registry formulas; a reproducible
5-stage decomposition of math-extraction correctness.

**Not supported (do not claim):** corpus-scale retrieval quality; calibrated
confidence or abstention; relational-KG contribution; Phase-2 rescue of
math detection; English-language math extraction; end-to-end formula
interpretation correctness (needs reviewer scoring).

# Where Does the Graph Help? A Staged, Ablation-Based Evaluation Protocol for Mathematical Knowledge Extraction from Classical Sanskrit

**Status:** draft for internal review. All quantitative statements are
generated from `benchmarks/results/*.json` by
`benchmarks/render_manuscript_tables.py`; the numbers cannot drift from the
code. Judgements in both benchmarks are AI-generated candidates awaiting a
Sanskrit-reading reviewer (see *Limitations*).

---

## Abstract

Classical Sanskrit corpora — the Vedas, the Śulbasūtras, and the siddhāntic
astronomy texts — carry a large body of mathematical knowledge expressed in
verse, but no existing work evaluates, quantitatively, a pipeline that
retrieves and formalises that knowledge. We present a staged evaluation
protocol for mathematical knowledge extraction from classical Sanskrit that
(a) fixes a judged query set over a reproducible document pool, (b) compares
keyword, dense and hybrid retrieval with per-signal ablations, and (c)
decomposes mathematical correctness into five independently measured stages:
passage detection, formula retrieval, formula extraction, symbolic validity,
and interpretation. We use the protocol to evaluate a 252k-document
production system whose hybrid retriever fuses dense, keyword and
knowledge-graph signals. On the 48-document evaluation pool the hybrid
retriever reaches MRR 0.877 versus 0.835 for keyword-only; the apparent
knowledge-graph contribution (ΔMRR +0.043) fails a paired randomisation test
(p = 0.063, n = 29) and is shown to be a document-keyword-richness prior —
the graph score is `min(#relations/10, 1)` over a graph whose edges are
uniform classification metadata. In the mathematical pipeline,extraction accuracy is 0.111: substring keyword routing mis-assigns formulas
("भुज" ⊂ "त्रिभुज" routes triangles to the Pythagorean theorem) and half the
expected formulas are absent from the 7-entry registry, while symbolic
parse validity is a vacuous 1.000. We argue that staged measurement plus
per-signal ablation — not end-to-end scores — is the minimum standard for
evaluating knowledge-extraction systems over low-resource ancient-language
corpora, and we release the benchmarks, harnesses and audit as reproducible
artifacts.

---

## 1. Introduction

Mathematical ideas in the Sanskrit tradition are embedded in verse: the
Śulbasūtras state geometric constructions as aphorisms, the Śulbasūtra and
siddhāntic literature encodes arithmetic and algebraic identities in
kārikā-style statements, and even the Ṛgveda contains number lore. Digitised
corpora of this material now exist at scale — our own reference corpus holds
252,226 annotated passages — yet the systems built over them are almost
always evaluated with a single headline number, if at all.

This paper makes an evaluation-methodology contribution, not an architecture
contribution. Our three claims are:

1. **Mathematical extraction from ancient Sanskrit must be measured in
   stages.** A single end-to-end "formula found correctly" score conflates
   at least five distinct capabilities, whose failure modes do not overlap
   (§4.2, Table T6).
2. **Graph signals in hybrid retrievers require ablation, because a metadata
   graph can masquerade as a relevance signal.** In our production system the
   graph component appears to add ΔMRR +0.043; controlled analysis shows it
   encodes only document keyword count, and the effect does not reach
   significance under a paired randomisation test (§4.3, Table T5).
3. **On a fixed judged pool, lexical retrieval beats dense retrieval for
   Sanskrit mathematical text, and hybrid fusion helps only where the two
   channels are complementary** — cross-lingual queries are the one category
   where dense retrieval is indispensable (§4.3, Table T4).

Every number in this draft is rendered mechanically from the benchmark JSON
outputs by `benchmarks/render_manuscript_tables.py --check`; the audit trail
is `reports/audit/implementation_audit.md`.

---

## 2. Related Work

**Sanskrit computational linguistics.** Sandhi, morphology, parsing and
lexicography for Sanskrit are mature (Sandarśana survey, 2025; the Sanskrit
Heritage platform; the first data-driven Vedic-Sanskrit parser, Hellwig
2023). Recent benchmarks target sandhi tools (SandhiKosh) and translation
(Itihasa, 93k shloka pairs). None targets mathematical content. The closest
prior *system* is the Ayurveda semantic-annotation and querying framework of
Terdalkar & Joshi (2023): a Sanskrit-domain knowledge graph, but with
qualitative evaluation, no retrieval experiments and no formalisation.

**Mathematical information retrieval.** Math-IR benchmarks exist for modern
formula-bearing corpora (MIRB, 2025; Formula-Text Cross-Retrieval, 2025) and
consistently find sparse retrieval surprisingly strong and dense retrieval
weak on formula-form text (also Dallas 2018). Our stage 2–3 results replicate
this pattern in an ancient-language setting — and show that when the "formula"
is implicit in verse rather than typeset as LaTeX, keyword heuristics fail in
a qualitatively different way (substring mis-routing, §4.4).

**Knowledge graphs and graph-augmented retrieval.** KG-for-cultural-heritage
is an active, largely qualitative area (Schimmenti et al. 2025; the ROSES
review 2026). Independently of us, controlled evaluations of GraphRAG report
that graph advantages often disappear under fair baselines (arXiv:2502.11371;
NeurIPS 2025 KG-RAG audit). Our ablation adds a *diagnosis*: it identifies
the exact statistic the graph signal computes and shows why it cannot carry
relational information in our graph.

**History of Indian mathematics.** Source-critical framing follows Plofker
(2009) and Srinivasiengar (1967); F01's gloss follows Sen & Bag's Baudhāyana
translation (1983), flagged for verification against a critical edition in the
item provenance.

---

## 3. System Under Evaluation

The system is a four-phase pipeline over a 252,226-document corpus
(Phase 1 ingestion/annotation; Phase 2 transformer classification of
veda/domain/branch; Phase 3 hybrid retrieval + typed metadata graph + math
engine; Phase 4 API). Table T1 gives the inventory; Table T2 the Phase-2
classifier metrics, whose macro-F1 collapse we treat as a measured fact about
the upstream labelling rather than a component we optimise.

The hybrid retriever computes
`final = 0.45·semantic + 0.35·keyword + 0.20·graph`, where the graph score is
`min(len(relations)/10, 1)`. The graph itself has six edge types
(BELONGS_TO, HAS_DOMAIN, HAS_BRANCH, FROM_SOURCE, HAS_KEYWORD, HAS_FORMULA)
giving every document exactly four structural edges plus one per keyword —
this fact, not any experiment, is what first motivates the ablation.

Two shipped-artifact defects are disclosed up front because they bound every
claim below: the production vector index contains only the 48 evaluation
documents (0.019 % corpus coverage) with pre-renumbering IDs (0/48 overlap
with the corpus), and the local graph backend answers relation queries by
scanning all 1.07M edges per call (~5 s per query). The benchmark therefore
rebuilds an index over the same 48 documents with corpus-consistent IDs and
precomputes relation counts once (identical scores, milliseconds instead of
seconds).

---

## 4. Evaluation

### 4.1 Protocol

**Retrieval.** A fixed pool of 48 annotated passages (the five sample
sources: Ṛgveda maṇḍalas 1–2, Atharvaveda sample, a Śulbasūtra sample, and
Vedic-mathematics sutras) is judged against 31 queries in five categories:
lexical (15), cross-lingual English↔Sanskrit (5), conceptual (8),
keyword-field (1), and negative/answerless (2). All methods run over the
identical pool — keyword retrieval over a pool-local inverted index, dense
retrieval over a freshly built 48-vector index — so the comparison is fair;
absolute scores are pool-bound. Metrics: Recall@k, Precision@k (k ∈ {1,3,5,10}),
MRR, MAP; per-query paired randomisation tests (100k permutations, seed 1729).

**Mathematics.** 21 items (16 mathematical, 5 negative) with
provenance-tagged citations and expected symbolic forms. Five stages are
measured independently: detection (all items), formula retrieval, formula
extraction (exact registry key), symbolic parse-validity and equivalence
(SymPy `simplify(a−b)==0`), and interpretation-name match (an automatic
proxy pending reviewer scoring).

### 4.2 Staged mathematical correctness

{{T6}}

{{T7}}

The decomposition is the result. Stage 4a (parse validity, 1.000) is
*information-free*: every template the generator can emit parses. Stage 4b
(equivalence to the expected expression, 0.077) is where correctness lives,
and it is 13× lower. Stage 3 (exact formula key, 0.111) isolates the cause to
the keyword router: of nine scored items, eight fail — four via *deterministic*
mis-routing (F06, F09, F11, F12: three triangle/square passages to
`pythagorean`, one circle passage to `triangle_area`), driven by substring
matching (`"भुじ" ⊂ "त्रिभुज"`) and
dict-order precedence; the other four (F01, F08, F14, F15) fail because the
expected formula is absent from the 7-entry registry and the item falls
through to the generic generator. Detection (recall 0.500) misses every English-language
mathematics passage because the Latin keyword list contains transliterations
only, and produces the single false positive via `धर्मक्षेत्रे` (dharma-field),
where `क्षेत्र` fires. Each of these is a one-line fix; the point of the
staging is that each would have been invisible in an end-to-end number.

### 4.3 Retrieval and the graph ablation

{{T3}}

Hybrid fusion (production weights) is the best variant on every metric
(MRR 0.877), and keyword-only beats semantic-only by a wide margin
(0.835 vs 0.581), replicating the Math-IR literature's lexical finding in a
low-resource setting.

{{T5}}

The graph contribution is ΔMRR +0.043 — and **not statistically significant**
(p = 0.063 under a two-sided paired randomisation test on 29 scored queries).
Three further facts complete the diagnosis:

1. `hybrid_no_graph` is numerically identical to `keyword_only`
   (MRR 0.835) on this pool: with the graph switched off, the semantic
   channel changes no ranking decision because keyword candidates already
   saturate the top ranks.
2. `graph_only` reaches MRR 0.306 — two orders below keyword-only — so the
   graph score does not identify relevant documents by itself.
3. Sweeping the graph weight gives a peak at 0.20 and *degradation* at 0.40
   (MRR 0.815), the signature of a useful-but-miscalibrated prior rather
   than a relevance signal.

Because the graph score is `min((4 + #keywords)/10, 1)` in this graph, the
"graph" effect is a keyword-richness prior: it re-ranks documents that the
keyword channel already favoured. We report it as such. An honest systems
paper would either drop the component or replace the statistic with a
relational one (e.g. two-hop keyword-node expansion); our contribution here
is the measurement that makes the distinction possible.

{{T4}}

Per-category, the picture is sharp: lexical and keyword-field queries are
solved (MRR 1.000) but two of five cross-lingual queries (Q17, Q20 — English
paraphrases of Sanskrit mathematical statements) collapse to MRR 0.125 and
0.071 under the best variant, because the tokenizer-level keyword channel
cannot match `वर्ग` to "square" and fusion dilutes the one channel that can.
Dense-only retrieval, though worse on average, is the only variant that
retrieves Q17/Q20's gold passage in its top-5. A cross-lingual query-detection
gate that routes to semantic-only would fix this at no cost; it is future
work, not a claim.

### 4.4 Error analysis

The stage decomposition plus per-category retrieval analysis yield a
mechanistic error taxonomy:

- **Deterministic router errors** (4/9 stage-3 failures: F06, F09, F11,
  F12): substring keyword matching and dict-order precedence. Fixable
  exactly; no model involved.
- **Registry coverage gaps** (the other 4/9: F01, F08, F14, F15): expected
  formulas outside the 7-entry `KNOWN_FORMULAS` registry degrade to generic
  template generation. A coverage problem, not a matching problem.
- **Lexicon gaps** (7/16 detection misses: F02, F04, F05, F07, F08 plus
  generated-path items with English mathematics): English mathematics is
  invisible to the transliteration-only Latin keyword list. Fixable by adding
  surface forms.
- **Compositional false positive** (1): `धर्मक्षेत्रे` shows that substring
  keywords fire inside non-mathematical compounds; a compound-aware longest-
  match rule is required for Devanagari, which lacks word boundaries inside
  compounds.
- **Cross-lingual channel interference** (2/5 cross-lingual queries): hybrid
  fusion actively harms queries whose evidence exists in only one channel.
- **No abstention anywhere.** Negative queries receive full top-k lists with
  top scores (0.357, 0.329) indistinguishable from weak positive queries; the
  confidence estimator is an uncalibrated weighted sum.

---

## 5. Threats to Validity and Limitations

1. **Judgements are AI-generated candidates** (`needs_reviewer: true` on all
   31 queries and 21 items). Every rate in this paper is therefore
   provisional pending a Sanskrit-reading reviewer; the harness regenerates
   all tables after review.
2. **48-document pool, 29 scored queries, 15 formula items.** Absolute scores
   are pool-bound and not corpus-representative; significance is only
   detectable for large effects (the graph effect sits at p = 0.063).
3. **The corpus is 93.7 % template-expanded synthetic text.** No claim is
   made about retrieval or classification over the synthetic majority; the
   evaluation pool and formula items are the only human-meaningful slices.
4. **Shipped artifacts are defective** (stale 48-vector index, O(E) graph
   lookups). The benchmark works around both without mutating them; the
   system as shipped does not have these workarounds.
5. **The KG formula path is unbenchmarked**: formula items run text-only, and
   only 1.6 % of corpus documents carry HAS_FORMULA edges.
6. **Interpretation correctness is an automatic name-match proxy** (stage 5,
   0.067); final interpretation scoring must be reviewer-judged.
7. Phase-2 macro-F1 (0.31 domain / 0.32 branch) bounds any label-driven
   downstream logic; the math detector was therefore measured text-only.

---

## 6. Conclusion

We contribute an evaluation protocol, not a retriever. The protocol's payoff
is concrete: it turns an apparent success (graph-augmented hybrid retrieval,
ΔMRR +0.043) into a diagnosed non-result (a keyword-richness prior, p = 0.063),
and it localises a 0.111 extraction accuracy to two deterministic string-
matching bugs rather than to any limitation of Sanskrit understanding.
Systems over ancient low-resource corpora should be required to report (i)
staged correctness decompositions, (ii) per-signal ablations with paired
significance tests, and (iii) abstention behaviour — otherwise headline
numbers cannot be distinguished from artifacts of keyword priors and
template expansion.

---

## Artifact Appendix

```
benchmarks/retrieval/queries.json            31 judged queries (provenance-tagged)
benchmarks/retrieval/run_retrieval_eval.py   retrieval harness + ablations + randomisation test
benchmarks/formula/items.json                21 formula items (provenance-tagged)
benchmarks/formula/run_formula_eval.py       5-stage math-correctness harness
benchmarks/render_manuscript_tables.py       renders every table in this draft from JSON
benchmarks/common.py                         shared IR metrics
tests/test_benchmarks.py                     20 unit tests (all pass)
reports/audit/implementation_audit.md        file-by-file claim↔code audit
reports/literature_review.md                 positioned prior work
```

Reproduction: `python benchmarks/retrieval/run_retrieval_eval.py` (~2 min),
`python benchmarks/formula/run_formula_eval.py` (~1 s),
`python -m pytest tests/test_benchmarks.py -q`,
`python benchmarks/render_manuscript_tables.py` (regenerates this manuscript).

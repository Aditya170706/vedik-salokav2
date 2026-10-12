# Focused Literature Review

Purpose: position this system against prior work and identify a defensible
novel contribution. Sources checked 2026-10-10 via web search; each entry
links to the primary source. This is a focused review, not exhaustive — a
submission would need a librarian-assisted expansion and citation of primary
papers where only secondary mentions appear here.

## 1. Sanskrit computational linguistics

| Work | What it does | Relation to this system |
| --- | --- | --- | --- |
| Sandarśana: survey of Sanskrit computational linguistics (Pradeep et al., 2025, ACM CSUR; dl.acm.org/doi/full/10.1145/3729530) | Comprehensive survey: sandhi, samāsa, morphology, WordNet, parsing, NLP tooling for Sanskrit | Establishes that mature *linguistic* tooling exists; none of it addresses mathematical content extraction |
| Huet's Sanskrit Heritage platform (Inria; heritage.inria.fr) | Morphological parser, sandhi splitter, lexicon — the reference infrastructure for Sanskrit NLP | Our pipeline deliberately does *not* use morphological analysis; keyword matching is a weaker but dependency-free baseline |
| Hellwig, data-driven dependency parsing of Vedic Sanskrit (2023) | First data-driven Vedic Sanskrit parser | Shows Vedic register needs its own models; supports our caveat that Modern-Sanskrit tools transfer poorly |
| SanskritWordNet / WordNet resources (cited in Sandarśana) | Structured semantic network for Sanskrit | The closest existing "graph" for Sanskrit; a natural source of relational edges our metadata graph lacks |
| SanskritBERT / IndicBERT / MuRIL (various) | Monolingual & multilingual PLMs covering Sanskrit | We use a generic multilingual sentence encoder instead; cross-lingual queries Q17/Q20 show the cost (MRR 0.07–0.13) |
| Itihasa: 93k Sanskrit-English shloka translation pairs (2026); SandhiKosh benchmark (2026) | Large-scale shloka parallel corpus; sandhi-tool benchmark | Recent benchmarks exist for translation and sandhi — **none for mathematical passage detection or formula extraction** |
| DharmaBench (Buddhist Sanskrit/Tibetan LLM evaluation) | LLM probing on Classical Tibetan/Buddhist Sanskrit | Confirms the trend of task-specific probing of ancient-language texts; again no math task |
| Terdalkar & Joshi, semantic annotation & querying framework for Ayurveda KG (IIT Kanpur, 2023; aclanthology.org/2023.wsc-csdh.11) | Builds a Sanskrit-domain knowledge graph (Ayurveda) with semantic annotation and SPARQL-style querying | **Closest prior system.** Domain-specific KG + querying, but: annotation is manual/semi-automatic, no retrieval evaluation, no math extraction, no ablations |
| Kumar, NLP as Digital Veda (2025) | Positional/conceptual essay | Context only |

## 2. Mathematical information retrieval

| Work | What it does | Relation to this system |
| --- | --- | --- |
| MIRB (Ju et al., 2025; arXiv:2505.15585) | Benchmark for mathematical IR over *modern* math corpora (LaTeX/MathML) for retrieval models | Defines the modern state of the art; operates on formula-embedded documents, not ancient prose where formulas are implicit in words |
| Formula-Text Cross-Retrieval benchmarking (Li et al., 2025, MathNLP @ EMNLP; aclanthology.org/2025.mathnlp-main.9) | BM25 vs dense vs cross-encoder for formula↔text retrieval; reports low Recall/MRR for sparse methods | Directly analogous to our stage 2–3 (text→formula); their finding that sparse retrieval struggles mirrors our keyword mis-routing failures |
| Greff et al., formula retrieval system (2016) + follow-ups; NTCIR MathIR tasks | Equation retrieval via structural/symbolic fingerprints | Pre-neural reference point; our stage-4 symbolic-equivalence check (SymPy `simplify(a−b)==0`) is the lightweight version of their structural matching |
| Dallas, Math IR with a text search engine (MSc thesis, Waterloo 2018) | BM25 adapted for math search | Evidence that lexical baselines remain strong in math IR — consistent with our keyword-only (MRR 0.835) > semantic-only (0.581) result |

## 3. Knowledge graphs in cultural heritage & graph-augmented retrieval

| Work | What it does | Relation to this system |
| --- | --- | --- |
| Schimmenti et al., KG generation from cultural-heritage texts (2025; arXiv:2511.10354) | LLM-driven KG construction pipeline for heritage texts (ATR4CH) | Modern template for what our Phase-1 KG builder does with hand-written rules |
| Zhu et al., KG applications in cultural heritage — ROSES review (2026, Information 17(3):269) | Systematic review of KGs for heritage | Confirms KG-for-heritage is an active area; evaluation is usually qualitative — supports our claim that *quantitative ablation* is rare |
| de Boer et al., KGs for cultural heritage & DH (SUMAC keynote) | Linked-data principles for heritage | Standards our flat JSON graph does not meet (no URIs, no ontological alignment) |
| RAG vs GraphRAG: systematic evaluation (2025; arXiv:2502.11371) | Controlled comparison; many GraphRAG advantages disappear under fair evaluation | Independent support for our central ablation finding: graph signals often fail to beat simpler baselines once controlled |
| Can KG-based RAG be trusted? (NeurIPS 2025 poster) | Audits KG-RAG reliability | Same direction: KG contributions need ablation, not assertion |
| CICHMKG (Fan et al., 2023, Heritage Science) | Large-scale multimodal heritage KG (Chinese) | Scale contrast: our KG is 252k nodes of *typed metadata*, not entity/event knowledge |

## 4. History of Indian mathematics (domain grounding)

- Plofker, *Mathematics in India* (Princeton, 2009) — standard scholarly
  treatment; source-critical caution for Śulbasūtra and Ṛgvedic material.
- Srinivasiengar, *The History of Indian Mathematics* (1967) — classical
  reference.
- Sen & Bag, *Baudhāyana Śulbasūtra* translation (1983) — the translation used
  for our F01 gloss; our `provenance.notes` flags it for checking against a
  critical edition.

## 5. The gap and the defensible contribution

**Gap (supported by the review):** no existing work provides a *quantitatively
evaluated* pipeline that (a) retrieves mathematically relevant passages from a
mixed Vedic/classical Sanskrit corpus with per-signal ablations, and (b)
decomposes mathematical-extraction correctness into independently measured
stages (detection → retrieval → extraction → symbolic validity →
interpretation). Sanskrit NLP has benchmarks for sandhi, translation and
parsing; math IR has benchmarks for modern LaTeX corpora; KG-for-heritage
work is almost entirely qualitative. The intersection — evaluation methodology
for mathematical knowledge extraction from ancient Sanskrit — is empty.

**What is genuinely novel here (and defensible on our evidence):**

1. **A two-sided benchmark artifact**: `benchmarks/retrieval/queries.json`
   (31 judged queries over a fixed 48-doc pool, incl. cross-lingual and
   answerless queries) and `benchmarks/formula/items.json` (21 items with
   provenance-tracked expected symbolic forms). Small, but each item is
   auditable and license/citation-tagged — the opposite of a scraped benchmark.
2. **A staged correctness decomposition** for mathematical extraction from
   ancient texts (5 decoupled stages), which localises failures that a single
   end-to-end score would conflate. Concretely: it separates "SymPy parses
   everything" (1.000) from "SymPy agrees with the expected formula" (0.077).
3. **A controlled ablation showing that the KG signal in a production hybrid
   retriever is a document-richness prior, not relational reasoning**
   (ΔMRR +0.043; graph_only MRR 0.306 ≪ keyword_only 0.835; graph weight
   sweep shows a peak at 0.20 and degradation at 0.40). This replicates, at
   small scale and in a low-resource ancient-language setting, the recent
   GraphRAG-sceptical findings — and diagnoses *why* (the score is
   `len(relations)/10` over a graph whose edges are uniform classification
   metadata).

**Explicitly not novel** (do not claim): the hybrid retriever architecture
itself, the use of sentence transformers for Sanskrit, KG construction for
heritage corpora, symbolic validation with SymPy.

**Positioning sentence for the manuscript:** *We do not propose a new
retrieval architecture; we propose the first quantitative evaluation protocol
— with staged correctness metrics and per-signal ablations — for mathematical
knowledge extraction from classical Sanskrit corpora, and we use it to show
that a graph signal that appears to help is, in fact, a keyword-richness
prior.*

"""
benchmarks/retrieval/run_retrieval_eval.py
──────────────────────────────────────────
Retrieval benchmark + knowledge-graph contribution ablation.

What it does
  1. Fixes an evaluation collection (the 48 annotated sample shlokas).
  2. Runs keyword, semantic (dense), and hybrid retrieval over that identical
     collection, using the *project's own* retriever classes.
  3. Scores every ranking with Recall@k, Precision@k and MRR against the
     reviewer-judged relevance sets in ``queries.json``.
  4. Sweeps signal weights to isolate the contribution of the knowledge-graph
     signal (graph on/off, graph weight 0 → 0.4) and the other two signals.

Fair-comparison note
  The shipped vector index only contains the 48 collection documents, while the
  inverted search index covers the full ~252k-document corpus. To compare
  methods on equal footing, keyword retrieval is run over a *pool-local*
  ``LocalSearchIndex`` built from exactly the 48 documents. Hybrid retrieval
  reuses the production ``HybridRetrievalEngine`` and only its keyword backend
  is swapped for the pool-local one (semantic is already collection-scoped).

Usage
  python benchmarks/retrieval/run_retrieval_eval.py
  python benchmarks/retrieval/run_retrieval_eval.py --rebuild-index
  python benchmarks/retrieval/run_retrieval_eval.py --limit 10   # smoke test
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmarks import common  # noqa: E402
from benchmarks.common import (  # noqa: E402
    K_VALUES, RETRIEVAL_DIR, RESULTS_DIR, SAMPLE_VECTOR_INDEX_DIR,
    SHIPPED_VECTOR_INDEX_DIR, aggregate_metrics, evaluate_ranking,
    load_collection_docs, load_queries, markdown_table, write_json, write_text,
)

POOL_SEARCH_INDEX_DIR = RETRIEVAL_DIR / "search_index_sample"

# variant name -> (semantic_weight, keyword_weight, graph_weight)
VARIANTS: "OrderedDict[str, tuple[float, float, float]]" = OrderedDict([
    ("semantic_only",     (1.00, 0.00, 0.00)),
    ("keyword_only",      (0.00, 1.00, 0.00)),
    ("graph_only",        (0.00, 0.00, 1.00)),
    ("hybrid_full",       (0.45, 0.35, 0.20)),   # production default
    ("hybrid_no_graph",   (0.45, 0.35, 0.00)),   # graph ablation
    ("hybrid_graph_0.10", (0.45, 0.35, 0.10)),
    ("hybrid_graph_0.40", (0.45, 0.35, 0.40)),
    ("hybrid_no_semantic", (0.00, 0.35, 0.20)),
    ("hybrid_no_keyword",  (0.45, 0.00, 0.20)),
])


# ── Collection-scoped keyword backend ────────────────────────────────────────

class PoolKeywordRetriever:
    """Keyword retriever limited to the evaluation collection.

    Implements the same duck-typed interface consumed by
    ``HybridRetrievalEngine.retrieve`` (``ready`` + ``keyword_search``) and
    reuses the production token-overlap scorer
    ``SymbolicSearchEngine._score``.
    """

    def __init__(self, local_index, pool_size: int):
        from retrieval.symbolic_search import SymbolicSearchEngine
        self._local = local_index
        self._pool_size = pool_size
        self._score = SymbolicSearchEngine._score

    @property
    def ready(self) -> bool:
        return True

    def keyword_search(self, query: str, filters: dict = None, top_k: int = 5) -> list[dict]:
        raw = self._local.keyword_search(query, filters=filters, size=self._pool_size)
        return self._score(raw, query)


class CachedGraphRelevance:
    """Drop-in replacement for ``KnowledgeGraphQueryEngine.get_shloka_relations``.

    The local KG backend implements ``find_related_shlokas`` as a full scan over
    ~1.07M edges *per call*, so scoring the 48 candidates of a single query
    costs ~5s. We precompute each collection document's relation count in a
    single pass and expose the same interface. ``HybridRetrievalEngine``
    consumes only ``len(relations)``, so the graph scores are identical while
    the ablation runs in milliseconds.
    """

    def __init__(self, counts: dict[str, int]):
        self._counts = counts

    def get_shloka_relations(self, shloka_id: str) -> list[dict]:
        return [{"rel": "cached"}] * self._counts.get(shloka_id, 0)



def compute_relation_counts(kg, docs: list[dict]) -> dict[str, int]:
    """Count outgoing KG edges per collection document in one pass."""
    backend_kg = getattr(getattr(kg, "_local", None), "_kg", None)
    counts: dict[str, int] = {}
    if backend_kg is None or not getattr(backend_kg, "edges", None):
        for d in docs:  # correct but O(E) per doc
            counts[d["id"]] = len(kg.get_shloka_relations(d["id"]))
        return counts
    wanted = {f"Shloka:{d['id']}": d["id"] for d in docs}
    for edge in backend_kg.edges:
        did = wanted.get(edge.get("from"))
        if did:
            counts[did] = counts.get(did, 0) + 1
    return counts


def aggregate_by_category(per_query: list[dict], k_values: Iterable[int]) -> dict:
    """Aggregate per-query metrics grouped by query category."""
    groups: dict[str, list[dict]] = {}
    for ev in per_query:
        groups.setdefault(ev.get("category") or "uncategorised", []).append(ev)
    return {cat: aggregate_metrics(rows, k_values) for cat, rows in groups.items()}


def paired_mrr_test(
    per_query_a: list[dict],
    per_query_b: list[dict],
    n_permutations: int = 100_000,
    seed: int = 1729,
) -> dict:
    """Paired randomisation (permutation) test on per-query MRR, variant A vs B.

    Under the null hypothesis the two variants are exchangeable per query, so
    we randomly swap the two MRR values within each pair, recompute the mean
    difference, and estimate a two-sided p-value. Queries where either variant
    has no relevant documents (MRR undefined / None) are excluded on both sides
    so the pairing stays intact.

    Deterministic for a given seed.
    """
    import random

    ids_b = {ev["query_id"]: ev for ev in per_query_b}
    diffs: list[float] = []
    for ev in per_query_a:
        other = ids_b.get(ev["query_id"])
        if other is None or ev["mrr"] is None or other["mrr"] is None:
            continue
        diffs.append(ev["mrr"] - other["mrr"])

    n = len(diffs)
    if n == 0:
        return {"n_pairs": 0, "mean_delta": None, "p_value": None, "n_permutations": 0}

    observed = sum(diffs) / n
    rng = random.Random(seed)
    count_extreme = 0
    for _ in range(n_permutations):
        acc = 0.0
        for d in diffs:
            acc += d if rng.random() < 0.5 else -d
        if abs(acc / n) >= abs(observed) - 1e-12:
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_permutations + 1)   # never exactly 0
    return {
        "n_pairs": n,
        "mean_delta": observed,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "seed": seed,
        "test": "two-sided paired randomisation on per-query MRR",
    }


def _quiet_loggers() -> None:
    """Silence the project's INFO logs for readable benchmark output."""
    for lg in list(logging.root.manager.loggerDict.values()):
        if isinstance(lg, logging.Logger):
            lg.setLevel(logging.WARNING)


def build_pool_search_index(docs: list[dict], index_dir: Path = POOL_SEARCH_INDEX_DIR):
    """Build a LocalSearchIndex over exactly the collection documents."""
    from scripts.module7_search_index import LocalSearchIndex

    if index_dir.exists():
        shutil.rmtree(index_dir)
    idx = LocalSearchIndex(index_dir)
    index_docs = [
        {
            "id": d["id"],
            "shloka_text": d["text"],
            "source": d["source"],
            "chapter": d["chapter"],
            "veda": d["veda"],
            "domain": d["domain"],
            "branch": d["branch"],
            "formula": "",
            "language": "Sanskrit",
            "keywords": d["keywords"],
        }
        for d in docs
    ]
    idx.bulk_index(index_docs)
    return idx


# ── Main evaluation ──────────────────────────────────────────────────────────

def validate_queries(queries: list[dict], ids: set[str]) -> None:
    for q in queries:
        for rid in q.get("relevant_ids", []) + q.get("partially_relevant_ids", []):
            if rid not in ids:
                raise ValueError(
                    f"Query {q['id']} references id {rid} which is not in the "
                    f"evaluation collection ({len(ids)} documents)."
                )


def run(limit: int | None = None, rebuild_index: bool = False) -> dict:
    t0 = time.time()
    docs = load_collection_docs()
    ids = {d["id"] for d in docs}
    qdata = load_queries()
    queries = qdata["queries"]
    if limit:
        queries = queries[:limit]
    validate_queries(queries, ids)

    # ── Vector index (benchmark-local, corpus-consistent IDs) ────────────────
    stale_report = common.vector_index_id_overlap(SHIPPED_VECTOR_INDEX_DIR)
    if rebuild_index or not (SAMPLE_VECTOR_INDEX_DIR / "embeddings.npy").exists():
        print("[index] rebuilding benchmark-local 48-doc vector index …")
        common.rebuild_sample_vector_index(index_dir=SAMPLE_VECTOR_INDEX_DIR)

    from retrieval.semantic_search import SemanticSearchEngine
    from retrieval.hybrid_engine import HybridRetrievalEngine
    from graph.graph_queries import KnowledgeGraphQueryEngine

    semantic = SemanticSearchEngine(index_dir=SAMPLE_VECTOR_INDEX_DIR)
    if not semantic.ready:
        raise RuntimeError("Benchmark vector index failed to load.")

    pool_index = build_pool_search_index(docs)
    pool_keyword = PoolKeywordRetriever(pool_index, pool_size=len(docs))

    print("[graph] loading local knowledge graph …")
    kg = KnowledgeGraphQueryEngine(use_neo4j=False)
    rel_counts = compute_relation_counts(kg, docs)
    graph_backend = CachedGraphRelevance(rel_counts)
    _quiet_loggers()

    # Production hybrid engine; swap in the collection-scoped backends.
    engine = HybridRetrievalEngine(kg_engine=graph_backend)
    engine._semantic = semantic
    engine._symbolic = pool_keyword

    # Pre-encode every query once (semantic.search re-encodes otherwise).
    k_values = list(K_VALUES)
    results: dict = {
        "config": {
            "collection": "sample48",
            "num_documents": len(docs),
            "num_queries": len(queries),
            "k_values": k_values,
            "variants": {k: list(v) for k, v in VARIANTS.items()},
            "vector_index": str(SAMPLE_VECTOR_INDEX_DIR),
            "shipped_vector_index_report": stale_report,
            "graph_relation_counts": rel_counts,
            "graph_scoring_note": (
                "Relation counts are precomputed in one pass; the shipped local "
                "backend's find_related_shlokas is O(E) per document (~5s/query on "
                "this pool). Scores are identical because only len(relations) is used."
            ),
            "elapsed_seconds_before_eval": round(time.time() - t0, 2),
        },
        "variants": {},
    }

    query_texts = [q["query"] for q in queries]
    relevant_sets = [set(q.get("relevant_ids", [])) for q in queries]

    for name, (sw, kw, gw) in VARIANTS.items():
        engine.semantic_weight = sw
        engine.keyword_weight = kw
        engine.graph_weight = gw
        engine.kg_engine = graph_backend if gw > 0 else None

        per_query = []
        for q, relevant in zip(queries, relevant_sets):
            ranked = engine.retrieve(q["query"], top_k=len(docs))
            retrieved = [r["id"] for r in ranked]
            ev = evaluate_ranking(retrieved, relevant, k_values)
            ev.update({
                "query_id": q["id"],
                "category": q.get("category"),
                "top5": retrieved[:5],
                "top_score": ranked[0]["final_score"] if ranked else None,
                "query": q["query"],
            })
            per_query.append(ev)

        results["variants"][name] = {
            "weights": {"semantic": sw, "keyword": kw, "graph": gw},
            "aggregate": aggregate_metrics(per_query, k_values),
            "by_category": aggregate_by_category(per_query, k_values),
            "per_query": per_query,
        }
        print(f"  [{name:<18}] MRR="
              f"{results['variants'][name]['aggregate']['mrr']!r}")

    # ── Paired significance tests for the pre-registered comparisons ─────────
    comparisons = {}
    for other in ("hybrid_no_graph", "hybrid_no_semantic", "hybrid_no_keyword",
                  "keyword_only", "semantic_only"):
        if other in results["variants"]:
            comparisons[f"hybrid_full_vs_{other}"] = paired_mrr_test(
                results["variants"]["hybrid_full"]["per_query"],
                results["variants"][other]["per_query"],
            )
    results["comparisons"] = comparisons

    results["config"]["elapsed_seconds"] = round(time.time() - t0, 2)
    return results, qdata, queries


# ── Reporting ────────────────────────────────────────────────────────────────

def render_markdown(results: dict, queries: list[dict]) -> str:
    k_values = results["config"]["k_values"]
    variants = results["variants"]

    scored = variants.get("hybrid_full", {}).get("aggregate", {}).get("num_scored_queries")
    lines = [
        "# Retrieval Benchmark — Results",
        "",
        f"- Collection: **{results['config']['num_documents']} documents** (fixed sample48 pool)",
        f"- Queries: **{results['config']['num_queries']}** "
        f"({scored} with at least one relevant passage; the remaining are negative/answerless and excluded from R/P/MRR)",
        f"- Judgment status: AI-generated candidates — **not yet reviewer-confirmed**",
        f"- Vector index: `{results['config']['vector_index']}`",
        "",
        "> **Coverage limitation.** The semantic index in this run contains only "
        "the 48 collection documents, so every method is compared on that identical "
        "fixed pool. Absolute scores are therefore not representative of retrieval "
        "over the full ~252k-document corpus, where keyword retrieval at least has "
        "full coverage. See `benchmarks/README.md`.",
        "",
        "## 1. Method comparison (weights are semantic / keyword / graph)",
        "",
    ]

    headers = ["Variant", "Weights (s/k/g)"]
    for k in k_values:
        headers += [f"R@{k}", f"P@{k}"]
    headers += ["MRR", "MAP"]
    rows = []
    for name, v in variants.items():
        a = v["aggregate"]
        w = v["weights"]
        row = [name, f"{w['semantic']:.2f}/{w['keyword']:.2f}/{w['graph']:.2f}"]
        for k in k_values:
            row += [a["recall_at_k"][k], a["precision_at_k"][k]]
        row += [a["mrr"], a["map"]]
        rows.append(row)
    lines.append(markdown_table(headers, rows))

    # Graph ablation delta
    lines += ["", "## 2. Knowledge-graph contribution (ablation)", "",
              "Difference of the production hybrid against the same hybrid with "
              "the graph signal removed (`hybrid_full` − `hybrid_no_graph`).", ""]
    full = variants.get("hybrid_full", {}).get("aggregate", {})
    nog = variants.get("hybrid_no_graph", {}).get("aggregate", {})
    if full and nog:
        h = ["Metric"] + [f"@{k}" for k in k_values]
        delta_r = ["ΔRecall@" + str(k) for k in k_values]
        r = ["ΔRecall"] + [(full["recall_at_k"][k] or 0) - (nog["recall_at_k"][k] or 0)
                           for k in k_values]
        p = ["ΔPrecision"] + [(full["precision_at_k"][k] or 0) - (nog["precision_at_k"][k] or 0)
                              for k in k_values]
        lines.append(markdown_table(h, [r, p]))
        d_mrr = (full.get("mrr") or 0) - (nog.get("mrr") or 0)
        lines.append(f"\nΔMRR (full − no-graph) = **{d_mrr:+.4f}**")

    comp = (results.get("comparisons") or {}).get("hybrid_full_vs_hybrid_no_graph")
    if comp and comp.get("p_value") is not None:
        lines.append(
            f"Paired randomisation test on per-query MRR "
            f"(n={comp['n_pairs']} queries, {comp['n_permutations']:,} permutations, "
            f"seed={comp['seed']}): **p = {comp['p_value']:.4f}** "
            f"(two-sided). Treat p > 0.05 as 'not distinguishable from noise' at "
            f"this sample size."
        )

    lines += ["", "## 3. Per-query breakdown (hybrid_full)", ""]
    h = ["Query", "Category", "n_rel", "R@5", "P@5", "RR"]
    rows = []
    for ev in variants.get("hybrid_full", {}).get("per_query", []):
        rows.append([
            ev["query_id"], ev["category"], ev["n_relevant"],
            ev["recall_at_k"].get(5), ev["precision_at_k"].get(5),
            ev["mrr"],
        ])
    lines.append(markdown_table(h, rows))

    # Per-category aggregation across variants
    lines += ["", "## 3b. Per-category aggregation", "",
              "MRR / R@5 per query category for every variant. Categories are "
              "defined in `queries.json` (lexical, cross-lingual, conceptual, "
              "keyword-field, negative).", ""]
    cats: list[str] = []
    for v in variants.values():
        for c in v.get("by_category", {}):
            if c not in cats:
                cats.append(c)
    h = ["Variant"] + [f"{c}: MRR / R@5" for c in cats]
    rows = []
    for name, v in variants.items():
        row = [name]
        for c in cats:
            ca = v.get("by_category", {}).get(c)
            if not ca or ca["mrr"] is None:
                row.append("—")
            else:
                row.append(f"{ca['mrr']:.3f} / "
                           f"{(ca['recall_at_k'].get(5) or 0):.3f}")
        rows.append(row)
    lines.append(markdown_table(h, rows))

    lines += ["", "## 4. Negative-query behaviour (hybrid_full)", "",
              "Answerless queries: a well-behaved system should return "
              "low-confidence results. This reports the top score it assigns.", ""]
    neg = [ev for ev in variants.get("hybrid_full", {}).get("per_query", [])
           if ev["n_relevant"] == 0]
    if neg:
        lines.append(markdown_table(
            ["Query", "Query text", "Top score", "Top-1 id"],
            [[ev["query_id"], (ev.get("query") or "")[:40], ev.get("top_score"),
              (ev["top5"][0] if ev["top5"] else None)] for ev in neg],
        ))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval benchmark + graph ablation")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only first N queries")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="force rebuild of the benchmark-local vector index")
    args = parser.parse_args()

    results, qdata, queries = run(limit=args.limit, rebuild_index=args.rebuild_index)

    write_json(RESULTS_DIR / "retrieval_metrics.json", results)
    write_text(RESULTS_DIR / "retrieval_metrics.md", render_markdown(results, queries))

    print("\nWrote:")
    print(f"  {RESULTS_DIR / 'retrieval_metrics.json'}")
    print(f"  {RESULTS_DIR / 'retrieval_metrics.md'}")

    stale = results["config"]["shipped_vector_index_report"]
    if stale.get("stale"):
        print(f"\n⚠ Shipped vector index is STALE: {stale['overlap']}/"
              f"{stale['num_vectors']} IDs overlap the current corpus. "
              f"See benchmarks/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

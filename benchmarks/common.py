"""
benchmarks/common.py
────────────────────
Shared utilities for the research benchmark suite.

Contents
  • Paths and the fixed evaluation collection definition
  • ``load_collection_docs``      — the fixed 48-document evaluation pool
  • ``load_queries`` / ``load_formula_items``
  • IR metrics: recall@k, precision@k, reciprocal rank, MRR
  • ``rebuild_sample_vector_index`` — rebuild the *same* 48-doc vector index
       with corpus-consistent IDs (fixes the stale-ID bug, see README)
  • Markdown / JSON result writers

The evaluation collection is deliberately fixed. Semantic retrieval can only
return documents that are present in the vector index, so every retrieval
method is scored on an identical collection with identical relevance
judgements. This is the only way a keyword-vs-semantic-vs-hybrid comparison is
methodologically valid.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = PROJECT_ROOT / "benchmarks"
RESULTS_DIR = BENCH_DIR / "results"
RETRIEVAL_DIR = BENCH_DIR / "retrieval"
FORMULA_DIR = BENCH_DIR / "formula"

ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
SHIPPED_VECTOR_INDEX_DIR = PROJECT_ROOT / "database" / "vector_index"
SAMPLE_VECTOR_INDEX_DIR = RETRIEVAL_DIR / "vector_index_sample"

# The five annotated files that were used to build the shipped 48-doc vector
# index (verified: 7 + 10 + 6 + 5 + 20 == 48 documents).
SAMPLE_SOURCES = [
    "atharvaveda_sample",
    "rigveda_mandala1",
    "rigveda_mandala2",
    "sulbasutras_sample",
    "vedic_mathematics_sutras",
]

# Evaluation cut-offs. The collection has 48 documents, so k<=10 keeps
# precision@k informative.
K_VALUES = [1, 3, 5, 10]


# ── Collection loading ───────────────────────────────────────────────────────

def load_collection_docs(
    sources: Iterable[str] = SAMPLE_SOURCES,
    annotated_dir: Path = ANNOTATED_DIR,
) -> list[dict]:
    """Load the fixed evaluation collection as index-style documents.

    Returns a list of dicts with a flat schema shared by every retriever:
        id, text, shloka_text, source, chapter, veda, domain, branch, keywords
    Order matches ``sorted(glob('*.json'))`` used by the project's index
    builders, so it is reproducible.
    """
    docs: list[dict] = []
    for name in sources:
        path = Path(annotated_dir) / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing collection source: {path}")
        with open(path, encoding="utf-8") as fh:
            shlokas = json.load(fh)
        for s in shlokas:
            ann = s.get("annotation", {}) or {}
            keywords = ann.get("keywords", []) or []
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split("|") if k.strip()]
            docs.append({
                "id":          s["id"],
                "text":        s.get("text", ""),
                "shloka_text": s.get("text", ""),
                "source":      s.get("source", name),
                "chapter":     s.get("chapter", ""),
                "veda":        ann.get("veda", "Unknown"),
                "domain":      ann.get("domain", "Unknown"),
                "branch":      ann.get("branch", "Unknown"),
                "keywords":    keywords,
            })
    return docs


def collection_ids(docs: Optional[list[dict]] = None) -> set[str]:
    return {d["id"] for d in (docs if docs is not None else load_collection_docs())}


# ── Query / item loading ─────────────────────────────────────────────────────

def load_queries(path: Optional[Path] = None) -> dict:
    path = Path(path or RETRIEVAL_DIR / "queries.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_formula_items(path: Optional[Path] = None) -> dict:
    path = Path(path or FORMULA_DIR / "items.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── IR metrics ───────────────────────────────────────────────────────────────

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> Optional[float]:
    """Fraction of relevant documents found in the top-k.

    Returns ``None`` when the query has no relevant documents (undefined),
    so callers can skip it rather than counting a spurious 0.
    """
    if not relevant:
        return None
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> Optional[float]:
    """Fraction of the top-k that is relevant (denominator is k)."""
    if not relevant:
        return None
    if k <= 0:
        return None
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> Optional[float]:
    """1 / rank of the first relevant document (0.0 if none retrieved)."""
    if not relevant:
        return None
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / idx
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> Optional[float]:
    """Mean of precision@i over all relevant retrieved documents."""
    if not relevant:
        return None
    hits = 0
    score = 0.0
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / idx
    return score / len(relevant)


def evaluate_ranking(
    retrieved: list[str],
    relevant: set[str],
    k_values: Iterable[int] = K_VALUES,
) -> dict:
    """Compute every metric for one ranking. Undefined metrics are ``None``."""
    return {
        "n_relevant":    len(relevant),
        "n_retrieved":   len(retrieved),
        "recall_at_k":   {k: recall_at_k(retrieved, relevant, k) for k in k_values},
        "precision_at_k": {k: precision_at_k(retrieved, relevant, k) for k in k_values},
        "mrr":           reciprocal_rank(retrieved, relevant),
        "ap":            average_precision(retrieved, relevant),
    }


def mean_defined(values: Iterable[Optional[float]]) -> Optional[float]:
    """Mean over the non-None values, or None if all are None."""
    kept = [v for v in values if v is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)


def aggregate_metrics(per_query: list[dict], k_values: Iterable[int] = K_VALUES) -> dict:
    """Aggregate per-query metric dicts into corpus-level means."""
    k_values = list(k_values)
    agg = {
        "num_queries": len(per_query),
        "num_scored_queries": sum(1 for q in per_query if q["n_relevant"] > 0),
        "recall_at_k":    {k: mean_defined(q["recall_at_k"][k] for q in per_query) for k in k_values},
        "precision_at_k": {k: mean_defined(q["precision_at_k"][k] for q in per_query) for k in k_values},
        "mrr":            mean_defined(q["mrr"] for q in per_query),
        "map":            mean_defined(q["ap"] for q in per_query),
    }
    return agg


def round_nested(obj, ndigits: int = 4):
    """Recursively round floats for stable, diff-friendly JSON output."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_nested(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_nested(v, ndigits) for v in obj]
    return obj


# ── Vector-index rebuild (stale-ID repair) ───────────────────────────────────

def rebuild_sample_vector_index(
    sources: Iterable[str] = SAMPLE_SOURCES,
    annotated_dir: Path = ANNOTATED_DIR,
    index_dir: Path = SAMPLE_VECTOR_INDEX_DIR,
    model_name: Optional[str] = None,
) -> dict:
    """Rebuild the 48-doc vector index from the *current* annotated files.

    The shipped ``database/vector_index/`` was built before the corpus was
    renumbered: its 48 metadata IDs (``SHLOKA_000001`` … ``SHLOKA_000048``)
    have **zero** overlap with the corpus IDs (``SHLOKA_002266`` …). Ranking
    still works because the stored text is correct, but returned IDs cannot be
    joined back to the corpus. This helper produces an equivalent index with
    corpus-consistent IDs so that retrieval metrics are well defined.

    The output schema matches ``retrieval.semantic_search.build_vector_index``
    exactly (embeddings.npy, metadata.json, index_config.json) and the document
    order is identical, so the two indexes are directly comparable.
    """
    import numpy as np

    if model_name is None:
        from phase3_config import EMBEDDING_MODEL as model_name  # noqa: N806

    docs = load_collection_docs(sources, annotated_dir)
    texts = [d["text"] for d in docs]
    metadata = [
        {
            "id":       d["id"],
            "text":     d["text"],
            "source":   d["source"],
            "chapter":  d["chapter"],
            "veda":     d["veda"],
            "domain":   d["domain"],
            "branch":   d["branch"],
            "keywords": d["keywords"],
        }
        for d in docs
    ]

    from retrieval.semantic_search import load_embedding_model
    model = load_embedding_model(model_name)
    embeddings = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", embeddings)
    with open(index_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    cfg = {
        "model_name":  model_name,
        "dim":         int(embeddings.shape[1]),
        "num_vectors": len(embeddings),
        "built_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note":        "benchmark-local rebuild of the fixed 48-document collection",
    }
    with open(index_dir / "index_config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    return cfg


def vector_index_id_overlap(index_dir: Path = SHIPPED_VECTOR_INDEX_DIR) -> dict:
    """Report the ID overlap between a vector index and the current collection."""
    index_dir = Path(index_dir)
    meta_path = index_dir / "metadata.json"
    if not meta_path.exists():
        return {"index_dir": str(index_dir), "exists": False}
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    vids = {m["id"] for m in meta}
    cids = collection_ids()
    return {
        "index_dir":    str(index_dir),
        "exists":       True,
        "num_vectors":  len(meta),
        "num_collection": len(cids),
        "overlap":      len(vids & cids),
        "stale":        len(vids & cids) == 0,
    }


# ── Output writers ───────────────────────────────────────────────────────────

def write_json(path: Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(round_nested(payload), fh, ensure_ascii=False, indent=2)
    return path


def write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def markdown_table(headers: list[str], rows: list[list]) -> str:
    """Render a GitHub-flavoured markdown table."""
    def cell(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)

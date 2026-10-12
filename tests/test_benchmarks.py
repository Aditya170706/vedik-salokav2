"""
tests/test_benchmarks.py
────────────────────────
Unit tests for the research benchmark suite.

These tests are fast and require no model downloads or external services:
they cover the metric implementations, the integrity of the fixed evaluation
collection and the judgement files, and the pure helper functions of the
formula harness.

The full retrieval run (which needs the SentenceTransformer model) is not
exercised here; it is run explicitly via
``python benchmarks/retrieval/run_retrieval_eval.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks import common                                   # noqa: E402
from benchmarks.common import (                                 # noqa: E402
    K_VALUES, aggregate_metrics, evaluate_ranking, load_collection_docs,
    load_formula_items, load_queries, precision_at_k, recall_at_k,
    reciprocal_rank, vector_index_id_overlap,
)
from benchmarks.formula.run_formula_eval import (               # noqa: E402
    name_matches, resolve_system_key, stage1_detection,
    stage2_to_5_formula, symbolic_checks,
)
from benchmarks.retrieval.run_retrieval_eval import (           # noqa: E402
    VARIANTS, PoolKeywordRetriever, aggregate_by_category,
    build_pool_search_index, paired_mrr_test, validate_queries,
)


# ── Metric correctness ───────────────────────────────────────────────────────

class TestMetrics:
    def test_recall_at_k_basic(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"b", "d", "z"}
        assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)
        assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)
        assert recall_at_k(retrieved, relevant, 10) == pytest.approx(2 / 3)

    def test_precision_at_k_denominator_is_k(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 4) == pytest.approx(0.5)
        assert precision_at_k(retrieved, relevant, 2) == pytest.approx(1.0)

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)
        assert reciprocal_rank(["a"], {"a"}) == pytest.approx(1.0)
        assert reciprocal_rank(["x", "y"], {"a"}) == pytest.approx(0.0)

    def test_undefined_for_empty_relevant(self):
        assert recall_at_k(["a"], set(), 1) is None
        assert precision_at_k(["a"], set(), 1) is None
        assert reciprocal_rank(["a"], set()) is None

    def test_evaluate_ranking_shape(self):
        ev = evaluate_ranking(["a", "b"], {"a"}, k_values=[1, 3])
        assert ev["recall_at_k"][1] == pytest.approx(1.0)
        assert ev["precision_at_k"][3] == pytest.approx(1 / 3)
        assert ev["mrr"] == pytest.approx(1.0)

    def test_aggregate_excludes_undefined(self):
        # one defined query, one undefined (negative) query
        q1 = evaluate_ranking(["a"], {"a"})
        q2 = evaluate_ranking(["a"], set())
        agg = aggregate_metrics([q1, q2])
        assert agg["num_queries"] == 2
        assert agg["num_scored_queries"] == 1
        assert agg["mrr"] == pytest.approx(1.0)   # not dragged to 0.5


# ── Collection / judgement integrity ─────────────────────────────────────────

class TestDataIntegrity:
    def test_collection_is_48_docs(self):
        docs = load_collection_docs()
        assert len(docs) == 48
        ids = {d["id"] for d in docs}
        assert len(ids) == 48
        for d in docs:
            assert d["text"]
            assert d["shloka_text"] == d["text"]

    def test_all_query_relevance_ids_exist(self):
        docs = load_collection_docs()
        queries = load_queries()["queries"]
        # raises if any relevant id is outside the collection
        validate_queries(queries, {d["id"] for d in docs})

    def test_query_ids_unique(self):
        queries = load_queries()["queries"]
        ids = [q["id"] for q in queries]
        assert len(ids) == len(set(ids))

    def test_shipped_vector_index_is_detected_as_stale(self):
        # documents the known data-consistency bug; update if the index is rebuilt
        report = vector_index_id_overlap()
        assert report["exists"] is True
        assert report["num_vectors"] == 48
        assert report["overlap"] == 0
        assert report["stale"] is True

    def test_variants_include_graph_ablation_pair(self):
        assert "hybrid_full" in VARIANTS
        assert "hybrid_no_graph" in VARIANTS
        s, k, g = VARIANTS["hybrid_full"]
        assert (s, k, g) == (0.45, 0.35, 0.20)
        assert VARIANTS["hybrid_no_graph"][2] == 0.0


# ── Formula item schema ──────────────────────────────────────────────────────

class TestFormulaItems:
    def test_schema_complete(self):
        items = load_formula_items()["items"]
        assert len(items) >= 15
        seen = set()
        for it in items:
            assert it["id"] not in seen
            seen.add(it["id"])
            assert isinstance(it["is_mathematical"], bool)
            exp = it["expected"]
            for field in ("formula_key", "system_key", "expr", "latex",
                          "variables", "name", "interpretation"):
                assert field in exp, f"{it['id']} missing expected.{field}"
            assert it["provenance"]["level"] in (
                "verified_secondary", "project_corpus", "needs_review")
            assert it["source"]["citation"]

    def test_system_keys_exist_in_registry(self):
        from phase3_config import KNOWN_FORMULAS
        for it in load_formula_items()["items"]:
            sys_key = it["expected"]["system_key"]
            if sys_key is not None:
                assert sys_key in KNOWN_FORMULAS, \
                    f"{it['id']} references unknown registry key {sys_key}"

    def test_has_negative_and_external_items(self):
        items = load_formula_items()["items"]
        assert any(not it["is_mathematical"] for it in items)
        assert any(it["provenance"]["level"] == "verified_secondary" for it in items)


# ── Formula harness helpers ──────────────────────────────────────────────────

class TestFormulaHelpers:
    def test_resolve_system_key(self):
        from phase3_config import KNOWN_FORMULAS
        formula = dict(KNOWN_FORMULAS["circle_area"])
        assert resolve_system_key(formula) == "circle_area"
        assert resolve_system_key(None) is None
        assert resolve_system_key({"expr": "zzz", "name": "nope"}) is None

    def test_symbolic_equivalence(self):
        assert symbolic_checks("a**2 + a**2", "2*a**2") == (True, True)
        assert symbolic_checks("a**2", "b**2")[1] is False
        assert symbolic_checks("not valid ((", None)[0] is False
        assert symbolic_checks("a**2", None) == (True, None)

    def test_name_matches(self):
        assert name_matches("Area of circle", "Area of circle") is True
        assert name_matches("Area of a Circle", "Area of circle") is True
        assert name_matches("", "Area of circle") is False
        assert name_matches("x", None) is None

    def test_stage2_5_matches_known_formula(self):
        items = [{
            "id": "T1",
            "query_text": "the area of a circle is pi times r squared",
            "is_mathematical": True,
            "expected": {
                "formula_key": "circle_area", "system_key": "circle_area",
                "expr": "pi*r**2", "latex": "", "variables": {},
                "name": "Area of circle", "interpretation": "",
            },
        }]
        rows, metrics = stage2_to_5_formula(items)
        assert rows[0]["retrieved_key"] == "circle_area"
        assert metrics["extraction"]["accuracy"] == pytest.approx(1.0)
        assert metrics["symbolic"]["equivalent_rate"] == pytest.approx(1.0)

    def test_stage1_detection_flags_math_keyword(self):
        items = [
            {"id": "M", "query_text": "क्षेत्रफलं गणितम्", "is_mathematical": True},
            {"id": "X", "query_text": "अग्निमीळे पुरोहितम्", "is_mathematical": False},
        ]
        rows, metrics = stage1_detection(items)
        by_id = {r["id"]: r for r in rows}
        assert by_id["M"]["pred"] is True
        assert by_id["X"]["pred"] is False
        assert metrics["accuracy"] == pytest.approx(1.0)


# ── Pool keyword retriever (collection-scoped) ───────────────────────────────

class TestPoolKeywordRetriever:
    def test_keyword_search_is_collection_scoped(self, tmp_path):
        docs = load_collection_docs()
        idx = build_pool_search_index(docs, index_dir=tmp_path / "idx")
        retriever = PoolKeywordRetriever(idx, pool_size=len(docs))
        assert retriever.ready
        results = retriever.keyword_search("क्षेत्रफलं", top_k=10)
        ids = {r["id"] for r in results}
        assert ids == {"SHLOKA_250749", "SHLOKA_250750",
                       "SHLOKA_250751", "SHLOKA_250752"}
        assert all("keyword_score" in r for r in results)


# ── Category aggregation & paired randomisation test ────────────────────────

class TestAnalysis:
    def _ev(self, qid, category, mrr):
        if mrr is None:
            ev = evaluate_ranking(["x"], set())   # negative/answerless query
        else:
            ev = evaluate_ranking(["a"], {"a"})
            ev["mrr"] = mrr                        # synthetic value
        ev.update({"query_id": qid, "category": category})
        return ev

    def test_aggregate_by_category_groups(self):
        per_query = [
            self._ev("Q1", "lexical", 1.0),
            self._ev("Q2", "lexical", 0.5),
            self._ev("Q3", "cross-lingual", 0.25),
            self._ev("Q4", "negative", None),
        ]
        cats = aggregate_by_category(per_query, K_VALUES)
        assert cats["lexical"]["mrr"] == pytest.approx(0.75)
        assert cats["lexical"]["num_queries"] == 2
        assert cats["cross-lingual"]["mrr"] == pytest.approx(0.25)
        # negative queries: MRR undefined, not zero
        assert cats["negative"]["mrr"] is None

    def test_paired_test_identical_variants_not_significant(self):
        a = [self._ev(f"Q{i}", "lexical", 0.8) for i in range(10)]
        b = [self._ev(f"Q{i}", "lexical", 0.8) for i in range(10)]
        res = paired_mrr_test(a, b, n_permutations=1000)
        assert res["mean_delta"] == pytest.approx(0.0)
        assert res["p_value"] > 0.5

    def test_paired_test_large_effect_is_significant(self):
        a = [self._ev(f"Q{i}", "lexical", 1.0) for i in range(20)]
        b = [self._ev(f"Q{i}", "lexical", 0.1) for i in range(20)]
        res = paired_mrr_test(a, b, n_permutations=1000)
        assert res["mean_delta"] == pytest.approx(0.9)
        assert res["p_value"] < 0.01

    def test_paired_test_deterministic(self):
        a = [self._ev("Q1", "lexical", 1.0), self._ev("Q2", "lexical", 0.2)]
        b = [self._ev("Q1", "lexical", 0.4), self._ev("Q2", "lexical", 0.9)]
        r1 = paired_mrr_test(a, b, n_permutations=5000)
        r2 = paired_mrr_test(a, b, n_permutations=5000)
        assert r1 == r2

    def test_paired_test_skips_unpaired_undefined(self):
        a = [self._ev("Q1", "lexical", 1.0), self._ev("Q2", "negative", None)]
        b = [self._ev("Q1", "lexical", 0.5), self._ev("Q2", "negative", None)]
        res = paired_mrr_test(a, b)
        assert res["n_pairs"] == 1
        assert res["mean_delta"] == pytest.approx(0.5)


# ── Manuscript rendering (tables come from JSON, never hand-typed) ──────────

class TestManuscriptRender:
    def test_renderer_outputs_exist_and_check_passes(self):
        import importlib
        mod = importlib.import_module("benchmarks.render_manuscript_tables")
        tables = mod.build_tables()
        assert set(tables) == {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}
        # every table must be non-empty markdown
        for key, body in tables.items():
            assert body.startswith("|"), f"{key} is not a table"

    def test_rendered_manuscript_is_fresh(self):
        """reports/manuscript.md must match what current results would render."""
        import importlib
        mod = importlib.import_module("benchmarks.render_manuscript_tables")
        template = mod.TEMPLATE_PATH.read_text(encoding="utf-8")
        rendered = mod.render(template, mod.build_tables())
        current = mod.MANUSCRIPT_PATH.read_text(encoding="utf-8")
        assert rendered == current, \
            "reports/manuscript.md is stale; run benchmarks/render_manuscript_tables.py"

    def test_no_unsubstituted_placeholders(self):
        text = (ROOT / "reports" / "manuscript.md").read_text(encoding="utf-8")
        assert "{{" not in text

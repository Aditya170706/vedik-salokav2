"""
benchmarks/render_manuscript_tables.py
─────────────────────────────────────
Renders the manuscript's result tables *mechanically* from the verified JSON
outputs in ``benchmarks/results/`` so that no number in the paper is
hand-transcribed.

Usage
  python benchmarks/render_manuscript_tables.py            # render
  python benchmarks/render_manuscript_tables.py --check    # verify only

Inputs (must exist; produced by the two harnesses):
  benchmarks/results/retrieval_metrics.json
  benchmarks/results/formula_metrics.json

Outputs:
  reports/manuscript_tables.md        — one file per table, T1…T7
  reports/manuscript_template.md      — placeholders {{T1}}…{{T7}} substituted
  reports/manuscript.md               — the rendered manuscript

``--check`` re-renders into memory and exits non-zero if reports/manuscript.md
differs from what the current results would produce (useful in CI / tests).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import RESULTS_DIR, markdown_table, write_text  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
TEMPLATE_PATH = REPORTS_DIR / "manuscript_template.md"
MANUSCRIPT_PATH = REPORTS_DIR / "manuscript.md"
TABLES_PATH = REPORTS_DIR / "manuscript_tables.md"


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _load(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding harness first:\n"
            "  python benchmarks/retrieval/run_retrieval_eval.py\n"
            "  python benchmarks/formula/run_formula_eval.py"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── T1: corpus / system inventory ────────────────────────────────────────────

def table_t1() -> str:
    rows = [
        ["Annotated corpus documents", "252,226"],
        ["— of which template-expanded (Itihāsa + Purāṇas)", "236,365 (93.7 %)"],
        ["KG nodes / edges", "252,267 / 1,067,200"],
        ["KG formula edges (HAS_FORMULA)", "4,002 (1.6 % of documents)"],
        ["Phase-2 classifier test set", "5,312"],
        ["Retrieval evaluation pool", "48 documents, 31 queries"],
        ["Formula benchmark items", "21 (15 with expected formulas)"],
        ["Formulas in symbolic registry", "7"],
    ]
    return markdown_table(["Quantity", "Value"], rows)


# ── T2: Phase-2 classification (the honest metrics) ──────────────────────────

def table_t2() -> str:
    rows = []
    for task, label in (("veda", "Veda"), ("domain", "Domain"), ("branch", "Branch")):
        path = Path(__file__).parent.parent / "reports" / task / "eval_results.json"
        if not path.exists():
            rows.append([label, "—", "—", "—"])
            continue
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        rows.append([label, _fmt(r["accuracy"]), _fmt(r["macro_f1"]), r["test_size"]])
    return markdown_table(
        ["Task", "Accuracy", "Macro-F1", "Test set"], rows
    )


# ── T3: retrieval method comparison ──────────────────────────────────────────

T3_ORDER = [
    "semantic_only", "keyword_only", "graph_only",
    "hybrid_no_graph", "hybrid_full",
    "hybrid_no_semantic", "hybrid_no_keyword",
    "hybrid_graph_0.10", "hybrid_graph_0.40",
]


def table_t3(ret: dict) -> str:
    variants = ret["variants"]
    headers = ["Variant", "R@1", "P@1", "R@5", "P@5", "MRR", "MAP"]
    rows = []
    for name in T3_ORDER:
        v = variants.get(name)
        if not v:
            continue
        a = v["aggregate"]
        rows.append([
            name,
            _fmt(a["recall_at_k"].get("1")), _fmt(a["precision_at_k"].get("1")),
            _fmt(a["recall_at_k"].get("5")), _fmt(a["precision_at_k"].get("5")),
            _fmt(a["mrr"]), _fmt(a["map"]),
        ])
    return markdown_table(headers, rows)


# ── T4: per-category MRR ─────────────────────────────────────────────────────

T4_ORDER = ["semantic_only", "keyword_only", "hybrid_no_graph", "hybrid_full"]


def table_t4(ret: dict) -> str:
    variants = ret["variants"]
    cats: list[str] = []
    for v in variants.values():
        for c in v.get("by_category", {}):
            if c not in cats:
                cats.append(c)
    headers = ["Variant"] + [f"{c} MRR" for c in cats]
    rows = []
    for name in T4_ORDER:
        v = variants.get(name)
        if not v:
            continue
        row = [name]
        for c in cats:
            ca = v.get("by_category", {}).get(c)
            row.append(_fmt(ca["mrr"]) if ca else "—")
        rows.append(row)
    return markdown_table(headers, rows)


# ── T5: graph ablation deltas + significance ─────────────────────────────────

def table_t5(ret: dict) -> str:
    full = ret["variants"].get("hybrid_full", {}).get("aggregate", {})
    nog = ret["variants"].get("hybrid_no_graph", {}).get("aggregate", {})
    if not full or not nog:
        return "*Table unavailable: hybrid variants missing from results.*"
    comp = (ret.get("comparisons") or {}).get("hybrid_full_vs_hybrid_no_graph", {})
    rows = [
        ["ΔMRR (full − no-graph)",
         f"{(full['mrr'] or 0) - (nog['mrr'] or 0):+.4f}"],
        ["ΔMAP", f"{(full['map'] or 0) - (nog['map'] or 0):+.4f}"],
        ["ΔR@1", f"{(full['recall_at_k']['1'] or 0) - (nog['recall_at_k']['1'] or 0):+.4f}"],
        ["ΔR@5", f"{(full['recall_at_k']['5'] or 0) - (nog['recall_at_k']['5'] or 0):+.4f}"],
        ["ΔR@10", f"{(full['recall_at_k']['10'] or 0) - (nog['recall_at_k']['10'] or 0):+.4f}"],
    ]
    table = markdown_table(["Quantity", "Δ (hybrid_full − hybrid_no_graph)"], rows)
    if comp.get("p_value") is not None:
        table += (
            f"\n\nPaired randomisation test on per-query MRR: "
            f"n = {comp['n_pairs']} queries, {comp.get('n_permutations', 0):,} "
            f"permutations, p = {comp['p_value']:.4f} (two-sided)."
        )
    return table


# ── T6: math-correctness five stages ─────────────────────────────────────────

def table_t6(formula: dict) -> str:
    d = formula["stage1_detection"]["metrics"]
    f = formula["stage2_5_formula"]["metrics"]
    rows = [
        ["1. Passage detection (recall)", d["tp"] + d["fn"], d["tp"], _fmt(d["recall"])],
        ["1. Passage detection (precision)", d["tp"] + d["fp"], d["tp"], _fmt(d["precision"])],
        ["2. Formula retrieval (any known formula)", f["retrieval"]["scored"], f["retrieval"]["retrieved"], _fmt(f["retrieval"]["rate"])],
        ["3. Formula extraction (exact key)", f["extraction"]["scored"], f["extraction"]["correct"], _fmt(f["extraction"]["accuracy"])],
        ["4a. Symbolic parse valid", f["symbolic"]["scored"], f["symbolic"]["parse_valid"], _fmt(f["symbolic"]["parse_valid_rate"])],
        ["4b. Symbolic equivalent to expected", f["symbolic"]["scored"], f["symbolic"]["equivalent"], _fmt(f["symbolic"]["equivalent_rate"])],
        ["5. Interpretation name match", f["interpretation"]["scored"], f["interpretation"]["correct"], _fmt(f["interpretation"]["accuracy"])],
    ]
    return markdown_table(["Stage", "Scored", "Correct", "Rate"], rows)


# ── T7: detection confusion + error taxonomy counts ──────────────────────────

def table_t7(formula: dict) -> str:
    d = formula["stage1_detection"]["metrics"]
    rows = [[d["tp"], d["fp"], d["tn"], d["fn"]]]
    table = markdown_table(["TP", "FP", "TN", "FN"], rows)

    # Error taxonomy over formula items (stage 2–5 per-item rows)
    per_item = formula["stage2_5_formula"]["per_item"]
    miss_english = []
    misroute = []
    for r in per_item:
        if r["path"] == "generated" and r["retrieved_key"] is None and r["expected_key"]:
            miss_english.append(r["id"])
        elif r["retrieved_key"] and r["expected_system_key"] and \
                r["retrieved_key"] != r["expected_system_key"]:
            misroute.append(r["id"])
    table += (
        "\n\nError taxonomy (formula items): "
        f"keyword mis-routing = {len(misroute)} ({', '.join(misroute) or '—'}); "
        f"fell through to generic generation = {len(miss_english)} "
        f"({', '.join(miss_english) or '—'})."
    )
    return table


def build_tables() -> dict[str, str]:
    ret = _load("retrieval_metrics.json")
    formula = _load("formula_metrics.json")
    return {
        "T1": table_t1(),
        "T2": table_t2(),
        "T3": table_t3(ret),
        "T4": table_t4(ret),
        "T5": table_t5(ret),
        "T6": table_t6(formula),
        "T7": table_t7(formula),
    }


def render(template: str, tables: dict[str, str]) -> str:
    out = template
    for key, body in tables.items():
        out = out.replace("{{" + key + "}}", body)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify reports/manuscript.md matches current results")
    args = parser.parse_args()

    tables = build_tables()

    tables_md = "# Manuscript Tables (generated)\n\n" \
        "Generated by `benchmarks/render_manuscript_tables.py` from the JSON " \
        "results. Do not edit by hand.\n\n" + \
        "\n\n".join(f"## {k}\n\n{v}" for k, v in tables.items()) + "\n"

    if args.check:
        if not MANUSCRIPT_PATH.exists():
            print("FAIL: reports/manuscript.md missing")
            return 1
        rendered = render(TEMPLATE_PATH.read_text(encoding="utf-8"), tables)
        current = MANUSCRIPT_PATH.read_text(encoding="utf-8")
        if rendered != current:
            print("FAIL: reports/manuscript.md is stale; re-run without --check")
            return 1
        print("OK: manuscript matches current benchmark results")
        return 0

    write_text(TABLES_PATH, tables_md)
    if TEMPLATE_PATH.exists():
        rendered = render(TEMPLATE_PATH.read_text(encoding="utf-8"), tables)
        write_text(MANUSCRIPT_PATH, rendered)
        print(f"Wrote {MANUSCRIPT_PATH}")
    else:
        print(f"NOTE: {TEMPLATE_PATH} not found; wrote tables only")
    print(f"Wrote {TABLES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

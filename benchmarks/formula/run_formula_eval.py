"""
benchmarks/formula/run_formula_eval.py
──────────────────────────────────────
Formula / mathematical-correctness harness.

The checklist requires the pipeline's mathematical ability to be measured as
*separate* capabilities rather than one end-to-end score. This harness reports
five decoupled stages:

  1. Passage detection        — is the passage flagged as mathematical?
                                (``MathContentDetector``, text-only)
  2. Formula retrieval        — does the pipeline retrieve *any* known formula
                                (``FormulaExtractor.extract`` non-None) instead of
                                falling back to generation?
  3. Formula extraction       — is the retrieved formula the *correct* one
                                (registry key == expected.system_key)?
  4. Symbolic validity        — does the returned expression parse in SymPy and
                                is it symbolically equivalent to the expected
                                expression (``simplify(retrieved - expected) == 0``)?
  5. Interpretation correctness — does the returned formula name match the
                                expected interpretation label?

Stages 2–5 are run independently of stage 1 so that a detection failure does not
mask an extraction success (and vice-versa).

Usage
  python benchmarks/formula/run_formula_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmarks.common import (  # noqa: E402
    FORMULA_DIR, RESULTS_DIR, load_formula_items, markdown_table,
    write_json, write_text,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(text.split())


def resolve_system_key(formula: dict) -> str | None:
    """Map a retrieved formula dict back to its KNOWN_FORMULAS registry key."""
    if not formula:
        return None
    from phase3_config import KNOWN_FORMULAS
    expr = formula.get("expr", "")
    name = formula.get("name", "")
    for key, entry in KNOWN_FORMULAS.items():
        if entry["expr"] == expr:
            return key
    for key, entry in KNOWN_FORMULAS.items():
        if _norm(entry.get("name")) == _norm(name):
            return key
    return None


def symbolic_checks(chosen_expr, expected_expr) -> tuple[bool, bool | None]:
    """Return (parse_valid, equivalent). ``equivalent`` is None if untestable."""
    if not chosen_expr:
        return False, None
    try:
        import sympy as sp
        parsed = sp.sympify(chosen_expr)
    except Exception:
        return False, None
    if expected_expr is None:
        return True, None
    try:
        import sympy as sp
        diff = sp.simplify(parsed - sp.sympify(expected_expr))
        return True, bool(diff == 0)
    except Exception:
        return True, False


_STOPWORDS = {"a", "an", "the", "of", "for", "and", "to"}


def name_matches(system_name: str, expected_name: str) -> bool | None:
    """Stopword-insensitive label match (proxy for interpretation correctness)."""
    if not expected_name:
        return None
    tok_a = set(_norm(system_name).split()) - _STOPWORDS
    tok_b = set(_norm(expected_name).split()) - _STOPWORDS
    if not tok_a:
        return False
    return tok_a == tok_b or tok_a <= tok_b or tok_b <= tok_a


# ── Stages ───────────────────────────────────────────────────────────────────

def stage1_detection(items: list[dict]) -> tuple[list[dict], dict]:
    from reasoning.math_engine import MathContentDetector
    detector = MathContentDetector()

    rows, tp, fp, tn, fn = [], 0, 0, 0, 0
    for it in items:
        result = detector.detect(it["query_text"])
        pred = bool(result["math_flag"])
        gold = bool(it["is_mathematical"])
        correct = pred == gold
        if gold and pred:
            tp += 1
        elif gold and not pred:
            fn += 1
        elif not gold and pred:
            fp += 1
        else:
            tn += 1
        rows.append({
            "id": it["id"], "gold": gold, "pred": pred, "correct": correct,
            "detector_confidence": result["confidence"],
            "triggers": result["triggers"],
        })

    total = len(items)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else None)
    metrics = {
        "n": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / total if total else None,
        "precision": precision, "recall": recall, "f1": f1,
    }
    return rows, metrics


def stage2_to_5_formula(items: list[dict]) -> tuple[list[dict], dict]:
    from reasoning.math_engine import FormulaExtractor, FormulaGenerator
    extractor = FormulaExtractor(kg_engine=None)   # text-only retrieval path
    generator = FormulaGenerator()

    rows = []
    n_formula = 0
    n_retrieved = 0
    n_extract_scored = 0
    n_extract_correct = 0
    n_sym_scored = 0
    n_parse_valid = 0
    n_equivalent = 0
    n_interp_scored = 0
    n_interp_correct = 0

    for it in items:
        exp = it.get("expected", {})
        expected_key = exp.get("formula_key")
        if expected_key is None:
            continue   # detection-only item
        n_formula += 1

        retrieved = extractor.extract(it["query_text"])
        retrieved_key = resolve_system_key(retrieved) if retrieved else None
        if retrieved is not None:
            n_retrieved += 1
            path = "registry" if retrieved_key else "registry_unnamed"
        else:
            generated = generator.generate(it["query_text"])
            path = "generated"

        chosen = retrieved if retrieved is not None else generated
        chosen_expr = chosen.get("expr")
        chosen_name = chosen.get("name", "")

        expected_sys = exp.get("system_key")
        extraction_correct = None
        if expected_sys is not None:
            n_extract_scored += 1
            extraction_correct = (retrieved_key == expected_sys)
            n_extract_correct += int(extraction_correct)

        parse_valid, equivalent = symbolic_checks(chosen_expr, exp.get("expr"))
        if exp.get("expr") is not None:
            n_sym_scored += 1
            n_parse_valid += int(parse_valid)
            n_equivalent += int(bool(equivalent))

        interp = name_matches(chosen_name, exp.get("name"))
        if interp is not None:
            n_interp_scored += 1
            n_interp_correct += int(interp)

        rows.append({
            "id": it["id"],
            "path": path,
            "expected_key": expected_key,
            "expected_system_key": expected_sys,
            "retrieved_key": retrieved_key,
            "retrieval_success": retrieved is not None,
            "extraction_correct": extraction_correct,
            "chosen_expr": chosen_expr,
            "expected_expr": exp.get("expr"),
            "parse_valid": parse_valid,
            "equivalent": equivalent,
            "chosen_name": chosen_name,
            "expected_name": exp.get("name"),
            "interpretation_correct": interp,
        })

    def rate(num, den):
        return (num / den) if den else None

    metrics = {
        "n_formula_items": n_formula,
        "retrieval": {
            "scored": n_formula,
            "retrieved": n_retrieved,
            "rate": rate(n_retrieved, n_formula),
        },
        "extraction": {
            "scored": n_extract_scored,
            "correct": n_extract_correct,
            "accuracy": rate(n_extract_correct, n_extract_scored),
        },
        "symbolic": {
            "scored": n_sym_scored,
            "parse_valid": n_parse_valid,
            "parse_valid_rate": rate(n_parse_valid, n_sym_scored),
            "equivalent": n_equivalent,
            "equivalent_rate": rate(n_equivalent, n_sym_scored),
        },
        "interpretation": {
            "scored": n_interp_scored,
            "correct": n_interp_correct,
            "accuracy": rate(n_interp_correct, n_interp_scored),
        },
    }
    return rows, metrics


# ── Reporting ────────────────────────────────────────────────────────────────

def render_markdown(det_metrics, det_rows, f_metrics, f_rows) -> str:
    lines = [
        "# Formula Benchmark — Math-Correctness Results",
        "",
        "Five decoupled stages. Passage detection is measured on all items; "
        "formula stages are measured on items that specify a formula, "
        "**independently of detection**.",
        "",
        "## Stage 1 — Passage detection",
        "",
        markdown_table(
            ["TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1"],
            [[det_metrics["tp"], det_metrics["fp"], det_metrics["tn"],
              det_metrics["fn"], det_metrics["accuracy"], det_metrics["precision"],
              det_metrics["recall"], det_metrics["f1"]]],
        ),
        "",
        "| Item | Gold | Predicted | Conf | Top trigger |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in det_rows:
        trig = r["triggers"][0] if r["triggers"] else "—"
        lines.append(f"| {r['id']} | {r['gold']} | {r['pred']} | "
                     f"{r['detector_confidence']:.2f} | {trig} |")

    r = f_metrics["retrieval"]
    e = f_metrics["extraction"]
    s = f_metrics["symbolic"]
    i = f_metrics["interpretation"]
    lines += [
        "",
        "## Stages 2–5 — Formula pipeline",
        "",
        markdown_table(
            ["Stage", "Scored", "Correct", "Rate"],
            [
                ["2. Formula retrieval (any known formula)", r["scored"], r["retrieved"], r["rate"]],
                ["3. Formula extraction (exact registry key)", e["scored"], e["correct"], e["accuracy"]],
                ["4a. Symbolic parse valid", s["scored"], s["parse_valid"], s["parse_valid_rate"]],
                ["4b. Symbolic equivalent to expected", s["scored"], s["equivalent"], s["equivalent_rate"]],
                ["5. Interpretation name match", i["scored"], i["correct"], i["accuracy"]],
            ],
        ),
        "",
        "## Per-item detail",
        "",
        markdown_table(
            ["Item", "Path", "Expected key", "Retrieved key", "Extract OK",
             "Parse", "Equiv", "Interp"],
            [[x["id"], x["path"], x["expected_key"], x["retrieved_key"],
              x["extraction_correct"], x["parse_valid"], x["equivalent"],
              x["interpretation_correct"]] for x in f_rows],
        ),
        "",
        "## Notes",
        "",
        "- Interpretation scoring is an automatic name-match *proxy*; final "
        "interpretation correctness must be scored by a domain reviewer "
        "(`provenance.needs_reviewer` is true for every item).",
        "- Stage 4b is `null` when the item has no expected closed-form expression.",
        "- The extractor is run text-only (no knowledge-graph `shloka_id`), which "
        "isolates the keyword/registry logic from the KG lookup path.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Formula benchmark harness")
    parser.add_argument("--only", default=None, help="run a single item id")
    args = parser.parse_args()

    data = load_formula_items()
    items = data["items"]
    if args.only:
        items = [it for it in items if it["id"] == args.only]

    det_rows, det_metrics = stage1_detection(items)
    f_rows, f_metrics = stage2_to_5_formula(items)

    results = {
        "config": {"n_items": len(items), "expr_convention": data["expr_convention"]},
        "stage1_detection": {"metrics": det_metrics, "per_item": det_rows},
        "stage2_5_formula": {"metrics": f_metrics, "per_item": f_rows},
    }
    write_json(RESULTS_DIR / "formula_metrics.json", results)
    write_text(RESULTS_DIR / "formula_metrics.md",
               render_markdown(det_metrics, det_rows, f_metrics, f_rows))

    print(f"Stage 1 detection accuracy : {det_metrics['accuracy']}")
    print(f"Stage 2 retrieval rate     : {f_metrics['retrieval']['rate']}")
    print(f"Stage 3 extraction accuracy: {f_metrics['extraction']['accuracy']}")
    print(f"Stage 4a parse-valid rate  : {f_metrics['symbolic']['parse_valid_rate']}")
    print(f"Stage 4b equivalent rate   : {f_metrics['symbolic']['equivalent_rate']}")
    print(f"Stage 5 interpretation     : {f_metrics['interpretation']['accuracy']}")
    print("\nWrote:")
    print(f"  {RESULTS_DIR / 'formula_metrics.json'}")
    print(f"  {RESULTS_DIR / 'formula_metrics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

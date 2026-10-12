"""
Module 4 — Annotation Engine
Vedic Shloka Intelligence and Mathematical Knowledge Extraction System

Supports manual and AI-assisted annotation of segmented shlokas.
Saves annotated data to data/annotated/
"""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SEGMENTED_DIR, ANNOTATED_DIR, LOGS_DIR, ANNOTATION_FIELDS
from scripts.logger import setup_logger, JSONLogger

logger = setup_logger("annotation", LOGS_DIR)
audit = JSONLogger(LOGS_DIR / "annotation_audit.jsonl")


# ---------------------------------------------------------------------------
# Annotation schema
# ---------------------------------------------------------------------------

VEDA_OPTIONS = ["Rigveda", "Samaveda", "Yajurveda", "Atharvaveda", "Other", "Unknown"]

DOMAIN_OPTIONS = [
    "Mathematics", "Astronomy", "Grammar", "Philosophy", "Ritual",
    "Medicine", "Linguistics", "Cosmology", "Ethics", "Unknown"
]

BRANCH_OPTIONS = [
    "Arithmetic", "Geometry", "Algebra", "Combinatorics", "Number Theory",
    "Kalpa", "Jyotisha", "Shiksha", "Chandas", "Nirukta", "Vyakarana",
    "Vedanta", "Mimamsa", "Unknown"
]

LANGUAGE_OPTIONS = ["Sanskrit", "Prakrit", "Mixed", "Unknown"]

# ---------------------------------------------------------------------------
# Keyword lexicon for heuristic domain classification
#
# "strong" terms are domain-specific and score 3 each; "weak" terms are common
# in ordinary narrative Sanskrit (e.g. धर्म, कर्म) and score 1 each. A verse is
# labelled only when its best domain reaches SCORE_THRESHOLD, so a single
# generic word can no longer mislabel an epic verse as Philosophy.
# ---------------------------------------------------------------------------

SCORE_STRONG = 3
SCORE_WEAK = 1
SCORE_THRESHOLD = 3

DOMAIN_LEXICON = {
    "Mathematics": {
        "branch": "Arithmetic",
        "strong": [
            "गणित", "गणना", "संख्यान", "संख्या", "क्षेत्रफल", "बीजगणित",
            "रेखागणित", "त्रैराशिक", "समीकरण", "श्रेणी", "पूर्णांक", "भिन्न",
            "वर्गमूल", "घनमूल", "समकोण", "त्रिभुज", "चतुर्भुज", "वृत्त",
            "परिधि", "व्यास", "त्रिज्या", "ज्या", "कोटिज्या", "कोण",
            "शून्य", "मापन",
            "ganita", "sankhya", "kshetra", "bijaganita", "rekha",
            "samikarana", "trijya", "vyasa", "paridhi", "shunya",
        ],
        "weak": ["वर्ग", "घन", "अंक", "योग", "मान", "राशि", "परिमाण", "प्रमाण", "varga", "ghana", "kona"],
    },
    "Astronomy": {
        "branch": "Jyotisha",
        "strong": [
            "ज्योतिष", "नक्षत्र", "ग्रहण", "संक्रांति", "राशिचक्र", "खगोल",
            "गोल", "यन्त्र", "क्रांति", "अयन", "दृक्प", "भास्कर", "वेध",
            "jyotisha", "nakshatra", "grahana", "samkranti", "khagola",
            "yantra", "kranti", "ayana", "bhaskara",
        ],
        "weak": ["ग्रह", "चन्द्र", "सूर्य", "तारा", "रवि", "मास", "छाया", "अर्क", "उदय", "अस्त", "पात", "तिथि", "मण्डल", "graha", "chandra", "surya"],
    },
    "Grammar": {
        "branch": "Vyakarana",
        "strong": [
            "व्याकरण", "प्रत्यय", "विभक्ति", "संधि", "समास", "कारक",
            "कृदंत", "तद्धित", "उपसर्ग", "अव्यय", "सर्वनाम", "लकार",
            "वाच्य", "अष्टाध्यायी", "पाणिनि", "निरुक्त", "शिक्षा", "छन्दस्",
            "vyakarana", "pratyaya", "vibhakti", "samasa", "karaka",
            "panini", "ashtadhyayi", "nirukta",
        ],
        "weak": ["सूत्र", "वर्ण", "स्वर", "व्यंजन", "लिङ्ग", "वचन", "छन्द", "धातु", "पद", "sutra", "dhatu"],
    },
    "Philosophy": {
        "branch": "Vedanta",
        "strong": [
            "ब्रह्म", "परमात्मन्", "आत्मन्", "मोक्ष", "अद्वैत", "द्वैत",
            "वेदान्त", "उपनिषद्", "उपनिषद", "माया", "पुरुषार्थ", "संसार",
            "सांख्य", "समाधि", "तत्त्व", "ईश्वर", "परमेश्वर", "जीव",
            "मुक्ति", "निर्वाण", "दर्शन", "न्याय", "वैशेषिक", "मीमांसा",
            "अविद्या", "आत्मा", "निर्वाण",
            "brahma", "atman", "moksha", "advaita", "vedanta", "upanishad",
            "maya", "purushartha", "samkhya", "tattva", "ishvara", "mukti",
            "nirvana", "darshana",
        ],
        "weak": ["धर्म", "कर्म", "ज्ञान", "योग", "ध्यान", "अध्यात्म", "प्रकृति", "जीव", "dharma", "karma", "jnana"],
    },
}


def _validate_annotation(annotation: dict) -> list[str]:
    """
    Validate annotation fields. Returns list of warning messages (empty = valid).
    """
    warnings = []
    for field in ANNOTATION_FIELDS:
        if field not in annotation:
            warnings.append(f"Missing field: {field}")
    if "keywords" in annotation and not isinstance(annotation["keywords"], list):
        warnings.append("'keywords' must be a list")
    return warnings


def _default_annotation() -> dict:
    """Return a blank annotation template."""
    return {
        "veda": "Unknown",
        "domain": "Unknown",
        "branch": "Unknown",
        "formula": "",
        "source": "",
        "language": "Sanskrit",
        "keywords": [],
        "annotated_by": "manual",
        "annotated_at": None,
        "confidence": 1.0,
        "notes": "",
    }


# ---------------------------------------------------------------------------
# AI-assisted annotation
# ---------------------------------------------------------------------------

def ai_annotate_shloka(shloka: dict) -> dict:
    """
    Generate AI-assisted annotation for a single shloka using heuristic rules.

    This module uses rule-based heuristics to classify shlokas without
    requiring external API calls. For production, replace with an LLM call.

    Returns:
        Annotation dict
    """
    text = shloka.get("text", "")
    source = shloka.get("source", "")

    annotation = _default_annotation()
    annotation["source"] = source
    annotation["annotated_by"] = "ai_heuristic"
    annotation["annotated_at"] = datetime.utcnow().isoformat() + "Z"
    annotation["confidence"] = 0.6  # Heuristic confidence

    # --- Veda detection ---
    source_lower = source.lower()
    if "rigveda" in source_lower or "rig veda" in source_lower:
        annotation["veda"] = "Rigveda"
    elif "samaveda" in source_lower or "sama veda" in source_lower:
        annotation["veda"] = "Samaveda"
    elif "yajurveda" in source_lower or "yajur veda" in source_lower:
        annotation["veda"] = "Yajurveda"
    elif "atharvaveda" in source_lower or "atharva veda" in source_lower:
        annotation["veda"] = "Atharvaveda"
    else:
        annotation["veda"] = "Unknown"

    # --- Domain detection (weighted keyword scoring, Devanagari + transliteration) ---
    text_lower = text.lower()
    scores = {}
    for domain, spec in DOMAIN_LEXICON.items():
        score = sum(SCORE_STRONG for kw in spec["strong"] if kw in text or kw in text_lower)
        score += sum(SCORE_WEAK for kw in spec["weak"] if kw in text or kw in text_lower)
        if score:
            scores[domain] = score

    if scores:
        best_domain, best_score = max(scores.items(), key=lambda kv: kv[1])
    else:
        best_domain, best_score = "Unknown", 0

    if best_score >= SCORE_THRESHOLD:
        annotation["domain"] = best_domain
        annotation["branch"] = DOMAIN_LEXICON[best_domain]["branch"]
    else:
        annotation["domain"] = "Unknown"
        annotation["branch"] = "Unknown"

    # --- Formula detection ---
    # Check for numeric patterns or mathematical terms
    import re
    if re.search(r"[\d०-९]+", text):  # ASCII or Devanagari digits
        annotation["formula"] = "contains_numeric_expression"
    elif "सूत्र" in text or "sutra" in text.lower():
        annotation["formula"] = "sutric_formula"
    else:
        annotation["formula"] = ""

    # --- Keywords ---
    keywords = []
    if annotation["domain"] != "Unknown":
        keywords.append(annotation["domain"].lower())
    if annotation["veda"] != "Unknown":
        keywords.append(annotation["veda"].lower())
    if annotation["branch"] != "Unknown":
        keywords.append(annotation["branch"].lower())
    annotation["keywords"] = keywords

    annotation["language"] = "Sanskrit"
    return annotation


# ---------------------------------------------------------------------------
# Annotation of files
# ---------------------------------------------------------------------------

def annotate_shloka(shloka: dict, annotation: Optional[dict] = None, use_ai: bool = False) -> dict:
    """
    Annotate a single shloka dict.

    Args:
        shloka: Shloka dict (id, text, source, chapter)
        annotation: Manual annotation dict (optional)
        use_ai: If True and no manual annotation provided, run AI heuristics

    Returns:
        Annotated shloka dict
    """
    result = dict(shloka)  # Copy

    if annotation:
        ann = {**_default_annotation(), **annotation}
        ann["annotated_by"] = annotation.get("annotated_by", "manual")
        ann["annotated_at"] = annotation.get("annotated_at", datetime.utcnow().isoformat() + "Z")
    elif use_ai:
        ann = ai_annotate_shloka(shloka)
    else:
        ann = _default_annotation()
        ann["source"] = shloka.get("source", "")

    # Validate
    warnings = _validate_annotation(ann)
    if warnings:
        logger.warning(f"Annotation warnings for {shloka.get('id')}: {warnings}")

    result["annotation"] = ann
    return result


def annotate_file(
    segmented_path: str | Path,
    output_path: Optional[str | Path] = None,
    use_ai: bool = True,
    manual_annotations: Optional[dict] = None,
) -> Path:
    """
    Annotate all shlokas in a segmented JSON file.

    Args:
        segmented_path: Path to segmented JSON file
        output_path: Destination (defaults to data/annotated/<stem>.json)
        use_ai: Use AI-assisted heuristic annotation as default
        manual_annotations: Dict mapping shloka_id → annotation dict for manual overrides

    Returns:
        Path to annotated JSON file
    """
    segmented_path = Path(segmented_path)
    manual_annotations = manual_annotations or {}

    if output_path is None:
        ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = ANNOTATED_DIR / segmented_path.name
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(segmented_path, encoding="utf-8") as f:
        shlokas = json.load(f)

    annotated = []
    ai_count = 0
    manual_count = 0

    for shloka in shlokas:
        sid = shloka.get("id", "")
        manual_ann = manual_annotations.get(sid)
        if manual_ann:
            result = annotate_shloka(shloka, annotation=manual_ann)
            manual_count += 1
        else:
            result = annotate_shloka(shloka, use_ai=use_ai)
            if use_ai:
                ai_count += 1

        annotated.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, ensure_ascii=False, indent=2)

    audit.log(
        "file_annotated",
        file=segmented_path.name,
        total=len(annotated),
        manual=manual_count,
        ai=ai_count,
        output=str(output_path),
    )
    logger.info(
        f"Annotated {len(annotated)} shlokas → {output_path} "
        f"(manual={manual_count}, ai={ai_count})"
    )
    return output_path


def annotate_directory(
    segmented_dir: str | Path = None,
    output_dir: str | Path = None,
    use_ai: bool = True,
) -> list[Path]:
    """
    Annotate all segmented JSON files in a directory.
    """
    segmented_dir = Path(segmented_dir or SEGMENTED_DIR)
    output_dir = Path(output_dir or ANNOTATED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(segmented_dir.glob("*.json"))
    logger.info(f"Found {len(files)} segmented files to annotate")

    results = []
    for f in files:
        try:
            out = annotate_file(f, output_dir / f.name, use_ai=use_ai)
            results.append(out)
        except Exception as exc:
            logger.error(f"Failed to annotate {f.name}: {exc}")
            audit.log("annotation_error", file=str(f), error=str(exc))

    logger.info(f"Annotation complete. {len(results)}/{len(files)} files processed.")
    return results


def load_annotated_shlokas(annotated_dir: str | Path = None) -> list[dict]:
    """Load all annotated JSON files into a flat list."""
    annotated_dir = Path(annotated_dir or ANNOTATED_DIR)
    all_shlokas = []
    for jf in sorted(annotated_dir.glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            shlokas = json.load(f)
        all_shlokas.extend(shlokas)
    logger.info(f"Loaded {len(all_shlokas):,} annotated shlokas")
    return all_shlokas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate segmented shlokas")
    parser.add_argument("path", help="Segmented JSON file or directory")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-ai", action="store_true", help="Disable AI-assisted annotation")
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        annotate_directory(p, args.output, use_ai=not args.no_ai)
    else:
        annotate_file(p, args.output, use_ai=not args.no_ai)

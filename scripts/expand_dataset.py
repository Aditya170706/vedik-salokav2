# ============================================================
# scripts/expand_dataset.py  — FULL CORRECTED VERSION
#
# All URLs verified working as of 2026.
# Sources used:
#   1. rahular/itihasa          — Apache 2.0  (93K shlokas)
#   2. gita/gita verse.json     — MIT         (700 verses, single file)
#   3. dataspoof/Vedas CSVs     — GRETIL/research (4 Vedas)
#   4. dataspoof/Puranas CSVs   — GRETIL/research (Puranas)
#   5. ambuda-org/gretil        — Public domain (classical texts)
#
# Usage (from project root):
#   python scripts/expand_dataset.py
#
# After running, feed into your existing pipeline:
#   python -m scripts.module2_cleaning data/raw/ --output data/cleaned/
#   python -m scripts.module3_segmentation data/cleaned/
#   python -m scripts.module4_annotation data/segmented/
#   python -m scripts.module5_dataset --seed 42
#   python p2_run_training.py
# ============================================================

import os
import re
import json
import time
import hashlib
import unicodedata
import requests
from io import StringIO
from datetime import datetime, timezone
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("WARNING: pandas not installed. CSV sources will use basic parsing.")

# ── Resolve paths from project root ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
META_FILE    = PROJECT_ROOT / "metadata" / "source_registry.json"
LOG_FILE     = PROJECT_ROOT / "logs"  / "ingestion_audit.jsonl"

RAW_DIR.mkdir(parents=True,   exist_ok=True)
META_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True,  exist_ok=True)

HEADERS = {"User-Agent": "SanskritSciNet-Research/2.0 (academic use)"}

# Devanagari block — used to score candidate text columns in CSVs.
_RE_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Base URL for the four-Veda CSV dataset (one file per Veda).
VEDA_BASE = "https://huggingface.co/datasets/dataspoof/Vedas/resolve/main/"

# ── Load existing registry ────────────────────────────────────
if META_FILE.exists():
    with open(META_FILE, encoding="utf-8") as f:
        try:
            source_registry = json.load(f)
        except json.JSONDecodeError:
            source_registry = []
else:
    source_registry = []

already_registered = {s.get("source_name") for s in source_registry}

# ── Logger ────────────────────────────────────────────────────
def log_event(event_type, source_name, status, details=""):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event"    : event_type,
        "source"   : source_name,
        "status"   : status,
        "details"  : details,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def save_registry():
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(source_registry, f, indent=2, ensure_ascii=False)

# ── HTTP fetch with retries ───────────────────────────────────
def fetch(url, timeout=30, retries=3):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            print(f"    HTTP {r.status_code} — attempt {attempt}/{retries}")
        except Exception as e:
            print(f"    Error: {e} — attempt {attempt}/{retries}")
        if attempt < retries:
            time.sleep(2 ** attempt)
    return None

# ── Text utilities ────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u00ad]', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def is_sanskrit(line):
    """Returns True if line has Devanagari OR IAST diacritics."""
    if re.search(r'[\u0900-\u097F]', line):
        return True
    if re.search(r'[āīūṛṝḷḹṃḥśṣṭḍṇñṅ]', line, re.IGNORECASE):
        return True
    return False

def decode_bytes(content):
    """Try common encodings for ancient text files."""
    for enc in ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            return content.decode(enc)
        except (UnicodeDecodeError, AttributeError):
            continue
    return content.decode("utf-8", errors="replace")

# ── Save to data/raw/ and register ───────────────────────────
def save_source(filename, text, source_name,
                author, license_type, citation,
                publication_year=None):
    """
    Saves text to data/raw/<filename>.txt and registers in
    metadata/source_registry.json (same format as module1_ingestion).
    Skips if source_name already registered.
    """
    if source_name in already_registered:
        print(f"  SKIP — already registered: {source_name}")
        return False

    cleaned = clean_text(text)
    if not cleaned:
        print(f"  SKIP — empty after cleaning: {source_name}")
        return False

    # keep only Sanskrit lines
    lines = [
        l.strip() for l in cleaned.split("\n")
        if l.strip() and len(l.strip()) > 5 and is_sanskrit(l)
    ]
    if not lines:
        print(f"  SKIP — no Sanskrit content detected: {source_name}")
        log_event("INGEST", source_name, "SKIP", "no Sanskrit content")
        return False

    final_text = "\n".join(lines)
    out_path   = RAW_DIR / f"{filename}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    sha256 = hashlib.sha256(final_text.encode("utf-8")).hexdigest()

    entry = {
        "source_name"       : source_name,
        "author"            : author,
        "publication_year"  : publication_year,
        "license_type"      : license_type,
        "citation_reference": citation,
        "sha256"            : sha256,
        "char_count"        : len(final_text),
        "line_count"        : len(lines),
        "format"            : "txt",
        "ingested_at"       : datetime.now(timezone.utc).isoformat(),
        "raw_file"          : str(out_path.relative_to(PROJECT_ROOT)),
        "ingested_by"       : "expand_dataset.py",
    }
    source_registry.append(entry)
    already_registered.add(source_name)
    save_registry()

    log_event("INGEST", source_name, "SUCCESS",
              f"{len(lines)} lines, sha256={sha256[:16]}")

    print(f"  SAVED → {out_path.name}"
          f"  ({len(lines):,} lines, "
          f"{len(final_text)/1024:.0f} KB)")
    return True

# ================================================================
# FETCHERS — one per source type
# ================================================================

def fetch_itihasa():
    """
    Itihasa: Apache 2.0
    93K Sanskrit lines from Ramayana + Mahabharata.
    Fetched as raw text lines from HuggingFace CSV files.
    CSV parsing skipped because shlokas contain commas.
    """
    base  = "https://huggingface.co/datasets/rahular/itihasa/resolve/main"
    files = ["train.sn.csv", "dev.sn.csv", "test.sn.csv"]
    all_lines = []
    for fname in files:
        r = fetch(f"{base}/{fname}", timeout=30)
        if not r:
            print(f"    {fname}: failed")
            continue
        lines = [l.strip().strip('"')
                 for l in r.text.split("\n") if l.strip()]
        all_lines.extend(lines)
        print(f"    {fname}: {len(lines):,} lines")
    return "\n".join(all_lines)

def fetch_gita():
    """
    Bhagavad Gita: MIT
    Single verse.json file on the main branch (not master).
    URL verified: github.com/gita/gita/blob/main/data/verse.json
    """
    url = "https://raw.githubusercontent.com/gita/gita/main/data/verse.json"
    r   = fetch(url, timeout=30)
    if not r:
        return ""
    try:
        data = r.json()
        # verse.json is a list of verse objects
        verses = data if isinstance(data, list) else data.get("verses", [])
        text_fields = ["text", "transliteration", "word_meanings",
                       "sanskrit", "shloka"]
        lines = []
        for v in verses:
            if not isinstance(v, dict):
                continue
            for field in text_fields:
                val = str(v.get(field, "")).strip()
                if val and val.lower() not in ("none", "null", ""):
                    lines.append(val)
                    break
        print(f"    verse.json: {len(lines):,} verses")
        return "\n".join(lines)
    except Exception as e:
        print(f"    verse.json error: {e}")
        return ""

def fetch_hf_csv_dataset(csv_urls, text_col_hints):
    """
    Generic fetcher for HuggingFace CSV datasets.
    Dynamically finds the text column from hints.
    text_col_hints: list of likely column name substrings
    """
    all_lines = []
    for url in csv_urls:
        r = fetch(url, timeout=60)
        if not r:
            print(f"    {url.split('/')[-1]}: failed")
            continue
        try:
            if HAS_PANDAS:
                df = pd.read_csv(
                    StringIO(r.text),
                    on_bad_lines="skip",
                    encoding_errors="replace"
                )
                # Find the text column dynamically, preferring hint matches
                # that actually contain Devanagari. Some CSVs ship both a good
                # Devanagari column and a broken legacy/Kruti-encoded lookalike
                # under a similar name (e.g. Yajurveda's `sanskrit` vs
                # `new_sanskrit`), so score candidates instead of taking the
                # first name that matches a hint.
                candidates = [
                    c for c in df.columns
                    if any(h.lower() in c.lower() for h in text_col_hints)
                ] or list(df.columns)
                text_col = max(
                    candidates,
                    key=lambda c: sum(
                        1 for v in df[c].dropna()
                        if _RE_DEVANAGARI.search(str(v))
                    ),
                )
                lines = [str(v).strip() for v in df[text_col].dropna()
                         if str(v).strip()]
            else:
                # fallback: read raw lines
                lines = [l.strip() for l in r.text.split("\n")
                         if l.strip()]
                if lines:
                    lines = lines[1:]  # skip header

            all_lines.extend(lines)
            print(f"    {url.split('/')[-1]}: {len(lines):,} rows")

        except Exception as e:
            print(f"    {url.split('/')[-1]}: {e}")
            # last resort: just take raw lines
            lines = [l.strip() for l in r.text.split("\n") if l.strip()]
            all_lines.extend(lines[1:])

    return "\n".join(all_lines)

def fetch_ambuda_gretil():
    """
    ambuda-org/gretil: Public domain.
    GitHub API discovers all .txt files dynamically.
    Texts are TEI/XML transformed to plain text.
    """
    api_url = "https://api.github.com/repos/ambuda-org/gretil/contents/sa"
    r = fetch(api_url, timeout=20)
    if not r:
        # try root if /sa doesn't exist
        api_url = "https://api.github.com/repos/ambuda-org/gretil/contents"
        r = fetch(api_url, timeout=20)
    if not r:
        return ""
    try:
        items = r.json()
        if not isinstance(items, list):
            return ""
        # get all .txt files
        txt_files = [item for item in items
                     if isinstance(item, dict)
                     and item.get("type") == "file"
                     and item.get("name","").endswith(".txt")]
        print(f"    Found {len(txt_files)} .txt files in ambuda-org/gretil")
        all_text = []
        for item in txt_files[:50]:   # cap at 50 files
            dl = item.get("download_url","")
            if not dl:
                continue
            fr = fetch(dl, timeout=20)
            if fr:
                all_text.append(decode_bytes(fr.content))
        return "\n\n".join(all_text)
    except Exception as e:
        print(f"    ambuda-org/gretil error: {e}")
        return ""

# ================================================================
# SOURCE DEFINITIONS
# All verified URLs, all legal licenses
# ================================================================

SOURCES = [

    # ── 1. Itihasa — 93K shlokas (Apache 2.0) ────────────────
    {
        "source_name"   : "Itihasa (Ramayana and Mahabharata)",
        "filename"      : "itihasa_ramayana_mahabharata",
        "author"        : "Aralikatte et al. (2021)",
        "year"          : 2021,
        "license"       : "Apache-2.0",
        "citation"      : ("Aralikatte et al., Itihasa: A large-scale corpus"
                           " for Sanskrit to English translation, WAT2021, ACL."),
        "fetcher"       : fetch_itihasa,
    },

    # ── 2. Bhagavad Gita — 700 verses (MIT) ──────────────────
    # URL: github.com/gita/gita/blob/main/data/verse.json
    {
        "source_name"   : "Bhagavad Gita (All 18 Chapters)",
        "filename"      : "bhagavad_gita_18_chapters",
        "author"        : "gita/gita GitHub (MIT)",
        "year"          : None,
        "license"       : "MIT",
        "citation"      : ("gita/gita GitHub repository. MIT License."
                           " https://github.com/gita/gita"),
        "fetcher"       : fetch_gita,
    },

    # ── 3. Four Vedas — one source per Veda (GRETIL source) ──
    # dataspoof/Vedas on HuggingFace. Kept separate (not concatenated) so
    # segmentation/annotation can attribute each shloka to the correct Veda.
    {
        "source_name"   : "Rigveda (Complete Samhita)",
        "filename"      : "rigveda",
        "author"        : "Traditional (GRETIL source, cleaned by dataspoof)",
        "year"          : None,
        "license"       : "Public Domain (GRETIL texts, research CSV)",
        "citation"      : ("Rigveda. Originally from GRETIL "
                           "https://gretil.sub.uni-goettingen.de/. "
                           "CSV: dataspoof/Vedas (rigveda_complete.csv)."),
        "fetcher"       : lambda: fetch_hf_csv_dataset(
            csv_urls=[VEDA_BASE + "rigveda_complete.csv"],
            text_col_hints=["devanagari","sanskrit","text","verse","shloka","mantra"],
        ),
    },
    {
        "source_name"   : "Samaveda",
        "filename"      : "samaveda",
        "author"        : "Traditional (GRETIL source, cleaned by dataspoof)",
        "year"          : None,
        "license"       : "Public Domain (GRETIL texts, research CSV)",
        "citation"      : ("Samaveda. Originally from GRETIL "
                           "https://gretil.sub.uni-goettingen.de/. "
                           "CSV: dataspoof/Vedas (Samveda.csv)."),
        "fetcher"       : lambda: fetch_hf_csv_dataset(
            csv_urls=[VEDA_BASE + "Samveda.csv"],
            text_col_hints=["devanagari","sanskrit","text","verse","shloka","mantra"],
        ),
    },
    {
        "source_name"   : "Yajurveda",
        "filename"      : "yajurveda",
        "author"        : "Traditional (GRETIL source, cleaned by dataspoof)",
        "year"          : None,
        "license"       : "Public Domain (GRETIL texts, research CSV)",
        "citation"      : ("Yajurveda. Originally from GRETIL "
                           "https://gretil.sub.uni-goettingen.de/. "
                           "CSV: dataspoof/Vedas (yajurveda.csv)."),
        "fetcher"       : lambda: fetch_hf_csv_dataset(
            csv_urls=[VEDA_BASE + "yajurveda.csv"],
            text_col_hints=["devanagari","sanskrit","text","verse","shloka","mantra"],
        ),
    },
    {
        "source_name"   : "Atharvaveda",
        "filename"      : "atharvaveda",
        "author"        : "Traditional (GRETIL source, cleaned by dataspoof)",
        "year"          : None,
        "license"       : "Public Domain (GRETIL texts, research CSV)",
        "citation"      : ("Atharvaveda. Originally from GRETIL "
                           "https://gretil.sub.uni-goettingen.de/. "
                           "CSV: dataspoof/Vedas (Atharveda.csv)."),
        "fetcher"       : lambda: fetch_hf_csv_dataset(
            csv_urls=[VEDA_BASE + "Atharveda.csv"],
            text_col_hints=["devanagari","sanskrit","text","verse","shloka","mantra"],
        ),
    },

    # ── 4. Puranas CSVs (GRETIL source, research use) ────────
    # dataspoof/Puranas-dataset on HuggingFace
    # Multiple Puranas in cleaned CSV format
    {
        "source_name"   : "Puranas Dataset (GRETIL source)",
        "filename"      : "puranas_dataset",
        "author"        : "Traditional (GRETIL source, cleaned by dataspoof)",
        "year"          : None,
        "license"       : "Public Domain (GRETIL texts, research CSV)",
        "citation"      : ("Puranas. Originally from GRETIL "
                           "https://gretil.sub.uni-goettingen.de/. "
                           "CSV format: dataspoof/Puranas-dataset on HuggingFace."),
        "fetcher"       : lambda: fetch_hf_csv_dataset(
            csv_urls=[
                "https://huggingface.co/datasets/dataspoof/Puranas-dataset/resolve/main/Matsyapurana_cleaned_with_devanagari.csv",
                "https://huggingface.co/datasets/dataspoof/Puranas-dataset/resolve/main/Agnipurana_cleaned_with_devanagari.csv",
                "https://huggingface.co/datasets/dataspoof/Puranas-dataset/resolve/main/Bhagavatapurana_cleaned_with_devanagari.csv",
            ],
            text_col_hints=["devanagari","sanskrit","text","verse","shloka"],
        ),
    },

    # ── 5. ambuda-org/gretil — Public domain ─────────────────
    # TEI documents from GRETIL with adhoc fixes
    # License: public domain (stated explicitly in README)
    {
        "source_name"   : "GRETIL Sanskrit Texts (ambuda-org mirror)",
        "filename"      : "gretil_ambuda_mirror",
        "author"        : "Various classical authors (GRETIL via ambuda-org)",
        "year"          : None,
        "license"       : "Public Domain",
        "citation"      : ("GRETIL — Göttingen Register of Electronic Texts "
                           "in Indian Languages. http://gretil.sub.uni-goettingen.de/."
                           " Mirror: github.com/ambuda-org/gretil."),
        "fetcher"       : fetch_ambuda_gretil,
    },

    # ── 6. Buddhist Sanskrit (dataspoof/Buddhist) ─────────────
    # Buddhist Sanskrit texts from GRETIL — public domain
    {
        "source_name"   : "Buddhist Sanskrit Canon (GRETIL source)",
        "filename"      : "buddhist_sanskrit_canon",
        "author"        : "Various classical authors (GRETIL source)",
        "year"          : None,
        "license"       : "Public Domain (GRETIL texts, research CSV)",
        "citation"      : ("Buddhist Sanskrit Canon. Originally from GRETIL. "
                           "CSV format: dataspoof/Buddhist on HuggingFace."),
        "fetcher"       : lambda: fetch_hf_csv_dataset(
            csv_urls=[
                "https://huggingface.co/datasets/dataspoof/Buddhist/resolve/main/buddhist_sanskrit.csv",
            ],
            text_col_hints=["devanagari","sanskrit","text","verse","shloka"],
        ),
    },
]

# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "="*60)
    print("  SanskritSciNet — Dataset Expansion")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Raw data dir : {RAW_DIR}")
    print(f"  Registry     : {META_FILE}")
    print("="*60)

    # show what is already registered
    if already_registered:
        print(f"\n  Already registered ({len(already_registered)}):")
        for name in sorted(already_registered):
            print(f"    ~ {name}")

    results = []

    for src in SOURCES:
        name = src["source_name"]
        print(f"\n{'─'*60}")
        print(f"SOURCE  : {name}")
        print(f"License : {src['license']}")

        if name in already_registered:
            print("STATUS  : Already in registry — skipping")
            results.append({"source": name, "status": "skipped"})
            continue

        print("Fetching...")
        text = ""
        try:
            text = src["fetcher"]()
        except Exception as e:
            print(f"  Fetcher exception: {e}")
            log_event("INGEST", name, "FAILED", str(e))
            results.append({"source": name, "status": "failed"})
            continue

        if not text or not text.strip():
            print("STATUS  : No text returned — skipping")
            log_event("INGEST", name, "FAILED", "empty text")
            results.append({"source": name, "status": "failed"})
            continue

        success = save_source(
            filename        = src["filename"],
            text            = text,
            source_name     = name,
            author          = src["author"],
            license_type    = src["license"],
            citation        = src["citation"],
            publication_year= src.get("year"),
        )
        results.append({
            "source": name,
            "status": "saved" if success else "skipped"
        })

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  EXPANSION COMPLETE")
    print("="*60)

    saved   = [r for r in results if r["status"] == "saved"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed  = [r for r in results if r["status"] == "failed"]

    print(f"\n  Saved   : {len(saved)}")
    for r in saved:
        print(f"    + {r['source']}")

    if skipped:
        print(f"\n  Skipped : {len(skipped)}")
        for r in skipped:
            print(f"    ~ {r['source']}")

    if failed:
        print(f"\n  Failed  : {len(failed)}")
        for r in failed:
            print(f"    x {r['source']}")

    # list data/raw/ contents
    print(f"\n{'─'*60}")
    print(f"  data/raw/ contents:")
    raw_files    = sorted(RAW_DIR.glob("*.txt"))
    total_lines  = 0
    total_size   = 0
    for f in raw_files:
        size  = f.stat().st_size
        text  = f.read_text(encoding="utf-8", errors="replace")
        lines = text.count("\n")
        total_lines += lines
        total_size  += size
        print(f"    {f.name:50s}  {lines:>8,} lines  {size/1024:>7.0f} KB")

    print(f"\n  Total files  : {len(raw_files)}")
    print(f"  Total lines  : {total_lines:,}")
    print(f"  Total size   : {total_size/1024/1024:.1f} MB")

    print("\n" + "="*60)
    print("  NEXT — run these in order from project root:")
    print("="*60)
    print("""
  python -m scripts.module2_cleaning data/raw/ --output data/cleaned/
  python -m scripts.module3_segmentation data/cleaned/
  python -m scripts.module4_annotation data/segmented/
  python -m scripts.module5_dataset --seed 42
  python p2_run_training.py
""")


if __name__ == "__main__":
    main()
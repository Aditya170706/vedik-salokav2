import os
import json
import csv
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
SEGMENTED_DIR = PROJECT_ROOT / "data" / "segmented"
TRAINING_DIR  = PROJECT_ROOT / "data" / "training"
LOG_DIR       = PROJECT_ROOT / "logs"

TRAINING_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def load_json_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def infer_domain_from_existing(data):
    domain  = data.get("domain", "Unknown")
    branch  = data.get("branch", "Unknown")
    formula = data.get("formula", "")
    if domain not in ("Unknown", "", None):
        return domain, branch, formula
    return "Unknown", branch, formula

def build_row(data, source_file):
    text = (data.get("text") or
            data.get("shloka") or
            data.get("content") or "").strip()
    if not text:
        return None

    domain, branch, formula = infer_domain_from_existing(data)

    return {
        "text"      : text,
        "source"    : data.get("source", source_file),
        "chapter"   : data.get("chapter", ""),
        "veda"      : data.get("veda", ""),
        "domain"    : domain,
        "branch"    : branch,
        "formula"   : formula,
        "language"  : data.get("language", "Sanskrit"),
        "keywords"  : data.get("keywords", ""),
        "annotated" : data.get("annotated", "ai_heuristic"),
        "confidence": data.get("confidence", 0.6),
        "notes"     : data.get("notes", ""),
    }

def load_all_records():
    records = []
    seen    = set()

    source_dir = ANNOTATED_DIR if ANNOTATED_DIR.exists() else SEGMENTED_DIR
    if not source_dir.exists():
        print(f"ERROR: Neither annotated/ nor segmented/ found.")
        return records

    files = sorted(source_dir.glob("*.json"))
    print(f"Found {len(files):,} JSON files in {source_dir.name}/")

    for f in files:
        data = load_json_file(f)
        if not data:
            continue

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("shlokas", data.get("items", data.get("verses", [data])))
        else:
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            row = build_row(item, f.stem)
            if not row:
                continue
            key = sha256_text(row["text"])
            if key in seen:
                continue
            seen.add(key)
            records.append(row)

    print(f"Unique records loaded: {len(records):,}")
    return records

def split_dataset(records, train=0.70, val=0.15, test=0.15):
    random.shuffle(records)
    n       = len(records)
    t_end   = int(n * train)
    v_end   = t_end + int(n * val)
    return records[:t_end], records[t_end:v_end], records[v_end:]

def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path.name}  ({len(rows):,} rows)")

def write_manifest(train, val, test, path):
    domain_counts = defaultdict(int)
    for r in train + val + test:
        domain_counts[r["domain"]] += 1

    manifest = {
        "created_at"       : datetime.now(timezone.utc).isoformat(),
        "random_seed"      : RANDOM_SEED,
        "total_records"    : len(train) + len(val) + len(test),
        "train_count"      : len(train),
        "validation_count" : len(val),
        "test_count"       : len(test),
        "split_ratio"      : {"train": 0.70, "validation": 0.15, "test": 0.15},
        "domain_breakdown" : dict(sorted(domain_counts.items(),
                                         key=lambda x: -x[1])),
        "source_script"    : "scripts/build_journal_dataset.py",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path.name}")
    return manifest

def main():
    print("=" * 55)
    print("  SanskritSciNet — Journal Dataset Builder")
    print("=" * 55 + "\n")

    records = load_all_records()

    if not records:
        print("No records found. Make sure module4_annotation has run.")
        return

    if len(records) < 100:
        print(f"WARNING: Only {len(records)} records found.")
        print("Journal publication typically requires 100+ records.")
        print("Run expand_dataset.py to add more sources first.")

    fieldnames = ["text","source","chapter","veda","domain",
                  "branch","formula","language","keywords",
                  "annotated","confidence","notes"]

    train, val, test = split_dataset(records)

    print(f"\nSplitting {len(records):,} records:")
    print(f"  Train      : {len(train):,} (70%)")
    print(f"  Validation : {len(val):,} (15%)")
    print(f"  Test       : {len(test):,} (15%)")
    print()

    write_csv(train, TRAINING_DIR / "train.csv",      fieldnames)
    write_csv(val,   TRAINING_DIR / "validation.csv", fieldnames)
    write_csv(test,  TRAINING_DIR / "test.csv",       fieldnames)

    manifest = write_manifest(
        train, val, test,
        TRAINING_DIR / "dataset_manifest.json"
    )

    print()
    print("=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  Total records  : {manifest['total_records']:,}")
    print(f"  Train          : {manifest['train_count']:,}")
    print(f"  Validation     : {manifest['validation_count']:,}")
    print(f"  Test           : {manifest['test_count']:,}")
    print(f"\n  Domain breakdown:")
    for domain, count in manifest["domain_breakdown"].items():
        bar = "█" * min(int(count / manifest["total_records"] * 40), 40)
        print(f"    {domain:20s} {count:>6,}  {bar}")

    journal_ready = manifest["total_records"] >= 100
    print(f"\n  Journal ready  : {'YES' if journal_ready else 'NO — need more data'}")
    print(f"\n  Next: python p2_run_training.py")
    print("=" * 55)

if __name__ == "__main__":
    main()
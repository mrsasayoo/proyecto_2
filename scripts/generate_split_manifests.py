#!/usr/bin/env python3
"""Generate per-split manifest.csv files from existing patches + candidates_V2.csv.

For each split directory (train/, val/, test/) under patches/, creates a
manifest.csv with columns: filename,label

The label is looked up from candidates_V2.csv using the 6-digit index
embedded in the filename (e.g., candidate_003633.npy → row 3633 → class=1).

Usage:
    python scripts/generate_split_manifests.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATCHES_DIR = PROJECT_ROOT / "datasets" / "luna_lung_cancer" / "patches"
CANDIDATES_CSV = (
    PROJECT_ROOT
    / "datasets"
    / "luna_lung_cancer"
    / "candidates_V2"
    / "candidates_V2.csv"
)


def extract_index(filename: str) -> int:
    """Extract the candidate row index from a patch filename.

    'candidate_003633.npy'      → 3633
    'candidate_003633_aug1.npy' → 3633
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    return int(parts[1])


def main() -> None:
    if not CANDIDATES_CSV.exists():
        print(f"ERROR: {CANDIDATES_CSV} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {CANDIDATES_CSV} ...")
    candidates = pd.read_csv(CANDIDATES_CSV)
    print(f"  {len(candidates)} candidates loaded")

    for split in ["train", "val", "test"]:
        split_dir = PATCHES_DIR / split
        if not split_dir.is_dir():
            print(f"  {split}/: directory not found — skipping")
            continue

        npy_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".npy"))
        print(f"  {split}/: {len(npy_files)} .npy files")

        rows = []
        errors = 0
        for fname in npy_files:
            try:
                idx = extract_index(fname)
                label = int(candidates.iloc[idx]["class"])
                rows.append({"filename": fname, "label": label})
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    WARNING: {fname}: {e}")

        if errors > 5:
            print(f"    ... and {errors - 5} more errors")

        df = pd.DataFrame(rows)
        n_pos = int((df["label"] == 1).sum())
        n_neg = int((df["label"] == 0).sum())

        manifest_path = split_dir / "manifest.csv"
        df.to_csv(manifest_path, index=False)
        print(f"    → {manifest_path}: {len(df)} entries (pos={n_pos}, neg={n_neg})")

    print("\nDone.")


if __name__ == "__main__":
    main()

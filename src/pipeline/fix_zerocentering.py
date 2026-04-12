#!/usr/bin/env python3
"""Fix over-centered LUNA16 patches.

Some patches had global_mean subtracted 2-3 times instead of once.
This script detects and corrects them by adding global_mean back until
the patch mean falls within the valid range [-GLOBAL_MEAN, +inf).

Idempotent: running this script multiple times produces the same result.

Usage:
    python src/pipeline/fix_zerocentering.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Suppress numpy overflow warnings from corrupt patches
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GLOBAL_MEAN: float = 0.09921630471944809
MIN_VALID_MEAN: float = -GLOBAL_MEAN  # patches with mean below this are over-centered

BASE_DIR = Path(__file__).resolve().parents[2]  # proyecto_2/
PATCHES_DIR = BASE_DIR / "datasets" / "luna_lung_cancer" / "patches"
SPLITS = ("train", "val", "test")
NUM_WORKERS = 8
CHUNK_SIZE = 100  # files per imap_unordered chunk
PROGRESS_EVERY = 500


# ---------------------------------------------------------------------------
# Single-file processing (runs inside worker)
# ---------------------------------------------------------------------------
def _process_one(path_str: str) -> tuple[str, str]:
    """Load a .npy patch, fix over-centering if needed, return status."""
    try:
        arr = np.load(path_str, allow_pickle=False).astype(np.float32)
        mean = float(arr.mean())

        # Guard: NaN / Inf / degenerate arrays → treat as corrupt
        if not np.isfinite(mean):
            return (path_str, "CORRUPT")

        if mean < MIN_VALID_MEAN:
            # Safety limit: at most 10 corrections (prevents infinite loop)
            for _ in range(10):
                arr += GLOBAL_MEAN
                if float(arr.mean()) >= MIN_VALID_MEAN:
                    break
            np.save(path_str, arr)
            return (path_str, "FIXED")

        return (path_str, "OK")

    except Exception:
        return (path_str, "CORRUPT")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"[{_ts()}] fix_zerocentering — START")
    print(f"  GLOBAL_MEAN   = {GLOBAL_MEAN}")
    print(f"  MIN_VALID_MEAN= {MIN_VALID_MEAN}")
    print(f"  PATCHES_DIR   = {PATCHES_DIR}")
    print(f"  SPLITS        = {SPLITS}")
    print(f"  WORKERS       = {NUM_WORKERS}")
    sys.stdout.flush()

    report: dict = {
        "global_mean": GLOBAL_MEAN,
        "splits": {},
        "timestamp": "",
    }

    # Use 'fork' context to avoid forkserver serialization issues on Python 3.14
    ctx = mp.get_context("fork")

    for split in SPLITS:
        split_dir = PATCHES_DIR / split
        if not split_dir.is_dir():
            print(f"[{_ts()}] WARNING: {split_dir} not found, skipping")
            continue

        files = sorted(
            str(split_dir / f) for f in os.listdir(split_dir) if f.endswith(".npy")
        )
        total = len(files)
        print(f"[{_ts()}] Processing split={split}  files={total}")
        sys.stdout.flush()

        counters: dict[str, int] = {"total": total, "fixed": 0, "ok": 0, "corrupt": 0}
        corrupt_files: list[str] = []
        processed = 0

        with ctx.Pool(processes=NUM_WORKERS) as pool:
            for path_str, status in pool.imap_unordered(
                _process_one, files, chunksize=CHUNK_SIZE
            ):
                if status == "FIXED":
                    counters["fixed"] += 1
                elif status == "OK":
                    counters["ok"] += 1
                else:
                    counters["corrupt"] += 1
                    corrupt_files.append(os.path.basename(path_str))

                processed += 1
                if processed % PROGRESS_EVERY == 0:
                    print(
                        f"  [{_ts()}] {split}: {processed}/{total}  "
                        f"fixed={counters['fixed']} ok={counters['ok']} "
                        f"corrupt={counters['corrupt']}"
                    )
                    sys.stdout.flush()

        report["splits"][split] = {**counters, "corrupt_files": corrupt_files}
        print(
            f"[{_ts()}] {split} DONE — "
            f"total={counters['total']} fixed={counters['fixed']} "
            f"ok={counters['ok']} corrupt={counters['corrupt']}"
        )
        sys.stdout.flush()

    report["timestamp"] = datetime.now(timezone.utc).isoformat()

    report_path = PATCHES_DIR / "fix_zerocentering_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[{_ts()}] Report saved → {report_path}")
    print(f"[{_ts()}] fix_zerocentering — COMPLETADO")
    sys.stdout.flush()


def _ts() -> str:
    """Return a compact timestamp for log lines."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()

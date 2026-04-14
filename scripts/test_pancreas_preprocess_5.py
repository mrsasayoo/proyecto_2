#!/usr/bin/env python3
"""Test pancreas preprocessing on 5 unprocessed volumes.

Validates the pipeline works end-to-end and measures per-volume timing.
Uses the existing _pancreas_preprocess_one function from pre_embeddings.py.

Usage:
    /home/mrsasayo_mesa/venv_global/bin/python scripts/test_pancreas_preprocess_5.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from pipeline.fase0.pre_embeddings import (
    _pancreas_preprocess_one,
    PANCREAS_CROP_SIZE,
)


def main() -> None:
    datasets_dir = PROJECT_ROOT / "datasets"
    zenodo_dir = datasets_dir / "zenodo_13715870"
    preprocessed_dir = zenodo_dir / "preprocessed"
    panorama_dir = datasets_dir / "panorama_labels"

    # Collect all NIfTI files
    search_dirs = [zenodo_dir, zenodo_dir / "batch_3", zenodo_dir / "batch_4"]
    all_nii: list[Path] = []
    for d in search_dirs:
        if d.is_dir():
            all_nii.extend(sorted(d.glob("*_0000.nii.gz")))

    # Identify already-processed case_ids
    existing_npys = {p.stem for p in preprocessed_dir.glob("*.npy")}

    # Find unprocessed volumes
    unprocessed: list[tuple[Path, str]] = []
    for nii_path in all_nii:
        stem = nii_path.name.replace(".nii.gz", "")
        case_id = stem[:-5] if stem.endswith("_0000") else stem
        if case_id not in existing_npys:
            unprocessed.append((nii_path, case_id))

    print(f"Total NIfTIs: {len(all_nii)}")
    print(f"Already processed: {len(existing_npys)}")
    print(f"Unprocessed: {len(unprocessed)}")
    print()

    if len(unprocessed) == 0:
        print("Nothing to process!")
        return

    # Take 5 samples spread across the list
    step = max(1, len(unprocessed) // 5)
    test_cases = unprocessed[: 5 * step : step][:5]

    print(f"Testing {len(test_cases)} volumes:")
    for _, cid in test_cases:
        print(f"  - {cid}")
    print()

    timings: list[float] = []
    results: list[dict] = []

    for nii_path, case_id in test_cases:
        out_npy = preprocessed_dir / f"{case_id}.npy"
        print(f"Processing {case_id}...", end=" ", flush=True)

        t0 = time.perf_counter()
        try:
            crop, meta = _pancreas_preprocess_one(
                str(nii_path),
                panorama_dir=panorama_dir,
            )
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)

            # Validate
            ok = True
            issues: list[str] = []
            if crop.shape != (PANCREAS_CROP_SIZE,) * 3:
                issues.append(
                    f"shape={crop.shape} (expected ({PANCREAS_CROP_SIZE},)*3)"
                )
                ok = False
            if crop.dtype != np.float32:
                issues.append(f"dtype={crop.dtype} (expected float32)")
                ok = False
            if np.isnan(crop).any():
                issues.append("contains NaN")
                ok = False
            if float(crop.min()) < 0.0 or float(crop.max()) > 1.0:
                issues.append(f"range=[{crop.min():.4f}, {crop.max():.4f}]")
                ok = False

            result = {
                "case_id": case_id,
                "ok": ok,
                "shape": crop.shape,
                "dtype": str(crop.dtype),
                "min": float(crop.min()),
                "max": float(crop.max()),
                "mean": float(crop.mean()),
                "std": float(crop.std()),
                "elapsed_s": round(elapsed, 2),
                "centroid_strategy": meta.get("centroid_strategy", "unknown"),
                "issues": issues,
            }
            results.append(result)

            # Save only if valid
            if ok:
                np.save(str(out_npy), crop)
                print(
                    f"OK ({elapsed:.1f}s) | mean={crop.mean():.3f} std={crop.std():.3f} | {meta.get('centroid_strategy', '?')}"
                )
            else:
                print(f"ISSUES ({elapsed:.1f}s): {issues}")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)
            print(f"ERROR ({elapsed:.1f}s): {e}")
            results.append(
                {
                    "case_id": case_id,
                    "ok": False,
                    "error": str(e),
                    "elapsed_s": round(elapsed, 2),
                }
            )

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_ok = sum(1 for r in results if r.get("ok", False))
    print(f"  Processed: {len(results)}")
    print(f"  OK:        {n_ok}")
    print(f"  Failed:    {len(results) - n_ok}")
    print()

    if timings:
        avg_time = sum(timings) / len(timings)
        remaining = len(unprocessed) - len(test_cases)
        print(f"  Avg time/volume: {avg_time:.1f}s")
        print(f"  Min time/volume: {min(timings):.1f}s")
        print(f"  Max time/volume: {max(timings):.1f}s")
        print()
        print(f"  Remaining volumes: {remaining}")
        # Estimate with 2 workers (conservative for 7GB RAM)
        for n_workers in [1, 2, 3]:
            est_total_s = (remaining * avg_time) / n_workers
            est_hours = est_total_s / 3600
            print(
                f"  Est. time ({n_workers} workers): {est_total_s:.0f}s = {est_hours:.1f}h"
            )

    print()
    print("Per-volume details:")
    for r in results:
        status = "✅" if r.get("ok") else "❌"
        cid = r["case_id"]
        t = r["elapsed_s"]
        if r.get("ok"):
            print(
                f"  {status} {cid}: {t}s | mean={r['mean']:.3f} std={r['std']:.3f} | {r.get('centroid_strategy', '?')}"
            )
        else:
            err = r.get("error", r.get("issues", ""))
            print(f"  {status} {cid}: {t}s | {err}")


if __name__ == "__main__":
    main()

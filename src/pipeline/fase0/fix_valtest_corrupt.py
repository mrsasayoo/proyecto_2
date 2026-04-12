#!/usr/bin/env python3
"""Fix corrupt patches in LUNA16 val/ and test/ splits.

Corrupt patches contain mostly normal float32 values in [0,1] but have 6-7
voxels with garbage values up to ~7.68×10³¹ (finite but astronomically large,
likely from uninitialized memory during extraction).

Identification criterion: np.mean(patch) > 0.9

For each corrupt patch this script:
  1. Parses the candidate index from the filename (candidate_XXXXXX.npy)
  2. Looks up seriesuid and world coordinates in candidates_V2.csv
  3. Re-extracts the 64³ patch using the inline 7-step pipeline
  4. Validates the result (mean in range, no NaN/Inf)
  5. Overwrites the corrupt file if valid; skips and logs if not

Usage:
    cd /mnt/hdd/datasets/carlos_andres_ferro/proyecto_2
    python src/pipeline/fix_valtest_corrupt.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = Path("/mnt/hdd/datasets/carlos_andres_ferro/proyecto_2")
PATCHES_DIR = BASE / "datasets" / "luna_lung_cancer" / "patches"
CT_DIR = BASE / "datasets" / "luna_lung_cancer" / "ct_volumes"
SEG_DIR = (
    BASE / "datasets" / "luna_lung_cancer" / "seg-lungs-LUNA16" / "seg-lungs-LUNA16"
)
CSV_PATH = (
    BASE / "datasets" / "luna_lung_cancer" / "candidates_V2" / "candidates_V2.csv"
)
REPORT_PATH = PATCHES_DIR / "fix_valtest_report.json"
GLOBAL_MEAN = 0.09921630471944809
PATCH_SIZE = 64
CORRUPT_THRESHOLD = 0.9


# ── Inline 7-step extraction pipeline ────────────────────────────────────
def extract_patch_inline(
    ct_path: str,
    seg_path: str,
    world_x: float,
    world_y: float,
    world_z: float,
) -> np.ndarray:
    """Extract a 64³ patch at 1 mm isotropic, zero-centered.

    Uses SimpleITK + scipy inline — no project imports.
    """
    import SimpleITK as sitk
    from scipy.ndimage import zoom

    # Step 1: Load + HU clamp
    img = sitk.ReadImage(str(ct_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    spacing = img.GetSpacing()  # (x_sp, y_sp, z_sp)
    origin = img.GetOrigin()
    direction = img.GetDirection()
    arr = np.clip(arr, -1000, None)

    # Step 2: Resample to 1×1×1 mm³
    zoom_factors = (spacing[2], spacing[1], spacing[0])  # z, y, x
    arr = zoom(arr, zoom_factors, order=1)

    # Step 3: Load + apply lung mask
    seg_img = sitk.ReadImage(str(seg_path))
    seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.float32)
    seg_arr = zoom(seg_arr, zoom_factors, order=0)
    seg_mask = (seg_arr > 0.5).astype(bool)

    # Handle shape mismatches between resampled CT and resampled mask
    min_z = min(arr.shape[0], seg_mask.shape[0])
    min_y = min(arr.shape[1], seg_mask.shape[1])
    min_x = min(arr.shape[2], seg_mask.shape[2])
    arr[:min_z, :min_y, :min_x][~seg_mask[:min_z, :min_y, :min_x]] = -1000.0
    if min_z < arr.shape[0]:
        arr[min_z:, :, :] = -1000.0
    if min_y < arr.shape[1]:
        arr[:, min_y:, :] = -1000.0
    if min_x < arr.shape[2]:
        arr[:, :, min_x:] = -1000.0

    # Step 4: Clip HU
    arr = np.clip(arr, -1000, 400)

    # Step 5: Min-max normalize to [0, 1]
    arr = (arr - (-1000.0)) / (400.0 - (-1000.0))
    arr = arr.astype(np.float32)

    # Step 6: Zero-center
    arr -= GLOBAL_MEAN

    # Step 7: Extract 64×64×64 patch
    new_img = sitk.GetImageFromArray(arr)
    new_img.SetOrigin(origin)
    new_img.SetDirection(direction)
    new_img.SetSpacing((1.0, 1.0, 1.0))

    idx = new_img.TransformPhysicalPointToIndex(
        (float(world_x), float(world_y), float(world_z))
    )
    # idx is (x_idx, y_idx, z_idx); arr is (Z, Y, X)
    vx, vy, vz = idx[0], idx[1], idx[2]

    half = PATCH_SIZE // 2
    patch = np.full((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), 0.0, dtype=np.float32)

    z_shape, y_shape, x_shape = arr.shape

    src_z0 = max(0, vz - half)
    src_z1 = min(z_shape, vz + half)
    src_y0 = max(0, vy - half)
    src_y1 = min(y_shape, vy + half)
    src_x0 = max(0, vx - half)
    src_x1 = min(x_shape, vx + half)

    dst_z0 = src_z0 - (vz - half)
    dst_z1 = dst_z0 + (src_z1 - src_z0)
    dst_y0 = src_y0 - (vy - half)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - (vx - half)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    patch[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = arr[
        src_z0:src_z1, src_y0:src_y1, src_x0:src_x1
    ]

    return patch


# ── Helpers ───────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def is_valid_patch(patch: np.ndarray) -> bool:
    """Validate that a re-extracted patch is sane."""
    if patch.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
        return False
    if np.any(np.isnan(patch)):
        return False
    if np.any(np.isinf(patch)):
        return False
    m = float(np.mean(patch))
    if m < -0.099 or m > CORRUPT_THRESHOLD:
        return False
    return True


def scan_corrupt(split_dir: Path) -> list[str]:
    """Return filenames with mean > CORRUPT_THRESHOLD."""
    corrupt: list[str] = []
    for f in sorted(split_dir.glob("*.npy")):
        arr = np.load(str(f))
        if float(np.mean(arr)) > CORRUPT_THRESHOLD:
            corrupt.append(f.name)
    return corrupt


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    print(f"[{_ts()}] === FIX VAL/TEST CORRUPT PATCHES ===")
    print(f"  GLOBAL_MEAN   = {GLOBAL_MEAN}")
    print(f"  THRESHOLD     = mean > {CORRUPT_THRESHOLD}")
    sys.stdout.flush()

    # ── Step 1: Build seriesuid → .mhd path map ─────────────────────────
    mhd_map: dict[str, str] = {}
    for subset_idx in range(10):
        subset_dir = CT_DIR / f"subset{subset_idx}"
        if subset_dir.exists():
            for mhd_file in subset_dir.glob("*.mhd"):
                mhd_map[mhd_file.stem] = str(mhd_file)
    print(f"[{_ts()}] CTs available: {len(mhd_map)}")
    sys.stdout.flush()

    # ── Step 2: Load candidates CSV ─────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"[{_ts()}] Candidates CSV: {len(df)} rows")
    sys.stdout.flush()

    # ── Step 3: Scan for corrupt patches ────────────────────────────────
    report: dict[str, dict] = {}

    for split in ["val", "test"]:
        split_dir = PATCHES_DIR / split
        print(f"\n[{_ts()}] Scanning {split}/ for corrupt patches ...")
        sys.stdout.flush()

        corrupt_files = scan_corrupt(split_dir)
        n_corrupt = len(corrupt_files)
        print(f"  Found {n_corrupt} corrupt patches in {split}/")
        sys.stdout.flush()

        fixed = 0
        failed = 0
        failed_files: list[str] = []

        for i, fname in enumerate(corrupt_files):
            idx = int(fname.replace("candidate_", "").replace(".npy", ""))

            try:
                if idx >= len(df):
                    reason = f"index {idx} out of range ({len(df)})"
                    print(f"  [{i + 1}/{n_corrupt}] FAIL {fname}: {reason}")
                    failed += 1
                    failed_files.append(fname)
                    continue

                row = df.iloc[idx]
                uid = str(row["seriesuid"])
                wx = float(row["coordX"])
                wy = float(row["coordY"])
                wz = float(row["coordZ"])

                if uid not in mhd_map:
                    reason = f"seriesuid not found in CT subsets"
                    print(f"  [{i + 1}/{n_corrupt}] FAIL {fname}: {reason}")
                    failed += 1
                    failed_files.append(fname)
                    continue

                ct_path = mhd_map[uid]
                seg_path = str(SEG_DIR / f"{uid}.mhd")

                if not Path(seg_path).exists():
                    reason = "lung segmentation mask not found"
                    print(f"  [{i + 1}/{n_corrupt}] FAIL {fname}: {reason}")
                    failed += 1
                    failed_files.append(fname)
                    continue

                # Re-extract
                new_patch = extract_patch_inline(ct_path, seg_path, wx, wy, wz)

                # Validate
                if not is_valid_patch(new_patch):
                    m = float(np.mean(new_patch))
                    reason = (
                        f"re-extracted patch invalid "
                        f"(shape={new_patch.shape}, mean={m:.6f})"
                    )
                    print(f"  [{i + 1}/{n_corrupt}] FAIL {fname}: {reason}")
                    failed += 1
                    failed_files.append(fname)
                    continue

                # Overwrite
                out_path = split_dir / fname
                np.save(str(out_path), new_patch)
                fixed += 1

                new_mean = float(np.mean(new_patch))
                print(f"  [{i + 1}/{n_corrupt}] FIXED {fname}: new mean={new_mean:.6f}")

            except Exception as e:
                print(f"  [{i + 1}/{n_corrupt}] ERROR {fname}: {e}")
                failed += 1
                failed_files.append(fname)

            sys.stdout.flush()

        report[split] = {
            "corrupt_found": n_corrupt,
            "fixed": fixed,
            "failed": failed,
            "failed_files": failed_files,
        }
        print(
            f"[{_ts()}] {split} done: "
            f"corrupt={n_corrupt}, fixed={fixed}, failed={failed}"
        )
        sys.stdout.flush()

    # ── Step 4: Post-fix validation ─────────────────────────────────────
    print(f"\n[{_ts()}] === POST-FIX VALIDATION ===")
    remaining: dict[str, list[str]] = {}
    total_remaining = 0

    for split in ["val", "test"]:
        split_dir = PATCHES_DIR / split
        still_corrupt = scan_corrupt(split_dir)
        remaining[split] = still_corrupt
        total_remaining += len(still_corrupt)
        print(f"  {split}: {len(still_corrupt)} remaining corrupt")
        if still_corrupt:
            for f in still_corrupt:
                print(f"    {f}")
    sys.stdout.flush()

    # ── Step 5: Build and save report ───────────────────────────────────
    total_corrupt = sum(r["corrupt_found"] for r in report.values())
    total_fixed = sum(r["fixed"] for r in report.values())
    total_failed = sum(r["failed"] for r in report.values())

    final_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "val": report["val"],
        "test": report["test"],
        "total_corrupt": total_corrupt,
        "total_fixed": total_fixed,
        "total_failed": total_failed,
        "remaining_corrupt_after_fix": {
            "val": remaining["val"],
            "test": remaining["test"],
            "total": total_remaining,
        },
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(final_report, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[{_ts()}] Report saved: {REPORT_PATH}")
    print(f"[{_ts()}] Elapsed: {elapsed:.1f}s")
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total corrupt found:  {total_corrupt}")
    print(f"  Total fixed:          {total_fixed}")
    print(f"  Total failed:         {total_failed}")
    print(f"  Remaining after fix:  {total_remaining}")
    print(f"{'=' * 60}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

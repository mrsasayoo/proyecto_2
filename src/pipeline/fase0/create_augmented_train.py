#!/usr/bin/env python3
"""Create augmented training set for LUNA16.

Copies all negatives as-is and generates augmented versions of positive
patches to reduce class imbalance from ~10:1 to ~2:1.

Only loadable (non-corrupt) patches are included in the output.

Output directory: datasets/luna_lung_cancer/patches/train_aug/
Manifest CSV:     datasets/luna_lung_cancer/patches/train_aug_manifest.csv
Report JSON:      datasets/luna_lung_cancer/patches/train_aug_report.json

Usage:
    python src/pipeline/create_augmented_train.py
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.ndimage import rotate as scipy_rotate
from scipy.ndimage import zoom as scipy_zoom

# Suppress numpy warnings from edge-case patches
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_RATIO = 2  # desired neg:pos ratio
GLOBAL_MEAN: float = 0.09921630471944809

BASE_DIR = Path(__file__).resolve().parents[3]  # proyecto_2/
PATCHES_DIR = BASE_DIR / "datasets" / "luna_lung_cancer" / "patches"
TRAIN_DIR = PATCHES_DIR / "train"
OUTPUT_DIR = PATCHES_DIR / "train_aug"
CANDIDATES_CSV = (
    BASE_DIR / "datasets" / "luna_lung_cancer" / "candidates_V2" / "candidates_V2.csv"
)

PROGRESS_EVERY = 500


# ---------------------------------------------------------------------------
# Augmentation pipeline (self-contained — does NOT import luna.py)
# ---------------------------------------------------------------------------
def augment_patch(volume: np.ndarray) -> np.ndarray:
    """Apply a stochastic augmentation pipeline to a 64x64x64 float32 patch."""
    D = volume.shape[0]  # 64

    # --- Flips (P=0.5 per axis) ---
    for axis in range(3):
        if random.random() < 0.5:
            volume = np.flip(volume, axis=axis)

    # --- Rotation +/-15 deg in 3 planes ---
    for axes in [(1, 2), (0, 2), (0, 1)]:
        angle = random.uniform(-15.0, 15.0)
        if abs(angle) > 0.5:
            volume = scipy_rotate(
                volume, angle=angle, axes=axes, reshape=False, order=1, mode="nearest"
            )

    # --- Zoom [0.80, 1.20] with crop/pad ---
    zoom_factor = random.uniform(0.80, 1.20)
    zoomed = scipy_zoom(volume, zoom_factor, order=1)
    zD = zoomed.shape[0]
    if zD > D:
        start = (zD - D) // 2
        volume = zoomed[start : start + D, start : start + D, start : start + D]
    elif zD < D:
        pb = (D - zD) // 2
        pa = D - zD - pb
        volume = np.pad(zoomed, ((pb, pa),) * 3, constant_values=0.0)
    else:
        volume = zoomed

    # --- Translation +/-4 voxels ---
    for axis in range(3):
        shift = random.randint(-4, 4)
        if shift == 0:
            continue
        shifted = np.zeros_like(volume)
        s = volume.shape[0]
        slices_src = [slice(None)] * 3
        slices_dst = [slice(None)] * 3
        if shift > 0:
            slices_src[axis] = slice(0, s - shift)
            slices_dst[axis] = slice(shift, s)
        else:
            slices_src[axis] = slice(-shift, s)
            slices_dst[axis] = slice(0, s + shift)
        shifted[tuple(slices_dst)] = volume[tuple(slices_src)]
        volume = shifted

    # --- Elastic deformation (P=0.5) ---
    if random.random() < 0.5:
        sigma = random.uniform(1.0, 3.0)
        alpha = random.uniform(0.0, 5.0)
        shape = volume.shape
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        zg, yg, xg = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )
        volume = map_coordinates(
            volume, [zg + dz, yg + dy, xg + dx], order=1, mode="nearest"
        ).astype(np.float32)

    # --- Gaussian noise (P=0.5) ---
    if random.random() < 0.5:
        sigma_n = random.uniform(0.0, 25.0) / 1400.0
        volume = volume + np.random.normal(0.0, sigma_n, volume.shape).astype(
            np.float32
        )

    # --- Brightness / contrast ---
    scale = random.uniform(0.9, 1.1)
    offset = random.uniform(-20.0, 20.0) / 1400.0
    volume = volume * scale + offset

    # --- Gaussian blur (P=0.5) ---
    if random.random() < 0.5:
        sigma_b = random.uniform(0.1, 0.5)
        volume = gaussian_filter(volume, sigma=sigma_b).astype(np.float32)

    return np.clip(
        np.ascontiguousarray(volume, dtype=np.float32),
        -GLOBAL_MEAN,
        1.0 - GLOBAL_MEAN,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _flush() -> None:
    sys.stdout.flush()


def _idx_from_filename(fname: str) -> int | None:
    """Extract integer index from 'candidate_XXXXXX.npy'."""
    m = re.match(r"candidate_(\d+)\.npy$", fname)
    return int(m.group(1)) if m else None


def _safe_load(path: Path) -> np.ndarray | None:
    """Try to load a .npy file. Return None if corrupt."""
    try:
        arr = np.load(str(path), allow_pickle=False).astype(np.float32)
        if arr.shape != (64, 64, 64):
            return None
        if not np.isfinite(arr.mean()):
            return None
        return arr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"[{_ts()}] create_augmented_train — START")
    print(f"  TARGET_RATIO  = {TARGET_RATIO}")
    print(f"  TRAIN_DIR     = {TRAIN_DIR}")
    print(f"  OUTPUT_DIR    = {OUTPUT_DIR}")
    _flush()

    # ---- Load labels ---------------------------------------------------
    df = pd.read_csv(CANDIDATES_CSV)
    print(f"[{_ts()}] Loaded candidates CSV: {len(df)} rows")

    # ---- Load corrupt list from fix_zerocentering report ---------------
    corrupt_set: set[str] = set()
    zc_report_path = PATCHES_DIR / "fix_zerocentering_report.json"
    if zc_report_path.exists():
        with open(zc_report_path) as f:
            zc_report = json.load(f)
        corrupt_set = set(
            zc_report.get("splits", {}).get("train", {}).get("corrupt_files", [])
        )
        print(
            f"[{_ts()}] Loaded {len(corrupt_set)} known corrupt files from zerocentering report"
        )

    # ---- Classify train files ------------------------------------------
    all_files = sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".npy"))
    pos_files: list[str] = []
    neg_files: list[str] = []
    skipped: list[str] = []

    for fname in all_files:
        # Skip known corrupt files
        if fname in corrupt_set:
            skipped.append(fname)
            continue
        idx = _idx_from_filename(fname)
        if idx is None or idx >= len(df):
            skipped.append(fname)
            continue
        label = int(df.iloc[idx]["class"])
        if label == 1:
            pos_files.append(fname)
        else:
            neg_files.append(fname)

    n_pos = len(pos_files)
    n_neg = len(neg_files)
    print(
        f"[{_ts()}] Train: pos={n_pos}  neg={n_neg}  skipped={len(skipped)} (corrupt/invalid)"
    )
    print(f"[{_ts()}] Current ratio neg:pos = {n_neg / max(n_pos, 1):.1f}:1")

    # ---- Compute augmentations -----------------------------------------
    target_pos = n_neg // TARGET_RATIO
    aug_per_pos = max(1, (target_pos - n_pos) // n_pos)
    total_pos_after = n_pos + n_pos * aug_per_pos
    total_files_expected = n_neg + total_pos_after
    print(f"[{_ts()}] aug_per_pos={aug_per_pos}  target total_pos={total_pos_after}")
    print(f"[{_ts()}] Expected final ratio = {n_neg / total_pos_after:.2f}:1")
    print(f"[{_ts()}] Expected total files in train_aug = {total_files_expected}")
    _flush()

    # ---- Prepare output directory --------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Manifest CSV --------------------------------------------------
    manifest_path = PATCHES_DIR / "train_aug_manifest.csv"
    manifest_f = open(manifest_path, "w", newline="")
    writer = csv.writer(manifest_f)
    writer.writerow(["filename", "label", "is_augmented", "source_file"])

    count_written = 0
    count_neg = 0
    count_pos_orig = 0
    count_pos_aug = 0
    corrupt_files: list[str] = []
    progress_counter = 0  # tracks total attempted for progress prints

    # ---- 1. Copy negatives (validated) ---------------------------------
    print(f"[{_ts()}] Copying {n_neg} negative patches ...")
    _flush()
    for fname in neg_files:
        src = TRAIN_DIR / fname
        dst = OUTPUT_DIR / fname
        try:
            # Validate by loading (ensures proper .npy format)
            arr = np.load(str(src), allow_pickle=False)
            if arr.shape != (64, 64, 64):
                raise ValueError(f"Bad shape: {arr.shape}")
            shutil.copy2(src, dst)
            writer.writerow([fname, 0, False, fname])
            count_neg += 1
            count_written += 1
        except Exception:
            corrupt_files.append(fname)

        progress_counter += 1
        if progress_counter % PROGRESS_EVERY == 0:
            total_pos_so_far = count_pos_orig + count_pos_aug
            print(
                f"  [{_ts()}] progress: {progress_counter}/{total_files_expected}  "
                f"neg={count_neg} pos_orig={count_pos_orig} pos_aug={count_pos_aug}  "
                f"ratio={count_neg / max(total_pos_so_far, 1):.2f}:1"
            )
            _flush()

    # ---- 2. Copy positive originals + generate augmentations -----------
    print(
        f"[{_ts()}] Processing {n_pos} positive patches (aug_per_pos={aug_per_pos}) ..."
    )
    _flush()
    for fname in pos_files:
        src_path = TRAIN_DIR / fname
        arr = _safe_load(src_path)
        if arr is None:
            corrupt_files.append(fname)
            progress_counter += 1 + aug_per_pos  # account for skipped augs
            continue

        # Copy original
        dst = OUTPUT_DIR / fname
        np.save(str(dst), arr)
        writer.writerow([fname, 1, False, fname])
        count_pos_orig += 1
        count_written += 1
        progress_counter += 1

        if progress_counter % PROGRESS_EVERY == 0:
            total_pos_so_far = count_pos_orig + count_pos_aug
            print(
                f"  [{_ts()}] progress: {progress_counter}/{total_files_expected}  "
                f"neg={count_neg} pos_orig={count_pos_orig} pos_aug={count_pos_aug}  "
                f"ratio={count_neg / max(total_pos_so_far, 1):.2f}:1"
            )
            _flush()

        # Generate augmented copies
        stem = fname.replace(".npy", "")
        for aug_i in range(1, aug_per_pos + 1):
            aug_name = f"{stem}_aug{aug_i}.npy"
            aug_arr = augment_patch(arr.copy())
            np.save(str(OUTPUT_DIR / aug_name), aug_arr)
            writer.writerow([aug_name, 1, True, fname])
            count_pos_aug += 1
            count_written += 1
            progress_counter += 1

            if progress_counter % PROGRESS_EVERY == 0:
                total_pos_so_far = count_pos_orig + count_pos_aug
                print(
                    f"  [{_ts()}] progress: {progress_counter}/{total_files_expected}  "
                    f"neg={count_neg} pos_orig={count_pos_orig} pos_aug={count_pos_aug}  "
                    f"ratio={count_neg / max(total_pos_so_far, 1):.2f}:1"
                )
                _flush()

    manifest_f.close()

    # ---- Final stats ---------------------------------------------------
    total_pos_final = count_pos_orig + count_pos_aug
    final_ratio = count_neg / max(total_pos_final, 1)

    print(f"\n[{_ts()}] ============ FINAL STATS ============")
    print(f"  Negatives copied:       {count_neg}")
    print(f"  Positives (original):   {count_pos_orig}")
    print(f"  Positives (augmented):  {count_pos_aug}")
    print(f"  Total positives:        {total_pos_final}")
    print(f"  Total files written:    {count_written}")
    print(f"  Final ratio neg:pos:    {final_ratio:.2f}:1")
    print(f"  Corrupt files skipped:  {len(corrupt_files)}")
    if corrupt_files:
        print(f"  Corrupt list (first 20): {corrupt_files[:20]}")
    print(f"  Manifest:               {manifest_path}")

    # ---- Report JSON ---------------------------------------------------
    report = {
        "target_ratio": TARGET_RATIO,
        "aug_per_pos": aug_per_pos,
        "negatives": count_neg,
        "positives_original": count_pos_orig,
        "positives_augmented": count_pos_aug,
        "total_positives": total_pos_final,
        "total_files": count_written,
        "final_ratio": round(final_ratio, 4),
        "corrupt_files": corrupt_files,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    report_path = PATCHES_DIR / "train_aug_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{_ts()}] Report saved → {report_path}")
    print(f"[{_ts()}] create_augmented_train — COMPLETADO")
    _flush()


if __name__ == "__main__":
    main()

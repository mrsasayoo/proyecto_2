#!/usr/bin/env python3
"""Re-extract 757 corrupt .npy patches from LUNA16 CT source volumes.

Corrupt patches have size 1,048,704 bytes but no valid numpy magic header
(\x93NUMPY) — they contain garbage data from a crashed extraction process.

This script:
  1. Reads the list of corrupt files from fix_zerocentering_report.json
  2. For each corrupt file, looks up the candidate index, loads the source CT,
     and re-runs the full 7-step extraction pipeline + zero-centering
  3. Overwrites the corrupt .npy in the corresponding split directory
  4. For train positives: also writes to train_aug/ and generates 3 augmented
     copies, then appends to train_aug_manifest.csv

Processing is sequential (SimpleITK is not thread-safe for reading the same
.mhd files concurrently). Estimated time: 2-4 hours for 757 patches.

Usage:
    cd /mnt/hdd/datasets/carlos_andres_ferro/proyecto_2
    nohup python src/pipeline/reextract_corrupt.py > logs/reextract.log 2>&1 &
"""

from __future__ import annotations

import csv
import json
import sys
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
REPORT_PATH = PATCHES_DIR / "fix_zerocentering_report.json"
GLOBAL_MEAN = float(np.load(PATCHES_DIR / "global_mean.npy"))
TRAIN_AUG_DIR = PATCHES_DIR / "train_aug"
TRAIN_AUG_MANIFEST = PATCHES_DIR / "train_aug_manifest.csv"

# Augmentation settings — must match create_augmented_train.py
AUG_PER_POS = 3


# ── Augmentation pipeline (copied from create_augmented_train.py) ─────────
def augment_patch(volume: np.ndarray) -> np.ndarray:
    """Apply stochastic augmentation pipeline to a 64x64x64 float32 patch."""
    import random

    from scipy.ndimage import gaussian_filter, map_coordinates
    from scipy.ndimage import rotate as scipy_rotate
    from scipy.ndimage import zoom as scipy_zoom

    D = volume.shape[0]

    for axis in range(3):
        if random.random() < 0.5:
            volume = np.flip(volume, axis=axis)

    for axes in [(1, 2), (0, 2), (0, 1)]:
        angle = random.uniform(-15.0, 15.0)
        if abs(angle) > 0.5:
            volume = scipy_rotate(
                volume, angle=angle, axes=axes, reshape=False, order=1, mode="nearest"
            )

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

    if random.random() < 0.5:
        sigma_n = random.uniform(0.0, 25.0) / 1400.0
        volume = volume + np.random.normal(0.0, sigma_n, volume.shape).astype(
            np.float32
        )

    scale = random.uniform(0.9, 1.1)
    offset = random.uniform(-20.0, 20.0) / 1400.0
    volume = volume * scale + offset

    if random.random() < 0.5:
        sigma_b = random.uniform(0.1, 0.5)
        volume = gaussian_filter(volume, sigma=sigma_b).astype(np.float32)

    return np.clip(np.ascontiguousarray(volume, dtype=np.float32), 0.0, 1.0)


# ── Extraction function ───────────────────────────────────────────────────
def extract_and_center(
    mhd_path: str, coord_world: list, seg_dir: Path, global_mean: float
) -> np.ndarray:
    """Extract a 64^3 patch at 1mm isotropic and apply zero-centering.

    Tries the project's LUNA16PatchExtractor first; falls back to an inline
    implementation if the import chain fails (it pulls torch, config, etc.).
    """
    try:
        sys.path.insert(0, str(BASE / "src" / "pipeline"))
        sys.path.insert(0, str(BASE / "src" / "pipeline" / "datasets"))
        from datasets.luna import LUNA16PatchExtractor

        patch = LUNA16PatchExtractor.extract(
            mhd_path=mhd_path,
            coord_world=coord_world,
            seg_dir=str(seg_dir),
            patch_size=64,
            clip_hu=(-1000, 400),
        )
    except Exception:
        patch = _extract_inline(mhd_path, coord_world, seg_dir)

    return patch.astype(np.float32) - global_mean


def _extract_inline(
    mhd_path: str,
    coord_world: list,
    seg_dir: Path,
    patch_size: int = 64,
    clip_hu: tuple = (-1000, 400),
) -> np.ndarray:
    """Inline 7-step extraction pipeline (no project imports)."""
    import SimpleITK as sitk
    from scipy.ndimage import zoom as scipy_zoom

    # Step 1: Load + HU
    image = sitk.ReadImage(str(mhd_path))
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direc = np.array(image.GetDirection())
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    array[array < -1000] = -1000.0

    # Step 2: Isotropic resampling to 1x1x1 mm^3
    zoom_factors = (spacing[2], spacing[1], spacing[0])
    array = scipy_zoom(array, zoom_factors, order=1)

    # Step 3: Lung segmentation mask
    uid = Path(mhd_path).stem
    seg_path = Path(seg_dir) / (uid + ".mhd")
    if seg_path.exists():
        mask_img = sitk.ReadImage(str(seg_path))
        mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.float32)
        mask_arr = scipy_zoom(mask_arr, zoom_factors, order=0)
        mask_arr = (mask_arr > 0.5).astype(np.uint8)
        min_z = min(array.shape[0], mask_arr.shape[0])
        min_y = min(array.shape[1], mask_arr.shape[1])
        min_x = min(array.shape[2], mask_arr.shape[2])
        array[:min_z, :min_y, :min_x][mask_arr[:min_z, :min_y, :min_x] == 0] = -1000.0
        if min_z < array.shape[0]:
            array[min_z:, :, :] = -1000.0
        if min_y < array.shape[1]:
            array[:, min_y:, :] = -1000.0
        if min_x < array.shape[2]:
            array[:, :, min_x:] = -1000.0

    # Step 4: Clip HU
    array = np.clip(array, clip_hu[0], clip_hu[1])

    # Step 5: Min-max normalization [0, 1]
    array = (array - clip_hu[0]) / (clip_hu[1] - clip_hu[0])

    # Step 7: Extract patch
    half = patch_size // 2
    cx, cy, cz = coord_world
    coord_shifted = np.array([cx, cy, cz]) - origin
    coord_voxel = np.linalg.solve(
        np.array(direc).reshape(3, 3) * spacing, coord_shifted
    )
    iz_orig, iy_orig, ix_orig = np.round(coord_voxel[::-1]).astype(int)
    iz = int(round(iz_orig * spacing[2]))
    iy = int(round(iy_orig * spacing[1]))
    ix = int(round(ix_orig * spacing[0]))

    z1, z2 = max(0, iz - half), min(array.shape[0], iz + half)
    y1, y2 = max(0, iy - half), min(array.shape[1], iy + half)
    x1, x2 = max(0, ix - half), min(array.shape[2], ix + half)
    raw = array[z1:z2, y1:y2, x1:x2]

    pb_z, pa_z = max(0, half - iz), max(0, (iz + half) - array.shape[0])
    pb_y, pa_y = max(0, half - iy), max(0, (iy + half) - array.shape[1])
    pb_x, pa_x = max(0, half - ix), max(0, (ix + half) - array.shape[2])
    if any(p > 0 for p in [pb_z, pa_z, pb_y, pa_y, pb_x, pa_x]):
        raw = np.pad(
            raw,
            ((pb_z, pa_z), (pb_y, pa_y), (pb_x, pa_x)),
            constant_values=0.0,
        )

    return raw.astype(np.float32)


# ── Helpers ───────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _flush():
    sys.stdout.flush()


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[{_ts()}] === RE-EXTRACTION OF 757 CORRUPT PATCHES ===")
    print(f"  GLOBAL_MEAN = {GLOBAL_MEAN}")
    print(f"  AUG_PER_POS = {AUG_PER_POS}")
    _flush()

    # ── Step 1: Load corrupt file list ────────────────────────────────────
    with open(REPORT_PATH) as f:
        report = json.load(f)

    total_corrupt = sum(
        len(report["splits"][s]["corrupt_files"]) for s in ["train", "val", "test"]
    )
    print(f"[{_ts()}] Corrupt files loaded: {total_corrupt}")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {len(report['splits'][s]['corrupt_files'])}")
    _flush()

    # ── Step 2: Build seriesuid -> .mhd path map ─────────────────────────
    mhd_map: dict[str, str] = {}
    for subset_idx in range(10):
        subset_dir = CT_DIR / f"subset{subset_idx}"
        if subset_dir.exists():
            for mhd_file in subset_dir.glob("*.mhd"):
                mhd_map[mhd_file.stem] = str(mhd_file)
    print(f"[{_ts()}] CTs available: {len(mhd_map)}")
    _flush()

    # ── Step 3: Load candidates CSV ──────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"[{_ts()}] Candidates CSV: {len(df)} rows")
    _flush()

    # ── Step 4: Identify train positives needing train_aug update ─────────
    train_corrupt = report["splits"]["train"]["corrupt_files"]
    train_pos_files: list[str] = []
    for fname in train_corrupt:
        idx = int(fname.replace("candidate_", "").replace(".npy", ""))
        if idx < len(df) and int(df.iloc[idx]["class"]) == 1:
            train_pos_files.append(fname)
    print(f"[{_ts()}] Train positives to add to train_aug: {len(train_pos_files)}")
    _flush()

    # ── Step 5: Process each split ───────────────────────────────────────
    results: dict[str, dict[str, int]] = {
        "train": {"ok": 0, "skip": 0, "error": 0},
        "val": {"ok": 0, "skip": 0, "error": 0},
        "test": {"ok": 0, "skip": 0, "error": 0},
    }

    # Cache: avoid reloading the same CT volume for consecutive candidates
    # from the same seriesuid. We only cache one CT at a time since they
    # are large (~400 MB resampled).
    _cache_uid: str = ""
    train_aug_new_rows: list[list] = []  # rows to append to manifest

    for split in ["train", "val", "test"]:
        corrupt_files = report["splits"][split]["corrupt_files"]
        split_dir = PATCHES_DIR / split

        print(f"\n[{_ts()}] Processing {split}: {len(corrupt_files)} files ...")
        _flush()

        for i, fname in enumerate(corrupt_files):
            idx = int(fname.replace("candidate_", "").replace(".npy", ""))

            try:
                if idx >= len(df):
                    print(
                        f"  [{i + 1}/{len(corrupt_files)}] SKIP {fname}: index {idx} out of range"
                    )
                    results[split]["skip"] += 1
                    continue

                row = df.iloc[idx]
                uid = row["seriesuid"]
                coords = [
                    float(row["coordX"]),
                    float(row["coordY"]),
                    float(row["coordZ"]),
                ]
                label = int(row["class"])

                if uid not in mhd_map:
                    print(
                        f"  [{i + 1}/{len(corrupt_files)}] SKIP {fname}: UID not in CTs"
                    )
                    results[split]["skip"] += 1
                    continue

                # Extract patch
                patch = extract_and_center(mhd_map[uid], coords, SEG_DIR, GLOBAL_MEAN)

                # Validate shape
                if patch.shape != (64, 64, 64):
                    print(
                        f"  [{i + 1}/{len(corrupt_files)}] ERROR {fname}: bad shape {patch.shape}"
                    )
                    results[split]["error"] += 1
                    continue

                # Write to split directory
                out_path = split_dir / fname
                np.save(str(out_path), patch)
                results[split]["ok"] += 1

                # If train positive, also update train_aug
                if split == "train" and label == 1 and TRAIN_AUG_DIR.exists():
                    # Save original (zero-centered) to train_aug — matches train/
                    np.save(str(TRAIN_AUG_DIR / fname), patch)
                    train_aug_new_rows.append([fname, 1, False, fname])

                    # Generate augmented copies.
                    # augment_patch() expects and outputs [0,1] range (clips at the end).
                    # Undo zero-centering before augmentation, keep [0,1] output as-is
                    # to match existing augmented copies in train_aug/.
                    patch_01 = patch + GLOBAL_MEAN  # undo zero-centering → [0, 1]
                    stem = fname.replace(".npy", "")
                    for aug_i in range(1, AUG_PER_POS + 1):
                        aug_name = f"{stem}_aug{aug_i}.npy"
                        aug_arr = augment_patch(patch_01.copy())  # output in [0, 1]
                        np.save(str(TRAIN_AUG_DIR / aug_name), aug_arr)
                        train_aug_new_rows.append([aug_name, 1, True, fname])

                # Progress
                if (i + 1) % 50 == 0 or (i + 1) == len(corrupt_files):
                    r = results[split]
                    print(
                        f"  [{_ts()}] {i + 1}/{len(corrupt_files)} | "
                        f"ok={r['ok']} skip={r['skip']} err={r['error']}"
                    )
                    _flush()

            except Exception as e:
                print(f"  [{i + 1}/{len(corrupt_files)}] ERROR {fname}: {e}")
                results[split]["error"] += 1

    # ── Step 6: Append new rows to train_aug_manifest.csv ─────────────────
    if train_aug_new_rows and TRAIN_AUG_MANIFEST.exists():
        print(
            f"\n[{_ts()}] Appending {len(train_aug_new_rows)} rows to train_aug_manifest.csv ..."
        )
        with open(TRAIN_AUG_MANIFEST, "a", newline="") as f:
            writer = csv.writer(f)
            for row in train_aug_new_rows:
                writer.writerow(row)
        print(
            f"  Done. Added {len(train_aug_new_rows)} rows ({len(train_pos_files)} originals + {len(train_pos_files) * AUG_PER_POS} augmented)"
        )
    elif train_aug_new_rows:
        print(
            f"\n[{_ts()}] WARNING: train_aug_manifest.csv not found, skipping manifest update"
        )

    # ── Step 7: Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RE-EXTRACTION SUMMARY")
    print(f"{'=' * 60}")
    total_ok = sum(r["ok"] for r in results.values())
    total_skip = sum(r["skip"] for r in results.values())
    total_err = sum(r["error"] for r in results.values())
    for split, r in results.items():
        print(
            f"  {split:5s}: ok={r['ok']:4d}  skip={r['skip']:3d}  error={r['error']:3d}"
        )
    print(
        f"  {'TOTAL':5s}: ok={total_ok:4d}  skip={total_skip:3d}  error={total_err:3d}"
    )
    print(f"  train_aug rows added: {len(train_aug_new_rows)}")
    print(f"{'=' * 60}")

    # ── Save report ──────────────────────────────────────────────────────
    report_out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "global_mean": GLOBAL_MEAN,
        "aug_per_pos": AUG_PER_POS,
        "total_corrupt_input": total_corrupt,
        "results": results,
        "totals": {"ok": total_ok, "skip": total_skip, "error": total_err},
        "train_aug_rows_added": len(train_aug_new_rows),
        "train_positives_reextracted": [f for f in train_pos_files],
    }
    report_path = PATCHES_DIR / "reextract_report.json"
    with open(report_path, "w") as f:
        json.dump(report_out, f, indent=2)
    print(f"\n[{_ts()}] Report saved: {report_path}")
    print(f"[{_ts()}] COMPLETADO")
    _flush()


if __name__ == "__main__":
    main()

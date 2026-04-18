#!/usr/bin/env python3
"""
pre_chestxray14.py — Fase 1 offline: preprocesamiento completo de NIH ChestX-ray14
===================================================================================
Pipeline: carga grayscale → validación → resize → CLAHE → float32 → guardado .npy + metadatos

Ejecutar como script standalone:
    python src/pipeline/fase0/pre_chestxray14.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fase1.pre_chestxray14")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELS_14: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

TARGET_SIZE: int = 256
MIN_DIM: int = 800
CLAHE_CLIP: float = 2.0
CLAHE_TILE: tuple[int, int] = (8, 8)

SPLIT_NAMES: dict[str, str] = {
    "train": "nih_train_list.txt",
    "val": "nih_val_list.txt",
    "test": "nih_test_list.txt",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class ProcessingStats:
    """Accumulates per-split processing statistics."""

    processed: int = 0
    skipped_exists: int = 0
    skipped_corrupt: int = 0
    skipped_small: int = 0
    skipped_missing: int = 0


def _build_label_lookup(csv_path: Path) -> dict[str, list[int]]:
    """Parse Data_Entry_2017.csv into {filename: 14-dim binary vector}."""
    df = pd.read_csv(csv_path)
    label_to_idx = {label: i for i, label in enumerate(LABELS_14)}
    lookup: dict[str, list[int]] = {}
    for fname, findings in zip(df["Image Index"], df["Finding Labels"]):
        vec = [0] * 14
        for lbl in str(findings).split("|"):
            lbl = lbl.strip()
            if lbl in label_to_idx:
                vec[label_to_idx[lbl]] = 1
        lookup[str(fname)] = vec
    return lookup


def _find_raw_image(nih_dir: Path, filename: str) -> Path | None:
    """Locate a raw image across images_001..images_012 subdirectories."""
    for subdir in sorted(nih_dir.glob("images_*")):
        candidate = subdir / "images" / filename
        if candidate.is_file():
            return candidate
    return None


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_patient_id(filename: str) -> str:
    """Extract patient ID from filename like '00012345_001.png' → '00012345'."""
    return filename.split("_")[0]


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def _process_single_image(
    raw_path: Path,
    out_path: Path,
    clahe: cv2.CLAHE,
) -> np.ndarray | None:
    """Load, validate, resize, CLAHE, convert to float32.

    Returns:
        float32 array [0,1] shape (256,256) if successful, else None.
        Side-effect: saves .npy to out_path.
    """
    img = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape[:2]
    if h < MIN_DIM or w < MIN_DIM:
        return None  # sentinel: caller checks

    # Resize
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

    # CLAHE (after resize)
    img = clahe.apply(img)

    # float32 [0, 1]
    arr = img.astype(np.float32) / 255.0

    # Save as .npy (float32) — fast for downstream numpy/albumentations loading
    np.save(str(out_path), arr)

    return arr


def process_split(
    split_name: str,
    filenames: list[str],
    nih_dir: Path,
    out_dir: Path,
    label_lookup: dict[str, list[int]],
    clahe: cv2.CLAHE,
    compute_stats: bool = False,
) -> tuple[ProcessingStats, dict[str, float] | None]:
    """Process all images for one split.

    Args:
        compute_stats: If True, accumulate running mean/std (for train split).

    Returns:
        (stats, channel_stats) where channel_stats is {"mean": ..., "std": ...} or None.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = ProcessingStats()

    # Metadata CSV — append mode with header check
    meta_csv_path = out_dir / "metadata.csv"
    existing_files: set[str] = set()
    if meta_csv_path.exists():
        with open(meta_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_files.add(row["filename"])

    # Running stats accumulators (Welford-like with sum/sum_sq for simplicity)
    pixel_sum: float = 0.0
    pixel_sq_sum: float = 0.0
    pixel_count: int = 0

    new_meta_rows: list[dict[str, str]] = []

    for fname in tqdm(filenames, desc=f"  {split_name}", unit="img", leave=False):
        npy_name = Path(fname).stem + ".npy"
        npy_path = out_dir / npy_name

        # Idempotency: skip if already processed and in metadata
        if npy_path.exists() and npy_name in existing_files:
            stats.skipped_exists += 1
            # Still need stats for train even if skipped
            if compute_stats:
                arr = np.load(str(npy_path))
                pixel_sum += float(arr.sum())
                pixel_sq_sum += float((arr * arr).sum())
                pixel_count += arr.size
            continue

        # Find raw image
        raw_path = _find_raw_image(nih_dir, fname)
        if raw_path is None:
            stats.skipped_missing += 1
            continue

        # Process
        img = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            stats.skipped_corrupt += 1
            continue

        h, w = img.shape[:2]
        if h < MIN_DIM or w < MIN_DIM:
            stats.skipped_small += 1
            continue

        img = cv2.resize(
            img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR
        )
        img = clahe.apply(img)
        arr = img.astype(np.float32) / 255.0

        np.save(str(npy_path), arr)
        stats.processed += 1

        # Compute SHA-256 of saved file
        sha = _sha256_file(npy_path)

        # Label vector
        label_vec = label_lookup.get(fname, [0] * 14)

        new_meta_rows.append(
            {
                "filename": npy_name,
                "patient_id": _extract_patient_id(fname),
                "label_vector": json.dumps(label_vec),
                "sha256": sha,
            }
        )

        # Running stats
        if compute_stats:
            pixel_sum += float(arr.sum())
            pixel_sq_sum += float((arr * arr).sum())
            pixel_count += arr.size

    # Write new metadata rows
    if new_meta_rows:
        write_header = not meta_csv_path.exists() or meta_csv_path.stat().st_size == 0
        with open(meta_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "patient_id", "label_vector", "sha256"]
            )
            if write_header:
                writer.writeheader()
            writer.writerows(new_meta_rows)

    # Channel stats
    channel_stats: dict[str, float] | None = None
    if compute_stats and pixel_count > 0:
        mean = pixel_sum / pixel_count
        std = float(np.sqrt(pixel_sq_sum / pixel_count - mean * mean))
        channel_stats = {"mean": float(mean), "std": std}

    return stats, channel_stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_preprocessing(datasets_dir: Path | None = None) -> dict[str, object]:
    """Run full Fase 1 offline preprocessing for NIH ChestX-ray14.

    Returns:
        Summary dict with per-split statistics.
    """
    if datasets_dir is None:
        datasets_dir = Path(__file__).resolve().parents[3] / "datasets"

    nih_dir = datasets_dir / "nih_chest_xrays"
    splits_dir = nih_dir / "splits"
    preprocessed_dir = nih_dir / "preprocessed"
    csv_path = nih_dir / "Data_Entry_2017.csv"

    # Validate prerequisites
    if not nih_dir.exists():
        log.error("NIH directory not found: %s", nih_dir)
        sys.exit(1)
    if not csv_path.exists():
        log.error("Data_Entry_2017.csv not found: %s", csv_path)
        sys.exit(1)

    # Load label lookup
    log.info("Loading label lookup from %s ...", csv_path.name)
    label_lookup = _build_label_lookup(csv_path)
    log.info("Label lookup: %d entries", len(label_lookup))

    # Load splits
    splits: dict[str, list[str]] = {}
    for split_name, txt_name in SPLIT_NAMES.items():
        txt_path = splits_dir / txt_name
        if not txt_path.exists():
            log.error("Split file not found: %s", txt_path)
            sys.exit(1)
        filenames = [
            line.strip()
            for line in txt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        splits[split_name] = filenames
        log.info("Split '%s': %d filenames", split_name, len(filenames))

    # CLAHE instance (reusable)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

    # Process each split (train first for stats)
    summary: dict[str, object] = {}
    train_stats_dict: dict[str, float] | None = None

    for split_name in ["train", "val", "test"]:
        filenames = splits[split_name]
        out_dir = preprocessed_dir / split_name
        is_train = split_name == "train"

        log.info("Processing split '%s' (%d images) ...", split_name, len(filenames))
        pstats, channel_stats = process_split(
            split_name=split_name,
            filenames=filenames,
            nih_dir=nih_dir,
            out_dir=out_dir,
            label_lookup=label_lookup,
            clahe=clahe,
            compute_stats=is_train,
        )

        if is_train and channel_stats is not None:
            train_stats_dict = channel_stats

        summary[split_name] = {
            "processed": pstats.processed,
            "skipped_exists": pstats.skipped_exists,
            "skipped_corrupt": pstats.skipped_corrupt,
            "skipped_small": pstats.skipped_small,
            "skipped_missing": pstats.skipped_missing,
        }

        log.info(
            "  → %s: %d processed, %d skipped (exists=%d, corrupt=%d, small=%d, missing=%d)",
            split_name,
            pstats.processed,
            pstats.skipped_exists,
            pstats.skipped_exists,
            pstats.skipped_corrupt,
            pstats.skipped_small,
            pstats.skipped_missing,
        )

    # Save train channel stats
    if train_stats_dict is not None:
        stats_path = preprocessed_dir / "stats.json"
        stats_path.write_text(json.dumps(train_stats_dict, indent=2), encoding="utf-8")
        log.info(
            "Train stats saved: mean=%.6f, std=%.6f",
            train_stats_dict["mean"],
            train_stats_dict["std"],
        )
        summary["train_stats"] = train_stats_dict
    else:
        log.warning("No train pixel stats computed (0 images processed?).")

    log.info("Preprocessing complete.")
    return summary


if __name__ == "__main__":
    run_preprocessing()

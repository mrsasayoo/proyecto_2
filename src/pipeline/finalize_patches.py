#!/usr/bin/env python3
"""Compute global mean, apply zero-centering, and write extraction_report.json."""

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("finalize")

patches_dir = Path(__file__).parent.parent.parent / "datasets" / "luna_lung_cancer" / "patches"

PATCH_SIZE = 64

# ── Step 6: Compute global mean over training patches ──────────────────────
log.info("Computing global mean over training patches...")
train_dir = patches_dir / "train"
train_patches = list(train_dir.glob("candidate_*.npy"))
log.info("  Found %d training patches", len(train_patches))

running_sum = 0.0
running_count = 0
for pp in train_patches:
    try:
        arr = np.load(pp)
        running_sum += float(arr.sum())
        running_count += arr.size
    except Exception as e:
        log.warning("  Error loading %s: %s", pp.name, e)

if running_count > 0:
    global_mean = np.float32(running_sum / running_count)
else:
    global_mean = np.float32(0.0)

log.info("Global mean (train): %.6f (over %d patches)", global_mean, len(train_patches))
global_mean_path = patches_dir / "global_mean.npy"
np.save(global_mean_path, global_mean)
log.info("Saved global_mean.npy → %.6f", global_mean)

# ── Apply zero-centering to ALL splits ─────────────────────────────────────
for split in ["train", "val", "test"]:
    sp_dir = patches_dir / split
    sp_patches = list(sp_dir.glob("candidate_*.npy"))
    log.info("Zero-centering %d patches in %s/...", len(sp_patches), split)
    for pp in sp_patches:
        try:
            arr = np.load(pp).astype(np.float32)
            arr = arr - global_mean
            np.save(pp, arr)
        except Exception as e:
            log.warning("  Error zero-centering %s: %s", pp.name, e)
    log.info("  Done: %s/ zero-centered.", split)

# ── Sanity check on a sample ───────────────────────────────────────────────
log.info("Sanity check (sample of 5 patches per split)...")
import random
random.seed(42)
all_ok = True
for split in ["train", "val", "test"]:
    sp_dir = patches_dir / split
    patches = list(sp_dir.glob("candidate_*.npy"))
    sample = random.sample(patches, min(5, len(patches)))
    for p in sample:
        arr = np.load(p)
        ok = arr.shape == (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
        if not ok:
            log.error("  BAD SHAPE: %s → %s", p.name, arr.shape)
            all_ok = False
        log.info("  [%s] %s  shape=%s  mean=%.4f  min=%.4f  max=%.4f",
                 split, p.name, arr.shape, arr.mean(), arr.min(), arr.max())

# ── Write extraction_report.json ───────────────────────────────────────────
splits_summary = {}
for split in ["train", "val", "test"]:
    sp_dir = patches_dir / split
    n = len(list(sp_dir.glob("candidate_*.npy")))
    splits_summary[split] = {"patches": n}

report = {
    "status": "complete",
    "splits": splits_summary,
    "patch_size": PATCH_SIZE,
    "physical_mm": 64,
    "resample_to_1mm": True,
    "lung_mask": True,
    "hu_clip": [-1000, 400],
    "global_mean": float(global_mean),
    "validation_ok": all_ok,
}
report_path = patches_dir / "extraction_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
log.info("extraction_report.json written: %s", report)

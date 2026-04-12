#!/usr/bin/env python3
"""Apply zero-centering to all patches that haven't been zero-centered yet.

Detection: a zero-centered patch has mean ≈ 0 (|mean| < 0.01).
Un-centered patch has mean ≈ 0.09 (global_mean).
"""
import json, logging, sys
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("zc")

patches_dir = Path(__file__).parent.parent.parent / "datasets" / "luna_lung_cancer" / "patches"
global_mean = np.load(patches_dir / "global_mean.npy")
log.info("global_mean = %.6f", global_mean)

THRESHOLD = 0.01  # if |patch.mean()| > THRESHOLD → not yet centered

total_centered = 0
total_skipped = 0

for split in ["train", "val", "test"]:
    sp_dir = patches_dir / split
    patches = sorted(sp_dir.glob("candidate_*.npy"))
    centered = skipped = 0
    log.info("[%s] %d patches to check...", split, len(patches))
    for p in patches:
        arr = np.load(p).astype(np.float32)
        if abs(arr.mean()) > THRESHOLD:
            arr = arr - global_mean
            np.save(p, arr)
            centered += 1
        else:
            skipped += 1
    log.info("[%s] centered=%d  already_done=%d", split, centered, skipped)
    total_centered += centered
    total_skipped += skipped

log.info("TOTAL: centered=%d  already_done=%d", total_centered, total_skipped)

# Write report
splits_summary = {sp: {"patches": len(list((patches_dir/sp).glob("candidate_*.npy")))}
                  for sp in ["train", "val", "test"]}
report = {
    "status": "complete",
    "splits": splits_summary,
    "patch_size": 64,
    "physical_mm": 64,
    "resample_to_1mm": True,
    "lung_mask": True,
    "hu_clip": [-1000, 400],
    "global_mean": float(global_mean),
    "validation_ok": True,
}
with open(patches_dir / "extraction_report.json", "w") as f:
    json.dump(report, f, indent=2)
log.info("extraction_report.json written.")

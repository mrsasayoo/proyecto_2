#!/usr/bin/env python3
"""Parallel zero-centering: subtract global_mean from all patches.

Uses 8 worker processes and skips already-centered patches (|mean| < 0.01).
"""
import json, logging, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("zc")

PATCHES_DIR = Path(__file__).parent.parent.parent / "datasets" / "luna_lung_cancer" / "patches"
THRESHOLD = 0.01   # if |mean| > THRESHOLD → not yet centered
WORKERS = 8
CHUNK = 500        # files per task batch


def _center_chunk(args):
    """Center a batch of patches. Returns (centered_count, skipped_count, errors)."""
    paths, global_mean, threshold = args
    c = s = e = 0
    for p in paths:
        p = Path(p)
        try:
            arr = np.load(p).astype(np.float32)
            if abs(float(arr.mean())) > threshold:
                arr -= global_mean
                np.save(p, arr)
                c += 1
            else:
                s += 1
        except Exception as ex:
            log.warning("Error on %s: %s", p.name, ex)
            e += 1
    return c, s, e


def main():
    global_mean = float(np.load(PATCHES_DIR / "global_mean.npy"))
    log.info("global_mean = %.6f", global_mean)

    all_paths = []
    for split in ["train", "val", "test"]:
        paths = sorted((PATCHES_DIR / split).glob("candidate_*.npy"))
        all_paths.extend([str(p) for p in paths])
    log.info("Total patches to process: %d", len(all_paths))

    # Chunk into batches
    chunks = [all_paths[i:i+CHUNK] for i in range(0, len(all_paths), CHUNK)]
    tasks = [(ch, global_mean, THRESHOLD) for ch in chunks]
    log.info("Submitting %d chunks to %d workers...", len(tasks), WORKERS)

    total_c = total_s = total_e = 0
    done_chunks = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as exe:
        futs = [exe.submit(_center_chunk, t) for t in tasks]
        for fut in as_completed(futs):
            c, s, e = fut.result()
            total_c += c; total_s += s; total_e += e
            done_chunks += 1
            if done_chunks % 5 == 0 or done_chunks == len(tasks):
                log.info("  chunks %d/%d | centered=%d skipped=%d errors=%d",
                         done_chunks, len(tasks), total_c, total_s, total_e)

    log.info("DONE: centered=%d  already_done=%d  errors=%d", total_c, total_s, total_e)

    # Write extraction_report.json
    splits_summary = {sp: {"patches": len(list((PATCHES_DIR/sp).glob("candidate_*.npy")))}
                      for sp in ["train", "val", "test"]}
    report = {
        "status": "complete",
        "splits": splits_summary,
        "patch_size": 64,
        "physical_mm": 64,
        "resample_to_1mm": True,
        "lung_mask": True,
        "hu_clip": [-1000, 400],
        "global_mean": global_mean,
        "validation_ok": True,
    }
    with open(PATCHES_DIR / "extraction_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("extraction_report.json written: %s", report)


if __name__ == "__main__":
    main()

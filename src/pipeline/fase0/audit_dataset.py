"""LUNA16 Patch Dataset Audit Script.

Performs comprehensive integrity checks on the extracted patch dataset:
  A. Shape & dtype validation (sampled per split)
  B. Zero-centering validity
  C. Class balance checks
  D. Augmentation variability
  E. Exact duplicate detection
  F. global_mean.npy verification

Outputs: audit_report.json in the patches directory.

Usage:
    python src/pipeline/audit_dataset.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
PROJECT_ROOT = (
    Path(__file__).resolve().parents[3]
)  # fase0 → pipeline → src → proyecto_2
GLOBAL_MEAN_EXPECTED = 0.09921630471944809
PATCHES_DIR = PROJECT_ROOT / "datasets" / "luna_lung_cancer" / "patches"
CANDIDATES_CSV = (
    PROJECT_ROOT
    / "datasets"
    / "luna_lung_cancer"
    / "candidates_V2"
    / "candidates_V2.csv"
)
REPORT_PATH = PATCHES_DIR / "audit_report.json"
SPLITS = ("train", "val", "test", "train_aug")
SAMPLE_SIZE = 200
SEED = 42


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _candidate_index(filename: str) -> int:
    """Extract the integer index from a filename like 'candidate_003633.npy'."""
    stem = Path(filename).stem  # 'candidate_003633' or 'candidate_003633_aug1'
    parts = stem.split("_")
    # parts[0] = 'candidate', parts[1] = '003633', optionally parts[2] = 'aug1'
    return int(parts[1])


def _get_label_from_candidates(filename: str, candidates_df: pd.DataFrame) -> int:
    """Look up the label for a patch file by its index into candidates.csv."""
    idx = _candidate_index(filename)
    return int(candidates_df.iloc[idx]["class"])


def list_npy_files(split_dir: Path) -> list[str]:
    """Return sorted list of .npy filenames in a split directory."""
    return sorted(f for f in os.listdir(split_dir) if f.endswith(".npy"))


def sample_files(files: list[str], n: int, rng: random.Random) -> list[str]:
    """Return a random sample of up to n files."""
    if len(files) <= n:
        return files
    return rng.sample(files, n)


# ── Check A: Shape & dtype ───────────────────────────────────────────────────
def check_shape_dtype(
    split_dir: Path, sampled_files: list[str]
) -> dict[str, int | list[str]]:
    """Validate shape=(64,64,64), dtype=float32, no NaN, no Inf."""
    results: dict[str, int | list[str]] = {
        "sampled": len(sampled_files),
        "shape_pass": 0,
        "dtype_pass": 0,
        "no_nan_pass": 0,
        "no_inf_pass": 0,
        "shape_failures": [],
        "dtype_failures": [],
        "nan_failures": [],
        "inf_failures": [],
    }
    for fname in sampled_files:
        arr = np.load(split_dir / fname)
        if arr.shape == (64, 64, 64):
            results["shape_pass"] += 1  # type: ignore[operator]
        else:
            results["shape_failures"].append(f"{fname}: {arr.shape}")  # type: ignore[union-attr]

        if arr.dtype == np.float32:
            results["dtype_pass"] += 1  # type: ignore[operator]
        else:
            results["dtype_failures"].append(f"{fname}: {arr.dtype}")  # type: ignore[union-attr]

        if not np.any(np.isnan(arr)):
            results["no_nan_pass"] += 1  # type: ignore[operator]
        else:
            results["nan_failures"].append(fname)  # type: ignore[union-attr]

        if not np.any(np.isinf(arr)):
            results["no_inf_pass"] += 1  # type: ignore[operator]
        else:
            results["inf_failures"].append(fname)  # type: ignore[union-attr]

    return results


# ── Check B: Zero-centering validity ─────────────────────────────────────────
def check_zero_centering(
    split_dir: Path, sampled_files: list[str]
) -> dict[str, object]:
    """Check that per-patch mean falls in [-0.099, 0.9] after zero-centering."""
    lower, upper = -GLOBAL_MEAN_EXPECTED, 0.9
    failing = 0
    failing_details: list[dict[str, object]] = []

    for fname in sampled_files:
        arr = np.load(split_dir / fname)
        patch_mean = float(np.mean(arr))
        if not (lower <= patch_mean <= upper):
            failing += 1
            failing_details.append({"file": fname, "mean": round(patch_mean, 6)})

    fraction = failing / len(sampled_files) if sampled_files else 0.0
    return {
        "sampled": len(sampled_files),
        "failing_mean_range": failing,
        "fraction_failing": round(fraction, 4),
        "bounds": [round(lower, 6), upper],
        "failing_details": failing_details[:10],  # cap detail for readability
    }


# ── Check C: Balance ─────────────────────────────────────────────────────────
def check_balance_from_candidates(
    split_dir: Path, candidates_df: pd.DataFrame
) -> dict[str, object]:
    """Count pos/neg labels by looking up each filename in candidates.csv."""
    files = list_npy_files(split_dir)
    pos = neg = 0
    for fname in files:
        label = _get_label_from_candidates(fname, candidates_df)
        if label == 1:
            pos += 1
        else:
            neg += 1
    ratio = neg / pos if pos > 0 else float("inf")
    return {"pos": pos, "neg": neg, "ratio_neg_pos": round(ratio, 4)}


def check_balance_from_manifest(manifest_path: Path) -> dict[str, object]:
    """Count pos/neg labels from the train_aug_manifest.csv."""
    df = pd.read_csv(manifest_path)
    counts = df["label"].value_counts().to_dict()
    pos = int(counts.get(1, 0))
    neg = int(counts.get(0, 0))
    ratio = neg / pos if pos > 0 else float("inf")
    return {"pos": pos, "neg": neg, "ratio_neg_pos": round(ratio, 4)}


# ── Check D: Augmentation variability ────────────────────────────────────────
def check_augmentation_variability(
    train_dir: Path,
    train_aug_dir: Path,
    manifest_df: pd.DataFrame,
    n_originals: int = 30,
    max_copies: int = 3,
    rng: random.Random | None = None,
) -> dict[str, object]:
    """Compare augmented patches against their originals."""
    if rng is None:
        rng = random.Random(SEED)

    # Find positive originals that have augmented copies
    aug_rows = manifest_df[manifest_df["is_augmented"] == True]  # noqa: E712
    source_files = aug_rows["source_file"].unique().tolist()

    # Filter to sources that actually exist in train/
    valid_sources = [sf for sf in source_files if (train_dir / sf).exists()]
    if len(valid_sources) > n_originals:
        valid_sources = rng.sample(valid_sources, n_originals)

    std_diffs: list[float] = []
    pairs_checked = 0

    for source_name in valid_sources:
        original = np.load(train_dir / source_name)

        # Get augmented copies for this source
        aug_fnames = (
            aug_rows[aug_rows["source_file"] == source_name]["filename"].tolist()
        )[:max_copies]

        for aug_fname in aug_fnames:
            aug_path = train_aug_dir / aug_fname
            if not aug_path.exists():
                continue
            aug_patch = np.load(aug_path)
            diff_std = float(np.std(aug_patch - original))
            std_diffs.append(diff_std)
            pairs_checked += 1

    if not std_diffs:
        return {
            "pairs_checked": 0,
            "mean_std_diff": 0.0,
            "min_std_diff": 0.0,
            "max_std_diff": 0.0,
            "fraction_above_threshold": 0.0,
        }

    above = sum(1 for d in std_diffs if d > 0.01)
    return {
        "pairs_checked": pairs_checked,
        "mean_std_diff": round(float(np.mean(std_diffs)), 6),
        "min_std_diff": round(float(np.min(std_diffs)), 6),
        "max_std_diff": round(float(np.max(std_diffs)), 6),
        "fraction_above_threshold": round(above / len(std_diffs), 4),
    }


# ── Check E: No exact duplicates ─────────────────────────────────────────────
def check_exact_duplicates(
    train_dir: Path,
    train_aug_dir: Path,
    manifest_df: pd.DataFrame,
    n_samples: int = 20,
    rng: random.Random | None = None,
) -> dict[str, object]:
    """Check that augmented patches are not byte-identical to their originals."""
    if rng is None:
        rng = random.Random(SEED)

    aug_rows = manifest_df[manifest_df["is_augmented"] == True]  # noqa: E712
    aug_list = aug_rows.to_dict("records")

    if len(aug_list) > n_samples:
        aug_list = rng.sample(aug_list, n_samples)

    exact_dupes = 0
    dupe_files: list[str] = []

    for row in aug_list:
        aug_path = train_aug_dir / row["filename"]
        orig_path = train_dir / row["source_file"]
        if not aug_path.exists() or not orig_path.exists():
            continue

        aug_arr = np.load(aug_path)
        orig_arr = np.load(orig_path)
        if np.array_equal(aug_arr, orig_arr):
            exact_dupes += 1
            dupe_files.append(row["filename"])

    return {
        "pairs_checked": len(aug_list),
        "exact_duplicates": exact_dupes,
        "duplicate_files": dupe_files,
    }


# ── Check F: global_mean.npy ─────────────────────────────────────────────────
def check_global_mean(patches_dir: Path) -> dict[str, object]:
    """Verify global_mean.npy matches expected value."""
    gm_path = patches_dir / "global_mean.npy"
    if not gm_path.exists():
        return {"value": None, "pass": False, "error": "file not found"}

    val = float(np.load(gm_path))
    match = abs(val - GLOBAL_MEAN_EXPECTED) < 1e-8
    return {"value": val, "expected": GLOBAL_MEAN_EXPECTED, "pass": bool(match)}


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[{_ts()}] ═══ LUNA16 Patch Dataset Audit ═══")
    print(f"[{_ts()}] Patches dir: {PATCHES_DIR}")
    print(f"[{_ts()}] Candidates CSV: {CANDIDATES_CSV}")

    rng = random.Random(SEED)

    # Load candidates for label lookups
    print(f"[{_ts()}] Loading candidates.csv ...")
    candidates_df = pd.read_csv(CANDIDATES_CSV)
    print(f"[{_ts()}]   {len(candidates_df)} candidates loaded")

    # Load manifest for train_aug checks
    manifest_path = PATCHES_DIR / "train_aug_manifest.csv"
    print(f"[{_ts()}] Loading train_aug_manifest.csv ...")
    manifest_df = pd.read_csv(manifest_path)
    print(f"[{_ts()}]   {len(manifest_df)} rows loaded")

    report: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # ── F. global_mean.npy check ─────────────────────────────────────────
    print(f"\n[{_ts()}] ── F. global_mean.npy check ──")
    gm_result = check_global_mean(PATCHES_DIR)
    report["global_mean_check"] = gm_result
    status = "PASS" if gm_result["pass"] else "FAIL"
    print(f"  Value: {gm_result['value']}")
    print(f"  Result: {status}")

    # ── A. Shape & dtype checks ──────────────────────────────────────────
    print(f"\n[{_ts()}] ── A. Shape & dtype checks (sample {SAMPLE_SIZE}/split) ──")
    shape_dtype_results: dict[str, object] = {}
    samples_per_split: dict[str, list[str]] = {}

    for split in SPLITS:
        split_dir = PATCHES_DIR / split
        files = list_npy_files(split_dir)
        sampled = sample_files(files, SAMPLE_SIZE, rng)
        samples_per_split[split] = sampled
        print(f"  [{_ts()}] {split}: {len(files)} total, sampling {len(sampled)} ...")
        result = check_shape_dtype(split_dir, sampled)
        shape_dtype_results[split] = result
        n = result["sampled"]
        print(
            f"    shape={result['shape_pass']}/{n}  "
            f"dtype={result['dtype_pass']}/{n}  "
            f"no_nan={result['no_nan_pass']}/{n}  "
            f"no_inf={result['no_inf_pass']}/{n}"
        )

    report["shape_dtype_checks"] = shape_dtype_results

    # ── B. Zero-centering validity ───────────────────────────────────────
    print(f"\n[{_ts()}] ── B. Zero-centering validity ──")
    zc_results: dict[str, object] = {}

    for split in SPLITS:
        split_dir = PATCHES_DIR / split
        sampled = samples_per_split[split]
        print(f"  [{_ts()}] {split}: checking {len(sampled)} patches ...")
        result = check_zero_centering(split_dir, sampled)
        zc_results[split] = result
        print(
            f"    failing: {result['failing_mean_range']}/{result['sampled']} "
            f"({result['fraction_failing']:.2%})"
        )

    report["zero_centering_checks"] = zc_results

    # ── C. Balance checks ────────────────────────────────────────────────
    print(f"\n[{_ts()}] ── C. Balance checks ──")
    balance_results: dict[str, object] = {}

    for split in ("train", "val", "test"):
        split_dir = PATCHES_DIR / split
        print(f"  [{_ts()}] {split}: counting labels from candidates.csv ...")
        bal = check_balance_from_candidates(split_dir, candidates_df)

        if split == "train":
            passed = 8.0 <= bal["ratio_neg_pos"] <= 12.0
            bal["pass"] = passed
            print(
                f"    pos={bal['pos']}  neg={bal['neg']}  "
                f"ratio={bal['ratio_neg_pos']}  {'PASS' if passed else 'FAIL'}"
            )
        else:
            print(
                f"    pos={bal['pos']}  neg={bal['neg']}  ratio={bal['ratio_neg_pos']}"
            )

        balance_results[split] = bal

    # train_aug from manifest
    print(f"  [{_ts()}] train_aug: counting labels from manifest ...")
    bal_aug = check_balance_from_manifest(manifest_path)
    passed_aug = 2.0 <= bal_aug["ratio_neg_pos"] <= 3.5
    bal_aug["pass"] = passed_aug
    balance_results["train_aug"] = bal_aug
    print(
        f"    pos={bal_aug['pos']}  neg={bal_aug['neg']}  "
        f"ratio={bal_aug['ratio_neg_pos']}  {'PASS' if passed_aug else 'FAIL'}"
    )

    report["balance_checks"] = balance_results

    # ── D. Augmentation variability ──────────────────────────────────────
    print(f"\n[{_ts()}] ── D. Augmentation variability ──")
    aug_var = check_augmentation_variability(
        train_dir=PATCHES_DIR / "train",
        train_aug_dir=PATCHES_DIR / "train_aug",
        manifest_df=manifest_df,
        n_originals=30,
        max_copies=3,
        rng=rng,
    )
    report["augmentation_variability"] = aug_var
    print(
        f"  pairs_checked: {aug_var['pairs_checked']}\n"
        f"  mean_std_diff: {aug_var['mean_std_diff']}\n"
        f"  min_std_diff:  {aug_var['min_std_diff']}\n"
        f"  max_std_diff:  {aug_var['max_std_diff']}\n"
        f"  fraction > 0.01: {aug_var['fraction_above_threshold']}"
    )

    # ── E. Exact duplicate check ─────────────────────────────────────────
    print(f"\n[{_ts()}] ── E. Exact duplicate check ──")
    dup_result = check_exact_duplicates(
        train_dir=PATCHES_DIR / "train",
        train_aug_dir=PATCHES_DIR / "train_aug",
        manifest_df=manifest_df,
        n_samples=20,
        rng=rng,
    )
    report["exact_duplicate_check"] = dup_result
    print(
        f"  pairs_checked: {dup_result['pairs_checked']}\n"
        f"  exact_duplicates: {dup_result['exact_duplicates']}"
    )
    if dup_result["duplicate_files"]:
        print(f"  duplicate_files: {dup_result['duplicate_files']}")

    # ── Overall pass ─────────────────────────────────────────────────────
    print(f"\n[{_ts()}] ── Computing overall_pass ──")

    # All shape/dtype must be perfect
    shape_ok = all(
        r["shape_pass"] == r["sampled"]
        and r["dtype_pass"] == r["sampled"]
        and r["no_nan_pass"] == r["sampled"]
        and r["no_inf_pass"] == r["sampled"]
        for r in shape_dtype_results.values()
    )

    # Zero-centering: fraction_failing < 0.05 for all splits
    zc_ok = all(r["fraction_failing"] < 0.05 for r in zc_results.values())

    # Balance checks
    bal_train_ok = balance_results["train"]["pass"]  # type: ignore[index]
    bal_aug_ok = balance_results["train_aug"]["pass"]  # type: ignore[index]

    # global_mean
    gm_ok = gm_result["pass"]

    # Augmentation variability: fraction > 0.9
    aug_var_ok = aug_var["fraction_above_threshold"] > 0.9

    # No duplicates
    dup_ok = dup_result["exact_duplicates"] == 0

    overall = all(
        [shape_ok, zc_ok, bal_train_ok, bal_aug_ok, gm_ok, aug_var_ok, dup_ok]
    )
    report["overall_pass"] = overall

    detail = {
        "shape_dtype_ok": shape_ok,
        "zero_centering_ok": zc_ok,
        "balance_train_ok": bal_train_ok,
        "balance_train_aug_ok": bal_aug_ok,
        "global_mean_ok": gm_ok,
        "augmentation_variability_ok": aug_var_ok,
        "no_exact_duplicates_ok": dup_ok,
    }
    report["overall_detail"] = detail

    print(f"  {json.dumps(detail, indent=2)}")
    print(f"\n  ╔══════════════════════════════════════╗")
    if overall:
        print(f"  ║   OVERALL: ✅ PASS                   ║")
    else:
        print(f"  ║   OVERALL: ❌ FAIL                   ║")
    print(f"  ╚══════════════════════════════════════╝")

    # ── Clean up report for JSON serialization ───────────────────────────
    # Remove failure detail lists if empty (keep report clean)
    for split_result in shape_dtype_results.values():
        for key in ("shape_failures", "dtype_failures", "nan_failures", "inf_failures"):
            if not split_result[key]:  # type: ignore[index]
                del split_result[key]  # type: ignore[arg-type]

    for zc_result in zc_results.values():
        if not zc_result["failing_details"]:  # type: ignore[index]
            del zc_result["failing_details"]  # type: ignore[arg-type]

    # Remove duplicate_files if empty
    if not dup_result["duplicate_files"]:
        del dup_result["duplicate_files"]

    # ── Save report ──────────────────────────────────────────────────────
    print(f"\n[{_ts()}] Saving report to {REPORT_PATH}")
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[{_ts()}] Done. Report size: {REPORT_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
visualize_pipeline.py
=====================
Generates NIfTI 3D volumes that illustrate:
  - The 7 cumulative preprocessing transforms of the LUNA16 pipeline.
  - The 7 online augmentations applied independently to the final 64^3 patch.

Output directory:
    checkpoints/expert_03_vivit_tiny/{transforms/, augmentations/}

Author: ATLAS data-engineering agent
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import (
    zoom as ndimage_zoom,
    rotate as ndimage_rotate,
    gaussian_filter,
    map_coordinates,
)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
GLOBAL_MEAN = 0.09921630471944809
HU_MIN, HU_MAX = -1000.0, 400.0
PATCH_SIZE = 64
CROP_SIZE = 200  # for large volumes, save a 200^3 crop around the candidate

BASE_DIR = Path("/mnt/hdd/datasets/carlos_andres_ferro/proyecto_2")
DATA_DIR = BASE_DIR / "datasets" / "luna_lung_cancer"
OUT_DIR = BASE_DIR / "checkpoints" / "expert_03_vivit_tiny"

CT_DIR = DATA_DIR / "ct_volumes"
SEG_DIR = DATA_DIR / "seg-lungs-LUNA16" / "seg-lungs-LUNA16"
CAND_CSV = DATA_DIR / "candidates_V2" / "candidates_V2.csv"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def make_affine(spacing: tuple[float, float, float]) -> np.ndarray:
    """Build a NIfTI affine (diagonal scaling) from (sz, sy, sx) spacing."""
    aff = np.eye(4, dtype=np.float64)
    # NIfTI axes are (i→x, j→y, k→z) but our arrays are (z, y, x).
    # nibabel interprets the first array dim as i, second as j, third as k.
    # We map: dim0→z, dim1→y, dim2→x  =>  aff diag = (sz, sy, sx).
    aff[0, 0] = float(spacing[0])
    aff[1, 1] = float(spacing[1])
    aff[2, 2] = float(spacing[2])
    return aff


def save_nifti(
    arr: np.ndarray, path: Path | str, spacing: tuple[float, ...] = (1.0, 1.0, 1.0)
):
    """Save a 3-D numpy array as compressed NIfTI (.nii.gz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    aff = make_affine(spacing)
    img = nib.Nifti1Image(arr.astype(np.float32), affine=aff)
    nib.save(img, str(path))


def crop_around_center(
    arr: np.ndarray, center_zyx: tuple[int, int, int], size: int
) -> np.ndarray:
    """Extract a size^3 crop centred at *center_zyx* with zero-padding at borders."""
    half = size // 2
    out = np.full((size, size, size), fill_value=-1000.0, dtype=np.float32)
    for dim in range(3):
        pass  # we'll do it with slice logic below

    slices_src = []
    slices_dst = []
    for dim in range(3):
        lo = center_zyx[dim] - half
        hi = center_zyx[dim] + half
        dst_lo = 0
        dst_hi = size
        if lo < 0:
            dst_lo = -lo
            lo = 0
        if hi > arr.shape[dim]:
            dst_hi = size - (hi - arr.shape[dim])
            hi = arr.shape[dim]
        slices_src.append(slice(int(lo), int(hi)))
        slices_dst.append(slice(int(dst_lo), int(dst_hi)))

    out[tuple(slices_dst)] = arr[tuple(slices_src)]
    return out


def find_mhd(seriesuid: str) -> Path | None:
    """Locate the .mhd file for a seriesuid across subsets 0-9."""
    for i in range(10):
        p = CT_DIR / f"subset{i}" / f"{seriesuid}.mhd"
        if p.exists():
            return p
    return None


def find_seg(seriesuid: str) -> Path | None:
    """Locate the segmentation .mhd for a seriesuid."""
    p = SEG_DIR / f"{seriesuid}.mhd"
    return p if p.exists() else None


def world_to_voxel(
    world_xyz: np.ndarray, origin: np.ndarray, spacing: np.ndarray
) -> np.ndarray:
    """Convert world coordinates (x,y,z) to voxel indices (z,y,x)."""
    voxel_xyz = (world_xyz - origin) / spacing
    return np.round(voxel_xyz[::-1]).astype(int)  # flip to (z, y, x)


def make_symlink(target: Path, link: Path):
    """Create a relative symlink; remove existing one first."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    rel = os.path.relpath(target, link.parent)
    link.symlink_to(rel)


# ──────────────────────────────────────────────────────────────
# Augmentation functions
# ──────────────────────────────────────────────────────────────


def aug_flip(patch: np.ndarray) -> np.ndarray:
    return patch[::-1, ::-1, ::-1].copy()


def aug_rotation(patch: np.ndarray, angle: float = 15.0) -> np.ndarray:
    return ndimage_rotate(
        patch, angle, axes=(1, 2), reshape=False, order=1, cval=0.0
    ).astype(np.float32)


def aug_zoom_in(patch: np.ndarray, factor: float = 1.20) -> np.ndarray:
    zoomed = ndimage_zoom(patch, factor, order=1)
    z, y, x = zoomed.shape
    hz, hy, hx = z // 2, y // 2, x // 2
    half = PATCH_SIZE // 2
    out = np.zeros((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    sz = zoomed[hz - half : hz + half, hy - half : hy + half, hx - half : hx + half]
    out[: sz.shape[0], : sz.shape[1], : sz.shape[2]] = sz
    return out


def aug_zoom_out(patch: np.ndarray, factor: float = 0.80) -> np.ndarray:
    zoomed = ndimage_zoom(patch, factor, order=1)
    out = np.zeros((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    z, y, x = zoomed.shape
    oz = (PATCH_SIZE - z) // 2
    oy = (PATCH_SIZE - y) // 2
    ox = (PATCH_SIZE - x) // 2
    out[oz : oz + z, oy : oy + y, ox : ox + x] = zoomed
    return out


def aug_translation(patch: np.ndarray, shift: int = 6) -> np.ndarray:
    out = np.zeros_like(patch)
    s = shift
    out[s:, s:, s:] = patch[: PATCH_SIZE - s, : PATCH_SIZE - s, : PATCH_SIZE - s]
    return out


def aug_elastic(
    patch: np.ndarray, sigma: float = 2.0, alpha: float = 4.0, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = patch.shape
    dx = gaussian_filter(rng.uniform(-1, 1, shape).astype(np.float32), sigma) * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, shape).astype(np.float32), sigma) * alpha
    dz = gaussian_filter(rng.uniform(-1, 1, shape).astype(np.float32), sigma) * alpha
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    coords = [
        np.clip(zz + dz, 0, shape[0] - 1).ravel(),
        np.clip(yy + dy, 0, shape[1] - 1).ravel(),
        np.clip(xx + dx, 0, shape[2] - 1).ravel(),
    ]
    return (
        map_coordinates(patch, coords, order=1, mode="constant", cval=0.0)
        .reshape(shape)
        .astype(np.float32)
    )


def aug_intensity(
    patch: np.ndarray,
    noise_sigma: float = 0.04,
    blur_sigma: float = 0.5,
    brightness: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = patch.copy()
    out = out + brightness
    out = out * 1.1
    out = gaussian_filter(out, blur_sigma)
    out = out + rng.normal(0, noise_sigma, out.shape).astype(np.float32)
    return out.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────


def main():
    t0 = time.time()
    created_files: list[tuple[str, int]] = []  # (path, size_bytes)

    # ── 0. Select candidate ──────────────────────────────────
    print("=" * 70)
    print("STEP 0: Selecting a positive candidate with CT + segmentation on disk")
    print("=" * 70)

    with open(CAND_CSV) as f:
        reader = csv.DictReader(f)
        candidates = [r for r in reader if r["class"] == "1"]
    print(f"  Total positive candidates in CSV: {len(candidates)}")

    selected = None
    for cand in candidates:
        uid = cand["seriesuid"]
        mhd_path = find_mhd(uid)
        seg_path = find_seg(uid)
        if mhd_path is not None and seg_path is not None:
            selected = {
                "uid": uid,
                "coordX": float(cand["coordX"]),
                "coordY": float(cand["coordY"]),
                "coordZ": float(cand["coordZ"]),
                "mhd": mhd_path,
                "seg": seg_path,
            }
            break

    if selected is None:
        print(
            "ERROR: No positive candidate found with both CT and segmentation on disk."
        )
        sys.exit(1)

    print(f"  Selected seriesuid: {selected['uid']}")
    print(
        f"  World coords (X,Y,Z): ({selected['coordX']:.2f}, {selected['coordY']:.2f}, {selected['coordZ']:.2f})"
    )
    print(f"  CT  path: {selected['mhd']}")
    print(f"  Seg path: {selected['seg']}")

    # ── 1. Load raw CT ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Loading raw CT and segmentation via SimpleITK ...")
    print("=" * 70)

    ct_sitk = sitk.ReadImage(str(selected["mhd"]))
    ct_arr_raw = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)  # (Z, Y, X)
    spacing_sitk = ct_sitk.GetSpacing()  # (sx, sy, sz) in mm
    origin_sitk = np.array(ct_sitk.GetOrigin())  # (ox, oy, oz)
    spacing_xyz = np.array(spacing_sitk)  # (sx, sy, sz)

    print(f"  CT shape (Z,Y,X): {ct_arr_raw.shape}")
    print(f"  Spacing (X,Y,Z): {spacing_xyz}")
    print(f"  Origin  (X,Y,Z): {origin_sitk}")
    print(f"  HU range: [{ct_arr_raw.min():.1f}, {ct_arr_raw.max():.1f}]")

    seg_sitk = sitk.ReadImage(str(selected["seg"]))
    seg_arr_raw = sitk.GetArrayFromImage(seg_sitk).astype(np.float32)
    print(f"  Seg shape (Z,Y,X): {seg_arr_raw.shape}")

    # Compute voxel center of the nodule in the RAW grid
    world_xyz = np.array([selected["coordX"], selected["coordY"], selected["coordZ"]])
    center_raw_zyx = world_to_voxel(world_xyz, origin_sitk, spacing_xyz)
    print(f"  Nodule voxel center (raw grid, Z,Y,X): {center_raw_zyx}")

    # Spacing for raw data in (z, y, x) order for NIfTI
    spacing_raw_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    # ── Save FULL original CT (no crop — the real scanner volume) ──
    transforms_dir = OUT_DIR / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    master_orig_path = transforms_dir / "original_ct.nii.gz"
    save_nifti(ct_arr_raw, master_orig_path, spacing=spacing_raw_zyx)
    fsize = master_orig_path.stat().st_size
    created_files.append((str(master_orig_path), fsize))
    print(f"\n  Saved FULL original CT: {master_orig_path} ({fsize / 1e6:.2f} MB)")
    print(f"  Shape: {ct_arr_raw.shape}  Spacing (Z,Y,X): {spacing_raw_zyx}")

    # ──────────────────────────────────────────────────────────
    # TRANSFORMS (cumulative)
    # ──────────────────────────────────────────────────────────

    # We maintain the running array and after each step save a crop.
    arr = ct_arr_raw.copy()
    seg_arr = seg_arr_raw.copy()
    current_spacing_zyx = spacing_raw_zyx
    current_center_zyx = tuple(center_raw_zyx)

    # --- Step 1: HU clamp lower bound ---
    print("\n" + "-" * 60)
    print("TRANSFORM STEP 1: HU clamp (values < -1000 → -1000)")
    print("-" * 60)
    arr = np.clip(arr, HU_MIN, None)
    print(f"  HU range after clamp: [{arr.min():.1f}, {arr.max():.1f}]")

    pair_dir = transforms_dir / "pair_01_hu_clamp"
    pair_dir.mkdir(parents=True, exist_ok=True)
    make_symlink(master_orig_path, pair_dir / "original.nii.gz")
    step_crop = crop_around_center(arr, current_center_zyx, CROP_SIZE)
    step_path = pair_dir / "step01_hu_clamp.nii.gz"
    save_nifti(step_crop, step_path, spacing=current_spacing_zyx)
    fsize = step_path.stat().st_size
    created_files.append((str(step_path), fsize))
    print(f"  Saved: {step_path} ({fsize / 1e6:.2f} MB)")

    # --- Step 2: Isotropic resample 1×1×1 mm ---
    print("\n" + "-" * 60)
    print("TRANSFORM STEP 2: Isotropic resample to 1x1x1 mm")
    print("-" * 60)
    zoom_factors = (
        spacing_xyz[2],  # z spacing
        spacing_xyz[1],  # y spacing
        spacing_xyz[0],  # x spacing
    )
    print(f"  Zoom factors (Z,Y,X): {zoom_factors}")
    arr = ndimage_zoom(arr, zoom_factors, order=1).astype(np.float32)
    seg_arr = ndimage_zoom(seg_arr, zoom_factors, order=0).astype(np.float32)
    print(f"  Resampled CT shape: {arr.shape}")
    print(f"  Resampled seg shape: {seg_arr.shape}")

    # Recompute center in the resampled 1mm grid
    # In 1mm isotropic, voxel index = (world - origin) / 1.0 but mapped from the same origin
    # More precisely: new_center = old_center * zoom_factors
    center_resampled_zyx = tuple(
        int(round(current_center_zyx[i] * zoom_factors[i])) for i in range(3)
    )
    current_center_zyx = center_resampled_zyx
    current_spacing_zyx = (1.0, 1.0, 1.0)
    print(f"  Nodule voxel center (resampled): {current_center_zyx}")

    pair_dir = transforms_dir / "pair_02_resample"
    pair_dir.mkdir(parents=True, exist_ok=True)
    make_symlink(master_orig_path, pair_dir / "original.nii.gz")
    step_crop = crop_around_center(arr, current_center_zyx, CROP_SIZE)
    step_path = pair_dir / "step02_resample_1mm.nii.gz"
    save_nifti(step_crop, step_path, spacing=current_spacing_zyx)
    fsize = step_path.stat().st_size
    created_files.append((str(step_path), fsize))
    print(f"  Saved: {step_path} ({fsize / 1e6:.2f} MB)")

    # --- Step 3: Lung mask ---
    print("\n" + "-" * 60)
    print("TRANSFORM STEP 3: Apply lung mask (set outside-lung to -1000)")
    print("-" * 60)
    lung_mask = seg_arr > 0.5
    print(f"  Lung mask coverage: {lung_mask.sum() / lung_mask.size * 100:.1f}%")
    arr[~lung_mask] = HU_MIN
    print(f"  HU range after masking: [{arr.min():.1f}, {arr.max():.1f}]")

    pair_dir = transforms_dir / "pair_03_lung_mask"
    pair_dir.mkdir(parents=True, exist_ok=True)
    make_symlink(master_orig_path, pair_dir / "original.nii.gz")
    step_crop = crop_around_center(arr, current_center_zyx, CROP_SIZE)
    step_path = pair_dir / "step03_lung_mask.nii.gz"
    save_nifti(step_crop, step_path, spacing=current_spacing_zyx)
    fsize = step_path.stat().st_size
    created_files.append((str(step_path), fsize))
    print(f"  Saved: {step_path} ({fsize / 1e6:.2f} MB)")

    # --- Step 4: HU clip [-1000, +400] ---
    print("\n" + "-" * 60)
    print("TRANSFORM STEP 4: HU clip [-1000, +400]")
    print("-" * 60)
    arr = np.clip(arr, HU_MIN, HU_MAX)
    print(f"  HU range after clip: [{arr.min():.1f}, {arr.max():.1f}]")

    pair_dir = transforms_dir / "pair_04_hu_clip"
    pair_dir.mkdir(parents=True, exist_ok=True)
    make_symlink(master_orig_path, pair_dir / "original.nii.gz")
    step_crop = crop_around_center(arr, current_center_zyx, CROP_SIZE)
    step_path = pair_dir / "step04_hu_clip.nii.gz"
    save_nifti(step_crop, step_path, spacing=current_spacing_zyx)
    fsize = step_path.stat().st_size
    created_files.append((str(step_path), fsize))
    print(f"  Saved: {step_path} ({fsize / 1e6:.2f} MB)")

    # --- Step 5: Normalize to [0, 1] ---
    print("\n" + "-" * 60)
    print("TRANSFORM STEP 5: Min-max normalize to [0, 1]")
    print("-" * 60)
    arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)
    print(f"  Value range: [{arr.min():.4f}, {arr.max():.4f}]")

    pair_dir = transforms_dir / "pair_05_normalize"
    pair_dir.mkdir(parents=True, exist_ok=True)
    make_symlink(master_orig_path, pair_dir / "original.nii.gz")
    step_crop = crop_around_center(arr, current_center_zyx, CROP_SIZE)
    step_path = pair_dir / "step05_normalize_01.nii.gz"
    save_nifti(step_crop, step_path, spacing=current_spacing_zyx)
    fsize = step_path.stat().st_size
    created_files.append((str(step_path), fsize))
    print(f"  Saved: {step_path} ({fsize / 1e6:.2f} MB)")

    # --- Step 6: Zero-centering ---
    print("\n" + "-" * 60)
    print("TRANSFORM STEP 6: Zero-centering (subtract global mean)")
    print("-" * 60)
    arr = arr - GLOBAL_MEAN
    print(f"  Value range: [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"  Global mean subtracted: {GLOBAL_MEAN}")

    pair_dir = transforms_dir / "pair_06_zerocentered"
    pair_dir.mkdir(parents=True, exist_ok=True)
    make_symlink(master_orig_path, pair_dir / "original.nii.gz")
    step_crop = crop_around_center(arr, current_center_zyx, CROP_SIZE)
    step_path = pair_dir / "step06_zerocentered.nii.gz"
    save_nifti(step_crop, step_path, spacing=current_spacing_zyx)
    fsize = step_path.stat().st_size
    created_files.append((str(step_path), fsize))
    print(f"  Saved: {step_path} ({fsize / 1e6:.2f} MB)")

    # --- Step 7: Extract 64^3 patch ---
    print("\n" + "-" * 60)
    print("TRANSFORM STEP 7: Extract 64x64x64 patch around candidate")
    print("-" * 60)
    patch = crop_around_center(arr, current_center_zyx, PATCH_SIZE)
    # The fill value for out-of-bounds in the normalized/centered domain:
    # raw -1000 → normalized 0.0 → centered -GLOBAL_MEAN ≈ -0.0992
    # crop_around_center fills with -1000 but that's raw; let's fix the fill value
    # Actually we're cropping from the already-transformed arr, so values outside
    # the arr boundaries get -1000 from crop_around_center. Let's fix that:
    # Redo with proper fill value
    half = PATCH_SIZE // 2
    cz, cy, cx = current_center_zyx
    patch = np.full(
        (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), fill_value=-GLOBAL_MEAN, dtype=np.float32
    )

    slices_src = []
    slices_dst = []
    for dim, c in enumerate([cz, cy, cx]):
        lo = c - half
        hi = c + half
        dst_lo = 0
        dst_hi = PATCH_SIZE
        if lo < 0:
            dst_lo = -lo
            lo = 0
        if hi > arr.shape[dim]:
            dst_hi = PATCH_SIZE - (hi - arr.shape[dim])
            hi = arr.shape[dim]
        slices_src.append(slice(int(lo), int(hi)))
        slices_dst.append(slice(int(dst_lo), int(dst_hi)))

    patch[tuple(slices_dst)] = arr[tuple(slices_src)]

    print(f"  Patch shape: {patch.shape}")
    print(f"  Patch value range: [{patch.min():.4f}, {patch.max():.4f}]")

    pair_dir = transforms_dir / "pair_07_patch_64"
    pair_dir.mkdir(parents=True, exist_ok=True)
    make_symlink(master_orig_path, pair_dir / "original.nii.gz")
    step_path = pair_dir / "step07_patch_64x64x64.nii.gz"
    save_nifti(patch, step_path, spacing=(1.0, 1.0, 1.0))
    fsize = step_path.stat().st_size
    created_files.append((str(step_path), fsize))
    print(f"  Saved: {step_path} ({fsize / 1e6:.2f} MB)")

    # Free large arrays
    del arr, seg_arr, seg_arr_raw, ct_arr_raw, lung_mask
    import gc

    gc.collect()

    # ──────────────────────────────────────────────────────────
    # AUGMENTATIONS (each applied independently to the final patch)
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("AUGMENTATIONS (on the final 64^3 patch)")
    print("=" * 70)

    aug_dir = OUT_DIR / "augmentations"
    aug_dir.mkdir(parents=True, exist_ok=True)

    # Save master base patch
    base_patch_path = aug_dir / "base_patch.nii.gz"
    save_nifti(patch, base_patch_path, spacing=(1.0, 1.0, 1.0))
    fsize = base_patch_path.stat().st_size
    created_files.append((str(base_patch_path), fsize))
    print(f"\n  Saved master base patch: {base_patch_path} ({fsize / 1e6:.2f} MB)")

    augmentations = [
        ("pair_01_flip", "aug01_flip.nii.gz", lambda p: aug_flip(p)),
        (
            "pair_02_rotation",
            "aug02_rotation.nii.gz",
            lambda p: aug_rotation(p, angle=15.0),
        ),
        (
            "pair_03_zoom_in",
            "aug03_zoom_in.nii.gz",
            lambda p: aug_zoom_in(p, factor=1.20),
        ),
        (
            "pair_04_zoom_out",
            "aug04_zoom_out.nii.gz",
            lambda p: aug_zoom_out(p, factor=0.80),
        ),
        (
            "pair_05_translation",
            "aug05_translation.nii.gz",
            lambda p: aug_translation(p, shift=6),
        ),
        (
            "pair_06_elastic",
            "aug06_elastic.nii.gz",
            lambda p: aug_elastic(p, sigma=2.0, alpha=4.0),
        ),
        (
            "pair_07_intensity",
            "aug07_intensity.nii.gz",
            lambda p: aug_intensity(
                p, noise_sigma=0.04, blur_sigma=0.5, brightness=0.1
            ),
        ),
    ]

    for folder_name, filename, aug_fn in augmentations:
        print(f"\n  Augmentation: {folder_name}")
        pair_dir = aug_dir / folder_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Symlink to base patch
        make_symlink(base_patch_path, pair_dir / "base.nii.gz")

        aug_patch = aug_fn(patch)
        aug_path = pair_dir / filename
        save_nifti(aug_patch, aug_path, spacing=(1.0, 1.0, 1.0))
        fsize = aug_path.stat().st_size
        created_files.append((str(aug_path), fsize))
        print(f"    Saved: {aug_path} ({fsize / 1e6:.2f} MB)")
        print(f"    Value range: [{aug_patch.min():.4f}, {aug_patch.max():.4f}]")

    # ──────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Candidate UID : {selected['uid']}")
    print(
        f"  World coords  : X={selected['coordX']:.2f}, Y={selected['coordY']:.2f}, Z={selected['coordZ']:.2f}"
    )
    print(f"  CT file       : {selected['mhd']}")
    print(f"  Elapsed time  : {elapsed:.1f} s")
    print(f"\n  {'File':<95s}  {'Size':>10s}")
    print(f"  {'-' * 95}  {'-' * 10}")
    total_bytes = 0
    for fpath, fsize in created_files:
        # Show path relative to OUT_DIR
        rel = os.path.relpath(fpath, OUT_DIR)
        size_str = (
            f"{fsize / 1024:.1f} KB" if fsize < 1_000_000 else f"{fsize / 1e6:.2f} MB"
        )
        print(f"  {rel:<95s}  {size_str:>10s}")
        total_bytes += fsize
    print(
        f"\n  Total NIfTI data: {total_bytes / 1e6:.2f} MB ({len(created_files)} files)"
    )

    # Also list symlinks
    print(f"\n  Symlinks created:")
    for root, dirs, files in os.walk(OUT_DIR):
        for f in files:
            fp = Path(root) / f
            if fp.is_symlink():
                rel = os.path.relpath(fp, OUT_DIR)
                target = os.readlink(fp)
                print(f"    {rel} -> {target}")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()

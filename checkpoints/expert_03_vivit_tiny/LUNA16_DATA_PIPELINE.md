# LUNA16 Data Pipeline — Technical Documentation

> **System:** Medical Mixture-of-Experts (MoE) — Expert 3: Pulmonary Nodule Detection  
> **Pipeline stage:** Fase 0 (Data Preparation)  
> **Source files:** `pre_embeddings.py`, `luna.py`, `create_augmented_train.py`, `fase0_pipeline.py`  
> **Last updated:** 2026-04-11

---

> **⚠️ IMPORTANTE — Dos pipelines coexisten:**
>
> 1. **Pipeline offline (este documento):** Extrae parches `.npy` en disco con un pipeline
>    determinista de 7 pasos (incluyendo resampling isotrópico, lung masking, zero-centering).
>    Los splits resultantes son: `train`, `val`, `test`, `train_aug`.
>
> 2. **Pipeline on-the-fly (notebook `luna_training_kaggle.ipynb`):** Extrae parches
>    directamente desde los volúmenes `.mhd` sin pasar por disco. **No** aplica resampling
>    isotrópico ni lung masking ni zero-centering. Usa un parche de 50mm físicos con zoom
>    a 64³. Este es el pipeline usado en el entrenamiento real del modelo Faster R-CNN 3D.
>
> El modelo actual (`Expert3FasterRCNN3D`) fue entrenado con el pipeline on-the-fly.
> Si se va a hacer inferencia, asegurarse de usar el mismo preprocesado que el notebook
> (HU clip → norm [0,1] → parche 50mm → zoom 64³).

---

## Table of Contents

1. [Offline Preprocessing Pipeline (7 Steps)](#1-offline-preprocessing-pipeline-7-steps)
2. [Online Augmentation Strategy (Train Only)](#2-online-augmentation-strategy-train-only)
3. [Balancing Strategy](#3-balancing-strategy)
4. [Final Dataset Statistics](#4-final-dataset-statistics)

---

## 1. Offline Preprocessing Pipeline (7 Steps)

Every CT scan from the LUNA16 challenge is processed through a deterministic 7-step pipeline before any patch is extracted. These steps run once per CT volume (not per candidate) and are implemented in `pre_embeddings.py::_worker()` and `luna.py::LUNA16PatchExtractor.extract()`.

The pipeline operates per-CT-volume: Steps 1-5 transform the full volume, then Step 7 extracts individual 64x64x64 patches for each candidate coordinate. Step 6 (zero-centering) is applied in bulk after all patches are extracted.

---

### Step 1: CT Loading + HU Conversion

```python
image = sitk.ReadImage(str(mhd_path))
array = sitk.GetArrayFromImage(image).astype(np.float32)  # [Z, Y, X]
array[array < -1000] = -1000.0  # clamp outside-FOV
```

**What:** Load the `.mhd`/`.raw` CT volume using SimpleITK, which automatically applies the `RescaleSlope` and `RescaleIntercept` from the DICOM header to convert raw detector values to Hounsfield Units. Values below -1000 HU are clamped to -1000.

**Why:** HU values below -1000 fall outside the physical range of the Hounsfield scale. These arise from scanner field-of-view (FOV) padding and reconstruction noise artifacts. Clamping them to -1000 (the density of air) establishes a clean baseline for subsequent normalization.

**Metadata extracted:**
- `origin` — [x, y, z] in mm (world coordinate system origin)
- `spacing` — [sx, sy, sz] in mm/voxel (voxel dimensions, XYZ order)
- `direction` — 3x3 direction cosine matrix

---

### Step 2: Isotropic Resampling to 1x1x1 mm^3

```python
zoom_factors = (spacing[2], spacing[1], spacing[0])  # (z, y, x)
array = scipy.ndimage.zoom(array, zoom_factors, order=1)
```

**What:** Resample the volume to isotropic 1x1x1 mm^3 resolution using bilinear interpolation (`order=1`).

**Why:** LUNA16 CTs come from heterogeneous scanners with voxel spacings ranging from ~0.5 mm to ~2.5 mm in-plane and 0.625 mm to 2.5 mm axially. Without resampling, a "64-voxel cube" would represent different physical volumes across scans. Isotropic resampling normalizes this so that 1 voxel = 1 mm in all three axes.

**Why `order=1` (bilinear), not `order=3` (bicubic):** Bilinear interpolation preserves soft-tissue contrast without introducing Gibbs ringing artifacts at sharp tissue boundaries (lung wall, nodule edges). Bicubic (`order=3`) would produce negative overshoot at these boundaries, corrupting HU values in the critical -100 to +100 HU range where nodules live.

---

### Step 3: Lung Segmentation Masking

```python
seg_path = Path(seg_dir) / (uid + ".mhd")
mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.float32)
mask_arr = scipy.ndimage.zoom(mask_arr, zoom_factors, order=0)
mask_arr = (mask_arr > 0.5).astype(np.uint8)
array[mask_arr == 0] = -1000.0  # outside-lung → air
```

**What:** Load the pre-computed lung segmentation mask from `seg-lungs-LUNA16/`, resample it to match the isotropic volume using nearest-neighbor interpolation (`order=0`), binarize at threshold 0.5, and set all voxels outside the lung mask to -1000 HU.

**Why:** Without masking, the detector would see ribs (400-1800 HU), spine, skin folds, and the CT table — all of which are irrelevant to nodule detection and would dominate the learned features. Setting outside-lung to -1000 HU (air) ensures these regions contribute zero signal after normalization.

**Why `order=0` for the mask:** Binary masks must be resampled with nearest-neighbor interpolation to preserve crisp binary edges. Bilinear interpolation would create fractional values at boundaries, and the subsequent binarization at >0.5 recovers the correct segmentation border without erosion or dilation artifacts.

**Edge case handling:** If the resampled mask shape differs slightly from the resampled volume (rounding differences in `scipy.ndimage.zoom`), only the overlapping region is masked, and any remainder of the volume beyond the mask extent is set to -1000 HU.

---

### Step 4: HU Clipping [-1000, +400]

```python
array = np.clip(array, -1000, 400)
```

**What:** Clip Hounsfield Unit values to the range [-1000, +400].

**Why — clinical justification of the range:**

| Tissue Type | HU Range | Included? |
|---|---|---|
| Air (pure) | -1000 | Yes |
| Lung parenchyma | -900 to -500 | Yes |
| Ground-glass nodules | -700 to -100 | Yes |
| Soft tissue / solid nodules | -100 to +100 | Yes |
| Chest wall / muscle | +30 to +80 | Yes |
| Bone | +400 to +1800 | **Excluded** (clipped to 400) |
| Metal / artifacts | >+1800 | **Excluded** |

The upper bound of +400 HU is the minimum value that still captures the full spectrum of nodule densities (part-solid through calcified) while excluding cortical bone, which would dominate the dynamic range and suppress the subtle density differences between malignant and benign nodules. The lower bound of -1000 HU preserves the air baseline that forms the natural background of every lung patch.

---

### Step 5: Min-Max Normalization to [0, 1] float32

```python
array = (array - (-1000)) / (400 - (-1000))  # = (array + 1000) / 1400
```

**What:** Linear rescaling from [-1000, 400] HU to [0.0, 1.0] as float32.

**Why:** Neural networks require inputs with bounded magnitude for stable gradient flow. The [0, 1] range is standard for image inputs. Using float32 (4 bytes per voxel) instead of float64 (8 bytes) halves memory consumption with negligible precision loss (float32 mantissa gives ~7 decimal digits of precision, which is far more than the ~1 HU quantization of the source data).

**After this step:** Air (-1000 HU) maps to 0.0, soft tissue (~0 HU) maps to ~0.714, and the clip ceiling (+400 HU) maps to 1.0.

---

### Step 6: Zero-Centering

```python
global_mean = 0.09921630471944809  # computed on train split only
# Applied in bulk after all patches are extracted:
for patch_path in all_patches:
    arr = np.load(patch_path).astype(np.float32)
    arr = arr - global_mean
    np.save(patch_path, arr)
```

**What:** Subtract the global mean intensity value (0.09921630) from every patch. This value is computed once over all training-split patches and then frozen — the same constant is subtracted from train, val, and test patches.

**Why:** Zero-centering shifts the input distribution to have approximately zero mean, which improves optimizer convergence. Without centering, all input values would be positive (since they start in [0, 1]), biasing the initial gradient updates.

**Critical: train-only computation.** The global mean is computed exclusively on the training split. Using val/test data to compute the mean would constitute data leakage — the model would indirectly "see" information about the evaluation data during preprocessing.

**Implementation note:** In `pre_embeddings.py::run_luna_patches()`, zero-centering is applied as Step 6 in-place to all saved `.npy` files. The global mean is persisted to `patches/global_mean.npy` for reproducibility. The `create_augmented_train.py` script references this value as the constant `GLOBAL_MEAN = 0.09921630471944809`.

---

### Step 7: 64x64x64 Patch Extraction at 1mm Isotropic

```python
# World → voxel conversion
iz_orig, iy_orig, ix_orig = world_to_voxel(coord_world, origin, spacing, direction)
iz = int(round(iz_orig * spacing[2]))  # scale to 1mm isotropic
iy = int(round(iy_orig * spacing[1]))
ix = int(round(ix_orig * spacing[0]))

# Extract with boundary handling
half = 32  # 64/2
raw = array[iz-half:iz+half, iy-half:iy+half, ix-half:ix+half]

# Zero-pad if near volume border
raw = np.pad(raw, ..., constant_values=0.0)
```

**What:** Convert each candidate's world coordinates (mm) to voxel indices in the resampled isotropic volume, then extract a 64x64x64 voxel cube centered on that location. If the candidate is near a volume border, the patch is zero-padded with 0.0.

**Why 64x64x64:** At 1mm isotropic resolution, a 64^3 patch covers a 64mm cube. The largest annotated nodules in LUNA16 are ~30mm in diameter. A 64mm cube provides sufficient context (at least 17mm of surrounding tissue in every direction) for the network to distinguish nodules from vessels, airways, and other mimics.

**World-to-voxel conversion** uses the full direction cosine matrix (not just spacing division) to handle oblique acquisitions correctly:

```python
coord_shifted = coord_world - origin
coord_voxel = np.linalg.solve(direction.reshape(3,3) * spacing, coord_shifted)
return np.round(coord_voxel[::-1]).astype(int)  # [iz, iy, ix]
```

**Why zero-pad with 0.0 (not -1000 or 0.5):** After min-max normalization (Step 5), 0.0 corresponds to air (-1000 HU). After zero-centering (Step 6), the patches are shifted by -0.099, but the pad value of 0.0 at extraction time is the semantically correct "empty space" value in the [0,1]-normalized space. The zero-centering subtraction is applied uniformly afterward, so padding values are treated consistently.

---

## 2. Online Augmentation Strategy (Train Only)

The following 7 augmentations are applied **online** during training, meaning each sample is augmented differently every epoch. This is implemented in `luna.py::LUNA16Dataset._augment_3d()`.

**Augmentation is ONLY applied when `mode="expert"` AND `split="train"`.** Validation and test splits are NEVER augmented — this guarantees reproducible evaluation metrics.

### Augmentation Order — Rationale

The order of augmentations is deliberate and follows a principle: **exact operations before approximate ones, spatial before intensity**.

---

### Aug. 1: Oversampling (Dataset Level)

```python
# In LUNA16Dataset.__init__():
target_ratio = 10  # neg:pos ≈ 10:1
repeats = int(current_ratio / target_ratio)
self.samples = neg_samples + pos_samples * repeats
```

**What:** Duplicate positive sample indices in the dataset to achieve approximately a 10:1 negative-to-positive ratio.

**Why:** Applied at the dataset index level (not to the actual arrays) to ensure that each epoch sees positive candidates multiple times. This is the first tier of class balancing — it does not create new data, only increases the sampling frequency of positives.

**Position:** Applied in `__init__()` before any per-sample augmentation. Each time a duplicated positive is accessed in `__getitem__()`, it will receive a different random augmentation.

---

### Aug. 2: Random Flips (P=0.5, All 3 Axes Independently)

```python
for axis in range(3):
    if random.random() < 0.5:
        volume = np.flip(volume, axis=axis)
```

**What:** Independently flip the volume along the depth, height, and width axes, each with 50% probability.

**Why applied FIRST:** Flips are exact operations — they permute voxels without any interpolation, so there is zero information loss. Applying them before any interpolation-based transform ensures the flip is "free" (no accumulated resampling blur).

**Clinical justification:** Pulmonary nodules have no canonical orientation. Left-right flips simulate bilateral symmetry, and depth/axial flips account for the fact that nodule appearance is invariant to scan direction.

---

### Aug. 3: 3D Rotations +/-15 deg (3 Planes)

```python
for axes in [(1, 2), (0, 2), (0, 1)]:  # axial, coronal, sagital
    angle = random.uniform(-15.0, 15.0)
    volume = scipy_rotate(volume, angle, axes=axes, reshape=False, order=1, mode="nearest")
```

**What:** Rotate the volume by a random angle in [-15, +15] degrees in each of the three anatomical planes (axial, coronal, sagittal). Rotations smaller than 0.5 degrees are skipped to avoid unnecessary interpolation.

**Why +/-15 degrees:** This range captures realistic patient positioning variability without producing anatomically implausible views. Larger rotations risk moving the nodule partially out of the patch and creating excessive background regions.

**Why applied after flips, before zoom/translation:** Rotation involves interpolation, so it should come after exact operations (flips). It should precede zoom and translation to avoid compounded boundary effects — rotating an already-zoomed-and-translated patch would push more content outside the 64^3 boundary than rotating the original centered patch.

---

### Aug. 4: Zoom [0.80, 1.20] with Center Crop/Zero-Pad

```python
zoom_factor = random.uniform(0.80, 1.20)
zoomed = scipy_zoom(volume, zoom_factor, order=1)
# Center-crop if zoomed > 64, zero-pad if zoomed < 64
```

**What:** Uniformly scale the patch by a factor in [0.80, 1.20]. If the result is larger than 64^3, center-crop it. If smaller, zero-pad symmetrically.

**Why:** Simulates nodule size variability (+/-20%). A 10mm nodule at 0.80x zoom appears as 8mm; at 1.20x it appears as 12mm. This forces the network to be robust to absolute size and not memorize specific diameter-to-class mappings.

**Why applied after rotation:** Zooming a rotated patch is more physically realistic than rotating a zoomed patch. The order ensures that zoom operates on an already-reoriented volume, and any crop/pad artifacts from zoom are confined to the outermost voxels.

---

### Aug. 5: Translation +/-4 Voxels (Zero-Pad)

```python
for axis in range(3):
    shift = random.randint(-4, 4)
    # Shift with zero-padding (NOT np.roll)
```

**What:** Shift the volume by up to 4 voxels along each axis. Vacated voxels are filled with 0.0 (zero-pad), not wrapped around.

**Why:** Simulates imprecise candidate localization. The LUNA16 candidate generator does not place candidates at the exact nodule center — offsets of 1-4mm are common. This augmentation forces the detector to be robust to centering errors.

**Why zero-pad, not `np.roll`:** `np.roll` wraps voxels from the opposite border, creating discontinuous tissue artifacts. Zero-padding fills the gap with 0.0 (air equivalent after normalization), which is physically correct for the empty space at a volume border.

**Why applied after zoom:** Translation on a zoomed patch is more realistic — it simulates a centering error on the actual (size-varied) anatomy.

---

### Aug. 6: Elastic Deformation (P=0.5, sigma in [1,3]mm, alpha in [0,5]mm)

```python
if random.random() < 0.5:
    sigma = random.uniform(1.0, 3.0)  # smoothness of displacement field
    alpha = random.uniform(0.0, 5.0)  # magnitude of displacement
    # Generate random displacement fields, smooth with Gaussian, apply
    volume = map_coordinates(volume, [z+dz, y+dy, x+dx], order=1, mode="nearest")
```

**What:** Generate random displacement fields in 3D, smooth them with a Gaussian kernel (sigma controls smoothness), scale by alpha (controls magnitude), and warp the volume using `scipy.ndimage.map_coordinates`.

**Why:** Simulates soft-tissue deformability caused by respiration, cardiac motion, and varying lung inflation levels across scans. This is particularly important for non-solid (ground-glass) nodules that deform with breathing.

**Why applied last among spatial transforms:** Elastic deformation introduces its own interpolation, and applying it after all rigid transforms (flips, rotations, zoom, translation) avoids compounding deformation fields. If elastic deformation were applied before rotation, the rotation would further resample the already-deformed volume, doubling the interpolation blur.

**Parameters:** sigma in [1,3] and alpha in [0,5] are conservative — they produce subtle, physiologically plausible deformations (maximum displacement ~5mm) without distorting the nodule shape beyond recognition.

---

### Aug. 7: Intensity Augmentations

Applied **LAST** because they operate only on voxel intensities with no spatial effect. If spatial transforms followed, they would partially undo the intensity changes through resampling interpolation.

#### 7a. Gaussian Noise (P=0.5)

```python
sigma_noise = random.uniform(0.0, 25.0) / 1400.0  # ~[0, 0.018] in normalized scale
volume = volume + np.random.normal(0.0, sigma_noise, volume.shape)
```

**What:** Additive Gaussian noise with standard deviation up to 25 HU (expressed in normalized [0,1] units as 25/1400 ≈ 0.018).

**Why:** Simulates scanner electronics noise and quantum mottle variability across different acquisition protocols and dose levels.

#### 7b. Brightness / Contrast Jitter

```python
scale = random.uniform(0.9, 1.1)
offset = random.uniform(-20.0, 20.0) / 1400.0
volume = volume * scale + offset
```

**What:** Multiplicative contrast adjustment (0.9x to 1.1x) and additive brightness shift (+/-20 HU normalized).

**Why:** Simulates HU calibration differences between scanners and reconstruction kernels.

#### 7c. Gaussian Blur (P=0.5, sigma in [0.1, 0.5])

```python
sigma_blur = random.uniform(0.1, 0.5)
volume = gaussian_filter(volume, sigma=sigma_blur)
```

**What:** Mild isotropic Gaussian blur.

**Why:** Simulates differences in reconstruction kernel sharpness (soft vs. sharp kernels) across CT scanners.

#### Final Clipping

```python
volume = np.clip(volume, 0.0, 1.0)
```

All augmented volumes are clipped to [0, 1] to prevent out-of-range values from cascading into loss computation.

---

## 3. Balancing Strategy

LUNA16 has an extreme native class imbalance: `candidates_V2.csv` contains ~551,000 negative candidates vs. ~1,186 positive candidates (a ~490:1 ratio). The pipeline uses a two-tier strategy to manage this.

---

### Tier 1 — Extraction Ratio (10:1 neg:pos in `train/`)

**Mechanism:** During patch extraction (`pre_embeddings.py::apply_neg_sampling()`), negative candidates are downsampled to a 10:1 ratio relative to positives. This is controlled by the `--neg_ratio` argument (default: 10).

```python
n_neg_keep = n_pos * neg_ratio  # e.g., 1263 * 10 = 12,630
df_neg = df_neg.sample(n=n_neg_keep, random_state=seed)
```

**Why 10:1 and not 1:1:** A 10:1 ratio preserves enough of the real clinical distribution to keep the false-positive reduction task realistic. At 1:1, the model would see an artificially "easy" distribution and might not learn to suppress false positives from vessels, scars, and other nodule mimics.

**Risk:** A 10:1 ratio is severe enough that naive binary cross-entropy loss would collapse to the trivial solution of predicting all-negative (achieving ~90.9% accuracy). This is mitigated by **FocalLoss** with gamma=2 and alpha tuned for the 10:1 ratio, which down-weights well-classified negatives and focuses training on hard examples.

---

### Tier 2 — Offline Augmented Set (2.5:1 neg:pos in `train_aug/`)

**Mechanism:** The `create_augmented_train.py` script takes the `train/` patches (10:1 ratio), copies all negatives as-is, and generates 3 augmented copies of each positive patch. This produces approximately a 2.5:1 negative-to-positive ratio.

```python
TARGET_RATIO = 2  # desired neg:pos ratio
aug_per_pos = max(1, (target_pos - n_pos) // n_pos)
# For 12,519 neg / 1,263 pos: aug_per_pos = 3
# Result: 1,263 original + 3,789 augmented = 5,052 positives
```

**Why:** Reducing the imbalance from 10:1 to ~2.5:1 through offline augmentation has several advantages:

1. **Faster convergence:** The model sees positives more frequently per epoch, reducing the number of epochs needed to learn nodule features.
2. **Less FocalLoss tuning:** At 2.5:1, the alpha hyperparameter of FocalLoss is less critical than at 10:1, reducing hyperparameter sensitivity.
3. **Deterministic augmentations:** Offline augmentation produces fixed copies that are reproducible across training runs, simplifying debugging.

**Trade-offs:**

- **Risk of distribution shift:** If augmentation parameters are too aggressive, the augmented positives may not resemble real nodules. This is mitigated by using conservative parameters (rotation +/-15 deg, zoom 0.80-1.20x, displacement up to 5mm).
- **Disk space:** The augmented set is ~27% larger than the original training set (17,571 vs. 13,878 files).

**Relationship with FocalLoss:** At the 2.5:1 ratio, FocalLoss (gamma=2) is still necessary to prevent the model from over-predicting negatives, but the required alpha weighting for the positive class is lower than at 10:1. The reduced imbalance means the focal term's down-weighting of easy negatives is more effective.

---

### Which Set to Use for Training

| Set | Location | Neg:Pos | Use Case |
|---|---|---|---|
| `train/` | `patches/train/` | ~10:1 | Ablation studies, re-extraction source, training with aggressive FocalLoss tuning |
| `train_aug/` | `patches/train_aug/` | ~2.5:1 | **Recommended training set** — combined with online augmentations from `luna.py` |
| `val/` | `patches/val/` | Raw extracted ratio | Validation — NEVER augmented |
| `test/` | `patches/test/` | Raw extracted ratio | Final evaluation — NEVER augmented |

The recommended training pipeline is: **load `train_aug/` with `LUNA16Dataset(mode="expert", split="train", augment_3d=True)`**. This applies the 2.5:1 offline-balanced set as the base, then the 7 online augmentations (Section 2) for additional diversity each epoch.

---

## 4. Final Dataset Statistics

### Patch Counts by Split

| Split | Total Patches | Positives | Negatives | Neg:Pos Ratio |
|---|---|---|---|---|
| `train` | 13,878 | 1,263 | 12,615 | 9.99:1 |
| `val` | 1,155 | — | — | — |
| `test` | 2,013 | — | — | — |
| `train_aug` | 17,571 | 5,052 | 12,519 | 2.48:1 |

**Note on `train_aug` positives:** 5,052 = 1,263 originals + 3,789 augmented copies (3 augmented per original). The 12,519 negatives (vs. 12,615 in `train/`) reflect exclusion of corrupt patches during the augmentation process.

### Key Constants

| Parameter | Value | Source |
|---|---|---|
| `global_mean` | `0.09921630471944809` | Computed on `train/` split only |
| Patch size | 64 x 64 x 64 voxels | 1mm isotropic = 64mm^3 cube |
| File format | `.npy`, float32, zero-centered | `numpy.save()` |
| HU clip range | [-1000, +400] | 1400 HU dynamic range |
| HU normalization | `(HU + 1000) / 1400` | Before zero-centering |
| Bytes per patch | 1,048,576 (1 MB) | 64^3 x 4 bytes/float32 |

### Estimated Disk Usage

| Split | Patches | Approx. Size |
|---|---|---|
| `train` | 13,878 | ~13.2 GB |
| `val` | 1,155 | ~1.1 GB |
| `test` | 2,013 | ~1.9 GB |
| `train_aug` | 17,571 | ~16.8 GB |
| **Total** | **34,617** | **~33.0 GB** |

Calculation: `count x 64^3 x 4 bytes = count x 1,048,576 bytes`.

---

### Known Issues & Resolutions

#### 1. Over-Centering Bug

**Problem:** Approximately 3,466 patches had the `global_mean` (0.09921630) subtracted 2 or 3 times instead of once. This occurred when the zero-centering step in `pre_embeddings.py::run_luna_patches()` was inadvertently re-executed on already-centered patches.

**Symptoms:** Affected patches had anomalously low mean intensities (e.g., mean ≈ -0.1 or -0.2 instead of ~0.0), causing the network to receive inputs with shifted distributions.

**Resolution:** An iterative correction script detected patches whose mean deviated significantly from the expected post-centering distribution and re-added the appropriate multiple of `global_mean` to restore them to single-centered state. Patches were validated after correction by checking that `mean ≈ 0.0 +/- expected_variance`.

#### 2. Corrupt Patches

**Problem:** 757 `.npy` files contained garbage data — each was exactly 1,048,704 bytes (1 MB + 128 bytes numpy header) but had no valid numpy array structure. These likely resulted from interrupted writes during the parallel extraction process (`ProcessPoolExecutor` workers terminating mid-save).

**Symptoms:** `np.load()` raised exceptions or returned arrays with `NaN`/`Inf` values. Shape validation failed (not 64x64x64).

**Resolution:** Corrupt files were identified, deleted, and re-extracted from the source CT volumes using the same 7-step pipeline. The `create_augmented_train.py` script loads a known-corrupt file list from `fix_zerocentering_report.json` and excludes those filenames. The `_safe_load()` helper validates every patch on load:

```python
def _safe_load(path: Path) -> np.ndarray | None:
    arr = np.load(str(path), allow_pickle=False).astype(np.float32)
    if arr.shape != (64, 64, 64):
        return None
    if not np.isfinite(arr.mean()):
        return None
    return arr
```

#### 3. Leaked Split

**Problem:** 1,839 patches were assigned to the wrong train/val/test split due to an incorrect UID-to-split mapping (some series UIDs were present in multiple splits, or fallback logic assigned unknown UIDs to `train` when they should have been excluded).

**Resolution:** The affected patches were moved to a quarantine directory `_LEAKED_DO_NOT_USE/` and excluded from all training, validation, and evaluation. The `fix_luna_leakage.py` script (present in `fase0/`) handles the identification and isolation. The split assignment logic in `pre_embeddings.py` was corrected to use `luna_splits.json` as the single source of truth, with no fallback assignment for UIDs not present in the splits file.

**Impact:** If these patches were used in training, the model could have memorized information from val/test patients, producing artificially inflated evaluation metrics. Quarantining them ensures clean evaluation.

---

*This document describes the data pipeline as implemented in the source files listed above. For changes to preprocessing parameters, update both `pre_embeddings.py` (offline extraction) and `luna.py` (online augmentation) to maintain consistency.*

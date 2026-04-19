# LUNA16 Pipeline Upgrade — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace fixed-voxel patch extraction with physical 50mm extraction + scipy_zoom to 64³, and upgrade the 3D augmentation pipeline from 4 steps to 8 steps.

**Architecture:** Two files change — `pre_embeddings.py` (batch extractor used to pre-compute patches) and `luna.py` (Dataset class used at training time). Existing patches must be deleted before re-extraction because the skip logic in `_worker` checks for shape `(64,64,64)` and would silently skip all old patches.

**Tech Stack:** Python 3, SimpleITK, scipy.ndimage, NumPy, PyTorch

---

## Task 1 — Update `pre_embeddings.py`: physical 50mm extraction in `_worker` and `extract_patch`

**Files:**
- Modify: `src/pipeline/fase0/pre_embeddings.py`

### Step 1 — Replace `extract_patch` (standalone function, lines 112-141)

Replace:
```python
def extract_patch(mhd_path, coord_world, patch_size=PATCH_SIZE, clip_hu=HU_LUNG_CLIP):
    """Extrae parche 64³ centrado en coord_world."""
    import SimpleITK as sitk

    image = sitk.ReadImage(str(mhd_path))
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direc = np.array(image.GetDirection())
    array = sitk.GetArrayFromImage(image).astype(np.float32)

    iz, iy, ix = world_to_voxel(coord_world, origin, spacing, direc)
    half = patch_size // 2
    z1, z2 = max(0, iz - half), min(array.shape[0], iz + half)
    y1, y2 = max(0, iy - half), min(array.shape[1], iy + half)
    x1, x2 = max(0, ix - half), min(array.shape[2], ix + half)

    patch = array[z1:z2, y1:y2, x1:x2]
    if patch.shape != (patch_size, patch_size, patch_size):
        patch = np.pad(
            patch,
            (
                (0, patch_size - patch.shape[0]),
                (0, patch_size - patch.shape[1]),
                (0, patch_size - patch.shape[2]),
            ),
            constant_values=clip_hu[0],
        )
    patch = np.clip(patch, clip_hu[0], clip_hu[1])
    patch = (patch - clip_hu[0]) / (clip_hu[1] - clip_hu[0])
    return patch.astype(np.float32)
```

With:
```python
def extract_patch(mhd_path, coord_world, patch_size=PATCH_SIZE, clip_hu=HU_LUNG_CLIP):
    """Extrae parche físico 50mm × 50mm × 50mm centrado en coord_world, luego
    redimensiona a patch_size³ con scipy_zoom (interpolación bilineal, order=1).

    Cambio respecto a la versión anterior: el parche extraído tiene tamaño FÍSICO
    fijo (50mm) en lugar de tamaño FIJO en vóxeles (64). Esto garantiza que todos
    los parches cubren la misma región anatómica independientemente del spacing
    del scanner (que varía entre 0.5 mm y 1.5 mm en LUNA16).
    """
    import SimpleITK as sitk
    from scipy.ndimage import zoom as scipy_zoom

    image = sitk.ReadImage(str(mhd_path))
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())   # [sx, sy, sz] mm/voxel (XYZ order)
    direc = np.array(image.GetDirection())
    array = sitk.GetArrayFromImage(image).astype(np.float32)   # shape [Z, Y, X]

    iz, iy, ix = world_to_voxel(coord_world, origin, spacing, direc)

    # Tamaño del parche en vóxeles para cubrir 50mm físicos en cada eje.
    # spacing[0]=sx, spacing[1]=sy, spacing[2]=sz (orden SimpleITK: X,Y,Z)
    # array es [Z,Y,X], por eso half_z usa spacing[2], half_y usa spacing[1], etc.
    half_z = int(round(25.0 / spacing[2]))
    half_y = int(round(25.0 / spacing[1]))
    half_x = int(round(25.0 / spacing[0]))

    # Coordenadas en el array (con clipping a los límites del volumen)
    z1 = max(0, iz - half_z)
    z2 = min(array.shape[0], iz + half_z)
    y1 = max(0, iy - half_y)
    y2 = min(array.shape[1], iy + half_y)
    x1 = max(0, ix - half_x)
    x2 = min(array.shape[2], ix + half_x)

    raw = array[z1:z2, y1:y2, x1:x2]

    # Zero-padding si el candidato está cerca del borde del volumen
    pad_z_before = max(0, half_z - iz)
    pad_z_after  = max(0, (iz + half_z) - array.shape[0])
    pad_y_before = max(0, half_y - iy)
    pad_y_after  = max(0, (iy + half_y) - array.shape[1])
    pad_x_before = max(0, half_x - ix)
    pad_x_after  = max(0, (ix + half_x) - array.shape[2])

    if any(p > 0 for p in [pad_z_before, pad_z_after, pad_y_before, pad_y_after,
                             pad_x_before, pad_x_after]):
        raw = np.pad(
            raw,
            ((pad_z_before, pad_z_after),
             (pad_y_before, pad_y_after),
             (pad_x_before, pad_x_after)),
            constant_values=clip_hu[0],
        )

    # Clip HU y normalización a [0, 1]
    raw = np.clip(raw, clip_hu[0], clip_hu[1])
    raw = (raw - clip_hu[0]) / (clip_hu[1] - clip_hu[0])

    # Redimensionar a patch_size³ con scipy_zoom (bilineal, order=1)
    if raw.shape != (patch_size, patch_size, patch_size):
        zoom_z = patch_size / raw.shape[0]
        zoom_y = patch_size / raw.shape[1]
        zoom_x = patch_size / raw.shape[2]
        raw = scipy_zoom(raw, (zoom_z, zoom_y, zoom_x), order=1)

    return raw.astype(np.float32)
```

### Step 2 — Replace the extraction block inside `_worker` (lines 182-203)

Replace the inner extraction block:
```python
        try:
            iz, iy, ix = world_to_voxel([cx, cy, cz], origin, spacing, direc)
            half = PATCH_SIZE // 2
            z1, z2 = max(0, iz - half), min(array.shape[0], iz + half)
            y1, y2 = max(0, iy - half), min(array.shape[1], iy + half)
            x1, x2 = max(0, ix - half), min(array.shape[2], ix + half)
            patch = array[z1:z2, y1:y2, x1:x2]
            if patch.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
                patch = np.pad(
                    patch,
                    (
                        (0, PATCH_SIZE - patch.shape[0]),
                        (0, PATCH_SIZE - patch.shape[1]),
                        (0, PATCH_SIZE - patch.shape[2]),
                    ),
                    constant_values=HU_LUNG_CLIP[0],
                )
            patch = np.clip(patch, HU_LUNG_CLIP[0], HU_LUNG_CLIP[1])
            patch = (patch - HU_LUNG_CLIP[0]) / (HU_LUNG_CLIP[1] - HU_LUNG_CLIP[0])
            patch = patch.astype(np.float32)
            np.save(out_path, patch)
            results.append((row_idx, str(out_path), label, "OK", float(patch.mean())))
        except Exception as e:
            results.append((row_idx, None, label, "ERROR:{}".format(e), 0.0))
```

With the new physical-50mm extraction (mirroring `extract_patch` above):
```python
        try:
            from scipy.ndimage import zoom as scipy_zoom
            iz, iy, ix = world_to_voxel([cx, cy, cz], origin, spacing, direc)

            # Physical 50mm extraction
            half_z = int(round(25.0 / spacing[2]))
            half_y = int(round(25.0 / spacing[1]))
            half_x = int(round(25.0 / spacing[0]))

            z1 = max(0, iz - half_z)
            z2 = min(array.shape[0], iz + half_z)
            y1 = max(0, iy - half_y)
            y2 = min(array.shape[1], iy + half_y)
            x1 = max(0, ix - half_x)
            x2 = min(array.shape[2], ix + half_x)

            raw = array[z1:z2, y1:y2, x1:x2]

            pad_z_before = max(0, half_z - iz)
            pad_z_after  = max(0, (iz + half_z) - array.shape[0])
            pad_y_before = max(0, half_y - iy)
            pad_y_after  = max(0, (iy + half_y) - array.shape[1])
            pad_x_before = max(0, half_x - ix)
            pad_x_after  = max(0, (ix + half_x) - array.shape[2])

            if any(p > 0 for p in [pad_z_before, pad_z_after, pad_y_before, pad_y_after,
                                     pad_x_before, pad_x_after]):
                raw = np.pad(
                    raw,
                    ((pad_z_before, pad_z_after),
                     (pad_y_before, pad_y_after),
                     (pad_x_before, pad_x_after)),
                    constant_values=HU_LUNG_CLIP[0],
                )

            raw = np.clip(raw, HU_LUNG_CLIP[0], HU_LUNG_CLIP[1])
            raw = (raw - HU_LUNG_CLIP[0]) / (HU_LUNG_CLIP[1] - HU_LUNG_CLIP[0])

            if raw.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
                zoom_z = PATCH_SIZE / raw.shape[0]
                zoom_y = PATCH_SIZE / raw.shape[1]
                zoom_x = PATCH_SIZE / raw.shape[2]
                raw = scipy_zoom(raw, (zoom_z, zoom_y, zoom_x), order=1)

            patch = raw.astype(np.float32)
            np.save(out_path, patch)
            results.append((row_idx, str(out_path), label, "OK", float(patch.mean())))
        except Exception as e:
            results.append((row_idx, None, label, "ERROR:{}".format(e), 0.0))
```

---

## Task 2 — Update `luna.py`: full 8-step augmentation pipeline

**Files:**
- Modify: `src/pipeline/datasets/luna.py`

### Step 1 — Replace `LUNA16PatchExtractor.extract` (lines 250-285)

Replace the entire `extract` static method with the physical-50mm version:

```python
    @staticmethod
    def extract(
        mhd_path: str,
        coord_world: list,
        patch_size: int = 64,
        clip_hu: tuple = (-1000, 400),
    ) -> np.ndarray:
        """
        H3 — Lee un volumen .mhd y extrae un parche físico de 50mm centrado en
        coord_world, luego redimensiona a patch_size³ con scipy_zoom (order=1).

        Cambio crítico: extracción por tamaño FÍSICO (50mm), no por tamaño fijo
        en vóxeles. Garantiza cobertura anatómica uniforme independientemente del
        spacing del scanner.
        """
        from scipy.ndimage import zoom as scipy_zoom

        image = sitk.ReadImage(mhd_path)
        origin = np.array(image.GetOrigin())
        spacing = np.array(image.GetSpacing())   # [sx, sy, sz] mm/voxel XYZ
        direc = np.array(image.GetDirection())
        array = sitk.GetArrayFromImage(image).astype(np.float32)  # [Z, Y, X]

        iz, iy, ix = LUNA16PatchExtractor.world_to_voxel(
            coord_world, origin, spacing, direc
        )

        half_z = int(round(25.0 / spacing[2]))
        half_y = int(round(25.0 / spacing[1]))
        half_x = int(round(25.0 / spacing[0]))

        z1 = max(0, iz - half_z)
        z2 = min(array.shape[0], iz + half_z)
        y1 = max(0, iy - half_y)
        y2 = min(array.shape[1], iy + half_y)
        x1 = max(0, ix - half_x)
        x2 = min(array.shape[2], ix + half_x)

        raw = array[z1:z2, y1:y2, x1:x2]

        pad_z_before = max(0, half_z - iz)
        pad_z_after  = max(0, (iz + half_z) - array.shape[0])
        pad_y_before = max(0, half_y - iy)
        pad_y_after  = max(0, (iy + half_y) - array.shape[1])
        pad_x_before = max(0, half_x - ix)
        pad_x_after  = max(0, (ix + half_x) - array.shape[2])

        if any(p > 0 for p in [pad_z_before, pad_z_after, pad_y_before, pad_y_after,
                                 pad_x_before, pad_x_after]):
            raw = np.pad(
                raw,
                ((pad_z_before, pad_z_after),
                 (pad_y_before, pad_y_after),
                 (pad_x_before, pad_x_after)),
                constant_values=clip_hu[0],
            )

        raw = np.clip(raw, clip_hu[0], clip_hu[1])
        raw = (raw - clip_hu[0]) / (clip_hu[1] - clip_hu[0])

        if raw.shape != (patch_size, patch_size, patch_size):
            zoom_z = patch_size / raw.shape[0]
            zoom_y = patch_size / raw.shape[1]
            zoom_x = patch_size / raw.shape[2]
            raw = scipy_zoom(raw, (zoom_z, zoom_y, zoom_x), order=1)

        return raw.astype(np.float32)
```

### Step 2 — Replace `_random_spatial_shift` (lines 563-592) with zero-pad version

Replace the entire method:
```python
    def _random_spatial_shift(
        self, volume: np.ndarray, max_shift: int = 4
    ) -> np.ndarray:
        """
        Desplazamiento espacial aleatorio con zero-padding (reemplaza np.roll circular).

        np.roll introducía artefactos: los vóxeles del borde opuesto aparecían
        en el borde desplazado. Con zero-padding, el borde desplazado se rellena
        con 0.0 (equivalente a aire = -1000 HU normalizado), que es el valor
        correcto del fondo de un parche pulmonar.

        Args:
            volume:    ndarray float32 shape (64, 64, 64)
            max_shift: desplazamiento máximo en vóxeles por eje. Default: 4

        Returns:
            ndarray float32 shape (64, 64, 64) con zero-padding en bordes
        """
        D = volume.shape[0]
        for axis in range(3):
            shift = random.randint(-max_shift, max_shift)
            if shift == 0:
                continue
            shifted = np.zeros_like(volume)
            slices_src = [slice(None)] * 3
            slices_dst = [slice(None)] * 3
            if shift > 0:
                slices_src[axis] = slice(0, D - shift)
                slices_dst[axis] = slice(shift, D)
            else:
                slices_src[axis] = slice(-shift, D)
                slices_dst[axis] = slice(0, D + shift)
            shifted[tuple(slices_dst)] = volume[tuple(slices_src)]
            volume = shifted
        return volume
```

### Step 3 — Replace `_augment_3d` (lines 594-668) with full 8-step pipeline

Replace the entire method:
```python
    def _augment_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Pipeline de data augmentation 3D completo (8 pasos) para parches CT.

        Orden de aplicación:
          1. Flip 3D (P=0.5 por eje)
          2. Rotación 3D completa ±15° en los 3 planos
          3. Zoom 3D aleatorio [0.85, 1.15] con crop/pad central
          4. Deformación elástica 3D (P=0.4)
          5. CutOut volumétrico / RandCoarseDropout (P=0.5, 4 cubos 8³)
          6. Variación de intensidad HU (offset + escala)
          7. Ruido gaussiano (P=0.3)
          8. Traslación espacial ±4 vóxeles con zero-padding

        Entrada y salida: ndarray float32 shape (64,64,64), rango [0,1].
        """
        from scipy.ndimage import zoom as scipy_zoom
        from scipy.ndimage import gaussian_filter, map_coordinates

        D = volume.shape[0]  # 64

        # ── 1. Flip 3D (P=0.5 por eje) ─────────────────────────────────────
        for axis in range(3):
            if random.random() < 0.5:
                volume = np.flip(volume, axis=axis)

        # ── 2. Rotación 3D ±15° en los 3 planos ────────────────────────────
        # axial=(1,2), coronal=(0,2), sagital=(0,1)
        for axes in [(1, 2), (0, 2), (0, 1)]:
            angle = random.uniform(-15.0, 15.0)
            if abs(angle) > 0.5:
                volume = scipy_rotate(
                    volume,
                    angle=angle,
                    axes=axes,
                    reshape=False,
                    order=1,
                    mode="nearest",
                )

        # ── 3. Zoom 3D aleatorio [0.85, 1.15] con crop/pad central ─────────
        zoom_factor = random.uniform(0.85, 1.15)
        zoomed = scipy_zoom(volume, zoom_factor, order=1)
        zD = zoomed.shape[0]
        if zD > D:
            # Crop central
            start = (zD - D) // 2
            volume = zoomed[start:start + D, start:start + D, start:start + D]
        elif zD < D:
            # Pad central con ceros (aire)
            pad_before = (D - zD) // 2
            pad_after = D - zD - pad_before
            volume = np.pad(
                zoomed,
                ((pad_before, pad_after),) * 3,
                constant_values=0.0,
            )
        else:
            volume = zoomed

        # ── 4. Deformación elástica 3D (P=0.4) ─────────────────────────────
        if random.random() < 0.4:
            sigma, alpha = 2.0, 8.0
            shape = volume.shape
            dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
            z_g, y_g, x_g = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing="ij",
            )
            volume = map_coordinates(
                volume, [z_g + dz, y_g + dy, x_g + dx], order=1, mode="nearest"
            ).astype(np.float32)

        # ── 5. CutOut volumétrico (P=0.5, 4 cubos 8³) ──────────────────────
        if random.random() < 0.5:
            for _ in range(4):
                z0 = random.randint(0, D - 8)
                y0 = random.randint(0, D - 8)
                x0 = random.randint(0, D - 8)
                volume[z0:z0 + 8, y0:y0 + 8, x0:x0 + 8] = 0.0

        # ── 6. Variación de intensidad HU ───────────────────────────────────
        hu_range = 1400.0
        offset_norm = random.uniform(-20.0, 20.0) / hu_range
        scale = random.uniform(0.95, 1.05)
        volume = volume * scale + offset_norm

        # ── 7. Ruido gaussiano (P=0.3) ──────────────────────────────────────
        if random.random() < 0.3:
            sigma_noise = random.uniform(0.0, 0.02)
            noise = np.random.normal(0.0, sigma_noise, size=volume.shape).astype(np.float32)
            volume = volume + noise

        volume = np.ascontiguousarray(volume, dtype=np.float32)

        # ── 8. Traslación espacial ±4 vóxeles con zero-padding ──────────────
        volume = self._random_spatial_shift(volume, max_shift=4)

        # Clip final [0, 1]
        volume = np.clip(volume, 0.0, 1.0)

        return volume
```

Also update the docstring in `__init__` that describes the augmentation steps (lines 362-366):

```python
    Augmentation 3D (solo mode="expert" + split="train"):
      - Flip aleatorio en ejes D, H, W (P=0.5 por eje)
      - Rotación 3D completa ±15° en los 3 planos (axial, coronal, sagital)
      - Zoom 3D aleatorio [0.85, 1.15] con crop/pad central
      - Deformación elástica 3D (P=0.4, sigma=2, alpha=8)
      - CutOut volumétrico P=0.5 (4 cubos 8³)
      - Variación de intensidad HU: offset ∈ [-20, +20], escala ∈ [0.95, 1.05]
      - Ruido gaussiano σ~U(0, 0.02) con P=0.3
      - Traslación espacial ±4 vóxeles con zero-padding
```

And update the log message in `__init__` (lines 395-398):

```python
            log.info(
                "[LUNA16] Augmentation 3D ACTIVO para mode='expert', split='train':\n"
                "    - Flip aleatorio 3 ejes (P=0.5 c/u)\n"
                "    - Rotación 3D completa ±15° (axial, coronal, sagital)\n"
                "    - Zoom 3D aleatorio [0.85, 1.15] con crop/pad central\n"
                "    - Deformación elástica 3D (P=0.4, sigma=2, alpha=8)\n"
                "    - CutOut volumétrico P=0.5 (4 cubos 8³ vóxeles)\n"
                "    - Variación intensidad HU offset∈[-20,+20] + escala∈[0.95,1.05]\n"
                "    - Ruido gaussiano σ~U(0,0.02), P=0.3\n"
                "    - Desplazamiento espacial ±4 vóxeles (zero-pad, no circular)"
            )
```

---

## Task 3 — Delete stale patches and re-extract

### Step 1 — Delete all existing patches

```bash
rm -rf /mnt/hdd/datasets/carlos_andres_ferro/proyecto_2/datasets/luna_lung_cancer/patches/train/
rm -rf /mnt/hdd/datasets/carlos_andres_ferro/proyecto_2/datasets/luna_lung_cancer/patches/val/
rm -rf /mnt/hdd/datasets/carlos_andres_ferro/proyecto_2/datasets/luna_lung_cancer/patches/test/
rm -f  /mnt/hdd/datasets/carlos_andres_ferro/proyecto_2/datasets/luna_lung_cancer/patches/extraction_report.json
```

### Step 2 — Re-extract patches

Run from `src/pipeline/`:
```bash
cd /mnt/hdd/datasets/carlos_andres_ferro/proyecto_2/src/pipeline
python - <<'EOF'
import logging, sys
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
sys.path.insert(0, str(Path(".").resolve()))
from fase0.pre_embeddings import run_luna_patches
result = run_luna_patches(
    datasets_dir=Path("../../datasets"),
    workers=6,
    neg_ratio=10,
)
print(result)
EOF
```

Expected: logs showing extraction with "OK" statuses, final report showing ~17k patches total.

### Step 3 — Verify a sample patch covers 50mm physically

```bash
cd /mnt/hdd/datasets/carlos_andres_ferro/proyecto_2/src/pipeline
python - <<'EOF'
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Load one patch and confirm it is 64³ and in [0,1]
patch_files = list(Path("../../datasets/luna_lung_cancer/patches/train").glob("candidate_*.npy"))[:5]
for p in patch_files:
    arr = np.load(p)
    print(f"{p.name}: shape={arr.shape}  min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}")
EOF
```

Expected output per patch: `shape=(64, 64, 64)  min≥0.0  max≤1.0  mean∈[0.05, 0.5]`

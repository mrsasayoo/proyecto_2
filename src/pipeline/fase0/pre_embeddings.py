#!/usr/bin/env python3
"""
pre_embeddings.py — Preparación de datos 3D para Fase 1
========================================================
Fase 0 — Preparación de Datos | Proyecto MoE Médico

Responsabilidad única: transformar datos 3D antes de que puedan ser usados
para extracción de embeddings en Fase 1.

  1. LUNA16: extraer parches .npy 64³ centrados en cada candidato
  2. Páncreas: validar pipeline de preprocesado isotrópico

Origen: extract_luna_patches.py completo, pancreas_preprocess.py
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("fase0.pre_embeddings")

# ── Constantes LUNA ──────────────────────────────────────────────────────────
HU_LUNG_CLIP = (-1000, 400)
PATCH_SIZE = 64
PATCH_HALF = 32  # half of 64 voxels at 1mm isotropic = 32mm each side
SEG_DIR_NAME = "seg-lungs-LUNA16"
RANDOM_SEED = 42

# Mapa ElementType → bytes por vóxel (MetaImage)
_MHD_ELEMENT_BYTES = {
    "MET_UCHAR": 1,
    "MET_CHAR": 1,
    "MET_USHORT": 2,
    "MET_SHORT": 2,
    "MET_UINT": 4,
    "MET_INT": 4,
    "MET_FLOAT": 4,
    "MET_DOUBLE": 8,
}


def _parse_mhd_expected_bytes(mhd_path):
    # type: (Path) -> int | None
    """Lee el header .mhd y calcula el tamaño esperado del .raw en bytes.

    Retorna None si el header no contiene la información necesaria.
    """
    dim_size = None
    element_type = None
    n_channels = 1

    try:
        text = Path(mhd_path).read_text(errors="replace")
    except Exception:
        return None

    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        if key == "DimSize":
            try:
                dim_size = [int(x) for x in value.split()]
            except ValueError:
                return None
        elif key == "ElementType":
            element_type = value
        elif key == "ElementNumberOfChannels":
            try:
                n_channels = int(value)
            except ValueError:
                n_channels = 1

    if dim_size is None or element_type is None:
        return None

    bpe = _MHD_ELEMENT_BYTES.get(element_type)
    if bpe is None:
        return None

    n_voxels = 1
    for d in dim_size:
        n_voxels *= d
    return n_voxels * bpe * n_channels


# ══════════════════════════════════════════════════════════════════════════════
#  LUNA16 — Utilidades de extracción
# ══════════════════════════════════════════════════════════════════════════════


def world_to_voxel(coord_world, origin, spacing, direction):
    """Conversión world→vóxel correcta (SimpleITK convention)."""
    coord_shifted = np.array(coord_world) - np.array(origin)
    coord_voxel = np.linalg.solve(
        np.array(direction).reshape(3, 3) * np.array(spacing),
        coord_shifted,
    )
    return np.round(coord_voxel[::-1]).astype(int)  # [iz, iy, ix]


def extract_patch(mhd_path, coord_world, seg_dir, clip_hu=HU_LUNG_CLIP):
    """Extrae parche 64×64×64 a 1mm isotrópico centrado en coord_world.

    Pipeline de 7 pasos:
      1. Carga + conversión HU (SimpleITK aplica slope/intercept)
      2. Resampleo isotrópico a 1×1×1 mm³
      3. Máscara de segmentación pulmonar (fuera-de-pulmón → -1000 HU)
      4. Clip HU [-1000, +400]
      5. Normalización min-max a [0, 1]
      6. (Zero-centering se aplica por separado en bulk)
      7. Extracción del parche 64³ con zero-pad en bordes

    Args:
        mhd_path:    ruta al .mhd del CT
        coord_world: [x, y, z] en mm (world coordinates)
        seg_dir:     ruta al directorio seg-lungs-LUNA16/seg-lungs-LUNA16/
        clip_hu:     tupla (min_hu, max_hu)

    Returns:
        float32 ndarray (64, 64, 64) en rango [0, 1]
    """
    import SimpleITK as sitk
    from scipy.ndimage import zoom as scipy_zoom

    # ── Step 1: Load + HU conversion ────────────────────────────────────
    image = sitk.ReadImage(str(mhd_path))
    origin = np.array(image.GetOrigin())  # [x, y, z] mm
    spacing = np.array(image.GetSpacing())  # [sx, sy, sz] mm/voxel (XYZ)
    direc = np.array(image.GetDirection())
    array = sitk.GetArrayFromImage(image).astype(np.float32)  # [Z, Y, X]
    array[array < -1000] = -1000.0  # clamp outside-FOV

    # ── Step 2: Isotropic resampling to 1×1×1 mm³ ───────────────────────
    zoom_factors = (spacing[2], spacing[1], spacing[0])  # (z, y, x)
    array = scipy_zoom(array, zoom_factors, order=1)

    # ── Step 3: Apply lung segmentation mask ─────────────────────────────
    seg_dir = Path(seg_dir)
    uid = Path(mhd_path).stem
    seg_path = seg_dir / (uid + ".mhd")
    if seg_path.exists():
        mask_img = sitk.ReadImage(str(seg_path))
        mask_arr = sitk.GetArrayFromImage(mask_img)  # [Z, Y, X]
        mask_arr = scipy_zoom(mask_arr.astype(np.float32), zoom_factors, order=0)
        mask_arr = (mask_arr > 0.5).astype(np.uint8)  # binarize after zoom
        # Trim or pad mask if shapes differ slightly due to rounding
        min_z = min(array.shape[0], mask_arr.shape[0])
        min_y = min(array.shape[1], mask_arr.shape[1])
        min_x = min(array.shape[2], mask_arr.shape[2])
        array[:min_z, :min_y, :min_x][mask_arr[:min_z, :min_y, :min_x] == 0] = -1000.0
        # If mask is smaller, set the remainder to air
        if min_z < array.shape[0]:
            array[min_z:, :, :] = -1000.0
        if min_y < array.shape[1]:
            array[:, min_y:, :] = -1000.0
        if min_x < array.shape[2]:
            array[:, :, min_x:] = -1000.0

    # ── Step 4: Clip HU ─────────────────────────────────────────────────
    array = np.clip(array, clip_hu[0], clip_hu[1])

    # ── Step 5: Min-max normalization to [0, 1] ─────────────────────────
    array = (array - clip_hu[0]) / (clip_hu[1] - clip_hu[0])

    # ── Step 7: Extract 64×64×64 voxel patch ────────────────────────────
    # Convert world→voxel in ORIGINAL space, then scale to resampled space
    iz_orig, iy_orig, ix_orig = world_to_voxel(coord_world, origin, spacing, direc)
    iz = int(round(iz_orig * spacing[2]))  # scale to 1mm isotropic
    iy = int(round(iy_orig * spacing[1]))
    ix = int(round(ix_orig * spacing[0]))

    half = PATCH_HALF  # 32

    # Slice with boundary handling
    z1 = max(0, iz - half)
    z2 = min(array.shape[0], iz + half)
    y1 = max(0, iy - half)
    y2 = min(array.shape[1], iy + half)
    x1 = max(0, ix - half)
    x2 = min(array.shape[2], ix + half)

    raw = array[z1:z2, y1:y2, x1:x2]

    # Zero-pad if near volume border (pad value 0.0 = air after normalization)
    pad_z_before = max(0, half - iz)
    pad_z_after = max(0, (iz + half) - array.shape[0])
    pad_y_before = max(0, half - iy)
    pad_y_after = max(0, (iy + half) - array.shape[1])
    pad_x_before = max(0, half - ix)
    pad_x_after = max(0, (ix + half) - array.shape[2])

    if any(
        p > 0
        for p in [
            pad_z_before,
            pad_z_after,
            pad_y_before,
            pad_y_after,
            pad_x_before,
            pad_x_after,
        ]
    ):
        raw = np.pad(
            raw,
            (
                (pad_z_before, pad_z_after),
                (pad_y_before, pad_y_after),
                (pad_x_before, pad_x_after),
            ),
            constant_values=0.0,
        )

    return raw.astype(np.float32)


# ── Worker para ProcessPoolExecutor ──────────────────────────────────────────


def _worker(args_tuple):
    """
    Procesa un lote de candidatos de un mismo seriesuid.
    Carga y resamplea CT + máscara UNA VEZ por CT, luego extrae todos los parches.
    Retorna lista de (row_idx, out_path, label, status, mean_val).
    """
    import SimpleITK as sitk
    from scipy.ndimage import zoom as scipy_zoom

    mhd_path, candidates_batch, out_dir, seg_dir = args_tuple
    results = []

    # ── Step 1: Load + HU conversion ────────────────────────────────────
    try:
        image = sitk.ReadImage(str(mhd_path))
        origin = np.array(image.GetOrigin())
        spacing = np.array(image.GetSpacing())  # [sx, sy, sz] XYZ
        direc = np.array(image.GetDirection())
        array = sitk.GetArrayFromImage(image).astype(np.float32)  # [Z, Y, X]
        array[array < -1000] = -1000.0
    except Exception as e:
        for row_idx, _cx, _cy, _cz, label in candidates_batch:
            results.append((row_idx, None, label, "ERROR_READ:{}".format(e), 0.0))
        return results

    # ── Step 2: Isotropic resampling to 1×1×1 mm³ ───────────────────────
    zoom_factors = (spacing[2], spacing[1], spacing[0])  # (z, y, x)
    try:
        array = scipy_zoom(array, zoom_factors, order=1)
    except Exception as e:
        for row_idx, _cx, _cy, _cz, label in candidates_batch:
            results.append((row_idx, None, label, "ERROR_RESAMPLE:{}".format(e), 0.0))
        return results

    # ── Step 3: Apply lung segmentation mask ─────────────────────────────
    uid = Path(mhd_path).stem
    seg_path = Path(seg_dir) / (uid + ".mhd")
    if seg_path.exists():
        try:
            mask_img = sitk.ReadImage(str(seg_path))
            mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.float32)
            mask_arr = scipy_zoom(mask_arr, zoom_factors, order=0)
            mask_arr = (mask_arr > 0.5).astype(np.uint8)
            min_z = min(array.shape[0], mask_arr.shape[0])
            min_y = min(array.shape[1], mask_arr.shape[1])
            min_x = min(array.shape[2], mask_arr.shape[2])
            array[:min_z, :min_y, :min_x][
                mask_arr[:min_z, :min_y, :min_x] == 0
            ] = -1000.0
            if min_z < array.shape[0]:
                array[min_z:, :, :] = -1000.0
            if min_y < array.shape[1]:
                array[:, min_y:, :] = -1000.0
            if min_x < array.shape[2]:
                array[:, :, min_x:] = -1000.0
        except Exception as e:
            log.warning(
                "[Worker] Error loading mask for %s: %s — continuing without mask",
                uid,
                e,
            )

    # ── Step 4: Clip HU ─────────────────────────────────────────────────
    array = np.clip(array, HU_LUNG_CLIP[0], HU_LUNG_CLIP[1])

    # ── Step 5: Min-max normalization to [0, 1] ─────────────────────────
    array = (array - HU_LUNG_CLIP[0]) / (HU_LUNG_CLIP[1] - HU_LUNG_CLIP[0])

    # ── Step 7: Extract patches for each candidate ──────────────────────
    half = PATCH_HALF  # 32

    for row_idx, cx, cy, cz, label in candidates_batch:
        out_path = Path(out_dir) / "candidate_{:06d}.npy".format(row_idx)
        if out_path.exists():
            try:
                arr = np.load(out_path, mmap_mode="r")
                if arr.shape == (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
                    results.append((row_idx, str(out_path), label, "SKIPPED", -1.0))
                    continue
                else:
                    out_path.unlink()
            except Exception:
                try:
                    out_path.unlink()
                except Exception:
                    pass
        try:
            # Convert world→voxel in original space, scale to resampled 1mm space
            iz_orig, iy_orig, ix_orig = world_to_voxel(
                [cx, cy, cz], origin, spacing, direc
            )
            iz = int(round(iz_orig * spacing[2]))
            iy = int(round(iy_orig * spacing[1]))
            ix = int(round(ix_orig * spacing[0]))

            z1 = max(0, iz - half)
            z2 = min(array.shape[0], iz + half)
            y1 = max(0, iy - half)
            y2 = min(array.shape[1], iy + half)
            x1 = max(0, ix - half)
            x2 = min(array.shape[2], ix + half)

            raw = array[z1:z2, y1:y2, x1:x2]

            pad_z_before = max(0, half - iz)
            pad_z_after = max(0, (iz + half) - array.shape[0])
            pad_y_before = max(0, half - iy)
            pad_y_after = max(0, (iy + half) - array.shape[1])
            pad_x_before = max(0, half - ix)
            pad_x_after = max(0, (ix + half) - array.shape[2])

            if any(
                p > 0
                for p in [
                    pad_z_before,
                    pad_z_after,
                    pad_y_before,
                    pad_y_after,
                    pad_x_before,
                    pad_x_after,
                ]
            ):
                raw = np.pad(
                    raw,
                    (
                        (pad_z_before, pad_z_after),
                        (pad_y_before, pad_y_after),
                        (pad_x_before, pad_x_after),
                    ),
                    constant_values=0.0,
                )

            patch = raw.astype(np.float32)
            np.save(out_path, patch)
            results.append((row_idx, str(out_path), label, "OK", float(patch.mean())))
        except Exception as e:
            results.append((row_idx, None, label, "ERROR:{}".format(e), 0.0))

    return results


def apply_neg_sampling(df_split, split_name, neg_ratio, seed=42):
    """Muestreo controlado de negativos (ratio N:1 respecto a positivos)."""
    n_pos = (df_split["class"] == 1).sum()
    n_neg_keep = n_pos * neg_ratio
    df_pos = df_split[df_split["class"] == 1]
    df_neg = df_split[df_split["class"] == 0]
    if len(df_neg) > n_neg_keep:
        df_neg = df_neg.sample(n=n_neg_keep, random_state=seed)
    df_sampled = pd.concat([df_pos, df_neg]).reset_index(drop=True)
    log.info(
        "[NegSampling/%s] pos=%d | neg=%d→%d (ratio %d:1) | total=%d",
        split_name,
        len(df_pos),
        int((df_split["class"] == 0).sum()),
        len(df_neg),
        neg_ratio,
        len(df_sampled),
    )
    return df_sampled


def validate_patches(patches_dir, n_sample=20):
    """Valida shape y media de intensidad en una muestra de parches."""
    patches = list(Path(patches_dir).glob("candidate_*.npy"))
    if not patches:
        log.error("[Validate] Sin parches en %s", patches_dir)
        return False

    random.seed(42)
    sample = random.sample(patches, min(n_sample, len(patches)))
    errors = 0
    for p in sample:
        arr = np.load(p)
        if arr.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
            log.error("[Validate] %s: shape %s ≠ (%d³)", p.name, arr.shape, PATCH_SIZE)
            errors += 1
        mean_v = arr.mean()
        status = "✓" if mean_v > 0.05 else "⚠ POSIBLE ERROR"
        log.info("  %s  mean=%.3f  %s", p.name, mean_v, status)

    log.info("[Validate] %d parches revisados | errores shape: %d", len(sample), errors)
    return errors == 0


# ══════════════════════════════════════════════════════════════════════════════
#  LUNA16 — Extracción principal de parches
# ══════════════════════════════════════════════════════════════════════════════


def run_luna_patches(
    datasets_dir,
    workers=6,
    neg_ratio=10,
    max_neg=None,
    luna_subsets=None,
    dry_run=False,
):
    # type: (Path, int, int, int|None, list|None, bool) -> dict
    """
    Extrae parches 3D de LUNA16 respetando los splits de luna_splits.json.
    Deposita en patches/train/, patches/val/, patches/test/.
    """
    luna_dir = datasets_dir / "luna_lung_cancer"
    ct_dir = luna_dir / "ct_volumes"
    csv_path = luna_dir / "candidates_V2" / "candidates_V2.csv"
    splits_file = luna_dir / "luna_splits.json"
    patches_dir = luna_dir / "patches"
    seg_dir = luna_dir / SEG_DIR_NAME / SEG_DIR_NAME

    # Descubrir subsets
    available = sorted(
        d for d in ct_dir.iterdir() if d.is_dir() and d.name.startswith("subset")
    )
    if not available:
        log.warning("[LUNA] No se encontraron subsets en %s", ct_dir)
        return {"status": "Warning", "reason": "sin CTs"}

    mhd_files = []
    for sd in available:
        found = list(sd.rglob("*.mhd"))
        log.info("[LUNA] %s: %d CTs (.mhd)", sd.name, len(found))
        mhd_files.extend(found)

    if not mhd_files:
        return {"status": "Warning", "reason": "sin .mhd"}

    mhd_map = {p.stem: p for p in mhd_files}

    # Filtrar por subsets solicitados
    if luna_subsets is not None:
        target_names = {"subset{}".format(s) for s in luna_subsets}
        mhd_map = {
            uid: path
            for uid, path in mhd_map.items()
            if any(tn in str(path) for tn in target_names)
        }
        log.info("[LUNA] Filtrado a subsets %s: %d CTs", luna_subsets, len(mhd_map))

    # Filtrar CTs con .raw ausente o corrompido (< 1 MB)
    MIN_RAW_BYTES = 1_048_576  # 1 MB
    valid_mhd_map = {}
    skipped_corrupt = []
    for uid, mhd_path in mhd_map.items():
        raw_path = mhd_path.with_suffix(".raw")
        if raw_path.exists() and raw_path.stat().st_size >= MIN_RAW_BYTES:
            valid_mhd_map[uid] = mhd_path
        else:
            skipped_corrupt.append(uid)
            log.warning("[LUNA] CT omitido (raw ausente o corrupto): %s", uid)
    if skipped_corrupt:
        log.warning("[LUNA] %d CTs omitidos por raw inválido", len(skipped_corrupt))
    mhd_map = valid_mhd_map

    # Segundo filtro: verificar tamaño real vs. declarado en header .mhd
    checked_mhd_map = {}
    skipped_truncated = []
    for uid, mhd_path in mhd_map.items():
        expected = _parse_mhd_expected_bytes(mhd_path)
        if expected is not None:
            raw_path = mhd_path.with_suffix(".raw")
            actual = raw_path.stat().st_size
            if actual < expected * 0.95:
                pct = 100.0 * actual / expected
                log.warning(
                    "[LUNA] CT omitido (raw truncado, %.1f%% completo): %s",
                    pct,
                    uid,
                )
                skipped_truncated.append(uid)
                continue
        checked_mhd_map[uid] = mhd_path
    if skipped_truncated:
        log.warning("[LUNA] %d CTs omitidos por raw truncado", len(skipped_truncated))
    mhd_map = checked_mhd_map

    log.info("[LUNA] Total: %d CTs en %d subsets", len(mhd_map), len(available))

    # Cargar CSV de candidatos
    if not csv_path.exists():
        return {"status": "error", "reason": "candidates_V2.csv no encontrado"}

    df = pd.read_csv(csv_path)
    df_sub = df[df["seriesuid"].isin(mhd_map)].copy()
    df_sub = df_sub.reset_index(drop=False).rename(columns={"index": "original_idx"})

    log.info(
        "[LUNA] %d candidatos en subsets disponibles (pos=%d, neg=%d)",
        len(df_sub),
        (df_sub["class"] == 1).sum(),
        (df_sub["class"] == 0).sum(),
    )

    # --max_neg
    if max_neg is not None:
        df_pos = df_sub[df_sub["class"] == 1]
        df_neg = df_sub[df_sub["class"] == 0]
        rng_neg = np.random.default_rng(RANDOM_SEED + 1)
        if len(df_neg) > max_neg:
            neg_idx = rng_neg.choice(len(df_neg), size=max_neg, replace=False)
            df_neg = df_neg.iloc[neg_idx].reset_index(drop=True)
        df_sub = pd.concat([df_pos, df_neg], ignore_index=True)
        log.info("[LUNA] --max_neg=%d: %d candidatos finales", max_neg, len(df_sub))

    if dry_run:
        est_gb = len(df_sub) * 64**3 * 4 / 1e9
        log.info(
            "[LUNA DRY-RUN] CTs=%d | candidatos=%d | ~%.2f GB",
            len(mhd_map),
            len(df_sub),
            est_gb,
        )
        return {"status": "Info", "dry_run": True, "candidates": len(df_sub)}

    # Cargar splits
    if splits_file.exists():
        with open(splits_file) as f:
            splits_data = json.load(f)
        train_uids = set(splits_data["train_uids"])
        val_uids = set(splits_data["val_uids"])
        test_uids = set(splits_data["test_uids"])
        log.info(
            "[LUNA] luna_splits.json: train=%d val=%d test=%d",
            len(train_uids),
            len(val_uids),
            len(test_uids),
        )
    else:
        log.warning(
            "[LUNA] luna_splits.json no encontrado — usando split 80/10/10 ad-hoc"
        )
        rng = np.random.default_rng(RANDOM_SEED)
        all_uids = np.array(list(mhd_map.keys()))
        rng.shuffle(all_uids)
        n_test = int(0.10 * len(all_uids))
        n_val = int(0.10 * len(all_uids))
        test_uids = set(all_uids[:n_test])
        val_uids = set(all_uids[n_test : n_test + n_val])
        train_uids = set(all_uids[n_test + n_val :])

    # Asignar split a cada candidato
    def uid_to_split(uid):
        if uid in train_uids:
            return "train"
        if uid in val_uids:
            return "val"
        if uid in test_uids:
            return "test"
        return "train"  # fallback para UIDs no en splits

    df_sub["split"] = df_sub["seriesuid"].apply(uid_to_split)

    # Crear directorios
    for sp in ["train", "val", "test"]:
        (patches_dir / sp).mkdir(parents=True, exist_ok=True)

    # Extraer por split
    report_data = {}
    for split_name in ["train", "val", "test"]:
        df_split = df_sub[df_sub["split"] == split_name].copy()
        if len(df_split) == 0:
            log.info("[LUNA/%s] Sin candidatos, saltando.", split_name)
            continue

        # Negative sampling
        if neg_ratio > 0:
            seed_offset = {"train": 42, "val": 43, "test": 44}[split_name]
            df_split = apply_neg_sampling(
                df_split, split_name.upper(), neg_ratio, seed=seed_offset
            )

        out_dir = patches_dir / split_name

        # Agrupar por seriesuid
        tasks = []
        for uid, group in df_split.groupby("seriesuid"):
            if uid not in mhd_map:
                continue
            batch = [
                (
                    int(row["original_idx"]),
                    row["coordX"],
                    row["coordY"],
                    row["coordZ"],
                    int(row["class"]),
                )
                for _, row in group.iterrows()
            ]
            tasks.append((mhd_map[uid], batch, str(out_dir), str(seg_dir)))

        total_cands = len(df_split)
        log.info(
            "[LUNA/%s] Extrayendo %d parches de %d CTs con %d workers...",
            split_name.upper(),
            total_cands,
            len(tasks),
            workers,
        )

        t0 = time.time()
        done = ok = skip = errs = 0
        pos_means = []

        with ProcessPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    batch_results = fut.result()
                except Exception as e:
                    log.error("[LUNA/%s] Worker falló: %s", split_name.upper(), e)
                    continue
                for row_idx, out_path, label, status, mean_v in batch_results:
                    done += 1
                    if status == "OK":
                        ok += 1
                        if label == 1:
                            pos_means.append(mean_v)
                    elif status == "SKIPPED":
                        skip += 1
                    else:
                        errs += 1
                    if done % 5000 == 0 or done == total_cands:
                        elapsed = time.time() - t0
                        rate = done / max(elapsed, 1)
                        log.info(
                            "  [%s] %d/%d (%.1f%%) | %.0f cand/s | OK=%d SKIP=%d ERR=%d",
                            split_name.upper(),
                            done,
                            total_cands,
                            100 * done / total_cands,
                            rate,
                            ok,
                            skip,
                            errs,
                        )

        elapsed = time.time() - t0
        log.info(
            "[LUNA/%s] Completado en %.0fs | OK=%d SKIP=%d ERR=%d",
            split_name.upper(),
            elapsed,
            ok,
            skip,
            errs,
        )

        if pos_means:
            mean_pos = np.mean(pos_means)
            if mean_pos < 0.1:
                log.error(
                    "[LUNA/%s] Media positivos=%.4f < 0.1 — posible error w2v",
                    split_name.upper(),
                    mean_pos,
                )

        n_patches = len(list(out_dir.glob("candidate_*.npy")))
        report_data[split_name] = {
            "patches": n_patches,
            "pos": int((df_split["class"] == 1).sum()),
            "neg": int((df_split["class"] == 0).sum()),
        }

        # ── Generate per-split manifest CSV ─────────────────────────────
        # Maps each .npy filename to its label so that downstream consumers
        # (e.g. Kaggle notebooks) don't need candidates_V2.csv.
        manifest_rows = []
        for _, row in df_split.iterrows():
            fname = "candidate_{:06d}.npy".format(int(row["original_idx"]))
            fpath = out_dir / fname
            if fpath.exists():
                manifest_rows.append({"filename": fname, "label": int(row["class"])})
        if manifest_rows:
            manifest_df = pd.DataFrame(manifest_rows).sort_values("filename")
            manifest_path = out_dir / "manifest.csv"
            manifest_df.to_csv(manifest_path, index=False)
            log.info(
                "[LUNA/%s] Manifest saved: %d entries → %s",
                split_name.upper(),
                len(manifest_df),
                manifest_path,
            )

    # Validación
    log.info("[LUNA] Validando muestra de parches...")
    ok_all = True
    for sp in ["train", "val", "test"]:
        sp_dir = patches_dir / sp
        if any(sp_dir.glob("candidate_*.npy")):
            ok_sp = validate_patches(sp_dir, n_sample=10)
            ok_all = ok_all and ok_sp

    # ── Step 6: Compute global mean over ALL training patches ────────────
    log.info("[LUNA] Computing global mean over training patches...")
    train_dir = patches_dir / "train"
    train_patches = list(train_dir.glob("candidate_*.npy"))
    global_mean = np.float32(0.0)
    if train_patches:
        running_sum = 0.0
        running_count = 0
        for pp in train_patches:
            try:
                arr = np.load(pp)
                running_sum += arr.sum()
                running_count += arr.size
            except Exception:
                pass
        if running_count > 0:
            global_mean = np.float32(running_sum / running_count)
        log.info(
            "[LUNA] Global mean (train): %.6f (computed over %d patches)",
            global_mean,
            len(train_patches),
        )
    else:
        log.warning("[LUNA] No training patches found — global_mean=0.0")

    global_mean_path = patches_dir / "global_mean.npy"
    np.save(global_mean_path, global_mean)
    log.info("[LUNA] Saved global mean to %s", global_mean_path)

    # Apply zero-centering (subtract global_mean) to ALL patches in-place
    log.info(
        "[LUNA] Applying zero-centering (subtracting global_mean=%.6f) to all patches...",
        global_mean,
    )
    for sp in ["train", "val", "test"]:
        sp_dir = patches_dir / sp
        sp_patches = list(sp_dir.glob("candidate_*.npy"))
        for pp in sp_patches:
            try:
                arr = np.load(pp).astype(np.float32)
                arr = arr - global_mean
                np.save(pp, arr)
            except Exception as e:
                log.warning("[LUNA] Error zero-centering %s: %s", pp.name, e)
        log.info("[LUNA] Zero-centered %d patches in %s/", len(sp_patches), sp)

    # Reporte
    report = {
        "cts_total": len(mhd_files),
        "subsets": [d.name for d in available],
        "splits": report_data,
        "neg_ratio": neg_ratio,
        "max_neg": max_neg,
        "patch_size": PATCH_SIZE,
        "physical_mm": 64,
        "resample_to_1mm": True,
        "lung_mask": True,
        "hu_clip": list(HU_LUNG_CLIP),
        "global_mean": float(global_mean),
        "validation_ok": ok_all,
    }
    report_path = patches_dir / "extraction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("[LUNA] Reporte: %s", report_path)
    return {"status": "✅", **report}


# ══════════════════════════════════════════════════════════════════════════════
#  Páncreas — Validación de preprocesado isotrópico
# ══════════════════════════════════════════════════════════════════════════════


def preprocess_pancreas_volume(nii_path):
    # type: (str) -> np.ndarray
    """Retorna array float32 [64,64,64] normalizado a [0,1]."""
    import nibabel as nib
    import scipy.ndimage as ndimage

    HU_MIN, HU_MAX = -100, 400
    TARGET_SHAPE = (64, 64, 64)

    nii = nib.load(nii_path)
    vol = nii.get_fdata(dtype=np.float32)
    sp = np.array(nii.header.get_zooms()[:3], dtype=np.float64)

    vol = np.clip(vol, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)

    min_sp = max(sp.min(), 1e-6)
    zoom_factors = sp / min_sp
    vol_iso = ndimage.zoom(vol, zoom_factors, order=1)

    final_zoom = np.array(TARGET_SHAPE, dtype=np.float64) / np.array(
        vol_iso.shape, dtype=np.float64
    )
    vol_final = ndimage.zoom(vol_iso, final_zoom, order=1)
    return vol_final.astype(np.float32)


def validar_preprocesado_pancreas(datasets_dir, n_sample=10):
    # type: (Path, int) -> dict
    """
    Aplica pipeline de preprocesado sobre una muestra de volúmenes y verifica
    shape (64,64,64) y rango [0,1].
    """
    zenodo_dir = datasets_dir / "zenodo_13715870"
    if not zenodo_dir.is_dir():
        log.warning("[PANCREAS] zenodo_13715870/ no encontrado.")
        return {"status": "Warning", "reason": "directorio no encontrado"}

    nii_files = sorted(zenodo_dir.glob("*.nii.gz"))
    if not nii_files:
        log.warning("[PANCREAS] Sin archivos .nii.gz en zenodo_13715870/")
        return {"status": "Warning", "reason": "sin .nii.gz"}

    random.seed(42)
    sample = random.sample(nii_files, min(n_sample, len(nii_files)))
    log.info(
        "[PANCREAS] Validando preprocesado en %d de %d volúmenes...",
        len(sample),
        len(nii_files),
    )

    results_list = []
    ok_count = 0
    for nii_path in sample:
        entry = {"file": nii_path.name}
        try:
            vol = preprocess_pancreas_volume(str(nii_path))
            entry["shape"] = list(vol.shape)
            entry["min"] = round(float(vol.min()), 4)
            entry["max"] = round(float(vol.max()), 4)
            entry["mean"] = round(float(vol.mean()), 4)
            entry["std"] = round(float(vol.std()), 4)
            entry["has_nan"] = bool(np.isnan(vol).any())
            entry["has_inf"] = bool(np.isinf(vol).any())

            ok = (
                vol.shape == (64, 64, 64)
                and vol.min() >= -0.01
                and vol.max() <= 1.01
                and not entry["has_nan"]
                and not entry["has_inf"]
            )
            entry["ok"] = ok
            if ok:
                ok_count += 1
            log.info(
                "  %s: shape=%s range=[%.3f,%.3f] std=%.3f %s",
                nii_path.name,
                vol.shape,
                vol.min(),
                vol.max(),
                vol.std(),
                "✓" if ok else "✗",
            )
        except Exception as e:
            entry["ok"] = False
            entry["error"] = str(e)
            log.error("  %s: ERROR %s", nii_path.name, e)
        results_list.append(entry)

    # Criterio: ≥5 de 10 con std > 0.01 (no vacíos/constantes)
    n_nontrivial = sum(
        1 for r in results_list if r.get("ok") and r.get("std", 0) > 0.01
    )

    report = {
        "n_sample": len(sample),
        "n_ok": ok_count,
        "n_nontrivial": n_nontrivial,
        "results": results_list,
    }

    report_path = datasets_dir / "pancreas_preprocess_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    all_ok = ok_count == len(sample) and n_nontrivial >= min(5, len(sample))
    log.info(
        "[PANCREAS] Validación: %d/%d OK | %d no triviales | %s",
        ok_count,
        len(sample),
        n_nontrivial,
        "✅ PASS" if all_ok else "⚠️ REVISAR",
    )
    return {"status": "✅" if all_ok else "⚠️", **report}


# ══════════════════════════════════════════════════════════════════════════════
#  Paso 6b — Zero-centering fix (inlined from fix_zerocentering.py)
# ══════════════════════════════════════════════════════════════════════════════

# Constants for zero-centering
_ZC_GLOBAL_MEAN = 0.09921630471944809
_ZC_MIN_VALID_MEAN = -_ZC_GLOBAL_MEAN
_ZC_SPLITS = ("train", "val", "test", "train_aug")
_ZC_NUM_WORKERS = 8
_ZC_CHUNK_SIZE = 100
_ZC_PROGRESS_EVERY = 500


def _zc_process_one(path_str):
    # type: (str) -> tuple
    """Load a .npy patch, fix over-centering if needed, return status.

    Runs inside a worker process. Idempotent: patches with mean >= MIN_VALID_MEAN
    are left untouched.
    """
    try:
        arr = np.load(path_str, allow_pickle=False).astype(np.float32)
        mean = float(arr.mean())

        # Guard: NaN / Inf / degenerate arrays → treat as corrupt
        if not np.isfinite(mean):
            return (path_str, "CORRUPT")

        if mean < _ZC_MIN_VALID_MEAN:
            # Safety limit: at most 10 corrections (prevents infinite loop)
            for _ in range(10):
                arr += _ZC_GLOBAL_MEAN
                if float(arr.mean()) >= _ZC_MIN_VALID_MEAN:
                    break
            np.save(path_str, arr)
            return (path_str, "FIXED")

        return (path_str, "OK")

    except Exception:
        return (path_str, "CORRUPT")


def _paso6b_fix_zerocentering(patches_dir, dry_run=False):
    # type: (Path, bool) -> dict
    """Fix over-centered LUNA16 patches (paso 6b).

    Idempotent: patches already within valid range are skipped. Running
    multiple times on already-fixed data is a no-op.
    """
    log.info("[6b] ── Zero-centering fix ──")
    log.info(
        "[6b] GLOBAL_MEAN=%.11f  MIN_VALID_MEAN=%.11f",
        _ZC_GLOBAL_MEAN,
        _ZC_MIN_VALID_MEAN,
    )
    log.info("[6b] patches_dir=%s", patches_dir)

    if dry_run:
        log.info("[6b] DRY-RUN: would scan & fix patches in %s", patches_dir)
        return {"status": "⏭️", "dry_run": True}

    # Suppress numpy overflow warnings from corrupt patches
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    ctx = mp.get_context("fork")
    all_splits_stats = {}
    total_fixed_all = 0

    for split in _ZC_SPLITS:
        split_dir = patches_dir / split
        if not split_dir.is_dir():
            log.info("[6b] Split %s/ not found, skipping", split)
            continue

        files = sorted(
            str(split_dir / f) for f in os.listdir(split_dir) if f.endswith(".npy")
        )
        total = len(files)
        log.info("[6b] Processing split=%s  files=%d", split, total)

        counters = {"total": total, "fixed": 0, "ok": 0, "corrupt": 0}
        processed = 0

        with ctx.Pool(processes=_ZC_NUM_WORKERS) as pool:
            for path_str, status in pool.imap_unordered(
                _zc_process_one, files, chunksize=_ZC_CHUNK_SIZE
            ):
                if status == "FIXED":
                    counters["fixed"] += 1
                elif status == "OK":
                    counters["ok"] += 1
                else:
                    counters["corrupt"] += 1

                processed += 1
                if processed % _ZC_PROGRESS_EVERY == 0:
                    log.info(
                        "[6b]   %s: %d/%d  fixed=%d ok=%d corrupt=%d",
                        split,
                        processed,
                        total,
                        counters["fixed"],
                        counters["ok"],
                        counters["corrupt"],
                    )

        all_splits_stats[split] = {
            "fixed": counters["fixed"],
            "ok": counters["ok"],
            "corrupt": counters["corrupt"],
        }
        total_fixed_all += counters["fixed"]
        log.info(
            "[6b] %s DONE — total=%d fixed=%d ok=%d corrupt=%d",
            split,
            total,
            counters["fixed"],
            counters["ok"],
            counters["corrupt"],
        )

    # Save report JSON
    try:
        report_path = patches_dir / "fix_zerocentering_report.json"
        import datetime as _dt

        zc_report = {
            "global_mean": _ZC_GLOBAL_MEAN,
            "splits": all_splits_stats,
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        with open(report_path, "w") as f:
            json.dump(zc_report, f, indent=2)
        log.info("[6b] Report saved → %s", report_path)
    except Exception as e:
        log.warning("[6b] Could not save report: %s", e)

    if total_fixed_all == 0:
        log.info("[6b] All patches already clean — nothing to fix")
        return {"status": "✅", "already_clean": True, "splits": all_splits_stats}

    return {"status": "✅", "splits": all_splits_stats}


# ══════════════════════════════════════════════════════════════════════════════
#  Paso 6c — Create augmented training set (calls create_augmented_train.py)
# ══════════════════════════════════════════════════════════════════════════════


def _paso6c_create_aug(patches_dir, dry_run=False):
    # type: (Path, bool) -> dict
    """Create train_aug/ with augmented positive patches (paso 6c).

    Idempotent: if train_aug/ and train_aug_manifest.csv already exist with
    sufficient rows, this step is skipped.
    """
    log.info("[6c] ── Create augmented training set ──")

    if dry_run:
        log.info("[6c] DRY-RUN: would create train_aug in %s", patches_dir)
        return {"status": "⏭️", "dry_run": True}

    # Idempotence check
    train_aug_dir = patches_dir / "train_aug"
    manifest_csv = patches_dir / "train_aug_manifest.csv"

    if train_aug_dir.is_dir() and manifest_csv.exists():
        try:
            df_manifest = pd.read_csv(manifest_csv)
            n_rows = len(df_manifest)
            if n_rows >= 15000:
                log.info(
                    "[6c] train_aug already exists with %d rows in manifest — skipping",
                    n_rows,
                )
                return {
                    "status": "✅",
                    "skipped": True,
                    "reason": "train_aug already exists with {} rows".format(n_rows),
                }
        except Exception as e:
            log.warning("[6c] Could not read manifest for idempotence check: %s", e)

    # Run create_augmented_train.py as subprocess
    script_path = Path(__file__).resolve().parent / "create_augmented_train.py"
    if not script_path.exists():
        log.error("[6c] Script not found: %s", script_path)
        return {
            "status": "❌",
            "error": "create_augmented_train.py not found at {}".format(script_path),
        }

    log.info("[6c] Running %s ...", script_path)
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        log.error("[6c] create_augmented_train.py timed out after 3600s")
        return {"status": "❌", "error": "timeout after 3600s"}
    except Exception as e:
        log.error("[6c] Failed to run create_augmented_train.py: %s", e)
        return {"status": "❌", "error": str(e)}

    if proc.returncode != 0:
        log.error(
            "[6c] create_augmented_train.py failed (rc=%d): %s",
            proc.returncode,
            proc.stderr[-2000:],
        )
        return {"status": "❌", "error": proc.stderr[-500:]}

    # Log stdout (last 30 lines)
    for line in proc.stdout.strip().splitlines()[-30:]:
        log.info("[6c]   %s", line)

    # Read results
    n_patches = 0
    ratio = 0.0
    try:
        report_json = patches_dir / "train_aug_report.json"
        if report_json.exists():
            with open(report_json) as f:
                aug_report = json.load(f)
            n_patches = aug_report.get("total_files", 0)
            ratio = aug_report.get("final_ratio", 0.0)
    except Exception as e:
        log.warning("[6c] Could not read train_aug_report.json: %s", e)

    log.info("[6c] train_aug created: %d patches, ratio=%.2f:1", n_patches, ratio)
    return {"status": "✅", "train_aug_patches": n_patches, "ratio": round(ratio, 2)}


# ══════════════════════════════════════════════════════════════════════════════
#  Paso 6d — Dataset audit (calls audit_dataset.py)
# ══════════════════════════════════════════════════════════════════════════════


def _paso6d_audit(patches_dir, dry_run=False):
    # type: (Path, bool) -> dict
    """Run comprehensive dataset audit (paso 6d).

    Idempotent: the audit is read-only (except for writing audit_report.json).
    Re-running always produces a fresh report.
    """
    log.info("[6d] ── Dataset audit ──")

    if dry_run:
        log.info("[6d] DRY-RUN: would run audit on %s", patches_dir)
        return {"status": "⏭️", "dry_run": True}

    # Run audit_dataset.py as subprocess
    script_path = Path(__file__).resolve().parent / "audit_dataset.py"
    if not script_path.exists():
        log.error("[6d] Script not found: %s", script_path)
        return {
            "status": "❌",
            "error": "audit_dataset.py not found at {}".format(script_path),
        }

    log.info("[6d] Running %s ...", script_path)
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        log.error("[6d] audit_dataset.py timed out after 600s")
        return {"status": "❌", "error": "timeout after 600s"}
    except Exception as e:
        log.error("[6d] Failed to run audit_dataset.py: %s", e)
        return {"status": "❌", "error": str(e)}

    if proc.returncode != 0:
        log.error(
            "[6d] audit_dataset.py failed (rc=%d): %s",
            proc.returncode,
            proc.stderr[-2000:],
        )
        return {"status": "❌", "error": proc.stderr[-500:]}

    # Log stdout (last 20 lines)
    for line in proc.stdout.strip().splitlines()[-20:]:
        log.info("[6d]   %s", line)

    # Read audit report
    audit_report_path = patches_dir / "audit_report.json"
    overall_pass = False
    try:
        if audit_report_path.exists():
            with open(audit_report_path) as f:
                audit_data = json.load(f)
            overall_pass = bool(audit_data.get("overall_pass", False))
        else:
            log.warning("[6d] audit_report.json not found after audit run")
    except Exception as e:
        log.warning("[6d] Could not read audit_report.json: %s", e)

    status = "✅" if overall_pass else "⚠️"
    log.info("[6d] Audit %s — overall_pass=%s", status, overall_pass)
    return {
        "status": status,
        "overall_pass": overall_pass,
        "audit_report_path": str(audit_report_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Orquestador
# ══════════════════════════════════════════════════════════════════════════════


def run_pre_embeddings(
    datasets_dir,
    active,
    workers=6,
    neg_ratio=10,
    max_neg=None,
    luna_subsets=None,
    dry_run=False,
    skip_zerocentering=False,
    skip_augmentation=False,
    skip_audit=False,
):
    # type: (Path, set, int, int, int|None, list|None, bool, bool, bool, bool) -> dict
    """Ejecuta preparación de datos 3D para los datasets activos."""
    results = {}

    if "luna_ct" in active or "luna_meta" in active:
        try:
            results["luna"] = run_luna_patches(
                datasets_dir,
                workers=workers,
                neg_ratio=neg_ratio,
                max_neg=max_neg,
                luna_subsets=luna_subsets,
                dry_run=dry_run,
            )
        except Exception as e:
            log.error("[LUNA] Error en extracción de parches: %s", e)
            results["luna"] = {"status": "error", "error": str(e)}

    if "pancreas" in active:
        try:
            results["pancreas"] = validar_preprocesado_pancreas(datasets_dir)
        except Exception as e:
            log.error("[PANCREAS] Error en validación: %s", e)
            results["pancreas"] = {"status": "error", "error": str(e)}

    # ── Sub-steps 6b, 6c, 6d (LUNA16 post-processing) ────────────────────
    patches_dir = datasets_dir / "luna_lung_cancer" / "patches"

    # ── 6b: Zero-centering fix ────────────────────────────────────────────
    if "luna_ct" in active and not skip_zerocentering:
        results["paso_6b_zerocentering"] = _paso6b_fix_zerocentering(
            patches_dir, dry_run
        )
    else:
        results["paso_6b_zerocentering"] = {"status": "⏭️", "skipped": True}

    # ── 6c: Create train_aug ──────────────────────────────────────────────
    if "luna_ct" in active and not skip_augmentation:
        results["paso_6c_train_aug"] = _paso6c_create_aug(patches_dir, dry_run)
    else:
        results["paso_6c_train_aug"] = {"status": "⏭️", "skipped": True}

    # ── 6d: Audit ─────────────────────────────────────────────────────────
    if "luna_ct" in active and not skip_audit:
        results["paso_6d_audit"] = _paso6d_audit(patches_dir, dry_run)
    else:
        results["paso_6d_audit"] = {"status": "⏭️", "skipped": True}

    return results

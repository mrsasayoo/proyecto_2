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
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

log = logging.getLogger("fase0.pre_embeddings")

# ── Constantes LUNA ──────────────────────────────────────────────────────────
HU_LUNG_CLIP = (-1000, 400)
PATCH_SIZE = 64
RANDOM_SEED = 42


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


def extract_patch(mhd_path, coord_world, patch_size=PATCH_SIZE,
                  clip_hu=HU_LUNG_CLIP):
    """Extrae parche 64³ centrado en coord_world."""
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
            ((0, patch_size - patch.shape[0]),
             (0, patch_size - patch.shape[1]),
             (0, patch_size - patch.shape[2])),
            constant_values=clip_hu[0],
        )
    patch = np.clip(patch, clip_hu[0], clip_hu[1])
    patch = (patch - clip_hu[0]) / (clip_hu[1] - clip_hu[0])
    return patch.astype(np.float32)


# ── Worker para ProcessPoolExecutor ──────────────────────────────────────────

def _worker(args_tuple):
    """
    Procesa un lote de candidatos de un mismo seriesuid.
    Retorna lista de (row_idx, out_path, label, status, mean_val).
    """
    mhd_path, candidates_batch, out_dir = args_tuple
    results = []
    try:
        image = sitk.ReadImage(str(mhd_path))
        origin = np.array(image.GetOrigin())
        spacing = np.array(image.GetSpacing())
        direc = np.array(image.GetDirection())
        array = sitk.GetArrayFromImage(image).astype(np.float32)
    except Exception as e:
        for row_idx, _cx, _cy, _cz, label in candidates_batch:
            results.append((row_idx, None, label, "ERROR_READ:{}".format(e), 0.0))
        return results

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
            iz, iy, ix = world_to_voxel([cx, cy, cz], origin, spacing, direc)
            half = PATCH_SIZE // 2
            z1, z2 = max(0, iz - half), min(array.shape[0], iz + half)
            y1, y2 = max(0, iy - half), min(array.shape[1], iy + half)
            x1, x2 = max(0, ix - half), min(array.shape[2], ix + half)
            patch = array[z1:z2, y1:y2, x1:x2]
            if patch.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
                patch = np.pad(
                    patch,
                    ((0, PATCH_SIZE - patch.shape[0]),
                     (0, PATCH_SIZE - patch.shape[1]),
                     (0, PATCH_SIZE - patch.shape[2])),
                    constant_values=HU_LUNG_CLIP[0],
                )
            patch = np.clip(patch, HU_LUNG_CLIP[0], HU_LUNG_CLIP[1])
            patch = (patch - HU_LUNG_CLIP[0]) / (HU_LUNG_CLIP[1] - HU_LUNG_CLIP[0])
            patch = patch.astype(np.float32)
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
    log.info("[NegSampling/%s] pos=%d | neg=%d→%d (ratio %d:1) | total=%d",
             split_name, len(df_pos), int((df_split["class"] == 0).sum()),
             len(df_neg), neg_ratio, len(df_sampled))
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

def run_luna_patches(datasets_dir, workers=6, neg_ratio=10, max_neg=None,
                     luna_subsets=None, dry_run=False):
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

    # Descubrir subsets
    available = sorted(
        d for d in ct_dir.iterdir()
        if d.is_dir() and d.name.startswith("subset")
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
            uid: path for uid, path in mhd_map.items()
            if any(tn in str(path) for tn in target_names)
        }
        log.info("[LUNA] Filtrado a subsets %s: %d CTs", luna_subsets, len(mhd_map))

    log.info("[LUNA] Total: %d CTs en %d subsets", len(mhd_map), len(available))

    # Cargar CSV de candidatos
    if not csv_path.exists():
        return {"status": "error", "reason": "candidates_V2.csv no encontrado"}

    df = pd.read_csv(csv_path)
    df_sub = df[df["seriesuid"].isin(mhd_map)].copy()
    df_sub = df_sub.reset_index(drop=False).rename(columns={"index": "original_idx"})

    log.info("[LUNA] %d candidatos en subsets disponibles (pos=%d, neg=%d)",
             len(df_sub), (df_sub["class"] == 1).sum(), (df_sub["class"] == 0).sum())

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
        log.info("[LUNA DRY-RUN] CTs=%d | candidatos=%d | ~%.2f GB",
                 len(mhd_map), len(df_sub), est_gb)
        return {"status": "Info", "dry_run": True, "candidates": len(df_sub)}

    # Cargar splits
    if splits_file.exists():
        with open(splits_file) as f:
            splits_data = json.load(f)
        train_uids = set(splits_data["train_uids"])
        val_uids = set(splits_data["val_uids"])
        test_uids = set(splits_data["test_uids"])
        log.info("[LUNA] luna_splits.json: train=%d val=%d test=%d",
                 len(train_uids), len(val_uids), len(test_uids))
    else:
        log.warning("[LUNA] luna_splits.json no encontrado — usando split 80/10/10 ad-hoc")
        rng = np.random.default_rng(RANDOM_SEED)
        all_uids = np.array(list(mhd_map.keys()))
        rng.shuffle(all_uids)
        n_test = int(0.10 * len(all_uids))
        n_val = int(0.10 * len(all_uids))
        test_uids = set(all_uids[:n_test])
        val_uids = set(all_uids[n_test:n_test + n_val])
        train_uids = set(all_uids[n_test + n_val:])

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
            df_split = apply_neg_sampling(df_split, split_name.upper(), neg_ratio,
                                          seed=seed_offset)

        out_dir = patches_dir / split_name

        # Agrupar por seriesuid
        tasks = []
        for uid, group in df_split.groupby("seriesuid"):
            if uid not in mhd_map:
                continue
            batch = [
                (int(row["original_idx"]), row["coordX"], row["coordY"],
                 row["coordZ"], int(row["class"]))
                for _, row in group.iterrows()
            ]
            tasks.append((mhd_map[uid], batch, str(out_dir)))

        total_cands = len(df_split)
        log.info("[LUNA/%s] Extrayendo %d parches de %d CTs con %d workers...",
                 split_name.upper(), total_cands, len(tasks), workers)

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
                        log.info("  [%s] %d/%d (%.1f%%) | %.0f cand/s | OK=%d SKIP=%d ERR=%d",
                                 split_name.upper(), done, total_cands,
                                 100 * done / total_cands, rate, ok, skip, errs)

        elapsed = time.time() - t0
        log.info("[LUNA/%s] Completado en %.0fs | OK=%d SKIP=%d ERR=%d",
                 split_name.upper(), elapsed, ok, skip, errs)

        if pos_means:
            mean_pos = np.mean(pos_means)
            if mean_pos < 0.1:
                log.error("[LUNA/%s] Media positivos=%.4f < 0.1 — posible error w2v",
                          split_name.upper(), mean_pos)

        n_patches = len(list(out_dir.glob("candidate_*.npy")))
        report_data[split_name] = {
            "patches": n_patches,
            "pos": int((df_split["class"] == 1).sum()),
            "neg": int((df_split["class"] == 0).sum()),
        }

    # Validación
    log.info("[LUNA] Validando muestra de parches...")
    ok_all = True
    for sp in ["train", "val", "test"]:
        sp_dir = patches_dir / sp
        if any(sp_dir.glob("candidate_*.npy")):
            ok_sp = validate_patches(sp_dir, n_sample=10)
            ok_all = ok_all and ok_sp

    # Reporte
    report = {
        "cts_total": len(mhd_files),
        "subsets": [d.name for d in available],
        "splits": report_data,
        "neg_ratio": neg_ratio,
        "max_neg": max_neg,
        "patch_size": PATCH_SIZE,
        "hu_clip": list(HU_LUNG_CLIP),
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

    final_zoom = (np.array(TARGET_SHAPE, dtype=np.float64)
                  / np.array(vol_iso.shape, dtype=np.float64))
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
    log.info("[PANCREAS] Validando preprocesado en %d de %d volúmenes...",
             len(sample), len(nii_files))

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

            ok = (vol.shape == (64, 64, 64)
                  and vol.min() >= -0.01 and vol.max() <= 1.01
                  and not entry["has_nan"] and not entry["has_inf"])
            entry["ok"] = ok
            if ok:
                ok_count += 1
            log.info("  %s: shape=%s range=[%.3f,%.3f] std=%.3f %s",
                     nii_path.name, vol.shape, vol.min(), vol.max(),
                     vol.std(), "✓" if ok else "✗")
        except Exception as e:
            entry["ok"] = False
            entry["error"] = str(e)
            log.error("  %s: ERROR %s", nii_path.name, e)
        results_list.append(entry)

    # Criterio: ≥5 de 10 con std > 0.01 (no vacíos/constantes)
    n_nontrivial = sum(1 for r in results_list
                       if r.get("ok") and r.get("std", 0) > 0.01)

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
    log.info("[PANCREAS] Validación: %d/%d OK | %d no triviales | %s",
             ok_count, len(sample), n_nontrivial,
             "✅ PASS" if all_ok else "⚠️ REVISAR")
    return {"status": "✅" if all_ok else "⚠️", **report}


# ══════════════════════════════════════════════════════════════════════════════
#  Orquestador
# ══════════════════════════════════════════════════════════════════════════════

def run_pre_embeddings(datasets_dir, active, workers=6, neg_ratio=10,
                       max_neg=None, luna_subsets=None, dry_run=False):
    # type: (Path, set, int, int, int|None, list|None, bool) -> dict
    """Ejecuta preparación de datos 3D para los datasets activos."""
    results = {}

    if "luna_ct" in active or "luna" in active:
        try:
            results["luna"] = run_luna_patches(
                datasets_dir, workers=workers, neg_ratio=neg_ratio,
                max_neg=max_neg, luna_subsets=luna_subsets, dry_run=dry_run,
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

    return results

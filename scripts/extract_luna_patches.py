"""
extract_luna_patches.py
=======================
Extrae parches 3D de LUNA16 (todos los subsets disponibles) y los guarda como .npy para FASE 0.

Reutiliza LUNA16PatchExtractor de fase0_extract_embeddings.py sin modificarlo.
Los parches se guardan como candidate_XXXXXX.npy (índice de fila en candidates_V2.csv).

Uso:
    python3 scripts/extract_luna_patches.py [--workers N] [--dry-run]

Salida:
    datasets/luna_lung_cancer/patches/train/  ← parches de train (80%)
    datasets/luna_lung_cancer/patches/val/    ← parches de val   (20%)
    datasets/luna_lung_cancer/patches/extraction_report.json

Después de ejecutar este script, relanzar FASE 0 con:
    --luna_patches datasets/luna_lung_cancer/patches/train
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

# ── Crear directorio de logs ────────────────────────────────────────────────
(Path(__file__).resolve().parent.parent / "logs").mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            str(Path(__file__).resolve().parent.parent / "logs" / "extract_luna_patches.log"),
            mode="w"
        ),
    ]
)
log = logging.getLogger("luna_extract")

# ── Constantes ────────────────────────────────────────────────────────────────
HU_LUNG_CLIP   = (-1000, 400)
PATCH_SIZE     = 64
TRAIN_RATIO    = 0.80
RANDOM_SEED    = 42


# ── Lógica de extracción (copiada fiel de LUNA16PatchExtractor en fase0) ─────
# Se duplica aquí para que el worker de ProcessPoolExecutor no tenga que
# importar fase0_extract_embeddings.py completo (evita cargar timm/torch/etc.)

def world_to_voxel(coord_world, origin, spacing, direction):
    """Conversión world→vóxel correcta (H3 de fase0)."""
    coord_shifted = np.array(coord_world) - np.array(origin)
    coord_voxel   = np.linalg.solve(
        np.array(direction).reshape(3, 3) * np.array(spacing),
        coord_shifted
    )
    return np.round(coord_voxel[::-1]).astype(int)  # [iz, iy, ix]


def extract_patch(mhd_path, coord_world, patch_size=PATCH_SIZE,
                  clip_hu=HU_LUNG_CLIP):
    """Extrae parche 64³ centrado en coord_world. Idéntico a LUNA16PatchExtractor.extract()."""
    image   = sitk.ReadImage(str(mhd_path))
    origin  = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direc   = np.array(image.GetDirection())
    array   = sitk.GetArrayFromImage(image).astype(np.float32)  # [Z,Y,X] HU

    iz, iy, ix = world_to_voxel(coord_world, origin, spacing, direc)

    half = patch_size // 2
    z1 = max(0, iz - half); z2 = min(array.shape[0], iz + half)
    y1 = max(0, iy - half); y2 = min(array.shape[1], iy + half)
    x1 = max(0, ix - half); x2 = min(array.shape[2], ix + half)

    patch = array[z1:z2, y1:y2, x1:x2]

    # Padding con -1000 HU (aire) si el candidato está en el borde
    if patch.shape != (patch_size, patch_size, patch_size):
        patch = np.pad(
            patch,
            ((0, patch_size - patch.shape[0]),
             (0, patch_size - patch.shape[1]),
             (0, patch_size - patch.shape[2])),
            constant_values=clip_hu[0]
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
        # Leer el CT solo una vez por seriesuid (costoso: 300-800 MB)
        image   = sitk.ReadImage(str(mhd_path))
        origin  = np.array(image.GetOrigin())
        spacing = np.array(image.GetSpacing())
        direc   = np.array(image.GetDirection())
        array   = sitk.GetArrayFromImage(image).astype(np.float32)
    except Exception as e:
        for row_idx, _cx, _cy, _cz, label in candidates_batch:
            results.append((row_idx, None, label, f"ERROR_READ:{e}", 0.0))
        return results

    for row_idx, cx, cy, cz, label in candidates_batch:
        out_path = Path(out_dir) / f"candidate_{row_idx:06d}.npy"
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
            z1 = max(0, iz - half); z2 = min(array.shape[0], iz + half)
            y1 = max(0, iy - half); y2 = min(array.shape[1], iy + half)
            x1 = max(0, ix - half); x2 = min(array.shape[2], ix + half)
            patch = array[z1:z2, y1:y2, x1:x2]
            if patch.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
                patch = np.pad(
                    patch,
                    ((0, PATCH_SIZE - patch.shape[0]),
                     (0, PATCH_SIZE - patch.shape[1]),
                     (0, PATCH_SIZE - patch.shape[2])),
                    constant_values=HU_LUNG_CLIP[0]
                )
            patch = np.clip(patch, HU_LUNG_CLIP[0], HU_LUNG_CLIP[1])
            patch = (patch - HU_LUNG_CLIP[0]) / (HU_LUNG_CLIP[1] - HU_LUNG_CLIP[0])
            patch = patch.astype(np.float32)
            np.save(out_path, patch)
            results.append((row_idx, str(out_path), label, "OK", float(patch.mean())))
        except Exception as e:
            results.append((row_idx, None, label, f"ERROR:{e}", 0.0))

    return results


# ── Validación post-extracción ────────────────────────────────────────────────

def validate_patches(patches_dir, n_sample=20):
    """
    Replica LUNA16PatchExtractor.validate_extraction():
    comprueba que los parches positivos tienen intensidad media > 0.1.
    Un parche de puro aire tendría intensidad ~0.0 → error de conversión.
    """
    patches = list(Path(patches_dir).glob("candidate_*.npy"))
    if not patches:
        log.error(f"[Validate] Sin parches en {patches_dir}")
        return False

    import random
    random.seed(42)
    sample = random.sample(patches, min(n_sample, len(patches)))
    errors = 0
    for p in sample:
        arr = np.load(p)
        if arr.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
            log.error(f"[Validate] {p.name}: shape {arr.shape} ≠ ({PATCH_SIZE}³)")
            errors += 1
        if arr.max() == arr.min():
            log.warning(f"[Validate] {p.name}: parche constante (val={arr.max():.3f}) — "
                        f"posible error de extracción")
        mean_v = arr.mean()
        status = "✓" if mean_v > 0.05 else "⚠ POSIBLE ERROR (casi todo aire)"
        log.info(f"  {p.name}  mean={mean_v:.3f}  {status}")

    log.info(f"[Validate] {len(sample)} parches revisados | errores de shape: {errors}")
    return errors == 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    LUNA_DIR    = repo_root / "datasets" / "luna_lung_cancer"
    CT_VOLUMES_DIR = LUNA_DIR  / "ct_volumes"
    CSV_PATH    = LUNA_DIR  / "candidates_V2" / "candidates_V2.csv"
    PATCHES_DIR = LUNA_DIR  / "patches"
    if args.output_dir is not None:
        PATCHES_DIR = Path(args.output_dir)
    TRAIN_DIR = PATCHES_DIR / "train"
    VAL_DIR   = PATCHES_DIR / "val"
    LOG_DIR     = repo_root / "logs"

    LOG_DIR.mkdir(exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    import shutil as _shutil
    _free_gb = _shutil.disk_usage(str(PATCHES_DIR)).free / 1e9
    log.info(f"[Setup] Espacio libre en disco: {_free_gb:.1f} GB")
    if args.max_neg is None and _free_gb < 100:
        log.warning(
            f"[Setup] Solo {_free_gb:.1f} GB libres y --max_neg no especificado. "
            f"Extraer todos los parches requiere ~788 GB con 10 subsets. "
            f"Considera usar: --max_neg 8000 (requiere ~9.3 GB). "
            f"Continuando de todas formas — el script fallará si se llena el disco."
        )
    log.info("LUNA16 — Extracción de parches 3D (todos los subsets)")
    log.info("=" * 65)

    # ── 1. Descubrir todos los subsets disponibles ───────────────────────────────────────
    available_subsets = sorted([
        d for d in CT_VOLUMES_DIR.iterdir()
        if d.is_dir() and d.name.startswith("subset")
    ])

    if not available_subsets:
        log.error(
            f"[Setup] No se encontró ningún directorio subsetN en {CT_VOLUMES_DIR}. "
            f"Verifica que los CTs estén extraídos."
        )
        sys.exit(1)

    mhd_files = []
    for subset_dir in available_subsets:
        found = list(subset_dir.rglob("*.mhd"))
        log.info(f"[Setup] {subset_dir.name}: {len(found)} CTs (.mhd)")
        mhd_files.extend(found)

    if not mhd_files:
        log.error(
            f"[Setup] Ningún archivo .mhd encontrado en los subsets de {CT_VOLUMES_DIR}. "
            f"Verifica que los ZIPs estén extraídos."
        )
        sys.exit(1)

    log.info(f"[Setup] Total: {len(mhd_files)} CTs (.mhd) en "
             f"{len(available_subsets)} subsets: "
             f"{[d.name for d in available_subsets]}")

    # Mapa seriesuid → mhd_path
    mhd_map = {p.stem: p for p in mhd_files}

    if args.single_subset is not None:
        _target = f"subset{args.single_subset}"
        _before = len(mhd_map)
        mhd_map = {
            uid: path for uid, path in mhd_map.items()
            if f"{_target}{os.sep}" in str(path) or f"/{_target}/" in str(path)
        }
        log.info(
            f"[Setup] --single_subset={args.single_subset}: filtrando a CTs de "
            f"'{_target}' \u2014 {len(mhd_map)}/{_before} CTs disponibles."
        )
        if not mhd_map:
            log.error(
                f"[Setup] Ning\u00fan CT encontrado para subset{args.single_subset}. "
                f"Verifica que el directorio "
                f"datasets/luna_lung_cancer/ct_volumes/subset{args.single_subset}/ "
                f"existe y contiene archivos .mhd."
            )
            sys.exit(1)

    # ── 2. Cargar candidates_V2.csv ───────────────────────────────────────────
    if not CSV_PATH.exists():
        log.error(f"[Setup] candidates_V2.csv no encontrado en {CSV_PATH}")
        sys.exit(1)

    log.info(f"[Setup] Cargando {CSV_PATH.name} ...")
    df = pd.read_csv(CSV_PATH)
    log.info(f"[Setup] {len(df):,} candidatos totales en el CSV")
    log.info(f"[Setup] Columnas: {list(df.columns)}")

    # Filtrar candidatos cuyos seriesuid están en los subsets disponibles
    df_sub = df[df["seriesuid"].isin(mhd_map)].copy()
    df_sub = df_sub.reset_index(drop=False)  # conserva el índice original como columna
    df_sub = df_sub.rename(columns={"index": "original_idx"})

    n_pos = (df_sub["class"] == 1).sum()
    n_neg = (df_sub["class"] == 0).sum()
    log.info(f"[Setup] Candidatos en subsets disponibles: {len(df_sub):,} "
             f"(pos={n_pos:,}, neg={n_neg:,}, ratio={n_neg//max(n_pos,1)}:1)")

    if args.max_neg is not None:
        df_pos = df_sub[df_sub["class"] == 1]
        df_neg = df_sub[df_sub["class"] == 0]
        rng_neg = np.random.default_rng(RANDOM_SEED + 1)
        if len(df_neg) > args.max_neg:
            neg_idx = rng_neg.choice(len(df_neg), size=args.max_neg, replace=False)
            df_neg = df_neg.iloc[neg_idx].reset_index(drop=True)
        df_sub = pd.concat([df_pos, df_neg], ignore_index=True)
        n_pos = (df_sub["class"] == 1).sum()
        n_neg = (df_sub["class"] == 0).sum()
        log.info(
            f"[Setup] --max_neg aplicado: {n_pos:,} positivos (todos) + "
            f"{n_neg:,} negativos muestreados = {len(df_sub):,} candidatos totales. "
            f"Espacio estimado: {len(df_sub) * 64**3 * 4 / 1e9:.2f} GB."
        )

    if args.dry_run:
        log.info("[DRY-RUN] Modo simulación — no se escriben archivos.")
        log.info(f"  Subsets detectados: {[d.name for d in available_subsets]}")
        log.info(f"  CTs a procesar    : {len(mhd_map)}")
        log.info(f"  Candidatos totales: {len(df_sub):,}")
        if args.neg_ratio > 0:
            n_pos_total = (df_sub["class"] == 1).sum()
            n_neg_sampled = n_pos_total * args.neg_ratio
            n_after_sampling = n_pos_total + min(n_neg_sampled,
                                                 (df_sub["class"] == 0).sum())
            log.info(f"  Tras negative sampling ({args.neg_ratio}:1): "
                     f"~{n_after_sampling:,} parches "
                     f"(~{n_after_sampling * 64**3 * 4 / 1e9:.2f} GB)")
        log.info(f"  Positivos (class=1): {n_pos:,}")
        log.info(f"  Negativos (class=0): {n_neg:,}")
        log.info(f"  Parches a generar (aprox): {len(df_sub):,}")
        _est_gb = len(df_sub) * 64**3 * 4 / 1e9
        _est_mb = len(df_sub) * 64**3 * 4 / 1e6
        log.info(f"  Espacio estimado: {_est_gb:.2f} GB ({_est_mb:,.0f} MB)")
        log.info(f"  Espacio libre actual: {_shutil.disk_usage(str(PATCHES_DIR)).free / 1e9:.1f} GB")
        _fits = _est_gb < _shutil.disk_usage(str(PATCHES_DIR)).free / 1e9
        log.info(f"  Cabe en disco: {'✓ SÍ' if _fits else '✗ NO — usar --max_neg para reducir'}")
        log.info("  → Ejecuta sin --dry-run para extraer.")
        return

    # ── 3. Split train/val por seriesuid (sin leakage) ───────────────────────
    rng       = np.random.default_rng(RANDOM_SEED)
    all_uids  = np.array(list(mhd_map.keys()))
    rng.shuffle(all_uids)
    n_train   = int(len(all_uids) * TRAIN_RATIO)
    train_uid = set(all_uids[:n_train])
    val_uid   = set(all_uids[n_train:])

    _subset_tag = f" [subset{args.single_subset}]" if args.single_subset is not None else ""
    log.info(f"[Split]{_subset_tag} Train: {len(train_uid)} CTs | Val: {len(val_uid)} CTs "
             f"(split por seriesuid — sin leakage)")

    df_sub["split"] = df_sub["seriesuid"].apply(
        lambda s: "train" if s in train_uid else "val"
    )
    df_train = df_sub[df_sub["split"] == "train"]
    df_val   = df_sub[df_sub["split"] == "val"]
    log.info(f"[Split] Candidatos train: {len(df_train):,} | val: {len(df_val):,}")

    # ── Negative sampling ───────────────────────────────────────────────────────
    if args.neg_ratio > 0:
        def apply_neg_sampling(df_split, split_name, neg_ratio, seed=42):
            n_pos = (df_split["class"] == 1).sum()
            n_neg_keep = n_pos * neg_ratio
            df_pos = df_split[df_split["class"] == 1]
            df_neg = df_split[df_split["class"] == 0]
            if len(df_neg) > n_neg_keep:
                df_neg = df_neg.sample(n=n_neg_keep, random_state=seed)
            df_sampled = pd.concat([df_pos, df_neg]).reset_index(drop=True)
            log.info(
                f"[NegSampling/{split_name}] "
                f"pos={len(df_pos):,} | neg original={int((df_split['class']==0).sum()):,} "
                f"→ neg_keep={len(df_neg):,} (ratio {neg_ratio}:1) | "
                f"total={len(df_sampled):,} | "
                f"espacio estimado: {len(df_sampled) * 64**3 * 4 / 1e9:.2f} GB"
            )
            return df_sampled

        df_train = apply_neg_sampling(df_train, "TRAIN", args.neg_ratio, seed=42)
        df_val   = apply_neg_sampling(df_val,   "VAL",   args.neg_ratio, seed=43)
    else:
        log.warning(
            "[NegSampling] --neg_ratio=0: sin sampling. "
            f"Se escribirán {len(df_train)+len(df_val):,} parches "
            f"(~{(len(df_train)+len(df_val)) * 64**3 * 4 / 1e9:.1f} GB). "
            "Asegúrate de tener suficiente espacio en disco."
        )

    # ── 4. Extraer parches en paralelo (un worker por CT) ────────────────────
    def build_tasks(df_split, out_dir):
        """Agrupa candidatos por seriesuid → un task por CT."""
        tasks = []
        for uid, group in df_split.groupby("seriesuid"):
            batch = [
                (int(row["original_idx"]), row["coordX"], row["coordY"],
                 row["coordZ"], int(row["class"]))
                for _, row in group.iterrows()
            ]
            tasks.append((mhd_map[uid], batch, str(out_dir)))
        return tasks

    for split_name, df_split, out_dir in [
        ("TRAIN", df_train, TRAIN_DIR),
        ("VAL",   df_val,   VAL_DIR),
    ]:
        tasks = build_tasks(df_split, out_dir)
        total_cands = len(df_split)
        log.info(f"\n[{split_name}] Extrayendo {total_cands:,} parches de "
                 f"{len(tasks)} CTs con {args.workers} workers...")

        t0    = time.time()
        done  = 0
        ok    = 0
        skip  = 0
        errs  = 0
        pos_means = []

        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futures = {exe.submit(_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    batch_results = fut.result()
                except Exception as e:
                    log.error(f"[{split_name}] Worker falló: {e}")
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
                        log.warning(f"  [{split_name}] candidate_{row_idx:06d}: {status}")

                    if done % 5000 == 0 or done == total_cands:
                        elapsed = time.time() - t0
                        rate    = done / max(elapsed, 1)
                        eta     = (total_cands - done) / max(rate, 1)
                        log.info(f"  [{split_name}] {done:>7,}/{total_cands:,} "
                                 f"({100*done/total_cands:.1f}%) | "
                                 f"{rate:.0f} cand/s | ETA {eta/60:.1f} min | "
                                 f"OK={ok} SKIP={skip} ERR={errs}")

        elapsed = time.time() - t0
        log.info(f"[{split_name}] Completado en {elapsed:.0f}s "
                 f"({elapsed/60:.1f} min) | OK={ok} SKIP={skip} ERR={errs}")

        # Validar intensidad media de positivos (debe ser > 0.1 si la conversión es correcta)
        if pos_means:
            mean_pos = np.mean(pos_means)
            if mean_pos < 0.1:
                log.error(f"[{split_name}] Intensidad media de positivos = {mean_pos:.4f} < 0.1. "
                          f"Posible error en conversión world→vóxel. "
                          f"Verifica que los .mhd y .zraw están en la misma carpeta.")
            else:
                log.info(f"[{split_name}] Intensidad media parches positivos: "
                         f"{mean_pos:.4f} (esperado > 0.1 para tejido/nódulo) ✓")

    # ── 5. Validación de muestra ──────────────────────────────────────────────
    log.info("\n[Validate] Verificando muestra de parches extraídos...")
    ok_train = validate_patches(TRAIN_DIR, n_sample=20)
    ok_val   = validate_patches(VAL_DIR,   n_sample=10)

    # ── 6. Reporte final ──────────────────────────────────────────────────────
    n_train_files = len(list(TRAIN_DIR.glob("candidate_*.npy")))
    n_val_files   = len(list(VAL_DIR.glob("candidate_*.npy")))

    report = {
        "cts_total":   len(mhd_files),
        "subsets":     [d.name for d in available_subsets],
        "train_cts":        len(train_uid),
        "val_cts":          len(val_uid),
        "train_patches":    n_train_files,
        "val_patches":      n_val_files,
        "train_pos":        int((df_train["class"] == 1).sum()),
        "train_neg":        int((df_train["class"] == 0).sum()),
        "val_pos":          int((df_val["class"] == 1).sum()),
        "val_neg":          int((df_val["class"] == 0).sum()),
        "neg_ratio":        args.neg_ratio,
        "patch_size":       PATCH_SIZE,
        "hu_clip":          list(HU_LUNG_CLIP),
        "validation_ok":    ok_train and ok_val,
        "fase0_command": (
            f"python3 src/pipeline/fase0_extract_embeddings.py "
            f"--backbone vit_tiny_patch16_224 "
            f"--luna_patches {TRAIN_DIR} "
            f"[... resto de args ...]"
        ),
    }

    report_path = PATCHES_DIR / "extraction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("\n" + "=" * 65)
    log.info("EXTRACCIÓN COMPLETADA")
    log.info("=" * 65)
    log.info(f"  Train: {n_train_files:,} parches en {TRAIN_DIR}")
    log.info(f"  Val  : {n_val_files:,} parches en {VAL_DIR}")
    log.info(f"  Reporte: {report_path}")
    log.info(f"\n  Siguiente paso — relanzar FASE 0 con:")
    log.info(f"    --luna_patches {TRAIN_DIR}")
    log.info(f"\n  Comando completo para vit_tiny:")
    log.info(
        f"    python3 src/pipeline/fase0_extract_embeddings.py \\\n"
        f"        --backbone vit_tiny_patch16_224 --batch_size 256 --workers 8 \\\n"
        f"        --output_dir embeddings/vit_tiny \\\n"
        f"        --luna_patches {TRAIN_DIR} \\\n"
        f"        --luna_csv datasets/luna_lung_cancer/candidates_V2/candidates_V2.csv \\\n"
        f"        --chest_csv datasets/nih_chest_xrays/Data_Entry_2017.csv \\\n"
        f"        --chest_imgs datasets/nih_chest_xrays/all_images \\\n"
        f"        --chest_train_list datasets/nih_chest_xrays/train_val_list.txt \\\n"
        f"        --chest_val_list datasets/nih_chest_xrays/test_list.txt \\\n"
        f"        --chest_view_filter PA \\\n"
        f"        --chest_bbox_csv datasets/nih_chest_xrays/BBox_List_2017.csv \\\n"
        f"        --isic_gt datasets/isic_2019/ISIC_2019_Training_GroundTruth.csv \\\n"
        f"        --isic_imgs datasets/isic_2019/isic_images \\\n"
        f"        --isic_metadata datasets/isic_2019/ISIC_2019_Training_Metadata.csv \\\n"
        f"        --oa_root datasets/osteoarthritis/oa_splits \\\n"
        f"        --pancreas_nii_dir datasets/zenodo_13715870 \\\n"
        f"        --pancreas_labels_dir datasets/panorama_labels \\\n"
        f"        --pancreas_labels_commit bf1d6ba3230f6b093e7ea959a4bf5e2eba2e3665 \\\n"
        f"        --pancreas_roi_strategy A \\\n"
        f"        2>&1 | tee logs/fase0_vit_tiny_v3.log"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrae parches LUNA16 de todos los subsets → .npy para FASE 0"
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Workers paralelos (uno por CT). Default=6. "
             "Máximo recomendado con 32 GB RAM: 8"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directorio de salida para los parches train/ y val/. "
             "Default: datasets/luna_lung_cancer/patches/ dentro del repo. "
             "Usa una ruta en el NVMe si el HDD no tiene espacio suficiente."
    )
    parser.add_argument(
        "--neg_ratio", type=int, default=10,
        help="Número de negativos por positivo (negative sampling). "
             "Default=10. Con 1,555 positivos → ~15,550 negativos → "
             "~17,105 parches totales (~4.5 GB). "
             "Usa 0 para desactivar el sampling y conservar todos los negativos "
             "(requiere ~197 GB de disco libre)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solo muestra estadísticas, no extrae parches"
    )
    parser.add_argument(
        "--max_neg", type=int, default=None,
        help=(
            "Máximo número de candidatos negativos (class=0) a extraer en total. "
            "TODOS los positivos (class=1) se extraen siempre. "
            "Para FASE 0 (routing embeddings) se recomienda --max_neg 8000. "
            "Default=None extrae todos los negativos (788 GB para 10 subsets \u2014 "
            "solo usar si hay suficiente espacio en disco)."
        )
    )
    parser.add_argument(
        "--single_subset", type=int, default=None,
        help=(
            "Si se especifica, procesa solo los CTs del subsetN indicado "
            "(0-9). Los parches se acumulan en train/ y val/ junto con los "
            "de ejecuciones anteriores. Usar junto con --max_neg para controlar "
            "el espacio por subset. Ejemplo: --single_subset 3 --max_neg 800"
        )
    )
    main(parser.parse_args())

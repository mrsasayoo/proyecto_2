"""
extract_luna_patches.py
=======================
Extrae parches 3D de LUNA16 (subset0) y los guarda como .npy para FASE 0.

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

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/extract_luna_patches.log", mode="w"),
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
        for row_idx, _, _, label in candidates_batch:
            results.append((row_idx, None, label, f"ERROR_READ:{e}", 0.0))
        return results

    for row_idx, cx, cy, cz, label in candidates_batch:
        out_path = Path(out_dir) / f"candidate_{row_idx:06d}.npy"
        if out_path.exists():
            results.append((row_idx, str(out_path), label, "SKIPPED", -1.0))
            continue
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
    SUBSET0_DIR = LUNA_DIR  / "ct_volumes" / "subset0"
    CSV_PATH    = LUNA_DIR  / "candidates_V2" / "candidates_V2.csv"
    PATCHES_DIR = LUNA_DIR  / "patches"
    TRAIN_DIR   = PATCHES_DIR / "train"
    VAL_DIR     = PATCHES_DIR / "val"
    LOG_DIR     = repo_root / "logs"

    LOG_DIR.mkdir(exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("LUNA16 — Extracción de parches 3D (subset0)")
    log.info("=" * 65)

    # ── 1. Verificar que subset0 está extraído ────────────────────────────────
    mhd_files = list(SUBSET0_DIR.rglob("*.mhd"))
    if len(mhd_files) < 5:
        zip_path = LUNA_DIR / "ct_volumes" / "subset0.zip"
        if zip_path.exists():
            log.warning(f"[Setup] Solo {len(mhd_files)} .mhd en {SUBSET0_DIR}. "
                        f"Extrayendo subset0.zip (~6 GB, tardará ~5 min)...")
            import subprocess
            subprocess.run(
                ["unzip", "-n", str(zip_path), "-d", str(SUBSET0_DIR)],
                check=True
            )
            mhd_files = list(SUBSET0_DIR.rglob("*.mhd"))
        else:
            log.error(f"[Setup] subset0.zip no encontrado en {zip_path}. "
                      f"Descárgalo desde https://luna16.grand-challenge.org/")
            sys.exit(1)

    log.info(f"[Setup] {len(mhd_files)} CTs (.mhd) encontrados en subset0")

    # Mapa seriesuid → mhd_path
    mhd_map = {p.stem: p for p in mhd_files}

    # ── 2. Cargar candidates_V2.csv ───────────────────────────────────────────
    if not CSV_PATH.exists():
        log.error(f"[Setup] candidates_V2.csv no encontrado en {CSV_PATH}")
        sys.exit(1)

    log.info(f"[Setup] Cargando {CSV_PATH.name} ...")
    df = pd.read_csv(CSV_PATH)
    log.info(f"[Setup] {len(df):,} candidatos totales en el CSV")
    log.info(f"[Setup] Columnas: {list(df.columns)}")

    # Filtrar solo los candidatos cuyo seriesuid está en subset0
    df_sub = df[df["seriesuid"].isin(mhd_map)].copy()
    df_sub = df_sub.reset_index(drop=False)  # conserva el índice original como columna
    df_sub = df_sub.rename(columns={"index": "original_idx"})

    n_pos = (df_sub["class"] == 1).sum()
    n_neg = (df_sub["class"] == 0).sum()
    log.info(f"[Setup] Candidatos en subset0: {len(df_sub):,} "
             f"(pos={n_pos:,}, neg={n_neg:,}, ratio={n_neg//max(n_pos,1)}:1)")

    if args.dry_run:
        log.info("[DRY-RUN] Modo simulación — no se escriben archivos.")
        log.info(f"  CTs a procesar   : {len(mhd_map)}")
        log.info(f"  Candidatos totales: {len(df_sub):,}")
        log.info(f"  Positivos (class=1): {n_pos:,}")
        log.info(f"  Negativos (class=0): {n_neg:,}")
        log.info(f"  Parches a generar (aprox): {len(df_sub):,}")
        log.info(f"  Espacio estimado: {len(df_sub) * 64**3 * 4 / 1e9:.1f} GB")
        log.info("  → Ejecuta sin --dry-run para extraer.")
        return

    # ── 3. Split train/val por seriesuid (sin leakage) ───────────────────────
    rng       = np.random.default_rng(RANDOM_SEED)
    all_uids  = np.array(list(mhd_map.keys()))
    rng.shuffle(all_uids)
    n_train   = int(len(all_uids) * TRAIN_RATIO)
    train_uid = set(all_uids[:n_train])
    val_uid   = set(all_uids[n_train:])

    log.info(f"[Split] Train: {len(train_uid)} CTs | Val: {len(val_uid)} CTs "
             f"(split por seriesuid — sin leakage)")

    df_sub["split"] = df_sub["seriesuid"].apply(
        lambda s: "train" if s in train_uid else "val"
    )
    df_train = df_sub[df_sub["split"] == "train"]
    df_val   = df_sub[df_sub["split"] == "val"]
    log.info(f"[Split] Candidatos train: {len(df_train):,} | val: {len(df_val):,}")

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
        "cts_subset0":      len(mhd_files),
        "train_cts":        len(train_uid),
        "val_cts":          len(val_uid),
        "train_patches":    n_train_files,
        "val_patches":      n_val_files,
        "train_pos":        int((df_train["class"] == 1).sum()),
        "train_neg":        int((df_train["class"] == 0).sum()),
        "val_pos":          int((df_val["class"] == 1).sum()),
        "val_neg":          int((df_val["class"] == 0).sum()),
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
        f"        --chest_train_list datasets/nih_chest_xrays/splits/nih_train_list.txt \\\n"
        f"        --chest_val_list datasets/nih_chest_xrays/splits/nih_val_list.txt \\\n"
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
        description="Extrae parches LUNA16 de subset0 → .npy para FASE 0"
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Workers paralelos (uno por CT). Default=6. "
             "Máximo recomendado con 32 GB RAM: 8"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solo muestra estadísticas, no extrae parches"
    )
    main(parser.parse_args())

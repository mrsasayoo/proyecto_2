"""
FASE 0 — Extracción de CLS tokens (Embeddings)
===============================================
Proyecto MoE — Incorporar Elementos de IA, Unidad II

Arquitectura MoE (diseño propio — 6 expertos):
  - Experto 0 → NIH ChestXray14   (radiografía tórax, multi-label BCE)
  - Experto 1 → ISIC 2019         (dermatoscopía, multiclase CE)
  - Experto 2 → OA Rodilla        (radiografía rodilla, ordinal QWK)
  - Experto 3 → LUNA16            (CT pulmón 3D, parches, FocalLoss FROC)
  - Experto 4 → Pancreas PANORAMA (CT abdomen 3D, volumen, FocalLoss AUC)
  - Experto 5 → OOD / Error       (MLP sin dominio clínico — ver nota abajo)

  ⚠ EXPERTO 6 (ID=5, OOD/Error) NO genera embeddings en FASE 0.
    No existe un "dataset de imágenes OOD" que extraer. El Experto 6
    es un MLP simple que se define e inicializa en FASE 1, cuando se
    entrena el router. Su rol es recibir imágenes que el router no sabe
    clasificar (entropía alta del gating score H(g)) y devolver una
    señal de alerta. La pérdida L_error penaliza al router si delega
    imágenes médicas válidas (Expertos 0–4) al Experto 6.
    → y_train/y_val solo contendrán etiquetas 0–4 al finalizar FASE 0.

Requerimientos del proyecto (sección 4.1):
  - Backbone: ViT-Tiny preentrenado de timm (o CvT / Swin-Tiny)
  - Backbone CONGELADO — no se entrena, solo extrae
  - Pasar los 5 datasets de dominio → guardar CLS tokens en disco
  - Resultado: Z_train [N_train, 192], Z_val [N_val, 192]
  - Etiquetas de experto en disco: 0=Chest, 1=ISIC, 2=OA, 3=LUNA, 4=Pancreas
    (Experto 5=OOD no aparece — no tiene datos propios)
  - Hardware objetivo: clúster con 2 GPUs × 12 GB VRAM

Uso:
  python fase0_extract_embeddings.py --backbone vit_tiny_patch16_224 \\
                                     --batch_size 64 \\
                                     --output_dir ./embeddings

Estructura modular (refactorizado):
  src/pipeline/
    config.py          → constantes compartidas (EXPERT_IDS, BACKBONE_CONFIGS, HU clips, etc.)
    logging_utils.py   → setup_logging()
    losses.py          → FocalLossMultiLabel, OrdinalLoss, FocalLoss
    backbones.py       → load_frozen_backbone() + CvT-13 monkey-patch
    preprocessing.py   → build_2d_transform, apply_clahe, normalize_hu, resize_volume_3d, etc.
    datasets/
      __init__.py      → re-exports
      chest.py         → ChestXray14Dataset (Experto 0)
      isic.py          → ISICDataset (Experto 1)
      osteoarthritis.py→ OAKneeDataset (Experto 2)
      luna.py          → LUNA16Dataset, LUNA16PatchExtractor, LUNA16FROCEvaluator (Experto 3)
      pancreas.py      → PancreasDataset, PanoramaLabelLoader, PancreasROIExtractor (Experto 4)
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from pathlib import Path

# ── Imports del paquete modular ────────────────────────────────────────────
from config import (
    EXPERT_IDS, N_EXPERTS_DOMAIN, N_EXPERTS_TOTAL,
    BACKBONE_CONFIGS, HU_ABDOMEN_CLIP,
)
from logging_utils import setup_logging
from backbones import load_frozen_backbone
from preprocessing import build_2d_transform, resize_volume_3d, volume_to_vit_input
from datasets.chest import ChestXray14Dataset
from datasets.isic import ISICDataset
from datasets.osteoarthritis import OAKneeDataset
from datasets.luna import LUNA16Dataset
from datasets.pancreas import PancreasDataset, PanoramaLabelLoader

# Logger global — se inicializa en main()
log = None


# ──────────────────────────────────────────────────────────
# EXTRACCIÓN DE EMBEDDINGS
# ──────────────────────────────────────────────────────────

def extract_embeddings(model, dataloader, device, d_model, desc=""):
    """
    Pasa el dataloader completo por el backbone y acumula CLS tokens.
    Detecta activamente: NaN/Inf en embeddings, VRAM excesiva, batches vacíos.
    """
    n_total   = len(dataloader.dataset)
    Z         = np.zeros((n_total, d_model), dtype=np.float32)
    y_expert  = np.zeros(n_total, dtype=np.int64)
    img_names = []
    cursor    = 0

    nan_batches  = 0
    inf_batches  = 0
    t_start      = time.time()

    log.info(f"[Extract/{desc}] Iniciando extracción: {n_total:,} imágenes | "
             f"d_model={d_model} | batch_size={dataloader.batch_size}")

    for batch_idx, (imgs, experts, names) in enumerate(dataloader):
        imgs = imgs.to(device)
        z    = model(imgs)
        z_np = z.cpu().numpy()
        B    = z_np.shape[0]

        if np.isnan(z_np).any():
            nan_batches += 1
            log.error(f"[Extract/{desc}] NaN detectado en batch {batch_idx} "
                      f"(imágenes {cursor}–{cursor+B}).")
        if np.isinf(z_np).any():
            inf_batches += 1
            log.error(f"[Extract/{desc}] Inf detectado en batch {batch_idx} "
                      f"(imágenes {cursor}–{cursor+B}).")

        Z[cursor:cursor + B]        = z_np
        y_expert[cursor:cursor + B] = experts.numpy()
        img_names.extend(names)
        cursor += B

        if batch_idx % 50 == 0:
            elapsed  = time.time() - t_start
            speed    = cursor / elapsed if elapsed > 0 else 0
            eta_s    = (n_total - cursor) / speed if speed > 0 else 0
            log.info(f"[Extract/{desc}] {cursor:>6,}/{n_total:,} "
                     f"({100*cursor/n_total:.1f}%) | "
                     f"{speed:.0f} img/s | ETA {eta_s/60:.1f} min")

        if device == "cuda" and batch_idx % 200 == 0:
            used_gb  = torch.cuda.memory_allocated() / 1e9
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            pct      = 100 * used_gb / total_gb
            log.debug(f"[Extract/{desc}] VRAM: {used_gb:.2f}/{total_gb:.1f} GB ({pct:.1f}%)")
            if pct > 90:
                log.warning(f"[Extract/{desc}] VRAM al {pct:.1f}% — riesgo de OOM. "
                            f"Considera reducir --batch_size.")

    elapsed_total = time.time() - t_start
    log.info(f"[Extract/{desc}] Completado: {cursor:,} embeddings en {elapsed_total:.1f}s "
             f"({cursor/elapsed_total:.0f} img/s)")

    if nan_batches or inf_batches:
        log.error(f"[Extract/{desc}] RESUMEN ANOMALÍAS: "
                  f"NaN en {nan_batches} batches | Inf en {inf_batches} batches. "
                  f"Los embeddings resultantes no son confiables.")
    else:
        log.info(f"[Extract/{desc}] Sin NaN ni Inf detectados ✓")

    norms = np.linalg.norm(Z[:cursor], axis=1)
    log.debug(f"[Extract/{desc}] Norma L2 embeddings — "
              f"media: {norms.mean():.2f} | min: {norms.min():.2f} | max: {norms.max():.2f}")
    if norms.mean() < 1.0:
        log.warning(f"[Extract/{desc}] Norma L2 media muy baja ({norms.mean():.3f}). "
                    f"¿El backbone devuelve el CLS token correctamente?")

    return Z[:cursor], y_expert[:cursor], img_names


# ──────────────────────────────────────────────────────────
# GUARD CLAUSE: LUNA16 PATCHES
# ──────────────────────────────────────────────────────────

def _check_luna_patches(luna_patches_arg=None):
    """Verificar que los parches 3D de LUNA16 fueron pre-extraídos."""
    if luna_patches_arg is not None:
        _base = Path(luna_patches_arg)
        # Modo A: <ruta>/train/ y <ruta>/val/ bajo el directorio padre
        if (_base / "train").exists() or (_base / "val").exists():
            patches_base = _base
        else:
            # Modo B: la ruta IS el train dir, val está al lado
            patches_base = _base.parent
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent
        patches_base = repo_root / "datasets" / "luna_lung_cancer" / "patches"
    missing = []
    for split_name in ("train", "val"):
        split_dir = patches_base / split_name
        if not split_dir.exists():
            missing.append(f"  ✗ {split_dir}  (no existe)")
        elif not any(split_dir.glob("*.npy")):
            missing.append(f"  ✗ {split_dir}  (vacía — sin archivos .npy)")
    if missing:
        log.error(
            "LUNA16: parches 3D no encontrados.\n"
            + "\n".join(missing)
            + "\n\nEjecuta primero:\n"
            "  python scripts/extract_luna_patches.py\n"
            "antes de correr fase0_extract_embeddings.py."
        )
        sys.exit(1)
    log.info("[Guard] LUNA16 patches verificados ✓ (train/ y val/ contienen .npy)")


# ──────────────────────────────────────────────────────────
# FÁBRICA DE DATASETS
# ──────────────────────────────────────────────────────────

def build_datasets(cfg):
    """
    Construye los datasets de train, val y test para los 5 expertos de dominio.
    Devuelve (train_dataset, val_dataset, test_dataset) como ConcatDataset.

    Splits 80 / 10 / 10 — lógica por dataset:
      NIH  : Patient-ID level, rebalanceando test_list.txt al 10 %
      ISIC : Lesion-ID level, dividiendo el 20 % restante en val + test
      OA   : Carpetas predefinidas train/val/test (setup_datasets.py)
      LUNA : train/ intacto (separación por seriesuid), val/ dividida 50/50
             determinista → val + test
      Panc : 80/10/10 aleatorio sobre pares NIfTI válidos
    """
    MIN_SAMPLES_FOR_TEST = 30
    transform_2d = build_2d_transform(img_size=224)

    train_datasets = []
    val_datasets   = []
    test_datasets  = []

    # ── Experto 0: NIH ChestXray14 (80/10/10 por Patient ID) ──
    if cfg.get("chest_csv") and cfg.get("chest_imgs"):
        log.info("[Dataset] Cargando NIH ChestXray14 (Experto 0)...")

        if not (cfg.get("chest_train_list") and cfg.get("chest_val_list")):
            log.error("[Chest] Se requieren --chest_train_list (train_val_list.txt) y "
                      "--chest_val_list (test_list.txt) para computar split 80/10/10.")
        else:
            # Mapeo Image Index → Patient ID desde el CSV
            _df_csv = pd.read_csv(cfg["chest_csv"],
                                  usecols=["Image Index", "Patient ID"])
            _img_to_pid = dict(zip(_df_csv["Image Index"], _df_csv["Patient ID"]))

            with open(cfg["chest_train_list"]) as f:
                train_val_imgs = [l.strip() for l in f if l.strip()]
            with open(cfg["chest_val_list"]) as f:
                official_test_imgs = [l.strip() for l in f if l.strip()]

            total_n = len(train_val_imgs) + len(official_test_imgs)
            test_frac = len(official_test_imgs) / total_n
            log.info(f"[Chest] Total imágenes: {total_n:,} | "
                     f"train_val_list: {len(train_val_imgs):,} | "
                     f"test_list: {len(official_test_imgs):,} ({test_frac:.1%})")

            rng_nih_test = np.random.RandomState(42)
            rng_nih_pool = np.random.RandomState(43)

            if test_frac > 0.10:
                desired_test_n = int(total_n * 0.10)
                log.info(f"[Chest] test_list ({test_frac:.1%}) excede 10 % "
                         f"→ recortando a ~{desired_test_n:,} imágenes")
                # Agrupar imágenes test por Patient ID
                _test_pid_imgs = {}
                for img in official_test_imgs:
                    pid = _img_to_pid.get(img)
                    if pid is not None:
                        _test_pid_imgs.setdefault(pid, []).append(img)
                _test_pids = list(_test_pid_imgs.keys())
                rng_nih_test.shuffle(_test_pids)
                final_test_imgs, excess_imgs = [], []
                for pid in _test_pids:
                    if len(final_test_imgs) < desired_test_n:
                        final_test_imgs.extend(_test_pid_imgs[pid])
                    else:
                        excess_imgs.extend(_test_pid_imgs[pid])
                pool_imgs = train_val_imgs + excess_imgs
                log.info(f"[Chest] test recortado: {len(final_test_imgs):,} imgs | "
                         f"excedente devuelto al pool: {len(excess_imgs):,}")
            else:
                final_test_imgs = official_test_imgs
                pool_imgs = train_val_imgs

            # Separar pool → train + val por Patient ID
            val_target = int(total_n * 0.10)
            _pool_pid_imgs = {}
            for img in pool_imgs:
                pid = _img_to_pid.get(img)
                if pid is not None:
                    _pool_pid_imgs.setdefault(pid, []).append(img)
            _pool_pids = list(_pool_pid_imgs.keys())
            rng_nih_pool.shuffle(_pool_pids)
            final_val_imgs, final_train_imgs = [], []
            for pid in _pool_pids:
                if len(final_val_imgs) < val_target:
                    final_val_imgs.extend(_pool_pid_imgs[pid])
                else:
                    final_train_imgs.extend(_pool_pid_imgs[pid])

            log.info(f"[Chest] Split final: train={len(final_train_imgs):,} | "
                     f"val={len(final_val_imgs):,} | test={len(final_test_imgs):,}")

            # Verificar disjunción de Patient IDs
            _tr_pids = {_img_to_pid[i] for i in final_train_imgs if i in _img_to_pid}
            _va_pids = {_img_to_pid[i] for i in final_val_imgs   if i in _img_to_pid}
            _te_pids = {_img_to_pid[i] for i in final_test_imgs  if i in _img_to_pid}
            _leak = (len(_tr_pids & _va_pids), len(_tr_pids & _te_pids),
                     len(_va_pids & _te_pids))
            if any(_leak):
                log.error(f"[Chest] ¡LEAKAGE Patient ID! "
                          f"train∩val={_leak[0]} | train∩test={_leak[1]} | "
                          f"val∩test={_leak[2]}")
            else:
                log.info("[Chest] Sin leakage de Patient ID entre splits ✓")

            # Escribir listas de split para ChestXray14Dataset
            splits_dir = Path(cfg["output_dir"]) / "_chest_splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            for fname, imgs in [("train.txt", final_train_imgs),
                                ("val.txt", final_val_imgs),
                                ("test.txt", final_test_imgs)]:
                (splits_dir / fname).write_text("\n".join(imgs))

            view_filter = cfg.get("chest_view_filter", None)
            if view_filter:
                log.info(f"[Chest] H7 — filter_view='{view_filter}' aplicado "
                         f"SIMÉTRICAMENTE a train, val y test.")

            chest_train = ChestXray14Dataset(
                csv_path=cfg["chest_csv"],
                img_dir=cfg["chest_imgs"],
                file_list=str(splits_dir / "train.txt"),
                transform=transform_2d,
                mode="embedding",
                patient_ids_other=_va_pids | _te_pids,
                filter_view=view_filter,
            )
            chest_val = ChestXray14Dataset(
                csv_path=cfg["chest_csv"],
                img_dir=cfg["chest_imgs"],
                file_list=str(splits_dir / "val.txt"),
                transform=transform_2d,
                mode="embedding",
                patient_ids_other=_tr_pids | _te_pids,
                filter_view=view_filter,
            )
            chest_test = ChestXray14Dataset(
                csv_path=cfg["chest_csv"],
                img_dir=cfg["chest_imgs"],
                file_list=str(splits_dir / "test.txt"),
                transform=transform_2d,
                mode="embedding",
                patient_ids_other=_tr_pids | _va_pids,
                filter_view=view_filter,
            )
            train_datasets.append(chest_train)
            val_datasets.append(chest_val)
            n_chest_total = len(chest_train) + len(chest_val) + len(chest_test)
            if n_chest_total < MIN_SAMPLES_FOR_TEST:
                log.warning(f"[Chest] Solo {n_chest_total} muestras — test omitido.")
            else:
                test_datasets.append(chest_test)
            log.info(f"  → train: {len(chest_train):,}  val: {len(chest_val):,}"
                     f"  test: {len(chest_test):,}")

        if cfg.get("chest_bbox_csv"):
            try:
                _ = ChestXray14Dataset.load_bbox_index(cfg["chest_bbox_csv"])
                log.info("[Chest] H5 — BBox index cargado.")
            except Exception as e:
                log.warning(f"[Chest] H5 — No se pudo cargar BBox_List_2017.csv: {e}")
        else:
            log.info("[Chest] H5 — BBox no cargado (pasa --chest_bbox_csv para activar).")
    else:
        log.warning("[Dataset] NIH ChestXray14 OMITIDO — falta --chest_csv o --chest_imgs")

    # ── Experto 1: ISIC 2019 (80/10/10 por lesion_id) ──
    if cfg.get("isic_gt") and cfg.get("isic_imgs"):
        log.info("[Dataset] Cargando ISIC 2019 (Experto 1)...")

        isic_train_df, isic_rest_df = ISICDataset.build_lesion_split(
            gt_csv       = cfg["isic_gt"],
            img_dir      = cfg.get("isic_imgs", None),
            metadata_csv = cfg.get("isic_metadata", None),
            frac_train   = 0.8,
            random_state = 42,
        )

        # Dividir rest (20 %) en val (10 %) + test (10 %) por lesion_id
        if "lesion_id" in isic_rest_df.columns:
            _rest_lesions = isic_rest_df["lesion_id"].unique()
            _rng_isic = np.random.RandomState(42)
            _rng_isic.shuffle(_rest_lesions)
            _n_half = len(_rest_lesions) // 2
            _val_lesion_set = set(_rest_lesions[:_n_half])
            isic_val_df  = isic_rest_df[
                isic_rest_df["lesion_id"].isin(_val_lesion_set)
            ].reset_index(drop=True)
            isic_test_df = isic_rest_df[
                ~isic_rest_df["lesion_id"].isin(_val_lesion_set)
            ].reset_index(drop=True)
            log.info(f"[ISIC] rest dividido por lesion_id: "
                     f"val={len(isic_val_df):,} | test={len(isic_test_df):,}")
        else:
            _n_rest = len(isic_rest_df)
            _idx_rest = np.arange(_n_rest)
            _rng_isic = np.random.RandomState(42)
            _rng_isic.shuffle(_idx_rest)
            _n_half = _n_rest // 2
            isic_val_df  = isic_rest_df.iloc[_idx_rest[:_n_half]].reset_index(drop=True)
            isic_test_df = isic_rest_df.iloc[_idx_rest[_n_half:]].reset_index(drop=True)
            log.info(f"[ISIC] rest dividido aleatoriamente: "
                     f"val={len(isic_val_df):,} | test={len(isic_test_df):,}")

        isic_train = ISICDataset(
            img_dir  = cfg["isic_imgs"],
            split_df = isic_train_df,
            mode     = "embedding",
        )
        isic_val = ISICDataset(
            img_dir  = cfg["isic_imgs"],
            split_df = isic_val_df,
            mode     = "embedding",
        )
        isic_test = ISICDataset(
            img_dir  = cfg["isic_imgs"],
            split_df = isic_test_df,
            mode     = "embedding",
        )
        train_datasets.append(isic_train)
        val_datasets.append(isic_val)
        n_isic_total = len(isic_train) + len(isic_val) + len(isic_test)
        if n_isic_total < MIN_SAMPLES_FOR_TEST:
            log.warning(f"[ISIC] Solo {n_isic_total} muestras — test omitido.")
        else:
            test_datasets.append(isic_test)
        log.info(f"  → train: {len(isic_train):,}  val: {len(isic_val):,}"
                 f"  test: {len(isic_test):,}")
    else:
        log.warning("[Dataset] ISIC 2019 OMITIDO — falta --isic_gt o --isic_imgs")

    # ── Experto 2: OA Rodilla (carpetas predefinidas train/val/test) ──
    if cfg.get("oa_root"):
        log.info("[Dataset] Cargando Osteoarthritis Knee (Experto 2)...")
        _oa_root = Path(cfg["oa_root"])
        _has_test = (_oa_root / "test").exists() and any((_oa_root / "test").iterdir())

        oa_train = OAKneeDataset(cfg["oa_root"], split="train", mode="embedding")
        oa_val   = OAKneeDataset(cfg["oa_root"], split="val",   mode="embedding")

        if _has_test:
            oa_test = OAKneeDataset(cfg["oa_root"], split="test", mode="embedding")
        else:
            log.warning("[OA] Carpeta test/ no encontrada — generando desde train/")
            _n_oa = len(oa_train)
            _oa_idx = list(range(_n_oa))
            _rng_oa = np.random.RandomState(42)
            _rng_oa.shuffle(_oa_idx)
            _n_oa_test = max(1, int(_n_oa * 0.111))  # ~10 % del total ≈ 10 % de (train+val)
            oa_test  = Subset(oa_train, _oa_idx[:_n_oa_test])
            oa_train = Subset(oa_train, _oa_idx[_n_oa_test:])

        train_datasets.append(oa_train)
        val_datasets.append(oa_val)
        n_oa_total = len(oa_train) + len(oa_val) + len(oa_test)
        if n_oa_total < MIN_SAMPLES_FOR_TEST:
            log.warning(f"[OA] Solo {n_oa_total} muestras — test omitido.")
        else:
            test_datasets.append(oa_test)
        log.info(f"  → train: {len(oa_train):,}  val: {len(oa_val):,}"
                 f"  test: {len(oa_test):,}")
    else:
        log.warning("[Dataset] OA Rodilla OMITIDO — falta --oa_root")

    # ── Experto 3: LUNA16 (split por seriesuid: train/ intacto, val/ → val+test) ──
    if cfg.get("luna_patches") and cfg.get("luna_csv"):
        log.info("[Dataset] Cargando LUNA16 parches 3D (Experto 3)...")
        _luna_base = Path(cfg["luna_patches"])
        if (_luna_base / "train").exists():
            _luna_train_dir = str(_luna_base / "train")
            _luna_val_dir   = str(_luna_base / "val")
            log.debug(f"[LUNA16] Modo A (padre): train={_luna_train_dir} | val={_luna_val_dir}")
        else:
            _luna_train_dir = str(_luna_base)
            _luna_val_dir   = str(_luna_base.parent / "val")
            log.debug(f"[LUNA16] Modo B (train directo): train={_luna_train_dir} | val={_luna_val_dir}")

        _luna_ds_a = LUNA16Dataset(_luna_train_dir, cfg["luna_csv"], mode="embedding")
        _luna_ds_b = LUNA16Dataset(_luna_val_dir,   cfg["luna_csv"], mode="embedding")
        _n_luna    = len(_luna_ds_a) + len(_luna_ds_b)

        if _n_luna < MIN_SAMPLES_FOR_TEST:
            log.warning(f"[LUNA16] Solo {_n_luna} parches — test omitido.")
            train_datasets.append(_luna_ds_a)
            val_datasets.append(_luna_ds_b)
            log.info(f"  → train: {len(_luna_ds_a):,}  val: {len(_luna_ds_b):,}  test: 0")
        else:
            # Respetar separación original por seriesuid de extract_luna_patches.py:
            # train/ ya está separado por seriesuid del val/ — no re-mezclar.
            # La carpeta val/ se divide en dos mitades deterministas: val y test.
            _n_b = len(_luna_ds_b)
            _n_half = _n_b // 2
            luna_train = _luna_ds_a
            luna_val   = Subset(_luna_ds_b, list(range(_n_half)))
            luna_test  = Subset(_luna_ds_b, list(range(_n_half, _n_b)))
            train_datasets.append(luna_train)
            val_datasets.append(luna_val)
            test_datasets.append(luna_test)
            log.info(f"[LUNA16] Split respeta separación original por seriesuid "
                     f"de extract_luna_patches.py (train/ intacto, val/ → val+test).")
            log.info(f"  → train: {len(luna_train):,}  val: {len(luna_val):,}"
                     f"  test: {len(luna_test):,}")
    else:
        log.warning("[Dataset] LUNA16 OMITIDO — falta --luna_patches o --luna_csv")

    # ── Experto 4: Pancreas PANORAMA ──
    if cfg.get("pancreas_nii_dir") and cfg.get("pancreas_labels_dir"):
        log.info("[Dataset] Cargando Pancreas CT 3D — PANORAMA (Experto 4)...")

        _NII_MIN_EXPECTED = 50
        nii_dir_path = Path(cfg["pancreas_nii_dir"])
        batch_zip    = nii_dir_path / "batch_1.zip"
        _nii_on_disk = list(nii_dir_path.rglob("*.nii.gz")) + \
                       list(nii_dir_path.rglob("*.nii"))
        _nii_count   = len(_nii_on_disk)

        if _nii_count >= _NII_MIN_EXPECTED:
            log.info(
                f"[Pancreas/ZIP] {_nii_count} archivos NIfTI encontrados en disco — "
                f"extraccion confirmada (minimo esperado: {_NII_MIN_EXPECTED}) ✓"
            )
        elif batch_zip.exists():
            zip_size_gb = batch_zip.stat().st_size / 1e9
            log.warning(
                f"[Pancreas/ZIP] Solo {_nii_count} archivos .nii.gz en disco "
                f"(minimo esperado: {_NII_MIN_EXPECTED}).\n"
                f"    batch_1.zip existe ({zip_size_gb:.1f} GB) pero parece no extraido.\n"
                f"    Extrayendo automaticamente..."
            )
            import subprocess as _sp
            _result = _sp.run(
                ["unzip", "-n", str(batch_zip)],
                cwd=str(nii_dir_path),
                capture_output=False,
                timeout=3600
            )
            if _result.returncode == 0:
                _nii_on_disk = list(nii_dir_path.rglob("*.nii.gz")) + \
                               list(nii_dir_path.rglob("*.nii"))
                _nii_count   = len(_nii_on_disk)
                log.info(f"[Pancreas/ZIP] Extraccion completada — {_nii_count} archivos NIfTI disponibles.")
            else:
                log.error(f"[Pancreas/ZIP] unzip termino con codigo {_result.returncode}.")
        else:
            log.error(
                f"[Pancreas/ZIP] {_nii_count} archivos NIfTI en disco y "
                f"batch_1.zip NO encontrado en '{nii_dir_path}'."
            )

        nii_files  = list(nii_dir_path.rglob("*.nii.gz")) + \
                     list(nii_dir_path.rglob("*.nii"))
        labels     = PanoramaLabelLoader.load_labels(
            cfg["pancreas_labels_dir"],
            expected_commit=cfg.get("pancreas_labels_commit", None),
        )
        all_pairs  = PanoramaLabelLoader.cross_match(labels, nii_files)

        if not all_pairs:
            log.error("[Pancreas] Sin pares válidos — verificar H1 (etiquetas + volúmenes).")
        else:
            import random as _random
            _random.seed(42)
            _random.shuffle(all_pairs)
            _n_panc  = len(all_pairs)
            _n_ptr   = int(_n_panc * 0.8)
            _n_pva   = int(_n_panc * 0.1)
            panc_train_pairs = all_pairs[:_n_ptr]
            panc_val_pairs   = all_pairs[_n_ptr:_n_ptr + _n_pva]
            panc_test_pairs  = all_pairs[_n_ptr + _n_pva:]

            roi = cfg.get("pancreas_roi_strategy", "A")
            panc_train = PancreasDataset(panc_train_pairs, mode="embedding",
                                         roi_strategy=roi)
            panc_val   = PancreasDataset(panc_val_pairs,   mode="embedding",
                                         roi_strategy=roi)
            train_datasets.append(panc_train)
            val_datasets.append(panc_val)
            if _n_panc < MIN_SAMPLES_FOR_TEST:
                log.warning(f"[Pancreas] Solo {_n_panc} pares — test omitido.")
                log.info(f"  → train: {len(panc_train):,}  val: {len(panc_val):,}  test: 0")
            else:
                panc_test = PancreasDataset(panc_test_pairs, mode="embedding",
                                            roi_strategy=roi)
                test_datasets.append(panc_test)
                log.info(f"  → train: {len(panc_train):,}  val: {len(panc_val):,}"
                         f"  test: {len(panc_test):,}")
    else:
        log.warning("[Dataset] Pancreas OMITIDO — pasa --pancreas_nii_dir y "
                    "--pancreas_labels_dir")

    # ── Experto 5 (OOD) — ausente intencionalmente ──
    log.info("[Dataset] Experto 5 (OOD) — sin dataset en FASE 0. Se inicializa en FASE 1.")

    if not train_datasets:
        log.error("[Dataset] ¡Ningún dataset cargado! Verifica que al menos una ruta sea válida.")

    for _sname, _ds_list in [("train", train_datasets),
                              ("val", val_datasets),
                              ("test", test_datasets)]:
        _sizes = [len(d) for d in _ds_list]
        if _sizes:
            log.info(f"[Dataset] Tamaños {_sname} por experto: {_sizes}")
            _nonzero = [s for s in _sizes if s > 0]
            if len(_nonzero) > 1:
                _ratio = max(_nonzero) / min(_nonzero)
                if _ratio > 10:
                    log.warning(f"[Dataset/{_sname}] Balance muy desigual: "
                                f"max/min = {_ratio:.1f}x.")
                else:
                    log.info(f"[Dataset/{_sname}] ratio max/min = {_ratio:.1f}x")
            _zero = [i for i, s in enumerate(_sizes) if s == 0]
            if _zero:
                log.warning(f"[Dataset/{_sname}] Expertos con 0 muestras (indices): {_zero}")

    return (ConcatDataset(train_datasets),
            ConcatDataset(val_datasets),
            ConcatDataset(test_datasets))


# ──────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────

def main(args):
    global log
    log = setup_logging(args.output_dir)

    log.info("=" * 60)
    log.info("FASE 0 — Extracción de CLS tokens (Embeddings)")
    log.info("=" * 60)
    log.info(f"Argumentos recibidos: {vars(args)}")

    # ── Guard clause: LUNA16 patches deben existir ──
    _check_luna_patches(args.luna_patches)

    # ── Setup de dispositivo ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[Setup] Dispositivo: {device}")
    if device == "cuda":
        log.info(f"[Setup] GPU           : {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"[Setup] VRAM total    : {total_vram:.1f} GB")
        expected_vram = BACKBONE_CONFIGS[args.backbone]["vram_gb"]
        if expected_vram > total_vram * 0.8:
            log.warning(f"[Setup] El backbone '{args.backbone}' requiere ~{expected_vram} GB "
                        f"pero solo hay {total_vram:.1f} GB. Considera reducir --batch_size.")
    else:
        log.warning("[Setup] Corriendo en CPU — la extracción será muy lenta. "
                    "Se recomienda GPU para un dataset de esta escala.")

    # ── Cargar backbone congelado ──
    log.info("[Setup] Cargando backbone...")
    model, d_model = load_frozen_backbone(args.backbone, device)

    # ── Construir datasets ──
    cfg = {
        "chest_csv":         args.chest_csv,
        "chest_imgs":        args.chest_imgs,
        "chest_train_list":  args.chest_train_list,
        "chest_val_list":    args.chest_val_list,
        "chest_view_filter": args.chest_view_filter,
        "chest_bbox_csv":    args.chest_bbox_csv,
        "isic_gt":           args.isic_gt,
        "isic_imgs":         args.isic_imgs,
        "isic_metadata":     args.isic_metadata,
        "oa_root":           args.oa_root,
        "luna_patches":      args.luna_patches,
        "luna_csv":          args.luna_csv,
        "pancreas_nii_dir":       args.pancreas_nii_dir,
        "pancreas_labels_dir":    args.pancreas_labels_dir,
        "pancreas_labels_commit": args.pancreas_labels_commit,
        "pancreas_roi_strategy":  args.pancreas_roi_strategy,
        "output_dir":             args.output_dir,
    }

    log.info("[Setup] Construyendo datasets...")
    train_dataset, val_dataset, test_dataset = build_datasets(cfg)
    log.info(f"[Dataset] Total train: {len(train_dataset):,}")
    log.info(f"[Dataset] Total val  : {len(val_dataset):,}")
    log.info(f"[Dataset] Total test : {len(test_dataset):,}")

    _dl_kw = dict(batch_size=args.batch_size, shuffle=False,
                  num_workers=args.workers, pin_memory=(device == "cuda"))
    train_loader = DataLoader(train_dataset, **_dl_kw)
    val_loader   = DataLoader(val_dataset,   **_dl_kw)

    # ── Extraer embeddings ──
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("[FASE 0] Extrayendo embeddings de TRAIN...")
    t0 = time.time()
    Z_train, y_train, names_train = extract_embeddings(
        model, train_loader, device, d_model, desc="train"
    )
    log.info(f"[FASE 0] TRAIN completado en {time.time()-t0:.1f}s")

    log.info("[FASE 0] Extrayendo embeddings de VAL...")
    t0 = time.time()
    Z_val, y_val, names_val = extract_embeddings(
        model, val_loader, device, d_model, desc="val"
    )
    log.info(f"[FASE 0] VAL completado en {time.time()-t0:.1f}s")

    # ── Test embeddings ──
    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, **_dl_kw)
        log.info("[FASE 0] Extrayendo embeddings de TEST...")
        t0 = time.time()
        Z_test, y_test, names_test = extract_embeddings(
            model, test_loader, device, d_model, desc="test"
        )
        log.info(f"[FASE 0] TEST completado en {time.time()-t0:.1f}s")
    else:
        Z_test     = np.zeros((0, d_model), dtype=np.float32)
        y_test     = np.zeros(0, dtype=np.int64)
        names_test = []
        log.warning("[FASE 0] Sin datos de test — se omite extracción de test.")

    # ── Guardar en disco ──
    out = Path(args.output_dir)
    log.info(f"[Guardado] Escribiendo archivos en '{out}'...")
    np.save(out / "Z_train.npy", Z_train)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "Z_val.npy",   Z_val)
    np.save(out / "y_val.npy",   y_val)
    np.save(out / "Z_test.npy",  Z_test)
    np.save(out / "y_test.npy",  y_test)
    log.debug(f"[Guardado] Z_train.npy: {Z_train.shape}  ({Z_train.nbytes/1e6:.1f} MB)")
    log.debug(f"[Guardado] Z_val.npy  : {Z_val.shape}  ({Z_val.nbytes/1e6:.1f} MB)")
    log.debug(f"[Guardado] Z_test.npy : {Z_test.shape}  ({Z_test.nbytes/1e6:.1f} MB)")

    with open(out / "names_train.txt", "w") as f:
        f.write("\n".join(names_train))
    with open(out / "names_val.txt", "w") as f:
        f.write("\n".join(names_val))
    with open(out / "names_test.txt", "w") as f:
        f.write("\n".join(names_test))

    backbone_meta = {
        "backbone": args.backbone,
        "d_model":  int(Z_train.shape[1]),
        "n_train":  int(Z_train.shape[0]),
        "n_val":    int(Z_val.shape[0]),
        "n_test":   int(Z_test.shape[0]),
        "vram_gb":  BACKBONE_CONFIGS[args.backbone]["vram_gb"],
    }
    with open(out / "backbone_meta.json", "w") as f:
        json.dump(backbone_meta, f, indent=2)
    log.info(f"[Guardado] backbone_meta.json escrito")

    # ── Reporte final ──
    log.info("=" * 55)
    log.info("FASE 0 completada")
    log.info("=" * 55)
    log.info(f"Z_train : {Z_train.shape}  ({Z_train.nbytes/1e6:.1f} MB)")
    log.info(f"Z_val   : {Z_val.shape}  ({Z_val.nbytes/1e6:.1f} MB)")
    log.info(f"Z_test  : {Z_test.shape}  ({Z_test.nbytes/1e6:.1f} MB)")

    for _sname, _y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if len(_y) == 0:
            continue
        log.info(f"Distribución de expertos en {_sname} (dominios clínicos 0–4):")
        for exp_name, exp_id in EXPERT_IDS.items():
            count = (_y == exp_id).sum()
            pct   = count / len(_y) * 100
            log.info(f"  Experto {exp_id} ({exp_name:<10}): {count:>6,}  ({pct:.1f}%)")
        log.info(f"  Experto 5 (ood       ):      0  (0.0%)  ← sin dataset en FASE 0")
        _counts = [(_y == i).sum() for i in range(N_EXPERTS_DOMAIN)]
        _nz = [c for c in _counts if c > 0]
        if len(_nz) > 1:
            _bal = max(_nz) / min(_nz)
            if _bal > 10:
                log.warning(f"[Balance/{_sname}] ratio max/min = {_bal:.1f}x — muy desigual.")
            else:
                log.info(f"[Balance/{_sname}] ratio max/min = {_bal:.1f}x")

    log.info(f"  [Experto 5 se inicializa en FASE 1 como MLP OOD — entrenado via L_error]")
    log.info(f"  N_EXPERTS_DOMAIN = {N_EXPERTS_DOMAIN} | N_EXPERTS_TOTAL = {N_EXPERTS_TOTAL}")
    log.info(f"Archivos guardados en: {args.output_dir}")
    log.info(f"Siguiente paso: python fase1_ablation_router.py --embeddings {args.output_dir}")


# ──────────────────────────────────────────────────────────
# ARGPARSE
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FASE 0 — Extracción de embeddings MoE")

    # ── Backbone ──
    parser.add_argument(
        "--backbone",
        default="vit_tiny_patch16_224",
        choices=list(BACKBONE_CONFIGS.keys()),
        help=(
            "Backbone para extraer CLS tokens. Opciones:\n"
            "  vit_tiny_patch16_224        → d_model=192, ~2GB VRAM (DEFAULT)\n"
            "  swin_tiny_patch4_window7_224→ d_model=768, ~4GB VRAM\n"
            "  cvt_13                      → d_model=384, ~3GB VRAM"
        )
    )
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--output_dir",   default="./embeddings",
                        help="Carpeta donde se guardan Z_{train,val,test}.npy")

    # ── Rutas NIH ChestXray14 ──
    parser.add_argument("--chest_csv",        default=None)
    parser.add_argument("--chest_imgs",       default=None,
                        help="Directorio con las imágenes de ChestXray14 (all_images/ recomendado)")
    parser.add_argument("--chest_train_list", default=None,
                        help="train_val_list.txt oficial")
    parser.add_argument("--chest_val_list",   default=None,
                        help="test_list.txt oficial (usado para computar split 80/10/10)")
    parser.add_argument(
        "--chest_view_filter", default=None, choices=["PA", "AP", None],
        help="H4 — Filtrar por vista radiográfica: PA, AP o None."
    )
    parser.add_argument(
        "--chest_bbox_csv", default=None,
        help="H5 — Ruta a BBox_List_2017.csv (opcional para validación de heatmaps)."
    )

    # ── Rutas ISIC 2019 ──
    parser.add_argument("--isic_gt", default=None,
                        help="ISIC_2019_Training_GroundTruth.csv")
    parser.add_argument("--isic_imgs", default=None,
                        help="Carpeta con imágenes .jpg de ISIC")
    parser.add_argument(
        "--isic_metadata", default=None,
        help="H2 — ISIC_2019_Training_Metadata.csv para split por lesion_id."
    )

    # ── Rutas OA Rodilla ──
    parser.add_argument("--oa_root", default=None,
                        help="Raíz con subcarpetas train/val/test y clases 0,1,2")

    # ── Rutas LUNA16 ──
    parser.add_argument("--luna_patches", default=None,
                        help="Carpeta con parches 3D .npy pre-extraídos (train/ y val/)")
    parser.add_argument("--luna_csv", default=None,
                        help="candidates_V2.csv")

    # ── Rutas Pancreas PANORAMA (modo completo) ──
    parser.add_argument(
        "--pancreas_nii_dir", default=None,
        help="H1 — Directorio con los .nii.gz del ZIP de Zenodo"
    )
    parser.add_argument(
        "--pancreas_labels_dir", default=None,
        help="H1 — Directorio del repo clonado panorama_labels"
    )
    parser.add_argument(
        "--pancreas_labels_commit", default=None,
        help="H1 — Hash SHA del commit del repo panorama_labels para reproducibilidad."
    )
    parser.add_argument(
        "--pancreas_roi_strategy", default="A", choices=["A", "B"],
        help="H2 — Estrategia ROI: A=resize completo (FASE 0), B=recorte Z (FASE 2)."
    )

    args = parser.parse_args()
    main(args)

"""
fase1_pipeline.py — Orquestador de Fase 1: Extracción de Embeddings
===================================================================

Responsabilidad única: definir el orden de ejecución de todos los módulos,
gestionar argumentos CLI y producir el reporte final.

No contiene lógica de negocio propia — solo llama funciones.
Nunca implementa transformaciones ni escribe arrays directamente.

Uso:
    python src/pipeline/fase1/fase1_pipeline.py \\
        --backbone vit_tiny_patch16_224 \\
        --batch_size 64 --workers 4 \\
        --output_dir embeddings/vit_tiny \\
        --nih_train_list datasets/nih_chest_xrays/splits/nih_train_list.txt \\
        --nih_val_list datasets/nih_chest_xrays/splits/nih_val_list.txt \\
        --nih_test_list datasets/nih_chest_xrays/splits/nih_test_list.txt \\
        ...
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────
# Permite ejecutar como script: python src/pipeline/fase1/fase1_pipeline.py
_THIS_DIR     = Path(__file__).resolve().parent          # src/pipeline/fase1/
_PIPELINE_DIR = _THIS_DIR.parent                         # src/pipeline/
_SRC_DIR      = _PIPELINE_DIR.parent                     # src/
_PROJECT_ROOT = _SRC_DIR.parent                          # proyecto_2/

for _p in [str(_THIS_DIR), str(_PIPELINE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Imports de fase1 (via sys.path → fase1/) ───────────────
from fase1_config import (
    BACKBONE_CONFIGS, DEFAULT_BATCH_SIZE, DEFAULT_WORKERS,
    PANCREAS_FOLD, HU_ABDOMEN_CLIP, MIN_L2_NORM,
)
import backbone_cvt13                   # noqa: F401 — activar registro en timm
from backbone_loader import load_frozen_backbone
from dataset_builder import build_datasets
from embeddings_extractor import extract_embeddings
from embeddings_storage import save_embeddings, log_distribution

# ── Imports de pipeline global (via sys.path → pipeline/) ───
from config import EXPERT_IDS, N_EXPERTS_DOMAIN, N_EXPERTS_TOTAL
from logging_utils import setup_logging

import numpy as np
import torch
from torch.utils.data import DataLoader

log = logging.getLogger("fase1")


# ══════════════════════════════════════════════════════════════
#  Guard Clauses
# ══════════════════════════════════════════════════════════════

def _check_luna_patches(luna_patches_dir):
    """Verificar que patches/train/, patches/val/, patches/test/ existen."""
    if luna_patches_dir is None:
        return
    base = Path(luna_patches_dir)
    missing = []
    for split_name in ("train", "val", "test"):
        split_dir = base / split_name
        if not split_dir.exists():
            missing.append("  ✗ {} (no existe)".format(split_dir))
        elif not any(split_dir.glob("*.npy")):
            missing.append("  ✗ {} (vacía — sin .npy)".format(split_dir))
    if missing:
        log.error(
            "LUNA16: parches 3D no encontrados.\n%s\n\n"
            "Ejecuta primero:\n"
            "  python fase0/fase0_pipeline.py --solo_pasos 6 --solo luna\n"
            "antes de correr fase1_pipeline.py.",
            "\n".join(missing),
        )
        sys.exit(1)
    log.info("[Guard] LUNA16 patches verificados ✓ (train/, val/, test/)")


def _check_fase0_artifacts(cfg):
    """Verificar que los artefactos de Fase 0 necesarios existen."""
    checks = []

    # NIH splits
    for key in ("nih_train_list", "nih_val_list", "nih_test_list"):
        path = cfg.get(key)
        if path and not Path(path).exists():
            checks.append("  ✗ {} = {}".format(key, path))

    # ISIC splits
    for key in ("isic_train_csv", "isic_val_csv", "isic_test_csv"):
        path = cfg.get(key)
        if path and not Path(path).exists():
            checks.append("  ✗ {} = {}".format(key, path))

    # OA splits
    oa_root = cfg.get("oa_root")
    if oa_root:
        oa_path = Path(oa_root)
        for split in ("train", "val", "test"):
            sd = oa_path / split
            if not sd.exists():
                checks.append("  ✗ oa_root/{} no existe".format(split))

    # Pancreas splits
    panc_csv = cfg.get("pancreas_splits_csv")
    if panc_csv and not Path(panc_csv).exists():
        checks.append("  ✗ pancreas_splits_csv = {}".format(panc_csv))

    if checks:
        log.error(
            "Artefactos de Fase 0 faltantes:\n%s\n\n"
            "Ejecuta Fase 0 para generarlos:\n"
            "  python fase0/fase0_pipeline.py",
            "\n".join(checks),
        )
        sys.exit(1)
    log.info("[Guard] Artefactos de Fase 0 verificados ✓")


# ══════════════════════════════════════════════════════════════
#  Pipeline Principal
# ══════════════════════════════════════════════════════════════

def main(args):
    """Orquesta la extracción completa de embeddings."""
    log_obj = setup_logging(args.output_dir)

    log.info("=" * 60)
    log.info("FASE 1 — Extracción de CLS tokens (Embeddings)")
    log.info("=" * 60)
    log.info("Argumentos: %s", vars(args))

    # ── 1. Guard clause: LUNA16 patches ──
    _check_luna_patches(args.luna_patches_dir)

    # ── 2. Guard clause: artefactos de Fase 0 ──
    cfg = {
        "chest_csv":             args.chest_csv,
        "chest_imgs":            args.chest_imgs,
        "nih_train_list":        args.nih_train_list,
        "nih_val_list":          args.nih_val_list,
        "nih_test_list":         args.nih_test_list,
        "chest_view_filter":     args.chest_view_filter,
        "chest_bbox_csv":        args.chest_bbox_csv,
        "isic_train_csv":        args.isic_train_csv,
        "isic_val_csv":          args.isic_val_csv,
        "isic_test_csv":         args.isic_test_csv,
        "isic_imgs":             args.isic_imgs,

        "oa_root":               args.oa_root,
        "luna_patches_dir":      args.luna_patches_dir,
        "luna_csv":              args.luna_csv,
        "pancreas_splits_csv":   args.pancreas_splits_csv,
        "pancreas_nii_dir":      args.pancreas_nii_dir,
        "pancreas_fold":         args.pancreas_fold,
        "pancreas_roi_strategy": args.pancreas_roi_strategy,
        "output_dir":            args.output_dir,
    }
    _check_fase0_artifacts(cfg)

    # ── 3. Setup de dispositivo ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("[Setup] Dispositivo: %s", device)
    if device == "cuda":
        log.info("[Setup] GPU           : %s", torch.cuda.get_device_name(0))
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("[Setup] VRAM total    : %.1f GB", total_vram)
        expected_vram = BACKBONE_CONFIGS[args.backbone]["vram_gb"]
        if expected_vram > total_vram * 0.8:
            log.warning(
                "[Setup] Backbone '%s' requiere ~%.0f GB pero hay %.1f GB. "
                "Considera reducir --batch_size.", args.backbone,
                expected_vram, total_vram,
            )
    else:
        log.warning("[Setup] CPU — la extracción será lenta. "
                    "Se recomienda GPU.")

    # ── 4. Cargar backbone congelado ──
    log.info("[Setup] Cargando backbone '%s'...", args.backbone)
    model, d_model = load_frozen_backbone(args.backbone, device)

    # ── 5. Construir datasets (lectura pura de artefactos Fase 0) ──
    log.info("[Setup] Construyendo datasets...")
    train_dataset, val_dataset, test_dataset = build_datasets(cfg)
    log.info("[Dataset] Total train: %s", f"{len(train_dataset):,}")
    log.info("[Dataset] Total val  : %s", f"{len(val_dataset):,}")
    log.info("[Dataset] Total test : %s", f"{len(test_dataset):,}")

    # ── 6. DataLoaders ──
    _dl_kw = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )
    train_loader = DataLoader(train_dataset, **_dl_kw)
    val_loader   = DataLoader(val_dataset,   **_dl_kw)
    test_loader  = DataLoader(test_dataset,  **_dl_kw) \
        if len(test_dataset) > 0 else None

    # ── 7. Extraer embeddings ──
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("[FASE 1] Extrayendo embeddings de TRAIN...")
    t0 = time.time()
    Z_train, y_train, names_train = extract_embeddings(
        model, train_loader, device, d_model, desc="train",
    )
    log.info("[FASE 1] TRAIN completado en %.1fs", time.time() - t0)

    log.info("[FASE 1] Extrayendo embeddings de VAL...")
    t0 = time.time()
    Z_val, y_val, names_val = extract_embeddings(
        model, val_loader, device, d_model, desc="val",
    )
    log.info("[FASE 1] VAL completado en %.1fs", time.time() - t0)

    if test_loader is not None:
        log.info("[FASE 1] Extrayendo embeddings de TEST...")
        t0 = time.time()
        Z_test, y_test, names_test = extract_embeddings(
            model, test_loader, device, d_model, desc="test",
        )
        log.info("[FASE 1] TEST completado en %.1fs", time.time() - t0)
    else:
        Z_test     = np.zeros((0, d_model), dtype=np.float32)
        y_test     = np.zeros(0, dtype=np.int64)
        names_test = []
        log.warning("[FASE 1] Sin datos de test — extracción omitida.")

    # ── 8. Guardar en disco ──
    save_embeddings(
        args.output_dir, args.backbone,
        Z_train, y_train, names_train,
        Z_val, y_val, names_val,
        Z_test, y_test, names_test,
    )

    # ── 9. Reporte final ──
    log.info("=" * 55)
    log.info("FASE 1 completada")
    log.info("=" * 55)
    log.info("Z_train : %s  (%.1f MB)", Z_train.shape, Z_train.nbytes / 1e6)
    log.info("Z_val   : %s  (%.1f MB)", Z_val.shape, Z_val.nbytes / 1e6)
    log.info("Z_test  : %s  (%.1f MB)", Z_test.shape, Z_test.nbytes / 1e6)

    log_distribution(y_train, y_val, y_test, EXPERT_IDS)

    log.info("  [Experto 5 se inicializa en FASE 2 como MLP OOD — "
             "entrenado via L_error]")
    log.info("  N_EXPERTS_DOMAIN = %d | N_EXPERTS_TOTAL = %d",
             N_EXPERTS_DOMAIN, N_EXPERTS_TOTAL)
    log.info("Archivos guardados en: %s", args.output_dir)

    # ── Siguiente paso ──
    fase2_path = _PIPELINE_DIR / "fase2_ablation_router.py"
    log.info("Siguiente paso: python %s --embeddings %s",
             fase2_path, args.output_dir)


# ══════════════════════════════════════════════════════════════
#  Argparse
# ══════════════════════════════════════════════════════════════

def _build_parser():
    """Define los argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="FASE 1 — Extracción de embeddings MoE",
    )

    # ── Backbone ──
    parser.add_argument(
        "--backbone",
        default="vit_tiny_patch16_224",
        choices=list(BACKBONE_CONFIGS.keys()),
        help="Backbone para extraer CLS tokens",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers",    type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--output_dir", default="./embeddings",
        help="Carpeta para Z_{train,val,test}.npy y backbone_meta.json",
    )

    # ── NIH ChestXray14 ──
    parser.add_argument("--chest_csv",  default=None)
    parser.add_argument("--chest_imgs", default=None,
                        help="Directorio con imágenes (all_images/)")
    parser.add_argument(
        "--nih_train_list", default=None,
        help="splits/nih_train_list.txt generado por Fase 0",
    )
    parser.add_argument(
        "--nih_val_list", default=None,
        help="splits/nih_val_list.txt generado por Fase 0",
    )
    parser.add_argument(
        "--nih_test_list", default=None,
        help="splits/nih_test_list.txt generado por Fase 0",
    )
    parser.add_argument(
        "--chest_view_filter", default=None, choices=["PA", "AP"],
        help="Filtrar por vista radiográfica",
    )
    parser.add_argument("--chest_bbox_csv", default=None,
                        help="BBox_List_2017.csv para validación de heatmaps")

    # ── ISIC 2019 ──
    parser.add_argument(
        "--isic_train_csv", default=None,
        help="splits/isic_train.csv generado por Fase 0",
    )
    parser.add_argument(
        "--isic_val_csv", default=None,
        help="splits/isic_val.csv generado por Fase 0",
    )
    parser.add_argument(
        "--isic_test_csv", default=None,
        help="splits/isic_test.csv generado por Fase 0",
    )
    parser.add_argument("--isic_imgs", default=None,
                        help="Carpeta con imágenes ISIC .jpg")

    # ── OA Rodilla ──
    parser.add_argument(
        "--oa_root", default=None,
        help="Directorio oa_splits/ generado por fase0/pre_modelo.py",
    )

    # ── LUNA16 ──
    parser.add_argument(
        "--luna_patches_dir", default=None,
        help="Directorio patches/ con subdirs train/, val/, test/",
    )
    parser.add_argument("--luna_csv", default=None,
                        help="candidates_V2.csv")

    # ── Páncreas PANORAMA ──
    parser.add_argument(
        "--pancreas_splits_csv", default=None,
        help="pancreas_splits.csv generado por Fase 0",
    )
    parser.add_argument(
        "--pancreas_nii_dir", default=None,
        help="Directorio con archivos .nii.gz de Zenodo",
    )
    parser.add_argument(
        "--pancreas_fold", type=int, default=PANCREAS_FOLD,
        help="Fold del k-fold CV para train/val (1-5, default: %(default)s)",
    )
    parser.add_argument(
        "--pancreas_roi_strategy", default="A", choices=["A", "B"],
        help="Estrategia ROI: A=resize completo, B=recorte Z (Fase 3)",
    )

    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()
    main(_args)

"""
fase1_pipeline.py — Orquestador de Fase 1: Extracción de Embeddings
===================================================================

Responsabilidad única: definir el orden de ejecución de todos los módulos,
gestionar argumentos CLI y producir el reporte final.

No contiene lógica de negocio propia — solo llama funciones.
Nunca implementa transformaciones ni escribe arrays directamente.

Mejoras implementadas:
  - Idempotencia: si los embeddings ya existen, se omite la extracción
    (configurable via --force para forzar re-extracción)
  - Reporte markdown: genera fase1_report.md con métricas detalladas

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
import datetime
from pathlib import Path
from collections import Counter

import pandas as pd

# ── Path setup ──────────────────────────────────────────────
# Permite ejecutar como script: python src/pipeline/fase1/fase1_pipeline.py
_THIS_DIR = Path(__file__).resolve().parent  # src/pipeline/fase1/
_PIPELINE_DIR = _THIS_DIR.parent  # src/pipeline/
_SRC_DIR = _PIPELINE_DIR.parent  # src/
_PROJECT_ROOT = _SRC_DIR.parent  # proyecto_2/

for _p in [str(_THIS_DIR), str(_PIPELINE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Imports de fase1 (via sys.path → fase1/) ───────────────
from fase1_config import (
    BACKBONE_CONFIGS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_WORKERS,
    PANCREAS_FOLD,
    HU_ABDOMEN_CLIP,
    MIN_L2_NORM,
)
import backbone_cvt13  # noqa: F401 — activar registro en timm
import backbone_densenet  # noqa: F401 — activar registro de DenseNet-121 custom
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
#  Verificación de idempotencia
# ══════════════════════════════════════════════════════════════


def _embeddings_exist(output_dir, backbone_name):
    """
    Verifica si los archivos de embeddings ya existen para el backbone dado.

    Comprueba la existencia de todos los archivos críticos que genera
    save_embeddings(): los arrays .npy, los nombres .txt y el metadato JSON.

    Args:
        output_dir: directorio de salida de embeddings
        backbone_name: nombre del backbone (para validar backbone_meta.json)

    Returns:
        True si todos los archivos existen y el backbone coincide, False en caso contrario
    """
    import json

    out = Path(output_dir)
    # Archivos mínimos requeridos para considerar la extracción completa
    required_files = [
        "Z_train.npy",
        "y_train.npy",
        "Z_val.npy",
        "y_val.npy",
        "Z_test.npy",
        "y_test.npy",
        "backbone_meta.json",
    ]
    for fname in required_files:
        if not (out / fname).exists():
            return False

    # Verificar que el backbone en el metadato coincide con el solicitado
    try:
        meta_path = out / "backbone_meta.json"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("backbone") != backbone_name:
            log.info(
                "[Idempotencia] backbone_meta.json existe pero con backbone '%s' "
                "(solicitado: '%s') — se re-extraerá",
                meta.get("backbone"),
                backbone_name,
            )
            return False
    except (json.JSONDecodeError, OSError) as e:
        log.warning("[Idempotencia] Error leyendo backbone_meta.json: %s", e)
        return False

    return True


# ══════════════════════════════════════════════════════════════
#  Generación del reporte Markdown
# ══════════════════════════════════════════════════════════════


def _compute_class_distribution(y_array, expert_ids):
    """
    Calcula la distribución de clases (expertos) en un array de etiquetas.

    Args:
        y_array: array de etiquetas de experto (int64)
        expert_ids: dict {nombre_experto: id_experto}

    Returns:
        dict {nombre_experto: conteo} ordenado por id de experto
    """
    if len(y_array) == 0:
        return {}
    counts = Counter(y_array.tolist())
    # Mapeo inverso: id → nombre
    id_to_name = {v: k for k, v in expert_ids.items()}
    distribution = {}
    for exp_id in sorted(id_to_name.keys()):
        name = id_to_name[exp_id]
        distribution[name] = counts.get(exp_id, 0)
    return distribution


def _generate_report(report_data, output_path):
    """
    Genera el archivo fase1_report.md con métricas detalladas de la ejecución.

    Args:
        report_data: dict acumulador con todas las métricas del pipeline
        output_path: ruta donde escribir el archivo .md
    """
    lines = []
    lines.append("# Fase 1 — Reporte de Extracción de Embeddings")
    lines.append("")
    lines.append(f"**Fecha de ejecución:** {report_data['timestamp']}")
    lines.append(
        f"**Tiempo total:** {report_data['total_time_s']:.1f}s "
        f"({report_data['total_time_s'] / 60:.1f} min)"
    )
    lines.append(f"**Dispositivo:** {report_data['device']}")
    lines.append(f"**Backbone:** `{report_data['backbone']}`")
    lines.append(f"**Directorio de salida:** `{report_data['output_dir']}`")
    lines.append("")

    # Estado de idempotencia
    if report_data.get("skipped"):
        lines.append("## Estado: OMITIDO (idempotencia)")
        lines.append("")
        lines.append(
            "Los embeddings ya existían en disco y `--force` no fue "
            "proporcionado. No se realizó extracción."
        )
        lines.append("")
    else:
        lines.append("## Resumen de Embeddings")
        lines.append("")
        lines.append("| Split | Muestras | Dimensión | Tamaño (MB) | Tiempo (s) |")
        lines.append("|-------|----------|-----------|-------------|------------|")

        for split_name in ("train", "val", "test"):
            info = report_data.get(f"split_{split_name}", {})
            n_samples = info.get("n_samples", 0)
            d_model = info.get("d_model", "—")
            size_mb = info.get("size_mb", 0.0)
            elapsed = info.get("time_s", 0.0)
            lines.append(
                f"| {split_name:5s} | {n_samples:>8,} | {d_model:>9} "
                f"| {size_mb:>11.1f} | {elapsed:>10.1f} |"
            )
        lines.append("")

        # Distribución por clase (por split)
        lines.append("## Distribución por Experto")
        lines.append("")
        for split_name in ("train", "val", "test"):
            info = report_data.get(f"split_{split_name}", {})
            dist = info.get("distribution", {})
            if not dist:
                continue
            lines.append(f"### {split_name.upper()}")
            lines.append("")
            lines.append("| Experto | ID | Muestras | Porcentaje |")
            lines.append("|---------|----|---------:|------------|")
            total = sum(dist.values())
            for exp_name, count in dist.items():
                exp_id = report_data.get("expert_ids", {}).get(exp_name, "?")
                pct = (count / total * 100) if total > 0 else 0.0
                lines.append(
                    f"| {exp_name:<10s} | {exp_id} | {count:>8,} | {pct:>9.1f}% |"
                )
            lines.append("")

    # Errores encontrados
    errors = report_data.get("errors", [])
    if errors:
        lines.append("## Errores Encontrados")
        lines.append("")
        for err in errors:
            lines.append(f"- {err}")
        lines.append("")
    else:
        lines.append("## Errores")
        lines.append("")
        lines.append("Sin errores detectados durante la ejecución. ✓")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generado automáticamente por fase1_pipeline.py*")

    # Escribir archivo
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("[Reporte] fase1_report.md escrito en: %s", report_path)


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
    t_pipeline_start = time.time()
    log_obj = setup_logging(args.output_dir, phase_name="fase1")

    # ── Acumulador para el reporte markdown ──
    report_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": args.backbone,
        "output_dir": args.output_dir,
        "device": "pendiente",
        "expert_ids": EXPERT_IDS,
        "errors": [],
        "skipped": False,
        "total_time_s": 0.0,
    }

    log.info("=" * 60)
    log.info("FASE 1 — Extracción de CLS tokens (Embeddings)")
    log.info("=" * 60)
    log.info("Argumentos: %s", vars(args))

    # ── 0. Verificación de idempotencia ──
    force = getattr(args, "force", False)
    if not force and _embeddings_exist(args.output_dir, args.backbone):
        log.info(
            "[SKIP] Embeddings ya existen para backbone '%s' en '%s' — "
            "omitiendo extracción. Usa --force para re-extraer.",
            args.backbone,
            args.output_dir,
        )
        report_data["skipped"] = True
        report_data["device"] = "N/A (omitido)"
        report_data["total_time_s"] = time.time() - t_pipeline_start

        # Generar reporte incluso cuando se omite
        report_path = Path(args.output_dir) / "fase1_report.md"
        _generate_report(report_data, report_path)
        return

    # ── 1. Guard clause: LUNA16 patches ──
    _check_luna_patches(args.luna_patches_dir)

    # ── 2. Guard clause: artefactos de Fase 0 ──
    cfg = {
        "chest_csv": args.chest_csv,
        "chest_imgs": args.chest_imgs,
        "nih_train_list": args.nih_train_list,
        "nih_val_list": args.nih_val_list,
        "nih_test_list": args.nih_test_list,
        "chest_view_filter": args.chest_view_filter,
        "chest_bbox_csv": args.chest_bbox_csv,
        "isic_train_csv": args.isic_train_csv,
        "isic_val_csv": args.isic_val_csv,
        "isic_test_csv": args.isic_test_csv,
        "isic_imgs": args.isic_imgs,
        "oa_root": args.oa_root,
        "luna_patches_dir": args.luna_patches_dir,
        "luna_csv": args.luna_csv,
        "pancreas_splits_csv": args.pancreas_splits_csv,
        "pancreas_nii_dir": args.pancreas_nii_dir,
        "pancreas_fold": args.pancreas_fold,
        "pancreas_roi_strategy": args.pancreas_roi_strategy,
        "output_dir": args.output_dir,
    }
    _check_fase0_artifacts(cfg)

    # ── 3. Setup de dispositivo ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report_data["device"] = device
    log.info("[Setup] Dispositivo: %s", device)
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        log.info("[Setup] GPU           : %s", gpu_name)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("[Setup] VRAM total    : %.1f GB", total_vram)
        report_data["device"] = f"cuda ({gpu_name}, {total_vram:.1f} GB)"
        expected_vram = BACKBONE_CONFIGS[args.backbone]["vram_gb"]
        if expected_vram > total_vram * 0.8:
            log.warning(
                "[Setup] Backbone '%s' requiere ~%.0f GB pero hay %.1f GB. "
                "Considera reducir --batch_size.",
                args.backbone,
                expected_vram,
                total_vram,
            )
    else:
        log.warning("[Setup] CPU — la extracción será lenta. Se recomienda GPU.")

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
    val_loader = DataLoader(val_dataset, **_dl_kw)
    test_loader = DataLoader(test_dataset, **_dl_kw) if len(test_dataset) > 0 else None

    # ── 7. Extraer embeddings por split ──
    os.makedirs(args.output_dir, exist_ok=True)

    # Diccionario para acumular resultados de cada split
    split_results = {}

    # --- TRAIN ---
    log.info("[FASE 1] Extrayendo embeddings de TRAIN...")
    t0 = time.time()
    Z_train, y_train, names_train = extract_embeddings(
        model,
        train_loader,
        device,
        d_model,
        desc="train",
    )
    t_train = time.time() - t0
    log.info("[FASE 1] TRAIN completado en %.1fs", t_train)
    report_data["split_train"] = {
        "n_samples": Z_train.shape[0],
        "d_model": d_model,
        "size_mb": Z_train.nbytes / 1e6,
        "time_s": t_train,
        "distribution": _compute_class_distribution(y_train, EXPERT_IDS),
    }

    # --- VAL ---
    log.info("[FASE 1] Extrayendo embeddings de VAL...")
    t0 = time.time()
    Z_val, y_val, names_val = extract_embeddings(
        model,
        val_loader,
        device,
        d_model,
        desc="val",
    )
    t_val = time.time() - t0
    log.info("[FASE 1] VAL completado en %.1fs", t_val)
    report_data["split_val"] = {
        "n_samples": Z_val.shape[0],
        "d_model": d_model,
        "size_mb": Z_val.nbytes / 1e6,
        "time_s": t_val,
        "distribution": _compute_class_distribution(y_val, EXPERT_IDS),
    }

    # --- TEST ---
    if test_loader is not None:
        log.info("[FASE 1] Extrayendo embeddings de TEST...")
        t0 = time.time()
        Z_test, y_test, names_test = extract_embeddings(
            model,
            test_loader,
            device,
            d_model,
            desc="test",
        )
        t_test = time.time() - t0
        log.info("[FASE 1] TEST completado en %.1fs", t_test)
    else:
        Z_test = np.zeros((0, d_model), dtype=np.float32)
        y_test = np.zeros(0, dtype=np.int64)
        names_test = []
        t_test = 0.0
        log.warning("[FASE 1] Sin datos de test — extracción omitida.")

    report_data["split_test"] = {
        "n_samples": Z_test.shape[0],
        "d_model": d_model,
        "size_mb": Z_test.nbytes / 1e6,
        "time_s": t_test,
        "distribution": _compute_class_distribution(y_test, EXPERT_IDS),
    }

    # ── 8. Guardar en disco ──
    try:
        save_embeddings(
            args.output_dir,
            args.backbone,
            Z_train,
            y_train,
            names_train,
            Z_val,
            y_val,
            names_val,
            Z_test,
            y_test,
            names_test,
        )
    except Exception as e:
        error_msg = f"Error guardando embeddings: {e}"
        log.error("[FASE 1] %s", error_msg)
        report_data["errors"].append(error_msg)

    # ── 9. Reporte final en log ──
    log.info("=" * 55)
    log.info("FASE 1 completada")
    log.info("=" * 55)
    log.info("Z_train : %s  (%.1f MB)", Z_train.shape, Z_train.nbytes / 1e6)
    log.info("Z_val   : %s  (%.1f MB)", Z_val.shape, Z_val.nbytes / 1e6)
    log.info("Z_test  : %s  (%.1f MB)", Z_test.shape, Z_test.nbytes / 1e6)

    log_distribution(y_train, y_val, y_test, EXPERT_IDS)

    log.info(
        "  [Experto 5 se inicializa en FASE 2 como MLP OOD — entrenado via L_error]"
    )
    log.info(
        "  N_EXPERTS_DOMAIN = %d | N_EXPERTS_TOTAL = %d",
        N_EXPERTS_DOMAIN,
        N_EXPERTS_TOTAL,
    )
    log.info("Archivos guardados en: %s", args.output_dir)

    # ── 10. Generar reporte markdown ──
    report_data["total_time_s"] = time.time() - t_pipeline_start
    report_path = Path(args.output_dir) / "fase1_report.md"
    try:
        _generate_report(report_data, report_path)
    except Exception as e:
        log.error("[Reporte] Error generando fase1_report.md: %s", e)

    # ── Siguiente paso ──
    fase2_path = _PIPELINE_DIR / "fase2" / "fase2_pipeline.py"
    log.info("Siguiente paso: python %s --embeddings %s", fase2_path, args.output_dir)


# ══════════════════════════════════════════════════════════════
#  Argparse
# ══════════════════════════════════════════════════════════════


def _build_parser():
    """Define los argumentos CLI.

    Los defaults apuntan a las rutas estándar del proyecto cuando existen
    en disco. Si una ruta no existe, el default queda en None y el dataset
    correspondiente se omite automáticamente (comportamiento existente).

    Flags útiles:
      --dry-run   Muestra configuración resuelta y datasets detectados
                  sin ejecutar la extracción de embeddings.
      --force     Fuerza re-extracción aunque los embeddings ya existan.
    """
    # ── Defaults inteligentes: solo se aplican si la ruta existe ──
    _DS = _PROJECT_ROOT / "datasets"

    def _default_if_exists(rel_path):
        """Devuelve la ruta absoluta como str si existe, None si no."""
        full = _DS / rel_path
        return str(full) if full.exists() else None

    parser = argparse.ArgumentParser(
        description="FASE 1 — Extracción de embeddings MoE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos de uso:\n"
            "  # Ejecución con defaults detectados automáticamente:\n"
            "  python fase1_pipeline.py\n\n"
            "  # Ver qué se ejecutaría sin extraer nada:\n"
            "  python fase1_pipeline.py --dry-run\n\n"
            "  # Forzar re-extracción con backbone específico:\n"
            "  python fase1_pipeline.py --backbone cvt_13 --force\n"
        ),
    )

    # ── Backbone ──
    parser.add_argument(
        "--backbone",
        default="vit_tiny_patch16_224",
        choices=list(BACKBONE_CONFIGS.keys()),
        help="Backbone para extraer CLS tokens",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--output_dir",
        default="./embeddings",
        help="Carpeta para Z_{train,val,test}.npy y backbone_meta.json",
    )
    # ── Idempotencia ──
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Forzar re-extracción aunque los embeddings ya existan en disco",
    )
    # ── Dry-run ──
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help=(
            "Modo simulación: muestra la configuración resuelta, "
            "datasets detectados y archivos de embeddings existentes, "
            "pero NO ejecuta la extracción. Útil para verificar que "
            "todo esté correcto antes de lanzar el proceso real."
        ),
    )

    # ── NIH ChestXray14 ──
    parser.add_argument(
        "--chest_csv",
        default=_default_if_exists("nih_chest_xrays/Data_Entry_2017.csv"),
        help="CSV de metadatos NIH (default: datasets/nih_chest_xrays/Data_Entry_2017.csv)",
    )
    parser.add_argument(
        "--chest_imgs",
        default=_default_if_exists("nih_chest_xrays/all_images"),
        help="Directorio con imágenes (all_images/)",
    )
    parser.add_argument(
        "--nih_train_list",
        default=_default_if_exists("nih_chest_xrays/splits/nih_train_list.txt"),
        help="splits/nih_train_list.txt generado por Fase 0",
    )
    parser.add_argument(
        "--nih_val_list",
        default=_default_if_exists("nih_chest_xrays/splits/nih_val_list.txt"),
        help="splits/nih_val_list.txt generado por Fase 0",
    )
    parser.add_argument(
        "--nih_test_list",
        default=_default_if_exists("nih_chest_xrays/splits/nih_test_list.txt"),
        help="splits/nih_test_list.txt generado por Fase 0",
    )
    parser.add_argument(
        "--chest_view_filter",
        default=None,
        choices=["PA", "AP"],
        help="Filtrar por vista radiográfica",
    )
    parser.add_argument(
        "--chest_bbox_csv",
        default=_default_if_exists("nih_chest_xrays/BBox_List_2017.csv"),
        help="BBox_List_2017.csv para validación de heatmaps",
    )

    # ── ISIC 2019 ──
    parser.add_argument(
        "--isic_train_csv",
        default=_default_if_exists("isic_2019/splits/isic_train.csv"),
        help="splits/isic_train.csv generado por Fase 0",
    )
    parser.add_argument(
        "--isic_val_csv",
        default=_default_if_exists("isic_2019/splits/isic_val.csv"),
        help="splits/isic_val.csv generado por Fase 0",
    )
    parser.add_argument(
        "--isic_test_csv",
        default=_default_if_exists("isic_2019/splits/isic_test.csv"),
        help="splits/isic_test.csv generado por Fase 0",
    )
    parser.add_argument(
        "--isic_imgs",
        default=_default_if_exists("isic_2019/ISIC_2019_Training_Input"),
        help="Carpeta con imágenes ISIC .jpg",
    )

    # ── OA Rodilla ──
    parser.add_argument(
        "--oa_root",
        default=_default_if_exists("osteoarthritis/oa_splits"),
        help="Directorio oa_splits/ generado por fase0/pre_modelo.py",
    )

    # ── LUNA16 ──
    parser.add_argument(
        "--luna_patches_dir",
        default=_default_if_exists("luna_lung_cancer/patches"),
        help="Directorio patches/ con subdirs train/, val/, test/",
    )
    parser.add_argument(
        "--luna_csv",
        default=_default_if_exists("luna_lung_cancer/candidates_V2/candidates_V2.csv"),
        help="candidates_V2.csv (ruta completa)",
    )

    # ── Páncreas PANORAMA ──
    parser.add_argument(
        "--pancreas_splits_csv",
        default=_default_if_exists("pancreas_splits.csv"),
        help="pancreas_splits.csv generado por Fase 0",
    )
    parser.add_argument(
        "--pancreas_nii_dir",
        default=_default_if_exists("zenodo_13715870"),
        help="Directorio con archivos .nii.gz de Zenodo",
    )
    parser.add_argument(
        "--pancreas_fold",
        type=int,
        default=PANCREAS_FOLD,
        help="Fold del k-fold CV para train/val (1-5, default: %(default)s)",
    )
    parser.add_argument(
        "--pancreas_roi_strategy",
        default="A",
        choices=["A", "B"],
        help="Estrategia ROI: A=resize completo, B=recorte Z (Fase 3)",
    )

    return parser


# ══════════════════════════════════════════════════════════════
#  Dry-run: resumen informativo sin ejecutar extracción
# ══════════════════════════════════════════════════════════════


def _count_files_in_dir(dir_path, extensions=None):
    """Cuenta archivos en un directorio (no recursivo). Retorna 0 si no existe."""
    d = Path(dir_path)
    if not d.is_dir():
        return 0
    if extensions:
        return sum(
            1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in extensions
        )
    return sum(1 for f in d.iterdir() if f.is_file())


def _count_lines(filepath):
    """Cuenta líneas de un archivo de texto. Retorna 0 si no existe."""
    p = Path(filepath)
    if not p.is_file():
        return 0
    with open(p, encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def _print_dry_run_summary(args):
    """Imprime un resumen completo de configuración sin ejecutar extracción.

    Incluye:
      - Argumentos resueltos (rutas con defaults aplicados)
      - Datasets que se cargarían vs. omitidos
      - Muestras aproximadas por split
      - Backbone seleccionado y configuración
      - Archivos de embeddings existentes (idempotencia)
    """
    import json

    log.info("=" * 60)
    log.info("[DRY-RUN] Resumen de configuración")
    log.info("=" * 60)

    # ── 1. Argumentos resueltos ──
    log.info("")
    log.info("┌─ Argumentos resueltos ─────────────────────────────────┐")
    for k, v in sorted(vars(args).items()):
        status = "✓" if v is not None else "✗ (None)"
        log.info("  %-22s = %s  %s", k, v, "" if v is not None else status)
    log.info("└────────────────────────────────────────────────────────┘")

    # ── 2. Backbone ──
    bb_cfg = BACKBONE_CONFIGS.get(args.backbone, {})
    log.info("")
    log.info("┌─ Backbone ─────────────────────────────────────────────┐")
    log.info("  Nombre  : %s", args.backbone)
    log.info("  d_model : %s", bb_cfg.get("d_model", "?"))
    log.info("  VRAM est: %.1f GB", bb_cfg.get("vram_gb", 0))
    log.info("└────────────────────────────────────────────────────────┘")

    # ── 3. Detección de datasets ──
    log.info("")
    log.info("┌─ Datasets detectados ──────────────────────────────────┐")

    datasets_info = []

    # --- NIH ChestXray14 ---
    nih_ready = all([args.nih_train_list, args.chest_csv, args.chest_imgs])
    if nih_ready:
        n_train = _count_lines(args.nih_train_list)
        n_val = _count_lines(args.nih_val_list) if args.nih_val_list else 0
        n_test = _count_lines(args.nih_test_list) if args.nih_test_list else 0
        log.info("  ✓ NIH ChestXray14 (Experto 0)")
        log.info("      train: ~%d  val: ~%d  test: ~%d", n_train, n_val, n_test)
        datasets_info.append(("chest", n_train, n_val, n_test))
    else:
        log.info("  ✗ NIH ChestXray14 — OMITIDO (faltan rutas)")
        missing = []
        if not args.chest_csv:
            missing.append("--chest_csv")
        if not args.chest_imgs:
            missing.append("--chest_imgs")
        if not args.nih_train_list:
            missing.append("--nih_train_list")
        log.info("      Faltan: %s", ", ".join(missing))

    # --- ISIC 2019 ---
    isic_ready = all([args.isic_train_csv, args.isic_imgs])
    if isic_ready:
        # Conteo real: leer CSVs y descontar imágenes _downsampled que no existen en disco
        _isic_counts = {}
        for _sname, _spath in [
            ("train", args.isic_train_csv),
            ("val", args.isic_val_csv),
            ("test", args.isic_test_csv),
        ]:
            if _spath:
                _sdf = pd.read_csv(_spath)
                _antes = len(_sdf)
                _sdf = _sdf[~_sdf["image"].str.contains("_downsampled", na=False)]
                _elim = _antes - len(_sdf)
                _isic_counts[_sname] = len(_sdf)
                if _elim > 0:
                    log.info(
                        "      [filtro] %s: %d _downsampled excluidas", _sname, _elim
                    )
            else:
                _isic_counts[_sname] = 0
        n_train = _isic_counts["train"]
        n_val = _isic_counts["val"]
        n_test = _isic_counts["test"]
        log.info("  ✓ ISIC 2019 (Experto 1)")
        log.info("      train: ~%d  val: ~%d  test: ~%d", n_train, n_val, n_test)
        datasets_info.append(("isic", n_train, n_val, n_test))
    else:
        log.info("  ✗ ISIC 2019 — OMITIDO (faltan rutas)")

    # --- OA Rodilla ---
    if args.oa_root:
        oa_base = Path(args.oa_root)
        n_train = _count_files_in_dir(oa_base / "train", {".png", ".jpg", ".jpeg"})
        n_val = _count_files_in_dir(oa_base / "val", {".png", ".jpg", ".jpeg"})
        n_test = _count_files_in_dir(oa_base / "test", {".png", ".jpg", ".jpeg"})
        # OA puede tener subdirectorios por clase
        if n_train == 0:
            for sub in (oa_base / "train").iterdir():
                if sub.is_dir():
                    n_train += _count_files_in_dir(sub, {".png", ".jpg", ".jpeg"})
            for sub in (oa_base / "val").iterdir():
                if sub.is_dir():
                    n_val += _count_files_in_dir(sub, {".png", ".jpg", ".jpeg"})
            for sub in (oa_base / "test").iterdir():
                if sub.is_dir():
                    n_test += _count_files_in_dir(sub, {".png", ".jpg", ".jpeg"})
        log.info("  ✓ OA Rodilla (Experto 2)")
        log.info("      train: ~%d  val: ~%d  test: ~%d", n_train, n_val, n_test)
        datasets_info.append(("oa", n_train, n_val, n_test))
    else:
        log.info("  ✗ OA Rodilla — OMITIDO (falta --oa_root)")

    # --- LUNA16 ---
    luna_ready = all([args.luna_patches_dir, args.luna_csv])
    if luna_ready:
        luna_base = Path(args.luna_patches_dir)
        n_train = _count_files_in_dir(luna_base / "train", {".npy"})
        n_val = _count_files_in_dir(luna_base / "val", {".npy"})
        n_test = _count_files_in_dir(luna_base / "test", {".npy"})
        log.info("  ✓ LUNA16 (Experto 3)")
        log.info("      train: ~%d  val: ~%d  test: ~%d", n_train, n_val, n_test)
        datasets_info.append(("luna", n_train, n_val, n_test))
    else:
        log.info("  ✗ LUNA16 — OMITIDO (faltan rutas)")

    # --- Páncreas ---
    panc_ready = all([args.pancreas_splits_csv, args.pancreas_nii_dir])
    if panc_ready:
        import csv

        try:
            n_train = n_val = n_test = 0
            fold = args.pancreas_fold
            with open(args.pancreas_splits_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    split = row.get("split", "")
                    if split == f"fold{fold}_train":
                        n_train += 1
                    elif split == f"fold{fold}_val":
                        n_val += 1
                    elif split == "test":
                        n_test += 1
            log.info("  ✓ Páncreas PANORAMA (Experto 4, fold %d)", fold)
            log.info("      train: ~%d  val: ~%d  test: ~%d", n_train, n_val, n_test)
            datasets_info.append(("pancreas", n_train, n_val, n_test))
        except Exception as e:
            log.info("  ⚠ Páncreas — error leyendo CSV: %s", e)
    else:
        log.info("  ✗ Páncreas — OMITIDO (faltan rutas)")

    log.info("  ─ Experto 5 (OOD) — sin dataset en Fase 1")

    # Resumen total
    total_train = sum(d[1] for d in datasets_info)
    total_val = sum(d[2] for d in datasets_info)
    total_test = sum(d[3] for d in datasets_info)
    log.info("  ─────────────────────────────────────────")
    log.info(
        "  TOTAL aprox: train=%d  val=%d  test=%d", total_train, total_val, total_test
    )
    log.info("  Datasets activos: %d/5", len(datasets_info))
    log.info("└────────────────────────────────────────────────────────┘")

    if not datasets_info:
        log.error("  ¡NINGÚN dataset sería cargado! Verifica las rutas.")

    # ── 4. Idempotencia: archivos de embeddings existentes ──
    log.info("")
    log.info("┌─ Estado de embeddings (idempotencia) ───────────────────┐")
    out = Path(args.output_dir)
    expected_files = [
        "Z_train.npy",
        "y_train.npy",
        "names_train.txt",
        "Z_val.npy",
        "y_val.npy",
        "names_val.txt",
        "Z_test.npy",
        "y_test.npy",
        "names_test.txt",
        "backbone_meta.json",
    ]
    any_exist = False
    for fname in expected_files:
        fpath = out / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1e6
            log.info("  ✓ %s (%.1f MB) — YA EXISTE", fname, size_mb)
            any_exist = True
        else:
            log.info("  · %s — se generaría", fname)

    if any_exist and not args.force:
        already_done = _embeddings_exist(args.output_dir, args.backbone)
        if already_done:
            log.info("  → Los embeddings ya están completos. Sin --force se omitirían.")
        else:
            log.info("  → Embeddings incompletos o de otro backbone. Se re-extraerían.")
    elif args.force:
        log.info("  → --force activo: se re-extraerían todos los archivos.")
    else:
        log.info("  → No hay embeddings previos. Se generarían desde cero.")
    log.info("└────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    # ── Dry-run: mostrar resumen y salir sin ejecutar ──
    if _args.dry_run:
        # Configurar logging mínimo para el dry-run
        setup_logging(_args.output_dir, phase_name="fase1")
        _print_dry_run_summary(_args)
        log.info("")
        log.info("[DRY-RUN] Fin. Sin cambios en disco.")
        sys.exit(0)

    main(_args)

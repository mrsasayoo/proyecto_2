"""
fase1_train_pipeline.py — Orquestador del Paso 4.1: Entrenamiento End-to-End
============================================================================

Entrena cada backbone desde cero sobre los 5 datasets de dominio usando
clasificación de dominio (expert_id) como tarea proxy.

Después del entrenamiento, el backbone congelado se usa en Paso 4.2
(fase1_pipeline.py) para extraer CLS tokens.

Uso:
    # Verificar sin entrenar:
    python src/pipeline/fase1/fase1_train_pipeline.py --backbone vit_tiny_patch16_224 --dry-run

    # Entrenar (producción):
    python src/pipeline/fase1/fase1_train_pipeline.py --backbone vit_tiny_patch16_224
"""

import os
import sys
import argparse
import time
import logging
import datetime
import json
from pathlib import Path

import pandas as pd

# ── Path setup (IDÉNTICO a fase1_pipeline.py) ──────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _THIS_DIR.parent
_SRC_DIR = _PIPELINE_DIR.parent
_PROJECT_ROOT = _SRC_DIR.parent

for _p in [str(_THIS_DIR), str(_PIPELINE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Imports de fase1 ───────────────────────────────────────────
from fase1_config import (
    BACKBONE_CONFIGS,
    BACKBONE_TO_CHECKPOINT_DIR,
    BACKBONE_CHECKPOINT_FILENAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_WORKERS,
    PANCREAS_FOLD,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_WEIGHT_DECAY,
    TRAIN_WARMUP_EPOCHS,
    TRAIN_BATCH_SIZE,
    TRAIN_WORKERS,
)
import backbone_cvt13  # noqa: F401 — registrar interceptor timm
import backbone_densenet  # noqa: F401 — registrar interceptor timm
from backbone_loader import load_trainable_backbone
from backbone_trainer import LinearHead, train_backbone, backbone_checkpoint_exists
from dataset_builder import build_datasets

# ── Imports de pipeline global ──────────────────────────────────
from config import EXPERT_IDS, N_EXPERTS_DOMAIN
from logging_utils import setup_logging

import torch
from torch.utils.data import DataLoader

log = logging.getLogger("fase1")


# ══════════════════════════════════════════════════════════════
#  Ruta canónica del checkpoint para un backbone dado
# ══════════════════════════════════════════════════════════════


def _get_checkpoint_path(backbone_name: str, checkpoint_base_dir: str) -> str:
    """
    Devuelve la ruta completa del archivo backbone.pth para un backbone dado.

    Ejemplo:
        backbone_name = "vit_tiny_patch16_224"
        → checkpoints/backbone_01_vit_tiny/backbone.pth
    """
    subdir = BACKBONE_TO_CHECKPOINT_DIR[backbone_name]
    return str(Path(checkpoint_base_dir) / subdir / BACKBONE_CHECKPOINT_FILENAME)


# ══════════════════════════════════════════════════════════════
#  Detección de dispositivo (idéntico a fase1_pipeline.py)
# ══════════════════════════════════════════════════════════════


def _detect_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("[Setup] GPU: %s (%.1f GB VRAM)", name, vram)
        return device, "gpu"
    else:
        n = os.cpu_count() or 4
        n_th = max(1, n // 2)
        torch.set_num_threads(n_th)
        try:
            torch.set_num_interop_threads(max(1, n_th // 2))
        except RuntimeError:
            pass
        log.info("[Setup] CPU: %d cores → torch threads: %d", n, n_th)
        return torch.device("cpu"), "cpu"


# ══════════════════════════════════════════════════════════════
#  Dry-run
# ══════════════════════════════════════════════════════════════


def _print_dry_run_summary(args):
    """
    Modo simulación: instanciar modelo + head, un forward pass con un batch sintético,
    imprimir configuración completa. NO entrena, NO modifica disco.
    """
    import torch.nn as nn

    log.info("=" * 60)
    log.info("[DRY-RUN] Paso 4.1 — Entrenamiento End-to-End de Backbones")
    log.info("=" * 60)

    device, device_type = _detect_device()

    # 1. Configuración
    checkpoint_path = _get_checkpoint_path(args.backbone, args.checkpoint_dir)
    bb_cfg = BACKBONE_CONFIGS[args.backbone]
    log.info("")
    log.info("┌─ Configuración de entrenamiento ────────────────────────┐")
    log.info("  Backbone         : %s", args.backbone)
    log.info("  d_model          : %d", bb_cfg["d_model"])
    log.info("  VRAM estimada    : %.1f GB", bb_cfg["vram_gb"])
    log.info("  Epochs           : %d", args.epochs)
    log.info("  LR               : %.2e", args.lr)
    log.info("  Weight decay     : %.4f", args.weight_decay)
    log.info("  Warmup epochs    : %d", args.warmup_epochs)
    log.info("  Batch size       : %d", args.batch_size)
    log.info("  Workers          : %d", args.workers)
    log.info("  Dispositivo      : %s (%s)", device, device_type)
    log.info("  Checkpoint → %s", checkpoint_path)
    ckpt_exists = backbone_checkpoint_exists(checkpoint_path)
    log.info("  Checkpoint existe: %s", "SÍ ✓" if ckpt_exists else "NO (se crearía)")
    if ckpt_exists and not args.force:
        log.info("  → Con --force=False, se SALTARÍA el entrenamiento.")
    log.info("└────────────────────────────────────────────────────────┘")

    # 2. Backbone + Head
    log.info("")
    log.info("┌─ Modelo ───────────────────────────────────────────────┐")
    backbone, d_model = load_trainable_backbone(args.backbone, device)
    head = LinearHead(d_model, N_EXPERTS_DOMAIN).to(device)
    n_bb = sum(p.numel() for p in backbone.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    log.info("  Backbone params  : %s", f"{n_bb:,}")
    log.info(
        "  Head params      : %s  [Linear(%d→%d)]",
        f"{n_head:,}",
        d_model,
        N_EXPERTS_DOMAIN,
    )
    log.info("  Total params     : %s", f"{n_bb + n_head:,}")
    log.info("└────────────────────────────────────────────────────────┘")

    # 3. Forward pass sintético (UN batch)
    log.info("")
    log.info("┌─ Forward pass sintético (1 batch) ─────────────────────┐")
    dummy_imgs = torch.randn(min(args.batch_size, 4), 3, 224, 224, device=device)
    dummy_labels = torch.randint(
        0, N_EXPERTS_DOMAIN, (min(args.batch_size, 4),), device=device
    )
    backbone.eval()
    head.eval()
    with torch.no_grad():
        z = backbone(dummy_imgs)
        logits = head(z)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, dummy_labels)
    log.info("  Input  : %s", list(dummy_imgs.shape))
    log.info("  CLS z  : %s", list(z.shape))
    log.info(
        "  Logits : %s  (N_EXPERTS_DOMAIN=%d)", list(logits.shape), N_EXPERTS_DOMAIN
    )
    log.info(
        "  Loss   : %.4f (esperado ~%.4f para clase aleatoria)",
        loss.item(),
        torch.log(torch.tensor(float(N_EXPERTS_DOMAIN))).item(),
    )
    log.info("  Forward pass OK ✓")
    log.info("└────────────────────────────────────────────────────────┘")

    # 4. Siguiente paso
    log.info("")
    log.info("[DRY-RUN] Para entrenar en producción:")
    log.info(
        "  python src/pipeline/fase1/fase1_train_pipeline.py --backbone %s",
        args.backbone,
    )
    log.info("[DRY-RUN] Fin. Sin cambios en disco.")


# ══════════════════════════════════════════════════════════════
#  Guard Clauses (reusar de fase1_pipeline.py)
# ══════════════════════════════════════════════════════════════


def _check_fase0_artifacts(cfg):
    """Verifica artefactos de Fase 0 necesarios. Igual que en fase1_pipeline.py."""
    checks = []
    for key in ("nih_train_list", "nih_val_list", "nih_test_list"):
        path = cfg.get(key)
        if path and not Path(path).exists():
            checks.append(f"  ✗ {key} = {path}")
    for key in ("isic_train_csv", "isic_val_csv", "isic_test_csv"):
        path = cfg.get(key)
        if path and not Path(path).exists():
            checks.append(f"  ✗ {key} = {path}")
    oa_root = cfg.get("oa_root")
    if oa_root:
        for split in ("train", "val", "test"):
            sd = Path(oa_root) / split
            if not sd.exists():
                checks.append(f"  ✗ oa_root/{split} no existe")
    panc_csv = cfg.get("pancreas_splits_csv")
    if panc_csv and not Path(panc_csv).exists():
        checks.append(f"  ✗ pancreas_splits_csv = {panc_csv}")
    if checks:
        log.error("Artefactos de Fase 0 faltantes:\n%s", "\n".join(checks))
        sys.exit(1)
    log.info("[Guard] Artefactos de Fase 0 verificados ✓")


# ══════════════════════════════════════════════════════════════
#  Generar reporte Markdown de entrenamiento
# ══════════════════════════════════════════════════════════════


def _generate_train_report(report_data: dict, output_path: str) -> None:
    """
    Genera fase1_train_report.md con métricas del entrenamiento.
    """
    lines = [
        "# Fase 1 — Reporte de Entrenamiento de Backbone",
        "",
        f"**Fecha:** {report_data['timestamp']}",
        f"**Backbone:** `{report_data['backbone']}`",
        f"**Dispositivo:** {report_data['device']}",
        f"**Tiempo total:** {report_data['total_time_s']:.1f}s "
        f"({report_data['total_time_s'] / 60:.1f} min)",
        "",
    ]

    if report_data.get("skipped"):
        lines += [
            "## Estado: OMITIDO (idempotencia)",
            "",
            "El checkpoint ya existía y `--force` no fue proporcionado.",
            "",
        ]
    else:
        lines += [
            "## Resultados",
            "",
            f"- **Mejor val_acc:** {report_data['best_val_acc']:.4f}",
            f"- **Mejor epoch:** {report_data['best_epoch']}",
            f"- **Checkpoint:** `{report_data['checkpoint_path']}`",
            "",
            "## Historia de Entrenamiento",
            "",
            "| Epoch | train_loss | train_acc | val_loss | val_acc |",
            "|-------|------------|-----------|----------|---------|",
        ]
        train_h = report_data.get("train_history", [])
        val_h = report_data.get("val_history", [])
        for t, v in zip(train_h, val_h):
            lines.append(
                f"| {t['epoch']:5d} | {t['loss']:10.4f} | {t['acc']:9.4f} "
                f"| {v['loss']:8.4f} | {v['acc']:7.4f} |"
            )
        lines.append("")

    lines += ["---", "*Generado automáticamente por fase1_train_pipeline.py*"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    log.info("[Reporte] fase1_train_report.md escrito: %s", output_path)


# ══════════════════════════════════════════════════════════════
#  Pipeline Principal
# ══════════════════════════════════════════════════════════════


def main(args):
    t_start = time.time()
    setup_logging(args.checkpoint_dir, phase_name="fase1_train")

    log.info("=" * 60)
    log.info("PASO 4.1 — Entrenamiento End-to-End del Backbone")
    log.info("=" * 60)
    log.info("Backbone: %s | epochs=%d | lr=%.2e", args.backbone, args.epochs, args.lr)

    checkpoint_path = _get_checkpoint_path(args.backbone, args.checkpoint_dir)

    report_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": args.backbone,
        "device": "pendiente",
        "checkpoint_path": checkpoint_path,
        "skipped": False,
        "total_time_s": 0.0,
        "best_val_acc": 0.0,
        "best_epoch": 0,
        "train_history": [],
        "val_history": [],
    }

    # ── 0. Idempotencia ──
    if not args.force and backbone_checkpoint_exists(checkpoint_path):
        log.info(
            "[SKIP] Checkpoint ya existe para '%s' en '%s'. "
            "Usa --force para reentrenar.",
            args.backbone,
            checkpoint_path,
        )
        report_data["skipped"] = True
        report_data["device"] = "N/A (omitido)"
        report_data["total_time_s"] = time.time() - t_start
        report_path = (
            Path(args.checkpoint_dir)
            / BACKBONE_TO_CHECKPOINT_DIR[args.backbone]
            / "fase1_train_report.md"
        )
        _generate_train_report(report_data, str(report_path))
        return

    # ── 1. Guard clause ──
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
        "output_dir": args.checkpoint_dir,
    }
    _check_fase0_artifacts(cfg)

    # ── 2. Setup de dispositivo ──
    device, device_type = _detect_device()
    report_data["device"] = str(device)

    # ── 3. Backbone entrenable ──
    backbone, d_model = load_trainable_backbone(args.backbone, device)

    # ── 4. Datasets (mode="embedding" — misma interfaz, sin augmentaciones) ──
    log.info("[Setup] Construyendo datasets...")
    train_ds, val_ds, _test_ds = build_datasets(cfg)
    log.info("[Dataset] train=%s  val=%s", f"{len(train_ds):,}", f"{len(val_ds):,}")

    # ── 5. DataLoaders ──
    n_workers = (
        args.workers
        if device_type == "gpu"
        else min(args.workers, max(1, (os.cpu_count() or 4) // 2 - 1))
    )
    _dl_kw = dict(
        batch_size=args.batch_size,
        shuffle=True,  # shuffle=True para entrenamiento
        num_workers=n_workers,
        pin_memory=(device_type == "gpu"),
        prefetch_factor=2 if n_workers > 0 else None,
        persistent_workers=(n_workers > 0),
        drop_last=True,  # evitar batches incompletos en última iteración
    )
    train_loader = DataLoader(train_ds, **_dl_kw)
    val_loader = DataLoader(val_ds, **{**_dl_kw, "shuffle": False, "drop_last": False})

    # ── 6. Entrenamiento ──
    train_cfg = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
    }
    metrics = train_backbone(
        backbone=backbone,
        d_model=d_model,
        backbone_name=args.backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_path=checkpoint_path,
        cfg=train_cfg,
    )

    # ── 7. Reporte ──
    report_data.update(
        {
            "total_time_s": time.time() - t_start,
            "best_val_acc": metrics["best_val_acc"],
            "best_epoch": metrics["best_epoch"],
            "train_history": metrics["train_history"],
            "val_history": metrics["val_history"],
        }
    )
    report_path = (
        Path(args.checkpoint_dir)
        / BACKBONE_TO_CHECKPOINT_DIR[args.backbone]
        / "fase1_train_report.md"
    )
    _generate_train_report(report_data, str(report_path))

    log.info(
        "[Paso 4.1] Completado. Siguiente: fase1_pipeline.py --backbone %s "
        "--checkpoint_path %s",
        args.backbone,
        checkpoint_path,
    )


# ══════════════════════════════════════════════════════════════
#  Argparse
# ══════════════════════════════════════════════════════════════


def _build_parser():
    _DS = _PROJECT_ROOT / "datasets"

    def _default_if_exists(rel_path):
        full = _DS / rel_path
        return str(full) if full.exists() else None

    parser = argparse.ArgumentParser(
        description="PASO 4.1 — Entrenamiento end-to-end de backbones MoE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Verificar sin entrenar:\n"
            "  python fase1_train_pipeline.py --backbone vit_tiny_patch16_224 --dry-run\n\n"
            "  # Entrenar:\n"
            "  python fase1_train_pipeline.py --backbone vit_tiny_patch16_224\n"
        ),
    )

    # ── Backbone ──
    parser.add_argument(
        "--backbone",
        default="vit_tiny_patch16_224",
        choices=list(BACKBONE_CONFIGS.keys()),
    )
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=TRAIN_WORKERS)

    # ── Entrenamiento ──
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=TRAIN_LR)
    parser.add_argument("--weight_decay", type=float, default=TRAIN_WEIGHT_DECAY)
    parser.add_argument("--warmup_epochs", type=int, default=TRAIN_WARMUP_EPOCHS)

    # ── Salida ──
    parser.add_argument(
        "--checkpoint_dir",
        default="./checkpoints",
        help="Directorio base de checkpoints (default: ./checkpoints)",
    )

    # ── Control ──
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Reentrenar aunque el checkpoint ya exista",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Verificar configuración + forward pass sin entrenar",
    )

    # ── Datasets (IDÉNTICO a fase1_pipeline.py) ──
    parser.add_argument(
        "--chest_csv", default=_default_if_exists("nih_chest_xrays/Data_Entry_2017.csv")
    )
    parser.add_argument(
        "--chest_imgs", default=_default_if_exists("nih_chest_xrays/all_images")
    )
    parser.add_argument(
        "--nih_train_list",
        default=_default_if_exists("nih_chest_xrays/splits/nih_train_list.txt"),
    )
    parser.add_argument(
        "--nih_val_list",
        default=_default_if_exists("nih_chest_xrays/splits/nih_val_list.txt"),
    )
    parser.add_argument(
        "--nih_test_list",
        default=_default_if_exists("nih_chest_xrays/splits/nih_test_list.txt"),
    )
    parser.add_argument("--chest_view_filter", default=None, choices=["PA", "AP"])
    parser.add_argument(
        "--chest_bbox_csv",
        default=_default_if_exists("nih_chest_xrays/BBox_List_2017.csv"),
    )
    parser.add_argument(
        "--isic_train_csv",
        default=_default_if_exists("isic_2019/splits/isic_train.csv"),
    )
    parser.add_argument(
        "--isic_val_csv", default=_default_if_exists("isic_2019/splits/isic_val.csv")
    )
    parser.add_argument(
        "--isic_test_csv", default=_default_if_exists("isic_2019/splits/isic_test.csv")
    )
    parser.add_argument(
        "--isic_imgs", default=_default_if_exists("isic_2019/ISIC_2019_Training_Input")
    )
    parser.add_argument(
        "--oa_root", default=_default_if_exists("osteoarthritis/oa_splits")
    )
    parser.add_argument(
        "--luna_patches_dir", default=_default_if_exists("luna_lung_cancer/patches")
    )
    parser.add_argument(
        "--luna_csv",
        default=_default_if_exists("luna_lung_cancer/candidates_V2/candidates_V2.csv"),
    )
    parser.add_argument(
        "--pancreas_splits_csv", default=_default_if_exists("pancreas_splits.csv")
    )
    parser.add_argument(
        "--pancreas_nii_dir", default=_default_if_exists("zenodo_13715870")
    )
    parser.add_argument("--pancreas_fold", type=int, default=PANCREAS_FOLD)
    parser.add_argument("--pancreas_roi_strategy", default="A", choices=["A", "B"])

    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    if _args.dry_run:
        setup_logging(_args.checkpoint_dir, phase_name="fase1_train")
        _print_dry_run_summary(_args)
        sys.exit(0)

    main(_args)

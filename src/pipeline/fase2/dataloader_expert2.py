"""
DataLoader para Expert 2 — ISIC 2019 (dermoscopía, 8 clases multiclase).

Construye DataLoaders para train, val y test del Expert 2 (ConvNeXt-Small).
Los splits (CSVs) fueron pre-construidos en Fase 0 usando
ISICDataset.build_lesion_split() para evitar leakage por lesion_id.

Transforms:
    - Gestionados internamente por ISICDataset según mode/split:
      * train: TRANSFORM_TRAIN (RandomCrop 224, flips, rot 360°, ColorJitter, etc.)
      * val/test: TRANSFORM_VAL (CenterCrop 224)
    - NO incluye BCNCrop — lo aplica ISICDataset internamente (apply_bcn_crop=True)

Cache:
    - Si cache_dir existe y contiene imágenes preprocesadas ({id}_pp_224.jpg),
      ISICDataset las carga directamente (DullRazor + resize ya aplicados).
    - Si no, carga las originales con resize a shorter_side=224.

Sampler:
    - Train usa WeightedRandomSampler (inverse-frequency) para compensar desbalance.
    - Val/Test usan shuffle=False.

Dependencias:
    - src/pipeline/datasets/isic.py: ISICDataset (mode="expert")
    - src/pipeline/fase2/expert2_config.py: hiperparámetros

Uso:
    from dataloader_expert2 import build_dataloaders_expert2
    train_loader, val_loader, test_loader, class_weights = build_dataloaders_expert2()
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── Agregar src/pipeline al path para imports ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from datasets.isic import ISICDataset
from fase2.expert2_config import EXPERT2_BATCH_SIZE

log = logging.getLogger("expert2_dataloader")

# ── Rutas de datos ─────────────────────────────────────────────────────
_ISIC_IMG_DIR = _PROJECT_ROOT / "datasets" / "isic_2019" / "ISIC_2019_Training_Input"
_ISIC_CACHE_DIR = (
    _PROJECT_ROOT / "datasets" / "isic_2019" / "ISIC_2019_Training_Input_preprocessed"
)
_ISIC_TRAIN_CSV = _PROJECT_ROOT / "splits" / "isic_train.csv"
_ISIC_VAL_CSV = _PROJECT_ROOT / "splits" / "isic_val.csv"
_ISIC_TEST_CSV = _PROJECT_ROOT / "splits" / "isic_test.csv"

_NUM_WORKERS = 4


def build_dataloaders_expert2(
    img_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    train_csv: str | Path | None = None,
    val_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int = _NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Construye DataLoaders para train, val y test del Expert 2 (ISIC 2019).

    Args:
        img_dir: directorio con imágenes ISIC .jpg originales.
            Default: datasets/isic_2019/ISIC_2019_Training_Input/
        cache_dir: directorio con imágenes preprocesadas (DullRazor + resize).
            Default: datasets/isic_2019/ISIC_2019_Training_Input_preprocessed/
            Si no existe o es None, se usan las originales con resize online.
        train_csv: CSV con split de entrenamiento. Default: splits/isic_train.csv
        val_csv: CSV con split de validación. Default: splits/isic_val.csv
        test_csv: CSV con split de test. Default: splits/isic_test.csv
        batch_size: tamaño de batch por GPU. Default: EXPERT2_BATCH_SIZE (32).
        num_workers: número de workers para DataLoader. Default: 4.

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
        - class_weights: Tensor[8] con pesos de clase para FocalLossMultiClass
    """
    img_dir = Path(img_dir) if img_dir else _ISIC_IMG_DIR
    train_csv = Path(train_csv) if train_csv else _ISIC_TRAIN_CSV
    val_csv = Path(val_csv) if val_csv else _ISIC_VAL_CSV
    test_csv = Path(test_csv) if test_csv else _ISIC_TEST_CSV
    batch_size = batch_size or EXPERT2_BATCH_SIZE

    # Resolver cache_dir: usar default si no se pasa, verificar existencia
    if cache_dir is not None:
        effective_cache_dir: Path | None = Path(cache_dir)
    elif _ISIC_CACHE_DIR.exists():
        effective_cache_dir = _ISIC_CACHE_DIR
    else:
        effective_cache_dir = None

    if effective_cache_dir is not None and not effective_cache_dir.exists():
        log.warning(
            f"[Expert2/DataLoader] cache_dir no existe: {effective_cache_dir}. "
            f"Usando imágenes originales con resize online."
        )
        effective_cache_dir = None

    # ── Verificar que los archivos existen ─────────────────────────────
    for label, path in [
        ("img_dir", img_dir),
        ("train_csv", train_csv),
        ("val_csv", val_csv),
        ("test_csv", test_csv),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"[Expert2/DataLoader] {label} no encontrado: {path}"
            )

    log.info(f"[Expert2/DataLoader] Images: {img_dir}")
    log.info(f"[Expert2/DataLoader] Cache:  {effective_cache_dir}")
    log.info(f"[Expert2/DataLoader] Train CSV: {train_csv}")
    log.info(f"[Expert2/DataLoader] Val CSV:   {val_csv}")
    log.info(f"[Expert2/DataLoader] Test CSV:  {test_csv}")
    log.info(f"[Expert2/DataLoader] Batch: {batch_size} | Workers: {num_workers}")

    # ── Cargar DataFrames ──────────────────────────────────────────────
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    log.info(
        f"[Expert2/DataLoader] Splits cargados — "
        f"train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}"
    )

    # ── Crear datasets ─────────────────────────────────────────────────
    # ISICDataset en mode="expert" selecciona TRANSFORM_TRAIN/TRANSFORM_VAL
    # internamente según split. apply_bcn_crop=True para BCN_20000.
    train_ds = ISICDataset(
        img_dir=img_dir,
        cache_dir=effective_cache_dir,
        split_df=train_df,
        mode="expert",
        split="train",
        apply_bcn_crop=True,
    )

    val_ds = ISICDataset(
        img_dir=img_dir,
        cache_dir=effective_cache_dir,
        split_df=val_df,
        mode="expert",
        split="val",
        apply_bcn_crop=True,
    )

    test_ds = ISICDataset(
        img_dir=img_dir,
        cache_dir=effective_cache_dir,
        split_df=test_df,
        mode="expert",
        split="test",
        apply_bcn_crop=True,
    )

    # ── Obtener class_weights ──────────────────────────────────────────
    if hasattr(train_ds, "class_weights") and train_ds.class_weights is not None:
        class_weights = train_ds.class_weights
        log.info(
            f"[Expert2/DataLoader] class_weights desde ISICDataset: "
            f"shape={class_weights.shape} | "
            f"min={class_weights.min():.3f} | max={class_weights.max():.3f}"
        )
    else:
        # Fallback: computar pesos por inverse-frequency
        log.warning(
            "[Expert2/DataLoader] ISICDataset no tiene class_weights. "
            "Computando por inverse-frequency..."
        )
        labels = [train_ds[i][1] for i in range(len(train_ds))]
        n_classes = ISICDataset.N_TRAIN_CLS
        counts = np.bincount(labels, minlength=n_classes)
        class_weights = torch.tensor(
            len(train_ds) / (n_classes * np.maximum(counts, 1)),
            dtype=torch.float32,
        )
        log.info(
            f"[Expert2/DataLoader] class_weights (fallback): "
            f"shape={class_weights.shape} | "
            f"min={class_weights.min():.3f} | max={class_weights.max():.3f}"
        )

    # ── Crear DataLoaders ──────────────────────────────────────────────
    # Train: WeightedRandomSampler para compensar desbalance
    sampler = ISICDataset.get_weighted_sampler(train_df, ISICDataset.CLASSES)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,  # reemplaza shuffle=True
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # ── Resumen ────────────────────────────────────────────────────────
    _print_summary(train_ds, val_ds, test_ds, train_loader, val_loader, test_loader)

    return train_loader, val_loader, test_loader, class_weights


def _print_summary(
    train_ds: ISICDataset,
    val_ds: ISICDataset,
    test_ds: ISICDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    """Imprime resumen de los datasets y dataloaders."""
    print("=" * 70)
    print("  RESUMEN — DataLoaders Expert 2 (ISIC 2019)")
    print("=" * 70)

    for name, ds, loader in [
        ("train", train_ds, train_loader),
        ("val", val_ds, val_loader),
        ("test", test_ds, test_loader),
    ]:
        print(f"  {name:5s}: {len(ds):>6,} muestras | batches={len(loader):>4,}")

    print("=" * 70)


def _test_dataloaders() -> None:
    """Verificación rápida: cargar un batch de cada split."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    train_loader, val_loader, test_loader, class_weights = build_dataloaders_expert2(
        num_workers=0,
    )

    print(f"\n[Expert2/DataLoader] class_weights: {class_weights}")

    for name, loader in [("train", train_loader), ("val", val_loader)]:
        batch = next(iter(loader))
        img, label, stem = batch
        print(f"\n[Expert2/DataLoader] Primer batch de {name}:")
        print(f"  img shape:   {list(img.shape)}")
        print(f"  img dtype:   {img.dtype}")
        print(f"  img min/max: {img.min():.4f} / {img.max():.4f}")
        print(f"  labels:      {label[:8]}")
        print(f"  stems:       {stem[:2]}...")


if __name__ == "__main__":
    _test_dataloaders()

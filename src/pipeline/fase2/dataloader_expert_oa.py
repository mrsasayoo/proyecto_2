"""
DataLoader para Expert OA — Osteoarthritis Knee (5 clases KL 0-4).

Construye DataLoaders para train, val y test del Expert OA (EfficientNet-B0, 5 clases KL 0-4).
Los splits están organizados en carpetas: {root_dir}/{train,val,test}/{0,1,2,3,4}/

Transforms:
    - TODOS los transforms son internos al OAKneeDataset.
    - Train: Resize(256)→RandomCrop(224)→HFlip→Rotation(±15°)→ColorJitter→RandomAutocontrast→Normalize
    - Val/Test: Resize(224)→Normalize  (sin CLAHE, sin augmentation)
      a menos que se detecte augmentation offline (H2).
    - En mode="expert", split="val"/"test": solo base_transform (ToTensor + Normalize).
    - NO se pasan transforms externos al constructor de OAKneeDataset.
    - PROHIBIDO: RandomVerticalFlip (orientación anatómica fija).
    - PROHIBIDO: RandomErasing/Cutout/CutMix/MixUp.

Dependencias:
    - src/pipeline/datasets/osteoarthritis.py: OAKneeDataset (mode="expert")
    - src/pipeline/fase2/expert_oa_config.py: hiperparámetros

Uso:
    from dataloader_expert_oa import get_oa_dataloaders
    train_loader, val_loader, test_loader, class_weights = get_oa_dataloaders()
"""

import sys
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ── Agregar src/pipeline al path para imports ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from datasets.osteoarthritis import OAKneeDataset
from fase2.expert_oa_config import (
    EXPERT_OA_BATCH_SIZE,
    EXPERT_OA_IMG_SIZE,
    EXPERT_OA_NUM_CLASSES,
)

log = logging.getLogger("expert_oa_dataloader")

# ── Rutas de datos ─────────────────────────────────────────────────────
_OA_ROOT_DIR = _PROJECT_ROOT / "datasets" / "osteoarthritis" / "oa_splits"

_NUM_WORKERS = 4


def get_oa_dataloaders(
    root_dir: str | Path | None = None,
    batch_size: int | None = None,
    img_size: int | None = None,
    num_workers: int = _NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Construye DataLoaders para train, val y test del Expert OA (Knee).

    IMPORTANTE: OAKneeDataset NO acepta transforms externos. Todos los
    transforms (CLAHE, resize, augmentation) son internos al dataset.

    Args:
        root_dir: directorio raíz del dataset OA (contiene train/, val/, test/).
            Default: datasets/oa_splits/
        batch_size: tamaño de batch por GPU. Default: EXPERT_OA_BATCH_SIZE (32).
        img_size: tamaño de imagen de entrada. Default: EXPERT_OA_IMG_SIZE (224).
        num_workers: número de workers para DataLoader. Default: 4.

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
        - class_weights: Tensor[5] con pesos de clase para CrossEntropyLoss (KL 0-4)
    """
    root_dir = Path(root_dir) if root_dir else _OA_ROOT_DIR
    batch_size = batch_size or EXPERT_OA_BATCH_SIZE
    img_size = img_size or EXPERT_OA_IMG_SIZE

    # ── Verificar que el directorio raíz existe ────────────────────
    if not root_dir.exists():
        raise FileNotFoundError(
            f"[ExpertOA/DataLoader] Directorio raíz no encontrado: {root_dir}"
        )

    log.info(f"[ExpertOA/DataLoader] Root dir: {root_dir}")
    log.info(f"[ExpertOA/DataLoader] Batch: {batch_size} | Workers: {num_workers}")
    log.info(f"[ExpertOA/DataLoader] Image size: {img_size}")

    # ── Crear datasets ─────────────────────────────────────────────
    # OAKneeDataset maneja transforms internamente:
    #   - CLAHE antes del resize (H4)
    #   - Augmentation online en train si no hay aug offline (H2)
    #   - Base transform (ToTensor + Normalize) para val/test
    train_ds = OAKneeDataset(
        root_dir=root_dir,
        split="train",
        img_size=img_size,
        mode="expert",
    )

    val_ds = OAKneeDataset(
        root_dir=root_dir,
        split="val",
        img_size=img_size,
        mode="expert",
    )

    test_ds = OAKneeDataset(
        root_dir=root_dir,
        split="test",
        img_size=img_size,
        mode="expert",
    )

    # ── Obtener class_weights ──────────────────────────────────────
    if hasattr(train_ds, "class_weights") and train_ds.class_weights is not None:
        class_weights = train_ds.class_weights
        log.info(
            f"[ExpertOA/DataLoader] class_weights desde OAKneeDataset: "
            f"shape={class_weights.shape} | "
            f"min={class_weights.min():.3f} | max={class_weights.max():.3f}"
        )
    else:
        # Fallback: pesos uniformes
        log.warning(
            "[ExpertOA/DataLoader] OAKneeDataset no tiene class_weights. "
            "Usando pesos uniformes."
        )
        class_weights = torch.ones(EXPERT_OA_NUM_CLASSES, dtype=torch.float32)

    # ── Crear DataLoaders ──────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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

    # ── Resumen ────────────────────────────────────────────────────
    _print_summary(train_ds, val_ds, test_ds, train_loader, val_loader, test_loader)

    return train_loader, val_loader, test_loader, class_weights


def _print_summary(
    train_ds: OAKneeDataset,
    val_ds: OAKneeDataset,
    test_ds: OAKneeDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    """Imprime resumen de los datasets y dataloaders."""
    print("=" * 70)
    print("  RESUMEN — DataLoaders Expert OA (Osteoarthritis Knee)")
    print("=" * 70)

    for name, ds, loader in [
        ("train", train_ds, train_loader),
        ("val", val_ds, val_loader),
        ("test", test_ds, test_loader),
    ]:
        print(f"  {name:5s}: {len(ds):>6,} muestras | batches={len(loader):>4,}")

    print("=" * 70)


def _test_dataloaders():
    """Verificación rápida: cargar un batch de cada split."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    train_loader, val_loader, test_loader, class_weights = get_oa_dataloaders(
        num_workers=0,
    )

    print(f"\n[ExpertOA/DataLoader] class_weights: {class_weights}")

    for name, loader in [("train", train_loader), ("val", val_loader)]:
        batch = next(iter(loader))
        img, label, img_name = batch
        print(f"\n[ExpertOA/DataLoader] Primer batch de {name}:")
        print(f"  img shape:   {list(img.shape)}")
        print(f"  img dtype:   {img.dtype}")
        print(f"  img min/max: {img.min():.4f} / {img.max():.4f}")
        print(f"  labels:      {label[:8]}")
        print(f"  img_names:   {img_name[:2]}...")


if __name__ == "__main__":
    _test_dataloaders()

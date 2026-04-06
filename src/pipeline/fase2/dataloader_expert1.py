"""
DataLoader para Expert 1 — NIH ChestXray14 (14 patologías, multilabel).

Construye DataLoaders para train, val y test del Expert 1 (ConvNeXt-Tiny).
Usa el pipeline de preprocesamiento 2D estándar (CLAHE → Resize → TVF → ToTensor → Normalize)
y augmentation 2D para training (flip, rotación, color jitter).

Dependencias:
    - src/pipeline/datasets/chest.py: ChestXray14Dataset
    - src/pipeline/fase1/transform_2d.py: build_2d_transform, build_2d_aug_transform
    - src/pipeline/fase2/expert1_config.py: hiperparámetros

Uso:
    from dataloader_expert1 import build_dataloaders_expert1
    train_loader, val_loader, test_loader, pos_weight = build_dataloaders_expert1()
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

from datasets.chest import ChestXray14Dataset
from fase1.transform_2d import build_2d_transform, build_2d_aug_transform
from fase2.expert1_config import EXPERT1_BATCH_SIZE

log = logging.getLogger("expert1_dataloader")

# ── Rutas de datos (relativas a PROJECT_ROOT) ─────────────────────────
_CHEST_CSV = _PROJECT_ROOT / "datasets" / "nih-chest-xrays" / "Data_Entry_2017.csv"
_CHEST_IMG_DIR = _PROJECT_ROOT / "datasets" / "nih-chest-xrays" / "all_images"
_CHEST_TRAIN_LIST = _PROJECT_ROOT / "splits" / "nih_train.txt"
_CHEST_VAL_LIST = _PROJECT_ROOT / "splits" / "nih_val.txt"
_CHEST_TEST_LIST = _PROJECT_ROOT / "splits" / "nih_test.txt"

_NUM_WORKERS = 4


def build_dataloaders_expert1(
    csv_path: str | Path | None = None,
    img_dir: str | Path | None = None,
    train_list: str | Path | None = None,
    val_list: str | Path | None = None,
    test_list: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int = _NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Construye DataLoaders para train, val y test del Expert 1 (ChestXray14).

    Retorna una 4-tupla con los loaders y el tensor de pos_weight para
    BCEWithLogitsLoss (computado del split de entrenamiento).

    Args:
        csv_path: ruta al Data_Entry_2017.csv. Default: datasets/nih-chest-xrays/Data_Entry_2017.csv
        img_dir: directorio de imágenes. Default: datasets/nih-chest-xrays/all_images/
        train_list: archivo con nombres de imágenes del split train.
        val_list: archivo con nombres de imágenes del split val.
        test_list: archivo con nombres de imágenes del split test.
        batch_size: tamaño de batch por GPU. Default: EXPERT1_BATCH_SIZE (32).
        num_workers: número de workers para DataLoader. Default: 4.

    Returns:
        (train_loader, val_loader, test_loader, pos_weight_tensor)
        donde pos_weight_tensor es Tensor[14] con n_neg/n_pos por clase.
    """
    csv_path = Path(csv_path) if csv_path else _CHEST_CSV
    img_dir = Path(img_dir) if img_dir else _CHEST_IMG_DIR
    train_list = Path(train_list) if train_list else _CHEST_TRAIN_LIST
    val_list = Path(val_list) if val_list else _CHEST_VAL_LIST
    test_list = Path(test_list) if test_list else _CHEST_TEST_LIST
    batch_size = batch_size or EXPERT1_BATCH_SIZE

    # ── Verificar rutas ────────────────────────────────────────────────
    for name, path in [
        ("CSV", csv_path),
        ("img_dir", img_dir),
        ("train_list", train_list),
        ("val_list", val_list),
        ("test_list", test_list),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"[Expert1/DataLoader] {name} no encontrado: {path}"
            )

    log.info(f"[Expert1/DataLoader] CSV: {csv_path}")
    log.info(f"[Expert1/DataLoader] Imágenes: {img_dir}")
    log.info(f"[Expert1/DataLoader] Batch: {batch_size} | Workers: {num_workers}")

    # ── Transforms ─────────────────────────────────────────────────────
    transform_standard = build_2d_transform()
    transform_aug = build_2d_aug_transform()

    # ── Crear datasets ─────────────────────────────────────────────────
    train_ds = ChestXray14Dataset(
        csv_path=str(csv_path),
        img_dir=str(img_dir),
        file_list=str(train_list),
        transform=transform_standard,
        mode="expert",
        split="train",
        aug_transform=transform_aug,
    )

    val_ds = ChestXray14Dataset(
        csv_path=str(csv_path),
        img_dir=str(img_dir),
        file_list=str(val_list),
        transform=transform_standard,
        mode="expert",
        split="val",
    )

    test_ds = ChestXray14Dataset(
        csv_path=str(csv_path),
        img_dir=str(img_dir),
        file_list=str(test_list),
        transform=transform_standard,
        mode="expert",
        split="test",
    )

    # ── Extraer pos_weight del dataset de entrenamiento ─────────────────
    pos_weight = train_ds.class_weights
    if pos_weight is None:
        raise RuntimeError(
            "[Expert1/DataLoader] train_ds.class_weights es None. "
            "Verifica que mode='expert' esté configurado correctamente."
        )
    log.info(
        f"[Expert1/DataLoader] pos_weight shape: {pos_weight.shape} | "
        f"min: {pos_weight.min():.1f} | max: {pos_weight.max():.1f}"
    )

    # ── Crear DataLoaders ──────────────────────────────────────────────
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

    # ── Resumen ────────────────────────────────────────────────────────
    _print_summary(train_ds, val_ds, test_ds, train_loader, val_loader, test_loader)

    return train_loader, val_loader, test_loader, pos_weight


def _print_summary(
    train_ds: ChestXray14Dataset,
    val_ds: ChestXray14Dataset,
    test_ds: ChestXray14Dataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    """Imprime resumen de los datasets y dataloaders."""
    print("=" * 70)
    print("  RESUMEN — DataLoaders Expert 1 (ChestXray14)")
    print("=" * 70)

    for name, ds, loader in [
        ("train", train_ds, train_loader),
        ("val", val_ds, val_loader),
        ("test", test_ds, test_loader),
    ]:
        aug_status = (
            "SÍ" if (ds.split == "train" and ds.aug_transform is not None) else "NO"
        )
        print(
            f"  {name:5s}: {len(ds):>6,} muestras | "
            f"batches={len(loader):>4,} | aug={aug_status}"
        )

    print("=" * 70)


def _test_dataloaders():
    """Verificación rápida: cargar un batch de cada split."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    train_loader, val_loader, _test_loader, pos_weight = build_dataloaders_expert1(
        num_workers=0
    )

    print(f"\n[Expert1/DataLoader] pos_weight: {pos_weight}")

    for name, loader in [("train", train_loader), ("val", val_loader)]:
        batch = next(iter(loader))
        imgs, labels, names = batch
        print(f"\n[Expert1/DataLoader] Primer batch de {name}:")
        print(f"  img shape: {list(imgs.shape)}")
        print(f"  img dtype:  {imgs.dtype}")
        print(f"  img min/max: {imgs.min():.4f} / {imgs.max():.4f}")
        print(f"  labels shape: {list(labels.shape)}")
        print(f"  labels dtype: {labels.dtype}")
        print(f"  labels sum per sample (primeros 4): {labels[:4].sum(dim=1).tolist()}")
        print(f"  names: {names[:2]}...")


if __name__ == "__main__":
    _test_dataloaders()

"""
DataLoader para Expert 1 — NIH ChestXray14 (14 patologías, multilabel).

Construye DataLoaders para train, val, test y test_flip (TTA) del Expert 1
(ConvNeXt-Tiny). Transforms gestionados con Albumentations.

Pipeline offline (dentro de ChestXray14Dataset):
    cv2 grayscale → CLAHE → multistage_resize → cache RAM (uint8 224×224)

Pipeline online (__getitem__ del dataset + transforms de este módulo):
    GRAY2RGB → Albumentations (aug + Normalize + ToTensorV2) → tensor float32

Dependencias:
    - src/pipeline/datasets/chest.py: ChestXray14Dataset / NIHChestDataset
    - src/pipeline/fase2/expert1_config.py: EXPERT1_BATCH_SIZE, EXPERT1_NUM_WORKERS

Uso:
    from dataloader_expert1 import build_expert1_dataloaders
    loaders = build_expert1_dataloaders(
        csv_path=..., images_dir=...,
        train_split_file=..., val_split_file=..., test_split_file=...,
        model_mean=model.model_mean, model_std=model.model_std,
    )
    train_loader = loaders["train"]
    test_loader  = loaders["test"]
    test_flip    = loaders["test_flip"]  # TTA con HorizontalFlip=1.0
    pos_weight   = loaders["pos_weight"]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

# ── Agregar src/pipeline al path para imports ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from datasets.chest import ChestXray14Dataset
from fase2.expert1_config import EXPERT1_BATCH_SIZE, EXPERT1_NUM_WORKERS

log = logging.getLogger("expert1_dataloader")


# ── Transforms (Albumentations) ───────────────────────────────────────


def _build_train_transform(
    model_mean: tuple[float, ...],
    model_std: tuple[float, ...],
) -> A.Compose:
    """Transform de entrenamiento con augmentation moderada para rayos X.

    Incluye HorizontalFlip, brightness/contrast, gamma y ruido gaussiano
    con probabilidades conservadoras apropiadas para imágenes médicas.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5,
            ),
            A.RandomGamma(gamma_limit=(85, 115), p=0.5),
            A.GaussNoise(std_range=(0.01, 0.02), p=0.1),
            A.Normalize(mean=model_mean, std=model_std),
            ToTensorV2(),
        ]
    )


def _build_val_transform(
    model_mean: tuple[float, ...],
    model_std: tuple[float, ...],
) -> A.Compose:
    """Transform de validación/test: solo normalización + tensor."""
    return A.Compose(
        [
            A.Normalize(mean=model_mean, std=model_std),
            ToTensorV2(),
        ]
    )


def _build_flip_transform(
    model_mean: tuple[float, ...],
    model_std: tuple[float, ...],
) -> A.Compose:
    """Transform para TTA: HorizontalFlip determinista (p=1.0) + normalización."""
    return A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=model_mean, std=model_std),
            ToTensorV2(),
        ]
    )


# ── API pública ───────────────────────────────────────────────────────


def build_expert1_dataloaders(
    csv_path: str,
    images_dir: str,
    train_split_file: str,
    val_split_file: str,
    test_split_file: str,
    model_mean: tuple[float, ...],
    model_std: tuple[float, ...],
    batch_size: int = EXPERT1_BATCH_SIZE,
    num_workers: int = EXPERT1_NUM_WORKERS,
    use_cache: bool = True,
) -> dict[str, DataLoader | torch.Tensor]:
    """Construye DataLoaders para train, val, test y test_flip (TTA).

    Args:
        csv_path: ruta al Data_Entry_2017.csv del NIH ChestXray14.
        images_dir: directorio con las imágenes (all_images/).
        train_split_file: archivo .txt con nombres de imágenes del split train.
        val_split_file: archivo .txt con nombres de imágenes del split val.
        test_split_file: archivo .txt con nombres de imágenes del split test.
        model_mean: tupla RGB de medias de normalización del backbone pretrained
            (obtenida de ``model.model_mean``).
        model_std: tupla RGB de desviaciones estándar de normalización del backbone
            (obtenida de ``model.model_std``).
        batch_size: tamaño de batch por GPU. Default: EXPERT1_BATCH_SIZE (32).
        num_workers: workers para DataLoader. Default: EXPERT1_NUM_WORKERS (4).
        use_cache: precargar imágenes en RAM. Default: True.

    Returns:
        dict con claves:
            - ``'train'``: DataLoader de entrenamiento (shuffle=True, drop_last=True).
            - ``'val'``: DataLoader de validación.
            - ``'test'``: DataLoader de test (sin augmentation).
            - ``'test_flip'``: DataLoader de test con HorizontalFlip=1.0 para TTA.
            - ``'pos_weight'``: Tensor[14] con n_neg/n_pos por clase para
              BCEWithLogitsLoss.

    Raises:
        FileNotFoundError: si alguna ruta de entrada no existe.
        RuntimeError: si el dataset no expone class_weights (mode != 'expert').
    """
    csv_path_p = Path(csv_path)
    images_dir_p = Path(images_dir)
    train_split_p = Path(train_split_file)
    val_split_p = Path(val_split_file)
    test_split_p = Path(test_split_file)

    # ── Verificar rutas ────────────────────────────────────────────────
    for name, path in [
        ("CSV", csv_path_p),
        ("images_dir", images_dir_p),
        ("train_split_file", train_split_p),
        ("val_split_file", val_split_p),
        ("test_split_file", test_split_p),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"[Expert1/DataLoader] {name} no encontrado: {path}"
            )

    log.info(f"[Expert1/DataLoader] CSV: {csv_path_p}")
    log.info(f"[Expert1/DataLoader] Imágenes: {images_dir_p}")
    log.info(
        f"[Expert1/DataLoader] Batch: {batch_size} | Workers: {num_workers} | "
        f"Cache: {use_cache}"
    )
    log.info(f"[Expert1/DataLoader] model_mean={model_mean}  model_std={model_std}")

    # ── Builds transforms ──────────────────────────────────────────────
    train_tfm = _build_train_transform(model_mean, model_std)
    val_tfm = _build_val_transform(model_mean, model_std)
    flip_tfm = _build_flip_transform(model_mean, model_std)

    # ── Crear datasets ─────────────────────────────────────────────────
    train_ds = ChestXray14Dataset(
        csv_path=str(csv_path_p),
        img_dir=str(images_dir_p),
        file_list=str(train_split_p),
        transform=train_tfm,
        mode="expert",
        split="train",
        use_cache=use_cache,
    )

    val_ds = ChestXray14Dataset(
        csv_path=str(csv_path_p),
        img_dir=str(images_dir_p),
        file_list=str(val_split_p),
        transform=val_tfm,
        mode="expert",
        split="val",
        use_cache=use_cache,
    )

    test_ds = ChestXray14Dataset(
        csv_path=str(csv_path_p),
        img_dir=str(images_dir_p),
        file_list=str(test_split_p),
        transform=val_tfm,
        mode="expert",
        split="test",
        use_cache=use_cache,
    )

    test_flip_ds = ChestXray14Dataset(
        csv_path=str(csv_path_p),
        img_dir=str(images_dir_p),
        file_list=str(test_split_p),
        transform=flip_tfm,
        mode="expert",
        split="test",
        use_cache=use_cache,
    )

    # ── Extraer pos_weight del dataset de entrenamiento ────────────────
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
    common_kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **common_kwargs,  # type: ignore[arg-type]
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,  # type: ignore[arg-type]
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,  # type: ignore[arg-type]
    )

    test_flip_loader = DataLoader(
        test_flip_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,  # type: ignore[arg-type]
    )

    # ── Resumen ────────────────────────────────────────────────────────
    _print_summary(
        train_ds,
        val_ds,
        test_ds,
        test_flip_ds,
        train_loader,
        val_loader,
        test_loader,
        test_flip_loader,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "test_flip": test_flip_loader,
        "pos_weight": pos_weight,
    }


def build_tta_loaders(
    ds_orig: Dataset,
    ds_flip: Dataset,
    batch_size: int = EXPERT1_BATCH_SIZE,
    num_workers: int = EXPERT1_NUM_WORKERS,
) -> dict[str, DataLoader]:
    """Construye loaders para Test-Time Augmentation a partir de datasets existentes.

    Útil cuando ya tienes instancias de dataset (por ejemplo, para crear
    loaders TTA con transforms distintos sin re-leer el CSV).

    Args:
        ds_orig: dataset con transform de val (sin flip).
        ds_flip: dataset con transform de flip (HorizontalFlip=1.0).
        batch_size: tamaño de batch. Default: EXPERT1_BATCH_SIZE.
        num_workers: workers. Default: EXPERT1_NUM_WORKERS.

    Returns:
        dict con claves ``'orig'`` y ``'flip'``, cada una un DataLoader.
    """
    common_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    return {
        "orig": DataLoader(ds_orig, **common_kwargs),  # type: ignore[arg-type]
        "flip": DataLoader(ds_flip, **common_kwargs),  # type: ignore[arg-type]
    }


# ── Compatibilidad con imports existentes (train_expert1.py) ──────────
# El trainer actual llama build_dataloaders_expert1() con la firma vieja.
# Este alias adapta la interfaz hasta que el trainer sea actualizado.


def build_dataloaders_expert1(
    csv_path: str | Path | None = None,
    img_dir: str | Path | None = None,
    train_list: str | Path | None = None,
    val_list: str | Path | None = None,
    test_list: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int = EXPERT1_NUM_WORKERS,
    model_mean: tuple[float, ...] | None = None,
    model_std: tuple[float, ...] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Wrapper de compatibilidad con la API anterior.

    Convierte la llamada legacy (positional 4-tupla) a la nueva API (dict).
    Si no se pasan model_mean/model_std, usa los defaults de ImageNet
    (ImageNet-1K: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]).

    Returns:
        (train_loader, val_loader, test_loader, pos_weight)
    """
    # ── Defaults de rutas ──────────────────────────────────────────────
    _csv = (
        str(csv_path)
        if csv_path
        else str(_PROJECT_ROOT / "datasets" / "nih_chest_xrays" / "Data_Entry_2017.csv")
    )
    _img = (
        str(img_dir)
        if img_dir
        else str(_PROJECT_ROOT / "datasets" / "nih_chest_xrays" / "all_images")
    )
    _train = (
        str(train_list)
        if train_list
        else str(
            _PROJECT_ROOT
            / "datasets"
            / "nih_chest_xrays"
            / "splits"
            / "nih_train_list.txt"
        )
    )
    _val = (
        str(val_list)
        if val_list
        else str(
            _PROJECT_ROOT
            / "datasets"
            / "nih_chest_xrays"
            / "splits"
            / "nih_val_list.txt"
        )
    )
    _test = (
        str(test_list)
        if test_list
        else str(
            _PROJECT_ROOT
            / "datasets"
            / "nih_chest_xrays"
            / "splits"
            / "nih_test_list.txt"
        )
    )

    # ── Defaults de normalización (ImageNet-1K) si no se pasan ─────────
    _mean = model_mean if model_mean is not None else (0.485, 0.456, 0.406)
    _std = model_std if model_std is not None else (0.229, 0.224, 0.225)

    result = build_expert1_dataloaders(
        csv_path=_csv,
        images_dir=_img,
        train_split_file=_train,
        val_split_file=_val,
        test_split_file=_test,
        model_mean=_mean,
        model_std=_std,
        batch_size=batch_size or EXPERT1_BATCH_SIZE,
        num_workers=num_workers,
    )

    return (
        result["train"],
        result["val"],
        result["test"],
        result["pos_weight"],
    )


# ── Utilidades internas ───────────────────────────────────────────────


def _print_summary(
    train_ds: ChestXray14Dataset,
    val_ds: ChestXray14Dataset,
    test_ds: ChestXray14Dataset,
    test_flip_ds: ChestXray14Dataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    test_flip_loader: DataLoader,
) -> None:
    """Imprime resumen de los datasets y dataloaders."""
    print("=" * 70)
    print("  RESUMEN — DataLoaders Expert 1 (ChestXray14 + Albumentations)")
    print("=" * 70)

    for name, ds, loader in [
        ("train", train_ds, train_loader),
        ("val", val_ds, val_loader),
        ("test", test_ds, test_loader),
        ("flip", test_flip_ds, test_flip_loader),
    ]:
        aug_tag = (
            "train-aug"
            if name == "train"
            else ("TTA-flip" if name == "flip" else "val/test")
        )
        print(
            f"  {name:5s}: {len(ds):>6,} muestras | "
            f"batches={len(loader):>4,} | transform={aug_tag}"
        )

    print("=" * 70)


# ── Script de prueba ──────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("[Expert1/DataLoader] Verificación de import exitosa.")
    print(
        f"[Expert1/DataLoader] build_expert1_dataloaders: {build_expert1_dataloaders}"
    )
    print(
        f"[Expert1/DataLoader] build_dataloaders_expert1: {build_dataloaders_expert1}"
    )
    print(f"[Expert1/DataLoader] build_tta_loaders: {build_tta_loaders}")

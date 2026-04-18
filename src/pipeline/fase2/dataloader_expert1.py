"""
DataLoader para Expert 1 — NIH ChestXray14 (14 patologías, multilabel).

Construye DataLoaders para train, val, test y test_flip (TTA) del Expert 1
(ConvNeXt-Tiny). Transforms gestionados con Albumentations.

Pipeline offline (pre_chestxray14.py):
    raw PNG → grayscale → CLAHE → multistage_resize → .npy float32 256×256

Pipeline online (__getitem__ del dataset + transforms de este módulo):
    .npy float32 (256,256) → expand (256,256,1) → Albumentations (aug + Normalize + ToTensorV2) → tensor float32 (1,256,256)

Fase 2 — Transform Train (pasos 9–16):
    9.  HorizontalFlip(p=0.5)
    10. ShiftScaleRotate(shift=0.06, scale=(-0.15,0.10), rotate=0, p=0.5)
    11. Rotate(limit=10, p=0.5)
    12. ElasticTransform(alpha=30, sigma=5, p=0.1)
    13. RandomBrightnessContrast(brightness=0.15, contrast=0.15, p=0.5)
    14. GaussianBlur(blur_limit=(3,5), p=0.1)
    15. GaussNoise(var_limit=(0.009²,0.022²), p=0.1) — std en [0,1]
    16. CoarseDropout(holes=1–3, h=8–24, w=8–24, fill=0, p=0.15)

Paso 17 — Normalización (train + val/test):
    A.Normalize(mean=[μ], std=[σ]) con stats de stats.json (canal único)

Paso 18 — Val/Test:
    Solo Normalize + ToTensorV2 (sin augmentación estocástica)

Dependencias:
    - src/pipeline/datasets/chest.py: ChestXray14Dataset
    - src/pipeline/fase2/expert1_config.py: EXPERT1_BATCH_SIZE, EXPERT1_NUM_WORKERS
    - datasets/nih_chest_xrays/preprocessed/stats.json: {"mean": float, "std": float}

Uso:
    from dataloader_expert1 import build_expert1_dataloaders
    loaders = build_expert1_dataloaders()
    train_loader = loaders["train"]
    test_loader  = loaders["test"]
    test_flip    = loaders["test_flip"]  # TTA con HorizontalFlip=1.0
    pos_weight   = loaders["pos_weight"]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, Subset

# ── Agregar src/pipeline al path para imports ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from datasets.chest import ChestXray14Dataset
from fase2.expert1_config import EXPERT1_BATCH_SIZE, EXPERT1_NUM_WORKERS

log = logging.getLogger("expert1_dataloader")

# ── Path por defecto al stats.json ────────────────────────────────────
_DEFAULT_STATS_PATH = (
    _PROJECT_ROOT / "datasets" / "nih_chest_xrays" / "preprocessed" / "stats.json"
)
_DEFAULT_PREPROCESSED_DIR = (
    _PROJECT_ROOT / "datasets" / "nih_chest_xrays" / "preprocessed"
)


# ── Carga de estadísticas ─────────────────────────────────────────────


def _load_dataset_stats(
    stats_path: str | Path = _DEFAULT_STATS_PATH,
) -> tuple[list[float], list[float]]:
    """Lee mean y std desde stats.json generado por pre_chestxray14.py.

    Returns:
        (mean, std) como listas de un solo elemento para canal único.

    Raises:
        RuntimeError: si el archivo no existe o tiene formato inválido.
    """
    stats_path = Path(stats_path)
    if not stats_path.is_file():
        raise RuntimeError(
            f"[Expert1/DataLoader] stats.json no encontrado: '{stats_path}'. "
            f"Ejecuta pre_chestxray14.py primero para calcular las estadísticas "
            f"del dataset preprocesado."
        )

    with open(stats_path) as f:
        stats = json.load(f)

    if "mean" not in stats or "std" not in stats:
        raise RuntimeError(
            f"[Expert1/DataLoader] stats.json debe contener 'mean' y 'std'. "
            f"Contenido actual: {stats}. Regenera con pre_chestxray14.py."
        )

    mean = [float(stats["mean"])]
    std = [float(stats["std"])]
    log.info(
        f"[Expert1/DataLoader] Dataset stats: mean={mean[0]:.6f}, std={std[0]:.6f}"
    )
    return mean, std


# ── Transforms (Albumentations) ───────────────────────────────────────


def _build_train_transform(
    mean: list[float],
    std: list[float],
) -> A.Compose:
    """Transform de entrenamiento — Fase 2, pasos 9–17.

    8 augmentaciones estocásticas (solo train) + normalización con stats
    del propio dataset (canal único).
    """
    return A.Compose(
        [
            # 9. HorizontalFlip — solo horizontal, NO vertical
            A.HorizontalFlip(p=0.5),
            # 10. ShiftScaleRotate — traslación + escala, sin rotación (rotate_limit=0)
            A.ShiftScaleRotate(
                shift_limit=0.06,
                scale_limit=(-0.15, 0.10),
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
            # 11. Rotate — rotación leve, separada de ShiftScaleRotate
            A.Rotate(
                limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
            # 12. ElasticTransform — deformación elástica suave
            A.ElasticTransform(alpha=30, sigma=5, p=0.1),
            # 13. RandomBrightnessContrast — ajuste de brillo/contraste
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5,
            ),
            # 14. GaussianBlur — blur leve
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            # 15. GaussNoise — ruido gaussiano, std ∈ [0.009, 0.022] en escala [0,1]
            #     var_limit espera varianza (σ²), así que (0.009², 0.022²)
            A.GaussNoise(var_limit=(0.009**2, 0.022**2), p=0.1),
            # 16. CoarseDropout — cutout con huecos pequeños
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(8, 24),
                hole_width_range=(8, 24),
                fill=0,
                p=0.15,
            ),
            # 17. Normalización con stats del propio dataset (canal único)
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def _build_val_transform(
    mean: list[float],
    std: list[float],
) -> A.Compose:
    """Transform de validación/test — Paso 18: solo normalización + tensor."""
    return A.Compose(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def _build_flip_transform(
    mean: list[float],
    std: list[float],
) -> A.Compose:
    """Transform para TTA: HorizontalFlip determinista (p=1.0) + normalización."""
    return A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


# ── API pública ───────────────────────────────────────────────────────


def build_expert1_dataloaders(
    preprocessed_dir: str | Path = _DEFAULT_PREPROCESSED_DIR,
    stats_path: str | Path | None = None,
    batch_size: int = EXPERT1_BATCH_SIZE,
    num_workers: int = EXPERT1_NUM_WORKERS,
    max_samples: int | None = None,
) -> dict[str, DataLoader | torch.Tensor]:
    """Construye DataLoaders para train, val, test y test_flip (TTA).

    Lee imágenes preprocesadas desde ``preprocessed_dir/{train,val,test}/``
    y normaliza con estadísticas de ``stats.json``.

    Args:
        preprocessed_dir: directorio raíz con subcarpetas train/, val/, test/
            y stats.json. Default: datasets/nih_chest_xrays/preprocessed/.
        stats_path: ruta al stats.json. Si None, se busca en
            ``preprocessed_dir/stats.json``.
        batch_size: tamaño de batch por GPU. Default: EXPERT1_BATCH_SIZE.
        num_workers: workers para DataLoader. Default: EXPERT1_NUM_WORKERS.
        max_samples: si se proporciona, limita cada dataset a este número de
            muestras. Útil para dry-run rápido. Default: None (todo el dataset).

    Returns:
        dict con claves:
            - ``'train'``: DataLoader de entrenamiento (shuffle=True, drop_last=True).
            - ``'val'``: DataLoader de validación.
            - ``'test'``: DataLoader de test (sin augmentation).
            - ``'test_flip'``: DataLoader de test con HorizontalFlip=1.0 para TTA.
            - ``'pos_weight'``: Tensor[14] con n_neg/n_pos por clase para
              BCEWithLogitsLoss.

    Raises:
        RuntimeError: si stats.json no existe o el directorio preprocesado
            no está disponible.
    """
    preprocessed_dir = Path(preprocessed_dir)

    # ── Resolver stats_path ────────────────────────────────────────────
    if stats_path is None:
        stats_path = preprocessed_dir / "stats.json"

    # ── Cargar estadísticas del dataset ────────────────────────────────
    mean, std = _load_dataset_stats(stats_path)

    log.info(f"[Expert1/DataLoader] preprocessed_dir: {preprocessed_dir}")
    log.info(f"[Expert1/DataLoader] Batch: {batch_size} | Workers: {num_workers}")

    # ── Builds transforms ──────────────────────────────────────────────
    train_tfm = _build_train_transform(mean, std)
    val_tfm = _build_val_transform(mean, std)
    flip_tfm = _build_flip_transform(mean, std)

    # ── Crear datasets ─────────────────────────────────────────────────
    train_ds = ChestXray14Dataset(
        split="train",
        preprocessed_dir=str(preprocessed_dir),
        transform=train_tfm,
        mode="expert",
    )

    val_ds = ChestXray14Dataset(
        split="val",
        preprocessed_dir=str(preprocessed_dir),
        transform=val_tfm,
        mode="expert",
    )

    test_ds = ChestXray14Dataset(
        split="test",
        preprocessed_dir=str(preprocessed_dir),
        transform=val_tfm,
        mode="expert",
    )

    test_flip_ds = ChestXray14Dataset(
        split="test",
        preprocessed_dir=str(preprocessed_dir),
        transform=flip_tfm,
        mode="expert",
    )

    # ── Extraer pos_weight ANTES de hacer Subset (necesita dataset completo) ─
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

    # ── Limitar muestras si max_samples está activo (dry-run) ──────────
    train_ds_final: Dataset = train_ds
    val_ds_final: Dataset = val_ds
    test_ds_final: Dataset = test_ds
    test_flip_ds_final: Dataset = test_flip_ds

    if max_samples is not None:
        log.info(
            f"[Expert1/DataLoader] max_samples={max_samples} — "
            f"cada dataset limitado a {max_samples} muestras"
        )
        for name, ds in [
            ("train", train_ds),
            ("val", val_ds),
            ("test", test_ds),
            ("test_flip", test_flip_ds),
        ]:
            log.info(
                f"[Expert1/DataLoader] {name}: {len(ds):,} → "
                f"{min(max_samples, len(ds)):,} muestras"
            )
        n_train = min(max_samples, len(train_ds))
        n_val = min(max_samples, len(val_ds))
        n_test = min(max_samples, len(test_ds))
        n_test_flip = min(max_samples, len(test_flip_ds))
        train_ds_final = Subset(train_ds, list(range(n_train)))
        val_ds_final = Subset(val_ds, list(range(n_val)))
        test_ds_final = Subset(test_ds, list(range(n_test)))
        test_flip_ds_final = Subset(test_flip_ds, list(range(n_test_flip)))

    # ── Crear DataLoaders ──────────────────────────────────────────────
    common_kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_ds_final,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **common_kwargs,  # type: ignore[arg-type]
    )

    val_loader = DataLoader(
        val_ds_final,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,  # type: ignore[arg-type]
    )

    test_loader = DataLoader(
        test_ds_final,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,  # type: ignore[arg-type]
    )

    test_flip_loader = DataLoader(
        test_flip_ds_final,
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
    preprocessed_dir: str | Path | None = None,
    stats_path: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Wrapper de compatibilidad con la API anterior.

    Convierte la llamada legacy (positional 4-tupla) a la nueva API (dict).
    Los parámetros legacy (csv_path, img_dir, train_list, val_list, test_list,
    model_mean, model_std) se ignoran con un warning — la nueva API lee
    desde preprocessed_dir y stats.json.

    Returns:
        (train_loader, val_loader, test_loader, pos_weight)
    """
    # Warn about legacy params
    _legacy = {
        k: v
        for k, v in [
            ("csv_path", csv_path),
            ("img_dir", img_dir),
            ("train_list", train_list),
            ("val_list", val_list),
            ("test_list", test_list),
            ("model_mean", model_mean),
            ("model_std", model_std),
        ]
        if v is not None
    }
    if _legacy:
        log.warning(
            f"[Expert1/DataLoader] Parámetros legacy ignorados: {list(_legacy.keys())}. "
            f"El dataloader ahora lee desde preprocessed_dir + stats.json."
        )

    _preprocessed = preprocessed_dir or _DEFAULT_PREPROCESSED_DIR

    result = build_expert1_dataloaders(
        preprocessed_dir=_preprocessed,
        stats_path=stats_path,
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
            "train-aug(8steps)"
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

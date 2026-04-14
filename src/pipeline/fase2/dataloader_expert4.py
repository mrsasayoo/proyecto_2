"""
DataLoader para Expert 4 — Páncreas CT 3D (ResNet 3D / R3D-18).

Construye DataLoaders para train y val del Expert 4 con k-fold CV.
Usa PancreasDataset de datasets/pancreas.py con mode="expert".

Estrategia de splits:
    1. Carga pancreas_splits.csv para obtener (case_id, label, split)
    2. Filtra los volúmenes .nii.gz que existen en disco
    3. Genera k-fold splits estratificados con StratifiedGroupKFold
       (agrupando por patient_id para evitar leakage)
    4. Retorna train_loader y val_loader para el fold especificado

Dependencias:
    - src/pipeline/datasets/pancreas.py: PancreasDataset, PancreasROIExtractor
    - src/pipeline/fase2/expert4_config.py: hiperparámetros

Uso:
    from dataloader_expert4 import build_dataloaders_expert4
    train_loader, val_loader = build_dataloaders_expert4(fold=0)
"""

import sys
import time
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── Agregar src/pipeline al path para imports ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from datasets.pancreas import PancreasDataset
from fase2.expert4_config import EXPERT4_BATCH_SIZE, EXPERT4_NUM_FOLDS

log = logging.getLogger("expert4_dataloader")

# ── Rutas de datos ─────────────────────────────────────────────────────
_SPLITS_CSV = _PROJECT_ROOT / "datasets" / "pancreas_splits.csv"
_LABELS_CSV = _PROJECT_ROOT / "datasets" / "pancreas_labels_binary.csv"
# Directorio raíz donde buscar volúmenes .nii.gz del páncreas
_VOLUMES_DIR = _PROJECT_ROOT / "datasets" / "zenodo_13715870"

_NUM_WORKERS = 2  # Volúmenes grandes → I/O limitante, 2 workers recomendado


def _load_valid_pairs_from_splits(
    splits_csv: Path | None = None,
    labels_csv: Path | None = None,
    volumes_dir: Path | None = None,
) -> list[tuple[str, int]]:
    """
    Carga pares (nii_path, label) desde pancreas_splits.csv + disco.

    Busca volúmenes .nii.gz que correspondan a los case_ids del CSV.
    Si no se encuentran volúmenes en disco, retorna pares sintéticos
    con paths del CSV (para dry-run sin datos reales).

    Returns:
        list de (nii_path_str, label_int)
    """
    splits_csv = splits_csv or _SPLITS_CSV
    labels_csv = labels_csv or _LABELS_CSV
    volumes_dir = volumes_dir or _VOLUMES_DIR

    # Cargar CSV de splits
    if splits_csv.exists():
        df = pd.read_csv(splits_csv)
        log.info(
            f"[Expert4/DataLoader] CSV de splits cargado: {splits_csv.name} | "
            f"{len(df):,} entradas"
        )
    elif labels_csv.exists():
        df = pd.read_csv(labels_csv)
        log.info(
            f"[Expert4/DataLoader] Usando labels CSV: {labels_csv.name} | "
            f"{len(df):,} entradas"
        )
    else:
        log.error(
            f"[Expert4/DataLoader] No se encontró ningún CSV de splits.\n"
            f"  Esperados: {splits_csv}\n"
            f"  Fallback:  {labels_csv}"
        )
        return []

    # Construir índice de case_id → label desde el CSV
    label_map: dict[str, int] = {}
    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        label = int(row["label"])
        label_map[case_id] = label

    # Buscar volúmenes en disco
    valid_pairs: list[tuple[str, int]] = []

    if volumes_dir.exists():
        nii_files = sorted(volumes_dir.rglob("*.nii.gz"))
        if not nii_files:
            nii_files = sorted(volumes_dir.rglob("*.nii"))
        log.info(
            f"[Expert4/DataLoader] Volúmenes en disco: {len(nii_files):,} en {volumes_dir}"
        )

        for nii_path in nii_files:
            # Extraer case_id del nombre: e.g. 100047_00001_0000.nii.gz → 100047_00001
            stem = nii_path.name
            for ext in [".nii.gz", ".nii"]:
                if stem.endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            # Intentar match con y sin sufijo nnU-Net (_0000)
            import re

            normalized = re.sub(r"_\d{4}$", "", stem)

            label = label_map.get(stem) or label_map.get(normalized)
            if label is not None:
                valid_pairs.append((str(nii_path), label))
    else:
        log.warning(
            f"[Expert4/DataLoader] Directorio de volúmenes no encontrado: {volumes_dir}. "
            f"Creando pares sintéticos desde CSV (solo para dry-run)."
        )
        # Pares sintéticos para dry-run (paths no existen, PancreasDataset
        # retornará tensores cero en __getitem__ al fallar la lectura)
        for case_id, label in label_map.items():
            fake_path = str(volumes_dir / f"{case_id}_0000.nii.gz")
            valid_pairs.append((fake_path, label))

    n_pos = sum(1 for _, l in valid_pairs if l == 1)
    n_neg = sum(1 for _, l in valid_pairs if l == 0)
    log.info(
        f"[Expert4/DataLoader] Pares válidos: {len(valid_pairs):,} | "
        f"PDAC+: {n_pos:,} | PDAC-: {n_neg:,} | "
        f"ratio: {n_neg / max(n_pos, 1):.1f}:1"
    )

    return valid_pairs


def build_dataloaders_expert4(
    fold: int = 0,
    num_folds: int | None = None,
    splits_csv: str | Path | None = None,
    labels_csv: str | Path | None = None,
    volumes_dir: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int = _NUM_WORKERS,
    roi_strategy: str = "B",
) -> tuple[DataLoader, DataLoader]:
    """
    Construye DataLoaders para train y val del Expert 4 (Páncreas) en un fold.

    Usa PancreasDataset.build_kfold_splits() para generar k-fold splits
    estratificados agrupados por patient_id.

    Args:
        fold: índice del fold (0 a num_folds-1). Default: 0.
        num_folds: número total de folds. Default: EXPERT4_NUM_FOLDS (5).
        splits_csv: ruta al CSV de splits. Default: datasets/pancreas_splits.csv.
        labels_csv: ruta al CSV de labels. Default: datasets/pancreas_labels_binary.csv.
        volumes_dir: directorio con volúmenes .nii.gz. Default: datasets/zenodo_13715870/.
        batch_size: batch size por GPU. Default: EXPERT4_BATCH_SIZE (2).
        num_workers: workers para DataLoader. Default: 2.
        roi_strategy: estrategia ROI ("A" o "B"). Default: "B".

    Returns:
        (train_loader, val_loader)

    Raises:
        ValueError: si fold >= num_folds o no hay pares válidos.
    """
    num_folds = num_folds or EXPERT4_NUM_FOLDS
    batch_size = batch_size or EXPERT4_BATCH_SIZE

    if fold >= num_folds or fold < 0:
        raise ValueError(
            f"[Expert4/DataLoader] fold={fold} inválido. Debe ser 0 <= fold < {num_folds}"
        )

    # ── Cargar pares válidos ───────────────────────────────────────────
    splits_path = Path(splits_csv) if splits_csv else None
    labels_path = Path(labels_csv) if labels_csv else None
    vols_path = Path(volumes_dir) if volumes_dir else None
    valid_pairs = _load_valid_pairs_from_splits(splits_path, labels_path, vols_path)

    if not valid_pairs:
        raise ValueError(
            "[Expert4/DataLoader] No se encontraron pares válidos. "
            "Verifica que existan volúmenes .nii.gz y el CSV de splits/labels."
        )

    # ── Generar k-fold splits ──────────────────────────────────────────
    t0 = time.time()
    folds = PancreasDataset.build_kfold_splits(valid_pairs, k=num_folds)
    elapsed = time.time() - t0
    log.info(
        f"[Expert4/DataLoader] k-fold splits generados: {num_folds} folds en {elapsed:.2f}s"
    )

    train_pairs, val_pairs = folds[fold]

    n_pos_tr = sum(1 for _, l in train_pairs if l == 1)
    n_neg_tr = sum(1 for _, l in train_pairs if l == 0)
    n_pos_va = sum(1 for _, l in val_pairs if l == 1)
    n_neg_va = sum(1 for _, l in val_pairs if l == 0)

    log.info(
        f"[Expert4/DataLoader] Fold {fold}/{num_folds - 1}:\n"
        f"    Train: {len(train_pairs):,} (PDAC+={n_pos_tr}, PDAC-={n_neg_tr})\n"
        f"    Val:   {len(val_pairs):,} (PDAC+={n_pos_va}, PDAC-={n_neg_va})"
    )

    # ── Crear datasets ─────────────────────────────────────────────────
    train_ds = PancreasDataset(
        valid_pairs=train_pairs,
        mode="expert",
        roi_strategy=roi_strategy,
        z_score_per_volume=True,
        split="train",
        augment_3d=True,
    )

    val_ds = PancreasDataset(
        valid_pairs=val_pairs,
        mode="expert",
        roi_strategy=roi_strategy,
        z_score_per_volume=True,
        split="val",
        augment_3d=False,
    )

    # ── Crear DataLoaders ──────────────────────────────────────────────
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )

    # ── Resumen ────────────────────────────────────────────────────────
    _print_summary(train_ds, val_ds, train_loader, val_loader, fold, num_folds)

    return train_loader, val_loader


def _print_summary(
    train_ds: PancreasDataset,
    val_ds: PancreasDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold: int,
    num_folds: int,
) -> None:
    """Imprime resumen de los datasets y dataloaders."""
    print("=" * 70)
    print(f"  RESUMEN — DataLoaders Expert 4 (Páncreas) — Fold {fold}/{num_folds - 1}")
    print("=" * 70)

    for name, ds, loader in [
        ("train", train_ds, train_loader),
        ("val", val_ds, val_loader),
    ]:
        n_pos = sum(1 for _, l in ds.samples if l == 1)
        n_neg = sum(1 for _, l in ds.samples if l == 0)
        ratio = n_neg / max(n_pos, 1)
        aug_status = "SI" if ds._apply_augment else "NO"
        print(
            f"  {name:5s}: {len(ds):>6,} muestras | "
            f"PDAC+={n_pos:>4,} | PDAC-={n_neg:>4,} | "
            f"ratio={ratio:>5.1f}:1 | "
            f"batches={len(loader):>4,} | aug={aug_status}"
        )

    print("=" * 70)


def _test_dataloaders():
    """Verificación rápida: cargar un batch del fold 0."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        train_loader, val_loader = build_dataloaders_expert4(
            fold=0,
            num_workers=0,
        )

        for name, loader in [("train", train_loader), ("val", val_loader)]:
            batch = next(iter(loader))
            volume, label, stem = batch
            print(f"\n[Expert4/DataLoader] Primer batch de {name}:")
            print(f"  volume shape: {list(volume.shape)}")
            print(f"  volume dtype:  {volume.dtype}")
            print(f"  volume min/max: {volume.min():.4f} / {volume.max():.4f}")
            print(f"  labels: {label}")
            print(f"  stems: {stem[:2]}...")
    except Exception as e:
        print(f"[Expert4/DataLoader] Error en test: {e}")
        print("  (Esperado si no hay volúmenes .nii.gz en disco)")


if __name__ == "__main__":
    _test_dataloaders()

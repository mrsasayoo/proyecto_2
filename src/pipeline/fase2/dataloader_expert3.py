"""
DataLoader para Expert 3 — LUNA16 (parches 3D de nódulos pulmonares).

Construye DataLoaders para train, val y test del Expert 3 (DenseNet 3D).
Los parches ya están extraídos y separados por split en disco.

Estrategia de carga eficiente:
    En lugar de iterar las 754K filas de candidates_V2.csv buscando archivos,
    se escanean los archivos .npy en disco y se busca su label en un dict
    indexado por el número de candidato. Esto reduce el init de ~3 minutos
    a ~1 segundo por split.

    La augmentation 3D se importa de LUNA16Dataset._augment_3d para reutilizar
    la misma lógica validada (flip, rotación, variación HU, ruido).

Dependencias:
    - src/pipeline/datasets/luna.py: LUNA16Dataset (se reutiliza _augment_3d)
    - src/pipeline/fase2/expert3_config.py: hiperparámetros

Uso:
    from dataloader_expert3 import build_dataloaders_expert3
    train_loader, val_loader, test_loader = build_dataloaders_expert3()
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ── Agregar src/pipeline al path para imports ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from datasets.luna import LUNA16Dataset
from fase2.expert3_config import EXPERT3_BATCH_SIZE

log = logging.getLogger("expert3_dataloader")

# ── Rutas de datos ─────────────────────────────────────────────────────
_PATCHES_BASE = _PROJECT_ROOT / "datasets" / "luna_lung_cancer" / "patches"
_CANDIDATES_V2_CSV = (
    _PROJECT_ROOT
    / "datasets"
    / "luna_lung_cancer"
    / "candidates_V2"
    / "candidates_V2.csv"
)
# Fallback si V2 no existe
_CANDIDATES_V1_CSV = _PROJECT_ROOT / "datasets" / "luna_lung_cancer" / "candidates.csv"

_NUM_WORKERS = 4


class LUNA16ExpertDataset(Dataset):
    """
    Dataset ligero para entrenamiento del Expert 3 (mode="expert").

    A diferencia de LUNA16Dataset (que itera todo el CSV buscando archivos),
    este dataset escanea los .npy en disco y busca sus labels en un dict
    pre-computado. Resultado: init ~1 segundo vs ~3 minutos.

    Reutiliza la lógica de augmentation 3D de LUNA16Dataset para consistencia.

    Args:
        patches_dir: carpeta con archivos .npy del split
        label_map: dict {candidate_index: class_label}
        split: "train", "val" o "test"
        augment_3d: aplicar augmentation 3D (solo en split="train")
    """

    def __init__(
        self,
        patches_dir: str | Path,
        label_map: dict[int, int],
        split: str = "train",
        augment_3d: bool = True,
    ):
        self.patches_dir = Path(patches_dir)
        self.split = split
        self._apply_augment = split == "train" and augment_3d

        # Instanciar un LUNA16Dataset dummy solo para reutilizar _augment_3d
        # No cargamos datos reales aquí — solo necesitamos el método
        self._augmenter = LUNA16Dataset.__new__(LUNA16Dataset)
        self._augmenter._apply_augment = self._apply_augment

        # Escanear archivos y mapear labels
        t0 = time.time()
        self.samples: list[tuple[Path, int]] = []
        n_unknown = 0

        for npy_file in sorted(self.patches_dir.glob("candidate_*.npy")):
            try:
                idx = int(npy_file.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            label = label_map.get(idx, -1)
            if label >= 0:
                self.samples.append((npy_file, label))
            else:
                n_unknown += 1

        elapsed = time.time() - t0
        n_pos = sum(1 for _, c in self.samples if c == 1)
        n_neg = sum(1 for _, c in self.samples if c == 0)

        log.info(
            f"[LUNA16Expert/{split}] {len(self.samples):,} parches cargados en {elapsed:.2f}s | "
            f"pos={n_pos:,} | neg={n_neg:,} | ratio={n_neg / max(n_pos, 1):.1f}:1"
            f"{f' | {n_unknown} sin label (ignorados)' if n_unknown else ''}"
        )
        if self._apply_augment:
            log.info(
                f"[LUNA16Expert/{split}] Augmentation 3D ACTIVO: "
                "flip 3 ejes, rotación ±15°, variación HU, ruido gaussiano, desplazamiento espacial ±4 vóx"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        patch_file, label = self.samples[idx]

        try:
            volume = np.load(patch_file)
        except Exception as e:
            log.warning(f"[LUNA16Expert] Error cargando '{patch_file}': {e}")
            return torch.zeros(1, 64, 64, 64), label, patch_file.stem

        # Augmentation 3D (solo train)
        if self._apply_augment:
            volume = self._augmenter._augment_3d(volume)

        # Convertir a tensor [1, 64, 64, 64]
        volume_t = torch.from_numpy(
            np.ascontiguousarray(volume, dtype=np.float32)
        ).unsqueeze(0)

        return volume_t, label, patch_file.stem


def _resolve_csv_path(candidates_csv: str | Path | None) -> Path:
    """
    Resuelve la ruta al CSV de candidatos, priorizando V2 sobre V1.

    Returns:
        Path al CSV existente.
    Raises:
        FileNotFoundError si no se encuentra ningún CSV.
    """
    if candidates_csv is not None:
        path = Path(candidates_csv)
        if path.exists():
            return path
        raise FileNotFoundError(f"CSV de candidatos no encontrado: {path}")

    # Priorizar V2
    if _CANDIDATES_V2_CSV.exists():
        return _CANDIDATES_V2_CSV
    if _CANDIDATES_V1_CSV.exists():
        log.warning(
            "[Expert3/DataLoader] candidates_V2.csv no encontrado. "
            "Usando candidates.csv (V1) como fallback. "
            "Los conteos de positivos pueden ser incorrectos."
        )
        return _CANDIDATES_V1_CSV

    raise FileNotFoundError(
        "[Expert3/DataLoader] No se encontró ningún CSV de candidatos.\n"
        f"  V2 esperado: {_CANDIDATES_V2_CSV}\n"
        f"  V1 esperado: {_CANDIDATES_V1_CSV}"
    )


def _load_label_map(csv_path: Path) -> dict[int, int]:
    """Carga el CSV y retorna un dict {index_fila: class_label}."""
    t0 = time.time()
    df = pd.read_csv(csv_path)
    label_map = dict(zip(df.index, df["class"]))
    elapsed = time.time() - t0
    n_pos = (df["class"] == 1).sum()
    n_neg = (df["class"] == 0).sum()
    log.info(
        f"[Expert3/DataLoader] CSV cargado: {csv_path.name} | "
        f"{len(df):,} candidatos (pos={n_pos:,}, neg={n_neg:,}) | {elapsed:.2f}s"
    )
    return label_map


def build_dataloaders_expert3(
    patches_base: str | Path | None = None,
    candidates_csv: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int = _NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construye DataLoaders para train, val y test del Expert 3 (LUNA16).

    Usa LUNA16ExpertDataset (scan de disco + lookup de labels) en lugar de
    LUNA16Dataset (iteración fila por fila del CSV) para eficiencia.

    Args:
        patches_base: directorio raíz con subcarpetas {train, val, test}/.
            Default: datasets/luna_lung_cancer/patches/
        candidates_csv: ruta al CSV de candidatos. Default: candidates_V2.csv
        batch_size: tamaño de batch por GPU. Default: EXPERT3_BATCH_SIZE (4).
        num_workers: número de workers para DataLoader. Default: 4.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    patches_base = Path(patches_base) if patches_base else _PATCHES_BASE
    batch_size = batch_size or EXPERT3_BATCH_SIZE

    # ── Resolver CSV ───────────────────────────────────────────────────
    csv_path = _resolve_csv_path(candidates_csv)

    # ── Verificar directorios ──────────────────────────────────────────
    for split_name in ("train", "val", "test"):
        split_dir = patches_base / split_name
        if not split_dir.exists():
            raise FileNotFoundError(
                f"[Expert3/DataLoader] Directorio no encontrado: {split_dir}"
            )

    log.info(f"[Expert3/DataLoader] Patches: {patches_base}")
    log.info(f"[Expert3/DataLoader] CSV: {csv_path}")
    log.info(f"[Expert3/DataLoader] Batch: {batch_size} | Workers: {num_workers}")

    # ── Cargar labels ──────────────────────────────────────────────────
    label_map = _load_label_map(csv_path)

    # ── Crear datasets ─────────────────────────────────────────────────
    train_ds = LUNA16ExpertDataset(
        patches_dir=patches_base / "train",
        label_map=label_map,
        split="train",
        augment_3d=True,
    )

    val_ds = LUNA16ExpertDataset(
        patches_dir=patches_base / "val",
        label_map=label_map,
        split="val",
        augment_3d=False,
    )

    test_ds = LUNA16ExpertDataset(
        patches_dir=patches_base / "test",
        label_map=label_map,
        split="test",
        augment_3d=False,
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

    return train_loader, val_loader, test_loader


def _print_summary(
    train_ds: LUNA16ExpertDataset,
    val_ds: LUNA16ExpertDataset,
    test_ds: LUNA16ExpertDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    """Imprime resumen de los datasets y dataloaders."""
    print("=" * 70)
    print("  RESUMEN — DataLoaders Expert 3 (LUNA16)")
    print("=" * 70)

    for name, ds, loader in [
        ("train", train_ds, train_loader),
        ("val", val_ds, val_loader),
        ("test", test_ds, test_loader),
    ]:
        n_pos = sum(1 for _, c in ds.samples if c == 1)
        n_neg = sum(1 for _, c in ds.samples if c == 0)
        ratio = n_neg / max(n_pos, 1)
        aug_status = "SÍ" if ds._apply_augment else "NO"
        print(
            f"  {name:5s}: {len(ds):>6,} muestras | "
            f"pos={n_pos:>5,} | neg={n_neg:>5,} | "
            f"ratio={ratio:>5.1f}:1 | "
            f"batches={len(loader):>4,} | aug={aug_status}"
        )

    print("=" * 70)


def _test_dataloaders():
    """Verificación rápida: cargar un batch de cada split."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    train_loader, val_loader, test_loader = build_dataloaders_expert3(num_workers=0)

    for name, loader in [("train", train_loader), ("val", val_loader)]:
        batch = next(iter(loader))
        volume, label, stem = batch
        print(f"\n[Expert3/DataLoader] Primer batch de {name}:")
        print(f"  volume shape: {list(volume.shape)}")
        print(f"  volume dtype:  {volume.dtype}")
        print(f"  volume min/max: {volume.min():.4f} / {volume.max():.4f}")
        print(f"  labels: {label}")
        print(f"  stems: {stem[:2]}...")


if __name__ == "__main__":
    _test_dataloaders()

"""
DataLoaders para el CAE multimodal (Expert 5, Fase 3).

Construye DataLoaders de train y val a partir de cae_splits.csv
usando MultimodalCAEDataset.
"""

import logging
from pathlib import Path

from torch.utils.data import DataLoader

from datasets.cae import MultimodalCAEDataset

log = logging.getLogger("cae_dataloader")


def get_cae_dataloaders(
    csv_path: str,
    project_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
) -> tuple:
    """
    Construye DataLoaders de train y val para el CAE.

    Args:
        csv_path: ruta a cae_splits.csv
        project_root: raíz del proyecto para resolver paths relativos
        batch_size: batch size (default=32)
        num_workers: workers para DataLoader (default=4)
        img_size: tamaño de imagen de salida (default=224)

    Returns:
        (train_loader, val_loader)
    """
    train_ds = MultimodalCAEDataset(
        csv_path=csv_path,
        split="train",
        img_size=img_size,
        project_root=project_root,
    )
    val_ds = MultimodalCAEDataset(
        csv_path=csv_path,
        split="val",
        img_size=img_size,
        project_root=project_root,
    )

    log.info(
        f"[CAE DataLoader] train: {len(train_ds):,} muestras | "
        f"val: {len(val_ds):,} muestras | "
        f"batch_size: {batch_size} | num_workers: {num_workers}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

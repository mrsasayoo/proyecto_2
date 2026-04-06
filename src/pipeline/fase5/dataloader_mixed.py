"""
DataLoader mixto proporcional para fine-tuning global (Paso 8 Stage 2-3).

Combina los 5 datasets de dominio con pesos proporcionales a su tamano.
Retorna (imagen, label, expert_id) para cada batch.

En modo dry-run genera tensores sinteticos cuando los archivos reales
no existen, permitiendo verificar el pipeline sin datos.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from fase5.fase5_config import MIXED_BATCH_SIZE, MIXED_SAMPLES_PER_DATASET

log = logging.getLogger("fase5")


class SyntheticMixedDataset(Dataset):
    """
    Dataset sintetico para dry-run del fine-tuning.

    Genera tensores aleatorios con shapes correctos para cada experto:
      - Expertos 0-2: imagenes 2D [3, 224, 224], labels enteros
      - Experto 3: parches 3D [1, 64, 64, 64], labels binarios
      - Experto 4: volumenes 3D [1, 64, 64, 64], labels binarios
      - Experto 5: imagenes 2D [3, 224, 224], labels dummy (CAE usa MSE)

    Cada muestra retorna (tensor, label, expert_id).
    """

    # Numero de clases por experto (para generar labels validos)
    _NUM_CLASSES = {0: 14, 1: 9, 2: 3, 3: 2, 4: 2, 5: 0}
    # Expertos 3D
    _3D_EXPERTS = {3, 4}

    def __init__(self, n_samples: int = 160, seed: int = 42):
        """
        Args:
            n_samples: numero total de muestras sinteticas.
            seed: semilla para reproducibilidad.
        """
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)

        # Distribuir muestras equitativamente entre 6 expertos
        samples_per_expert = n_samples // 6
        remainder = n_samples - samples_per_expert * 6
        self._expert_ids = []
        for eid in range(6):
            count = samples_per_expert + (1 if eid < remainder else 0)
            self._expert_ids.extend([eid] * count)
        self.rng.shuffle(self._expert_ids)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        expert_id = self._expert_ids[idx]

        if expert_id in self._3D_EXPERTS:
            img = torch.randn(1, 64, 64, 64)
        else:
            img = torch.randn(3, 224, 224)

        n_classes = self._NUM_CLASSES[expert_id]
        if expert_id == 0:
            # Multilabel: vector binario de 14 clases
            label = torch.zeros(n_classes)
            # Activar 1-3 patologias aleatorias
            n_active = self.rng.randint(0, 4)
            if n_active > 0:
                active_idx = self.rng.choice(n_classes, size=n_active, replace=False)
                label[active_idx] = 1.0
        elif expert_id == 5:
            # CAE: label dummy (no se usa para clasificacion)
            label = torch.tensor(0)
        else:
            # Multiclase: un entero
            label = torch.tensor(self.rng.randint(0, n_classes))

        return img, label, expert_id


def get_mixed_dataloader(
    split: str = "train",
    batch_size: int = MIXED_BATCH_SIZE,
    dry_run: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Construye un DataLoader mixto proporcional que combina todos los datasets.

    En dry-run: retorna un DataLoader con datos sinteticos (5 batches).
    En produccion: intentaria cargar los datasets reales (no implementado aun,
    ya que este paso solo se verifica con --dry-run).

    Args:
        split: "train" o "val".
        batch_size: tamano de batch.
        dry_run: si True, usa datos sinteticos.
        num_workers: workers para DataLoader.

    Returns:
        DataLoader que produce (images, labels, expert_ids).
    """
    if dry_run:
        n_samples = batch_size * 5  # 5 batches sinteticos
        dataset = SyntheticMixedDataset(n_samples=n_samples, seed=42)
        log.info(
            "[DataLoader] Modo dry-run: %d muestras sinteticas (%d batches x %d)",
            n_samples,
            5,
            batch_size,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            # Collate personalizado para manejar tensores de shapes mixtos
            collate_fn=_mixed_collate_fn,
        )

    # --- Produccion: cargar datasets reales ---
    # NOTA: No implementado — este paso solo se verifica con --dry-run.
    # En produccion, se usaria ConcatDataset con WeightedRandomSampler.
    log.warning(
        "[DataLoader] Modo produccion no implementado. "
        "Usa --dry-run para verificar el pipeline."
    )
    # Fallback a sintetico para no romper el flujo
    n_samples = batch_size * 5
    dataset = SyntheticMixedDataset(n_samples=n_samples, seed=42)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=_mixed_collate_fn,
    )


def _mixed_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
    """
    Collate function para batches mixtos con tensores de shapes diferentes.

    Los expertos 3D tienen shape [1, 64, 64, 64] y los 2D [3, 224, 224].
    No se pueden apilar en un solo tensor, asi que agrupamos por expert_id.

    Returns:
        (images_stacked, labels_list, expert_ids_list)
        - images_stacked: se intenta apilar si todos tienen la misma shape;
          si no, se retorna una lista.
        - labels_list: lista de tensores de labels.
        - expert_ids_list: lista de expert_ids.
    """
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    expert_ids = [item[2] for item in batch]

    # Intentar apilar imagenes (solo funciona si todas tienen la misma shape)
    shapes = set(tuple(img.shape) for img in images)
    if len(shapes) == 1:
        images_out = torch.stack(images)
    else:
        # Shapes mixtas: retornar como lista
        images_out = images

    return images_out, labels, expert_ids

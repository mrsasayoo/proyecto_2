"""
Osteoarthritis Knee — Experto 2: Ordinal, 5 clases KL (0-4).

Clases:
  0 = Normal (KL0)
  1 = Dudoso (KL1)
  2 = Leve (KL2)
  3 = Moderado (KL3)
  4 = Severo (KL4)
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

from config import (
    EXPERT_IDS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OA_CLASS_NAMES,
    OA_N_CLASSES,
)

log = logging.getLogger("fase0")

# ── Transforms para EfficientNet-B3 (5 clases KL 0-4) ──────────────────
TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.Resize((256, 256), antialias=True),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

TRANSFORM_VAL = transforms.Compose(
    [
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class OAKneeDataset(Dataset):
    """
    Osteoarthritis Knee — Ordinal, 5 clases KL (0-4).

    Clases: Normal (0), Dudoso (1), Leve (2), Moderado (3), Severo (4).
    El label de cada imagen es su grado KL original sin remapeo.

    Dos modos:
      mode="embedding" → FASE 0: (img, expert_id=2, img_name)
      mode="expert"    → FASE 2: (img, kl_label_int, img_name) — aug diferenciado
    """

    def __init__(self, root_dir, split="train", img_size=224, mode="embedding"):
        assert mode in ("embedding", "expert"), (
            f"[OA] mode debe ser 'embedding' o 'expert', recibido: '{mode}'"
        )

        self.expert_id = EXPERT_IDS["oa"]
        self.img_size = img_size
        self.mode = mode
        self.split = split
        self.samples = []

        split_dir = Path(root_dir) / split
        if not split_dir.exists():
            log.error(f"[OA] Directorio de split no encontrado: {split_dir}")
            return

        # ── Leer clases desde carpetas ──────────────────────────────────
        class_counts = {}
        found_classes = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                class_id = int(class_dir.name)
            except ValueError:
                log.warning(
                    f"[OA] Carpeta con nombre no numérico ignorada: '{class_dir.name}'."
                )
                continue
            imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            for img_path in imgs:
                self.samples.append((img_path, class_id))
            class_counts[class_dir.name] = len(imgs)
            found_classes.append(class_dir.name)

        n_found = len(found_classes)
        log.info(
            f"[OA] split='{split}' mode='{mode}': {len(self.samples):,} imágenes | "
            f"clases: {found_classes}"
        )

        if len(self.samples) == 0:
            log.error(f"[OA] Dataset vacío en '{split_dir}'.")
            return

        # ── Verificar número de clases ────────────────────────────────
        if n_found != 5:
            log.warning(
                f"[OA] Se encontraron {n_found} clase(s) — se esperaban 5 "
                f"(grados KL 0-4)."
            )
        else:
            log.debug("[OA] 5 clases ordinales KL 0-4 confirmadas ✓")

        # ── Distribución por clase + pesos ────────────────────────────
        total = len(self.samples)
        counts_arr = np.zeros(OA_N_CLASSES, dtype=float)
        for k, v in class_counts.items():
            try:
                idx = int(k)
                if 0 <= idx < OA_N_CLASSES:
                    counts_arr[idx] = v
            except ValueError:
                pass

        log.info("[OA] Distribución por clase:")
        for i, (name, count) in enumerate(zip(OA_CLASS_NAMES, counts_arr)):
            bar = "█" * max(1, int(30 * count / total))
            log.info(
                f"    Clase {i} ({name:<18}): {int(count):>5,} "
                f"({100 * count / total:.1f}%) {bar}"
            )

        counts_safe = np.maximum(counts_arr, 1)
        cw = torch.tensor(total / (OA_N_CLASSES * counts_safe), dtype=torch.float32)
        self.class_weights = cw
        log.info(
            f"[OA] class_weights: "
            + " | ".join(f"Clase{i}:{cw[i]:.2f}" for i in range(OA_N_CLASSES))
        )

        # ── Transform pipeline ────────────────────────────────────────────
        # Selección automática según split: TRANSFORM_TRAIN para train,
        # TRANSFORM_VAL para val/test. Incluye Resize, crop, augmentation
        # y normalización ImageNet.
        if split == "train":
            self._transform = TRANSFORM_TRAIN
        else:
            self._transform = TRANSFORM_VAL

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, kl_class = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning(
                f"[OA] Error abriendo '{img_path}': {e}. Reemplazando con tensor cero."
            )
            dummy = torch.zeros(3, 224, 224)
            return (
                dummy,
                self.expert_id if self.mode == "embedding" else kl_class,
                str(img_path.name),
            )

        img_tensor = self._transform(img)

        if self.mode == "embedding":
            return img_tensor, self.expert_id, str(img_path.name)
        else:
            return img_tensor, kl_class, str(img_path.name)

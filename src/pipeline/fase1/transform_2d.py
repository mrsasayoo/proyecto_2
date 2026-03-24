"""
transform_2d.py — Pipeline Estándar para Imágenes 2D
=====================================================

Denominador común de todos los datasets 2D (Chest, ISIC, OA).
No tiene conocimiento de ningún dominio clínico específico.

Orden estricto: Resize (PIL) → ToTensor (HWC→CHW, [0,1]) → Normalize (ImageNet).
"""

import logging

import torch
from torchvision import transforms

from fase1_config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

log = logging.getLogger("fase1")


def build_2d_transform(img_size=IMG_SIZE):
    """
    Transform estándar para todos los datasets 2D.

    PIL opera sobre su representación comprimida interna — redimensionar
    en PIL antes de convertir a tensor evita copias de arrays flotantes
    grandes. ToTensor precede a Normalize porque este último asume [0,1].
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def inspect_tensor_stats(tensor):
    """
    Verifica que un tensor transformado tiene distribución razonable
    post-normalización ImageNet. Útil para diagnóstico cuando los
    embeddings tienen normas anómalamente bajas.

    Returns:
        dict con media y std por canal
    """
    if tensor.dim() != 3 or tensor.shape[0] != 3:
        log.warning("[inspect] Tensor no es [3, H, W]: %s", tensor.shape)
        return {}

    stats = {}
    for c in range(3):
        ch = tensor[c]
        stats[f"ch{c}"] = {"mean": ch.mean().item(), "std": ch.std().item()}
    return stats

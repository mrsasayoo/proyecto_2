"""
transform_3d.py — Procesamiento de Volúmenes CT 3D
===================================================

Normalización HU, interpolación trilineal, proyección multiplanar 3D→2D.
Transforma volúmenes [D, H, W] al formato [3, 224, 224] que el backbone
ViT puede procesar.
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    from fase1_config import IMAGENET_MEAN, IMAGENET_STD, PATCH_3D_SIZE, IMG_SIZE
except ModuleNotFoundError:
    from fase1.fase1_config import IMAGENET_MEAN, IMAGENET_STD, PATCH_3D_SIZE, IMG_SIZE


def normalize_hu(volume, min_hu=-1000, max_hu=400):
    """
    Clipping al rango HU clínico + normalización lineal a [0, 1].

    Recibe límites como parámetros explícitos para servir a
    LUNA16 (pulmonar: -1000, 400) y Páncreas (abdominal: -100, 400)
    con la misma función.
    """
    volume = np.clip(volume, min_hu, max_hu)
    volume = (volume - min_hu) / (max_hu - min_hu)
    return volume.astype(np.float32)


def resize_volume_3d(volume, target=PATCH_3D_SIZE):
    """
    Resize trilineal de volumen [D, H, W] a tamaño cúbico objetivo.

    La interpolación trilineal preserva la continuidad anatómica de
    estructuras tubulares que nearest-neighbor fragmentaría.

    Returns:
        tensor [1, D', H', W'] float32
    """
    t = torch.from_numpy(volume).float()
    t = t.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    t = F.interpolate(t, size=target, mode="trilinear", align_corners=False)
    return t.squeeze(0)  # [1, D', H', W']


def volume_to_vit_input(volume_3d_tensor, img_size=IMG_SIZE):
    """
    Convierte [1, 64, 64, 64] a [3, 224, 224] para backbone ViT 2D.

    Extrae 3 cortes centrales (axial, coronal, sagital) y los apila como RGB.
    Representación aproximada válida para routing — para clasificación 3D real,
    los expertos usarán el tensor volumétrico completo [1, 64, 64, 64].
    """
    v = volume_3d_tensor.squeeze(0)  # [64, 64, 64]
    d, h, w = v.shape

    axial = v[d // 2, :, :]  # corte central eje Z
    coronal = v[:, h // 2, :]  # corte central eje Y
    sagittal = v[:, :, w // 2]  # corte central eje X

    # Stack como RGB → [3, 64, 64]
    rgb = torch.stack([axial, coronal, sagittal], dim=0)

    # Resize a img_size×img_size
    rgb = F.interpolate(
        rgb.unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # [3, 224, 224]

    # Normalizar con estadísticas ImageNet (volumen ya en [0,1])
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (rgb - mean) / std


def full_3d_pipeline(
    volume_np, min_hu=-1000, max_hu=400, target=PATCH_3D_SIZE, img_size=IMG_SIZE
):
    """
    Pipeline completo: normalización HU → resize 3D → proyección + normalización.
    Conveniencia para que dataset_builder.py llame una sola función por volumen.
    """
    normed = normalize_hu(volume_np, min_hu=min_hu, max_hu=max_hu)
    resized = resize_volume_3d(normed, target=target)
    return volume_to_vit_input(resized, img_size=img_size)

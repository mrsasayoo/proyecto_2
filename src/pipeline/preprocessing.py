"""
Funciones de preprocesamiento para imágenes 2D y volúmenes 3D.

- build_2d_transform: pipeline estándar para datasets 2D (resize + normalize)
- apply_clahe: realce de contraste en resolución original (antes del resize)
- apply_circular_crop: elimina bordes negros de dermoscopio (ISIC BCN_20000)
- normalize_hu: normalización de Unidades Hounsfield para CT 3D
- resize_volume_3d: resize trilineal de volúmenes 3D a tamaño fijo
- volume_to_vit_input: proyección 3D → 2D (3 slices centrales como RGB)
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2

from config import IMAGENET_MEAN, IMAGENET_STD


def build_2d_transform(img_size: int = 224):
    """DEPRECATED — Delega a ``fase1.transform_2d.build_2d_transform``.

    Esta versión solo hacía Resize → ToTensor → Normalize y omitía TVF,
    CLAHE y Gamma Correction.  Ahora emite un DeprecationWarning y
    redirige a la implementación canónica en ``transform_2d.py``.

    Importa directamente desde ``fase1.transform_2d`` en su lugar.
    """
    warnings.warn(
        "build_2d_transform() en preprocessing.py está DEPRECATED. "
        "Importa desde fase1.transform_2d en su lugar.",
        DeprecationWarning,
        stacklevel=2,
    )
    from fase1.transform_2d import build_2d_transform as _canonical

    return _canonical(img_size=img_size)


def apply_clahe(
    img_pil: Image.Image, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)
) -> Image.Image:
    """
    H4 — CLAHE sobre imagen a resolución ORIGINAL (antes del resize).

    SIEMPRE llamar ANTES de img.resize() — nunca después.

    Flujo interno:
      img_pil (RGB, canales iguales) → canal L uint8 → CLAHE → RGB (canales iguales)
    """
    img_gray = np.array(img_pil.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(img_gray)
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))


def apply_circular_crop(img_pil):
    """
    H6/Item-8 — Elimina los bordes negros circulares de las imágenes BCN_20000.

    Las imágenes de dermoscopio de BCN tienen un campo de visión circular
    con fondo negro. Recortar evita que el modelo aprenda el artefacto.
    Operación idempotente — si no hay bordes negros, devuelve sin modificar.
    """
    img_np = np.array(img_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img_pil
    x, y, w, h = cv2.boundingRect(coords)
    img_h, img_w = img_np.shape[:2]
    if w >= 0.95 * img_w and h >= 0.95 * img_h:
        return img_pil
    return Image.fromarray(img_np[y : y + h, x : x + w])


def normalize_hu(volume, min_hu=-1000, max_hu=400):
    """Normalización HU para CT 3D (LUNA16, Pancreas)."""
    volume = np.clip(volume, min_hu, max_hu)
    volume = (volume - min_hu) / (max_hu - min_hu)
    return volume.astype(np.float32)


def resize_volume_3d(volume, target=(64, 64, 64)):
    """
    Redimensiona un volumen 3D [D, H, W] a target usando
    interpolación trilineal vía PyTorch.
    """
    t = torch.from_numpy(volume).float()
    t = t.unsqueeze(0).unsqueeze(0)
    t = nn.functional.interpolate(t, size=target, mode="trilinear", align_corners=False)
    return t.squeeze(0)  # → [1, 64, 64, 64]


def volume_to_vit_input(volume_3d_tensor):
    """
    Convierte [1, 64, 64, 64] a [3, 224, 224] para que ViT-2D
    lo procese: toma 3 slices centrales (axial, coronal, sagital)
    y los apila como canales RGB.

    Nota: esto es una aproximación válida para extracción de
    embeddings de routing. Para clasificación 3D real se usa
    la arquitectura 3D del experto (R3D-18, Swin3D-Tiny).
    """
    v = volume_3d_tensor.squeeze(0)  # [64, 64, 64]
    d, h, w = v.shape

    axial = v[d // 2, :, :]  # corte central eje Z
    coronal = v[:, h // 2, :]  # corte central eje Y
    sagittal = v[:, :, w // 2]  # corte central eje X

    # Stack como RGB → [3, 64, 64]
    rgb = torch.stack([axial, coronal, sagittal], dim=0)

    # Resize a 224×224
    rgb = nn.functional.interpolate(
        rgb.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)  # [3, 224, 224]

    # Normalizar como ImageNet (el volumen ya está en [0,1])
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (rgb - mean) / std

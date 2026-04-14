"""
transform_domain.py — Transformaciones Clínicas Especializadas
==============================================================

- CLAHE: realce de contraste adaptativo para OA Rodilla
- Crop circular: eliminación de bordes de dermoscopio para ISIC BCN_20000

Se aplican sobre la imagen en resolución original ANTES del pipeline estándar.
"""

import numpy as np
import cv2
from PIL import Image

try:
    from fase1_config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID
except ModuleNotFoundError:
    from fase1.fase1_config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID


def apply_clahe(img_pil, clip_limit=CLAHE_CLIP_LIMIT, tile_grid=CLAHE_TILE_GRID):
    """
    CLAHE sobre imagen a resolución ORIGINAL (antes del resize).

    Convierte a escala de grises → realce adaptativo → RGB (3 canales iguales).
    La conversión a gris antes del realce es crítica — si se realzaran los
    tres canales RGB por separado, se alteraría el balance de color.

    Llamar SIEMPRE antes de transform_2d — operar a alta resolución preserva
    la densidad estadística del histograma local.
    """
    img_gray = np.array(img_pil.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(img_gray)
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))


def apply_circular_crop(img_pil):
    """
    Elimina bordes negros circulares de imágenes BCN_20000.

    Umbralización en 10 (no 0) para tolerar artefactos JPEG en los bordes.
    Recorta al bounding rect del contenido visible.
    Idempotente: si el contenido ocupa >95% del frame, devuelve sin modificar
    (segura para HAM10000 y MSK que no tienen el artefacto).
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

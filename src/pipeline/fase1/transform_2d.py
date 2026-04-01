"""
transform_2d.py — Pipeline Estándar para Imágenes 2D
=====================================================

Denominador común de todos los datasets 2D (Chest, ISIC, OA).
No tiene conocimiento de ningún dominio clínico específico.

Orden estricto según §3.3 de arquitectura_moe.md:
  CLAHE → Resize → TVF → GammaCorrection → [guardar transform] → ToTensor → Normalize

Referencias:
  - PMC9340712: TVF + Gamma mejora accuracy y convergencia.
  - Regla del profesor: guardar el transform antes de normalizar para Grad-CAM.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
from torchvision import transforms

from fase1_config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    DEFAULT_GAMMA,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    TVF_N_ITER,
    TVF_WEIGHT,
)

log = logging.getLogger("fase1")


# =====================================================================
# Clase: TotalVariationFilter
# =====================================================================
class TotalVariationFilter:
    """Filtro de Variación Total (TVF) compatible con torchvision.transforms.

    Suaviza ruido pixel-a-pixel mientras preserva bordes, paso 2 de §6.2.

    Implementación:
        Usa ``skimage.restoration.denoise_tv_chambolle`` (algoritmo de
        Chambolle 2004) que es referencia estándar para TV denoising en
        imágenes médicas.  Opera sobre PIL Image (PIL → numpy float → PIL)
        para mantener compatibilidad con el pipeline de torchvision.

    Parámetros:
        weight: intensidad del suavizado.  Equivale al parámetro λ de la
                formulación TV:  argmin_u { ||u - f||² + λ·TV(u) }.
                Valores más altos → más suavizado.
                Valores típicos: 0.05–0.3 para imágenes normalizadas [0,1].
                El peso en fase1_config (TVF_WEIGHT) se escala internamente
                dividiendo por 255 para operar en rango [0,1].
        n_iter: iteraciones máximas del solver (más = mejor convergencia).
    """

    def __init__(self, weight: float = TVF_WEIGHT, n_iter: int = TVF_N_ITER):
        self.weight = weight
        self.n_iter = n_iter

    def __call__(self, img: Image.Image) -> Image.Image:
        """Aplica TVF sobre imagen PIL RGB o L."""
        img_np = np.array(img, dtype=np.float64) / 255.0

        # denoise_tv_chambolle opera sobre float [0,1] y soporta
        # imágenes 2D (grises) y 3D (RGB, con channel_axis=-1)
        if img_np.ndim == 2:
            denoised = denoise_tv_chambolle(
                img_np,
                weight=self.weight / 255.0,
                max_num_iter=self.n_iter,
            )
        elif img_np.ndim == 3:
            denoised = denoise_tv_chambolle(
                img_np,
                weight=self.weight / 255.0,
                max_num_iter=self.n_iter,
                channel_axis=-1,
            )
        else:
            log.warning(
                "[TVF] Dimensión inesperada %s, devolviendo sin filtrar", img_np.shape
            )
            return img

        denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(denoised, mode=img.mode)

    def __repr__(self) -> str:
        return f"TotalVariationFilter(weight={self.weight}, n_iter={self.n_iter})"


# =====================================================================
# Clase: GammaCorrection
# =====================================================================
class GammaCorrection:
    """Corrección gamma compatible con torchvision.transforms.

    Paso 3 de §6.2: realza brillo y estructuras, mejora accuracy y
    convergencia más rápida (PMC9340712).

    Fórmula: output = input^γ   (sobre valores normalizados [0, 1])

    Opera sobre PIL Image (uint8): normaliza a [0,1], aplica potencia,
    desnormaliza a [0,255].

    Parámetros:
        gamma: exponente de corrección.
               γ < 1 → imagen más brillante (realza sombras)
               γ = 1 → identidad
               γ > 1 → imagen más oscura (realza highlights)
               Rango clínico típico: 0.8–1.2
    """

    def __init__(self, gamma: float = DEFAULT_GAMMA):
        if gamma <= 0:
            raise ValueError(f"gamma debe ser > 0, recibido: {gamma}")
        self.gamma = gamma
        # Pre-calcular LUT de 256 entradas para aplicación O(1) por pixel
        self._lut = np.array(
            [np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)],
            dtype=np.uint8,
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        """Aplica corrección gamma sobre imagen PIL."""
        img_np = np.array(img, dtype=np.uint8)
        corrected = self._lut[img_np]
        return Image.fromarray(corrected, mode=img.mode)

    def __repr__(self) -> str:
        return f"GammaCorrection(gamma={self.gamma})"


# =====================================================================
# Clase: CLAHETransform (wrapper torchvision-compatible)
# =====================================================================
class CLAHETransform:
    """Wrapper torchvision-compatible alrededor de apply_clahe de transform_domain.

    Paso 4 de §6.2: realce de contraste adaptativo local.
    Delega la lógica real a transform_domain.apply_clahe() para no
    duplicar código.

    Parámetros:
        clip_limit: umbral de contraste de CLAHE.
        tile_grid:  tamaño de la grilla de tiles.
    """

    def __init__(
        self,
        clip_limit: float = CLAHE_CLIP_LIMIT,
        tile_grid: tuple[int, int] = CLAHE_TILE_GRID,
    ):
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    def __call__(self, img: Image.Image) -> Image.Image:
        """Aplica CLAHE. Importación diferida para evitar importaciones circulares."""
        from transform_domain import apply_clahe

        return apply_clahe(img, clip_limit=self.clip_limit, tile_grid=self.tile_grid)

    def __repr__(self) -> str:
        return (
            f"CLAHETransform(clip_limit={self.clip_limit}, tile_grid={self.tile_grid})"
        )


# =====================================================================
# Serialización del transform (paso 5 de §6.2)
# =====================================================================
@dataclass
class TransformRecord:
    """Registro serializable de la cadena de preprocesamiento aplicada.

    Almacena los parámetros de cada paso para reproducibilidad y para
    que Grad-CAM (Funcionalidad 4 del dashboard) pueda mapear
    activaciones al espacio original de la imagen.
    """

    img_size: int = IMG_SIZE
    tvf_weight: float = TVF_WEIGHT
    tvf_n_iter: int = TVF_N_ITER
    gamma: float = DEFAULT_GAMMA
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT
    clahe_tile_grid: tuple[int, int] = CLAHE_TILE_GRID
    imagenet_mean: list[float] = field(default_factory=lambda: list(IMAGENET_MEAN))
    imagenet_std: list[float] = field(default_factory=lambda: list(IMAGENET_STD))
    pipeline_order: list[str] = field(
        default_factory=lambda: [
            "CLAHE",
            "Resize",
            "TotalVariationFilter",
            "GammaCorrection",
            "ToTensor",
            "Normalize",
        ]
    )


def save_transform(transform_record: TransformRecord, path: str | Path) -> Path:
    """Guarda el registro del transform en disco (pickle + JSON).

    Genera dos archivos:
      - ``<path>.pkl``:  objeto TransformRecord serializado con pickle
                         (reconstrucción exacta para código Python).
      - ``<path>.json``: representación legible para inspección manual
                         y portabilidad entre lenguajes.

    Args:
        transform_record: registro con los parámetros de la cadena.
        path: ruta base sin extensión (se añaden .pkl y .json).

    Returns:
        Ruta del archivo .pkl generado.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Pickle: reconstrucción exacta
    pkl_path = path.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(transform_record, f, protocol=pickle.HIGHEST_PROTOCOL)

    # JSON: inspección humana y portabilidad
    json_path = path.with_suffix(".json")
    record_dict = asdict(transform_record)
    # Convertir tuplas a listas para JSON
    record_dict["clahe_tile_grid"] = list(record_dict["clahe_tile_grid"])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record_dict, f, indent=2, ensure_ascii=False)

    log.info("[Transform] Guardado en %s (.pkl + .json)", path)
    return pkl_path


def load_transform(path: str | Path) -> TransformRecord:
    """Carga un TransformRecord desde disco (pickle).

    Args:
        path: ruta al archivo .pkl (o base sin extensión).

    Returns:
        TransformRecord reconstruido.
    """
    path = Path(path)
    if path.suffix != ".pkl":
        path = path.with_suffix(".pkl")
    with open(path, "rb") as f:
        record = pickle.load(f)
    if not isinstance(record, TransformRecord):
        raise TypeError(
            f"Se esperaba TransformRecord, se obtuvo {type(record).__name__}"
        )
    return record


# =====================================================================
# Constructor principal del pipeline 2D
# =====================================================================
def build_2d_transform(
    img_size: int = IMG_SIZE,
    gamma: float = DEFAULT_GAMMA,
    tvf_weight: float = TVF_WEIGHT,
    tvf_n_iter: int = TVF_N_ITER,
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
    clahe_tile_grid: tuple[int, int] = CLAHE_TILE_GRID,
    save_path: str | Path | None = None,
) -> transforms.Compose:
    """Transform estándar para todos los datasets 2D (§6.2).

    Cadena completa:
        CLAHE → Resize → TotalVariationFilter → GammaCorrection → ToTensor → Normalize

    CLAHE se aplica primero, a resolución original, antes del resize
    (§3.3: "CLAHE siempre antes del resize").
    Los pasos restantes operan sobre PIL Image (uint8) en resolución
    ``img_size × img_size``.  ToTensor convierte HWC uint8 → CHW float [0,1].
    Normalize aplica la estandarización ImageNet final.

    Args:
        img_size:         tamaño cuadrado de salida (default: 224 de fase1_config).
        gamma:            exponente de corrección gamma (default: 1.0).
        tvf_weight:       intensidad del filtro de variación total.
        tvf_n_iter:       iteraciones del solver TVF.
        clahe_clip_limit: umbral de contraste para CLAHE.
        clahe_tile_grid:  tamaño de grilla CLAHE.
        save_path:        si se proporciona, guarda TransformRecord en esta ruta.

    Returns:
        transforms.Compose con la cadena completa.

    Nota sobre compatibilidad:
        La firma ``build_2d_transform(img_size=IMG_SIZE)`` sigue funcionando
        exactamente igual que antes — todos los parámetros nuevos tienen
        defaults que preservan el comportamiento original cuando no se
        especifican explícitamente.
    """
    pipeline = transforms.Compose(
        [
            # Paso 1: CLAHE — realce de contraste adaptativo local
            # (a resolución original, ANTES del resize — §3.3)
            CLAHETransform(clip_limit=clahe_clip_limit, tile_grid=clahe_tile_grid),
            # Paso 2: Resize estandarizado (PIL, sin interpolación agresiva)
            transforms.Resize((img_size, img_size)),
            # Paso 3: Total Variation Filter — denoising preservando bordes
            TotalVariationFilter(weight=tvf_weight, n_iter=tvf_n_iter),
            # Paso 4: Corrección gamma — realza brillo y estructuras
            GammaCorrection(gamma=gamma),
            # --- Paso 5: guardar transform (se hace fuera del Compose) ---
            # Paso 6: Generar tensor normalizado
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Paso 5: guardar registro del transform si se solicitó
    if save_path is not None:
        record = TransformRecord(
            img_size=img_size,
            tvf_weight=tvf_weight,
            tvf_n_iter=tvf_n_iter,
            gamma=gamma,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid=clahe_tile_grid,
        )
        save_transform(record, save_path)

    log.info(
        "[Transform 2D] Cadena construida: CLAHE(clip=%.1f) → Resize(%d) → "
        "TVF(w=%.1f) → Gamma(γ=%.2f) → ToTensor → Normalize",
        clahe_clip_limit,
        img_size,
        tvf_weight,
        gamma,
    )

    return pipeline


# =====================================================================
# Utilidad de diagnóstico (sin cambios respecto al original)
# =====================================================================
def inspect_tensor_stats(tensor: torch.Tensor) -> dict[str, dict[str, float]]:
    """Verifica que un tensor transformado tiene distribución razonable
    post-normalización ImageNet. Útil para diagnóstico cuando los
    embeddings tienen normas anómalamente bajas.

    Returns:
        dict con media y std por canal
    """
    if tensor.dim() != 3 or tensor.shape[0] != 3:
        log.warning("[inspect] Tensor no es [3, H, W]: %s", tensor.shape)
        return {}

    stats: dict[str, dict[str, float]] = {}
    for c in range(3):
        ch = tensor[c]
        stats[f"ch{c}"] = {"mean": ch.mean().item(), "std": ch.std().item()}
    return stats

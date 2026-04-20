"""
AdaptivePreprocessor del sistema MoE.

Recibe un tensor 2D (rank 4) o 3D (rank 5) SIN metadatos, detecta la modalidad
por rank del tensor, y produce la vista 224x224x3 que el router ViT-Tiny consume.

Ver diseno en agent-docs/moe-backbone/02-adaptive-preprocessor-design.md.

Importante: segun agent-docs/moe-backbone/01-pipelines-inferencia-consolidados.md
seccion 0, la implementacion REAL usada para los embeddings en `embeddings/` es
un pipeline generico (Resize 224 + ToTensor + Normalize ImageNet). Esta clase
ofrece dos modos:
  - "generic" (default): replica exactamente el pipeline usado para los embeddings.
  - "domain_aware": aplica hook determinista por dominio (CLAHE, Shades of Gray, etc.)
    Este modo esta pensado para el dashboard / re-extraccion coherente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PreprocessOutput:
    tensor: torch.Tensor            # (B_eff, 3, 224, 224) float32 normalizado
    modality: Literal["2d", "3d"]
    n_slices: int | None            # None para 2D; 16 para 3D
    expected_domain: int | None     # siempre None (nunca se pasa a la red)


class AdaptivePreprocessor:
    def __init__(
        self,
        router_resolution: int = 224,
        vit_mean: tuple[float, float, float] = IMAGENET_MEAN,
        vit_std: tuple[float, float, float] = IMAGENET_STD,
        n_slices_3d: int = 16,
        mode: Literal["generic", "domain_aware"] = "generic",
        domain_hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.router_resolution = router_resolution
        self.n_slices_3d = n_slices_3d
        self.mean = torch.tensor(vit_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(vit_std).view(1, 3, 1, 1)
        self.mode = mode
        self.domain_hook = domain_hook
        if mode == "domain_aware" and domain_hook is None:
            raise ValueError("mode='domain_aware' requiere domain_hook.")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def _ensure_3_channels(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        if x.size(1) == 3:
            return x
        raise ValueError(f"Canales no soportados: {x.size(1)}. Esperado 1 o 3.")

    def _scale_to_01(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.uint8, torch.int32, torch.int64):
            return x.float() / 255.0
        if x.float().max() > 1.5:
            return x.float() / 255.0
        return x.float()

    def _process_2d(self, x: torch.Tensor) -> PreprocessOutput:
        B, C, H, W = x.shape
        if H < 32 or W < 32:
            raise ValueError(f"Resolucion 2D insuficiente: {H}x{W}. Minimo 32.")
        x = self._scale_to_01(x)
        if self.mode == "domain_aware":
            x = self.domain_hook(x)  # type: ignore
        x = self._ensure_3_channels(x)
        R = self.router_resolution
        mode = "area" if H > R or W > R else "bilinear"
        x = F.interpolate(x, size=(R, R), mode=mode, align_corners=False if mode == "bilinear" else None)
        x = self._normalize(x)
        return PreprocessOutput(tensor=x, modality="2d", n_slices=None, expected_domain=None)

    def _process_3d(self, x: torch.Tensor) -> PreprocessOutput:
        B, C, D, H, W = x.shape
        if D < self.n_slices_3d:
            raise ValueError(
                f"Volumen 3D con D={D} < n_slices_3d={self.n_slices_3d}. "
                "Considera interpolar antes o reducir n_slices_3d."
            )
        idx = torch.linspace(0, D - 1, self.n_slices_3d).round().long()
        slices = x[:, :, idx, :, :]                                  # (B, C, N, H, W)
        slices = slices.permute(0, 2, 1, 3, 4).contiguous()          # (B, N, C, H, W)
        slices = slices.view(B * self.n_slices_3d, C, H, W)          # (B*N, C, H, W)
        slices = self._scale_to_01(slices)
        if self.mode == "domain_aware":
            slices = self.domain_hook(slices)  # type: ignore
        slices = self._ensure_3_channels(slices)
        R = self.router_resolution
        slices = F.interpolate(slices, size=(R, R), mode="bilinear", align_corners=False)
        slices = self._normalize(slices)
        return PreprocessOutput(
            tensor=slices, modality="3d", n_slices=self.n_slices_3d, expected_domain=None
        )

    def __call__(self, x: torch.Tensor) -> PreprocessOutput:
        rank = x.dim()
        if rank == 4:
            return self._process_2d(x)
        if rank == 5:
            return self._process_3d(x)
        raise ValueError(
            f"Rank {rank} no soportado. Esperado 4 (2D BCHW) o 5 (3D BCDHW)."
        )

"""
Expert 2 — ConvNeXt-Small para clasificación multiclase de lesiones dermatoscópicas.

Arquitectura:
    Backbone: timm convnext_small (Liu et al., 2022) con pesos ImageNet
    Entrada:  [B, 3, 224, 224] — imagen dermoscópica RGB
    Salida:   [B, 8] — logits para 8 clases de entrenamiento

Clases (ISIC 2019):
    0=MEL, 1=NV, 2=BCC, 3=AK, 4=BKL, 5=DF, 6=VASC, 7=SCC

Migración:
    Reemplaza Expert2EfficientNetB3 (EfficientNet-B3, ~10.7M params, sin pretrained)
    por ConvNeXt-Small (~50M params, pretrained ImageNet) con head personalizado
    y soporte para fine-tuning diferencial (freeze/unfreeze por stages).

Head personalizado (reemplaza head original de timm):
    LayerNorm(768) → Dropout(0.4) → Linear(768→256) → GELU()
    → Dropout(0.3) → Linear(256→8)

Parámetros: ~50M totales (backbone ConvNeXt-Small + head personalizado).

Autor: Pipeline Expert2 — Fase 2 (migración ConvNeXt)
"""

from __future__ import annotations

from typing import Iterator

import timm
import torch
import torch.nn as nn
from timm.layers import trunc_normal_


class Expert2ConvNeXtSmall(nn.Module):
    """
    ConvNeXt-Small adaptado para ISIC 2019 — 8 clases multiclase.

    Backbone timm convnext_small con pesos ImageNet pretrained.
    El head original se reemplaza por un clasificador personalizado
    con dos capas lineales, GELU, dropout y LayerNorm.

    Entrada:  [B, 3, 224, 224] — imagen dermoscópica RGB normalizada
    Salida:   [B, 8] — logits crudos (antes de softmax)

    Args:
        num_classes: número de clases de salida. Default: 8.
        pretrained: cargar pesos ImageNet en el backbone. Default: True.
    """

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ── Backbone ConvNeXt-Small (timm) ──────────────────────────────
        # num_classes=0 + global_pool='avg' → forward retorna [B, 768]
        # El head original de timm (pool + norm + flatten + fc) queda con
        # fc=Identity(), así self.model(x) produce embeddings [B, 768].
        backbone = timm.create_model(
            "convnext_small",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.model = backbone

        # ── Head personalizado ──────────────────────────────────────────
        # Opera sobre embeddings [B, 768] producidos por self.model(x).
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(p=0.4),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        # ── Inicialización de la capa final ─────────────────────────────
        final_linear: nn.Linear = self.head[5]  # type: ignore[assignment]
        trunc_normal_(final_linear.weight, std=0.02)
        nn.init.zeros_(final_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo ConvNeXt-Small.

        Args:
            x: tensor [B, 3, 224, 224] — imagen RGB normalizada

        Returns:
            logits: tensor [B, 8] — logits crudos (antes de softmax)
        """
        features = self.model(x)  # [B, 768]
        logits = self.head(features)  # [B, 8]
        return logits

    # ── Métodos para optimizer diferencial ──────────────────────────────

    def get_head_params(self) -> Iterator[nn.Parameter]:
        """Retorna parámetros del head (para lr más alto)."""
        return self.head.parameters()

    def get_backbone_params(self) -> Iterator[nn.Parameter]:
        """Retorna parámetros del backbone sin incluir el head."""
        return self.model.parameters()

    # ── Métodos de congelamiento / descongelamiento ─────────────────────

    def freeze_backbone(self) -> None:
        """Congela todos los parámetros del backbone (self.model)."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last_stages(self, n: int = 2) -> None:
        """
        Descongela los últimos ``n`` stages del backbone para fine-tuning.

        ConvNeXt-Small de timm tiene 4 stages (índices 0-3) en
        ``self.model.stages``. Con ``n=2`` se descongelan stages 2 y 3,
        que contienen la mayor parte de la capacidad representacional.

        Args:
            n: cantidad de stages finales a descongelar. Default: 2.
        """
        stages = self.model.stages
        total_stages = len(stages)
        start = max(0, total_stages - n)
        for i in range(start, total_stages):
            for param in stages[i].parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Descongela todos los parámetros del modelo (backbone + head)."""
        for param in self.parameters():
            param.requires_grad = True

    # ── Utilidades ──────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Retorna el número total de parámetros entrenables del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Retorna el número total de parámetros (entrenables + congelados)."""
        return sum(p.numel() for p in self.parameters())


def _test_model() -> None:
    """Verificación rápida: instanciar, forward pass, conteo de parámetros."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Expert2/ConvNeXt-Small] Dispositivo: {device}")

    model = Expert2ConvNeXtSmall(num_classes=8, pretrained=True).to(device)

    # Forward pass con tensor dummy
    dummy = torch.randn(2, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)

    n_params = model.count_parameters()
    print(f"[Expert2/ConvNeXt-Small] Input shape:  {list(dummy.shape)}")
    print(f"[Expert2/ConvNeXt-Small] Output shape: {list(out.shape)}")
    print(f"[Expert2/ConvNeXt-Small] Parámetros entrenables: {n_params:,}")
    print(f"[Expert2/ConvNeXt-Small] Output values: {out}")

    # Validaciones
    assert out.shape == (2, 8), (
        f"Shape de salida incorrecto: {out.shape}, esperado (2, 8)"
    )
    assert n_params > 0, "Modelo sin parámetros entrenables"

    # Verificar freeze/unfreeze
    model.freeze_backbone()
    frozen_params = model.count_parameters()
    print(f"[Expert2/ConvNeXt-Small] Params tras freeze_backbone: {frozen_params:,}")

    model.unfreeze_last_stages(n=2)
    partial_params = model.count_parameters()
    print(
        f"[Expert2/ConvNeXt-Small] Params tras unfreeze_last_stages(2): {partial_params:,}"
    )

    model.unfreeze_all()
    all_params = model.count_parameters()
    print(f"[Expert2/ConvNeXt-Small] Params tras unfreeze_all: {all_params:,}")

    assert frozen_params < partial_params < all_params, (
        "Freeze/unfreeze no funciona correctamente"
    )

    print("[Expert2/ConvNeXt-Small] Verificación completada exitosamente")
    return model


# ── Alias de compatibilidad ────────────────────────────────────────────
# Permite que imports existentes como `from ... import Expert2EfficientNetB3`
# sigan funcionando tras la migración a ConvNeXt-Small.
Expert2EfficientNetB3 = Expert2ConvNeXtSmall


if __name__ == "__main__":
    _test_model()

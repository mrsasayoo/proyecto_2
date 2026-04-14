"""
Expert 1 — ConvNeXt-Tiny (timm, pretrained) para clasificación multilabel de ChestXray14.

Arquitectura:
    Backbone: timm convnext_tiny.in12k_ft_in1k (Liu et al., 2022)
              Pre-entrenado en ImageNet-12K, fine-tuned en ImageNet-1K.
    Adapter:  domain_conv — bloque convolucional residual post-backbone
              que adapta las features al dominio médico sin destruir
              las representaciones pretrained.
    Head:     AdaptiveAvgPool2d(1) → Flatten → Dropout → Linear(768, 14)
    Entrada:  [B, 3, 224, 224] — radiografía de tórax RGB, float32
    Salida:   [B, 14] — logits para 14 patologías (multilabel, BCEWithLogitsLoss)

Estrategia de entrenamiento: LP-FT (Linear Probing → Fine-Tuning).
    - LP: freeze_backbone() — solo head + domain_conv entrenables.
    - FT: unfreeze_backbone() — todo el modelo con LR reducido.

MODEL_MEAN y MODEL_STD se resuelven programáticamente desde timm para
garantizar consistencia con el backbone pretrained.

Autor: Pipeline Expert1 — Fase 2
"""

from __future__ import annotations

import timm
import timm.data
import torch
import torch.nn as nn

from pipeline.fase2.expert1_config import (
    EXPERT1_BACKBONE,
    EXPERT1_DROPOUT_FC,
    EXPERT1_NUM_CLASSES,
)


class Expert1ConvNeXt(nn.Module):
    """ConvNeXt-Tiny pretrained para clasificación multilabel de 14 patologías.

    Usa ``timm.create_model`` con ``num_classes=0`` para obtener features puras
    del backbone. Un adapter residual (``domain_conv``) adapta las features al
    dominio de rayos X antes de pasar por pooling global y la cabeza clasificadora.

    Attributes:
        model_mean: tupla RGB de medias de normalización del backbone.
        model_std: tupla RGB de desviaciones de normalización del backbone.

    Args:
        backbone: nombre del modelo timm. Default: config.EXPERT1_BACKBONE.
        num_classes: clases de salida. Default: config.EXPERT1_NUM_CLASSES (14).
        dropout_fc: dropout antes de la FC final. Default: config.EXPERT1_DROPOUT_FC.
        pretrained: cargar pesos pretrained. Default: True.
    """

    def __init__(
        self,
        backbone: str = EXPERT1_BACKBONE,
        num_classes: int = EXPERT1_NUM_CLASSES,
        dropout_fc: float = EXPERT1_DROPOUT_FC,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ── Backbone pretrained (sin cabeza clasificadora) ──────────────
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
        )
        feat_dim: int = self.backbone.num_features

        # ── Resolver estadísticas de normalización del backbone ─────────
        _cfg = timm.data.resolve_data_config({}, model=self.backbone)
        self.model_mean: tuple[float, ...] = _cfg["mean"]
        self.model_std: tuple[float, ...] = _cfg["std"]

        # ── domain_conv — adapter residual post-backbone ────────────────
        self.domain_conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, 1, bias=False),
            nn.BatchNorm2d(feat_dim),
        )

        # ── Pooling global + cabeza clasificadora ───────────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_fc),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: tensor ``[B, 3, 224, 224]`` — imagen RGB normalizada.

        Returns:
            Logits ``[B, num_classes]`` (antes de sigmoid).
        """
        feat = self.backbone.forward_features(x)  # [B, C, H, W]
        feat = feat + self.domain_conv(feat)  # residual
        feat = self.pool(feat)  # [B, C, 1, 1]
        return self.head(feat)  # [B, num_classes]

    def freeze_backbone(self) -> None:
        """Congela todos los parámetros del backbone (fase LP)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Descongela todos los parámetros del backbone (fase FT)."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def count_parameters(self) -> int:
        """Número de parámetros entrenables (requires_grad=True)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Número total de parámetros (entrenables + congelados)."""
        return sum(p.numel() for p in self.parameters())


# ── Alias de compatibilidad con imports existentes ──────────────────────
Expert1ConvNeXtTiny = Expert1ConvNeXt


def _test_model() -> None:
    """Verificación rápida: instanciar, forward pass, freeze/unfreeze, conteo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Expert1/ConvNeXt] Dispositivo: {device}")

    model = Expert1ConvNeXt().to(device)

    print(f"[Expert1/ConvNeXt] mean={model.model_mean}  std={model.model_std}")

    x = torch.randn(2, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        out = model(x)

    total = model.count_all_parameters()
    trainable = model.count_parameters()
    print(f"[Expert1/ConvNeXt] Input shape:  {list(x.shape)}")
    print(f"[Expert1/ConvNeXt] Output shape: {list(out.shape)}")
    print(f"[Expert1/ConvNeXt] Params total: {total:,}  trainable: {trainable:,}")

    model.freeze_backbone()
    frozen_trainable = model.count_parameters()
    print(f"[Expert1/ConvNeXt] Trainable after freeze: {frozen_trainable:,}")

    model.unfreeze_backbone()
    unfrozen_trainable = model.count_parameters()
    print(f"[Expert1/ConvNeXt] Trainable after unfreeze: {unfrozen_trainable:,}")

    assert out.shape == (2, 14), f"Shape incorrecto: {out.shape}"
    assert frozen_trainable < trainable, "freeze_backbone no redujo trainables"
    assert unfrozen_trainable == trainable, "unfreeze_backbone no restauró trainables"
    print("[Expert1/ConvNeXt] Verificación completada exitosamente")


if __name__ == "__main__":
    _test_model()

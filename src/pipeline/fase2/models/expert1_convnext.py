"""
Expert 1 — ConvNeXt-Tiny para clasificación multilabel de ChestXray14.

Arquitectura:
    Backbone: torchvision.models.convnext_tiny (Liu et al., 2022)
    Entrada:  [B, 3, 224, 224] — radiografía de tórax RGB, float32
    Salida:   [B, 14] — logits para 14 patologías (multilabel, BCEWithLogitsLoss)

Adaptaciones respecto al ConvNeXt-Tiny original (ImageNet, 1000 clases):
    1. Sin pesos preentrenados (weights=None) — entrenamiento from scratch.
    2. Cabeza clasificadora: 1000 → 14 clases.
    3. Dropout(p=0.3) antes de la capa de clasificación final.

Conteo de parámetros:
    ConvNeXt-Tiny tiene ~28.6M parámetros totales.
    Con ~86K muestras de entrenamiento, el ratio params/datos es ~332:1,
    manejable con regularización estándar (dropout, weight_decay, augmentation).

Autor: Pipeline Expert1 — Fase 2
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny


class Expert1ConvNeXtTiny(nn.Module):
    """
    ConvNeXt-Tiny para clasificación multilabel de 14 patologías torácicas.

    Usa torchvision.models.convnext_tiny sin pesos preentrenados.
    La cabeza clasificadora original (1000 clases ImageNet) se reemplaza
    por una capa con Dropout + Linear(768, 14).

    Entrada:  [B, 3, 224, 224] — imagen RGB normalizada (ImageNet stats)
    Salida:   [B, 14] — logits crudos (antes de sigmoid)

    Args:
        fc_dropout_p: probabilidad de Dropout antes de la FC final.
            Default: 0.3 (de expert1_config.EXPERT1_DROPOUT_FC).
        num_classes: número de clases de salida. Default: 14.
    """

    def __init__(
        self,
        fc_dropout_p: float = 0.3,
        num_classes: int = 14,
    ):
        super().__init__()

        # ── Cargar backbone ConvNeXt-Tiny sin pesos preentrenados ───────
        backbone = convnext_tiny(weights=None)

        # ── Adaptar cabeza clasificadora: 1000 → 14 clases ─────────────
        # Estructura original del classifier en torchvision ConvNeXt:
        #   backbone.classifier = Sequential(
        #       [0] LayerNorm2d(768),
        #       [1] Flatten(start_dim=1),
        #       [2] Linear(768, 1000)
        #   )
        # Reemplazamos solo el Linear final con Dropout + Linear(768, num_classes)
        in_features = backbone.classifier[2].in_features  # 768
        backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=fc_dropout_p),
            nn.Linear(in_features, num_classes),
        )

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo ConvNeXt-Tiny.

        Args:
            x: tensor [B, 3, 224, 224] — imagen RGB normalizada

        Returns:
            logits: tensor [B, 14] — logits crudos (antes de sigmoid)
        """
        return self.model(x)

    def count_parameters(self) -> int:
        """Retorna el número total de parámetros entrenables del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Retorna el número total de parámetros (entrenables + congelados)."""
        return sum(p.numel() for p in self.parameters())


def _test_model():
    """Verificación rápida: instanciar, forward pass, conteo de parámetros."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Expert1/ConvNeXt-Tiny] Dispositivo: {device}")

    model = Expert1ConvNeXtTiny(
        fc_dropout_p=0.3,
        num_classes=14,
    ).to(device)

    # Forward pass con tensor dummy
    x = torch.randn(2, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        out = model(x)

    n_params = model.count_parameters()
    print(f"[Expert1/ConvNeXt-Tiny] Input shape:  {list(x.shape)}")
    print(f"[Expert1/ConvNeXt-Tiny] Output shape: {list(out.shape)}")
    print(f"[Expert1/ConvNeXt-Tiny] Params: {n_params:,}")
    print(f"[Expert1/ConvNeXt-Tiny] Output values: {out}")

    # Validaciones
    assert out.shape == (2, 14), (
        f"Shape de salida incorrecto: {out.shape}, esperado (2, 14)"
    )
    assert n_params > 0, "Modelo sin parámetros entrenables"
    print(f"[Expert1/ConvNeXt-Tiny] Verificación completada exitosamente")
    return model


if __name__ == "__main__":
    _test_model()

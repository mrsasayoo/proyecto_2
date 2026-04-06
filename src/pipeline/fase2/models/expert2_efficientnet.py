"""
Expert 2 — EfficientNet-B3 para clasificación multiclase de lesiones dermatoscópicas.

Arquitectura:
    Backbone: torchvision.models.efficientnet_b3 (Compound Scaling, Tan & Le 2019)
    Entrada:  [B, 3, 224, 224] — imagen dermoscópica RGB
    Salida:   [B, 9] — logits para 8 clases de entrenamiento + 1 slot UNK

Clases (ISIC 2019):
    0=MEL, 1=NV, 2=BCC, 3=AK, 4=BKL, 5=DF, 6=VASC, 7=SCC, 8=UNK

Adaptaciones respecto al EfficientNet-B3 original (ImageNet, 1000 clases):
    1. Sin pesos preentrenados (weights=None) — entrenamiento desde cero.
    2. Cabeza clasificadora: 1000 → 9 clases (8 train + 1 UNK).
    3. Dropout configurable en la capa FC final (default: 0.3).

El slot UNK (índice 8) no recibe supervisión durante el entrenamiento
(CrossEntropyLoss con labels 0-7). Su peso en class_weights es 1.0.
En inferencia, el softmax sobre 9 neuronas permite al Experto 5 (OOD)
detectar distribuciones anómalas vía entropía o probabilidad UNK.

Parámetros: ~10.7M entrenables (EfficientNet-B3 con clasificador adaptado).
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3


class Expert2EfficientNetB3(nn.Module):
    """
    EfficientNet-B3 adaptado para ISIC 2019 — 9 clases multiclase.

    Entrada:  [B, 3, 224, 224] — imagen dermoscópica RGB normalizada
    Salida:   [B, 9] — logits crudos (antes de softmax)

    Args:
        fc_dropout_p: probabilidad de Dropout en la capa FC final.
            Default: 0.3 (de expert2_config.EXPERT2_DROPOUT_FC).
        num_classes: número de clases de salida. Default: 9
            (8 clases de entrenamiento + 1 slot UNK).
    """

    def __init__(
        self,
        fc_dropout_p: float = 0.3,
        num_classes: int = 9,
    ):
        super().__init__()

        # ── Cargar backbone EfficientNet-B3 sin pesos preentrenados ─────
        backbone = efficientnet_b3(weights=None)

        # ── Reemplazar cabeza clasificadora ─────────────────────────────
        # EfficientNet-B3 original en torchvision:
        #   backbone.classifier = Sequential(Dropout(p=0.3), Linear(1536, 1000))
        # Reemplazamos con nuestro dropout y número de clases:
        in_features = backbone.classifier[1].in_features  # 1536
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=fc_dropout_p),
            nn.Linear(in_features, num_classes),
        )

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo EfficientNet-B3.

        Args:
            x: tensor [B, 3, 224, 224] — imagen RGB normalizada

        Returns:
            logits: tensor [B, 9] — logits crudos (antes de softmax)
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
    print(f"[Expert2/EfficientNet-B3] Dispositivo: {device}")

    model = Expert2EfficientNetB3(
        fc_dropout_p=0.3,
        num_classes=9,
    ).to(device)

    # Forward pass con tensor dummy
    dummy = torch.randn(2, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)

    n_params = model.count_parameters()
    print(f"[Expert2/EfficientNet-B3] Input shape:  {list(dummy.shape)}")
    print(f"[Expert2/EfficientNet-B3] Output shape: {list(out.shape)}")
    print(f"[Expert2/EfficientNet-B3] Parámetros entrenables: {n_params:,}")
    print(f"[Expert2/EfficientNet-B3] Output values: {out}")

    # Validaciones
    assert out.shape == (2, 9), (
        f"Shape de salida incorrecto: {out.shape}, esperado (2, 9)"
    )
    assert n_params > 0, "Modelo sin parámetros entrenables"
    print(f"[Expert2/EfficientNet-B3] ✓ Verificación completada exitosamente")
    return model


if __name__ == "__main__":
    _test_model()

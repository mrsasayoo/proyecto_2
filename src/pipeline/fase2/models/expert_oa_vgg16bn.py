"""
Expert OA — VGG16-BN para clasificación ordinal de osteoartritis de rodilla.

Arquitectura:
    Backbone: torchvision.models.vgg16_bn (Simonyan & Zisserman, 2015 + Batch Norm)
    Entrada:  [B, 3, 224, 224] — radiografía de rodilla RGB (CLAHE aplicado)
    Salida:   [B, 3] — logits para 3 clases ordinales

Clases (Kellgren-Lawrence consolidado):
    0 = Normal  (KL 0)
    1 = Leve    (KL 1-2)
    2 = Severo  (KL 3-4)

Adaptaciones respecto al VGG16-BN original (ImageNet, 1000 clases):
    1. Sin pesos preentrenados (weights=None) — entrenamiento desde cero.
    2. Clasificador modificado: 1000 → 3 clases con BN + Dropout agresivo (0.5).
    3. BN en las capas FC para estabilizar entrenamiento en dataset pequeño.

Parámetros: ~131M entrenables (VGG16-BN con clasificador adaptado).
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16_bn


class ExpertOAVGG16BN(nn.Module):
    """
    VGG16-BN adaptado para Osteoarthritis Knee — 3 clases ordinales.

    Entrada:  [B, 3, 224, 224] — radiografía de rodilla con CLAHE, normalizada
    Salida:   [B, 3] — logits crudos (antes de softmax)

    Args:
        num_classes: número de clases de salida. Default: 3
            (Normal, Leve, Severo).
        dropout: probabilidad de Dropout en las capas FC del clasificador.
            Default: 0.5 (agresivo por ratio alto parámetros/muestras).
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        # ── Cargar backbone VGG16-BN sin pesos preentrenados ───────
        base = vgg16_bn(weights=None)

        # ── Reusar features y avgpool del backbone ─────────────────
        self.features = base.features
        self.avgpool = base.avgpool

        # ── Clasificador modificado: BN + Dropout agresivo ─────────
        # VGG16-BN original: FC(25088→4096) → ReLU → Drop → FC(4096→4096) → ReLU → Drop → FC(4096→1000)
        # Modificación: se agrega BN antes de ReLU en cada FC para
        # estabilizar el entrenamiento en un dataset pequeño (~4.7K imgs).
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo VGG16-BN.

        Args:
            x: tensor [B, 3, 224, 224] — imagen RGB normalizada

        Returns:
            logits: tensor [B, 3] — logits crudos (antes de softmax)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        """Retorna el número total de parámetros entrenables del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Retorna el número total de parámetros (entrenables + congelados)."""
        return sum(p.numel() for p in self.parameters())


def _test_model():
    """Verificación rápida: instanciar, forward pass, conteo de parámetros."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ExpertOA/VGG16-BN] Dispositivo: {device}")

    model = ExpertOAVGG16BN(
        num_classes=3,
        dropout=0.5,
    ).to(device)

    # Forward pass con tensor dummy
    dummy = torch.randn(2, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)

    n_params = model.count_parameters()
    print(f"[ExpertOA/VGG16-BN] Input shape:  {list(dummy.shape)}")
    print(f"[ExpertOA/VGG16-BN] Output shape: {list(out.shape)}")
    print(f"[ExpertOA/VGG16-BN] Parámetros entrenables: {n_params:,}")
    print(f"[ExpertOA/VGG16-BN] Output values: {out}")

    # Validaciones
    assert out.shape == (2, 3), (
        f"Shape de salida incorrecto: {out.shape}, esperado (2, 3)"
    )
    assert n_params > 0, "Modelo sin parámetros entrenables"
    print(f"[ExpertOA/VGG16-BN] ✓ Verificación completada exitosamente")
    return model


if __name__ == "__main__":
    _test_model()

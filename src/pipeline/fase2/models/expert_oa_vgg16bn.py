"""
Expert OA — EfficientNet-B0 para clasificación de osteoartritis de rodilla (KL 0-4).

Arquitectura:
    Backbone: torchvision.models.efficientnet_b0 (preentrenado ImageNet1K)
    16 bloques MBConv con Squeeze-and-Excitation → [B, 1280, 7, 7]
    AdaptiveAvgPool2d(1) → [B, 1280]
    Classifier (HEAD reemplazada):
        Dropout(p=0.4)
        Linear(1280 → 5)
    Salida: [B, 5] — logits para 5 grados Kellgren-Lawrence

Clases (Kellgren-Lawrence):
    0 = KL 0 — Normal
    1 = KL 1 — Dudoso
    2 = KL 2 — Leve
    3 = KL 3 — Moderado
    4 = KL 4 — Severo

Adaptaciones respecto al EfficientNet-B0 original (ImageNet, 1000 clases):
    1. Pesos preentrenados ImageNet1K (transfer learning).
    2. Clasificador reemplazado: 1000 → 5 clases con Dropout(0.4).

Parámetros: ~5.3M totales (EfficientNet-B0 con head adaptada).

Nota: el archivo conserva su nombre original (expert_oa_vgg16bn.py) para no
romper imports existentes en el pipeline. La clase interna ahora es
ExpertOAEfficientNetB0. Se exporta un alias ExpertOAVGG16BN para
compatibilidad hacia atrás.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ExpertOAEfficientNetB0(nn.Module):
    """EfficientNet-B0 preentrenado adaptado para Osteoarthritis Knee — 5 grados KL.

    Reemplaza el head original de 1000 clases por un clasificador ligero
    (Dropout + Linear) ajustado para los 5 grados de Kellgren-Lawrence.

    Entrada:  [B, 3, 224, 224] — radiografía de rodilla con CLAHE, normalizada
    Salida:   [B, 5] — logits crudos (antes de softmax)

    Args:
        num_classes: número de clases de salida. Default: 5
            (KL 0, KL 1, KL 2, KL 3, KL 4).
        dropout: probabilidad de Dropout en el clasificador.
            Default: 0.4.
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        # ── Cargar backbone EfficientNet-B0 con pesos ImageNet1K ───
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # ── Reemplazar classifier (head) ───────────────────────────
        # Original: Sequential(Dropout(0.2), Linear(1280, 1000))
        # Nueva:    Sequential(Dropout(0.4), Linear(1280, 5))
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del modelo EfficientNet-B0.

        Args:
            x: tensor [B, 3, 224, 224] — imagen RGB normalizada.

        Returns:
            logits: tensor [B, 5] — logits crudos (antes de softmax).
        """
        return self.model(x)

    def get_backbone_params(self) -> list[nn.Parameter]:
        """Retorna los parámetros del backbone (features) para fine-tuning diferencial."""
        return list(self.model.features.parameters())

    def get_head_params(self) -> list[nn.Parameter]:
        """Retorna los parámetros del clasificador (head) para fine-tuning diferencial."""
        return list(self.model.classifier.parameters())

    def count_parameters(self) -> int:
        """Retorna el número total de parámetros entrenables del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Retorna el número total de parámetros (entrenables + congelados)."""
        return sum(p.numel() for p in self.parameters())


# ── Alias de compatibilidad hacia atrás ────────────────────────────
# Otros módulos del pipeline importan `ExpertOAVGG16BN` desde este archivo.
# Este alias evita romper esos imports mientras se migra gradualmente.
ExpertOAVGG16BN = ExpertOAEfficientNetB0


def _test_model() -> None:
    """Verificación rápida: instanciar, forward pass, conteo de parámetros."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ExpertOA/EfficientNet-B0] Dispositivo: {device}")

    model = ExpertOAEfficientNetB0(
        num_classes=5,
        dropout=0.4,
    ).to(device)

    # Forward pass con tensor dummy
    dummy = torch.randn(2, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)

    n_params = model.count_parameters()
    print(f"[ExpertOA/EfficientNet-B0] Input shape:  {list(dummy.shape)}")
    print(f"[ExpertOA/EfficientNet-B0] Output shape: {list(out.shape)}")
    print(f"[ExpertOA/EfficientNet-B0] Parámetros entrenables: {n_params:,}")
    print(f"[ExpertOA/EfficientNet-B0] Output values: {out}")

    # Validaciones
    assert out.shape == (2, 5), (
        f"Shape de salida incorrecto: {out.shape}, esperado (2, 5)"
    )
    assert n_params > 0, "Modelo sin parámetros entrenables"

    # Verificar backbone y head params
    backbone_p = sum(p.numel() for p in model.get_backbone_params())
    head_p = sum(p.numel() for p in model.get_head_params())
    print(f"[ExpertOA/EfficientNet-B0] Backbone params: {backbone_p:,}")
    print(f"[ExpertOA/EfficientNet-B0] Head params:     {head_p:,}")

    # Verificar alias de compatibilidad
    alias_model = ExpertOAVGG16BN()
    assert isinstance(alias_model, ExpertOAEfficientNetB0), (
        "Alias ExpertOAVGG16BN no apunta a ExpertOAEfficientNetB0"
    )
    print("[ExpertOA/EfficientNet-B0] ✓ Alias ExpertOAVGG16BN OK")

    print("[ExpertOA/EfficientNet-B0] ✓ Verificación completada exitosamente")
    return model


if __name__ == "__main__":
    _test_model()

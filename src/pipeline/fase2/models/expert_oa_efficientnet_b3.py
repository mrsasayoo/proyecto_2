"""
Expert OA — EfficientNet-B3 para clasificación de osteoartritis de rodilla (KL 0-4).

Arquitectura:
    Backbone: timm efficientnet_b3 (preentrenado ImageNet1K)
    Bloques MBConv con Squeeze-and-Excitation → [B, 1536]
    Classifier (HEAD reemplazada):
        Dropout(p=0.4)
        Linear(1536 → 5)
    Salida: [B, 5] — logits para 5 grados Kellgren-Lawrence

Clases (Kellgren-Lawrence):
    0 = KL 0 — Normal
    1 = KL 1 — Dudoso
    2 = KL 2 — Leve
    3 = KL 3 — Moderado
    4 = KL 4 — Severo

Adaptaciones respecto al EfficientNet-B3 original (ImageNet, 1000 clases):
    1. Pesos preentrenados ImageNet1K via timm (transfer learning).
    2. Clasificador reemplazado: 1000 → 5 clases con Dropout(0.4).

Parámetros: ~10.7M totales (EfficientNet-B3 con head adaptada).

Nota: se exportan aliases ExpertOAVGG16BN y ExpertOAEfficientNetB0 para
compatibilidad hacia atrás con imports existentes.
"""

import timm
import torch
import torch.nn as nn


class ExpertOAEfficientNetB3(nn.Module):
    """EfficientNet-B3 preentrenado adaptado para Osteoarthritis Knee — 5 grados KL.

    Usa timm.create_model para cargar pesos pretrained, evitando el hash
    mismatch de torchvision.models.efficientnet_b3.

    EfficientNet-B3 tiene feature dimension = 1536 (vs 1280 de B0),
    lo que proporciona mayor capacidad representacional (~10.7M params totales).

    Entrada:  [B, 3, 224, 224] — radiografía de rodilla con CLAHE, normalizada
    Salida:   [B, 5] — logits crudos (antes de softmax)

    Args:
        num_classes: número de clases de salida. Default: 5
            (KL 0, KL 1, KL 2, KL 3, KL 4).
        dropout: probabilidad de Dropout en el clasificador.
            Default: 0.4.
        pretrained: cargar pesos ImageNet pretrained. Default: True.
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ── Cargar backbone EfficientNet-B3 con pesos ImageNet1K via timm ──
        # num_classes=0 → retorna features [B, 1536] sin head de clasificación
        self.model = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        # ── Clasificador personalizado (head) ──────────────────────
        # Reemplaza el head original de 1000 clases
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1536, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del modelo EfficientNet-B3.

        Args:
            x: tensor [B, 3, 224, 224] — imagen RGB normalizada.

        Returns:
            logits: tensor [B, 5] — logits crudos (antes de softmax).
        """
        features = self.model(x)  # [B, 1536]
        return self.classifier(features)  # [B, num_classes]

    def get_backbone_params(self) -> list[nn.Parameter]:
        """Retorna los parámetros del backbone (features) para fine-tuning diferencial."""
        return list(self.model.parameters())

    def get_head_params(self) -> list[nn.Parameter]:
        """Retorna los parámetros del clasificador (head) para fine-tuning diferencial."""
        return list(self.classifier.parameters())

    def count_parameters(self) -> int:
        """Retorna el número total de parámetros entrenables del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Retorna el número total de parámetros (entrenables + congelados)."""
        return sum(p.numel() for p in self.parameters())


# ── Aliases de compatibilidad hacia atrás ──────────────────────────
# Otros módulos del pipeline importan `ExpertOAVGG16BN` o
# `ExpertOAEfficientNetB0` desde este archivo.
# Estos aliases evitan romper esos imports mientras se migra gradualmente.
ExpertOAVGG16BN = ExpertOAEfficientNetB3
ExpertOAEfficientNetB0 = ExpertOAEfficientNetB3


def _test_model() -> None:
    """Verificación rápida: instanciar, forward pass, conteo de parámetros."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ExpertOA/EfficientNet-B3] Dispositivo: {device}")

    model = ExpertOAEfficientNetB3(
        num_classes=5,
        dropout=0.4,
    ).to(device)

    # Forward pass con tensor dummy
    dummy = torch.randn(2, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)

    n_params = model.count_parameters()
    n_all_params = model.count_all_parameters()
    print(f"[ExpertOA/EfficientNet-B3] Input shape:  {list(dummy.shape)}")
    print(f"[ExpertOA/EfficientNet-B3] Output shape: {list(out.shape)}")
    print(f"[ExpertOA/EfficientNet-B3] Parámetros entrenables: {n_params:,}")
    print(f"[ExpertOA/EfficientNet-B3] Parámetros totales:     {n_all_params:,}")
    print(f"[ExpertOA/EfficientNet-B3] Output values: {out}")

    # ── Validar shape de salida ────────────────────────────────────
    assert out.shape == (2, 5), (
        f"Shape de salida incorrecto: {out.shape}, esperado (2, 5)"
    )
    assert n_params > 0, "Modelo sin parámetros entrenables"

    # ── Validar conteo de parámetros ~10.7M ────────────────────────
    expected_min = 10_000_000  # ~10M mínimo
    expected_max = 12_000_000  # ~12M máximo
    assert expected_min < n_params < expected_max, (
        f"Parámetros fuera de rango esperado (~10.7M): {n_params:,}. "
        f"Rango aceptable: [{expected_min:,}, {expected_max:,}]"
    )
    print(f"[ExpertOA/EfficientNet-B3] ✓ Parámetros en rango (~10.7M)")

    # ── Verificar backbone y head params ───────────────────────────
    backbone_p = sum(p.numel() for p in model.get_backbone_params())
    head_p = sum(p.numel() for p in model.get_head_params())
    print(f"[ExpertOA/EfficientNet-B3] Backbone params: {backbone_p:,}")
    print(f"[ExpertOA/EfficientNet-B3] Head params:     {head_p:,}")

    # ── Verificar alias ExpertOAVGG16BN ────────────────────────────
    alias_model = ExpertOAVGG16BN()
    assert isinstance(alias_model, ExpertOAEfficientNetB3), (
        "Alias ExpertOAVGG16BN no apunta a ExpertOAEfficientNetB3"
    )
    print("[ExpertOA/EfficientNet-B3] ✓ Alias ExpertOAVGG16BN OK")

    # ── Verificar alias ExpertOAEfficientNetB0 ─────────────────────
    alias_model_b0 = ExpertOAEfficientNetB0()
    assert isinstance(alias_model_b0, ExpertOAEfficientNetB3), (
        "Alias ExpertOAEfficientNetB0 no apunta a ExpertOAEfficientNetB3"
    )
    print("[ExpertOA/EfficientNet-B3] ✓ Alias ExpertOAEfficientNetB0 OK")

    print("[ExpertOA/EfficientNet-B3] ✓ Verificación completada exitosamente")
    return model


if __name__ == "__main__":
    _test_model()

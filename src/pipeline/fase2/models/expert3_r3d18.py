"""
Expert 3 — MC3-18 adaptado para clasificación binaria de nódulos pulmonares 3D.

Arquitectura:
    Backbone: torchvision.models.video.mc3_18 (ResNet 3D mixto: Conv3D en stem/layer1, Conv2D(1×3×3) en layers 2-4)
    Entrada:  [B, 1, 64, 64, 64] — parche CT monocanal, float32
    Salida:   [B, 2] — logits para clasificación binaria (0=no nódulo, 1=nódulo)

Adaptaciones respecto al MC3-18 original (Kinetics400, 3 canales RGB, 400 clases):
    1. Capa conv1: in_channels 3 → 1 (CT monocanal).
       Estrategia: se inicializa desde cero con kaiming_normal_ (He et al., 2015).
       NO se promedian pesos preentrenados porque el modelo se entrena from scratch
       (sin pesos ImageNet/Kinetics — dominio médico 3D incompatible con video RGB).
    2. Cabeza clasificadora: fc 400 → 2 clases.
    3. SpatialDropout3d(p=0.15) después del stem (primera capa convolucional).
    4. Dropout(p=0.4) antes de la capa de clasificación final.

NOTA SOBRE CONTEO DE PARÁMETROS:
    MC3-18 (torchvision) usa Conv3D (kernel 3×3×3) solo en stem y layer1, y
    Conv2D(1×3×3) sin componente temporal en layers 2-4. Esto reduce los
    parámetros significativamente respecto a R3D-18 (convoluciones 3D puras).
    Total: ~11.7M params (~3× menos que R3D-18 con ~33.2M params).
    Con 14,728 muestras el ratio params/datos es ~795:1 (vs ~2,250:1 con R3D-18).

Regularización aplicada (justificada por ratio params/datos desfavorable):
    - SpatialDropout3d(p=0.15): desactiva canales completos en feature maps 3D
    - Dropout(p=0.4): antes de la capa FC final
    - WeightDecay=0.03 en AdamW (definido en expert3_config.py)
    - FocalLoss(gamma=2, alpha=0.85) (definido en losses.py)

Autor: Pipeline Expert3 — Fase 2

NOTA: El modelo se llama MC3-18 (no R3D-18). R3D-18 se menciona aquí solo para
comparación de parámetros. El modelo real usado es MC3-18.
"""

import torch
import torch.nn as nn
from torchvision.models.video import mc3_18


class SpatialDropout3d(nn.Module):
    """
    Dropout espacial 3D: desactiva canales completos del feature map.

    A diferencia de nn.Dropout3d estándar (que opera por elemento), este módulo
    desactiva canales enteros. Esto es más efectivo para datos volumétricos donde
    las activaciones son espacialmente correlacionadas (ej: tejido pulmonar en CT).

    Args:
        p: probabilidad de desactivar un canal completo. Default: 0.15.
    """

    def __init__(self, p: float = 0.15):
        super().__init__()
        self.dropout = nn.Dropout3d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor [B, C, D, H, W]
        Returns:
            tensor [B, C, D, H, W] con canales desactivados aleatoriamente (solo en train)
        """
        return self.dropout(x)


class Expert3MC318(nn.Module):
    """
    MC3-18 adaptado para clasificación binaria de parches 3D CT (LUNA16).

    Usa Conv3D (3×3×3) en stem/layer1 y Conv2D (1×3×3) en layers 2-4,
    reduciendo parámetros a ~11.7M (vs ~33.2M de R3D-18).

    NOTA: Este modelo usa MC3-18, NO R3D-18. La mención de R3D-18 es solo
    para comparación de número de parámetros.

    Entrada:  [B, 1, 64, 64, 64] — parche CT monocanal normalizado
    Salida:   [B, 2] — logits para {no-nódulo, nódulo}

    Args:
        spatial_dropout_p: probabilidad de SpatialDropout3d después del stem.
            Default: 0.15 (de expert3_config.EXPERT3_SPATIAL_DROPOUT_3D).
        fc_dropout_p: probabilidad de Dropout antes de la FC final.
            Default: 0.4 (de expert3_config.EXPERT3_DROPOUT_FC).
        num_classes: número de clases de salida. Default: 2.
    """

    def __init__(
        self,
        spatial_dropout_p: float = 0.15,
        fc_dropout_p: float = 0.4,
        num_classes: int = 2,
    ):
        super().__init__()

        # ── Cargar backbone MC3-18 sin pesos preentrenados ──────────────
        backbone = mc3_18(weights=None)

        # ── Adaptar conv1: 3 canales → 1 canal (CT monocanal) ──────────
        # Estrategia: inicializar desde cero con kaiming_normal_.
        # Justificación: no se usan pesos preentrenados (dominio médico 3D ≠ video RGB),
        # así que promediar canales no tiene sentido. Kaiming init es el estándar
        # para ReLU networks (He et al., 2015, arXiv:1502.01852).
        old_conv1 = backbone.stem[0]
        new_conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )
        nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
        backbone.stem[0] = new_conv1

        # ── Construir el stem + spatial dropout ─────────────────────────
        self.stem = nn.Sequential(
            backbone.stem,
            SpatialDropout3d(p=spatial_dropout_p),
        )

        # ── Bloques residuales (sin modificación) ───────────────────────
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # ── Average pooling adaptativo ──────────────────────────────────
        self.avgpool = backbone.avgpool

        # ── Cabeza clasificadora con dropout ────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=fc_dropout_p),
            nn.Linear(512, num_classes),
        )

        # ── Inicializar la capa FC final ────────────────────────────────
        nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo MC3-18 adaptado.

        Args:
            x: tensor [B, 1, 64, 64, 64] — parche CT monocanal

        Returns:
            logits: tensor [B, 2] — logits crudos (antes de softmax/sigmoid)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Expert3/MC3-18] Dispositivo: {device}")

    model = Expert3MC318(
        spatial_dropout_p=0.15,
        fc_dropout_p=0.4,
        num_classes=2,
    ).to(device)

    # Forward pass con tensor dummy
    dummy = torch.zeros(2, 1, 64, 64, 64, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)

    n_params = model.count_parameters()
    print(f"[Expert3/MC3-18] Input shape:  {list(dummy.shape)}")
    print(f"[Expert3/MC3-18] Output shape: {list(out.shape)}")
    print(f"[Expert3/MC3-18] Parámetros entrenables: {n_params:,}")
    print(f"[Expert3/MC3-18] Output values: {out}")

    # Validaciones
    assert out.shape == (2, 2), (
        f"Shape de salida incorrecto: {out.shape}, esperado (2, 2)"
    )
    assert n_params > 0, "Modelo sin parámetros entrenables"
    print(f"[Expert3/MC3-18] ✓ Verificación completada exitosamente")
    return model


if __name__ == "__main__":
    _test_model()

"""
Expert 4 -- ResNet 3D (R3D-18) para clasificacion binaria de volumenes CT abdominales (PDAC).

Arquitectura:
    Backbone: torchvision.models.video.r3d_18 (ResNet 3D con Conv3D puras 3x3x3)
    Entrada:  [B, 1, 64, 64, 64] -- volumen CT abdominal monocanal
    Salida:   [B, 2] -- logits para clasificacion binaria (PDAC- / PDAC+)

Adaptaciones respecto al R3D-18 original (Kinetics400, 3 canales RGB, 400 clases):
    1. Capa conv1 (stem[0]): in_channels 3 -> 1 (CT monocanal).
       Inicializacion kaiming_normal_ desde cero (He et al., 2015).
       No se promedian pesos preentrenados: dominio medico 3D incompatible con video RGB.
    2. Cabeza clasificadora: se reemplaza fc(512->400) por
       nn.Sequential(nn.Dropout(p=0.5), nn.Linear(512, 2)).
    3. Inicializacion desde cero con weights=None.

Conteo de parametros:
    R3D-18 usa Conv3D puras (3x3x3) en TODOS los layers (a diferencia de MC3-18
    que usa Conv2D(1x3x3) en layers 2-4). Esto resulta en ~33.4M parametros,
    significativamente mas que los ~15M objetivo.

    Justificacion de usar R3D-18 (~33.4M) en lugar de un ResNet-10 3D custom (~15M):
    - R3D-18 es una arquitectura bien estudiada y reproducible (Tran et al., 2018,
      "A Closer Look at Spatiotemporal Convolutions for Action Recognition").
    - Las Conv3D puras capturan dependencias espaciotemporales completas en las 3
      dimensiones del volumen CT, critico para detectar patrones volumetricos de PDAC.
    - La regularizacion agresiva (Dropout p=0.5, WeightDecay, FocalLoss, k-fold CV,
      early stopping) compensa el exceso de parametros para el dataset pequeno.
    - Un modelo custom sin validacion publicada introduce riesgo de bugs silenciosos.

Regularizacion (dataset pequeno: ~281 volumenes, ratio params/datos ~ 119,000:1):
    - Dropout(p=0.5) en la cabeza clasificadora (mas agresivo que Expert 3)
    - WeightDecay=0.05 en AdamW (definido en expert4_config.py)
    - FocalLoss(gamma=2, alpha=0.75) (definido en expert4_config.py)
    - k-fold CV (k=5) obligatorio por tamano del dataset
    - Early stopping (patience=15) para prevenir overfitting
    - Se recomienda data augmentation volumetrica (rotaciones, flips, elastic deformation)

Nota sobre el nombre del archivo:
    El archivo se llama expert4_swin3d.py por compatibilidad historica con el pipeline.
    La clase principal es ExpertPancreasResNet3D. Se provee un alias
    ExpertPancreasSwin3D = ExpertPancreasResNet3D para compatibilidad con imports existentes.

Autor: Pipeline Expert4 -- Fase 2
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


class ExpertPancreasResNet3D(nn.Module):
    """
    R3D-18 adaptado para clasificacion binaria PDAC+/PDAC- en CT abdominal.

    Usa torchvision.models.video.r3d_18 como backbone con Conv3D puras (3x3x3)
    en todos los layers. Adaptado para entrada monocanal y 2 clases de salida.

    Arquitectura interna:
        stem:     Conv3d(1, 64, 3x7x7, stride=1x2x2) + BN + ReLU
        layer1:   2x BasicBlock(64, 64)   -- Conv3D 3x3x3
        layer2:   2x BasicBlock(64, 128)  -- Conv3D 3x3x3, downsample 2x
        layer3:   2x BasicBlock(128, 256) -- Conv3D 3x3x3, downsample 2x
        layer4:   2x BasicBlock(256, 512) -- Conv3D 3x3x3, downsample 2x
        avgpool:  AdaptiveAvgPool3d(1, 1, 1)
        head:     Dropout(0.5) -> Linear(512, 2)

    Entrada:  [B, 1, 64, 64, 64]
    Salida:   [B, 2] logits

    Parametros totales: ~33.4M (Conv3D puras en todos los layers).
    Justificacion: arquitectura bien validada (Tran et al., 2018) + regularizacion
    agresiva para compensar ratio params/datos desfavorable (~119,000:1).

    Args:
        in_channels: canales de entrada. Default: 1 (CT monocanal).
        num_classes: clases de salida. Default: 2 (PDAC-/PDAC+).
        dropout_p: probabilidad de dropout en la cabeza clasificadora.
            Default: 0.5. Mas agresivo que Expert 3 (0.4) por dataset mas
            pequeno (~281 vs ~14,728 volumenes).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()

        # ── Cargar backbone R3D-18 sin pesos preentrenados ─────────────
        backbone = r3d_18(weights=None)

        # ── Adaptar conv1 (stem[0]): 3 canales -> 1 canal (CT monocanal) ──
        # Estrategia: inicializar desde cero con kaiming_normal_.
        # Justificacion: no se usan pesos preentrenados (dominio medico 3D
        # incompatible con video RGB), kaiming init es el estandar para ReLU
        # networks (He et al., 2015, arXiv:1502.01852).
        old_conv1 = backbone.stem[0]
        new_conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )
        nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
        backbone.stem[0] = new_conv1

        # ── Extraer componentes del backbone ───────────────────────────
        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # ── Cabeza clasificadora con dropout agresivo ──────────────────
        # Dropout p=0.5: mas agresivo que Expert 3 (p=0.4) porque el dataset
        # PANORAMA es mucho mas pequeno (~281 vs ~14,728 volumenes).
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )

        # ── Inicializar la capa FC final ───────────────────────────────
        nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del R3D-18 adaptado.

        Args:
            x: tensor [B, 1, 64, 64, 64] -- volumen CT abdominal monocanal

        Returns:
            logits: tensor [B, 2] -- logits crudos (antes de softmax/sigmoid)
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
        """Retorna el numero total de parametros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Retorna el numero total de parametros (entrenables + congelados)."""
        return sum(p.numel() for p in self.parameters())


# ── Alias de compatibilidad ─────────────────────────────────────────────
# El pipeline (train_expert4.py y otros) importa ExpertPancreasSwin3D.
# Este alias garantiza retrocompatibilidad sin tocar otros archivos.
ExpertPancreasSwin3D = ExpertPancreasResNet3D


def _test_model() -> None:
    """Verificacion rapida: instanciar, forward pass, conteo de parametros, alias."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Expert4/R3D-18] Dispositivo: {device}")

    # ── Instanciar modelo ──────────────────────────────────────────
    model = ExpertPancreasResNet3D(
        in_channels=1,
        num_classes=2,
        dropout_p=0.5,
    ).to(device)

    # ── Forward pass con tensor dummy ──────────────────────────────
    dummy = torch.randn(1, 1, 64, 64, 64, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)

    n_params = model.count_parameters()
    n_all_params = model.count_all_parameters()

    print(f"[Expert4/R3D-18] Input shape:  {list(dummy.shape)}")
    print(f"[Expert4/R3D-18] Output shape: {list(out.shape)}")
    print(f"[Expert4/R3D-18] Parametros entrenables: {n_params:,}")
    print(f"[Expert4/R3D-18] Parametros totales:     {n_all_params:,}")
    print(f"[Expert4/R3D-18] Output values: {out}")

    # ── Validacion: output shape ───────────────────────────────────
    assert out.shape == (1, 2), (
        f"Shape de salida incorrecto: {out.shape}, esperado (1, 2)"
    )

    # ── Validacion: parametros > 0 ─────────────────────────────────
    assert n_params > 0, "Modelo sin parametros entrenables"

    # ── Validacion: conteo de parametros en rango esperado ─────────
    # R3D-18 con Conv3D puras: ~33.4M params.
    # Rango permisivo: 30M-40M (variaciones por in_channels y num_classes).
    assert 30_000_000 < n_params < 40_000_000, (
        f"Parametros fuera de rango esperado para R3D-18: {n_params:,}. "
        f"Esperado ~33.4M (30M-40M)"
    )
    print(f"[Expert4/R3D-18] Conteo de parametros OK (~33.4M para R3D-18)")

    # ── Validacion: alias de compatibilidad ────────────────────────
    alias_model = ExpertPancreasSwin3D(in_channels=1, num_classes=2)
    assert isinstance(alias_model, ExpertPancreasResNet3D), (
        "ExpertPancreasSwin3D debe ser alias de ExpertPancreasResNet3D"
    )
    print(f"[Expert4/R3D-18] Alias ExpertPancreasSwin3D -> ExpertPancreasResNet3D OK")

    # ── Resumen ────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  Expert 4 — ResNet 3D (R3D-18) para PDAC CT abdominal")
    print(f"  Arquitectura: torchvision.models.video.r3d_18 (Conv3D puras)")
    print(f"  Entrada:  [B, 1, 64, 64, 64]")
    print(f"  Salida:   [B, 2] logits")
    print(f"  Params:   {n_params:,} (~33.4M, Conv3D puras en todos los layers)")
    print(f"  Head:     Dropout(p=0.5) -> Linear(512, 2)")
    print(f"  Exports:  ExpertPancreasResNet3D, ExpertPancreasSwin3D (alias)")
    print(f"{'=' * 65}")
    print(f"[Expert4/R3D-18] Verificacion completada exitosamente")

    return model


if __name__ == "__main__":
    _test_model()

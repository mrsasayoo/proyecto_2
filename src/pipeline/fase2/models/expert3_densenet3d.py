"""
Expert 3 — DenseNet 3D para clasificación binaria de nódulos pulmonares 3D.

Arquitectura:
    Backbone: DenseNet 3D implementado desde cero (Huang et al., 2017)
    con convoluciones 3D para datos volumétricos CT.

    Configuración:
        growth_rate  = 32
        block_layers = [4, 8, 16, 12]  (4 dense blocks)
        init_features = 64
        compression  = 0.5 (factor de reducción en transition layers)
        bottleneck   = True (cada dense layer usa 1×1×1 → 3×3×3)

    Entrada:  [B, 1, 64, 64, 64] — parche CT monocanal, float32
    Salida:   [B, 2] — logits para clasificación binaria (0=no nódulo, 1=nódulo)

    Parámetros totales: ~6.7M (~7M objetivo)
    Con 14,728 muestras: ratio params/datos ≈ 455:1

Diseño DenseNet 3D:
    A diferencia de torchvision (que no ofrece DenseNet 3D nativo), esta
    implementación usa Conv3d(3×3×3) en los dense layers para procesar
    volúmenes CT tridimensionales directamente.

    Cada _DenseLayer aplica el patrón bottleneck:
        BN → ReLU → Conv3d(1×1×1) → BN → ReLU → Conv3d(3×3×3)
    Esto reduce el costo computacional manteniendo la conectividad densa.

    Cada _Transition layer aplica:
        BN → ReLU → Conv3d(1×1×1) → AvgPool3d(2×2×2)
    Reduciendo features por factor 'compression' y resolución espacial ×2.

Regularización aplicada (justificada por ratio params/datos):
    - SpatialDropout3d(p=0.15): desactiva canales completos en feature maps 3D
    - Dropout(p=0.4): antes de la capa FC final
    - WeightDecay=0.03 en AdamW (definido en expert3_config.py)
    - FocalLoss(gamma=2, alpha=0.85) (definido en losses.py)

Alias de compatibilidad:
    Expert3MC318 = Expert3DenseNet3D  (para imports existentes en train_expert3.py)

Autor: Pipeline Expert3 — Fase 2
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Componentes DenseNet 3D
# ═══════════════════════════════════════════════════════════════════════


class SpatialDropout3d(nn.Module):
    """
    Dropout espacial 3D: desactiva canales completos del feature map.

    A diferencia de nn.Dropout3d estándar (que opera por elemento), este módulo
    desactiva canales enteros. Esto es más efectivo para datos volumétricos donde
    las activaciones son espacialmente correlacionadas (ej: tejido pulmonar en CT).

    Args:
        p: probabilidad de desactivar un canal completo. Default: 0.15.
    """

    def __init__(self, p: float = 0.15) -> None:
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


class _DenseLayer(nn.Module):
    """
    Una capa dentro de un DenseBlock 3D (patrón bottleneck).

    Secuencia: BN → ReLU → Conv3d(1×1×1) → BN → ReLU → Conv3d(3×3×3)
    La salida se concatena con la entrada (conectividad densa).

    Args:
        in_features: número de canales de entrada (acumulados del bloque).
        growth_rate: número de filtros producidos por esta capa (k en el paper).
        bn_size: factor multiplicativo para el bottleneck.
            El Conv1×1×1 produce bn_size * growth_rate features intermedios.
    """

    def __init__(
        self,
        in_features: int,
        growth_rate: int,
        bn_size: int = 4,
    ) -> None:
        super().__init__()
        mid_features = bn_size * growth_rate

        # Bottleneck: 1×1×1 reduce dimensionalidad
        self.bn1 = nn.BatchNorm3d(in_features)
        self.conv1 = nn.Conv3d(
            in_features,
            mid_features,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        # Conv 3×3×3 produce growth_rate features nuevos
        self.bn2 = nn.BatchNorm3d(mid_features)
        self.conv2 = nn.Conv3d(
            mid_features,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor [B, C_in, D, H, W]
        Returns:
            tensor [B, C_in + growth_rate, D, H, W] — concatenación densa
        """
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    """
    Bloque denso: secuencia de n_layers DenseLayers con conectividad densa.

    Cada capa recibe como entrada la concatenación de todos los feature maps
    anteriores dentro del bloque, implementando la conectividad densa
    x_l = H_l([x_0, x_1, ..., x_{l-1}]) del paper DenseNet.

    Args:
        n_layers: número de DenseLayers en este bloque.
        in_features: canales de entrada al bloque.
        growth_rate: features nuevos por capa (k).
        bn_size: factor bottleneck.
    """

    def __init__(
        self,
        n_layers: int,
        in_features: int,
        growth_rate: int,
        bn_size: int = 4,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                _DenseLayer(
                    in_features=in_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    """
    Capa de transición entre DenseBlocks.

    Reduce el número de features (compresión) y la resolución espacial:
        BN → ReLU → Conv3d(1×1×1) → AvgPool3d(2×2×2)

    Args:
        in_features: canales de entrada.
        out_features: canales de salida (típicamente in_features * compression).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm3d(in_features)
        self.conv = nn.Conv3d(
            in_features,
            out_features,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(F.relu(self.bn(x), inplace=True))
        return self.pool(out)


# ═══════════════════════════════════════════════════════════════════════
# Modelo principal
# ═══════════════════════════════════════════════════════════════════════


class Expert3DenseNet3D(nn.Module):
    """
    DenseNet 3D para clasificación binaria de parches 3D CT (LUNA16).

    Implementación from-scratch de DenseNet (Huang et al., 2017) con
    convoluciones 3D, diseñada para volúmenes CT médicos.

    Configuración por defecto: ~6.7M parámetros.
        growth_rate=32, block_layers=[4, 8, 16, 12], init_features=64, compression=0.5

    Entrada:  [B, 1, 64, 64, 64] — parche CT monocanal normalizado
    Salida:   [B, 2] — logits para {no-nódulo, nódulo}

    Args:
        in_channels: canales de entrada. Default: 1 (CT monocanal).
        num_classes: clases de salida. Default: 2.
        growth_rate: features nuevos por dense layer (k). Default: 32.
        block_layers: número de layers por dense block. Default: [4, 8, 16, 12].
        init_features: features iniciales del stem conv. Default: 64.
        bn_size: factor bottleneck (mid = bn_size * growth_rate). Default: 4.
        compression: factor de reducción en transitions. Default: 0.5.
        spatial_dropout_p: probabilidad de SpatialDropout3d después del stem.
            Default: 0.15 (de expert3_config.EXPERT3_SPATIAL_DROPOUT_3D).
        fc_dropout_p: probabilidad de Dropout antes de la FC final.
            Default: 0.4 (de expert3_config.EXPERT3_DROPOUT_FC).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        growth_rate: int = 32,
        block_layers: list[int] | None = None,
        init_features: int = 64,
        bn_size: int = 4,
        compression: float = 0.5,
        spatial_dropout_p: float = 0.15,
        fc_dropout_p: float = 0.4,
    ) -> None:
        super().__init__()

        if block_layers is None:
            block_layers = [4, 8, 16, 12]

        self.growth_rate = growth_rate
        self.block_layers = block_layers

        # ── Stem: Conv3d(7×7×7) + BN + ReLU + MaxPool + SpatialDropout ──
        self.stem = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv3d(
                            in_channels,
                            init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm3d(init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
                    ("spatial_dropout", SpatialDropout3d(p=spatial_dropout_p)),
                ]
            )
        )

        # ── Dense blocks + Transition layers ────────────────────────────
        n_features = init_features
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, n_layers in enumerate(block_layers):
            block = _DenseBlock(
                n_layers=n_layers,
                in_features=n_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
            )
            self.dense_blocks.append(block)
            n_features = n_features + n_layers * growth_rate

            # Transition después de cada bloque excepto el último
            if i < len(block_layers) - 1:
                out_features = int(n_features * compression)
                transition = _Transition(n_features, out_features)
                self.transitions.append(transition)
                n_features = out_features

        # ── BN final ────────────────────────────────────────────────────
        self.final_bn = nn.BatchNorm3d(n_features)

        # ── Global Average Pooling ──────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # ── Cabeza clasificadora con dropout ────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=fc_dropout_p),
            nn.Linear(n_features, num_classes),
        )

        # ── Inicialización de pesos ─────────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Inicialización de pesos siguiendo las convenciones de DenseNet:
        - Conv3d: Kaiming normal (He et al., 2015) para ReLU
        - BatchNorm3d: weight=1, bias=0
        - Linear: normal(0, 0.01), bias=0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del DenseNet 3D.

        Args:
            x: tensor [B, 1, 64, 64, 64] — parche CT monocanal

        Returns:
            logits: tensor [B, 2] — logits crudos (antes de softmax/sigmoid)
        """
        # Stem: [B,1,64,64,64] → [B,64,16,16,16]
        x = self.stem(x)

        # Dense blocks + transitions
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        # Final BN + ReLU
        x = F.relu(self.final_bn(x), inplace=True)

        # Global average pooling + flatten
        x = self.avgpool(x)
        x = x.flatten(1)

        # Clasificador
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        """Retorna el número total de parámetros entrenables del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Retorna el número total de parámetros (entrenables + congelados)."""
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════
# Alias de compatibilidad
# ═══════════════════════════════════════════════════════════════════════

# train_expert3.py importa `Expert3MC318` — este alias mantiene la
# compatibilidad sin necesidad de modificar otros archivos.
Expert3MC318 = Expert3DenseNet3D


# ═══════════════════════════════════════════════════════════════════════
# Test de verificación
# ═══════════════════════════════════════════════════════════════════════


def _test_model() -> Expert3DenseNet3D:
    """Verificación rápida: instanciar, forward pass, conteo de parámetros."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Expert3/DenseNet3D] Dispositivo: {device}")

    # ── Instanciar modelo con config por defecto ────────────────────
    model = Expert3DenseNet3D(
        in_channels=1,
        num_classes=2,
        growth_rate=32,
        block_layers=[4, 8, 16, 12],
        init_features=64,
        spatial_dropout_p=0.15,
        fc_dropout_p=0.4,
    ).to(device)

    # ── Forward pass con tensor dummy ───────────────────────────────
    dummy = torch.zeros(2, 1, 64, 64, 64, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)

    n_params = model.count_parameters()
    n_all_params = model.count_all_parameters()

    print(f"[Expert3/DenseNet3D] Input shape:  {list(dummy.shape)}")
    print(f"[Expert3/DenseNet3D] Output shape: {list(out.shape)}")
    print(f"[Expert3/DenseNet3D] Parámetros entrenables: {n_params:,}")
    print(f"[Expert3/DenseNet3D] Parámetros totales:     {n_all_params:,}")
    print(
        f"[Expert3/DenseNet3D] Config: growth_rate=32, "
        f"block_layers=[4, 8, 16, 12], init_features=64"
    )
    print(f"[Expert3/DenseNet3D] Output values: {out}")

    # ── Validaciones ────────────────────────────────────────────────
    assert out.shape == (2, 2), (
        f"Shape de salida incorrecto: {out.shape}, esperado (2, 2)"
    )
    assert n_params > 0, "Modelo sin parámetros entrenables"
    assert 3_000_000 <= n_params <= 15_000_000, (
        f"Parámetros fuera de rango esperado (~7M): {n_params:,}"
    )

    # ── Verificar alias de compatibilidad ───────────────────────────
    alias_model = Expert3MC318(
        spatial_dropout_p=0.15,
        fc_dropout_p=0.4,
        num_classes=2,
    )
    assert isinstance(alias_model, Expert3DenseNet3D), (
        "Expert3MC318 no es alias de Expert3DenseNet3D"
    )
    print(f"[Expert3/DenseNet3D] ✓ Alias Expert3MC318 funciona correctamente")

    print(f"[Expert3/DenseNet3D] ✓ Verificación completada exitosamente")
    return model


if __name__ == "__main__":
    _test_model()

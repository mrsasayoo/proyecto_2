"""
backbone_densenet3d.py — DenseNet-121 3D para Clasificación de Nódulos Pulmonares
==================================================================================

Implementación pura con torch.nn para volúmenes médicos 3D (LUNA16).
Sin pesos preentrenados, sin dependencias externas más allá de PyTorch.

Arquitectura DenseNet-121 3D (adaptación volumétrica de Huang et al., CVPR 2017):
  - Entrada: [B, 1, 64, 64, 64] — parche 3D de CT pulmonar, 1 canal
  - Stem: Conv3d 7×7×7 stride 2 → BN → ReLU → MaxPool3d 3×3×3 stride 2
  - 4 DenseBlocks con config [6, 12, 24, 16], growth_rate=32
  - 3 Transition layers: BN → ReLU → Conv3d 1×1×1 (compresión 0.5) → AvgPool3d 2×2×2
  - BN final → AdaptiveAvgPool3d(1) → Dropout → Linear → logits [B, 2]
  - ~11.14M parámetros

Gradient checkpointing disponible en denseblock2..4 para reducir VRAM.

Interface pública:
  DenseNet3D(growth_rate, block_config, ...) → nn.Module
  forward(x) → [B, num_classes]

Referencia:
    Huang et al., "Densely Connected Convolutional Networks",
    CVPR 2017. https://arxiv.org/abs/1608.06993
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

log = logging.getLogger("fase1")


# ── Capa densa individual (3D) ─────────────────────────────────────────────
class _DenseLayer3D(nn.Module):
    """Single dense layer with bottleneck pattern for 3D volumes.

    Architecture: BN → ReLU → Conv3d(1×1×1) → BN → ReLU → Conv3d(3×3×3) → Dropout3d.

    Each layer produces ``growth_rate`` new feature maps that are concatenated
    with all previous feature maps in the dense block, enabling feature reuse
    and direct gradient flow.

    Args:
        num_input_features: Number of input channels (accumulated via concatenation).
        growth_rate: Number of output feature maps produced by this layer (k).
        bn_size: Bottleneck factor — intermediate channels = bn_size × growth_rate.
        dropout_rate: Spatial dropout probability after the 3×3×3 convolution.
        memory_efficient: If True, use gradient checkpointing to trade compute for memory.
    """

    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int = 4,
        dropout_rate: float = 0.0,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        # Canales intermedios del bottleneck
        inter_channels = bn_size * growth_rate

        # Bottleneck: BN → ReLU → Conv3d 1×1×1
        self.bn1 = nn.BatchNorm3d(num_input_features)
        self.conv1 = nn.Conv3d(
            num_input_features, inter_channels, kernel_size=1, bias=False
        )

        # Extracción espacial: BN → ReLU → Conv3d 3×3×3
        self.bn2 = nn.BatchNorm3d(inter_channels)
        self.conv2 = nn.Conv3d(
            inter_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )

        # Dropout espacial 3D para regularización
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else None
        self.memory_efficient = memory_efficient

    def _bottleneck(self, x: Tensor) -> Tensor:
        """Bottleneck: BN → ReLU → Conv 1×1×1."""
        return self.conv1(F.relu(self.bn1(x), inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        """Compute new feature maps from accumulated input features.

        Args:
            x: Concatenated feature maps [B, C_accumulated, D, H, W].

        Returns:
            New feature maps [B, growth_rate, D, H, W].
        """
        # Gradient checkpointing solo en el bottleneck (mayor consumo de memoria)
        if self.memory_efficient and x.requires_grad:
            bottleneck_out = grad_checkpoint(self._bottleneck, x, use_reentrant=False)
        else:
            bottleneck_out = self._bottleneck(x)

        # Convolución espacial 3×3×3
        out = self.conv2(F.relu(self.bn2(bottleneck_out), inplace=True))

        # Dropout espacial
        if self.dropout is not None:
            out = self.dropout(out)

        return out


# ── Bloque denso (DenseBlock 3D) ───────────────────────────────────────────
class _DenseBlock3D(nn.ModuleDict):
    """Dense block: stack of dense layers with skip connections via concatenation.

    Each layer receives ALL preceding feature maps concatenated along the
    channel dimension, enabling maximum feature reuse and gradient flow.

    Args:
        num_layers: Number of dense layers in this block.
        num_input_features: Input channels to the block.
        growth_rate: Feature maps produced per layer (k).
        bn_size: Bottleneck factor for each layer.
        dropout_rate: Dropout probability for each layer.
        memory_efficient: Whether to use gradient checkpointing per layer.
    """

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        growth_rate: int,
        bn_size: int = 4,
        dropout_rate: float = 0.0,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            # Cada capa recibe num_input_features + i * growth_rate canales
            layer = _DenseLayer3D(
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout_rate=dropout_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, x: Tensor) -> Tensor:
        """Forward: concatenate each layer's output with all previous features.

        Args:
            x: Input tensor [B, C_in, D, H, W].

        Returns:
            Concatenated features [B, C_in + num_layers × growth_rate, D, H, W].
        """
        features = [x]
        for layer in self.values():
            concat_input = torch.cat(features, dim=1)
            new_features = layer(concat_input)
            features.append(new_features)
        return torch.cat(features, dim=1)


# ── Capa de transición (3D) ────────────────────────────────────────────────
class _Transition3D(nn.Sequential):
    """Transition layer between dense blocks: reduces channels and spatial resolution.

    Architecture: BN → ReLU → Conv3d(1×1×1, compression) → AvgPool3d(2×2×2).

    Args:
        num_input_features: Input channels (output of preceding dense block).
        num_output_features: Output channels (typically input × compression_rate).
    """

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm3d(num_input_features)
        self.conv = nn.Conv3d(
            num_input_features, num_output_features, kernel_size=1, bias=False
        )
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        """BN → ReLU → Conv 1×1×1 → AvgPool 2×2×2."""
        out = self.conv(F.relu(self.bn(x), inplace=True))
        return self.pool(out)


# ── DenseNet3D completa ────────────────────────────────────────────────────
class DenseNet3D(nn.Module):
    """3D DenseNet for binary classification of volumetric medical images.

    Designed for pulmonary nodule classification on LUNA16 patches.
    Accepts single-channel 3D volumes and outputs class logits.

    Architecture:
        1. Stem: Conv3d 7×7×7 stride 2 → BN → ReLU → MaxPool3d 3×3×3 stride 2
        2. 4 DenseBlocks interleaved with 3 Transition layers
        3. Final BN → AdaptiveAvgPool3d(1) → Flatten → Dropout → Linear

    Gradient checkpointing can be activated on denseblock2..4 via
    ``model.set_grad_checkpointing(True)`` to reduce peak VRAM usage.

    Args:
        growth_rate: Feature maps produced per dense layer (k).
        block_config: Tuple with number of layers per dense block.
        num_init_features: Output channels of the initial convolution.
        bn_size: Bottleneck factor in dense layers.
        dropout_rate: Spatial dropout rate inside dense layers.
        fc_dropout: Dropout rate before the final linear classifier.
        num_classes: Number of output classes (2 for binary nodule classification).
        in_channels: Input channels (1 for single-channel CT volumes).
        compression_rate: Channel compression factor in transition layers (θ).
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: tuple[int, ...] | Sequence[int] = (6, 12, 24, 16),
        num_init_features: int = 32,
        bn_size: int = 4,
        dropout_rate: float = 0.2,
        fc_dropout: float = 0.4,
        num_classes: int = 2,
        in_channels: int = 1,
        compression_rate: float = 0.5,
    ) -> None:
        super().__init__()

        self._num_classes = num_classes
        self._grad_checkpointing = False

        # ── Stem: Conv3d 7×7×7 stride 2 → BN → ReLU → MaxPool3d 3×3×3 ────
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv3d(
                            in_channels,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm3d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    (
                        "pool0",
                        nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                    ),
                ]
            )
        )

        # ── DenseBlocks + Transition layers ────────────────────────────────
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Bloque denso
            block = _DenseBlock3D(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout_rate=dropout_rate,
                memory_efficient=False,  # se activa después con set_grad_checkpointing
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate

            # Transición después de cada bloque excepto el último
            if i < len(block_config) - 1:
                out_features = int(num_features * compression_rate)
                transition = _Transition3D(num_features, out_features)
                self.features.add_module(f"transition{i + 1}", transition)
                num_features = out_features

        # BN final (antes del pooling global)
        self.features.add_module("norm_final", nn.BatchNorm3d(num_features))

        # Guardar el número de canales finales (útil para reemplazar el head)
        self._num_features = num_features

        # ── Clasificador ───────────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc_drop = nn.Dropout(p=fc_dropout)
        self.classifier = nn.Linear(num_features, num_classes)

        # ── Inicialización de pesos ────────────────────────────────────────
        self._initialize_weights()

        log.info(
            "[DenseNet3D] Arquitectura construida:\n"
            f"    in_channels      : {in_channels}\n"
            f"    num_init_features: {num_init_features}\n"
            f"    growth_rate      : {growth_rate}\n"
            f"    block_config     : {block_config}\n"
            f"    compression_rate : {compression_rate}\n"
            f"    dropout_rate     : {dropout_rate}\n"
            f"    fc_dropout       : {fc_dropout}\n"
            f"    num_classes      : {num_classes}\n"
            f"    canales finales  : {num_features}\n"
            f"    parámetros       : {sum(p.numel() for p in self.parameters()):,}"
        )

    # ── Propiedades públicas ───────────────────────────────────────────────

    @property
    def num_features(self) -> int:
        """Number of channels before the classifier head (for head replacement)."""
        return self._num_features

    # ── Gradient checkpointing ─────────────────────────────────────────────

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing on denseblock2..4.

        When enabled, intermediate activations in these blocks are recomputed
        during backprop instead of stored, reducing peak VRAM at the cost of
        ~20-30% extra compute.

        Args:
            enable: Whether to enable (True) or disable (False) checkpointing.
        """
        self._grad_checkpointing = enable
        # Activar en los bloques más pesados: 2, 3, 4 (no en el primero)
        for name, module in self.features.named_children():
            if name in ("denseblock2", "denseblock3", "denseblock4"):
                for layer in module.values():
                    if isinstance(layer, _DenseLayer3D):
                        layer.memory_efficient = enable
        log.info(
            "[DenseNet3D] Gradient checkpointing %s en denseblock2..4",
            "activado" if enable else "desactivado",
        )

    # ── Inicialización de pesos ────────────────────────────────────────────

    def _initialize_weights(self) -> None:
        """Initialize weights following standard conventions for deep residual-like networks.

        - Conv3d: Kaiming normal (fan_out, relu).
        - BatchNorm3d: weight=1, bias=0 (identity transform).
        - Linear: Normal(mean=0, std=0.01), bias=0.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # ── Forward pass ───────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: 3D volume → class logits.

        Args:
            x: Input tensor [B, 1, D, H, W] (single-channel CT volume).

        Returns:
            Logits tensor [B, num_classes].
        """
        # Extracción de features: stem + bloques densos + transiciones + BN final
        features = self.features(x)

        # ReLU tras la BN final (convención DenseNet)
        out = F.relu(features, inplace=True)

        # Pooling global → vector plano
        out = self.global_pool(out)  # [B, C, 1, 1, 1]
        out = torch.flatten(out, 1)  # [B, C]

        # Clasificador con dropout
        out = self.fc_drop(out)
        out = self.classifier(out)  # [B, num_classes]

        return out

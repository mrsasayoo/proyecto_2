"""
DenseNet-3D custom del experto LUNA16 + LIDC-IDRI.

Arquitectura reconstruida en detalle desde el state_dict de
`models/LUNA-LIDCIDRI_best.pt.zip` (no hay codigo fuente disponible; la
reconstruccion es determinista sobre los tensores guardados).

Parametros inferidos del checkpoint:
  - num_init_features=64 (stem.conv0 → out_channels=64)
  - kernel_stem=7 (stem.conv0.weight shape [64,1,7,7,7])
  - growth_rate=32 (conv2 output ch en todas las DenseLayer)
  - bn_size=4 (conv1 expande a 128 = bn_size*growth_rate)
  - compression=0.5 (transitions halve channels)
  - block_config=(4, 8, 16, 12) (transitions in: 192, 352, 688; final_bn: 728)
  - dropout_fc=0.4 (config del checkpoint, clasificador con Dropout previo al Linear)
  - num_classes=2

Keys esperadas del state_dict (verificado en checkpoint):
  stem.{conv0, norm0.*}
  dense_blocks.N.layers.M.{bn1, conv1, bn2, conv2}
  transitions.N.{bn, conv}
  final_bn.*
  classifier.1.{weight, bias}        # classifier[0] = Dropout (sin params)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm3d(num_input_features)
        self.conv1 = nn.Conv3d(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.conv2 = nn.Conv3d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                _DenseLayer(
                    num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm3d(num_input_features)
        self.conv = nn.Conv3d(
            num_input_features, num_output_features, kernel_size=1, stride=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(F.relu(self.bn(x), inplace=True))
        return F.avg_pool3d(x, kernel_size=2, stride=2)


class _Stem(nn.Module):
    def __init__(self, in_channels: int = 1, num_init_features: int = 64) -> None:
        super().__init__()
        self.conv0 = nn.Conv3d(
            in_channels,
            num_init_features,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm0 = nn.BatchNorm3d(num_init_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = F.relu(self.norm0(x), inplace=True)
        return F.max_pool3d(x, kernel_size=3, stride=2, padding=1)


class DenseNet3DLUNA(nn.Module):
    """
    DenseNet-3D custom del checkpoint LUNA-LIDCIDRI_best.pt.zip.

    Entrada esperada: (B, 1, 64, 64, 64).
    Salida: logits (B, 2) para clasificacion binaria [no_nodule, nodule].
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_init_features: int = 64,
        growth_rate: int = 32,
        bn_size: int = 4,
        block_config: tuple[int, int, int, int] = (4, 8, 16, 12),
        compression: float = 0.5,
        dropout_fc: float = 0.4,
    ) -> None:
        super().__init__()
        self.stem = _Stem(in_channels=1, num_init_features=num_init_features)

        num_features = num_init_features
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
            )
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                out_features = int(num_features * compression)
                self.transitions.append(
                    _Transition(num_input_features=num_features, num_output_features=out_features)
                )
                num_features = out_features

        self.final_bn = nn.BatchNorm3d(num_features)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_fc),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        x = F.relu(self.final_bn(x), inplace=True)
        x = F.adaptive_avg_pool3d(x, output_size=1)
        x = torch.flatten(x, 1)
        return self.classifier(x)

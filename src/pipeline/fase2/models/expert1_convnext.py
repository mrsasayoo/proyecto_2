"""
Expert 1 — Hybrid-Deep-Vision para clasificación multilabel de ChestXray14.

Arquitectura custom entrenada desde cero (sin pretrained, sin timm).

Entrada:  [B, 1, 256, 256] — escala de grises, float32.

Fase 1 — Backbone Dense-Inception (5 bloques):
    Cada bloque: 4 ramas Inception paralelas → concat → bottleneck Conv 1×1
    → MaxPool 2×2 stride=2. Todas las Conv van seguidas de BatchNorm + ReLU.

    Bloque 1: [B,1,256,256]    → [B,64,128,128]
    Bloque 2: [B,64,128,128]   → [B,128,64,64]
    Bloque 3: [B,128,64,64]    → [B,256,32,32]
    Bloque 4: [B,256,32,32]    → [B,512,16,16]
    Bloque 5: [B,512,16,16]    → [B,1024,8,8]

Fase 2 — Bottleneck de transición:
    Tres Conv 1×1 consecutivas con BN+ReLU:
    1024 → 512 → 256 → 128.  Salida: [B,128,8,8]

Fase 3 — 3 bloques ResNet:
    Cada bloque: Conv3×3 → BN → ReLU → Conv3×3 → BN → residual add → ReLU.
    Entrada y salida: [B,128,8,8].

Fase 4 — Cabezal de clasificación:
    GAP + GMP → concat [B,256] → Linear(256,128) → ReLU → Dropout(0.4)
    → Linear(128,14) → Sigmoid.

Salida: [B, 14] — probabilidades para 14 patologías (multilabel).

Autor: Pipeline Expert1 — Fase 2
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fase2.expert1_config import (
    EXPERT1_DROPOUT_FC,
    EXPERT1_NUM_CLASSES,
)


class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU block.

    Args:
        in_ch: input channels.
        out_ch: output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        padding: convolution padding.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InceptionBlock(nn.Module):
    """Inception block with 4 parallel branches, bottleneck, and downsampling.

    Branches:
        1. Conv 1×1
        2. Conv 3×3 (padding=same)
        3. Conv 5×5 (padding=same)
        4. MaxPool 3×3 stride=1 padding=1 → Conv 1×1

    After concat: bottleneck Conv 1×1 → MaxPool 2×2 stride=2.

    Args:
        in_ch: input channels.
        br1: branch 1 output channels (Conv 1×1).
        br2: branch 2 output channels (Conv 3×3).
        br3: branch 3 output channels (Conv 5×5).
        br4: branch 4 output channels (MaxPool → Conv 1×1).
        bottleneck_ch: bottleneck output channels after concat.
    """

    def __init__(
        self,
        in_ch: int,
        br1: int,
        br2: int,
        br3: int,
        br4: int,
        bottleneck_ch: int,
    ) -> None:
        super().__init__()
        self.branch1 = ConvBnRelu(in_ch, br1, kernel_size=1)
        self.branch2 = ConvBnRelu(in_ch, br2, kernel_size=3, padding=1)
        self.branch3 = ConvBnRelu(in_ch, br3, kernel_size=5, padding=2)
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = ConvBnRelu(in_ch, br4, kernel_size=1)
        self.bottleneck = ConvBnRelu(
            br1 + br2 + br3 + br4, bottleneck_ch, kernel_size=1
        )
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4_conv(self.branch4_pool(x))
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.bottleneck(out)
        return self.downsample(out)


class ResBlock(nn.Module):
    """Basic residual block: Conv3×3 → BN → ReLU → Conv3×3 → BN → add → ReLU.

    Input and output have the same shape [B, channels, H, W].

    Args:
        channels: number of input/output channels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class HybridDeepVision(nn.Module):
    """Hybrid-Deep-Vision: custom architecture for ChestXray14 multilabel classification.

    Trained from scratch on grayscale chest X-rays. No pretrained weights.

    Architecture phases:
        1. Dense-Inception backbone (5 blocks): [B,1,256,256] → [B,1024,8,8]
        2. Transition bottleneck (3× Conv 1×1): [B,1024,8,8] → [B,128,8,8]
        3. ResNet blocks (3×): [B,128,8,8] → [B,128,8,8]
        4. Classification head: GAP+GMP → [B,256] → FC → [B,14] (sigmoid)

    Args:
        num_classes: number of output classes. Default: 14.
        dropout_fc: dropout probability in classifier head. Default: from config.
    """

    def __init__(
        self,
        num_classes: int = EXPERT1_NUM_CLASSES,
        dropout_fc: float = EXPERT1_DROPOUT_FC,
    ) -> None:
        super().__init__()

        # ── Phase 1: Dense-Inception backbone ──────────────────────
        self.inception_blocks = nn.Sequential(
            InceptionBlock(in_ch=1, br1=128, br2=64, br3=32, br4=32, bottleneck_ch=64),
            InceptionBlock(
                in_ch=64, br1=128, br2=64, br3=32, br4=32, bottleneck_ch=128
            ),
            InceptionBlock(
                in_ch=128, br1=64, br2=128, br3=32, br4=32, bottleneck_ch=256
            ),
            InceptionBlock(
                in_ch=256, br1=64, br2=128, br3=32, br4=32, bottleneck_ch=512
            ),
            InceptionBlock(
                in_ch=512, br1=32, br2=64, br3=128, br4=32, bottleneck_ch=1024
            ),
        )

        # ── Phase 2: Transition bottleneck ─────────────────────────
        self.transition = nn.Sequential(
            ConvBnRelu(1024, 512, kernel_size=1),
            ConvBnRelu(512, 256, kernel_size=1),
            ConvBnRelu(256, 128, kernel_size=1),
        )

        # ── Phase 3: ResNet blocks ─────────────────────────────────
        self.res_blocks = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
        )

        # ── Phase 4: Classification head ───────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )

        # ── Weight initialization (Kaiming for training from scratch) ──
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming Normal for Conv/Linear, ones/zeros for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: tensor [B, 1, 256, 256] — grayscale chest X-ray, float32.

        Returns:
            Probabilities [B, num_classes] after sigmoid (multilabel).
        """
        # Phase 1: Inception backbone
        x = self.inception_blocks(x)  # [B, 1024, 8, 8]

        # Phase 2: Transition bottleneck
        x = self.transition(x)  # [B, 128, 8, 8]

        # Phase 3: ResNet blocks
        x = self.res_blocks(x)  # [B, 128, 8, 8]

        # Phase 4: Classification head
        avg = self.gap(x).flatten(1)  # [B, 128]
        mx = self.gmp(x).flatten(1)  # [B, 128]
        x = torch.cat([avg, mx], dim=1)  # [B, 256]
        return self.classifier(x)  # [B, num_classes]

    def count_parameters(self) -> int:
        """Number of trainable parameters (requires_grad=True)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Total number of parameters (trainable + frozen)."""
        return sum(p.numel() for p in self.parameters())


# ── Alias for backward compatibility with existing imports ──────────────
Expert1ConvNeXtTiny = HybridDeepVision
Expert1ConvNeXt = HybridDeepVision


def _test_model() -> None:
    """Quick verification: instantiate, forward pass, check output shape."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Expert1/HybridDeepVision] Device: {device}")

    model = HybridDeepVision().to(device)

    x = torch.randn(2, 1, 256, 256, device=device)
    model.eval()
    with torch.no_grad():
        out = model(x)

    total = model.count_all_parameters()
    trainable = model.count_parameters()
    print(f"[Expert1/HybridDeepVision] Input shape:  {list(x.shape)}")
    print(f"[Expert1/HybridDeepVision] Output shape: {list(out.shape)}")
    print(
        f"[Expert1/HybridDeepVision] Params total: {total:,}  trainable: {trainable:,}"
    )
    print(
        f"[Expert1/HybridDeepVision] Output range: [{out.min().item():.4f}, {out.max().item():.4f}]"
    )

    assert out.shape == (2, 14), f"Wrong shape: {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), (
        "Output not in [0, 1] — sigmoid missing?"
    )
    print("[Expert1/HybridDeepVision] Verification completed successfully ✓")


if __name__ == "__main__":
    _test_model()

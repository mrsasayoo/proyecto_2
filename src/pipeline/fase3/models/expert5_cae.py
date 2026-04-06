"""
ConvAutoEncoder — Expert 5: CAE 2D para detección OOD multimodal.

Arquitectura encoder-latente-decoder puramente 2D.
Input/Output: [B, 3, 224, 224]
Latent space: [B, 512]

El CAE aprende la distribución conjunta de las 5 modalidades médicas
(Chest, ISIC, OA, LUNA16 slices, Páncreas slices). Imágenes que no
pertenecen a esta distribución producen alto error de reconstrucción
→ detección OOD.

Entrenado desde cero (weights=None). NO usa Conv3d.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoEncoder(nn.Module):
    """
    CAE 2D para detección OOD multimodal.

    Encoder:
        Conv2d(3, 64, 3, stride=2)   -> [B, 64, 112, 112]
        Conv2d(64, 128, 3, stride=2)  -> [B, 128, 56, 56]
        Conv2d(128, 256, 3, stride=2) -> [B, 256, 28, 28]
        Flatten -> Linear(200704, 512) -> [B, 512]

    Decoder:
        Linear(512, 200704) -> Unflatten -> [B, 256, 28, 28]
        ConvTranspose2d(256, 128, ...) -> [B, 128, 56, 56]
        ConvTranspose2d(128, 64, ...)  -> [B, 64, 112, 112]
        ConvTranspose2d(64, 3, ...)    -> [B, 3, 224, 224]
        Sigmoid

    Args:
        in_channels: canales de entrada (default=3)
        latent_dim: dimensión del espacio latente (default=512)
        img_size: tamaño espacial de entrada (default=224, debe ser divisible por 8)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 512,
        img_size: int = 224,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size

        # Tamaño del feature map después del encoder conv
        # 224 -> 112 -> 56 -> 28
        self._feat_size = img_size // 8  # 28 para img_size=224
        self._feat_flat = 256 * self._feat_size * self._feat_size  # 200704

        # ── ENCODER ────────────────────────────────────────────────────
        self.encoder_conv = nn.Sequential(
            # [B, 3, 224, 224] -> [B, 64, 112, 112]
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # [B, 64, 112, 112] -> [B, 128, 56, 56]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # [B, 128, 56, 56] -> [B, 256, 28, 28]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.encoder_fc = nn.Linear(self._feat_flat, latent_dim)

        # ── DECODER ────────────────────────────────────────────────────
        self.decoder_fc = nn.Linear(latent_dim, self._feat_flat)
        self.decoder_conv = nn.Sequential(
            # [B, 256, 28, 28] -> [B, 128, 56, 56]
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # [B, 128, 56, 56] -> [B, 64, 112, 112]
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # [B, 64, 112, 112] -> [B, 3, 224, 224]
            nn.ConvTranspose2d(
                64, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder: [B, C, H, W] -> [B, latent_dim]"""
        x = self.encoder_conv(x)
        x = torch.flatten(x, 1)
        return self.encoder_fc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder: [B, latent_dim] -> [B, C, H, W]"""
        x = self.decoder_fc(z)
        x = x.view(-1, 256, self._feat_size, self._feat_size)
        return self.decoder_conv(x)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass completo.

        Args:
            x: tensor de entrada [B, C, H, W]

        Returns:
            (recon, z) — reconstrucción [B, C, H, W] y latente [B, latent_dim]
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula MSE de reconstrucción por sample (para OOD detection).

        Args:
            x: tensor de entrada [B, C, H, W]

        Returns:
            tensor [B] con el MSE por sample
        """
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2, 3])

    def count_parameters(self) -> int:
        """Retorna el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

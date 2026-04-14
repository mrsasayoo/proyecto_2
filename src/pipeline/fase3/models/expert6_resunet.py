"""
Res-U-Net Autoencoder — Expert 6 (posición 5 en ModuleList del MoE).

Arquitectura encoder-decoder con skip connections tipo U-Net y bloques
residuales pre-activación (He et al., 2016 — ResNet v2).

Justificación del diseño:
  - Skip connections (U-Net): permiten al decoder recuperar detalles de
    alta frecuencia directamente del encoder, mejorando la fidelidad de
    reconstrucción respecto al CAE simple (Expert 5) que pierde
    información espacial al aplanar a un vector latente.
  - Bloques residuales pre-activación: BN → GELU → Conv garantiza
    gradientes estables durante el entrenamiento profundo, evitando
    degradación en redes con muchas capas.
  - Bottleneck con AdaptiveAvgPool2d: produce un vector latente compacto
    [B, 512] para detección OOD vía error de reconstrucción, manteniendo
    compatibilidad con el pipeline existente.

Input:  [B, 3, 224, 224]  — imágenes médicas RGB (o grayscale ×3)
Output: ([B, 3, 224, 224] reconstrucción, [B, 512] vector latente)

Mapa de dimensiones (base_ch=64):
  stem   → [B,  64, 224, 224]
  enc1   → [B, 128, 112, 112]
  enc2   → [B, 256,  56,  56]
  enc3   → [B, 512,  28,  28]
  enc4   → [B, 512,  14,  14]
  btl_dn → [B, 512,   7,   7]
  bottleneck feat=[B,512,7,7], z=[B,512]
  dec4   → [B, 256,  14,  14]
  dec3   → [B, 128,  28,  28]
  dec2   → [B,  64,  56,  56]
  dec1   → [B,  32, 112, 112]
  head   → [B,   3, 224, 224]

Nota sobre skip_ch en el decoder:
  Los valores de skip_ch se corresponden con los canales reales de salida
  de cada etapa del encoder (s1=128, s2=256, s3=512, s4=512), no con los
  canales de entrada del decoder. Esto es necesario para que la
  concatenación [x, skip] sea dimensionalmente correcta.

Dependencias: solo torch, torch.nn, torch.nn.functional (sin extras).
FP32 estricto — no usar AMP ni autocast.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlockPreAct(nn.Module):
    """Pre-activation residual block (He et al., 2016 ResNet v2).

    Secuencia: BN → GELU → Conv3×3 → BN → GELU → Dropout → Conv3×3.
    La entrada se suma a la salida del bloque (conexión residual).

    Args:
        channels: número de canales de entrada y salida (iguales).
        dropout: probabilidad de Dropout2d entre las dos convoluciones.
    """

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """Bloque encoder: downsample ×2 seguido de bloques residuales.

    Downsample: Conv3×3 stride=2 → BN → GELU.
    Luego ``n_res_blocks`` bloques :class:`ResBlockPreAct`.

    Args:
        in_ch: canales de entrada.
        out_ch: canales de salida (tras downsample).
        n_res_blocks: cantidad de bloques residuales.
        dropout: dropout para cada ResBlockPreAct.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_res_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlockPreAct(out_ch, dropout) for _ in range(n_res_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.res_blocks(x)
        return x


class DecoderBlock(nn.Module):
    """Bloque decoder: upsample + fusión con skip connection + bloques residuales.

    Upsample vía ``F.interpolate`` bilinear al tamaño del skip, luego
    concatenación en canales, fusión 1×1, y ``n_res_blocks`` residuales.

    Args:
        in_ch: canales del feature map que sube del nivel inferior.
        skip_ch: canales del skip connection del encoder.
        out_ch: canales de salida tras fusión.
        n_res_blocks: cantidad de bloques residuales.
        dropout: dropout para cada ResBlockPreAct.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        n_res_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlockPreAct(out_ch, dropout) for _ in range(n_res_blocks)]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.res_blocks(x)


class Bottleneck(nn.Module):
    """Bottleneck: bloques residuales profundos + proyección a vector latente.

    Produce tanto el feature map espacial (para el decoder) como un
    vector latente compacto [B, channels] vía AdaptiveAvgPool2d → Flatten.

    Args:
        channels: canales del feature map (y dimensión del vector latente).
        n_res_blocks: cantidad de bloques residuales en el bottleneck.
        dropout: dropout para cada ResBlockPreAct.
    """

    def __init__(
        self,
        channels: int = 512,
        n_res_blocks: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResBlockPreAct(channels, dropout) for _ in range(n_res_blocks)]
        )
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.blocks(x)
        z = self.to_latent(feat)
        return feat, z


class FiLMGenerator(nn.Module):
    """Genera parámetros gamma y beta para FiLM conditioning.

    Mapea un domain ID (int) a vectores de escala (gamma) y sesgo (beta)
    para modular los feature maps del bottleneck.

    n_domains=6: 0=CXR, 1=ISIC, 2=Panorama, 3=LUNA16, 4=OA, 5=Unknown.

    Inicialización: el último Linear se inicializa a cero → gamma=1, beta=0
    al inicio del entrenamiento (identidad). El modelo aprende desviaciones
    desde la identidad, lo que garantiza estabilidad en las primeras épocas.

    Args:
        n_domains: número total de dominios (incluyendo "Unknown").
        embed_dim: dimensión del embedding de dominio.
        feature_channels: número de canales del feature map a modular.
    """

    def __init__(
        self,
        n_domains: int = 6,
        embed_dim: int = 64,
        feature_channels: int = 512,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.domain_embed = nn.Embedding(n_domains, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_channels * 2),  # gamma y beta concatenados
        )
        # Inicialización crítica: gamma_raw=0, beta=0 → FiLM actúa como
        # identidad al inicio. El modelo aprende desviaciones desde 1 y 0.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, domain_ids: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Genera gamma y beta para los domain_ids dados.

        Args:
            domain_ids: [B] tensor int64 con IDs de dominio (0–5).

        Returns:
            Tupla (gamma [B, C, 1, 1], beta [B, C, 1, 1]) para broadcasting
            sobre feature maps [B, C, H, W].
        """
        e = self.domain_embed(domain_ids)  # [B, embed_dim]
        params = self.mlp(e)  # [B, 2*C]
        gamma_raw, beta = params.chunk(2, dim=-1)  # cada uno [B, C]
        # Formulación residual: base gamma=1 (identidad), aprende delta
        gamma = (1.0 + gamma_raw).view(-1, self.feature_channels, 1, 1)
        beta = beta.view(-1, self.feature_channels, 1, 1)
        return gamma, beta


class FiLMLayer(nn.Module):
    """Aplica FiLM modulation: x_cond = gamma * x + beta.

    Operación sin parámetros entrenables propios — los parámetros
    de modulación vienen de FiLMGenerator.
    """

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: feature map [B, C, H, W].
            gamma: escala [B, C, 1, 1].
            beta: sesgo [B, C, 1, 1].

        Returns:
            Feature map modulado [B, C, H, W].
        """
        return gamma * x + beta


class ConditionedBottleneck(nn.Module):
    """Bottleneck con FiLM domain conditioning.

    Flujo:
      x [B,512,7,7] → ResBlocks×4 → FiLM(domain_ids) → feat_cond [B,512,7,7]
                                                        → GAP → z [B,512]

    El FiLM se aplica DESPUÉS de los bloques residuales, modulando el
    espacio latente espacial antes de producir el vector z y antes de
    pasarlo al decoder. Esto permite al decoder "saber" qué dominio
    debe reconstruir.

    Args:
        channels: canales del feature map (= dimensión del vector latente).
        n_res_blocks: cantidad de ResBlockPreAct en el bottleneck.
        dropout: dropout para cada ResBlockPreAct.
        n_domains: número de dominios para el FiLMGenerator.
        embed_dim: dimensión del embedding de dominio en FiLMGenerator.
    """

    def __init__(
        self,
        channels: int = 512,
        n_res_blocks: int = 4,
        dropout: float = 0.15,
        n_domains: int = 6,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResBlockPreAct(channels, dropout) for _ in range(n_res_blocks)]
        )
        self.film_gen = FiLMGenerator(
            n_domains=n_domains,
            embed_dim=embed_dim,
            feature_channels=channels,
        )
        self.film = FiLMLayer()
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(
        self,
        x: torch.Tensor,
        domain_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: feature map de entrada [B, channels, H, W].
            domain_ids: [B] tensor int64 con IDs de dominio.

        Returns:
            Tupla (feat_cond [B, channels, H, W], z [B, channels]).
        """
        feat = self.blocks(x)
        gamma, beta = self.film_gen(domain_ids)
        feat = self.film(feat, gamma, beta)
        z = self.to_latent(feat)
        return feat, z


class ConditionedResUNetAE(nn.Module):
    """Res-U-Net Autoencoder con FiLM domain conditioning para detección OOD.

    Encoder simétrico con 4 etapas de downsample, bottleneck condicionado
    por dominio (FiLM), extracción de vector latente, y decoder con skip
    connections.

    El FiLM conditioning permite al modelo aprender reconstrucciones
    específicas por dominio médico, eliminando falsos positivos OOD
    causados por confusión entre modalidades.

    Input:  [B, 3, 224, 224]
    Output: ([B, 3, 224, 224] reconstrucción, [B, 512] vector latente)

    Canales (base_ch=64):
      stem   → [B,  64, 224, 224]
      enc1   → [B, 128, 112, 112]
      enc2   → [B, 256,  56,  56]
      enc3   → [B, 512,  28,  28]
      enc4   → [B, 512,  14,  14]
      btl_dn → [B, 512,   7,   7]
      bottleneck (feat=[B,512,7,7], z=[B,512]) + FiLM conditioning
      dec4(b, s4)  → [B, 256,  14,  14]
      dec3(d4, s3) → [B, 128,  28,  28]
      dec2(d3, s2) → [B,  64,  56,  56]
      dec1(d2, s1) → [B,  32, 112, 112]
      head         → [B,   3, 224, 224]

    Args:
        in_ch: canales de entrada (default=3 para RGB).
        base_ch: canales base; las etapas escalan como base×{1,2,4,8}.
        dropout: dropout en bloques residuales del encoder/decoder.
        n_domains: número total de dominios para FiLM conditioning
            (0=CXR, 1=ISIC, 2=Panorama, 3=LUNA16, 4=OA, 5=Unknown).
        embed_dim: dimensión del embedding de dominio en FiLMGenerator.
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 64,
        dropout: float = 0.1,
        n_domains: int = 6,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()

        # ── Stem ────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
            ResBlockPreAct(base_ch, dropout),
        )

        # ── Encoder ─────────────────────────────────────────────────
        self.enc1 = EncoderBlock(base_ch, base_ch * 2, n_res_blocks=2, dropout=dropout)
        self.enc2 = EncoderBlock(
            base_ch * 2, base_ch * 4, n_res_blocks=2, dropout=dropout
        )
        self.enc3 = EncoderBlock(
            base_ch * 4, base_ch * 8, n_res_blocks=2, dropout=dropout
        )
        self.enc4 = EncoderBlock(
            base_ch * 8, base_ch * 8, n_res_blocks=2, dropout=dropout
        )

        # ── Bottleneck ──────────────────────────────────────────────
        self.bottleneck_down = nn.Sequential(
            nn.Conv2d(
                base_ch * 8,
                base_ch * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_ch * 8),
            nn.GELU(),
        )
        self.bottleneck = ConditionedBottleneck(
            channels=base_ch * 8,
            n_res_blocks=4,
            dropout=0.15,
            n_domains=n_domains,
            embed_dim=embed_dim,
        )

        # ── Decoder ─────────────────────────────────────────────────
        # skip_ch corregidos para coincidir con los canales reales del
        # encoder: s4=base_ch*8, s3=base_ch*8, s2=base_ch*4, s1=base_ch*2.
        self.dec4 = DecoderBlock(
            base_ch * 8,
            base_ch * 8,
            base_ch * 4,
            n_res_blocks=2,
            dropout=dropout,
        )
        self.dec3 = DecoderBlock(
            base_ch * 4,
            base_ch * 8,
            base_ch * 2,
            n_res_blocks=2,
            dropout=dropout,
        )
        self.dec2 = DecoderBlock(
            base_ch * 2,
            base_ch * 4,
            base_ch,
            n_res_blocks=2,
            dropout=dropout,
        )
        self.dec1 = DecoderBlock(
            base_ch,
            base_ch * 2,
            base_ch // 2,
            n_res_blocks=2,
            dropout=dropout,
        )

        # ── Head ────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
            ResBlockPreAct(base_ch // 2, dropout),
            nn.Conv2d(base_ch // 2, in_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        domain_ids: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass completo con FiLM domain conditioning.

        Args:
            x: tensor de entrada [B, C, H, W].
            domain_ids: [B] tensor int64 con IDs de dominio (0–5).
                Si None, se usa dominio 5 ("Unknown") para todo el batch.

        Returns:
            Tupla (x_hat, z) con la reconstrucción [B, C, H, W] y el
            vector latente [B, 512].
        """
        if domain_ids is None:
            domain_ids = torch.full(
                (x.shape[0],),
                5,
                dtype=torch.long,
                device=x.device,
            )

        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        b = self.bottleneck_down(s4)
        b, z = self.bottleneck(b, domain_ids)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        x_hat = self.head(d1)
        return x_hat, z

    @torch.no_grad()
    def reconstruction_error(
        self,
        x: torch.Tensor,
        domain_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Calcula error de reconstrucción combinado (MSE + 0.1*L1) por sample.

        Usado en inferencia para detección OOD: imágenes fuera de la
        distribución de entrenamiento producen errores más altos.

        Args:
            x: tensor de entrada [B, C, H, W].
            domain_ids: [B] tensor int64 con IDs de dominio (0–5).
                Si None, se usa dominio 5 ("Unknown") para todo el batch.

        Returns:
            Tensor [B] con MSE + 0.1*L1 por sample (solo el error,
            no una tupla — los callers esperan un tensor plano).
        """
        x_hat, z = self.forward(x, domain_ids)
        mse = ((x - x_hat) ** 2).mean(dim=[1, 2, 3])
        l1 = (x - x_hat).abs().mean(dim=[1, 2, 3])
        return mse + 0.1 * l1

    def count_parameters(self) -> int:
        """Retorna el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ReconstructionLoss(nn.Module):
    """Loss combinada MSE + λ·L1 para entrenamiento del autoencoder.

    Promueve reconstrucciones fieles (MSE) y nítidas (L1 penaliza
    reconstrucciones borrosas promedio).

    Args:
        l1_weight: peso del término L1 en la loss total.
    """

    def __init__(self, l1_weight: float = 0.1) -> None:
        super().__init__()
        self.l1_weight = l1_weight

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Calcula loss = MSE(x_hat, x) + l1_weight * L1(x_hat, x).

        Args:
            x_hat: reconstrucción [B, C, H, W].
            x: entrada original [B, C, H, W].

        Returns:
            Escalar con la loss combinada.
        """
        return F.mse_loss(x_hat, x) + self.l1_weight * F.l1_loss(x_hat, x)


# Backward-compat alias — code that imports ResUNetAutoencoder still works.
ResUNetAutoencoder = ConditionedResUNetAE


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ConditionedResUNetAE(
        in_ch=3,
        base_ch=64,
        dropout=0.1,
        n_domains=6,
        embed_dim=64,
    ).to(device)

    n_params = model.count_parameters()
    print(f"Parámetros entrenables: {n_params:,} ({n_params / 1e6:.2f}M)")

    dummy = torch.randn(2, 3, 224, 224, device=device)
    model.eval()

    # --- Test 1: forward SIN domain_ids (debe usar Unknown=5 por defecto) ---
    print("\n--- Test 1: forward sin domain_ids (default Unknown=5) ---")
    with torch.no_grad():
        x_hat, z = model(dummy)
    print(f"Input shape:  {list(dummy.shape)}")
    print(f"Output shape: {list(x_hat.shape)}")
    print(f"Latent shape: {list(z.shape)}")

    # --- Test 2: forward CON domain_ids explícitos ---
    print("\n--- Test 2: forward con domain_ids=[0, 1] ---")
    domain_ids = torch.tensor([0, 1], dtype=torch.long, device=device)
    with torch.no_grad():
        x_hat2, z2 = model(dummy, domain_ids)
    print(f"Output shape: {list(x_hat2.shape)}")
    print(f"Latent shape: {list(z2.shape)}")

    # --- Test 3: reconstruction_error devuelve tensor, NO tupla ---
    print("\n--- Test 3: reconstruction_error devuelve Tensor[B] ---")
    error = model.reconstruction_error(dummy)
    assert isinstance(error, torch.Tensor), (
        f"reconstruction_error debe devolver Tensor, got {type(error)}"
    )
    assert error.ndim == 1 and error.shape[0] == 2, (
        f"Esperado shape [2], got {list(error.shape)}"
    )
    print(f"Recon error shape: {list(error.shape)} ✓ (tensor, no tupla)")

    # --- Test 4: reconstruction_error con domain_ids ---
    print("\n--- Test 4: reconstruction_error con domain_ids ---")
    error2 = model.reconstruction_error(dummy, domain_ids)
    print(f"Recon error shape: {list(error2.shape)} ✓")

    # --- Test 5: ReconstructionLoss ---
    print("\n--- Test 5: ReconstructionLoss ---")
    loss_fn = ReconstructionLoss(l1_weight=0.1)
    loss_val = loss_fn(x_hat, dummy)
    print(f"ReconstructionLoss: {loss_val.item():.6f}")

    # --- Test 6: alias backward-compat ---
    print("\n--- Test 6: alias ResUNetAutoencoder ---")
    assert ResUNetAutoencoder is ConditionedResUNetAE
    print("ResUNetAutoencoder is ConditionedResUNetAE ✓")

    # Validar shapes
    assert list(x_hat.shape) == [2, 3, 224, 224], (
        f"Output shape incorrecto: {list(x_hat.shape)}"
    )
    assert list(z.shape) == [2, 512], f"Latent shape incorrecto: {list(z.shape)}"
    print(f"\n✓ Shapes verificados: output [2, 3, 224, 224], latent [2, 512].")
    print(f"✓ Parámetros: {n_params:,} ({n_params / 1e6:.2f}M)")
    print("✓ FiLM domain conditioning activo.")

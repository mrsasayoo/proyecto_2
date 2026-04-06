"""
Expert 4 — Swin3D-Tiny para clasificación binaria de volúmenes CT abdominales (PDAC).

Arquitectura:
    Backbone: torchvision.models.video.swin_transformer.SwinTransformer3d
    configurado con los parámetros del model_card:
        embed_dim=48, depths=[2,2,6,2], num_heads=[3,6,12,24],
        window_size=[4,4,4], patch_size=[4,4,4], mlp_ratio=4.0
    Entrada:  [B, 1, 64, 64, 64] — volumen CT abdominal monocanal
    Salida:   [B, 2] — logits para clasificación binaria (PDAC− / PDAC+)

Adaptaciones respecto al SwinTransformer3d de torchvision (Kinetics400):
    1. Patch embedding: in_channels 3 → 1 (CT monocanal).
       patch_size y stride: (4,4,4) isótropo (vs (2,4,4) del preset swin3d_t).
    2. embed_dim: 48 (vs 96 en swin3d_t) para reducir parámetros.
       Con ~281 muestras, un modelo más pequeño es preferible.
    3. Cabeza clasificadora: 2 clases (vs 400 en Kinetics).
    4. Inicialización desde cero con weights=None.

Parámetros totales: ~6.9M (vs ~28.2M del swin3d_t estándar).
Con ~281 volúmenes: ratio params/datos ≈ 24,600:1 — requiere regularización
agresiva + k-fold CV + early stopping.

Autor: Pipeline Expert4 — Fase 2
"""

import torch
import torch.nn as nn
from torchvision.models.video.swin_transformer import SwinTransformer3d


class ExpertPancreasSwin3D(nn.Module):
    """
    Swin3D-Tiny para clasificación binaria PDAC+/PDAC- en CT abdominal.

    Usa la clase SwinTransformer3d de torchvision con parámetros del model_card:
    - embed_dim=48, depths=[2,2,6,2], num_heads=[3,6,12,24]
    - window_size=[4,4,4], patch_size=[4,4,4]
    - 1 canal de entrada (CT monocanal)
    - 2 clases de salida (PDAC-/PDAC+)

    Entrada:  [B, 1, 64, 64, 64]
    Salida:   [B, 2] logits

    Args:
        in_channels: canales de entrada. Default: 1 (CT monocanal).
        num_classes: clases de salida. Default: 2 (PDAC-/PDAC+).
        embed_dim: dimensión del embedding. Default: 48.
        depths: profundidad de cada stage. Default: [2, 2, 6, 2].
        num_heads: heads de atención por stage. Default: [3, 6, 12, 24].
        window_size: tamaño de ventana para atención local. Default: [4, 4, 4].
        patch_size: tamaño del patch embedding. Default: [4, 4, 4].
        mlp_ratio: ratio de expansión del MLP. Default: 4.0.
        dropout: dropout general. Default: 0.0.
        attention_dropout: dropout en atención. Default: 0.0.
        stochastic_depth_prob: probabilidad de stochastic depth. Default: 0.0.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dim: int = 48,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        window_size: list[int] | None = None,
        patch_size: list[int] | None = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
    ):
        super().__init__()

        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if window_size is None:
            window_size = [4, 4, 4]
        if patch_size is None:
            patch_size = [4, 4, 4]

        # ── Instanciar SwinTransformer3d de torchvision ─────────────────
        # Se crea con 3 canales (default de torchvision) y luego se reemplaza
        # la capa de patch embedding para aceptar 'in_channels' canales.
        self.backbone = SwinTransformer3d(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            num_classes=num_classes,
        )

        # ── Adaptar patch_embed: 3 canales → in_channels (1 para CT) ───
        old_proj = self.backbone.patch_embed.proj
        new_proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        # Inicialización Kaiming para Conv3d con entrada monocanal
        nn.init.kaiming_normal_(new_proj.weight, mode="fan_out", nonlinearity="relu")
        if new_proj.bias is not None:
            nn.init.zeros_(new_proj.bias)
        self.backbone.patch_embed.proj = new_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del Swin3D-Tiny.

        Args:
            x: tensor [B, 1, 64, 64, 64] — volumen CT abdominal monocanal

        Returns:
            logits: tensor [B, 2] — logits crudos (antes de softmax/sigmoid)
        """
        return self.backbone(x)

    def count_parameters(self) -> int:
        """Retorna el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Retorna el número total de parámetros (entrenables + congelados)."""
        return sum(p.numel() for p in self.parameters())


def _test_model():
    """Verificación rápida: instanciar, forward pass, conteo de parámetros."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Expert4/Swin3D-Tiny] Dispositivo: {device}")

    model = ExpertPancreasSwin3D(
        in_channels=1,
        num_classes=2,
    ).to(device)

    # Forward pass con tensor dummy
    dummy = torch.randn(1, 1, 64, 64, 64, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)

    n_params = model.count_parameters()
    print(f"[Expert4/Swin3D-Tiny] Input shape:  {list(dummy.shape)}")
    print(f"[Expert4/Swin3D-Tiny] Output shape: {list(out.shape)}")
    print(f"[Expert4/Swin3D-Tiny] Parámetros entrenables: {n_params:,}")
    print(f"[Expert4/Swin3D-Tiny] Output values: {out}")

    # Validaciones
    assert out.shape == (1, 2), (
        f"Shape de salida incorrecto: {out.shape}, esperado (1, 2)"
    )
    assert n_params > 0, "Modelo sin parámetros entrenables"
    print(f"[Expert4/Swin3D-Tiny] Verificación completada exitosamente")
    return model


if __name__ == "__main__":
    _test_model()

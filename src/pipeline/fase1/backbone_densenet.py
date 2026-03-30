"""
backbone_densenet.py — DenseNet-121 Custom desde Cero
=====================================================

Implementación pura con torch.nn, sin pesos preentrenados, sin timm,
sin torchvision.models. Cumple el requisito del proyecto de construir
toda la arquitectura desde cero.

Arquitectura DenseNet-121 (Huang et al., CVPR 2017):
  - 4 DenseBlocks con config [6, 12, 24, 16]
  - Growth rate: 32
  - Transition layers: Conv 1×1 + AvgPool 2×2
  - Primer conv: 7×7, stride 2, padding 3 → 64 canales
  - Global Average Pooling → embedding de dimensión configurable

Interface pública:
  build_densenet(in_channels, embed_dim, growth_rate, block_config) → nn.Module
  forward(x) → [B, embed_dim]

Referencia:
    Huang et al., "Densely Connected Convolutional Networks",
    CVPR 2017. https://arxiv.org/abs/1608.06993
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("fase1")


# ── Capa densa individual ───────────────────────────────────────────────────
class _DenseLayer(nn.Module):
    """
    Una capa densa del DenseBlock: BN → ReLU → Conv1×1 → BN → ReLU → Conv3×3.

    Usa el patrón bottleneck (BN-ReLU-Conv1×1-BN-ReLU-Conv3×3) que reduce
    el número de feature maps intermedios a 4 × growth_rate antes del Conv3×3,
    mejorando la eficiencia computacional sin perder capacidad representativa.
    """

    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        """
        Args:
            in_channels: número de canales de entrada (acumulados por concatenación)
            growth_rate: número de feature maps que esta capa agrega (k en el paper)
            bn_size: factor bottleneck — canales intermedios = bn_size × growth_rate
        """
        super().__init__()
        # Canales intermedios del bottleneck
        inter_channels = bn_size * growth_rate

        # Bottleneck: reduce dimensionalidad antes del conv 3×3
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)

        # Conv 3×3: extrae features espaciales
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(
            inter_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward con conexión densa: la salida se concatena con la entrada
        en el DenseBlock (no aquí — el bloque maneja la concatenación).

        Args:
            x: tensor [B, C_in, H, W] — feature maps acumulados

        Returns:
            out: tensor [B, growth_rate, H, W] — nuevos feature maps
        """
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out


# ── Bloque denso (DenseBlock) ───────────────────────────────────────────────
class _DenseBlock(nn.Module):
    """
    DenseBlock: secuencia de capas densas con conexiones skip por concatenación.

    Cada capa recibe TODOS los feature maps anteriores (entrada original +
    salidas de capas previas) concatenados en la dimensión de canales.
    Esto permite reutilización de features y flujo directo del gradiente.
    """

    def __init__(self, num_layers: int, in_channels: int, growth_rate: int):
        """
        Args:
            num_layers: número de capas densas en este bloque
            in_channels: canales de entrada al bloque
            growth_rate: feature maps que cada capa agrega (k)
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Cada capa recibe in_channels + i * growth_rate canales
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: concatena la salida de cada capa con todas las anteriores.

        Args:
            x: tensor [B, C_in, H, W]

        Returns:
            features: tensor [B, C_in + num_layers * growth_rate, H, W]
        """
        features = [x]
        for layer in self.layers:
            # Concatenar todos los feature maps previos
            concat_input = torch.cat(features, dim=1)
            new_features = layer(concat_input)
            features.append(new_features)
        return torch.cat(features, dim=1)


# ── Capa de transición (Transition Layer) ───────────────────────────────────
class _TransitionLayer(nn.Module):
    """
    Transition Layer entre DenseBlocks: reduce canales y resolución espacial.

    Compresión: BN → ReLU → Conv 1×1 (reduce canales al 50%) → AvgPool 2×2.
    El factor de compresión θ=0.5 es el valor por defecto en DenseNet-BC
    (Bottleneck-Compression), que reduce el número de feature maps a la mitad
    sin pérdida significativa de información.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: canales de entrada (salida del DenseBlock anterior)
            out_channels: canales de salida (típicamente in_channels // 2)
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """BN → ReLU → Conv 1×1 → AvgPool 2×2."""
        out = self.conv(F.relu(self.bn(x), inplace=True))
        out = self.pool(out)
        return out


# ── DenseNet completa ───────────────────────────────────────────────────────
class DenseNet(nn.Module):
    """
    DenseNet custom construida desde cero (sin pesos preentrenados).

    Arquitectura:
      1. Stem: Conv 7×7 stride 2 → BN → ReLU → MaxPool 3×3 stride 2
      2. 4 DenseBlocks intercalados con 3 Transition Layers
      3. BN final → ReLU → Global Average Pooling
      4. Proyección lineal al embed_dim deseado

    En forward(x) retorna un tensor [B, embed_dim] — el embedding global,
    listo para el pipeline de extracción de embeddings de Fase 1.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 1024,
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        init_features: int = 64,
        compression: float = 0.5,
    ):
        """
        Args:
            in_channels: canales de la imagen de entrada (1=grises, 3=RGB)
            embed_dim: dimensión del vector de embedding de salida
            growth_rate: feature maps nuevos por capa densa (k en el paper)
            block_config: tupla con el número de capas por DenseBlock
                         (6,12,24,16) = DenseNet-121
                         (6,12,32,32) = DenseNet-169
                         (6,12,48,32) = DenseNet-201
            init_features: canales de salida del stem (conv inicial)
            compression: factor de compresión θ en Transition Layers (0.5 = BC)
        """
        super().__init__()

        self.embed_dim = embed_dim

        # ── Stem: conv 7×7 + BN + ReLU + MaxPool ───────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── DenseBlocks + Transition Layers ────────────────────────────────
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        num_features = init_features
        for i, num_layers in enumerate(block_config):
            # DenseBlock: agrega num_layers * growth_rate canales
            block = _DenseBlock(num_layers, num_features, growth_rate)
            self.blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            # Transition Layer después de cada bloque excepto el último
            if i < len(block_config) - 1:
                out_features = int(num_features * compression)
                transition = _TransitionLayer(num_features, out_features)
                self.transitions.append(transition)
                num_features = out_features

        # Número de canales tras el último DenseBlock (antes de la proyección)
        self._final_channels = num_features

        # ── BN final + proyección al embed_dim ─────────────────────────────
        self.final_bn = nn.BatchNorm2d(num_features)
        self.projection = nn.Linear(num_features, embed_dim)

        # ── Inicialización de pesos (Kaiming/He) ──────────────────────────
        self._initialize_weights()

        log.info(
            "[DenseNet] Arquitectura construida desde cero:\n"
            f"    in_channels    : {in_channels}\n"
            f"    embed_dim      : {embed_dim}\n"
            f"    growth_rate    : {growth_rate}\n"
            f"    block_config   : {block_config}\n"
            f"    init_features  : {init_features}\n"
            f"    compression    : {compression}\n"
            f"    canales finales: {num_features}\n"
            f"    parámetros     : {sum(p.numel() for p in self.parameters()):,}"
        )

    def _initialize_weights(self):
        """
        Inicialización de pesos con el método de Kaiming/He.

        - Conv2d: Kaiming normal (fan_out, ReLU) — estándar para redes profundas
        - BatchNorm2d: weight=1, bias=0 — identidad inicial
        - Linear: Kaiming uniform — estándar para capas fully connected
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: imagen → embedding global.

        Args:
            x: tensor [B, C, H, W] — imagen de entrada (típicamente 224×224)

        Returns:
            embedding: tensor [B, embed_dim] — vector de embedding global
        """
        # Stem: reduce resolución 4× (224→56 con stride 2 + maxpool stride 2)
        out = self.stem(x)

        # DenseBlocks + Transition Layers
        for i, block in enumerate(self.blocks):
            out = block(out)
            if i < len(self.transitions):
                out = self.transitions[i](out)

        # BN final + ReLU + Global Average Pooling
        out = F.relu(self.final_bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))  # [B, C_final, 1, 1]
        out = out.view(out.size(0), -1)  # [B, C_final]

        # Proyección lineal al embed_dim
        embedding = self.projection(out)  # [B, embed_dim]

        return embedding


# ── Interface pública ───────────────────────────────────────────────────────


def build_densenet(
    in_channels: int = 3,
    embed_dim: int = 1024,
    growth_rate: int = 32,
    block_config: tuple = (6, 12, 24, 16),
    **kwargs,
) -> nn.Module:
    """
    Construye DenseNet custom desde cero sin pesos preentrenados.

    Esta función es la interface pública para crear un DenseNet compatible
    con el pipeline de Fase 1. El modelo retorna embeddings de dimensión
    `embed_dim` directamente (sin cabeza de clasificación).

    Args:
        in_channels: canales de entrada (1 para rayos X grises, 3 para RGB)
        embed_dim: dimensión del embedding de salida (default 1024)
        growth_rate: feature maps nuevos por capa densa (k=32 en DenseNet-121)
        block_config: número de capas por DenseBlock
                     (6,12,24,16) = DenseNet-121 (default)
                     (6,12,32,32) = DenseNet-169
                     (6,12,48,32) = DenseNet-201
        **kwargs: argumentos adicionales ignorados (compatibilidad futura)

    Returns:
        nn.Module con forward(x) → [B, embed_dim]
    """
    if kwargs:
        log.debug(
            "[DenseNet] Argumentos adicionales ignorados: %s", list(kwargs.keys())
        )

    model = DenseNet(
        in_channels=in_channels,
        embed_dim=embed_dim,
        growth_rate=growth_rate,
        block_config=block_config,
    )

    return model


# ── Registro en el interceptor de backbone_loader ───────────────────────────

_DENSENET_REGISTERED = False


def _register_densenet_interceptor():
    """
    Registra DenseNet-121 como backbone disponible en backbone_loader.

    Funciona como el interceptor de CvT-13: al importar este módulo,
    el nombre 'densenet121_custom' queda disponible en BACKBONE_CONFIGS
    y timm.create_model lo intercepta automáticamente.

    Idempotente: seguro para múltiples importaciones.
    """
    global _DENSENET_REGISTERED
    if _DENSENET_REGISTERED:
        return

    try:
        import timm
        from fase1_config import BACKBONE_CONFIGS

        # Registrar en BACKBONE_CONFIGS si no existe
        if "densenet121_custom" not in BACKBONE_CONFIGS:
            BACKBONE_CONFIGS["densenet121_custom"] = {
                "d_model": 1024,
                "vram_gb": 3.0,
            }
            log.debug(
                "[DenseNet] Registrado en BACKBONE_CONFIGS: "
                "densenet121_custom (d_model=1024, ~3 GB VRAM)"
            )

        # Interceptar timm.create_model para el nombre 'densenet121_custom'
        _original_create = timm.create_model

        def _patched_create_model(model_name, *args, **kwargs):
            if model_name == "densenet121_custom":
                log.info(
                    "[DenseNet/patch] densenet121_custom interceptado → DenseNet custom"
                )
                device = kwargs.get("device", "cpu")
                model = build_densenet(
                    in_channels=3,
                    embed_dim=1024,
                    growth_rate=32,
                    block_config=(6, 12, 24, 16),
                )
                # Congelar y mover al dispositivo
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()
                model.to(device)
                return model
            return _original_create(model_name, *args, **kwargs)

        # Solo parchear si no fue ya parcheado por otro módulo
        # (evita encadenar interceptores infinitamente)
        if not getattr(timm.create_model, "_densenet_patched", False):
            _patched_create_model._densenet_patched = True
            # Preservar el flag de CvT-13 si existe
            if hasattr(timm.create_model, "_densenet_patched"):
                pass  # ya parcheado
            timm.create_model = _patched_create_model

        _DENSENET_REGISTERED = True
        log.debug("[DenseNet/patch] Interceptor registrado en timm.create_model")

    except ImportError as e:
        log.warning(
            "[DenseNet] No se pudo registrar interceptor (timm o fase1_config "
            "no disponible): %s. El modelo sigue siendo usable directamente "
            "via build_densenet().",
            e,
        )
        _DENSENET_REGISTERED = True  # Marcar como intentado para no reintentar


# Activar al importar este módulo (como backbone_cvt13.py)
_register_densenet_interceptor()

# Modelos de expertos — Fase 2
#
# Lazy imports: cada experto se carga bajo demanda para evitar que
# dependencias pesadas (timm, torchvision) se importen innecesariamente.
# Para usar un modelo, importa directamente desde su módulo:
#   from fase2.models.expert1_convnext import HybridDeepVision
#   from fase2.models.expert_oa_efficientnet_b3 import ExpertOAEfficientNetB3


def get_expert_oa_efficientnet_b3():
    from .expert_oa_efficientnet_b3 import ExpertOAEfficientNetB3

    return ExpertOAEfficientNetB3


def get_expert_oa_efficientnet_b0():
    from .expert_oa_efficientnet_b3 import ExpertOAEfficientNetB0

    return ExpertOAEfficientNetB0


def get_expert_oa_vgg16bn():
    from .expert_oa_efficientnet_b3 import ExpertOAVGG16BN

    return ExpertOAVGG16BN


__all__ = [
    "get_expert_oa_efficientnet_b3",
    "get_expert_oa_efficientnet_b0",
    "get_expert_oa_vgg16bn",
]

"""
moe: paquete del sistema Mixture of Experts medico.

API publica:
  - MoESystem: orquestador end-to-end (preproc + ViT + router + 5 expertos + OOD)
  - MoEResponse: respuesta agregada por batch
  - ExpertWrapper: wrapper unificado por experto con logit adjustment opcional
  - ExpertSpec, EXPERT_SPECS: metadatos por experto (arq, input_hw, clases, thresholds)
  - ExpertOutput: salida heterogenea de cada experto
  - AdaptivePreprocessor: 2D/3D -> (B_eff, 3, 224, 224) para el router
  - PreprocessOutput: tensor + modality + n_slices
  - LinearRouter: head Linear sobre CLS tokens 192-dim
  - MahalanobisOOD: distancia diagonal a 5 centroides para OOD

Ejemplo de inferencia end-to-end:

    import torch
    from moe import MoESystem

    system = MoESystem(device="cuda" if torch.cuda.is_available() else "cpu")
    x_2d = torch.randint(0, 255, (1, 3, 1024, 1024), dtype=torch.uint8).float()
    resp = system(x_2d)
    print("experto:", int(resp.selected_expert[0]))
    print("probs:", resp.expert_outputs[0].probs)
    print("is_ood:", bool(resp.is_ood[0]))

Artefactos cargados:
  - router Linear: outputs/moe_phase2/router_linear_best.pt
  - ViT-Tiny: timm `vit_tiny_patch16_224.augreg_in21k_ft_in1k` (pretrained=True)
  - 5 expertos: ver moe/experts/wrappers.py::EXPERT_SPECS
"""

from __future__ import annotations

from moe.moe_system import MoESystem, MoEResponse
from moe.experts.wrappers import (
    EXPERT_SPECS,
    ExpertSpec,
    ExpertWrapper,
    ExpertOutput,
)
from moe.preprocessing.adaptive import AdaptivePreprocessor, PreprocessOutput
from moe.routing.linear_router import LinearRouter, MahalanobisOOD

__all__ = [
    "MoESystem",
    "MoEResponse",
    "EXPERT_SPECS",
    "ExpertSpec",
    "ExpertWrapper",
    "ExpertOutput",
    "AdaptivePreprocessor",
    "PreprocessOutput",
    "LinearRouter",
    "MahalanobisOOD",
]

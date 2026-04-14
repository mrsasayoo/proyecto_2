"""
MoESystem: wrapper que ensambla router + 6 expertos para fine-tuning.

NO carga pesos reales — en dry-run usa arquitecturas vacias (inicializacion
aleatoria). En produccion, fase5_finetune_global.py le inyecta los
state_dicts cargados de los checkpoints de Fases 2-4.

Componentes:
    - experts: nn.ModuleList de 6 expertos (0-5)
    - router: modulo de routing (LinearGatingHead)
    - backbone: opcional, si existe como modulo separado

Forward pass:
    x: imagen de entrada [B, C, H, W] o [B, C, D, H, W] para 3D
    expert_id: int — que experto procesa este batch (para L_task)
    returns: dict con 'logits', 'gates', 'expert_id'
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger("fase5")


class MoESystem(nn.Module):
    """
    Sistema MoE completo para fine-tuning global.

    Ensambla los 6 expertos y el router en un unico nn.Module para que
    PyTorch pueda rastrear todos los parametros y aplicar el optimizer
    con param_groups diferenciados.

    Args:
        experts: nn.ModuleList con 6 expertos (indices 0-5).
        router: modulo de routing (LinearGatingHead o equivalente).
        backbone: modulo backbone opcional (puede no existir como modulo separado).
    """

    def __init__(
        self,
        experts: nn.ModuleList,
        router: nn.Module,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.experts = experts
        self.router = router
        self.backbone = backbone
        self.n_experts_domain = 5  # expertos 0-4, no CAE
        self.n_experts_total = 6

    def forward(
        self,
        x: torch.Tensor,
        expert_id: int,
        domain_ids: Optional[torch.LongTensor] = None,
    ) -> dict:
        """
        Ejecuta el experto indicado y el router sobre el mismo input.

        Para 2D experts (0-2, 5): x debe ser [B, 3, 224, 224]
        Para 3D experts (3-4): x debe ser [B, 1, 64, 64, 64]

        El router opera sobre embeddings (en Stage 1) o directamente
        sobre features extraidas (en Stage 2-3). En el dry-run,
        se genera un embedding sintetico para el router.

        Args:
            x: tensor de entrada con shape apropiado para el experto.
            expert_id: indice del experto (0-5).
            domain_ids: [B] tensor int64 con IDs de dominio (0-5) para
                Expert 5 (Res-U-Net). Si None, Expert 5 usa dominio 5
                ("Unknown") por defecto. Ignorado para expertos 0-4.

        Returns:
            dict con keys:
                'logits': salida del experto (shape depende del experto)
                'gates': probabilidades del router [B, n_experts]
                'expert_id': el expert_id de entrada
                'recon': reconstruccion si expert_id==5, None si no
        """
        result = {
            "logits": None,
            "gates": None,
            "expert_id": expert_id,
            "recon": None,
        }

        # --- Forward del experto indicado ---
        expert = self.experts[expert_id]
        if expert_id == 5:
            # ConditionedResUNetAE retorna (recon, z)
            recon, z = expert(x, domain_ids)
            result["logits"] = z  # latente como "logits"
            result["recon"] = recon
        else:
            logits = expert(x)
            result["logits"] = logits

        # --- Forward del router ---
        # El router espera embeddings [B, d_model].
        # En dry-run, generamos un embedding sintetico del tamano correcto.
        d_model = self.router.gate.in_features
        batch_size = x.shape[0]
        # Crear embedding sintetico (en produccion se usaria el backbone)
        dummy_embedding = torch.randn(
            batch_size, d_model, device=x.device, dtype=x.dtype
        )
        gates = self.router(dummy_embedding)
        result["gates"] = gates

        return result

    def get_trainable_params_by_component(self) -> dict[str, int]:
        """
        Retorna dict con numero de parametros entrenables por componente.

        Returns:
            dict con keys: 'expert_0' ... 'expert_5', 'router', 'backbone', 'total'
        """
        counts = {}
        total = 0

        for i, expert in enumerate(self.experts):
            n = sum(p.numel() for p in expert.parameters() if p.requires_grad)
            counts[f"expert_{i}"] = n
            total += n

        n_router = sum(p.numel() for p in self.router.parameters() if p.requires_grad)
        counts["router"] = n_router
        total += n_router

        if self.backbone is not None:
            n_backbone = sum(
                p.numel() for p in self.backbone.parameters() if p.requires_grad
            )
            counts["backbone"] = n_backbone
            total += n_backbone

        counts["total"] = total
        return counts


def build_moe_system_dry_run(d_model: int = 192) -> MoESystem:
    """
    Construye un MoESystem sintetico para dry-run.

    Instancia los 6 expertos y el router con arquitecturas reales pero sin
    cargar checkpoints (weights=None, inicializacion aleatoria).

    Args:
        d_model: dimension del embedding del backbone (default=192 para ViT-Tiny).

    Returns:
        MoESystem listo para forward pass sintetico.
    """
    from fase2.models.expert1_convnext import Expert1ConvNeXtTiny
    from fase2.models.expert2_efficientnet import Expert2ConvNeXtSmall
    from fase2.models.expert_oa_vgg16bn import ExpertOAEfficientNetB0
    from fase2.models.expert3_r3d18 import Expert3MC318
    from fase2.models.expert4_swin3d import ExpertPancreasSwin3D
    from fase3.models.expert6_resunet import ConditionedResUNetAE

    # Import LinearGatingHead directly — fase2/routers/__init__.py uses bare
    # imports (from linear import ...) that only work when routers/ is on sys.path.
    # linear.py itself imports from config, fase2_config, router_metrics which
    # live in fase2/ and pipeline/, so we ensure both are on sys.path.
    _pipeline_dir = str(Path(__file__).resolve().parents[1])
    _fase2_dir = str(Path(__file__).resolve().parents[1] / "fase2")
    _routers_dir = str(Path(__file__).resolve().parents[1] / "fase2" / "routers")
    for _d in (_pipeline_dir, _fase2_dir, _routers_dir):
        if _d not in sys.path:
            sys.path.insert(0, _d)
    from linear import LinearGatingHead

    log.info("[MoESystem] Construyendo sistema MoE sintetico para dry-run...")

    # --- Instanciar expertos ---
    experts = nn.ModuleList(
        [
            Expert1ConvNeXtTiny(dropout_fc=0.3, num_classes=14),  # Expert 0
            Expert2ConvNeXtSmall(),  # Expert 1
            ExpertOAEfficientNetB0(),  # Expert 2 — EfficientNet-B0, 5 clases KL
            Expert3MC318(
                spatial_dropout_p=0.15, fc_dropout_p=0.4, num_classes=2
            ),  # Expert 3
            ExpertPancreasSwin3D(in_channels=1, num_classes=2),  # Expert 4
            ConditionedResUNetAE(
                in_ch=3, base_ch=64, n_domains=6
            ),  # Expert 5 → Res-U-Net v6 condicionado (OOD)
        ]
    )

    # --- Instanciar router ---
    # n_experts = N_EXPERTS_DOMAIN = 5 (el CAE no participa en routing directo)
    router = LinearGatingHead(d_model=d_model, n_experts=5)

    log.info("[MoESystem] 6 expertos + 1 router (LinearGatingHead) creados")

    return MoESystem(experts=experts, router=router, backbone=None)

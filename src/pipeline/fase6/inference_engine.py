"""
inference_engine.py
-------------------
Forward de INFERENCIA del sistema MoE.
Diferencia crítica respecto a moe_model.py (entrenamiento):
  - Entrenamiento: forward(x, expert_id) — expert_id conocido
  - Inferencia:    forward(x) — router decide qué experto activar

Flujo de inferencia:
  1. backbone(x) → z  (embeddings)
  2. router(z)   → gates  (distribución sobre N_EXPERTS_DOMAIN expertos)
  3. H = -sum(gates * log(gates + eps))  (entropía de Shannon)
  4. if H > entropy_threshold:
         output = experts[CAE_EXPERT_IDX](x)   # OOD → experto CAE
     else:
         expert_id = argmax(gates)
         output = experts[expert_id](x)         # experto de dominio
  5. return output, expert_id_used, gates, H
"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.pipeline.fase6.fase6_config import (
    ENTROPY_THRESHOLD_PATH,
    CAE_EXPERT_IDX,
    N_EXPERTS_DOMAIN,
)

logger = logging.getLogger(__name__)


class InferenceEngine(nn.Module):
    """
    Wraps MoESystem para inferencia sin conocer expert_id.

    Durante entrenamiento, MoESystem.forward(x, expert_id) recibe el expert_id
    como ground truth. En inferencia, el router decide qué experto activar
    basándose en la entropía de Shannon de la distribución de gating.

    Args:
        moe_system: MoESystem instance (ya cargado con checkpoint).
        entropy_threshold: float — umbral H para OOD. Si None, se carga
            de ENTROPY_THRESHOLD_PATH o se usa fallback log(N)/2.
        device: torch.device donde ejecutar la inferencia.
    """

    def __init__(
        self,
        moe_system: nn.Module,
        entropy_threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.moe = moe_system
        self.device = device or torch.device("cpu")
        self.entropy_threshold = entropy_threshold

        if self.entropy_threshold is None:
            self._load_threshold()

    def _load_threshold(self) -> None:
        """Carga el umbral de entropía desde disco o usa fallback."""
        path = Path(ENTROPY_THRESHOLD_PATH)
        if path.exists():
            with open(path, "rb") as f:
                self.entropy_threshold = pickle.load(f)
            logger.info("Entropy threshold loaded: %.4f", self.entropy_threshold)
        else:
            # fallback: usar log(N)/2 como umbral por defecto
            self.entropy_threshold = math.log(N_EXPERTS_DOMAIN) / 2.0
            logger.warning(
                "Entropy threshold file not found at %s. Using default: %.4f",
                path,
                self.entropy_threshold,
            )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
    ) -> dict:
        """
        Inferencia completa: backbone → router → (OOD check) → expert.

        Args:
            x: tensor de entrada [B, C, H, W] (2D) o [B, C, D, H, W] (3D).

        Returns:
            dict con keys:
                logits    : list[torch.Tensor] — output del experto seleccionado
                            por muestra. Lista (no stack) porque dimensiones
                            pueden diferir entre expertos 2D/3D.
                expert_ids: list[int] — índice del experto usado por muestra
                            (0-4 dominio, 5 CAE/OOD).
                gates     : torch.Tensor — distribución del router
                            (shape: [B, N_EXPERTS_DOMAIN]).
                entropy   : torch.Tensor — entropía de Shannon por muestra
                            (shape: [B]).
                is_ood    : torch.Tensor — máscara bool OOD (shape: [B]).
        """
        x = x.to(self.device)

        # 1. Backbone → embeddings
        if self.moe.backbone is not None:
            z = self.moe.backbone(x)
        else:
            # backbone=None: generate synthetic embedding for the router
            # (same approach as MoESystem.forward in moe_model.py)
            d_model = self.moe.router.gate.in_features
            z = torch.randn(x.shape[0], d_model, device=x.device, dtype=x.dtype)

        # 2. Router → gates  (softmax output, shape [B, N_EXPERTS_DOMAIN])
        #    LinearGatingHead ya aplica softmax internamente
        gates = self.moe.router(z)

        # 3. Entropía de Shannon por muestra
        eps = 1e-8
        entropy = -(gates * torch.log(gates + eps)).sum(dim=-1)  # [B]

        # 4. OOD check: entropía alta → muestra fuera de distribución
        is_ood = entropy > self.entropy_threshold  # [B] bool

        # 5. Expert selection — per-sample routing (loop sobre batch)
        #    Loop intencional: los expertos pueden tener dimensiones de salida
        #    heterogéneas (2D vs 3D, distintas num_classes).
        batch_size = x.shape[0]
        logits_list = []
        expert_ids = []

        for i in range(batch_size):
            xi = x[i : i + 1]  # [1, C, H, W] o [1, C, D, H, W]

            if is_ood[i]:
                # OOD → experto CAE (índice 5)
                out_i = self.moe.experts[CAE_EXPERT_IDX](xi)
                # CAE returns (recon, z) tuple — use z (latent) as logit
                logit_i = out_i[1] if isinstance(out_i, (list, tuple)) else out_i
                expert_ids.append(CAE_EXPERT_IDX)
            else:
                eid = int(gates[i].argmax().item())
                logit_i = self.moe.experts[eid](xi)
                expert_ids.append(eid)

            logits_list.append(logit_i)

        return {
            "logits": logits_list,  # list of tensors (heterogéneo)
            "expert_ids": expert_ids,  # list of ints
            "gates": gates,  # [B, N_EXPERTS_DOMAIN]
            "entropy": entropy,  # [B]
            "is_ood": is_ood,  # [B] bool
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        moe_system_class: type,
        moe_system_kwargs: dict,
        entropy_threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> InferenceEngine:
        """
        Factory: carga MoESystem desde checkpoint y crea InferenceEngine.

        Si el checkpoint no existe, el MoESystem se inicializa con pesos
        aleatorios (útil para dry-run / smoke test, pero las métricas
        serán meaningless).

        Args:
            checkpoint_path: ruta al .pt del MoESystem.
            moe_system_class: clase MoESystem (para instanciar).
            moe_system_kwargs: kwargs para MoESystem.__init__.
            entropy_threshold: umbral de entropía (opcional, se carga de disco
                si no se proporciona).
            device: torch.device (default: cuda si disponible, sino cpu).

        Returns:
            InferenceEngine listo para forward(x).
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = Path(checkpoint_path)

        moe = moe_system_class(**moe_system_kwargs)

        if path.exists():
            state = torch.load(path, map_location=device, weights_only=True)
            # El checkpoint puede guardar 'model_state_dict' o directamente
            # el state dict completo
            if isinstance(state, dict) and "model_state_dict" in state:
                moe.load_state_dict(state["model_state_dict"])
            elif isinstance(state, dict) and "state_dict" in state:
                moe.load_state_dict(state["state_dict"])
            else:
                moe.load_state_dict(state)
            logger.info("MoESystem checkpoint loaded from %s", path)
        else:
            logger.warning(
                "Checkpoint not found at %s. "
                "Using randomly initialized weights — metrics will be meaningless.",
                path,
            )

        moe = moe.to(device)
        moe.eval()

        return cls(
            moe_system=moe,
            entropy_threshold=entropy_threshold,
            device=device,
        )

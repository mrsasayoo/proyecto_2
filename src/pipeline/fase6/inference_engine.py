"""
inference_engine.py
-------------------
Forward de INFERENCIA del sistema MoE con feedback loop CAE completo.

Diferencia crítica respecto a moe_model.py (entrenamiento):
  - Entrenamiento: forward(x, expert_id) — expert_id conocido
  - Inferencia:    forward(x) — router decide qué experto activar

Flujo de inferencia:
  1. backbone(x) → z  (embeddings)
  2. router(z)   → gates  (distribución sobre N_EXPERTS_DOMAIN expertos)
  3. H = -sum(gates * log(gates + eps))  (entropía de Shannon)
  4. if H ≤ entropy_threshold:
         expert_id = argmax(gates)
         output = experts[expert_id](x)         # experto de dominio
     else:
         → feedback loop CAE (ver _cae_feedback_loop)
  5. return resultado con ood_status

Feedback loop CAE (3 paths basados en ε = MSE(x, CAE(x))):
  - ε ≤ θ_leve           → "clean": falsa alarma, re-enrutar x original
  - θ_leve < ε ≤ θ_OOD   → "denoised": denoising con x̂=CAE(x), re-routing
      - Si H(g(x̂)) > entropy_threshold → "professional_review"
      - Si no → enrutar al experto con g_k* máximo
  - ε > θ_OOD            → "ood_rejected": rechazar
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.pipeline.fase6.fase6_config import (
    ENTROPY_THRESHOLD_PATH,
    OOD_THRESHOLDS_JSON_PATH,
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

    Cuando la entropía supera el umbral calibrado, se activa el feedback loop
    del CAE con tres paths según el error de reconstrucción.

    Args:
        moe_system: MoESystem instance (ya cargado con checkpoint).
        entropy_threshold: float — umbral H para OOD. Si None, se carga
            de OOD_THRESHOLDS_JSON_PATH / ENTROPY_THRESHOLD_PATH o fallback.
        cae_theta_leve: float | None — umbral bajo de MSE. Si None, se carga
            desde OOD_THRESHOLDS_JSON_PATH.
        cae_theta_ood: float | None — umbral alto de MSE. Si None, se carga
            desde OOD_THRESHOLDS_JSON_PATH.
        device: torch.device donde ejecutar la inferencia.
    """

    def __init__(
        self,
        moe_system: nn.Module,
        entropy_threshold: float | None = None,
        cae_theta_leve: float | None = None,
        cae_theta_ood: float | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.moe = moe_system
        self.device = device or torch.device("cpu")
        self.entropy_threshold = entropy_threshold
        self.cae_theta_leve = cae_theta_leve
        self.cae_theta_ood = cae_theta_ood

        # Cargar umbrales que no fueron proporcionados
        self._load_thresholds()

    def _load_thresholds(self) -> None:
        """Carga umbrales faltantes desde JSON (preferido) o pickle (legacy)."""
        json_path = Path(OOD_THRESHOLDS_JSON_PATH)

        if json_path.exists():
            with open(json_path, "r") as f:
                data: dict = json.load(f)

            if self.entropy_threshold is None:
                self.entropy_threshold = data.get("entropy_threshold")
                if self.entropy_threshold is not None:
                    logger.info(
                        "Entropy threshold loaded from JSON: %.4f",
                        self.entropy_threshold,
                    )

            if self.cae_theta_leve is None:
                self.cae_theta_leve = data.get("cae_theta_leve")
                if self.cae_theta_leve is not None:
                    logger.info(
                        "CAE θ_leve loaded from JSON: %.6f", self.cae_theta_leve
                    )

            if self.cae_theta_ood is None:
                self.cae_theta_ood = data.get("cae_theta_ood")
                if self.cae_theta_ood is not None:
                    logger.info("CAE θ_OOD loaded from JSON: %.6f", self.cae_theta_ood)

        # Fallback para entropy_threshold: pickle legacy
        if self.entropy_threshold is None:
            pkl_path = Path(ENTROPY_THRESHOLD_PATH)
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    self.entropy_threshold = pickle.load(f)
                logger.info(
                    "Entropy threshold loaded from pickle: %.4f",
                    self.entropy_threshold,
                )

        # Último fallback: valor heurístico
        if self.entropy_threshold is None:
            self.entropy_threshold = math.log(N_EXPERTS_DOMAIN) / 2.0
            logger.warning(
                "No entropy threshold found on disk. Using default: %.4f",
                self.entropy_threshold,
            )

        # Log si faltan umbrales CAE (no es error fatal: el feedback loop
        # seguirá funcionando pero sin los tres paths diferenciados)
        if self.cae_theta_leve is None or self.cae_theta_ood is None:
            logger.warning(
                "CAE thresholds not fully loaded (θ_leve=%s, θ_OOD=%s). "
                "Feedback loop will fall back to legacy behavior (latent-only). "
                "Run OODDetector.calibrate_cae_thresholds() to calibrate.",
                self.cae_theta_leve,
                self.cae_theta_ood,
            )

    # ------------------------------------------------------------------
    # Routing helpers (sin side-effects sobre el estado del engine)
    # ------------------------------------------------------------------

    def _compute_gates_and_entropy(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ejecuta backbone + router y calcula entropía de Shannon.

        Args:
            x: tensor [B, C, H, W] (o 3D).

        Returns:
            (gates, entropy) — gates [B, N_EXPERTS_DOMAIN], entropy [B].
        """
        if self.moe.backbone is not None:
            z = self.moe.backbone(x)
        else:
            d_model = self.moe.router.gate.in_features
            z = torch.randn(x.shape[0], d_model, device=x.device, dtype=x.dtype)

        gates = self.moe.router(z)

        eps = 1e-8
        entropy = -(gates * torch.log(gates + eps)).sum(dim=-1)
        return gates, entropy

    def _route_to_domain_expert(
        self, xi: torch.Tensor, gates_i: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """
        Enruta una sola muestra al experto de dominio con gate máximo.

        Args:
            xi: tensor [1, C, H, W].
            gates_i: tensor [N_EXPERTS_DOMAIN] — gates para esta muestra.

        Returns:
            (logits, expert_id).
        """
        eid = int(gates_i.argmax().item())
        logits = self.moe.experts[eid](xi)
        return logits, eid

    # ------------------------------------------------------------------
    # CAE feedback loop
    # ------------------------------------------------------------------

    def _cae_feedback_loop(self, xi: torch.Tensor) -> dict:
        """
        Feedback loop completo del CAE para una muestra con alta entropía.

        Tres paths basados en ε = MSE(x, CAE(x)):
          1. ε ≤ θ_leve           → "clean" (falsa alarma): re-enrutar x original
          2. θ_leve < ε ≤ θ_OOD   → denoising: x̂=CAE(x), re-routing
             - Si H(g(x̂)) > entropy_threshold → "professional_review"
             - Else → enrutar x̂ al experto con gate máximo
          3. ε > θ_OOD            → "ood_rejected": rechazar

        Args:
            xi: tensor [1, C, H, W], FP32.

        Returns:
            dict con keys: logits, expert_id, ood_status.
        """
        cae = self.moe.experts[CAE_EXPERT_IDX]

        # Asegurar FP32 para todo el path del CAE
        xi_fp32 = xi.float()

        # Calcular error de reconstrucción ε
        epsilon = cae.reconstruction_error(xi_fp32)  # [1]
        eps_val = epsilon.item()

        # Si los umbrales CAE no están calibrados, fallback a legacy
        if self.cae_theta_leve is None or self.cae_theta_ood is None:
            logger.debug(
                "CAE thresholds not calibrated — legacy fallback (latent-only)"
            )
            recon, z = cae(xi_fp32)
            return {
                "logits": z,
                "expert_id": CAE_EXPERT_IDX,
                "ood_status": "ood_rejected",
            }

        # PATH 3: ε > θ_OOD → OOD absoluto, rechazar
        if eps_val > self.cae_theta_ood:
            logger.debug(
                "CAE feedback: ε=%.6f > θ_OOD=%.6f → ood_rejected",
                eps_val,
                self.cae_theta_ood,
            )
            recon, z = cae(xi_fp32)
            return {
                "logits": z,
                "expert_id": CAE_EXPERT_IDX,
                "ood_status": "ood_rejected",
            }

        # PATH 1: ε ≤ θ_leve → imagen limpia, falsa alarma del router
        if eps_val <= self.cae_theta_leve:
            logger.debug(
                "CAE feedback: ε=%.6f ≤ θ_leve=%.6f → clean (false alarm), re-routing",
                eps_val,
                self.cae_theta_leve,
            )
            # Re-enrutar la imagen original al experto de dominio
            gates_orig, _ = self._compute_gates_and_entropy(xi)
            logits, eid = self._route_to_domain_expert(xi, gates_orig[0])
            return {
                "logits": logits,
                "expert_id": eid,
                "ood_status": "clean",
            }

        # PATH 2: θ_leve < ε ≤ θ_OOD → ruido leve, denoising
        logger.debug(
            "CAE feedback: θ_leve=%.6f < ε=%.6f ≤ θ_OOD=%.6f → denoising",
            self.cae_theta_leve,
            eps_val,
            self.cae_theta_ood,
        )

        # Denoising: usar la reconstrucción del CAE como nueva entrada
        recon, _ = cae(xi_fp32)
        x_hat = recon  # [1, C, H, W], ya FP32

        # Re-ejecutar routing sobre x̂
        gates_denoised, entropy_denoised = self._compute_gates_and_entropy(x_hat)

        # ¿Sigue con alta entropía después del denoising?
        if entropy_denoised[0].item() > self.entropy_threshold:
            logger.debug(
                "CAE feedback (denoised): H(g(x̂))=%.4f > threshold=%.4f "
                "→ professional_review",
                entropy_denoised[0].item(),
                self.entropy_threshold,
            )
            return {
                "logits": x_hat,  # devolver la reconstrucción como output
                "expert_id": CAE_EXPERT_IDX,
                "ood_status": "professional_review",
            }

        # Entropía baja tras denoising → enrutar al experto con gate máximo
        logits, eid = self._route_to_domain_expert(x_hat, gates_denoised[0])
        logger.debug(
            "CAE feedback (denoised): H(g(x̂))=%.4f ≤ threshold=%.4f "
            "→ routed to expert %d",
            entropy_denoised[0].item(),
            self.entropy_threshold,
            eid,
        )
        return {
            "logits": logits,
            "expert_id": eid,
            "ood_status": "denoised",
        }

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
    ) -> dict:
        """
        Inferencia completa: backbone → router → (OOD check) → expert / CAE feedback.

        Args:
            x: tensor de entrada [B, C, H, W] (2D) o [B, C, D, H, W] (3D).

        Returns:
            dict con keys:
                logits     : list[torch.Tensor] — output del experto seleccionado
                             por muestra. Lista (no stack) porque dimensiones
                             pueden diferir entre expertos 2D/3D.
                expert_ids : list[int] — índice del experto usado por muestra
                             (0-4 dominio, 5 CAE/OOD).
                gates      : torch.Tensor — distribución del router
                             (shape: [B, N_EXPERTS_DOMAIN]).
                entropy    : torch.Tensor — entropía de Shannon por muestra
                             (shape: [B]).
                is_ood     : torch.Tensor — máscara bool OOD (shape: [B]).
                ood_status : list[str] — estado OOD por muestra. Valores:
                             "normal" | "clean" | "denoised" |
                             "ood_rejected" | "professional_review".
        """
        x = x.to(self.device)

        # 1-3. Backbone → Router → Entropía
        gates, entropy = self._compute_gates_and_entropy(x)

        # 4. OOD check: entropía alta → muestra fuera de distribución
        is_ood = entropy > self.entropy_threshold  # [B] bool

        # 5. Expert selection — per-sample routing (loop sobre batch)
        #    Loop intencional: los expertos pueden tener dimensiones de salida
        #    heterogéneas (2D vs 3D, distintas num_classes).
        batch_size = x.shape[0]
        logits_list: list[torch.Tensor] = []
        expert_ids: list[int] = []
        ood_status_list: list[str] = []

        for i in range(batch_size):
            xi = x[i : i + 1]  # [1, C, H, W] o [1, C, D, H, W]

            if is_ood[i]:
                # Alta entropía → feedback loop CAE
                fb = self._cae_feedback_loop(xi)
                logits_list.append(fb["logits"])
                expert_ids.append(fb["expert_id"])
                ood_status_list.append(fb["ood_status"])
            else:
                # Entropía normal → enrutar al experto de dominio
                logit_i, eid = self._route_to_domain_expert(xi, gates[i])
                logits_list.append(logit_i)
                expert_ids.append(eid)
                ood_status_list.append("normal")

        return {
            "logits": logits_list,  # list of tensors (heterogéneo)
            "expert_ids": expert_ids,  # list of ints
            "gates": gates,  # [B, N_EXPERTS_DOMAIN]
            "entropy": entropy,  # [B]
            "is_ood": is_ood,  # [B] bool
            "ood_status": ood_status_list,  # list of str
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        moe_system_class: type,
        moe_system_kwargs: dict,
        entropy_threshold: float | None = None,
        cae_theta_leve: float | None = None,
        cae_theta_ood: float | None = None,
        device: torch.device | None = None,
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
            cae_theta_leve: umbral bajo MSE del CAE (opcional, se carga de disco).
            cae_theta_ood: umbral alto MSE del CAE (opcional, se carga de disco).
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
            cae_theta_leve=cae_theta_leve,
            cae_theta_ood=cae_theta_ood,
            device=device,
        )

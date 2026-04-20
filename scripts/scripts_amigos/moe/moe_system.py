"""
MoESystem: orquestador end-to-end del Mixture of Experts medico.

Flujo:
  x_raw (BCHW 2D o BCDHW 3D)
    -> AdaptivePreprocessor -> (B_eff, 3, 224, 224) vista router
    -> ViT-Tiny.forward_features -> CLS token (B_eff, 192)
    -> si 3D: mean-pool N slices -> (B, 192)
    -> LinearRouter -> probs (B, 5)
    -> hard_gate (argmax) -> expert_idx (B,)
    -> entropia normalizada + Mahalanobis -> ood_score -> is_ood
    -> por muestra: resize/normalize a resolucion nativa del experto seleccionado
    -> ExpertWrapper.forward -> ExpertOutput heterogeneo
    -> MoEResponse agregada.

Notas de diseno:
  - El router siempre corre, incluso si is_ood=True. El consumidor decide si
    presentar la respuesta del experto o marcarla como fuera de dominio.
  - Para modalidad 3D, la CLS final por sujeto es el promedio de las CLS
    por slice (n_slices_3d=16). Esta es la misma agregacion usada en la
    extraccion de embeddings v2.
  - Los expertos pueden no estar cargados (checkpoint ausente o incompatible).
    En ese caso devuelven placeholder determinista (_PlaceholderHead),
    util para validar el cableado del sistema MoE sin pesos finales.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe.preprocessing.adaptive import AdaptivePreprocessor, PreprocessOutput
from moe.routing.linear_router import LinearRouter, MahalanobisOOD
from moe.experts.wrappers import EXPERT_SPECS, ExpertSpec, ExpertWrapper, ExpertOutput


DEFAULT_VIT_MODEL = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"


@dataclass
class MoEResponse:
    selected_expert: torch.Tensor                # (B,) int64
    routing_probs: torch.Tensor                  # (B, 5)
    gating_entropy: torch.Tensor                 # (B,) normalizada 0-1
    ood_score: np.ndarray                        # (B,) combinado entropy+mahal
    is_ood: np.ndarray                           # (B,) bool
    expert_outputs: list[ExpertOutput]           # len=B
    modality: Literal["2d", "3d"] = "2d"
    timing_ms: dict = field(default_factory=dict)


class MoESystem:
    def __init__(
        self,
        router_ckpt_path: str | Path = "outputs/moe_phase2/router_linear_best.pt",
        device: str = "cpu",
        vit_model: str = DEFAULT_VIT_MODEL,
        load_experts: bool = True,
        n_slices_3d: int = 16,
        preprocessor_mode: Literal["generic", "domain_aware"] = "generic",
    ) -> None:
        self.device = device
        self.preprocessor = AdaptivePreprocessor(
            router_resolution=224, n_slices_3d=n_slices_3d, mode=preprocessor_mode
        )

        self.vit = self._build_vit(vit_model).to(device).eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)

        self.router, self.mahal, self.ood_threshold = self._load_router(
            Path(router_ckpt_path)
        )
        self.router = self.router.to(device).eval()
        for p in self.router.parameters():
            p.requires_grad_(False)

        self.experts: dict[int, ExpertWrapper] = {}
        if load_experts:
            for label, spec in EXPERT_SPECS.items():
                w = ExpertWrapper(spec, strict_load=False).to(device)
                self.experts[label] = w

    @staticmethod
    def _build_vit(vit_model: str) -> nn.Module:
        import timm

        model = timm.create_model(vit_model, pretrained=True, num_classes=0, global_pool="")
        return model

    def _load_router(
        self, ckpt_path: Path
    ) -> tuple[LinearRouter, MahalanobisOOD, float]:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        embed_dim = ckpt.get("embed_dim", 192)
        num_experts = ckpt.get("num_experts", 5)
        router = LinearRouter(embed_dim=embed_dim, num_experts=num_experts)
        router.load_state_dict(ckpt["state_dict"])

        mahal = MahalanobisOOD()
        mahal.means_ = ckpt["mahalanobis_means"].astype(np.float32)
        mahal.inv_var_ = ckpt["mahalanobis_inv_var"].astype(np.float32)

        threshold = float(ckpt.get("ood_threshold_p95", 1.4728))
        return router, mahal, threshold

    @torch.no_grad()
    def _extract_cls(self, x_router: torch.Tensor) -> torch.Tensor:
        """x_router: (B_eff, 3, 224, 224) -> CLS (B_eff, 192)."""
        feats = self.vit.forward_features(x_router)
        if feats.dim() == 3:
            return feats[:, 0]                                          # CLS en pos 0
        if feats.dim() == 2:
            return feats                                                # ya pooleado
        raise RuntimeError(f"forward_features shape inesperado: {feats.shape}")

    def _aggregate_3d(self, cls: torch.Tensor, n_slices: int, batch_size: int) -> torch.Tensor:
        """(B*N, D) -> (B, N, D) -> mean -> (B, D)."""
        return cls.view(batch_size, n_slices, -1).mean(dim=1)

    @torch.no_grad()
    def _expert_native_view(
        self, x_raw: torch.Tensor, spec: ExpertSpec
    ) -> torch.Tensor:
        """Adapta x_raw (una muestra, sin batch collapsed) a la resolucion nativa del experto.

        Admite:
          - x_raw 2D (1, C, H, W) -> 2D experto (1, C_out, H_native, W_native)
          - x_raw 3D (1, 1, D, H, W) -> 3D experto (1, 1, D_native, H_native, W_native)

        Si hay mismatch de modalidad (ej. 2D raw contra experto 3D), devuelve None;
        la muestra se marca como mismatch y se reporta OOD por construccion.
        """
        mean = torch.tensor(spec.normalize_mean, device=x_raw.device).view(
            1, -1, *([1] * (x_raw.dim() - 2))
        )
        std = torch.tensor(spec.normalize_std, device=x_raw.device).view(
            1, -1, *([1] * (x_raw.dim() - 2))
        )

        if spec.modality == "2d":
            if x_raw.dim() != 4:
                return None                                             # mismatch
            x = x_raw.float()
            if x.max() > 1.5:
                x = x / 255.0
            if x.size(1) == 1 and spec.input_channels == 3:
                x = x.repeat(1, 3, 1, 1)
            H, W = spec.input_hw
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            x = (x - mean) / std
            return x

        if spec.modality == "3d":
            if x_raw.dim() != 5:
                return None
            x = x_raw.float()
            if x.size(1) != spec.input_channels:
                if x.size(1) == 1 and spec.input_channels == 1:
                    pass
                else:
                    return None
            D, H, W = spec.input_hw
            x = F.interpolate(x, size=(D, H, W), mode="trilinear", align_corners=False)
            x = (x - mean) / std
            return x

        raise ValueError(f"Modalidad no soportada: {spec.modality}")

    @torch.no_grad()
    def forward(self, x_raw: torch.Tensor) -> MoEResponse:
        t0 = time.perf_counter()

        pre: PreprocessOutput = self.preprocessor(x_raw)
        x_router = pre.tensor.to(self.device)
        t_prep = time.perf_counter()

        cls_all = self._extract_cls(x_router)                           # (B_eff, 192)
        if pre.modality == "3d":
            B = x_raw.size(0)
            cls = self._aggregate_3d(cls_all, pre.n_slices, B)
        else:
            cls = cls_all
        t_vit = time.perf_counter()

        logits, probs = self.router(cls)
        expert_idx = LinearRouter.hard_gate(probs)
        entropy = LinearRouter.entropy(probs)

        mahal_d = self.mahal.score(cls.detach().cpu().numpy().astype(np.float32))
        mahal_norm = mahal_d / (np.median(mahal_d) + 1e-9)
        entropy_np = entropy.detach().cpu().numpy()
        ood_score = 0.5 * entropy_np + 0.5 * mahal_norm
        is_ood = ood_score > self.ood_threshold
        t_router = time.perf_counter()

        expert_outputs: list[ExpertOutput] = []
        for i in range(x_raw.size(0)):
            lbl = int(expert_idx[i].item())
            spec = EXPERT_SPECS[lbl]
            sample = x_raw[i : i + 1]
            x_native = self._expert_native_view(sample, spec)
            if x_native is None:
                expert_outputs.append(
                    ExpertOutput(
                        logits=torch.zeros(1, spec.num_classes, device=self.device),
                        probs=torch.zeros(1, spec.num_classes, device=self.device),
                        prediction=torch.tensor([-1], device=self.device),
                        confidence=torch.tensor([0.0], device=self.device),
                        extra={"mismatch_modality": True, "label": lbl},
                    )
                )
                continue
            eo = self.experts[lbl].forward(x_native.to(self.device))
            expert_outputs.append(eo)
        t_exp = time.perf_counter()

        timing = {
            "preprocess_ms": (t_prep - t0) * 1000.0,
            "vit_ms": (t_vit - t_prep) * 1000.0,
            "router_ms": (t_router - t_vit) * 1000.0,
            "expert_ms": (t_exp - t_router) * 1000.0,
            "total_ms": (t_exp - t0) * 1000.0,
        }

        return MoEResponse(
            selected_expert=expert_idx,
            routing_probs=probs,
            gating_entropy=entropy,
            ood_score=ood_score,
            is_ood=is_ood,
            expert_outputs=expert_outputs,
            modality=pre.modality,
            timing_ms=timing,
        )

    def __call__(self, x_raw: torch.Tensor) -> MoEResponse:
        return self.forward(x_raw)

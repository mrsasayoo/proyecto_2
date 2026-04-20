"""
Wrappers para los 5 expertos del MoE.

Cada ExpertSpec describe el contrato publico:
  - label (0-4) para routing
  - dataset
  - arch (nombre descriptivo, no timm id necesariamente)
  - input_hw y modalidad (2d / 3d)
  - num_classes y class_names
  - task ("multilabel" / "multiclass" / "binary")
  - normalize (mean, std) de la vista nativa del experto
  - postproc (sigmoid / softmax)
  - threshold (por clase o escalar)

ExpertWrapper unifica el forward:
  forward(x_native) -> ExpertOutput { logits, probs, prediction, gating_info }

Si el checkpoint no puede cargarse (arquitectura mismatch, archivo ausente),
el wrapper queda en modo "not_loaded" y devuelve predicciones placeholder
deterministas; util para que el sistema MoE no se rompa en ausencia de pesos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExpertSpec:
    label: int
    dataset: str
    arch: str
    input_hw: tuple[int, ...]                # (H, W) 2D o (D, H, W) 3D
    modality: Literal["2d", "3d"]
    input_channels: int
    num_classes: int
    class_names: list[str]
    task: Literal["multilabel", "multiclass", "binary"]
    normalize_mean: tuple[float, ...]
    normalize_std: tuple[float, ...]
    threshold: float | list[float] = 0.5
    checkpoint_path: str | None = None
    notes: str = ""


@dataclass
class ExpertOutput:
    logits: torch.Tensor                     # (B, C) con C nativo del experto
    probs: torch.Tensor                      # softmax o sigmoid segun task
    prediction: torch.Tensor                 # argmax multiclass; multi-hot multilabel; 0/1 binary
    confidence: torch.Tensor                 # max prob o prob positiva
    extra: dict = field(default_factory=dict)


EXPERT_SPECS: dict[int, ExpertSpec] = {
    0: ExpertSpec(
        label=0,
        dataset="NIH ChestX-ray14",
        arch="CXRExpertSingleHead (ConvNeXt-V2 Base 384 features_only + LSEPool + Linear)",
        input_hw=(384, 384),
        modality="2d",
        input_channels=3,
        num_classes=6,
        class_names=["Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass", "Pneumothorax"],
        task="multilabel",
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        threshold=[0.48, 0.54, 0.58, 0.63, 0.66, 0.69],
        checkpoint_path="hf://mitgar14/moe-medical-experts/exp1v21/ckpt_exp1v21_f0_best.pt",
        notes="v21 oficial. EMA state_dict. tau_logit_adjustment=0.5. AUROC_val=0.8064, F1_opt=0.4979.",
    ),
    1: ExpertSpec(
        label=1,
        dataset="ISIC 2019",
        arch="efficientnet_b3 (torchvision) + classifier(Dropout, Linear(1536, 8))",
        input_hw=(300, 300),
        modality="2d",
        input_channels=3,
        num_classes=8,
        class_names=["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"],
        task="multiclass",
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        threshold=0.5,
        checkpoint_path="models/expert3_isic_best.pth.zip",   # naming local invertido
        notes="F1_macro_val=0.7414. La clase 9 'None of the others' se maneja via OOD router entropy.",
    ),
    2: ExpertSpec(
        label=2,
        dataset="Osteoarthritis Knee",
        arch="efficientnet_b0 (torchvision) + classifier(Dropout, Linear(1280, 5))",
        input_hw=(224, 224),
        modality="2d",
        input_channels=3,
        num_classes=5,
        class_names=["Normal", "Dudoso", "Leve", "Moderado", "Severo"],
        task="multiclass",
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        threshold=0.5,
        checkpoint_path="models/expert2_osteo_best.pth.zip",
        notes="F1_macro_val=0.7987. Taxonomia 5 clases KL (Kellgren-Lawrence), no 3 como decia spec previo.",
    ),
    3: ExpertSpec(
        label=3,
        dataset="LUNA16 + LIDC-IDRI",
        arch="DenseNet-3D custom reproducida (block=(4,8,16,12), growth=32, init=64, final_bn=728)",
        input_hw=(64, 64, 64),
        modality="3d",
        input_channels=1,
        num_classes=2,
        class_names=["no_nodule", "nodule"],
        task="binary",
        normalize_mean=(0.0,),
        normalize_std=(1.0,),
        threshold=0.5,
        checkpoint_path="models/LUNA-LIDCIDRI_best.pt.zip",
        notes="Val F1_macro=0.9438, AUC=0.9911. Arq reconstruida desde shapes (no habia fuente). Preprocessing offline HU+mask+clip.",
    ),
    4: ExpertSpec(
        label=4,
        dataset="PANORAMA Pancreas",
        arch="r3d_18 (torchvision) + fc(Linear(512, 2))",
        input_hw=(64, 64, 64),
        modality="3d",
        input_channels=3,                                      # checkpoint espera 3 ch (replica 1->3)
        num_classes=2,
        class_names=["no_tumor", "tumor"],
        task="binary",
        normalize_mean=(0.43, 0.43, 0.43),                     # HU mean 43.40 normalizada (HU[-150,250] clip a [0,1])
        normalize_std=(1.2013, 1.2013, 1.2013),                # std 120.13 escalada
        threshold=0.5813,
        checkpoint_path="models/exp5_best.pth.zip",
        notes="Best F1=0.6558. HU [-150,250], mean=43.40 std=120.13 foreground. OOD thr=-1.0503. Input replicado 1->3 ch.",
    ),
}


class _PlaceholderHead(nn.Module):
    """Cabeza deterministica usada cuando no se puede cargar el checkpoint real.
    Produce logits basados en una proyeccion aleatoria fija (seed=spec.label) para
    que el sistema MoE no se rompa pero claramente marque 'not_loaded'.
    """

    def __init__(self, spec: ExpertSpec):
        super().__init__()
        torch.manual_seed(spec.label)
        flat_dim = 1
        for d in spec.input_hw:
            flat_dim *= d
        flat_dim *= spec.input_channels
        self.proj = nn.Linear(flat_dim, spec.num_classes)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.reshape(x.size(0), -1))


class ExpertWrapper:
    def __init__(self, spec: ExpertSpec, strict_load: bool = False,
                 hf_token: str | None = None) -> None:
        self.spec = spec
        self.loaded = False
        self.model: nn.Module = _PlaceholderHead(spec)
        self.extras: dict = {}
        self.device = "cpu"
        self.logit_offset: torch.Tensor | None = None
        self._try_load(strict=strict_load, hf_token=hf_token)

    def _try_load(self, strict: bool, hf_token: str | None = None) -> None:
        from moe.experts.loaders import (
            build_exp1_cxr14_v21,
            build_exp2_isic,
            build_exp3_osteo,
            build_exp4_luna,
            build_exp5_pancreas,
        )

        builders = {
            0: lambda: build_exp1_cxr14_v21(hf_token=hf_token),
            1: lambda: build_exp2_isic(self.spec.checkpoint_path),
            2: lambda: build_exp3_osteo(self.spec.checkpoint_path),
            3: lambda: build_exp4_luna(self.spec.checkpoint_path),
            4: lambda: build_exp5_pancreas(self.spec.checkpoint_path),
        }
        fn = builders.get(self.spec.label)
        if fn is None:
            return

        try:
            model, extras = fn()
            self.model = model
            self.model.eval()
            self.extras = extras
            self.loaded = True
            self._maybe_build_logit_offset()
        except Exception as e:
            if strict:
                raise
            self.loaded = False
            self.extras = {"error": str(e)}
            print(f"[ExpertWrapper] load fallback para label {self.spec.label}: {e}")

    def _maybe_build_logit_offset(self) -> None:
        """Precomputa tau * log(prevalence) si el training del experto aplicó
        logit adjustment (Menon 2020). En inferencia, el offset debe restarse de
        los logits ANTES del sigmoid/softmax para que los probs sean consistentes
        con los thresholds calibrados.

        Convencion: extras['tau_logit_adjustment'] (float) y extras['prevalence_train']
        (list[float]) presentes => self.logit_offset: Tensor(1, C). Si no, None.
        """
        tau = self.extras.get("tau_logit_adjustment")
        prev = self.extras.get("prevalence_train")
        if tau is None or prev is None:
            self.logit_offset = None
            return
        prev_np = np.asarray(prev, dtype=np.float64)
        log_prior = np.log(prev_np + 1e-12)
        offset = float(tau) * log_prior                                    # (C,)
        self.logit_offset = torch.tensor(offset, dtype=torch.float32).unsqueeze(0)

    def to(self, device: str) -> "ExpertWrapper":
        self.model = self.model.to(device)
        if getattr(self, "logit_offset", None) is not None:
            self.logit_offset = self.logit_offset.to(device)
        self.device = device
        return self

    def forward(self, x: torch.Tensor) -> ExpertOutput:
        with torch.no_grad():
            logits = self.model(x)
        if getattr(self, "logit_offset", None) is not None:
            logits = logits - self.logit_offset                            # Menon 2020
        if self.spec.task == "multilabel" or self.spec.task == "binary":
            probs = torch.sigmoid(logits)
            if self.spec.task == "binary":
                pred = (probs[:, -1:] > self.spec.threshold).long().squeeze(-1)
                conf = probs[:, -1]
            else:
                thr = self.spec.threshold
                if isinstance(thr, list):
                    thr_t = torch.tensor(thr, device=probs.device)
                    pred = (probs > thr_t).long()
                else:
                    pred = (probs > thr).long()
                conf = probs.max(dim=-1).values
        else:  # multiclass
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            conf = probs.max(dim=-1).values
        return ExpertOutput(
            logits=logits,
            probs=probs,
            prediction=pred,
            confidence=conf,
            extra={"loaded": self.loaded, "label": self.spec.label},
        )

    def __call__(self, x: torch.Tensor) -> ExpertOutput:
        return self.forward(x)

"""
Funciones de pérdida especializadas para el pipeline MoE.

- FocalLossMultiLabel: para ChestXray14 (alternativa a BCEWithLogitsLoss)
- OrdinalLoss: para OA Knee (Chen et al., 2019)
- FocalLoss: para LUNA16 y Pancreas (desbalance extremo)
"""

import torch
import torch.nn as nn


class FocalLossMultiLabel(nn.Module):
    """
    Focal Loss para clasificación multi-label (alternativa a BCEWithLogitsLoss).

    H6 — útil cuando BCEWithLogitsLoss con pesos no converge bien
    en patologías muy raras (Hernia ~0.2%, Pneumonia ~1.2%).

    Uso en FASE 2:
        criterion = FocalLossMultiLabel(gamma=2.0)
        loss = criterion(logits, labels)   # logits: [B,14], labels: [B,14] float
    """
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t  = torch.exp(-bce)
        loss = (1 - p_t) ** self.gamma * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss   # "none"


class OrdinalLoss(nn.Module):
    """
    H1 — OrdinalLoss (Chen et al., 2019 — PMC doi:10.1016/j.compmedimag.2019.05.007).

    Reformula la clasificación ordinal como K-1 clasificaciones binarias
    acumulativas: P(y > k) para k = 0, ..., K-2.

    Para K=3 clases (Normal/Leve/Severo):
      P(y > 0) = P(Leve o Severo)
      P(y > 1) = P(Severo)

    Uso en FASE 2:
        criterion = OrdinalLoss(n_classes=3)
        loss = criterion(logits, labels)   # logits: [B, K-1], labels ∈ {0, 1, 2}
    """
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.n_classes = n_classes
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        K = self.n_classes
        bin_targets = torch.zeros(targets.size(0), K - 1, device=logits.device)
        for k in range(K - 1):
            bin_targets[:, k] = (targets > k).float()
        return self.bce(logits, bin_targets)


class FocalLoss(nn.Module):
    """
    H2 — FocalLoss (Lin et al., ICCV 2017 — arXiv:1708.02002).

    Reduce el gradiente de los ejemplos negativos fáciles y concentra el
    aprendizaje en los positivos y los negativos difíciles.

    Parámetros recomendados:
      LUNA16:   gamma=2, alpha=0.25
      Pancreas: gamma=2, alpha=0.75 (más peso a PDAC+)

    Uso en FASE 2:
        criterion = FocalLoss(gamma=2, alpha=0.25)
        loss = criterion(logits, labels.float())  # logits: [B], labels: [B] ∈ {0,1}
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce    = nn.functional.binary_cross_entropy_with_logits(
                     logits, targets, reduction="none")
        p_t    = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss   = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()

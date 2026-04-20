"""
Router Linear del MoE: Linear + Softmax sobre CLS tokens 192-dim.

Arquitectura:
  Input: CLS token [B, 192] del ViT-Tiny congelado.
  Output: probabilidades sobre 5 expertos [B, 5].

Incluye utilidades para:
  - Hard gating via argmax.
  - Entropia del softmax (senal OOD primaria).
  - Mahalanobis min a centroides por clase (senal OOD secundaria).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class LinearRouter(nn.Module):
    def __init__(self, embed_dim: int = 192, num_experts: int = 5) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.fc = nn.Linear(embed_dim, num_experts)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(z)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

    @staticmethod
    def hard_gate(probs: torch.Tensor) -> torch.Tensor:
        return probs.argmax(dim=-1)

    @staticmethod
    def entropy(probs: torch.Tensor) -> torch.Tensor:
        eps = 1e-9
        H = -(probs * torch.log(probs + eps)).sum(dim=-1)
        return H / math.log(probs.size(-1))


class MahalanobisOOD:
    """
    Mahalanobis min distance sobre centroides por clase del training set.
    Usa covarianza diagonal (rapida, estable en alta dim) con floor de varianza.
    """

    def __init__(self, var_floor: float = 1e-4) -> None:
        self.means_: np.ndarray | None = None
        self.inv_var_: np.ndarray | None = None
        self.var_floor = var_floor

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int = 5) -> "MahalanobisOOD":
        means = np.stack([X[y == c].mean(axis=0) for c in range(num_classes)])
        var_global = X.var(axis=0) + self.var_floor
        self.means_ = means.astype(np.float32)
        self.inv_var_ = (1.0 / var_global).astype(np.float32)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None or self.inv_var_ is None:
            raise RuntimeError("MahalanobisOOD no ha sido fit().")
        diff = X[:, None, :] - self.means_[None, :, :]          # [B, C, D]
        d2 = (diff ** 2 * self.inv_var_[None, None, :]).sum(-1) # [B, C]
        return d2.min(axis=1)                                    # [B]


def combined_ood_score(
    router_probs: torch.Tensor,
    cls_tokens: torch.Tensor,
    mahal_detector: MahalanobisOOD,
    entropy_weight: float = 0.5,
    mahal_weight: float = 0.5,
) -> np.ndarray:
    H = LinearRouter.entropy(router_probs).detach().cpu().numpy()
    d = mahal_detector.score(cls_tokens.detach().cpu().numpy())
    d_norm = d / (np.median(d) + 1e-9)
    return entropy_weight * H + mahal_weight * d_norm

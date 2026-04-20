"""
CXRExpertSingleHead — wrapper del exp1 CXR14 v21.

Replica exactamente la arquitectura usada en el training (ver
`scripts/exp1_v21/train_exp1v21.py`):

  - Backbone: timm `features_only=True` sobre `convnextv2_base.fcmae_ft_in22k_in1k_384`
    con `out_indices=(last_idx,)`. Devuelve feature map (B, 1024, H/32, W/32).
  - Pool: LSEPool2d(r=10) — Log-Sum-Exp Pooling (Wang et al., CVPR 2017).
  - Head: Sequential(Dropout(0.3), Linear(1024, 6)).

Checkpoint EMA esperado (`backbone.stages_N.*`, `head.1.weight (6,1024)`); el loader
remapea `stem_N/stages_N/downsample_N` → `stem.N/stages.N/downsample.N` antes de cargar.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class LSEPool2d(nn.Module):
    """Log-Sum-Exp Pooling (Wang et al., CVPR 2017). r=10 optimo empirico CXR14."""

    def __init__(self, r: float = 10.0) -> None:
        super().__init__()
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        x_max = x_flat.max(dim=2, keepdim=True).values
        exp_sum = torch.exp(self.r * (x_flat - x_max)).mean(dim=2, keepdim=True)
        out = x_max + (1.0 / self.r) * torch.log(exp_sum + 1e-8)
        return out.view(b, c, 1, 1)


class CXRExpertSingleHead(nn.Module):
    def __init__(
        self,
        model_name: str = "convnextv2_base.fcmae_ft_in22k_in1k_384",
        num_classes: int = 6,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        _tmp = timm.create_model(model_name, pretrained=False, features_only=True)
        last_idx = len(_tmp.feature_info) - 1
        del _tmp
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
            out_indices=(last_idx,),
            drop_path_rate=drop_path,
        )
        feat_dim = self.backbone.feature_info[-1]["num_chs"]
        self.pool = LSEPool2d(r=10.0)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[0]                                    # (B, C, h, w)
        pooled = self.pool(feats).flatten(1)                           # (B, C)
        return self.head(pooled)

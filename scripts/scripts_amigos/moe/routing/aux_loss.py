"""
Auxiliary Loss del Switch Transformer (Fedus et al., 2021) para el router.

L_aux = alpha * N * sum_i(f_i * P_i)

  f_i = fraccion de muestras asignadas al experto i (hard, no diferenciable).
  P_i = probabilidad media asignada al experto i (soft, diferenciable).

Implementacion clave: f_i se computa sobre un buffer de ventana acumulada
(global-batch accumulated) y no sobre cada micro-batch individualmente.
Qiu et al. 2025 (arXiv:2501.11873) demuestra que calcular la LBL sobre el
global-batch es clave para especializacion real con desbalance severo
(en nuestro caso Pancreas 0.9% vs CXR14 70%).

Ver: investigaciones/2026-04-19-moe-profundizacion.md seccion 4.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SwitchAuxLoss(nn.Module):
    def __init__(
        self,
        num_experts: int = 5,
        alpha: float = 0.02,
        window_size: int = 1024,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.window_size = window_size
        self.register_buffer("expert_counts", torch.zeros(num_experts))

    def reset_buffer(self) -> None:
        self.expert_counts.zero_()

    @torch.no_grad()
    def _update_buffer(self, expert_idx: torch.Tensor) -> None:
        B = expert_idx.numel()
        batch_hist = torch.bincount(
            expert_idx.to(self.expert_counts.device),
            minlength=self.num_experts,
        ).float()
        decay = max(0.0, (self.window_size - B) / self.window_size)
        self.expert_counts.mul_(decay).add_(batch_hist)

    def forward(
        self,
        router_probs: torch.Tensor,
        expert_idx: torch.Tensor,
    ) -> torch.Tensor:
        self._update_buffer(expert_idx)
        total = self.expert_counts.sum()
        if total < 1e-9:
            return router_probs.sum() * 0.0
        f_i = self.expert_counts / total                  # [N]
        P_i = router_probs.mean(dim=0)                     # [N] diferenciable
        return self.alpha * self.num_experts * (f_i * P_i).sum()

    def current_load(self) -> dict:
        total = self.expert_counts.sum().item()
        if total < 1e-9:
            f_i = [0.0] * self.num_experts
        else:
            f_i = (self.expert_counts / total).tolist()
        ratio = max(f_i) / max(min(f_i), 1e-9) if total > 0 else 0.0
        return {
            "f_i": f_i,
            "max_over_min": ratio,
            "violates_rubric": ratio > 1.30,
        }

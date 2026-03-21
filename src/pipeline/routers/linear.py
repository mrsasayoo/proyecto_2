"""
Router A — Linear + Softmax (baseline deep learning).

En el ablation study: salida = N_EXPERTS_DOMAIN = 5 logits.
En FASE 1 real: salida = N_EXPERTS_TOTAL = 6 (slot OOD entrenado via L_error).
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from ..config import N_EXPERTS_DOMAIN, N_EXPERTS_TOTAL
from ..router_metrics import (
    per_expert_accuracy, log_per_expert,
    check_load_balance, calibrate_entropy_threshold,
)

log = logging.getLogger("fase1")


class LinearGatingHead(nn.Module):
    """
    Cabeza de gating lineal.
    n_experts=5  → ablation study (solo dominios clínicos)
    n_experts=6  → FASE 1 real   (incluye slot OOD entrenado via L_error)
    """
    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(z), dim=-1)


def train_linear_router(Z_train, y_train, Z_val, y_val, d_model,
                        epochs=50, lr=1e-3, batch_size=512):
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    model   = LinearGatingHead(d_model, N_EXPERTS_DOMAIN).to(device)
    opt     = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Z_t = torch.from_numpy(Z_train).float().to(device)
    y_t = torch.from_numpy(y_train).long().to(device)

    best_acc     = 0.0
    best_weights = None

    log.info(f"  [Linear] Entrenando en {device} | epochs={epochs} lr={lr} batch={batch_size}")
    log.info(f"  [Linear] n_experts={N_EXPERTS_DOMAIN} (ablation) | "
             f"nota: FASE 1 real usará n_experts={N_EXPERTS_TOTAL} (+OOD slot)")

    Z_v = torch.from_numpy(Z_val).float().to(device)

    for epoch in range(epochs):
        model.train()
        idx        = torch.randperm(len(Z_t))
        epoch_loss = 0.0

        for i in range(0, len(Z_t), batch_size):
            batch_idx = idx[i:i + batch_size]
            z_b, y_b  = Z_t[batch_idx], y_t[batch_idx]
            logits    = model.gate(z_b)
            loss      = loss_fn(logits, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                probs = model(Z_v).cpu().numpy()
                preds = probs.argmax(axis=1)
                acc   = accuracy_score(y_val, preds)

            improved = acc > best_acc
            if improved:
                best_acc     = acc
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}

            log.info(f"  [Linear] época {epoch+1:3d}/{epochs} | "
                     f"loss {epoch_loss:.4f} | val acc {acc:.4f}"
                     + (" ✓ mejor" if improved else ""))

            if epoch_loss < 1e-6:
                log.warning(f"  [Linear] Loss ≈ 0 en época {epoch+1} — posible overfitting "
                            f"o etiquetas con fuga de información.")

    if best_weights is None:
        log.error("[Linear] best_weights es None — ninguna época de validación mejoró. "
                  "Revisa que y_train tenga las 5 clases representadas.")
        best_weights = model.state_dict()

    model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        Z_v   = torch.from_numpy(Z_val).float().to(device)
        probs = model(Z_v).cpu().numpy()
        preds = probs.argmax(axis=1)

    per_exp   = per_expert_accuracy(y_val, preds)
    log_per_expert("Linear", per_exp)
    balance   = check_load_balance(preds, "Linear")
    threshold = calibrate_entropy_threshold(probs, y_val, "Linear")

    return model, best_acc, probs, balance, threshold

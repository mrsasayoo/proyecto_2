"""
Router A — Linear + Softmax (baseline deep learning).

Responsabilidad única: entrenar y devolver el router Linear + Softmax con
la Auxiliary Loss del Switch Transformer para forzar balance de carga.

En el ablation study: salida = N_EXPERTS_DOMAIN = 5 logits.
En Fase 1 real: salida = N_EXPERTS_TOTAL = 6 (slot OOD entrenado via L_error).
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from config import N_EXPERTS_DOMAIN, N_EXPERTS_TOTAL
from fase2_config import ALPHA_L_AUX, LINEAR_EPOCHS, LINEAR_LR, LINEAR_BATCH_SIZE
from router_metrics import (
    per_expert_accuracy,
    log_per_expert,
    check_load_balance,
    calibrate_entropy_threshold,
)

log = logging.getLogger("fase2")


class LinearGatingHead(nn.Module):
    """
    Cabeza de gating lineal.

    n_experts=5  → ablation study (solo dominios clínicos)
    n_experts=6  → Fase 1 real   (incluye slot OOD entrenado via L_error)
    """

    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(z), dim=-1)


def train_linear_router(
    Z_train,
    y_train,
    Z_val,
    y_val,
    d_model,
    epochs=LINEAR_EPOCHS,
    lr=LINEAR_LR,
    batch_size=LINEAR_BATCH_SIZE,
    alpha=ALPHA_L_AUX,
    Z_test=None,
    y_test=None,
):
    """
    Entrena el router Linear + Softmax sobre embeddings congelados.

    Loss total = CrossEntropy + L_aux (Switch Transformer auxiliary loss).

    L_aux = α · N · Σ f_i · P_i
      donde:
        α   = ALPHA_L_AUX de fase2_config.py (default 0.01)
        N   = N_EXPERTS_DOMAIN = 5
        f_i = fracción real de muestras del batch asignadas al experto i
              (distribución empírica calculada sobre las etiquetas del batch)
        P_i = probabilidad media que el router asigna al experto i sobre el batch
              (calculada como softmax(logits).mean(dim=0))

    L_aux penaliza la concentración de carga en pocos expertos: si el router
    aprende a enviar todo a un experto, f_winner ≈ 1 y P_winner ≈ 1, haciendo
    L_aux ≈ α · N · 1. Si la carga está perfectamente balanceada, f_i = 1/N
    para todos, haciendo L_aux = α (el valor mínimo).

    El modelo retornado tiene sus pesos restaurados al mejor checkpoint de
    validación encontrado durante el entrenamiento.

    Parameters
    ----------
    Z_train, y_train : np.ndarray
    Z_val, y_val     : np.ndarray
    d_model          : int    — dimensión del embedding
    epochs           : int    — épocas de entrenamiento
    lr               : float  — learning rate del optimizador Adam
    batch_size       : int    — batch size
    alpha            : float  — coeficiente α de L_aux
    Z_test, y_test   : np.ndarray | None — si se proporcionan, calcula acc_test

    Returns
    -------
    (model, best_acc, probs_val, balance, threshold, acc_test)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LinearGatingHead(d_model, N_EXPERTS_DOMAIN).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Z_t = torch.from_numpy(Z_train).float().to(device)
    y_t = torch.from_numpy(y_train).long().to(device)
    Z_v = torch.from_numpy(Z_val).float().to(device)

    best_acc = 0.0
    best_weights = None

    log.info(
        "  [Linear] Entrenando en %s | epochs=%d lr=%g batch=%d alpha_aux=%g",
        device,
        epochs,
        lr,
        batch_size,
        alpha,
    )
    log.info(
        "  [Linear] n_experts=%d (ablation) | "
        "nota: Fase 1 real usará n_experts=%d (+OOD slot)",
        N_EXPERTS_DOMAIN,
        N_EXPERTS_TOTAL,
    )

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(Z_t))
        epoch_loss = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_aux = 0.0

        for i in range(0, len(Z_t), batch_size):
            batch_idx = idx[i : i + batch_size]
            z_b, y_b = Z_t[batch_idx], y_t[batch_idx]
            logits = model.gate(z_b)
            loss_ce = loss_fn(logits, y_b)

            # ── L_aux: Switch Transformer auxiliary loss ──────────────
            # f_i: fracción de muestras del batch por experto (distribución empírica)
            f_i = torch.zeros(N_EXPERTS_DOMAIN, device=device)
            for exp_id in range(N_EXPERTS_DOMAIN):
                f_i[exp_id] = (y_b == exp_id).float().sum() / len(y_b)

            # P_i: probabilidad media del router por experto sobre el batch
            probs_batch = torch.softmax(logits, dim=-1)  # [B, N_EXPERTS_DOMAIN]
            P_i = probs_batch.mean(dim=0)  # [N_EXPERTS_DOMAIN]

            # L_aux Switch Transformer: L_aux = α · N · Σ f_i · P_i
            L_aux = alpha * N_EXPERTS_DOMAIN * (f_i * P_i).sum()
            loss = loss_ce + L_aux

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            epoch_loss_ce += loss_ce.item()
            epoch_loss_aux += L_aux.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                probs = model(Z_v).cpu().numpy()
                preds = probs.argmax(axis=1)
                acc = accuracy_score(y_val, preds)

            improved = acc > best_acc
            if improved:
                best_acc = acc
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}

            log.info(
                "  [Linear] época %3d/%d | loss_ce %.4f | L_aux %.4f | "
                "loss_total %.4f | val acc %.4f%s",
                epoch + 1,
                epochs,
                epoch_loss_ce,
                epoch_loss_aux,
                epoch_loss,
                acc,
                " ✓ mejor" if improved else "",
            )

            if epoch_loss < 1e-6:
                log.warning(
                    "  [Linear] Loss ≈ 0 en época %d — posible overfitting "
                    "o etiquetas con fuga de información.",
                    epoch + 1,
                )

    if best_weights is None:
        log.error(
            "[Linear] best_weights es None — ninguna época de validación mejoró. "
            "Revisa que y_train tenga las %d clases representadas.",
            N_EXPERTS_DOMAIN,
        )
        best_weights = model.state_dict()

    model.load_state_dict(best_weights)

    # ── Evaluación final sobre val ───────────────────────────────────────
    model.eval()
    with torch.no_grad():
        Z_v = torch.from_numpy(Z_val).float().to(device)
        probs = model(Z_v).cpu().numpy()
        preds = probs.argmax(axis=1)

    per_exp = per_expert_accuracy(y_val, preds)
    log_per_expert("Linear", per_exp)
    balance = check_load_balance(preds, "Linear")
    threshold = calibrate_entropy_threshold(probs, y_val, "Linear")

    # ── Evaluación sobre test (si disponible) ────────────────────────────
    acc_test = None
    if Z_test is not None and y_test is not None and len(Z_test) > 0:
        with torch.no_grad():
            Z_te = torch.from_numpy(Z_test).float().to(device)
            test_probs = model(Z_te).cpu().numpy()
            test_preds = test_probs.argmax(axis=1)
            acc_test = float(accuracy_score(y_test, test_preds))
        log.info("  [Linear] Routing Accuracy test: %.4f", acc_test)

    return model, best_acc, probs, balance, threshold, acc_test

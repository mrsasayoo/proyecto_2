"""
Router B — GMM (paramétrico EM).

Ajusta N_EXPERTS_DOMAIN componentes gaussianas sin supervisión,
luego mapea cada componente al experto de dominio por voto mayoritario.
"""

import logging

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

from ..config import N_EXPERTS_DOMAIN, EXPERT_NAMES
from ..router_metrics import (
    per_expert_accuracy, log_per_expert,
    check_load_balance, calibrate_entropy_threshold,
)

log = logging.getLogger("fase1")


def train_gmm_router(Z_train, y_train, Z_val, y_val):
    log.info(f"  [GMM] Ajustando GaussianMixture "
             f"({N_EXPERTS_DOMAIN} comp., full covariance) ...")

    cov_type = "full"
    try:
        gmm = GaussianMixture(
            n_components=N_EXPERTS_DOMAIN,
            covariance_type="full",
            max_iter=200,
            random_state=42,
            verbose=0
        )
        gmm.fit(Z_train)
        if not gmm.converged_:
            log.warning("  [GMM] EM no convergió en 200 iteraciones con full covariance. "
                        "Considera aumentar max_iter o cambiar a 'diag'.")
    except Exception as e:
        log.warning(f"  [GMM] full covariance falló ({e}) → reintentando con 'diag'")
        cov_type = "diag"
        gmm = GaussianMixture(
            n_components=N_EXPERTS_DOMAIN,
            covariance_type="diag",
            max_iter=200,
            random_state=42
        )
        gmm.fit(Z_train)
        if not gmm.converged_:
            log.warning("  [GMM] EM tampoco convergió con 'diag'. Resultados poco confiables.")

    log.debug(f"  [GMM] covariance_type usado: '{cov_type}'")

    # Mapeo componente → experto de dominio por voto mayoritario
    train_comp     = gmm.predict(Z_train)
    comp_to_expert = {}
    for comp in range(N_EXPERTS_DOMAIN):
        mask = train_comp == comp
        if mask.sum() == 0:
            log.warning(f"  [GMM] Componente {comp} vacía en train — mapeada a experto 0.")
            comp_to_expert[comp] = 0
            continue
        labels = y_train[mask]
        winner = int(np.bincount(labels, minlength=N_EXPERTS_DOMAIN).argmax())
        comp_to_expert[comp] = winner

    # Verificar que todos los expertos de dominio tienen al menos una componente asignada
    assigned_experts = set(comp_to_expert.values())
    unassigned = set(range(N_EXPERTS_DOMAIN)) - assigned_experts
    if unassigned:
        log.warning(f"  [GMM] Expertos sin componente asignada: "
                    f"{[EXPERT_NAMES[e] for e in unassigned]}. "
                    f"El GMM nunca rutará a estos expertos.")

    log.info(f"  [GMM] Mapeo componente→experto: "
             + " | ".join(f"C{c}→{EXPERT_NAMES[e]}" for c, e in comp_to_expert.items()))

    val_comp  = gmm.predict(Z_val)
    val_preds = np.array([comp_to_expert[c] for c in val_comp])
    acc       = accuracy_score(y_val, val_preds)
    log.info(f"  [GMM] Routing Accuracy val: {acc:.4f}")

    val_probs    = gmm.predict_proba(Z_val)
    per_exp      = per_expert_accuracy(y_val, val_preds)
    log_per_expert("GMM", per_exp)
    balance      = check_load_balance(val_preds, "GMM")
    threshold    = calibrate_entropy_threshold(val_probs, y_val, "GMM")

    return gmm, comp_to_expert, acc, val_probs, balance, threshold

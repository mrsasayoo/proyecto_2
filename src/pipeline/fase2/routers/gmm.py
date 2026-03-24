"""
Router B — GMM (paramétrico EM sin supervisión).

Responsabilidad única: ajustar una mezcla gaussiana no supervisada sobre los
embeddings de entrenamiento, mapear cada componente al experto de dominio
mayoritario, y devolver predicciones y probabilidades comparables.

Decisión de diseño — supervisión indirecta: el GMM se ajusta sin etiquetas
(EM puro). Las etiquetas solo se usan después del ajuste, para asignar cada
componente al experto cuyas muestras son mayoría (voto mayoritario). Esta
indirección evalúa si el espacio de embeddings tiene estructura geométrica
suficiente para separar los dominios clínicos sin supervisión.
"""

import logging

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

from config import N_EXPERTS_DOMAIN, EXPERT_NAMES
from fase2_config import GMM_COV_TYPE, GMM_MAX_ITER
from router_metrics import (
    per_expert_accuracy,
    log_per_expert,
    check_load_balance,
    calibrate_entropy_threshold,
)

log = logging.getLogger("fase2")


def train_gmm_router(Z_train, y_train, Z_val, y_val, Z_test=None, y_test=None):
    """
    Entrena el router GMM sobre embeddings congelados.

    Ajusta N_EXPERTS_DOMAIN componentes gaussianas sin supervisión (EM),
    luego mapea cada componente al experto de dominio por voto mayoritario.

    Las probabilidades de componente (gmm.predict_proba) se convierten a
    probabilidades de experto sumando las responsabilidades de todas las
    componentes mapeadas al mismo experto, y normalizando por fila.

    Reasignación greedy: si dos componentes se asignan al mismo experto
    y otro queda sin cobertura, se reasigna la componente con más muestras
    del experto sin cobertura. Esto previene que el GMM sea descalificado
    por un artefacto del algoritmo EM.

    Returns
    -------
    (gmm, comp_to_expert, acc, expert_probs, balance, threshold, cov_type, acc_test)
    """
    log.info(
        "  [GMM] Ajustando GaussianMixture (%d comp., %s covariance) ...",
        N_EXPERTS_DOMAIN,
        GMM_COV_TYPE,
    )

    cov_type = GMM_COV_TYPE
    try:
        gmm = GaussianMixture(
            n_components=N_EXPERTS_DOMAIN,
            covariance_type=GMM_COV_TYPE,
            max_iter=GMM_MAX_ITER,
            random_state=42,
            verbose=0,
        )
        gmm.fit(Z_train)
        if not gmm.converged_:
            log.warning(
                "  [GMM] EM no convergió en %d iteraciones con %s covariance. "
                "Considera aumentar GMM_MAX_ITER en fase2_config.py o cambiar a 'diag'.",
                GMM_MAX_ITER,
                GMM_COV_TYPE,
            )
    except Exception as e:
        log.warning(
            "  [GMM] %s covariance falló (%s) → reintentando con 'diag'",
            GMM_COV_TYPE,
            e,
        )
        cov_type = "diag"
        gmm = GaussianMixture(
            n_components=N_EXPERTS_DOMAIN,
            covariance_type="diag",
            max_iter=GMM_MAX_ITER,
            random_state=42,
        )
        gmm.fit(Z_train)
        if not gmm.converged_:
            log.warning(
                "  [GMM] EM tampoco convergió con 'diag'. Resultados poco confiables."
            )

    log.debug("  [GMM] covariance_type usado: '%s'", cov_type)

    # ── Mapeo componente → experto de dominio por voto mayoritario ───────
    train_comp = gmm.predict(Z_train)
    comp_to_expert = {}
    for comp in range(N_EXPERTS_DOMAIN):
        mask = train_comp == comp
        if mask.sum() == 0:
            log.warning(
                "  [GMM] Componente %d vacía en train — mapeada a experto 0.", comp
            )
            comp_to_expert[comp] = 0
            continue
        labels = y_train[mask]
        winner = int(np.bincount(labels, minlength=N_EXPERTS_DOMAIN).argmax())
        comp_to_expert[comp] = winner

    # ── Reasignación greedy si hay expertos sin cobertura ────────────────
    covered_experts = set(comp_to_expert.values())
    uncovered_experts = [e for e in range(N_EXPERTS_DOMAIN) if e not in covered_experts]

    if uncovered_experts:
        log.warning(
            "  [GMM] Expertos sin componente: %s. Aplicando reasignación greedy.",
            uncovered_experts,
        )
        for unc_exp in uncovered_experts:
            best_comp = max(
                range(N_EXPERTS_DOMAIN),
                key=lambda c: (y_train[train_comp == c] == unc_exp).sum(),
            )
            comp_to_expert[best_comp] = unc_exp
            log.warning(
                "  [GMM] Experto %d → componente %d (reasignación forzada)",
                unc_exp,
                best_comp,
            )

    log.info(
        "  [GMM] Mapeo final componente→experto: %s",
        " | ".join(f"C{c}→{EXPERT_NAMES[e]}" for c, e in comp_to_expert.items()),
    )

    # ── Evaluación sobre val ─────────────────────────────────────────────
    val_comp = gmm.predict(Z_val)
    val_preds = np.array([comp_to_expert[c] for c in val_comp])
    acc = accuracy_score(y_val, val_preds)
    log.info("  [GMM] Routing Accuracy val: %.4f", acc)

    # ── Convertir probabilidades de componente → probabilidades de experto ─
    comp_probs = gmm.predict_proba(Z_val)  # [N, n_components]
    expert_probs = np.zeros((len(Z_val), N_EXPERTS_DOMAIN), dtype=np.float64)
    for comp_id, exp_id in comp_to_expert.items():
        expert_probs[:, exp_id] += comp_probs[:, comp_id]
    row_sums = np.maximum(expert_probs.sum(axis=1, keepdims=True), 1e-12)
    expert_probs = expert_probs / row_sums

    # Verificar consistencia entre mapeo y argmax de probabilidades
    expert_preds = expert_probs.argmax(axis=1)
    mismatch = (expert_preds != val_preds).sum()
    if mismatch > 0:
        log.warning(
            "  [GMM] %d muestras con predicción inconsistente "
            "entre comp→expert mapping y expert_probs.argmax. "
            "Usando expert_probs.argmax para consistencia.",
            mismatch,
        )
        val_preds = expert_preds

    per_exp = per_expert_accuracy(y_val, val_preds)
    log_per_expert("GMM", per_exp)
    balance = check_load_balance(val_preds, "GMM")
    threshold = calibrate_entropy_threshold(expert_probs, y_val, "GMM")

    # ── Evaluación sobre test (si disponible) ────────────────────────────
    acc_test = None
    if Z_test is not None and y_test is not None and len(Z_test) > 0:
        test_comp = gmm.predict(Z_test)
        test_preds = np.array([comp_to_expert[c] for c in test_comp])
        acc_test = float(accuracy_score(y_test, test_preds))
        log.info("  [GMM] Routing Accuracy test: %.4f", acc_test)

    return (
        gmm,
        comp_to_expert,
        acc,
        expert_probs,
        balance,
        threshold,
        cov_type,
        acc_test,
    )

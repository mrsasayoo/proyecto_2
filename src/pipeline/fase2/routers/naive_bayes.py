"""
Router C — Naive Bayes (paramétrico MLE analítico).

Responsabilidad única: ajustar un clasificador Naive Bayes gaussiano sobre
los embeddings de entrenamiento y devolver predicciones y probabilidades
evaluadas sobre el set de validación.

Por qué como baseline: GaussianNB tiene solución analítica (MLE de medias y
varianzas por clase). No requiere iteraciones de EM, no requiere convergencia,
no requiere GPU. Si su accuracy es competitivo con el GMM pero su tiempo de
entrenamiento es 100× menor, el tradeoff puede ser favorable en contextos de
recursos limitados.
"""

import logging

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from config import N_EXPERTS_DOMAIN, EXPERT_NAMES
from router_metrics import (
    per_expert_accuracy,
    log_per_expert,
    check_load_balance,
    calibrate_entropy_threshold,
)

log = logging.getLogger("fase2")


def train_nb_router(Z_train, y_train, Z_val, y_val, Z_test=None, y_test=None):
    """
    Entrena el router Naive Bayes (GaussianNB, MLE analítico).

    Parameters
    ----------
    Z_train, y_train : np.ndarray
    Z_val, y_val     : np.ndarray
    Z_test, y_test   : np.ndarray | None — si se proporcionan, calcula acc_test

    Returns
    -------
    (nb_model, acc, val_probs, balance, threshold, acc_test)
    """
    log.info("  [NB] Ajustando GaussianNB (MLE analítico)...")

    classes_present = np.unique(y_train)
    missing = set(range(N_EXPERTS_DOMAIN)) - set(classes_present.tolist())
    if missing:
        log.warning(
            "  [NB] Clases ausentes en train: %s. "
            "GaussianNB no podrá aprender estos expertos.",
            [EXPERT_NAMES[m] for m in missing],
        )

    nb = GaussianNB()
    nb.fit(Z_train, y_train)

    val_preds = nb.predict(Z_val)
    val_probs = nb.predict_proba(Z_val)
    acc = accuracy_score(y_val, val_preds)
    log.info("  [NB] Routing Accuracy val: %.4f", acc)

    per_exp = per_expert_accuracy(y_val, val_preds)
    log_per_expert("NB", per_exp)
    balance = check_load_balance(val_preds, "NB")
    threshold = calibrate_entropy_threshold(val_probs, y_val, "NB")

    # ── Evaluación sobre test (si disponible) ────────────────────────────
    acc_test = None
    if Z_test is not None and y_test is not None and len(Z_test) > 0:
        test_preds = nb.predict(Z_test)
        acc_test = float(accuracy_score(y_test, test_preds))
        log.info("  [NB] Routing Accuracy test: %.4f", acc_test)

    return nb, acc, val_probs, balance, threshold, acc_test

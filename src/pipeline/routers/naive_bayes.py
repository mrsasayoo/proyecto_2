"""
Router C — Naive Bayes (paramétrico MLE analítico).

GaussianNB estima medias y varianzas por clase de forma analítica.
"""

import logging

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from ..config import N_EXPERTS_DOMAIN, EXPERT_NAMES
from ..router_metrics import (
    per_expert_accuracy, log_per_expert,
    check_load_balance, calibrate_entropy_threshold,
)

log = logging.getLogger("fase1")


def train_nb_router(Z_train, y_train, Z_val, y_val):
    log.info("  [NB] Ajustando GaussianNB (MLE analítico)...")

    classes_present = np.unique(y_train)
    missing = set(range(N_EXPERTS_DOMAIN)) - set(classes_present.tolist())
    if missing:
        log.warning(f"  [NB] Clases ausentes en train: "
                    f"{[EXPERT_NAMES[m] for m in missing]}. "
                    f"GaussianNB no podrá aprender estos expertos.")

    nb = GaussianNB()
    nb.fit(Z_train, y_train)

    val_preds = nb.predict(Z_val)
    val_probs = nb.predict_proba(Z_val)
    acc       = accuracy_score(y_val, val_preds)
    log.info(f"  [NB] Routing Accuracy val: {acc:.4f}")

    per_exp   = per_expert_accuracy(y_val, val_preds)
    log_per_expert("NB", per_exp)
    balance   = check_load_balance(val_preds, "NB")
    threshold = calibrate_entropy_threshold(val_probs, y_val, "NB")

    return nb, acc, val_probs, balance, threshold

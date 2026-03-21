"""
Router D — k-NN con FAISS (no paramétrico, distancia coseno).

Normalización L2 + IndexFlatIP para similitud coseno.
Probabilidades suaves construidas por fracción de votos.
"""

import logging

import faiss
import numpy as np
from sklearn.metrics import accuracy_score

from ..config import N_EXPERTS_DOMAIN
from ..router_metrics import (
    per_expert_accuracy, log_per_expert,
    check_load_balance, calibrate_entropy_threshold,
)

log = logging.getLogger("fase1")


def train_knn_router(Z_train, y_train, Z_val, y_val, k=5):
    log.info(f"  [kNN] Construyendo índice FAISS coseno (k={k})...")
    d = Z_train.shape[1]

    Z_t_norm = Z_train.copy().astype(np.float32)
    faiss.normalize_L2(Z_t_norm)
    Z_v_norm = Z_val.copy().astype(np.float32)
    faiss.normalize_L2(Z_v_norm)

    zero_train = (np.linalg.norm(Z_train, axis=1) < 1e-9).sum()
    zero_val   = (np.linalg.norm(Z_val,   axis=1) < 1e-9).sum()
    if zero_train or zero_val:
        log.error(f"  [kNN] Vectores cero detectados: "
                  f"{zero_train} en train, {zero_val} en val. "
                  f"normalize_L2 producirá NaN. Verifica la extracción de embeddings.")

    index = faiss.IndexFlatIP(d)
    index.add(Z_t_norm)
    log.debug(f"  [kNN] Índice construido con {index.ntotal:,} vectores")

    distances, I = index.search(Z_v_norm, k)

    min_sim = distances.max(axis=1)
    low_sim_count = (min_sim < 0.5).sum()
    if low_sim_count > 0:
        log.info(f"  [kNN] {low_sim_count} muestras de val con similitud coseno < 0.5 "
                 f"con su vecino más cercano — candidatas naturales a OOD.")

    neighbor_labels = y_train[I]
    val_preds = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=N_EXPERTS_DOMAIN).argmax(),
        axis=1, arr=neighbor_labels
    )

    val_probs = np.zeros((len(Z_val), N_EXPERTS_DOMAIN), dtype=np.float32)
    for i, row in enumerate(neighbor_labels):
        counts = np.bincount(row, minlength=N_EXPERTS_DOMAIN)
        val_probs[i] = counts / counts.sum()

    acc = accuracy_score(y_val, val_preds)
    log.info(f"  [kNN] Routing Accuracy val: {acc:.4f}")

    per_exp   = per_expert_accuracy(y_val, val_preds)
    log_per_expert("kNN", per_exp)
    balance   = check_load_balance(val_preds, "kNN")
    threshold = calibrate_entropy_threshold(val_probs, y_val, "kNN")

    return index, y_train, acc, val_probs, balance, threshold

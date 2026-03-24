"""
Router D — k-NN con FAISS (no paramétrico, distancia coseno).

Responsabilidad única: construir un índice FAISS de similitud coseno sobre
los embeddings de entrenamiento y evaluar el routing por votación de k
vecinos más cercanos.

Decisión de diseño — similitud coseno: los embeddings de un backbone ViT
están en un espacio de alta dimensión donde la distancia euclidea sufre el
"curse of dimensionality". La similitud coseno normaliza las magnitudes y
mide solo la orientación, que es la representación apropiada para embeddings
de transformers donde la magnitud no tiene semántica clínica.

Suavizado de Laplace: con k=5, las probabilidades de voto sin suavizar son
{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, produciendo entropías artificialmente
discretas que hacen el umbral OOD no comparable con los otros routers
(continuos). LAPLACE_EPSILON de fase2_config.py hace las probabilidades
continuas sin alterar materialmente las predicciones.
"""

import logging

import faiss
import numpy as np
from sklearn.metrics import accuracy_score

from config import N_EXPERTS_DOMAIN
from fase2_config import KNN_K, LAPLACE_EPSILON
from router_metrics import (
    per_expert_accuracy,
    log_per_expert,
    check_load_balance,
    calibrate_entropy_threshold,
)

log = logging.getLogger("fase2")


def train_knn_router(
    Z_train,
    y_train,
    Z_val,
    y_val,
    k=KNN_K,
    Z_test=None,
    y_test=None,
    epsilon=LAPLACE_EPSILON,
):
    """
    Construye el índice kNN-FAISS (coseno) y evalúa routing.

    Parameters
    ----------
    Z_train, y_train : np.ndarray
    Z_val, y_val     : np.ndarray
    k                : int   — número de vecinos (default: KNN_K de fase2_config.py)
    Z_test, y_test   : np.ndarray | None — si se proporcionan, calcula acc_test
    epsilon          : float — suavizado de Laplace (default: LAPLACE_EPSILON)

    Returns
    -------
    (index, y_train, acc, val_probs, balance, threshold, acc_test)
        El índice FAISS y las etiquetas de entrenamiento se devuelven juntos
        porque son inseparables para la inferencia (el índice solo devuelve
        distancias e índices, no etiquetas).
    """
    log.info("  [kNN] Construyendo índice FAISS coseno (k=%d)...", k)
    d = Z_train.shape[1]

    Z_t_norm = Z_train.copy().astype(np.float32)
    faiss.normalize_L2(Z_t_norm)
    Z_v_norm = Z_val.copy().astype(np.float32)
    faiss.normalize_L2(Z_v_norm)

    zero_train = (np.linalg.norm(Z_train, axis=1) < 1e-9).sum()
    zero_val = (np.linalg.norm(Z_val, axis=1) < 1e-9).sum()
    if zero_train or zero_val:
        log.error(
            "  [kNN] Vectores cero detectados: %d en train, %d en val. "
            "normalize_L2 producirá NaN. Verifica la extracción de embeddings.",
            zero_train,
            zero_val,
        )

    index = faiss.IndexFlatIP(d)
    index.add(Z_t_norm)
    log.debug("  [kNN] Índice construido con %d vectores", index.ntotal)

    distances, I = index.search(Z_v_norm, k)

    best_sim = distances.max(axis=1)
    low_sim_count = (best_sim < 0.5).sum()
    if low_sim_count > 0:
        log.info(
            "  [kNN] %d muestras de val con similitud coseno < 0.5 "
            "con su vecino más cercano — candidatas naturales a OOD.",
            low_sim_count,
        )

    neighbor_labels = y_train[I]
    val_preds = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=N_EXPERTS_DOMAIN).argmax(),
        axis=1,
        arr=neighbor_labels,
    )

    # Suavizado de Laplace para entropías continuas y comparables
    val_probs = np.zeros((len(Z_val), N_EXPERTS_DOMAIN), dtype=np.float32)
    for i, row in enumerate(neighbor_labels):
        counts = np.bincount(row, minlength=N_EXPERTS_DOMAIN).astype(np.float32)
        counts_smooth = counts + epsilon
        val_probs[i] = counts_smooth / counts_smooth.sum()

    acc = accuracy_score(y_val, val_preds)
    log.info("  [kNN] Routing Accuracy val: %.4f", acc)

    per_exp = per_expert_accuracy(y_val, val_preds)
    log_per_expert("kNN", per_exp)
    balance = check_load_balance(val_preds, "kNN")
    threshold = calibrate_entropy_threshold(val_probs, y_val, "kNN")

    # ── Evaluación sobre test (si disponible) ────────────────────────────
    acc_test = None
    if Z_test is not None and y_test is not None and len(Z_test) > 0:
        Z_te_norm = Z_test.copy().astype(np.float32)
        faiss.normalize_L2(Z_te_norm)
        _, I_test = index.search(Z_te_norm, k)
        test_neighbor_labels = y_train[I_test]
        test_preds = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=N_EXPERTS_DOMAIN).argmax(),
            axis=1,
            arr=test_neighbor_labels,
        )
        acc_test = float(accuracy_score(y_test, test_preds))
        log.info("  [kNN] Routing Accuracy test: %.4f", acc_test)

    return index, y_train, acc, val_probs, balance, threshold, acc_test

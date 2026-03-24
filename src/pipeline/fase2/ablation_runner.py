"""
ablation_runner.py — Ejecución del ablation study.

Responsabilidad única: ejecutar los cuatro routers del ablation en secuencia,
recopilar sus resultados en un dict uniforme, y devolver ese dict al módulo
de reporte.

El orquestador (fase2_pipeline.py) no necesita conocer los detalles de
entrenamiento de cada router. Si se añade un quinto router, solo se modifica
este módulo y el __init__.py de routers/. El orquestador y el reportero no
necesitan cambios.

Este módulo NO escribe ningún archivo en disco — esa es la responsabilidad
de ablation_reporter.py.
"""

import logging
import time

import numpy as np
import torch

from config import N_EXPERTS_DOMAIN
from fase2_config import (
    LINEAR_EPOCHS,
    LINEAR_LR,
    LINEAR_BATCH_SIZE,
    ALPHA_L_AUX,
    KNN_K,
)
from router_metrics import measure_latency
from routers import (
    train_linear_router,
    train_gmm_router,
    train_nb_router,
    train_knn_router,
)

log = logging.getLogger("fase2")


def run_ablation(data: dict, args) -> dict:
    """
    Ejecuta los cuatro routers del ablation study y devuelve los resultados.

    Parameters
    ----------
    data : dict
        Dict devuelto por embeddings_loader.load_embeddings(). Claves:
        Z_train, y_train, Z_val, y_val, Z_test, y_test, d_model, has_test.
    args : argparse.Namespace
        Argumentos CLI. Se usan: args.epochs, args.l_aux_alpha, args.knn_k.

    Returns
    -------
    dict
        {
          "Linear":     {acc, acc_test, train_time_s, latency_ms, params,
                         needs_gpu, load_balance, entropy_thresh, model},
          "GMM":        {...},
          "NaiveBayes": {...},
          "kNN-FAISS":  {...},
        }
        No escribe ningún archivo en disco.
    """
    Z_train = data["Z_train"]
    y_train = data["y_train"]
    Z_val = data["Z_val"]
    y_val = data["y_val"]
    Z_test = data["Z_test"] if data["has_test"] else None
    y_test = data["y_test"] if data["has_test"] else None
    d_model = data["d_model"]

    # Muestras de val para medición de latencia (batch fijo de 32)
    sample_np = Z_val[:32].astype(np.float32)

    results = {}

    # ── A) LINEAR ────────────────────────────────────────────────────────
    log.info("\n[A] Entrenando Linear + Softmax (baseline DL)...")
    t0 = time.time()
    (
        linear_model,
        linear_acc,
        linear_probs,
        linear_balance,
        linear_thresh,
        linear_acc_test,
    ) = train_linear_router(
        Z_train,
        y_train,
        Z_val,
        y_val,
        d_model,
        epochs=getattr(args, "epochs", LINEAR_EPOCHS),
        lr=LINEAR_LR,
        batch_size=LINEAR_BATCH_SIZE,
        alpha=getattr(args, "l_aux_alpha", ALPHA_L_AUX),
        Z_test=Z_test,
        y_test=y_test,
    )
    linear_time = time.time() - t0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_t = torch.from_numpy(sample_np).float().to(device)
    linear_latency = measure_latency(lambda: linear_model(sample_t))

    results["Linear"] = {
        "acc": linear_acc,
        "acc_test": linear_acc_test,
        "train_time_s": linear_time,
        "latency_ms": linear_latency,
        "params": d_model * N_EXPERTS_DOMAIN + N_EXPERTS_DOMAIN,
        "needs_gpu": True,
        "load_balance": linear_balance,
        "entropy_thresh": linear_thresh,
        "model": linear_model,
    }

    # ── B) GMM ───────────────────────────────────────────────────────────
    log.info("\n[B] Entrenando GMM (paramétrico EM)...")
    t0 = time.time()
    (
        gmm_model,
        gmm_map,
        gmm_acc,
        gmm_probs,
        gmm_balance,
        gmm_thresh,
        gmm_cov_type,
        gmm_acc_test,
    ) = train_gmm_router(Z_train, y_train, Z_val, y_val, Z_test=Z_test, y_test=y_test)
    gmm_time = time.time() - t0

    gmm_latency = measure_latency(lambda: gmm_model.predict_proba(sample_np))

    if gmm_cov_type == "full":
        gmm_params = N_EXPERTS_DOMAIN * (d_model + d_model * d_model)
    else:
        gmm_params = N_EXPERTS_DOMAIN * 2 * d_model

    results["GMM"] = {
        "acc": gmm_acc,
        "acc_test": gmm_acc_test,
        "train_time_s": gmm_time,
        "latency_ms": gmm_latency,
        "params": gmm_params,
        "needs_gpu": False,
        "load_balance": gmm_balance,
        "entropy_thresh": gmm_thresh,
        "model": (gmm_model, gmm_map),
    }

    # ── C) NAIVE BAYES ───────────────────────────────────────────────────
    log.info("\n[C] Entrenando Naive Bayes (MLE analítico)...")
    t0 = time.time()
    nb_model, nb_acc, nb_probs, nb_balance, nb_thresh, nb_acc_test = train_nb_router(
        Z_train, y_train, Z_val, y_val, Z_test=Z_test, y_test=y_test
    )
    nb_time = time.time() - t0

    nb_latency = measure_latency(lambda: nb_model.predict_proba(sample_np))

    results["NaiveBayes"] = {
        "acc": nb_acc,
        "acc_test": nb_acc_test,
        "train_time_s": nb_time,
        "latency_ms": nb_latency,
        "params": N_EXPERTS_DOMAIN * 2 * d_model,
        "needs_gpu": False,
        "load_balance": nb_balance,
        "entropy_thresh": nb_thresh,
        "model": nb_model,
    }

    # ── D) kNN FAISS ─────────────────────────────────────────────────────
    log.info("\n[D] Construyendo índice kNN-FAISS (no paramétrico)...")
    t0 = time.time()
    knn_index, knn_labels, knn_acc, knn_probs, knn_balance, knn_thresh, knn_acc_test = (
        train_knn_router(
            Z_train,
            y_train,
            Z_val,
            y_val,
            k=getattr(args, "knn_k", KNN_K),
            Z_test=Z_test,
            y_test=y_test,
        )
    )
    knn_time = time.time() - t0

    try:
        import faiss as _faiss

        sample_norm = sample_np.copy()
        _faiss.normalize_L2(sample_norm)
        knn_latency = measure_latency(
            lambda: knn_index.search(sample_norm, getattr(args, "knn_k", KNN_K))
        )
    except ImportError:
        log.warning("[kNN] faiss no disponible para medición de latencia — saltando.")
        knn_latency = float("nan")

    results["kNN-FAISS"] = {
        "acc": knn_acc,
        "acc_test": knn_acc_test,
        "train_time_s": knn_time,
        "latency_ms": knn_latency,
        "params": int(len(Z_train)) * d_model,
        "needs_gpu": False,
        "load_balance": knn_balance,
        "entropy_thresh": knn_thresh,
        "model": (knn_index, knn_labels),
    }

    return results

"""
FASE 1 — Ablation Study del Router
====================================
Proyecto MoE — Incorporar Elementos de IA, Unidad II

Lee Z_train.npy / Z_val.npy generados por FASE 0 y compara los 4 mecanismos
de routing sobre el mismo espacio de embeddings.

Arquitectura MoE (6 expertos — diseño propio):
  Expertos de dominio (ID 0–4) — targets del ablation study:
    0 → NIH ChestXray14   4 → Pancreas PANORAMA
    1 → ISIC 2019         5 → OOD/Error (ver nota abajo)
    2 → OA Rodilla
    3 → LUNA16

  Experto 5 (OOD/Error) — NO es un target de routing del ablation study.
    El router aprende a enviar imágenes a los expertos 0–4. El Experto 5 se
    activa en inferencia por umbral de entropía: H(g) ≥ ENTROPY_THRESHOLD.
    Este script calibra ese umbral analizando H(g) en el set de validación.

  LinearGatingHead en el ablation: salida = N_EXPERTS_DOMAIN = 5 logits.
  LinearGatingHead en FASE 1 real: salida = N_EXPERTS_TOTAL  = 6 logits
    (el slot 5 recibe L_error durante el entrenamiento del router).

Mecanismos comparados (sección 4.2 del proyecto):
  A) ViT + Linear + Softmax  — paramétrico, gradiente
  B) ViT + GMM               — paramétrico, EM
  C) ViT + Naive Bayes       — paramétrico, MLE analítico
  D) ViT + k-NN FAISS        — no paramétrico, distancia coseno

Uso:
  python fase1_ablation_router.py --embeddings ./embeddings --epochs 50
"""

import argparse
import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from .config import (
    N_EXPERTS_DOMAIN, N_EXPERTS_TOTAL,
    EXPERT_NAMES, EXPERT_NOTES,
)
from .logging_utils import setup_logging
from .router_metrics import measure_latency
from .routers import (
    train_linear_router, train_gmm_router,
    train_nb_router, train_knn_router,
)

log = logging.getLogger("fase1")


# ──────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────


def main(args):
    global log
    log = setup_logging(args.embeddings)

    log.info("=" * 65)
    log.info("FASE 1 — Ablation Study del Router")
    log.info(f"Arquitectura: {N_EXPERTS_DOMAIN} expertos de dominio + "
             f"1 OOD (total={N_EXPERTS_TOTAL})")
    log.info("=" * 65)

    emb_dir = Path(args.embeddings)

    # ── Cargar backbone_meta.json (generado por FASE 0) ──────────────────
    meta_path = emb_dir / "backbone_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            backbone_meta = json.load(f)
        log.info(f"[Setup] backbone_meta.json cargado: {backbone_meta}")
    else:
        backbone_meta = {}
        log.warning("[Setup] backbone_meta.json no encontrado. "
                    "d_model se inferirá de Z_train.shape[1]. "
                    "Ejecuta FASE 0 con la versión actualizada para generarlo.")

    # ── Cargar embeddings ─────────────────────────────────────────────────
    for fname in ["Z_train.npy", "y_train.npy", "Z_val.npy", "y_val.npy"]:
        if not (emb_dir / fname).exists():
            log.error(f"[Setup] Archivo faltante: {emb_dir / fname}. "
                      f"Ejecuta FASE 0 primero.")
            raise FileNotFoundError(emb_dir / fname)

    Z_train = np.load(emb_dir / "Z_train.npy")
    y_train = np.load(emb_dir / "y_train.npy")
    Z_val   = np.load(emb_dir / "Z_val.npy")
    y_val   = np.load(emb_dir / "y_val.npy")
    d_model = Z_train.shape[1]

    log.info(f"[Setup] Z_train: {Z_train.shape}  |  Z_val: {Z_val.shape}")
    log.info(f"[Setup] d_model: {d_model}")

    if backbone_meta.get("d_model") and backbone_meta["d_model"] != d_model:
        log.error(f"[Setup] d_model del archivo ({d_model}) ≠ backbone_meta "
                  f"({backbone_meta['d_model']}). Los embeddings pueden estar mezclados.")

    # ── Distribución de expertos ─────────────────────────────────────────
    log.info("[Setup] Distribución de expertos en train:")
    for exp_id, exp_name in EXPERT_NAMES.items():
        count = (y_train == exp_id).sum()
        pct   = 100 * count / len(y_train)
        log.info(f"  Experto {exp_id} ({exp_name:<12}): {count:>6,}  ({pct:.1f}%)")
    log.info(f"  Experto 5 (OOD        ):      0  (0.0%)  "
             f"← sin dataset, activado por entropía en inferencia")

    classes_in_train = set(np.unique(y_train).tolist())
    expected = set(range(N_EXPERTS_DOMAIN))
    if classes_in_train != expected:
        log.error(f"[Setup] Clases en y_train: {classes_in_train} — "
                  f"se esperaban {expected}. El ablation study no será válido.")

    if np.isnan(Z_train).any() or np.isnan(Z_val).any():
        log.error("[Setup] NaN detectado en embeddings. Regenera con FASE 0.")
    if np.isinf(Z_train).any() or np.isinf(Z_val).any():
        log.error("[Setup] Inf detectado en embeddings. Regenera con FASE 0.")

    # ── Recordatorios por experto ────────────────────────────────────────
    log.info("[Setup] Recordatorios de diseño por experto para FASE 2:")
    for exp_id, note in EXPERT_NOTES.items():
        log.info(f"  Experto {exp_id} ({EXPERT_NAMES[exp_id]:<12}): {note}")
    log.info("  [IMPORTANTE] Este ablation study solo evalúa ROUTING (expert_id 0–4). "
             "Las métricas de patología (AUC, F1, QWK, CPM) se evalúan en FASE 2.")

    results = {}

    # ── A) LINEAR ────────────────────────────────────────────────────────
    log.info("\n[A] Entrenando Linear + Softmax (baseline DL)...")
    t0 = time.time()
    linear_model, linear_acc, linear_probs, linear_balance, linear_thresh = \
        train_linear_router(Z_train, y_train, Z_val, y_val, d_model,
                            epochs=args.epochs, lr=1e-3)
    linear_time = time.time() - t0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = torch.from_numpy(Z_val[:32]).float().to(device)
    linear_latency = measure_latency(lambda: linear_model(sample))

    results["Linear"] = {
        "acc":            linear_acc,
        "train_time_s":   linear_time,
        "latency_ms":     linear_latency,
        "params":         d_model * N_EXPERTS_DOMAIN + N_EXPERTS_DOMAIN,
        "needs_gpu":      True,
        "load_balance":   linear_balance,
        "entropy_thresh": linear_thresh,
        "model":          linear_model,
    }

    # ── B) GMM ───────────────────────────────────────────────────────────
    log.info("\n[B] Entrenando GMM (paramétrico EM)...")
    t0 = time.time()
    gmm_model, gmm_map, gmm_acc, gmm_probs, gmm_balance, gmm_thresh = \
        train_gmm_router(Z_train, y_train, Z_val, y_val)
    gmm_time = time.time() - t0

    sample_np = Z_val[:32].astype(np.float32)
    gmm_latency = measure_latency(lambda: gmm_model.predict_proba(sample_np))

    results["GMM"] = {
        "acc":            gmm_acc,
        "train_time_s":   gmm_time,
        "latency_ms":     gmm_latency,
        "params":         N_EXPERTS_DOMAIN * (d_model + d_model * d_model),
        "needs_gpu":      False,
        "load_balance":   gmm_balance,
        "entropy_thresh": gmm_thresh,
        "model":          (gmm_model, gmm_map),
    }

    # ── C) NAIVE BAYES ───────────────────────────────────────────────────
    log.info("\n[C] Entrenando Naive Bayes (MLE analítico)...")
    t0 = time.time()
    nb_model, nb_acc, nb_probs, nb_balance, nb_thresh = \
        train_nb_router(Z_train, y_train, Z_val, y_val)
    nb_time = time.time() - t0

    nb_latency = measure_latency(lambda: nb_model.predict_proba(sample_np))

    results["NaiveBayes"] = {
        "acc":            nb_acc,
        "train_time_s":   nb_time,
        "latency_ms":     nb_latency,
        "params":         N_EXPERTS_DOMAIN * 2 * d_model,
        "needs_gpu":      False,
        "load_balance":   nb_balance,
        "entropy_thresh": nb_thresh,
        "model":          nb_model,
    }

    # ── D) kNN FAISS ─────────────────────────────────────────────────────
    log.info("\n[D] Construyendo índice kNN-FAISS (no paramétrico)...")
    t0 = time.time()
    knn_index, knn_labels, knn_acc, knn_probs, knn_balance, knn_thresh = \
        train_knn_router(Z_train, y_train, Z_val, y_val, k=args.knn_k)
    knn_time = time.time() - t0

    sample_norm = sample_np.copy(); faiss.normalize_L2(sample_norm)
    knn_latency = measure_latency(lambda: knn_index.search(sample_norm, args.knn_k))

    results["kNN-FAISS"] = {
        "acc":            knn_acc,
        "train_time_s":   knn_time,
        "latency_ms":     knn_latency,
        "params":         len(Z_train) * d_model,
        "needs_gpu":      False,
        "load_balance":   knn_balance,
        "entropy_thresh": knn_thresh,
        "model":          (knn_index, knn_labels),
    }

    # ── TABLA COMPARATIVA ────────────────────────────────────────────────
    col = 78
    log.info("\n" + "=" * col)
    log.info(f"{'ABLATION STUDY — Tabla comparativa (sección 4.3 del proyecto)':^{col}}")
    log.info("=" * col)
    log.info(f"{'Router':<14} {'Tipo':<24} {'Acc':>6} {'Lat(ms)':>8} "
             f"{'Train(s)':>9} {'Bal':>5} {'H_thr':>6} {'GPU':>4}")
    log.info("-" * col)

    ROUTER_TYPE = {
        "Linear":    "Paramétrico (gradiente)",
        "GMM":       "Paramétrico (EM)",
        "NaiveBayes":"Paramétrico (MLE)",
        "kNN-FAISS": "No paramétrico",
    }
    for name, r in sorted(results.items(), key=lambda x: -x[1]["acc"]):
        log.info(
            f"{name:<14} {ROUTER_TYPE[name]:<24} "
            f"{r['acc']:>6.4f} "
            f"{r['latency_ms']:>8.2f} "
            f"{r['train_time_s']:>9.1f} "
            f"{r['load_balance']:>5.2f} "
            f"{r['entropy_thresh']:>6.3f} "
            f"{'Sí' if r['needs_gpu'] else 'No':>4}"
        )
    log.info("=" * col)
    log.info("  Bal = max(f_i)/min(f_i) objetivo < 1.30 | "
             "H_thr = umbral entropía OOD (p95)")

    # ── GANADOR ──────────────────────────────────────────────────────────
    winner_name, winner = max(results.items(), key=lambda x: x[1]["acc"])
    log.info(f"\nROUTER GANADOR: {winner_name}  "
             f"(Routing Accuracy = {winner['acc']:.4f})")
    log.info(f"  Latencia de inferencia : {winner['latency_ms']:.2f} ms (batch=32)")
    log.info(f"  Balance de carga       : {winner['load_balance']:.2f}x")
    log.info(f"  Entropy threshold OOD  : {winner['entropy_thresh']:.4f}")
    log.info(f"\n  → Usar este router en FASE 1 del entrenamiento MoE")
    log.info(f"  → En FASE 1 real: LinearGatingHead(d_model={d_model}, "
             f"n_experts={N_EXPERTS_TOTAL})  ← añade slot OOD")
    log.info(f"  → El slot OOD (ID=5) se entrena con L_error, "
             f"no con labels de dominio")
    log.info(f"  → Registrar tabla completa en Reporte Técnico (sección 3)")

    # ── GUARDAR RESULTADOS ───────────────────────────────────────────────
    report = {}
    for k, v in results.items():
        report[k] = {
            "acc":            float(v["acc"]),
            "train_time_s":   float(v["train_time_s"]),
            "latency_ms":     float(v["latency_ms"]),
            "params":         int(v["params"]),
            "needs_gpu":      v["needs_gpu"],
            "load_balance":   float(v["load_balance"]),
            "entropy_thresh": float(v["entropy_thresh"]),
        }

    output = {
        "results":          report,
        "winner":           winner_name,
        "n_experts_domain": N_EXPERTS_DOMAIN,
        "n_experts_total":  N_EXPERTS_TOTAL,
        "d_model":          d_model,
        "backbone":         backbone_meta.get("backbone", "unknown"),
        "entropy_threshold_winner": float(winner["entropy_thresh"]),
        "note_ood": (
            f"Experto 5 (OOD) se activa en inferencia cuando H(g) >= "
            f"{winner['entropy_thresh']:.4f}. "
            f"En FASE 1 real usar LinearGatingHead(n_experts={N_EXPERTS_TOTAL})."
        ),
        "expert_notes_fase2": EXPERT_NOTES,
    }

    out_path = emb_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nResultados guardados en {out_path}")
    log.info(f"Siguiente paso: entrenar FASE 1 MoE con router='{winner_name}' "
             f"y n_experts={N_EXPERTS_TOTAL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FASE 1 — Ablation Study del Router (arquitectura 6 expertos)"
    )
    parser.add_argument(
        "--embeddings", default="./embeddings",
        help="Carpeta con Z_train.npy, Z_val.npy y backbone_meta.json (output de FASE 0)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Épocas para entrenar el Linear router (ablation). "
             "FASE 1 real usa ~50 épocas con LR=1e-3."
    )
    parser.add_argument(
        "--knn_k", type=int, default=5,
        help="Número de vecinos para kNN-FAISS (default=5)"
    )
    args = parser.parse_args()
    main(args)
    main(args)
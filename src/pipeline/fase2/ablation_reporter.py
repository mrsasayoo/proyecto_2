"""
ablation_reporter.py — Consolidación y reporte del ablation study.

Responsabilidad única: recibir el dict de resultados del ablation, imprimir
la tabla comparativa, seleccionar el ganador, guardar el modelo ganador y
escribir ablation_results.json.

La lógica de selección del ganador es una política de negocio (priorizar
balance sobre accuracy). Si esta política cambia, solo se modifica este
módulo, sin tocar el código de entrenamiento de ningún router.

Criterio de selección (dos pasos):
  1. Filtrar: safe = {r | r.load_balance ≤ LOAD_BALANCE_THRESHOLD}
  2. Seleccionar: winner = max(safe, key=acc_val)
  Si safe vacío → min(results, key=load_balance) + warning.
"""

import json
import logging
from pathlib import Path

import torch

from config import N_EXPERTS_DOMAIN, N_EXPERTS_TOTAL, EXPERT_NOTES
from fase2_config import LOAD_BALANCE_THRESHOLD, ENTROPY_PERCENTILE

log = logging.getLogger("fase2")

ROUTER_TYPE = {
    "Linear": "Paramétrico (gradiente)",
    "GMM": "Paramétrico (EM)",
    "NaiveBayes": "Paramétrico (MLE)",
    "kNN-FAISS": "No paramétrico",
}


def report_and_save(results: dict, data: dict, args) -> None:
    """
    Imprime la tabla comparativa, selecciona el ganador, guarda el modelo y
    escribe ablation_results.json en el directorio de embeddings.

    Parameters
    ----------
    results : dict
        Dict devuelto por ablation_runner.run_ablation().
    data : dict
        Dict devuelto por embeddings_loader.load_embeddings().
    args : argparse.Namespace
        Argumentos CLI. Se usa args.embeddings para determinar emb_dir.
    """
    emb_dir = Path(args.embeddings)
    backbone_meta = data["backbone_meta"]
    d_model = data["d_model"]

    # ── Tabla comparativa ────────────────────────────────────────────────
    col = 89
    log.info("\n" + "=" * col)
    log.info(
        "%-*s",
        col,
        "ABLATION STUDY — Tabla comparativa (sección 4.3 del proyecto)".center(col),
    )
    log.info("=" * col)
    log.info(
        "%-14s %-24s %7s %8s %8s %9s %5s %6s %4s",
        "Router",
        "Tipo",
        "Acc_val",
        "Acc_test",
        "Lat(ms)",
        "Train(s)",
        "Bal",
        "H_thr",
        "GPU",
    )
    log.info("-" * col)

    for name, r in sorted(results.items(), key=lambda x: -x[1]["acc"]):
        acc_test = results[name].get("acc_test", float("nan"))
        log.info(
            "%-14s %-24s %7.4f %8.4f %8.2f %9.1f %5.2f %6.3f %4s",
            name,
            ROUTER_TYPE.get(name, ""),
            r["acc"],
            acc_test if acc_test is not None else float("nan"),
            r["latency_ms"],
            r["train_time_s"],
            r["load_balance"],
            r["entropy_thresh"],
            "Sí" if r["needs_gpu"] else "No",
        )
    log.info("=" * col)
    log.info(
        "  Bal = max(f_i)/min(f_i) objetivo < %.2f | H_thr = umbral entropía OOD (p%d)",
        LOAD_BALANCE_THRESHOLD,
        ENTROPY_PERCENTILE,
    )

    # ── Selección del ganador ────────────────────────────────────────────
    # Paso 1: filtrar por balance seguro
    safe_routers = {
        k: v
        for k, v in results.items()
        if v.get("load_balance", float("inf")) <= LOAD_BALANCE_THRESHOLD
    }

    if safe_routers:
        winner_name, winner = max(safe_routers.items(), key=lambda x: x[1]["acc"])
        log.info(
            "[Winner] Seleccionado entre %d routers con balance seguro (≤%.2f): %s",
            len(safe_routers),
            LOAD_BALANCE_THRESHOLD,
            winner_name,
        )
    else:
        winner_name, winner = min(
            results.items(),
            key=lambda x: x[1].get("load_balance", float("inf")),
        )
        log.warning(
            "[Winner] Ningún router tiene balance ≤ %.2f. "
            "Seleccionado por menor ratio: %s (%.2fx)",
            LOAD_BALANCE_THRESHOLD,
            winner_name,
            winner.get("load_balance", 0),
        )

    log.info(
        "\nROUTER GANADOR: %s  (Routing Accuracy = %.4f)", winner_name, winner["acc"]
    )
    log.info("  Latencia de inferencia : %.2f ms (batch=32)", winner["latency_ms"])
    log.info("  Balance de carga       : %.2fx", winner["load_balance"])
    log.info("  Entropy threshold OOD  : %.4f", winner["entropy_thresh"])
    log.info("\n  → Usar este router en Fase 3 del entrenamiento MoE")
    log.info(
        "  → En Fase 3 real: LinearGatingHead(d_model=%d, n_experts=%d)  ← añade slot OOD",
        d_model,
        N_EXPERTS_TOTAL,
    )
    log.info("  → El slot OOD (ID=5) se entrena con L_error, no con labels de dominio")
    log.info("  → Registrar tabla completa en Reporte Técnico (sección 3)")

    # ── Guardar modelo ganador ───────────────────────────────────────────
    best_router_path = _save_winner(winner_name, winner, emb_dir)

    # ── Construir y guardar ablation_results.json ────────────────────────
    report = {}
    for k, v in results.items():
        report[k] = {
            "acc": float(v["acc"]),
            "acc_test": float(v["acc_test"]) if v["acc_test"] is not None else None,
            "train_time_s": float(v["train_time_s"]),
            "latency_ms": float(v["latency_ms"]),
            "params": int(v["params"]),
            "needs_gpu": bool(v["needs_gpu"]),
            "load_balance": float(v["load_balance"]),
            "entropy_thresh": float(v["entropy_thresh"]),
        }

    output = {
        "results": report,
        "winner": winner_name,
        "best_router_path": best_router_path,
        "n_experts_domain": N_EXPERTS_DOMAIN,
        "n_experts_total": N_EXPERTS_TOTAL,
        "d_model": d_model,
        "backbone": backbone_meta.get("backbone", "unknown"),
        "load_balance_threshold": LOAD_BALANCE_THRESHOLD,
        "entropy_threshold_winner": float(winner["entropy_thresh"]),
        "note_ood": (
            f"Experto 5 (OOD) se activa en inferencia cuando H(g) >= "
            f"{winner['entropy_thresh']:.4f}. "
            f"En Fase 3 real usar LinearGatingHead(n_experts={N_EXPERTS_TOTAL})."
        ),
        "expert_notes_fase2": EXPERT_NOTES,
    }

    out_path = emb_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("\nResultados guardados en %s", out_path)


def _save_winner(winner_name: str, winner: dict, emb_dir: Path) -> str | None:
    """
    Guarda el modelo ganador en el formato apropiado según el tipo de router.

    Todos los artefactos se guardan en emb_dir (directorio de embeddings de
    Fase 1), no en directorios de Fase 0 ni en src/.

    Returns
    -------
    str | None
        Ruta absoluta del archivo guardado, o None si falla.
    """
    winner_path = emb_dir / "best_router"
    best_router_path = None

    try:
        if winner_name == "Linear":
            save_file = str(winner_path) + "_linear.pt"
            torch.save(winner["model"].state_dict(), save_file)
            best_router_path = save_file
            log.info("[Save] Router Linear guardado: %s", save_file)

        elif winner_name == "GMM":
            import joblib

            save_file = str(winner_path) + "_gmm.joblib"
            joblib.dump(winner["model"], save_file)
            best_router_path = save_file
            log.info("[Save] Router GMM guardado: %s", save_file)

        elif winner_name == "NaiveBayes":
            import joblib

            save_file = str(winner_path) + "_nb.joblib"
            joblib.dump(winner["model"], save_file)
            best_router_path = save_file
            log.info("[Save] Router NaiveBayes guardado: %s", save_file)

        elif winner_name == "kNN-FAISS":
            import faiss as _faiss_save

            save_file = str(winner_path) + "_knn.faiss"
            _faiss_save.write_index(winner["model"][0], save_file)
            best_router_path = save_file
            log.info("[Save] Router kNN-FAISS guardado: %s", save_file)

    except Exception as e:
        log.error(
            "[Save] No se pudo guardar el modelo ganador (%s): %s",
            winner_name,
            e,
        )

    return best_router_path

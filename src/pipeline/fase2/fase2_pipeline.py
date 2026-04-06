"""
fase2_pipeline.py — Orquestador de Fase 2: Ablation Study del Router.

Responsabilidad única: definir el orden de ejecución de todos los módulos
anteriores, gestionar los argumentos CLI y producir el reporte final.

No contiene lógica propia — solo llama funciones.
Nunca implementa métricas, nunca toma decisiones sobre routers,
nunca guarda modelos directamente.

Uso:
    python src/pipeline/fase2/fase2_pipeline.py \\
        --embeddings ./embeddings/vit_tiny

Verificación del criterio 1 de éxito del plan:
    python -c "from fase2.fase2_pipeline import main" (desde src/pipeline/)
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────
# Permite ejecutar como script: python src/pipeline/fase2/fase2_pipeline.py
#
# Se añaden tres directorios al sys.path (en orden de prioridad):
#   1. src/pipeline/fase2/routers/ — para que routers/__init__.py resuelva
#      sus submódulos (linear, gmm, knn, naive_bayes) con imports directos
#   2. src/pipeline/fase2/ — para resolver módulos del paquete fase2
#      (fase2_config, router_metrics, routers, ablation_runner, ablation_reporter)
#   3. src/pipeline/ — para resolver módulos globales del pipeline
#      (config, logging_utils)
#
# Todos los imports dentro del paquete fase2/ son absolutos (sin '..'), lo
# que elimina la ambigüedad de contexto de paquete que causó los problemas
# de importación del estado anterior.
_THIS_DIR = Path(__file__).resolve().parent  # src/pipeline/fase2/
_ROUTERS_DIR = _THIS_DIR / "routers"  # src/pipeline/fase2/routers/
_PIPELINE_DIR = _THIS_DIR.parent  # src/pipeline/
_SRC_DIR = _PIPELINE_DIR.parent  # src/
_PROJECT_ROOT = _SRC_DIR.parent  # proyecto_2/

# Insert in reverse priority order so highest-priority ends up at index 0:
#   _PIPELINE_DIR inserted first (lowest priority)
#   _THIS_DIR inserted second (overrides _PIPELINE_DIR)
#   _ROUTERS_DIR inserted last (highest priority — ensures 'linear', 'gmm', etc.
#     resolve to fase2/routers/ modules before any same-named top-level module)
for _p in [str(_PIPELINE_DIR), str(_THIS_DIR), str(_ROUTERS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fase2_config import (
    LINEAR_EPOCHS,
    ALPHA_L_AUX,
    KNN_K,
)
from config import (
    N_EXPERTS_DOMAIN,
    N_EXPERTS_TOTAL,
)
from logging_utils import setup_logging
from router_metrics import log_split_distribution

import embeddings_loader
import ablation_runner
import ablation_reporter

log = logging.getLogger("fase2")


def _generate_synthetic_data(d_model: int) -> dict:
    """Genera embeddings sintéticos para dry-run (sin acceder a disco)."""
    import numpy as np

    rng = np.random.RandomState(42)
    n_train, n_val = 200, 40

    # 40 muestras por clase en train, 8 por clase en val
    y_train = np.repeat(np.arange(N_EXPERTS_DOMAIN), n_train // N_EXPERTS_DOMAIN)
    y_val = np.repeat(np.arange(N_EXPERTS_DOMAIN), n_val // N_EXPERTS_DOMAIN)

    Z_train = rng.randn(n_train, d_model).astype(np.float32)
    Z_val = rng.randn(n_val, d_model).astype(np.float32)

    return {
        "Z_train": Z_train,
        "y_train": y_train,
        "Z_val": Z_val,
        "y_val": y_val,
        "Z_test": np.zeros((0, d_model), dtype=np.float32),
        "y_test": np.zeros(0, dtype=np.int64),
        "backbone_meta": {
            "backbone": "synthetic-dry-run",
            "d_model": d_model,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": 0,
            "vram_gb": 0.0,
        },
        "d_model": d_model,
        "has_test": False,
    }


def main(args):
    """Orquesta el ablation study completo."""
    global log

    dry_run = getattr(args, "dry_run", False)

    # ── Dry-run: crear directorio temporal para logs si --embeddings no existe
    if dry_run:
        import tempfile, os

        _emb_dir = Path(args.embeddings)
        if not _emb_dir.exists():
            _log_dir = tempfile.mkdtemp(prefix="fase2_dryrun_")
            args.embeddings = _log_dir
        log = setup_logging(args.embeddings, phase_name="fase2")
        log.info("=" * 70)
        log.info("  [DRY-RUN] Step 6: Ablation Study del Router — modo verificación")
        log.info("=" * 70)
    else:
        log = setup_logging(args.embeddings, phase_name="fase2")

    log.info("=" * 65)
    log.info("FASE 2 — Ablation Study del Router")
    log.info(
        "Arquitectura: %d expertos de dominio + 1 OOD (total=%d)",
        N_EXPERTS_DOMAIN,
        N_EXPERTS_TOTAL,
    )
    log.info("=" * 65)

    # ── 1. Cargar y validar embeddings ───────────────────────────────────
    if dry_run:
        from config import BACKBONE_CONFIGS

        d_model = BACKBONE_CONFIGS["vit_tiny_patch16_224"]["d_model"]  # 192
        data = _generate_synthetic_data(d_model)
        log.info(
            "[DRY-RUN] Embeddings sintéticos generados: "
            "Z_train=%s, Z_val=%s, d_model=%d",
            data["Z_train"].shape,
            data["Z_val"].shape,
            d_model,
        )
    else:
        data = embeddings_loader.load_embeddings(args.embeddings)

    # ── 2. Distribución de expertos ──────────────────────────────────────
    log_split_distribution(data["y_train"], "train")
    log_split_distribution(data["y_val"], "val")

    # ── 3. Ejecutar ablation ─────────────────────────────────────────────
    results = ablation_runner.run_ablation(data, args, dry_run=dry_run)

    # ── 4. Reporte y guardado ────────────────────────────────────────────
    ablation_reporter.report_and_save(results, data, args, dry_run=dry_run)

    # ── 5. Siguiente paso / dry-run banner ───────────────────────────────
    if dry_run:
        log.info("")
        log.info("=" * 70)
        log.info("  DRY-RUN COMPLETADO — Step 6: Ablation Study (Router Selection)")
        log.info("  Routers verificados: Linear, GMM, NaiveBayes, kNN-FAISS")
        log.info(
            "  Embeddings: sintéticos (%d train / %d val, d=%d)",
            len(data["Z_train"]),
            len(data["Z_val"]),
            data["d_model"],
        )
        log.info("=" * 70)
        log.info(
            "[DRY-RUN] Pipeline verificado exitosamente. "
            "Ejecuta sin --dry-run para el ablation real."
        )
    else:
        fase3_path = _PIPELINE_DIR / "fase3" / "fase3_train_experts.py"
        log.info(
            "Siguiente paso: python %s --embeddings %s --ablation_results %s",
            fase3_path,
            args.embeddings,
            Path(args.embeddings) / "ablation_results.json",
        )


def _build_parser() -> argparse.ArgumentParser:
    """Define los argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="FASE 2 — Ablation Study del Router MoE (arquitectura 6 expertos)"
    )
    parser.add_argument(
        "--embeddings",
        default="./embeddings",
        help=(
            "Carpeta con Z_train.npy, Z_val.npy, Z_test.npy y backbone_meta.json "
            "(output de Fase 1). También es el destino de ablation_results.json "
            "y del modelo ganador."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=LINEAR_EPOCHS,
        help=f"Épocas para entrenar el router Linear. Default: {LINEAR_EPOCHS}",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=KNN_K,
        help=f"Número de vecinos para kNN-FAISS. Default: {KNN_K}",
    )
    parser.add_argument(
        "--l_aux_alpha",
        type=float,
        default=ALPHA_L_AUX,
        help=(
            f"Coeficiente α de la Auxiliary Loss L_aux del Switch Transformer "
            f"para el router Linear. Default: {ALPHA_L_AUX}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Ejecuta el ablation study con embeddings sintéticos y parámetros "
            "reducidos para verificar el pipeline sin datos reales."
        ),
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()
    main(_args)

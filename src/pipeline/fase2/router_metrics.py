"""
router_metrics.py — Métricas compartidas del ablation study.

Responsabilidad única: proporcionar las funciones de evaluación y diagnóstico
que son comparables entre todos los routers del ablation study.

Las métricas deben ser idénticas para todos los routers para que la
comparación sea válida. Un módulo central garantiza que se aplica exactamente
el mismo criterio de evaluación a todos los algoritmos.

Este módulo no escribe ningún archivo en disco. Todas las funciones devuelven
valores Python y escriben en el logger.
"""

import logging
import time

import numpy as np
from sklearn.metrics import accuracy_score

from config import N_EXPERTS_DOMAIN, EXPERT_NAMES
from fase2_config import LOAD_BALANCE_THRESHOLD, ENTROPY_PERCENTILE

log = logging.getLogger("fase2")


def compute_entropy(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Entropía de Shannon por muestra: H(g) = -Σ g_i · log(g_i + eps)

    Rango: [0, ln(N_EXPERTS_DOMAIN)]
      H=0   → router completamente seguro (toda la masa en un experto)
      H=max → router completamente inseguro (distribución uniforme)

    Umbral OOD: si H(g) >= ENTROPY_THRESHOLD → Experto 5 (OOD)
    El epsilon previene log(0).
    """
    return -(probs * np.log(probs + eps)).sum(axis=1)


def per_expert_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Accuracy de routing por experto de dominio.

    Útil para detectar si el router colapsa hacia un subconjunto de expertos
    o ignora sistemáticamente a alguno.

    Returns
    -------
    dict {nombre_experto: float | None}
        None si el experto no tiene muestras en y_true.
    """
    result = {}
    for exp_id, exp_name in EXPERT_NAMES.items():
        mask = y_true == exp_id
        if mask.sum() == 0:
            result[exp_name] = None
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        result[exp_name] = acc
    return result


def log_per_expert(tag: str, acc_dict: dict) -> None:
    """Imprime la tabla de accuracy por experto en el log con barra ASCII."""
    log.info("  [%s] Accuracy por experto de dominio:", tag)
    for name, acc in acc_dict.items():
        if acc is None:
            log.warning("    %-12s: sin muestras en val", name)
        else:
            bar = "█" * int(acc * 20)
            log.info("    %-12s: %.4f  %s", name, acc, bar)


def check_load_balance(y_pred: np.ndarray, tag: str) -> float:
    """
    Calcula el ratio max(f_i)/min(f_i) sobre las predicciones del set de validación.

    Detecta expert collapse total (0 muestras a un experto → ratio infinito)
    ANTES de aplicar el floor. Si el ratio supera LOAD_BALANCE_THRESHOLD de
    fase2_config.py, emite un error explícito de penalización.

    Returns
    -------
    float
        Ratio max/min. float('inf') si algún experto tiene 0 muestras.
    """
    counts = np.array(
        [(y_pred == i).sum() for i in range(N_EXPERTS_DOMAIN)], dtype=float
    )

    # Detectar expert collapse ANTES del floor
    zero_experts = np.where(counts == 0)[0].tolist()
    if zero_experts:
        log.error(
            "  [%s] EXPERT COLLAPSE TOTAL: expertos con 0 muestras routeadas: %s. "
            "Ratio real = infinito. Distribución: %s",
            tag,
            zero_experts,
            counts.astype(int).tolist(),
        )
        return float("inf")

    counts_floored = np.maximum(counts, 1)
    ratio = float(counts_floored.max() / counts_floored.min())

    if ratio > LOAD_BALANCE_THRESHOLD:
        log.warning(
            "  [%s] Balance de carga: max/min = %.2fx — supera el objetivo %.2fx. "
            "Distribución: %s",
            tag,
            ratio,
            LOAD_BALANCE_THRESHOLD,
            counts.astype(int).tolist(),
        )
        log.error(
            "  [%s] EXPERT COLLAPSE RISK: max/min = %.2fx > %.2f. "
            "PENALIZACIÓN del 40%% si se mantiene en evaluación final.",
            tag,
            ratio,
            LOAD_BALANCE_THRESHOLD,
        )
    else:
        log.info(
            "  [%s] Balance de carga: max/min = %.2fx ✓  Distribución: %s",
            tag,
            ratio,
            counts.astype(int).tolist(),
        )
    return ratio


def calibrate_entropy_threshold(
    probs: np.ndarray, y_true: np.ndarray, tag: str
) -> float:
    """
    Calibra el umbral de entropía para el Experto 5 (OOD).

    Estrategia: el umbral es el percentil ENTROPY_PERCENTILE de H(g) sobre el
    conjunto completo de validación. Esto significa que las muestras más
    confusas del router serán tratadas como OOD en inferencia.

    Returns
    -------
    float
        Umbral de entropía calibrado.
    """
    entropies = compute_entropy(probs)
    correct_mask = probs.argmax(axis=1) == y_true

    h_correct = entropies[correct_mask]
    h_incorrect = entropies[~correct_mask]

    threshold = float(np.percentile(entropies, ENTROPY_PERCENTILE))

    log.info("  [%s] Calibración entropía OOD (Experto 5):", tag)
    if len(h_correct) > 0:
        log.info("    H(g) media (correctas)  : %.4f", h_correct.mean())
    if len(h_incorrect) > 0:
        log.info("    H(g) media (incorrectas): %.4f", h_incorrect.mean())
    log.info("    H(g) máxima posible     : %.4f  (uniforme)", np.log(N_EXPERTS_DOMAIN))
    log.info("    ENTROPY_THRESHOLD (p%d)  : %.4f", ENTROPY_PERCENTILE, threshold)
    log.info(
        "    → Muestras que irían a Experto 5 OOD: %d/%d (%.1f%%)",
        (entropies >= threshold).sum(),
        len(entropies),
        100.0 * (entropies >= threshold).mean(),
    )

    if threshold < 0.3:
        log.warning(
            "  [%s] Umbral muy bajo (%.4f). El router es muy seguro "
            "o los embeddings están colapsados. Verifica con otra semilla.",
            tag,
            threshold,
        )
    return threshold


def measure_latency(fn, n_runs: int = 10) -> float:
    """
    Mide la latencia promedio de inferencia en ms sobre un batch de 32 muestras.

    Descarta la primera corrida (JIT/warm-up). Devuelve la media de n_runs
    corridas en milisegundos.

    Parameters
    ----------
    fn : callable
        Función sin argumentos que ejecuta la inferencia sobre el batch.
    n_runs : int
        Número de corridas para promediar (excluye el warm-up).

    Returns
    -------
    float
        Latencia media en milisegundos.
    """
    fn()  # warm-up
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)) * 1000.0


def log_split_distribution(y: np.ndarray, split_name: str) -> None:
    """Log per-expert sample counts and percentages for a data split."""
    total = len(y)
    log.info("=== Distribución de expertos — %s ===", split_name)
    for expert_id in range(N_EXPERTS_DOMAIN):
        count = int((y == expert_id).sum())
        pct = 100.0 * count / total if total > 0 else 0.0
        name = EXPERT_NAMES.get(expert_id, f"Expert {expert_id}")
        log.info("  Experto %d (%s): %d muestras (%.1f%%)", expert_id, name, count, pct)

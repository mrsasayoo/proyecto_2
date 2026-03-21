"""
Métricas y utilidades agnósticas al algoritmo de routing.

Funciones de evaluación compartidas por los 4 routers del ablation study
(Linear, GMM, Naive Bayes, kNN) y reutilizables en FASE 2.

- compute_entropy: entropía de Shannon para OOD detection
- per_expert_accuracy: desglose de accuracy por experto
- log_per_expert: visualización ASCII en log
- check_load_balance: ratio max/min de distribución
- calibrate_entropy_threshold: calibración p95 del umbral OOD
- measure_latency: benchmarking de inferencia
"""

import logging
import time

import numpy as np
from sklearn.metrics import accuracy_score

from .config import N_EXPERTS_DOMAIN, EXPERT_NAMES

log = logging.getLogger("fase1")


def compute_entropy(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Entropía de Shannon por muestra: H(g) = -sum(g_i * log(g_i + eps))
    Rango: [0, ln(N_EXPERTS_DOMAIN)]
      H=0   → router completamente seguro (toda la masa en un experto)
      H=max → router completamente inseguro (distribución uniforme)

    Umbral OOD: si H(g) >= ENTROPY_THRESHOLD → Experto 5 (OOD)
    """
    return -(probs * np.log(probs + eps)).sum(axis=1)


def per_expert_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Accuracy de routing por experto de dominio.
    Útil para detectar si el router ignora sistemáticamente un experto.
    Objetivo del proyecto: max(f_i)/min(f_i) < 1.30 en FASE 1 real.
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
    """Imprime la tabla de accuracy por experto en el log."""
    log.info(f"  [{tag}] Accuracy por experto de dominio:")
    for name, acc in acc_dict.items():
        if acc is None:
            log.warning(f"    {name:<12}: sin muestras en val")
        else:
            bar = "█" * int(acc * 20)
            log.info(f"    {name:<12}: {acc:.4f}  {bar}")


def check_load_balance(y_pred: np.ndarray, tag: str) -> float:
    """
    Verifica que el router no ignore expertos.
    Objetivo del proyecto: max(f_i)/min(f_i) < 1.30
    Retorna el ratio para incluirlo en la tabla comparativa.
    """
    counts = np.array([(y_pred == i).sum() for i in range(N_EXPERTS_DOMAIN)], dtype=float)
    counts = np.maximum(counts, 1)
    ratio  = counts.max() / counts.min()
    if ratio > 1.30:
        log.warning(f"  [{tag}] Balance de carga: max/min = {ratio:.2f}x — supera el objetivo 1.30x. "
                    f"Distribución: {counts.astype(int).tolist()}")
    else:
        log.info(f"  [{tag}] Balance de carga: max/min = {ratio:.2f}x ✓  "
                 f"Distribución: {counts.astype(int).tolist()}")
    return float(ratio)


def calibrate_entropy_threshold(probs: np.ndarray, y_true: np.ndarray, tag: str) -> float:
    """
    Calibra el umbral de entropía para el Experto 5 (OOD) analizando la
    distribución de H(g) en el set de validación.

    Estrategia: el umbral es el percentil 95 de H(g) sobre muestras bien
    clasificadas. Esto significa que el 5% de las muestras más confusas del
    router (aunque sean dominio conocido) se tratarán como OOD en inferencia.
    """
    entropies = compute_entropy(probs)
    correct_mask = (probs.argmax(axis=1) == y_true)

    h_correct   = entropies[correct_mask]
    h_incorrect = entropies[~correct_mask]

    threshold = float(np.percentile(entropies, 95))

    log.info(f"  [{tag}] Calibración entropía OOD (Experto 5):")
    log.info(f"    H(g) media (correctas)  : {h_correct.mean():.4f}")
    log.info(f"    H(g) media (incorrectas): {h_incorrect.mean():.4f}")
    log.info(f"    H(g) máxima posible     : {np.log(N_EXPERTS_DOMAIN):.4f}  (uniforme)")
    log.info(f"    ENTROPY_THRESHOLD (p95) : {threshold:.4f}")
    log.info(f"    → Muestras que irían a Experto 5 OOD: "
             f"{(entropies >= threshold).sum()}/{len(entropies)} "
             f"({100*(entropies >= threshold).mean():.1f}%)")

    if threshold < 0.3:
        log.warning(f"  [{tag}] Umbral muy bajo ({threshold:.4f}). El router es muy seguro "
                    f"o los embeddings están colapsados. Verifica con otra semilla.")
    return threshold


def measure_latency(fn, n_runs: int = 10) -> float:
    """
    Mide la latencia promedio de inferencia en ms sobre un batch de 32 muestras.
    Descarta la primera corrida (JIT/warm-up).
    """
    times = []
    fn()   # warm-up
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)) * 1000

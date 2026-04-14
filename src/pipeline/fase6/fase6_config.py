"""
fase6_config.py — Constantes exclusivas del Paso 9: inferencia y evaluación batch.

Fuente de verdad para umbrales de métricas, rutas de checkpoints y parámetros
de evaluación del sistema MoE en modo inferencia.

Umbrales de aceptación derivados de proyecto_moe.md §10 y §13.
"""

# =====================================================================
# Rutas de entrada
# =====================================================================
MOE_CHECKPOINT = "checkpoints/fase5/moe_final.pt"
"""Checkpoint del MoESystem entrenado en Fase 5 (Stage 3 final)."""

ENTROPY_THRESHOLD_PATH = "checkpoints/entropy_threshold.pkl"
"""Umbral de entropía calibrado sobre el validation set.
Generado por calibrate_entropy_threshold() en Fase 2."""

OOD_THRESHOLDS_JSON_PATH = "checkpoints/ood_thresholds.json"
"""Archivo JSON consolidado con todos los umbrales OOD:
  - entropy_threshold: umbral de entropía del router
  - cae_theta_leve: umbral bajo de MSE (imagen limpia vs. ruido leve)
  - cae_theta_ood: umbral alto de MSE (ruido leve vs. OOD absoluto)
Generado por OODDetector.calibrate_threshold() y calibrate_cae_thresholds()."""


# =====================================================================
# Rutas de resultados
# =====================================================================
RESULTS_DIR = "results/paso9"
"""Directorio raíz para artefactos de evaluación del Paso 9."""

FIGURES_DIR = "results/paso9/figures"
"""Directorio para figuras de evaluación (confusion matrices, ROC, etc.)."""


# =====================================================================
# Umbrales de métricas (de proyecto_moe.md §10 y §13)
# =====================================================================
F1_THRESHOLD_2D = 0.65
"""F1 mínimo aceptable para expertos 2D (Chest, ISIC, OA)."""

F1_THRESHOLD_3D = 0.58
"""F1 mínimo aceptable para expertos 3D (LUNA, Páncreas)."""

F1_FULL_2D = 0.72
"""F1 full marks para expertos 2D."""

F1_FULL_3D = 0.65
"""F1 full marks para expertos 3D."""

ROUTING_ACCURACY_MIN = 0.80
"""Routing accuracy mínima aceptable del router."""

OOD_AUROC_MIN = 0.80
"""AUROC mínimo para detección OOD (entropía vs. in-distribution)."""

LOAD_BALANCE_MAX_RATIO = 1.30
"""Ratio máximo de carga aceptable entre expertos (max/min)."""


# =====================================================================
# OOD
# =====================================================================
OOD_ENTROPY_PERCENTILE = 95
"""Percentil sobre el validation set para calibrar el umbral de entropía."""

CAE_MSE_PERCENTILE_LEVE = 50
"""Percentil del MSE in-distribution para θ_leve (imagen limpia vs. ruido leve)."""

CAE_MSE_PERCENTILE_OOD = 99
"""Percentil del MSE in-distribution para θ_OOD (ruido leve vs. OOD absoluto).
Umbral elevado de percentil 95 → 99 con la nueva arquitectura Res-U-Net
(mayor especificidad OOD, menor tasa de falsos positivos).
En ausencia de muestras OOD etiquetadas, se usa la cola alta de la distribución
in-distribution como proxy."""


# =====================================================================
# Evaluación batch
# =====================================================================
EVAL_BATCH_SIZE = 32
"""Batch size para evaluación (sin gradientes, permite batches más grandes)."""

EVAL_NUM_WORKERS = 4
"""Número de workers para DataLoader de evaluación."""


# =====================================================================
# Expertos
# =====================================================================
EXPERT_NAMES = ["chest", "isic", "oa_knee", "luna", "pancreas"]
"""Nombres de los 5 expertos de dominio, indexados 0-4."""

CAE_EXPERT_IDX = 5
"""Índice del experto CAE (OOD / autoencoder)."""

N_EXPERTS_DOMAIN = 5
"""Número de expertos de dominio (0-4)."""

N_EXPERTS_TOTAL = 6
"""Número total de expertos (dominio + CAE)."""

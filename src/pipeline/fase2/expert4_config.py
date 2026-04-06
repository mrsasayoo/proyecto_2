"""
Configuración de entrenamiento para Expert 4 — Páncreas CT 3D (Swin3D-Tiny).

Fuente de verdad para todos los hiperparámetros del experto 4 en Fase 2.

Dataset: PANORAMA / Zenodo — volúmenes CT abdominales con etiqueta PDAC+/PDAC-.
Arquitectura: Swin3D-Tiny (Swin Transformer adaptado a 3D).
Entrada: [B, 1, 64, 64, 64] — volumen CT abdominal monocanal.

Diferencias clave con Expert 3 (LUNA16):
  - HU clip [-100, 400] (abdomen) vs [-1000, 400] (pulmón)
  - FocalLoss alpha=0.75 (vs 0.85): desbalance menos extremo (~2.3:1 vs ~10:1)
  - k-fold CV (k=5) obligatorio por tamaño limitado (~281 volúmenes)
  - Batch size 2 (vs 4): Swin3D consume más VRAM que MC3-18
  - LR 5e-5 (vs 3e-4): Swin Transformers requieren LR más conservador
"""

# ── Optimizador ─────────────────────────────────────────────
EXPERT4_LR = 5e-5
"""Learning rate inicial para AdamW. Conservador para Swin Transformer 3D."""

EXPERT4_WEIGHT_DECAY = 0.05
"""Weight decay en AdamW. Valor estándar para Swin Transformers (Liu et al., 2021)."""

# ── Batch y entrenamiento ───────────────────────────────────
EXPERT4_BATCH_SIZE = 2
"""Batch size real por GPU. Limitado por VRAM — Swin3D con 4 stages
jerárquicos y shifted window attention consume más que MC3-18."""

EXPERT4_ACCUMULATION_STEPS = 8
"""Gradient accumulation steps. Batch efectivo = 2 × 8 = 16.
Menor que Expert 3 (4×8=32) porque el dataset es más pequeño (~281 volúmenes)."""

EXPERT4_FP16 = True
"""Mixed precision (torch.amp). Obligatorio para Swin3D en 12-16 GB VRAM."""

# ── Scheduler y stopping ───────────────────────────────────
EXPERT4_MAX_EPOCHS = 100
"""Máximo de épocas. Early stopping detendrá antes si val_loss estanca."""

EXPERT4_EARLY_STOPPING_PATIENCE = 15
"""Épocas sin mejora antes de detener. Menor que Expert 3 (20) porque el
dataset es más pequeño y el overfitting se manifiesta antes."""

EXPERT4_EARLY_STOPPING_MONITOR = "val_loss"
"""Métrica a monitorear para early stopping."""

# ── Clases y entrada ───────────────────────────────────────
EXPERT4_NUM_CLASSES = 2
"""Clases de salida: 0 = PDAC negativo, 1 = PDAC positivo."""

EXPERT4_INPUT_SIZE = (1, 64, 64, 64)
"""Forma del tensor de entrada: [C, D, H, W] — CT monocanal."""

EXPERT4_EXPERT_ID = 4
"""ID del experto en el pipeline MoE."""

# ── k-fold CV ──────────────────────────────────────────────
EXPERT4_NUM_FOLDS = 5
"""Número de folds para cross-validation. Obligatorio por tamaño del dataset."""

# ── HU clipping ───────────────────────────────────────────
EXPERT4_HU_CLIP = (-100, 400)
"""Rango HU para abdomen. NO usar (-1000, 400) de LUNA16.
Parénquima pancreático: +30 a +150 HU. Tumor PDAC: -20 a +80 HU.
El rango [-100, 400] maximiza contraste entre páncreas y tumor."""

# ── Loss ────────────────────────────────────────────────────
EXPERT4_FOCAL_ALPHA = 0.75
"""Peso de la clase positiva (PDAC+) en FocalLoss.
Menor que Expert 3 (0.85) porque el desbalance es menos extremo (~2.3:1 vs ~10:1).
alpha=0.75 penaliza más los FN de PDAC+ que los FP."""

EXPERT4_FOCAL_GAMMA = 2.0
"""Exponente de modulación de FocalLoss. gamma=2 es el valor de referencia
de Lin et al. 2017 — down-pondera negativos fáciles."""

# ── Scheduler CosineAnnealingWarmRestarts ──────────────────
EXPERT4_SCHEDULER_T0 = 10
"""Período inicial del cosine annealing (T_0). Primer restart a época 10."""

EXPERT4_SCHEDULER_T_MULT = 2
"""Multiplicador del período tras cada restart. Períodos: 10, 20, 40, ..."""

# ── Resumen ejecutivo para logs ─────────────────────────────
EXPERT4_CONFIG_SUMMARY = (
    "Expert 4 (Pancreas/Swin3D-Tiny): LR=5e-5 | WD=0.05 | "
    "FocalLoss(gamma=2, alpha=0.75) | "
    "batch=2 | accum=8 (efectivo=16) | FP16=True | "
    "patience=15 | max_epochs=100 | k-fold=5"
)

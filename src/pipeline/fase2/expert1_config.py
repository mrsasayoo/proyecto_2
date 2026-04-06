"""
Configuración de entrenamiento para Expert 1 — NIH ChestXray14 (14 patologías).

Fuente de verdad para todos los hiperparámetros del experto 1 en Fase 2.

Arquitectura: ConvNeXt-Tiny entrenado desde cero (weights=None).
Tarea: multilabel — 14 etiquetas binarias independientes.
Loss: BCEWithLogitsLoss con pos_weight computado del split de entrenamiento.

Análisis del dataset:
  - ~86,524 imágenes de entrenamiento (split oficial NIH)
  - 14 patologías con prevalencia muy variable (0.2% Hernia → 17.7% Infiltration)
  - ~53% son "No Finding" (vector todo-ceros) → pos_weight obligatorio
  - Etiquetas generadas por NLP (>90% precisión reportada por Wang et al.)
"""

# ── Optimizador ─────────────────────────────────────────────
EXPERT1_LR = 1e-4
"""Learning rate inicial para AdamW. Conservador porque ChestXray14 es un
dataset grande (~86K train) y ConvNeXt-Tiny (~28M params) se entrena
desde cero — LR muy alto causa inestabilidad con BCEWithLogitsLoss."""

EXPERT1_WEIGHT_DECAY = 0.05
"""Weight decay (L2 regularization) en AdamW. Valor estándar para
ConvNeXt (Liu et al., 2022 — A ConvNet for the 2020s)."""

# ── Regularización del modelo ───────────────────────────────
EXPERT1_DROPOUT_FC = 0.3
"""Dropout en la capa fully-connected final antes de la clasificación.
Valor moderado: el dataset es grande pero el desbalance por clase es
significativo, y el modelo se entrena desde cero."""

# ── Batch y entrenamiento ───────────────────────────────────
EXPERT1_BATCH_SIZE = 32
"""Batch size real por GPU. Con imágenes 224×224 RGB y ConvNeXt-Tiny
en FP16, batch_size=32 cabe cómodamente en 12 GB VRAM."""

EXPERT1_ACCUMULATION_STEPS = 4
"""Gradient accumulation steps. Batch efectivo = batch_size × accumulation_steps
= 32 × 4 = 128. Mínimo obligatorio del proyecto es 4.
Batch efectivo de 128 es estándar para entrenamiento de ConvNeXt."""

EXPERT1_FP16 = True
"""Usar mixed precision (torch.amp). Reduce consumo de VRAM ~40% y
acelera el entrenamiento ~1.5x en GPUs con soporte de Tensor Cores."""

# ── Scheduler y stopping ───────────────────────────────────
EXPERT1_MAX_EPOCHS = 50
"""Máximo de épocas. El early stopping detendrá antes si val_loss estanca.
50 es suficiente para convergencia con CosineAnnealingWarmRestarts."""

EXPERT1_EARLY_STOPPING_PATIENCE = 10
"""Épocas sin mejora en val_loss antes de detener el entrenamiento.
10 épocas es razonable para un dataset grande con augmentation."""

EXPERT1_EARLY_STOPPING_MONITOR = "val_loss"
"""Métrica a monitorear para early stopping. val_loss es más estable
que AUC o F1 para multilabel con 14 clases desbalanceadas."""

# ── Resumen ejecutivo para logs ─────────────────────────────
EXPERT1_CONFIG_SUMMARY = (
    "Expert 1 (ChestXray14): LR=1e-4 | WD=0.05 | "
    "BCEWithLogitsLoss(pos_weight) | dropout_fc=0.3 | "
    "batch=32 | accum=4 (efectivo=128) | FP16=True | "
    "patience=10 | max_epochs=50"
)

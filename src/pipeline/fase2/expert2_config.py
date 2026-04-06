"""
Configuración de entrenamiento para Expert 2 — ISIC 2019 (Dermoscopía, 8+1 clases).

Fuente de verdad para todos los hiperparámetros del experto 2 en Fase 2.

Arquitectura: EfficientNet-B3 entrenado desde cero (weights=None).
Tarea: multiclase — 8 clases de entrenamiento + 1 slot UNK (solo inferencia).
Loss: CrossEntropyLoss con class_weights (inverse-frequency, 9 posiciones).
Métrica principal: BMCA (Balanced Multi-Class Accuracy = balanced_accuracy_score).

Análisis del dataset:
  - ~25,331 imágenes (split por lesion_id sin leakage)
  - 8 clases con desbalance severo: NV ~53%, VASC ~0.9%
  - 3 fuentes con bias de dominio: HAM10000, BCN_20000, MSK
  - Augmentation diferenciado: estándar para mayoría, agresivo para minoría
  - Contramedida de bias: apply_circular_crop() para BCN_20000
"""

# ── Optimizador ─────────────────────────────────────────────
EXPERT2_LR = 3e-4
"""Learning rate inicial para AdamW. Valor moderado para EfficientNet-B3
entrenado desde cero sobre ISIC 2019."""

EXPERT2_WEIGHT_DECAY = 0.05
"""Weight decay (L2 regularization) en AdamW. Valor estándar para
modelos EfficientNet (Tan & Le, 2019)."""

# ── Regularización del modelo ───────────────────────────────
EXPERT2_DROPOUT_FC = 0.3
"""Dropout en la capa fully-connected final antes de la clasificación.
Valor moderado: EfficientNet-B3 ya incluye dropout interno, pero
el desbalance severo entre clases requiere regularización adicional."""

# ── Batch y entrenamiento ───────────────────────────────────
EXPERT2_BATCH_SIZE = 32
"""Batch size real por GPU. Con imágenes 224×224 RGB y EfficientNet-B3
en FP16, batch_size=32 cabe cómodamente en 12 GB VRAM."""

EXPERT2_ACCUMULATION_STEPS = 4
"""Gradient accumulation steps. Batch efectivo = batch_size × accumulation_steps
= 32 × 4 = 128. Mínimo obligatorio del proyecto es 4."""

EXPERT2_FP16 = True
"""Usar mixed precision (torch.amp). Reduce consumo de VRAM ~40% y
acelera el entrenamiento ~1.5x en GPUs con soporte de Tensor Cores."""

# ── Scheduler y stopping ───────────────────────────────────
EXPERT2_MAX_EPOCHS = 50
"""Máximo de épocas. El early stopping detendrá antes si val_loss estanca.
50 es suficiente para convergencia con CosineAnnealingWarmRestarts."""

EXPERT2_EARLY_STOPPING_PATIENCE = 10
"""Épocas sin mejora en val_loss antes de detener el entrenamiento.
10 épocas es razonable para un dataset de ~20K imágenes con augmentation."""

EXPERT2_EARLY_STOPPING_MONITOR = "val_loss"
"""Métrica a monitorear para early stopping. val_loss es más estable
que BMCA o AUC para multiclase con 8 clases desbalanceadas."""

# ── Resumen ejecutivo para logs ─────────────────────────────
EXPERT2_CONFIG_SUMMARY = (
    "Expert 2 (ISIC2019): LR=3e-4 | WD=0.05 | "
    "CrossEntropyLoss(class_weights) | dropout_fc=0.3 | "
    "batch=32 | accum=4 (efectivo=128) | FP16=True | "
    "patience=10 | max_epochs=50"
)

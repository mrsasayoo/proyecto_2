"""
Configuración de entrenamiento para Expert OA Knee — VGG16-BN (expert_id=2).

Fuente de verdad para todos los hiperparámetros del experto OA en Fase 2.

Arquitectura: VGG16 con Batch Normalization, entrenado desde cero (weights=None).
Tarea: clasificación ordinal — 3 clases (Normal KL0, Leve KL1-2, Severo KL3-4).
Loss: CrossEntropyLoss con class_weights (inverse-frequency).
Métrica principal: QWK (Quadratic Weighted Kappa).

Análisis del dataset:
  - ~4,766 imágenes KL-graded (train: 3,814 / val: 480 / test: 472)
  - 3 clases ordinales con desbalance moderado
  - Sin metadatos de paciente: split por carpeta OAI predefinido
  - Augmentation: RandomHorizontalFlip, RandomRotation(10), ColorJitter ligero
  - PROHIBIDO: RandomVerticalFlip (orientación anatómica fija)
  - PROHIBIDO: RandomErasing/Cutout/CutMix/MixUp (proyecto MoE)
  - Preprocesado: CLAHE antes del resize (interno en OAKneeDataset)
"""

# ── Optimizador ─────────────────────────────────────────────
EXPERT_OA_LR = 1e-4
"""Learning rate inicial para AdamW. Valor conservador para VGG16-BN
entrenado desde cero sobre un dataset pequeño (~4.7K imágenes)."""

EXPERT_OA_WEIGHT_DECAY = 0.05
"""Weight decay (L2 regularization) en AdamW. Valor estándar para
regularización en datasets pequeños con modelos grandes (~131M params)."""

# ── Regularización del modelo ───────────────────────────────
EXPERT_OA_DROPOUT_FC = 0.5
"""Dropout en las capas FC del clasificador. Agresivo (0.5) por el ratio
alto parámetros/muestras (~131M params vs ~4.7K muestras)."""

# ── Batch y entrenamiento ───────────────────────────────────
EXPERT_OA_BATCH_SIZE = 32
"""Batch size real por GPU. Con imágenes 224×224 RGB y VGG16-BN
en FP16, batch_size=32 cabe en ~6-8 GB VRAM."""

EXPERT_OA_ACCUMULATION_STEPS = 4
"""Gradient accumulation steps. Batch efectivo = batch_size × accumulation_steps
= 32 × 4 = 128. Mínimo obligatorio del proyecto es 4."""

EXPERT_OA_FP16 = True
"""Usar mixed precision (torch.amp). Reduce consumo de VRAM ~40% y
acelera el entrenamiento ~1.5x en GPUs con soporte de Tensor Cores.
NOTA: BN requiere estadísticas estables; AMP con precaución."""

# ── Scheduler y stopping ───────────────────────────────────
EXPERT_OA_MAX_EPOCHS = 100
"""Máximo de épocas. El early stopping detendrá antes si val_loss estanca.
100 es razonable para un dataset pequeño (~4.7K imgs) con convergencia lenta."""

EXPERT_OA_EARLY_STOPPING_PATIENCE = 10
"""Épocas sin mejora en val_loss antes de detener el entrenamiento.
10 épocas es razonable para un dataset pequeño con augmentation."""

EXPERT_OA_EARLY_STOPPING_MONITOR = "val_loss"
"""Métrica a monitorear para early stopping. val_loss es más estable
que QWK para early stopping en un problema ordinal con 3 clases."""

# ── Constantes del modelo ───────────────────────────────────
EXPERT_OA_NUM_CLASSES = 3
"""Número de clases de salida: 0=Normal (KL0), 1=Leve (KL1-2), 2=Severo (KL3-4)."""

EXPERT_OA_IMG_SIZE = 224
"""Tamaño de imagen de entrada (224×224 px). Estándar VGG16."""

EXPERT_OA_EXPERT_ID = 2
"""ID del experto en el sistema MoE. Coincide con EXPERT_IDS['oa'] = 2."""

# ── Resumen ejecutivo para logs ─────────────────────────────
EXPERT_OA_CONFIG_SUMMARY = (
    "Expert OA (Knee): LR=1e-4 | WD=0.05 | "
    "CrossEntropyLoss(class_weights) | dropout_fc=0.5 | "
    "batch=32 | accum=4 (efectivo=128) | FP16=True | "
    "patience=10 | max_epochs=100 | VGG16-BN"
)

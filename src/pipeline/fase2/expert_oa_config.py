"""
Configuración de entrenamiento para Expert OA Knee — EfficientNet-B0 (expert_id=2).

Fuente de verdad para todos los hiperparámetros del experto OA en Fase 2.

Arquitectura: EfficientNet-B0 (pretrained ImageNet), fine-tuning con Adam diferencial.
Tarea: clasificación de 5 clases — grados Kellgren-Lawrence 0-4.
Loss: CrossEntropyLoss con class_weights (inverse-frequency).
Métrica principal: val_f1_macro (F1-score macro).

Análisis del dataset:
  - ~4,766 imágenes KL-graded (train: 3,814 / val: 480 / test: 472)
  - 5 clases KL (0=Normal, 1=Dudoso, 2=Leve, 3=Moderado, 4=Severo)
  - Sin metadatos de paciente: split por carpeta OAI predefinido
  - Augmentation: RandomHorizontalFlip, RandomRotation(10), ColorJitter ligero
  - PROHIBIDO: RandomVerticalFlip (orientación anatómica fija)
  - PROHIBIDO: RandomErasing/Cutout/CutMix/MixUp (proyecto MoE)
  - Preprocesado: CLAHE antes del resize (interno en OAKneeDataset)

Cambios respecto a la versión anterior (VGG16-BN):
  - Modelo: VGG16-BN (from scratch) → EfficientNet-B0 (pretrained, fine-tune)
  - Clases: 3 (Normal/Leve/Severo) → 5 (KL 0-4)
  - Optimizador: AdamW uniforme → Adam diferencial (lr_backbone ≠ lr_head)
  - Scheduler: CosineAnnealingWarmRestarts → CosineAnnealingLR
  - Épocas: 100 → 30
  - Dropout: 0.5 → 0.4
  - Accum steps: 4 → 2 (batch efectivo 128 → 64)
  - Métrica principal: QWK → val_f1_macro
"""

# ── Identidad del experto ───────────────────────────────────
EXPERT_OA_EXPERT_ID = 2
"""ID del experto en el sistema MoE. Coincide con EXPERT_IDS['oa'] = 2."""

EXPERT_OA_NUM_CLASSES = 5
"""Número de clases de salida: grados Kellgren-Lawrence 0-4."""

EXPERT_OA_CLASS_NAMES: list[str] = [
    "Normal",
    "Dudoso",
    "Leve",
    "Moderado",
    "Severo",
]
"""Nombres legibles de las 5 clases KL (índice = grado KL)."""

# ── Modelo ──────────────────────────────────────────────────
EXPERT_OA_MODEL_NAME = "efficientnet_b0"
"""Arquitectura base. EfficientNet-B0 pretrained en ImageNet."""

EXPERT_OA_DROPOUT_FC = 0.4
"""Dropout antes de la capa de clasificación. Reducido respecto a VGG16-BN
(0.5 → 0.4) porque EfficientNet-B0 tiene ~5.3M params vs ~131M de VGG16."""

EXPERT_OA_IMG_SIZE = 224
"""Tamaño de imagen de entrada (224×224 px). Estándar EfficientNet-B0."""

# ── Optimizador (Adam diferencial) ──────────────────────────
EXPERT_OA_OPTIMIZER = "Adam"
"""Optimizador. Adam con learning rates diferenciales para backbone y head."""

EXPERT_OA_LR = 5e-5
"""Learning rate para el backbone (capas pretrained). Conservador para
preservar features de ImageNet durante fine-tuning."""

EXPERT_OA_LR_BACKBONE = 5e-5
"""Learning rate para el backbone (alias explícito). Igual a EXPERT_OA_LR
para compatibilidad con el import existente en train_expert_oa.py."""

EXPERT_OA_LR_HEAD = 5e-4
"""Learning rate para el classification head (10× mayor que backbone).
Permite al head adaptarse rápido a las 5 clases KL."""

EXPERT_OA_WEIGHT_DECAY = 1e-4
"""Weight decay (L2 regularization) en Adam. Valor moderado para
EfficientNet-B0 pretrained (~5.3M params)."""

# ── Batch y entrenamiento ───────────────────────────────────
EXPERT_OA_BATCH_SIZE = 32
"""Batch size real por GPU. Con imágenes 224×224 RGB y EfficientNet-B0
en FP16, batch_size=32 cabe cómodamente en ~4-6 GB VRAM."""

EXPERT_OA_ACCUMULATION_STEPS = 2
"""Gradient accumulation steps. Batch efectivo = batch_size × accumulation_steps
= 32 × 2 = 64."""

EXPERT_OA_FP16 = True
"""Usar mixed precision (torch.amp con GradScaler). Reduce VRAM ~40% y
acelera ~1.5x en GPUs con Tensor Cores."""

# ── Scheduler y stopping ───────────────────────────────────
EXPERT_OA_MAX_EPOCHS = 30
"""Máximo de épocas. Con EfficientNet-B0 pretrained y Adam diferencial,
la convergencia es más rápida que VGG16-BN from scratch."""

EXPERT_OA_SCHEDULER = "CosineAnnealingLR"
"""Scheduler de learning rate. Cosine annealing simple (sin warm restarts)."""

EXPERT_OA_SCHEDULER_T_MAX = 30
"""T_max para CosineAnnealingLR. Igual al número máximo de épocas para
un ciclo completo de cosine decay."""

EXPERT_OA_SCHEDULER_ETA_MIN = 1e-6
"""Learning rate mínimo al final del cosine schedule."""

EXPERT_OA_EARLY_STOPPING_PATIENCE = 10
"""Épocas sin mejora en la métrica monitor antes de detener el entrenamiento."""

EXPERT_OA_EARLY_STOPPING_MONITOR = "val_f1_macro"
"""Métrica a monitorear para early stopping y selección de mejor checkpoint."""

# ── Checkpoint ──────────────────────────────────────────────
EXPERT_OA_CHECKPOINT_DIR = "checkpoints/expert_02_vgg16_bn"
"""Directorio de checkpoints. Se mantiene la ruta existente para no
romper otros scripts que referencien esta ubicación."""

EXPERT_OA_MONITOR_METRIC = "val_f1_macro"
"""Métrica para seleccionar el mejor checkpoint."""

EXPERT_OA_MONITOR_MODE = "max"
"""Modo de monitoreo: 'max' porque F1-macro es mejor cuanto mayor."""

EXPERT_OA_TARGET_METRIC_THRESHOLD = 0.72
"""Umbral mínimo de val_f1_macro para considerar el modelo aceptable."""

# ── Resumen ejecutivo para logs ─────────────────────────────
EXPERT_OA_CONFIG_SUMMARY = (
    "Expert OA (Knee): EfficientNet-B0 | 5 clases KL | "
    "Adam diferencial (backbone=5e-5, head=5e-4) | WD=1e-4 | "
    "CosineAnnealingLR(T_max=30) | dropout=0.4 | "
    "batch=32 | accum=2 (efectivo=64) | FP16=True | "
    "patience=10 | max_epochs=30 | monitor=val_f1_macro"
)

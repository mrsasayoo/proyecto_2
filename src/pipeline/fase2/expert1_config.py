"""
Configuración de entrenamiento para Expert 1 — NIH ChestXray14 (14 patologías).

Fuente de verdad para todos los hiperparámetros del experto 1 en Fase 2.

Estrategia: LP-FT (Linear Probing → Fine-Tuning) con backbone pretrained.
  - Fase LP: backbone congelado, solo se entrena head + domain_conv.
  - Fase FT: todo el modelo descongelado con LR más bajo.

Arquitectura: ConvNeXt-Tiny pretrained (timm: convnext_tiny.in12k_ft_in1k).
Tarea: multilabel — 14 etiquetas binarias independientes.
Loss: BCEWithLogitsLoss con pos_weight computado del split de entrenamiento.

Análisis del dataset:
  - ~86,524 imágenes de entrenamiento (split oficial NIH)
  - 14 patologías con prevalencia muy variable (0.2% Hernia → 17.7% Infiltration)
  - ~53% son "No Finding" (vector todo-ceros) → pos_weight obligatorio
  - Etiquetas generadas por NLP (>90% precisión reportada por Wang et al.)

Nota: MODEL_MEAN y MODEL_STD no se hardcodean aquí. Se resuelven
programáticamente via `timm.data.resolve_data_config(model)` para garantizar
consistencia con las estadísticas de normalización del backbone pretrained.
"""

# ── Backbone ────────────────────────────────────────────────
EXPERT1_BACKBONE = "convnext_tiny.in12k_ft_in1k"
"""Backbone pretrained de timm. ConvNeXt-Tiny (~28M params) pre-entrenado
en ImageNet-12K y fine-tuned en ImageNet-1K. Buen balance entre capacidad
y eficiencia para transfer learning en imágenes médicas."""

# ── Épocas LP-FT ────────────────────────────────────────────
EXPERT1_LP_EPOCHS = 5
"""Épocas de Linear Probing (backbone congelado). Solo se entrenan
head + domain_conv. Pocas épocas bastan para alinear el clasificador
con las features pretrained antes de descongelar el backbone."""

EXPERT1_FT_EPOCHS = 30
"""Épocas de Fine-Tuning (todo descongelado). Suficiente para
convergencia con early stopping y scheduler coseno."""

# ── Learning rates LP-FT ────────────────────────────────────
EXPERT1_LP_LR = 1e-3
"""LR para la fase LP (solo head + domain_conv). Más agresivo porque
las capas entrenables son pocas y el backbone está congelado — no hay
riesgo de destruir features pretrained."""

EXPERT1_FT_LR = 1e-4
"""LR para la fase FT (todo el modelo). Conservador para preservar
las features del backbone pretrained mientras se adapta al dominio
médico de ChestXray14."""

# ── Regularización del modelo ───────────────────────────────
EXPERT1_WEIGHT_DECAY = 0.05
"""Weight decay (L2 regularization) en AdamW. Valor estándar para
ConvNeXt (Liu et al., 2022 — A ConvNet for the 2020s)."""

EXPERT1_DROPOUT_FC = 0.3
"""Dropout en la capa fully-connected final antes de la clasificación.
Valor moderado: el dataset es grande pero el desbalance por clase es
significativo."""

# ── Batch y entrenamiento ───────────────────────────────────
EXPERT1_BATCH_SIZE = 32
"""Batch size real por GPU. Con imágenes 224x224 RGB y ConvNeXt-Tiny
en FP16, batch_size=32 cabe cómodamente en 12 GB VRAM."""

EXPERT1_NUM_WORKERS = 4
"""Workers para DataLoader. 4 es suficiente para saturar el I/O
con imágenes 224x224 desde SSD."""

EXPERT1_ACCUMULATION_STEPS = 4
"""Gradient accumulation steps. Batch efectivo = batch_size x accumulation_steps
= 32 x 4 = 128. Mínimo obligatorio del proyecto es 4.
Batch efectivo de 128 es estándar para entrenamiento de ConvNeXt."""

EXPERT1_FP16 = True
"""Usar mixed precision (torch.amp). Reduce consumo de VRAM ~40% y
acelera el entrenamiento ~1.5x en GPUs con soporte de Tensor Cores."""

# ── Imagen y clases ─────────────────────────────────────────
EXPERT1_IMG_SIZE = 224
"""Tamaño de imagen de entrada (224x224). Resolución nativa del
backbone pretrained convnext_tiny.in12k_ft_in1k."""

EXPERT1_NUM_CLASSES = 14
"""Número de patologías del dataset NIH ChestXray14."""

# ── Scheduler y stopping ───────────────────────────────────
EXPERT1_EARLY_STOPPING_PATIENCE = 10
"""Épocas sin mejora en val_loss antes de detener el entrenamiento.
10 épocas es razonable para un dataset grande con augmentation."""

EXPERT1_EARLY_STOPPING_MONITOR = "val_loss"
"""Métrica a monitorear para early stopping. val_loss es más estable
que AUC o F1 para multilabel con 14 clases desbalanceadas."""

# ── Resumen ejecutivo para logs ─────────────────────────────
EXPERT1_CONFIG_SUMMARY = (
    "Expert 1 (ChestXray14) LP-FT: "
    f"backbone={EXPERT1_BACKBONE} | "
    f"LP={EXPERT1_LP_EPOCHS}ep@LR={EXPERT1_LP_LR} | "
    f"FT={EXPERT1_FT_EPOCHS}ep@LR={EXPERT1_FT_LR} | "
    f"WD={EXPERT1_WEIGHT_DECAY} | dropout_fc={EXPERT1_DROPOUT_FC} | "
    f"batch={EXPERT1_BATCH_SIZE} | accum={EXPERT1_ACCUMULATION_STEPS} "
    f"(efectivo={EXPERT1_BATCH_SIZE * EXPERT1_ACCUMULATION_STEPS}) | "
    f"FP16={EXPERT1_FP16} | patience={EXPERT1_EARLY_STOPPING_PATIENCE}"
)

"""
Configuración de entrenamiento para Expert 1 — NIH ChestXray14 (14 patologías).

Fuente de verdad para todos los hiperparámetros del experto 1 en Fase 2.

Estrategia: entrenamiento directo desde cero (sin LP-FT, sin pretrained).
Arquitectura: Hybrid-Deep-Vision (custom Dense-Inception + ResNet).
Tarea: multilabel — 14 etiquetas binarias independientes.
Loss: BCELoss (modelo produce probabilidades post-sigmoid).

Entrada: [B, 1, 256, 256] — escala de grises.
Normalización: stats.json del dataset preprocesado (canal único).

Análisis del dataset:
  - ~86,524 imágenes de entrenamiento (split oficial NIH)
  - 14 patologías con prevalencia muy variable (0.2% Hernia → 17.7% Infiltration)
  - ~53% son "No Finding" (vector todo-ceros) → pos_weight obligatorio
  - Etiquetas generadas por NLP (>90% precisión reportada por Wang et al.)
"""

# ── Épocas ──────────────────────────────────────────────────
EXPERT1_EPOCHS = 100
"""Épocas de entrenamiento directo desde cero. Sin fases LP/FT.
100 épocas con early stopping (patience=20) para convergencia completa."""

# ── Learning rate ───────────────────────────────────────────
EXPERT1_LR = 3e-4
"""Learning rate para AdamW. Valor conservador para entrenamiento desde cero
con pos_weight alto — reduce riesgo de gradientes explosivos."""

# ── Regularización del modelo ───────────────────────────────
EXPERT1_WEIGHT_DECAY = 1e-4
"""Weight decay (L2 regularization) en AdamW. Valor moderado para
entrenamiento desde cero — menor que el 0.05 típico de transfer learning."""

EXPERT1_DROPOUT_FC = 0.4
"""Dropout en la capa fully-connected del cabezal clasificador.
Valor moderado-alto para compensar el entrenamiento desde cero en un
dataset con desbalance significativo entre clases."""

# ── Batch y entrenamiento ───────────────────────────────────
EXPERT1_BATCH_SIZE = 48
"""Batch size total (dividido entre GPUs en DDP). Con 2 GPUs → 24 per GPU.
Incrementado de 32→48 aprovechando ~4.3 GB VRAM libres por GPU (7.8/12 usados)."""

EXPERT1_NUM_WORKERS = 8
"""Workers para DataLoader. 8 workers para saturar GPU y evitar
cuello de botella en carga de datos con imágenes 256×256."""

EXPERT1_ACCUMULATION_STEPS = 2
"""Gradient accumulation steps. Batch efectivo = batch_size × accum
= 48 × 2 = 96. Reducido de 4→2 (menos pasadas lentas) manteniendo
batch efectivo suficientemente grande. Mínimo obligatorio del proyecto es 2."""

EXPERT1_FP16 = True
"""FP16 habilitado: FocalLoss aplica alpha fuera de BCE (no como pos_weight
interno), alpha clamped a max=50, y gradient clipping activo. Seguro en FP16."""

# ── Imagen y clases ─────────────────────────────────────────
EXPERT1_IMG_SIZE = 256
"""Tamaño de imagen de entrada (256×256). Resolución nativa de la
arquitectura Hybrid-Deep-Vision (escala de grises, 1 canal)."""

EXPERT1_NUM_CLASSES = 14
"""Número de patologías del dataset NIH ChestXray14."""

# ── Scheduler y stopping ───────────────────────────────────
EXPERT1_EARLY_STOPPING_PATIENCE = 20
"""Épocas sin mejora en val_macro_auc antes de detener el entrenamiento.
10 épocas es razonable para un dataset grande con augmentation."""

EXPERT1_EARLY_STOPPING_MONITOR = "val_macro_auc"
"""Métrica a monitorear para early stopping."""

# ── Resumen ejecutivo para logs ─────────────────────────────
EXPERT1_CONFIG_SUMMARY = (
    "Expert 1 (ChestXray14) Hybrid-Deep-Vision from scratch: "
    f"epochs={EXPERT1_EPOCHS} | LR={EXPERT1_LR} | "
    f"WD={EXPERT1_WEIGHT_DECAY} | dropout_fc={EXPERT1_DROPOUT_FC} | "
    f"batch={EXPERT1_BATCH_SIZE} | accum={EXPERT1_ACCUMULATION_STEPS} "
    f"(effective={EXPERT1_BATCH_SIZE * EXPERT1_ACCUMULATION_STEPS}) | "
    f"FP16={EXPERT1_FP16} (FocalLoss + alpha clamp=50) | "
    f"img={EXPERT1_IMG_SIZE} | "
    f"patience={EXPERT1_EARLY_STOPPING_PATIENCE}"
)

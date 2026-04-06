"""
Configuración de entrenamiento para Expert 5 — CAE multimodal (Fase 3).

Fuente de verdad para todos los hiperparámetros del CAE (expert_id=5).

Arquitectura: Convolutional AutoEncoder 2D entrenado desde cero (weights=None).
Tarea: detección OOD via error de reconstrucción — NO clasifica patologías.
Loss: MSE (reconstrucción pixel-a-pixel) + lambda * L1 (nitidez).
Métrica principal: val_mse (MSE de reconstrucción en validación).

Datos:
  - 5 datasets combinados: Chest, ISIC, OA, LUNA16, Páncreas
  - Datos 3D convertidos a slices 2D representativos
  - Todas las imágenes normalizadas a [3, 224, 224]
"""

# ── Optimizador ─────────────────────────────────────────────
EXPERT5_LR = 1e-3
"""Learning rate inicial para Adam. Valor estándar para autoencoders
convolucionales con batch normalization."""

EXPERT5_WEIGHT_DECAY = 1e-5
"""Weight decay (L2 regularization) ligero. Autoencoders no necesitan
regularización agresiva: el bottleneck ya restringe la capacidad."""

# ── Batch y entrenamiento ───────────────────────────────────
EXPERT5_BATCH_SIZE = 32
"""Batch size real por GPU. Con imágenes 224x224 RGB y el CAE 2D
en FP32, batch_size=32 cabe en ~4 GB VRAM."""

EXPERT5_ACCUMULATION_STEPS = 1
"""Gradient accumulation steps. Sin acumulación: batch efectivo = 32.
El CAE es ligero y no requiere batches grandes para convergencia."""

EXPERT5_FP16 = False
"""FP32 obligatorio. MSE de reconstrucción requiere precisión numérica:
una diferencia de 0.001 en MSE puede decidir OOD vs in-distribution.
Los errores de cuantización de FP16 pueden causar falsos positivos/negativos."""

# ── Scheduler y stopping ───────────────────────────────────
EXPERT5_MAX_EPOCHS = 100
"""Máximo de épocas. El early stopping detendrá antes si val_mse estanca.
100 es conservador para un dataset multi-modal de ~130K imágenes."""

EXPERT5_EARLY_STOPPING_PATIENCE = 15
"""Épocas sin mejora en val_mse antes de detener. 15 épocas es razonable
dado que ReduceLROnPlateau necesita margen para reducir el LR."""

EXPERT5_EARLY_STOPPING_MONITOR = "val_mse"
"""Métrica a monitorear para early stopping. val_mse es la métrica
natural del autoencoder — menor es mejor."""

# ── Arquitectura ────────────────────────────────────────────
EXPERT5_LATENT_DIM = 512
"""Dimensión del espacio latente. Compresión 150,528 -> 512 (ratio 294:1).
Suficiente para capturar la distribución de 5 modalidades médicas."""

EXPERT5_IMG_SIZE = 224
"""Tamaño de entrada/salida. Consistente con el resto del pipeline."""

EXPERT5_IN_CHANNELS = 3
"""Canales de entrada. RGB (o grayscale replicado a 3 canales)."""

EXPERT5_EXPERT_ID = 5
"""ID del experto en el sistema MoE."""

# ── Loss ────────────────────────────────────────────────────
EXPERT5_LOSS_LAMBDA_L1 = 0.1
"""Peso del término L1 adicional al MSE en la loss total:
    loss = MSE(recon, input) + lambda * L1(recon, input)
L1 promueve reconstrucciones más nítidas (menos borrosas)."""

# ── Scheduler ───────────────────────────────────────────────
EXPERT5_SCHEDULER_FACTOR = 0.5
"""Factor de reducción de LR en ReduceLROnPlateau."""

EXPERT5_SCHEDULER_PATIENCE = 5
"""Épocas sin mejora en val_mse antes de reducir el LR.
Más agresivo que early stopping (5 vs 15) para dar oportunidad
al LR reducido de mejorar antes de detener."""

# ── OOD threshold (post-entrenamiento) ─────────────────────
EXPERT5_OOD_THRESHOLD_PERCENTILE = 95
"""Percentil del MSE en val set para definir el umbral OOD.
Solo el 5% de las imágenes de validación con mayor error serían
consideradas OOD. Se calibra post-entrenamiento."""

# ── Resumen ejecutivo para logs ─────────────────────────────
EXPERT5_CONFIG_SUMMARY = (
    "Expert 5 (CAE multimodal): LR=1e-3 | WD=1e-5 | "
    "MSE + 0.1*L1 | latent=512 | "
    "batch=32 | accum=1 (efectivo=32) | FP32 | "
    "patience=15 | max_epochs=100"
)

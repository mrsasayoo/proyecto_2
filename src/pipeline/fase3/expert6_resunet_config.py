"""
Configuración de entrenamiento para Expert 6 — Res-U-Net Autoencoder (Fase 3).

Fuente de verdad para todos los hiperparámetros del Res-U-Net (expert_id=5,
versión 6 del modelo OOD). Reemplaza al CAE simple del Expert 5 con una
arquitectura más potente que incorpora skip connections y bloques residuales
pre-activación.

Arquitectura: Res-U-Net Autoencoder entrenado desde cero (weights=None).
Tarea: detección OOD via error de reconstrucción — NO clasifica patologías.
Loss: MSE + 0.1 * L1 (reconstrucción + nitidez).
Métrica principal: val_mse (MSE de reconstrucción en validación).

Datos:
  - 5 datasets combinados: Chest, ISIC, OA, LUNA16, Páncreas
  - Datos 3D convertidos a slices 2D representativos
  - Todas las imágenes normalizadas a [3, 224, 224]
"""

# ── Arquitectura ────────────────────────────────────────────
EXPERT6_BASE_CH: int = 64
"""Canales base del encoder. Las etapas escalan como base×{1,2,4,8}.
Con base_ch=64 el modelo tiene ~30M parámetros, suficiente para capturar
la distribución de 5 modalidades médicas sin overfitting."""

EXPERT6_DROPOUT: float = 0.1
"""Dropout en ResBlockPreAct del encoder y decoder. Ligero para evitar
underfitting en bloques residuales poco profundos."""

EXPERT6_BOTTLENECK_DROPOUT: float = 0.15
"""Dropout en los bloques residuales del bottleneck. Ligeramente mayor
que en encoder/decoder porque el bottleneck tiene 4 bloques consecutivos
y es más propenso a memorización."""

EXPERT6_IN_CHANNELS: int = 3
"""Canales de entrada. RGB (o grayscale replicado a 3 canales)."""

EXPERT6_IMG_SIZE: int = 224
"""Tamaño espacial de entrada/salida. Consistente con el resto del
pipeline MoE y los transforms de preprocesamiento."""

EXPERT6_LATENT_DIM: int = 512
"""Dimensión del vector latente producido por Bottleneck.to_latent
(AdaptiveAvgPool2d(1) → Flatten). Igual que Expert 5 para
compatibilidad con el pipeline de detección OOD."""

EXPERT6_EXPERT_ID: int = 5
"""Posición del experto en el ModuleList del MoE. Aunque el modelo
es la versión 6, ocupa la posición 5 (0-indexed) del sistema."""

EXPERT6_N_DOMAINS: int = 5
"""Número total de dominios para el FiLMGenerator.
Dominios 0-4: CXR14, ISIC, Panorama, LUNA16, OA Knee."""

EXPERT6_EMBED_DIM: int = 64
"""Dimensión del embedding de dominio en FiLMGenerator.
Valor pequeño (64) suficiente para distinguir 5 dominios sin sobreajustar."""

# ── Optimizador ─────────────────────────────────────────────
EXPERT6_LR: float = 1e-3
"""Learning rate inicial para Adam/AdamW. Valor estándar para
autoencoders con batch normalization y bloques residuales."""

EXPERT6_WEIGHT_DECAY: float = 1e-5
"""Weight decay (L2 regularization) ligero. El bottleneck y las skip
connections ya actúan como regularizadores implícitos."""

# ── Batch y entrenamiento ───────────────────────────────────
EXPERT6_BATCH_SIZE: int = 32
"""Batch size real por GPU. Con imágenes 224×224 RGB y el Res-U-Net
en FP32, batch_size=32 requiere ~8-10 GB VRAM (más que el CAE simple
debido a los feature maps de las skip connections)."""

EXPERT6_MAX_EPOCHS: int = 50
"""Máximo de épocas. Reducido respecto al Expert 5 (100) porque
la arquitectura más profunda con skip connections converge más
rápido. El early stopping detendrá antes si val_mse estanca."""

EXPERT6_EARLY_STOPPING_PATIENCE: int = 10
"""Épocas sin mejora en val_mse antes de detener. Reducido respecto
al Expert 5 (15) por la convergencia más rápida del Res-U-Net y
el uso de CosineAnnealing en lugar de ReduceLROnPlateau."""

EXPERT6_EARLY_STOPPING_MONITOR: str = "val_mse"
"""Métrica a monitorear para early stopping. val_mse es la métrica
natural del autoencoder — menor es mejor."""

EXPERT6_FP16: bool = False
"""FP32 estricto. MSE de reconstrucción requiere precisión numérica:
una diferencia de 0.001 en MSE puede decidir OOD vs in-distribution.
Los errores de cuantización de FP16 causan falsos positivos/negativos.
Mismo criterio que Expert 5."""

EXPERT6_ACCUMULATION_STEPS: int = 1
"""Gradient accumulation steps. Sin acumulación: batch efectivo = 32.
El Res-U-Net es más pesado que el CAE pero batch_size=32 es suficiente
para convergencia estable con BN en cada bloque."""

# ── Scheduler ───────────────────────────────────────────────
EXPERT6_T_MAX: int = 50
"""T_max para CosineAnnealingLR — número total de épocas del ciclo
coseno. Reemplaza ReduceLROnPlateau del Expert 5 por un schedule
determinístico que decae suavemente el LR hasta 0 en T_max épocas.
CosineAnnealing es preferible para redes profundas porque evita
mesetas prolongadas en la curva de aprendizaje."""

# ── Loss ────────────────────────────────────────────────────
EXPERT6_LOSS_LAMBDA_L1: float = 0.1
"""Peso del término L1 adicional al MSE en la loss total:
    loss = MSE(recon, input) + lambda * L1(recon, input)
L1 promueve reconstrucciones más nítidas. Mismo valor que Expert 5."""

# ── OOD thresholds (post-entrenamiento) ────────────────────
EXPERT6_OOD_THRESHOLD_PERCENTILE_LEVE: int = 50
"""Percentil 50 del MSE in-distribution para definir θ_leve.
Imágenes con error de reconstrucción por encima de este umbral
reciben un flag de OOD leve (sospechosas). Se calibra
post-entrenamiento sobre el validation set."""

EXPERT6_OOD_THRESHOLD_PERCENTILE_OOD: int = 99
"""Percentil 99 del MSE in-distribution para definir θ_OOD.
Imágenes con error de reconstrucción por encima de este umbral
se clasifican como OOD con alta confianza. Solo el 1% de las
imágenes in-distribution más anómalas superarían este umbral."""

# ── Resumen ejecutivo para logs ─────────────────────────────
EXPERT6_CONFIG_SUMMARY: str = (
    "Expert 6 (Res-U-Net Autoencoder + FiLM Domain Conditioning, pos=5): "
    "LR=1e-3 | WD=1e-5 | MSE + 0.1*L1 | latent=512 | base_ch=64 | "
    "n_domains=5 | embed_dim=64 | "
    "batch=32 | accum=1 (efectivo=32) | FP32 | "
    "CosineAnnealing T_max=50 | patience=10 | max_epochs=50 | "
    "OOD θ_leve=p50, θ_OOD=p99"
)
"""Resumen de configuración en una línea para logging al inicio
del entrenamiento. Facilita identificación rápida del experimento."""

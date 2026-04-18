"""
Configuración de entrenamiento para Expert 3 — LUNA16 (Nódulos Pulmonares CT 3D).

Fuente de verdad para todos los hiperparámetros del experto 3 en Fase 2.

Arquitectura: DenseNet 3D (~6.7M parámetros), implementado from scratch con
convoluciones 3D. growth_rate=32, blocks=[4,8,16,12], compression=0.5.

Análisis de riesgo de overfitting:
  - 14,728 muestras de entrenamiento (1,258 positivas, 13,470 negativas)
  - Ratio desbalance efectivo (post-fix leakage): ~10.7:1
  - Con DenseNet 3D (~6.7M params): ratio params/datos ≈ 455:1 — riesgo moderado
  - Regularización aplicada: SpatialDropout3d, dropout_fc, weight_decay, FocalLoss
"""

# ── Optimizador ─────────────────────────────────────────────
EXPERT3_LR = 3e-4
"""Learning rate inicial para AdamW. Valor conservador para fine-tuning 3D."""

EXPERT3_WEIGHT_DECAY = 0.03
"""Weight decay (L2 regularization) en AdamW. Más alto que el default (0.01)
para combatir overfitting con ratio params/datos alto."""

# ── Loss ────────────────────────────────────────────────────
EXPERT3_FOCAL_GAMMA = 2.0
"""Exponente de modulación de FocalLoss. gamma=2 es el valor de referencia
de Lin et al. 2017."""

EXPERT3_FOCAL_ALPHA = 0.85
"""Peso de la clase positiva (nódulos) en FocalLoss.
CORRECCIÓN: alpha=0.85 pondera la clase positiva (nódulos, minoritaria).
Con ratio 10:1 neg:pos en disco (n_neg/n_total = 13470/14728 ≈ 0.915),
alpha=0.85 es una aproximación conservadora que penaliza más los falsos
negativos sin ignorar completamente los falsos positivos."""

EXPERT3_LABEL_SMOOTHING = 0.05
"""Label smoothing para regularizar las predicciones del modelo.
Convierte labels {0,1} → {0.025, 0.975}, previniendo sobreconfianza
en predicciones. Valor bajo porque el desbalance ya es extremo."""

# ── Regularización del modelo ───────────────────────────────
EXPERT3_DROPOUT_FC = 0.4
"""Dropout en la capa fully-connected final. Valor alto (0.4 vs 0.1 típico)
porque el ratio params/datos es muy desfavorable."""

EXPERT3_SPATIAL_DROPOUT_3D = 0.15
"""Spatial dropout 3D (SpatialDropout3d) en las capas convolucionales.
Desactiva canales completos en lugar de activaciones individuales,
más efectivo para datos volumétricos donde las activaciones son
espacialmente correlacionadas."""

# ── Batch y entrenamiento ───────────────────────────────────
EXPERT3_BATCH_SIZE = 8
"""Batch size real por GPU. Con Titan Xp (12 GB) y volúmenes 64³ en FP16,
batch_size=8 utiliza ~4-5 GB VRAM — margen amplio sin gradient checkpointing."""

EXPERT3_ACCUMULATION_STEPS = 4
"""Gradient accumulation steps. Batch efectivo = batch_size × accumulation_steps
= 8 × 4 = 32. Mismo batch efectivo que antes (4 × 8 = 32).
NOTA: ACCUMULATION_STEPS=4 es el mínimo obligatorio del proyecto."""

EXPERT3_FP16 = True
"""Usar mixed precision (torch.cuda.amp). Obligatorio para 12 GB VRAM con
volúmenes 3D — reduce consumo de VRAM ~50% sin pérdida significativa de
precisión numérica."""

# ── Scheduler y stopping ───────────────────────────────────
EXPERT3_MAX_EPOCHS = 100
"""Máximo de épocas. El early stopping detendrá antes si val_loss estanca."""

EXPERT3_EARLY_STOPPING_PATIENCE = 20
"""Épocas sin mejora en val_loss antes de detener el entrenamiento.
20 épocas es conservador dado que el augmentation 3D introduce
variabilidad entre épocas."""

EXPERT3_EARLY_STOPPING_MONITOR = "val_loss"
"""Métrica a monitorear para early stopping. val_loss en lugar de val_auc
porque con desbalance extremo, la loss es más estable que métricas
derivadas de un threshold."""

# ── Resumen ejecutivo para logs ─────────────────────────────
EXPERT3_CONFIG_SUMMARY = (
    "Expert 3 (LUNA16 / DenseNet3D): LR=3e-4 | WD=0.03 | "
    "FocalLoss(γ=2, α=0.85) | label_smooth=0.05 | "
    "dropout_fc=0.4 | spatial_drop3d=0.15 | "
    "batch=8 | accum=4 (efectivo=32) | FP16=True | "
    "patience=20 | max_epochs=100"
)

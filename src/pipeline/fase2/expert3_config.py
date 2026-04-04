"""
Configuración de entrenamiento para Expert 3 — LUNA16 (Nódulos Pulmonares CT 3D).

Fuente de verdad para todos los hiperparámetros del experto 3 en Fase 2.
Este archivo NO define la arquitectura del modelo (ViViT-Tiny vs MC3-18);
esa decisión está pendiente del usuario.

NOTA ARQUITECTÓNICA: La spec prescribe ViViT-Tiny 3D (~25M parámetros).
Con el dataset actual (14,728 muestras, ratio params/datos ≈ 2,038:1),
se recomienda considerar MC3-18 (~11.2M params, ratio ≈ 761:1) como alternativa
antes de Phase 2. Ver docs/overfitting_analysis_expert3.md para análisis completo.

Análisis de riesgo de overfitting:
  - 14,728 muestras de entrenamiento (1,258 positivas, 13,470 negativas)
  - Ratio desbalance efectivo (post-fix leakage): ~10.7:1
  - Con ViViT-Tiny (~25M params): ratio params/datos ≈ 1,700:1 → ALTO RIESGO
  - Regularización agresiva obligatoria: dropout, weight_decay, augmentation 3D
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
EXPERT3_BATCH_SIZE = 4
"""Batch size real por GPU. Limitado por VRAM (12 GB) con volúmenes 64³.
Con FP16 + gradient checkpointing, batch_size=4 es viable."""

EXPERT3_ACCUMULATION_STEPS = 8
"""Gradient accumulation steps. Batch efectivo = batch_size × accumulation_steps
= 4 × 8 = 32. Simula un batch mayor sin requerir más VRAM.
NOTA: ACCUMULATION_STEPS=4 es el mínimo obligatorio del proyecto;
aquí usamos 8 para mayor estabilidad del gradiente con FocalLoss."""

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
    "Expert 3 (LUNA16): LR=3e-4 | WD=0.03 | "
    "FocalLoss(γ=2, α=0.85) | label_smooth=0.05 | "
    "dropout_fc=0.4 | spatial_drop3d=0.15 | "
    "batch=4 | accum=8 (efectivo=32) | FP16=True | "
    "patience=20 | max_epochs=100"
)

"""
expert2_config.py — Configuración de entrenamiento para Expert 2 (ISIC 2019)
=============================================================================

Pipeline de 3 fases con descongelamiento progresivo del backbone ConvNeXt-Small:

  Fase 1 (épocas  1-5):   Solo head, backbone congelado.
                           CosineAnnealingLR.
  Fase 2 (épocas  6-20):  Fine-tuning diferencial (head + backbone con LR distintos).
                           CosineAnnealingWarmRestarts.
  Fase 3 (épocas 21-40):  Full fine-tuning + early stopping.
                           CosineAnnealingWarmRestarts.

Dataset ISIC 2019 — 8 clases de entrenamiento:
  MEL  (Melanoma)
  NV   (Melanocytic nevus)
  BCC  (Basal cell carcinoma)
  AK   (Actinic keratosis)
  BKL  (Benign keratosis)
  DF   (Dermatofibroma)
  VASC (Vascular lesion)
  SCC  (Squamous cell carcinoma)
"""

# ── GENERAL ──────────────────────────────────────────────────────────────────
EXPERT2_NUM_CLASSES: int = 8
EXPERT2_IMG_SIZE: int = 224
EXPERT2_BATCH_SIZE: int = 32
EXPERT2_ACCUMULATION_STEPS: int = 3  # batch efectivo = 96
EXPERT2_NUM_WORKERS: int = 4
EXPERT2_LABEL_SMOOTHING: float = 0.1
EXPERT2_CHECKPOINT_DIR: str = "checkpoints/expert_01_convnext_small"
EXPERT2_CHECKPOINT_NAME: str = "expert2_best.pt"
EXPERT2_MONITOR: str = "val_f1_macro"  # métrica para checkpoint

# ── FASE 1 (épocas 1-5): solo head, backbone congelado ──────────────────────
EXPERT2_PHASE1_EPOCHS: int = 5
EXPERT2_PHASE1_LR: float = 3e-4
EXPERT2_PHASE1_WD: float = 1e-4
EXPERT2_PHASE1_ETA_MIN: float = 3e-5  # CosineAnnealingLR eta_min

# ── FASE 2 (épocas 6-20): fine-tuning diferencial ───────────────────────────
EXPERT2_PHASE2_EPOCHS: int = 15
EXPERT2_PHASE2_HEAD_LR: float = 3e-4
EXPERT2_PHASE2_BACKBONE_LR: float = 1e-5
EXPERT2_PHASE2_WD: float = 1e-4
EXPERT2_PHASE2_T0: int = 10  # CosineAnnealingWarmRestarts T_0
EXPERT2_PHASE2_T_MULT: int = 2
EXPERT2_PHASE2_ETA_MIN: float = 1e-7

# ── FASE 3 (épocas 21-40): full fine-tuning + early stopping ────────────────
EXPERT2_PHASE3_EPOCHS: int = 20
EXPERT2_PHASE3_HEAD_LR: float = 1e-4
EXPERT2_PHASE3_BACKBONE_LR: float = 5e-6
EXPERT2_PHASE3_WD: float = 1e-4
EXPERT2_PHASE3_T0: int = 10  # CosineAnnealingWarmRestarts T_0
EXPERT2_PHASE3_T_MULT: int = 2
EXPERT2_PHASE3_ETA_MIN: float = 1e-7
EXPERT2_EARLY_STOPPING_PATIENCE: int = 8

# ── Derivada ─────────────────────────────────────────────────────────────────
EXPERT2_TOTAL_EPOCHS: int = (
    EXPERT2_PHASE1_EPOCHS + EXPERT2_PHASE2_EPOCHS + EXPERT2_PHASE3_EPOCHS
)

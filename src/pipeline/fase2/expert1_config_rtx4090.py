"""
Configuración optimizada para RTX 4090 (24GB VRAM, Ada Lovelace, Tensor Cores 4th gen).

Drop-in replacement para expert1_config.py cuando se entrena en RTX 4090.
Diferencias vs Titan Xp (expert1_config.py):
  - batch_size: 48 → 128 (2× VRAM disponible)
  - accum_steps: 2 → 1 (no necesario con batch grande)
  - num_workers: 8 → 12 (Ada tiene mejor throughput PCIe)
  - num_epochs: 100 → 20 (validación rápida / dry-run extendido)
  - torch_compile: True (Ada soporta torch.compile con speedup real)

Batch efectivo: 128 × 1 = 128 (vs 48 × 2 = 96 en Titan Xp).
VRAM estimado: ~14-16 GB con FP16 (24 GB disponibles → margen amplio).
"""

# ── Épocas ──────────────────────────────────────────────────
EXPERT1_EPOCHS = 20
"""20 épocas para validación rápida en RTX 4090. Subir a 100 para producción."""

# ── Learning rate ───────────────────────────────────────────
EXPERT1_LR = 3e-4
"""Sin cambios: LR independiente de GPU."""

# ── Regularización del modelo ───────────────────────────────
EXPERT1_WEIGHT_DECAY = 1e-4
EXPERT1_DROPOUT_FC = 0.4

# ── Batch y entrenamiento (OPTIMIZADO RTX 4090) ────────────
EXPERT1_BATCH_SIZE = 128
"""Batch total. Con 1 GPU RTX 4090 → 128 per GPU.
Con 2 GPUs → 64 per GPU. VRAM 24GB permite holgadamente."""

EXPERT1_NUM_WORKERS = 12
"""12 workers para saturar RTX 4090 + NVMe."""

EXPERT1_ACCUMULATION_STEPS = 1
"""Sin acumulación: batch 128 ya es suficiente como batch efectivo."""

EXPERT1_FP16 = True
"""FP16 obligatorio en RTX 4090 para aprovechar Tensor Cores FP16."""

# ── torch.compile ───────────────────────────────────────────
EXPERT1_TORCH_COMPILE = True
"""Habilitar torch.compile() para Ada Lovelace. Speedup ~10-30% tras warmup."""

# ── Imagen y clases ─────────────────────────────────────────
EXPERT1_IMG_SIZE = 256
EXPERT1_NUM_CLASSES = 14

# ── Scheduler y stopping ───────────────────────────────────
EXPERT1_EARLY_STOPPING_PATIENCE = 10
"""Patience reducido para 20 épocas."""

EXPERT1_EARLY_STOPPING_MONITOR = "val_macro_auc"

# ── Resumen ─────────────────────────────────────────────────
EXPERT1_CONFIG_SUMMARY = (
    "Expert 1 (ChestXray14) Hybrid-Deep-Vision RTX4090: "
    f"epochs={EXPERT1_EPOCHS} | LR={EXPERT1_LR} | "
    f"WD={EXPERT1_WEIGHT_DECAY} | dropout_fc={EXPERT1_DROPOUT_FC} | "
    f"batch={EXPERT1_BATCH_SIZE} | accum={EXPERT1_ACCUMULATION_STEPS} "
    f"(effective={EXPERT1_BATCH_SIZE * EXPERT1_ACCUMULATION_STEPS}) | "
    f"FP16={EXPERT1_FP16} | torch.compile={EXPERT1_TORCH_COMPILE} | "
    f"img={EXPERT1_IMG_SIZE} | "
    f"patience={EXPERT1_EARLY_STOPPING_PATIENCE}"
)

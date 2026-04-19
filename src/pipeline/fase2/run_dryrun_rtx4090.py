#!/usr/bin/env python3
"""
Dry-run wrapper para RTX 4090 — parchea expert1_config en runtime.

Uso:
    torchrun --nproc_per_node=1 src/pipeline/fase2/run_dryrun_rtx4090.py

No modifica archivos existentes. Aplica la configuración RTX 4090 via monkey-patch
antes de importar el script de entrenamiento.
"""

import sys
import time
from pathlib import Path

# ── Setup path ─────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

# ── Monkey-patch expert1_config ANTES de importar train ────────────────
import fase2.expert1_config as cfg

# RTX 4090 overrides
cfg.EXPERT1_BATCH_SIZE = 128
cfg.EXPERT1_ACCUMULATION_STEPS = 1
cfg.EXPERT1_NUM_WORKERS = 12
cfg.EXPERT1_EPOCHS = 20
cfg.EXPERT1_EARLY_STOPPING_PATIENCE = 10
cfg.EXPERT1_CONFIG_SUMMARY = (
    f"Expert 1 RTX4090: epochs={cfg.EXPERT1_EPOCHS} | "
    f"batch={cfg.EXPERT1_BATCH_SIZE} | accum={cfg.EXPERT1_ACCUMULATION_STEPS} | "
    f"workers={cfg.EXPERT1_NUM_WORKERS} | FP16={cfg.EXPERT1_FP16}"
)

# ── Ahora importar train (usará los valores parcheados) ───────────────
import torch

# Detectar GPU y reportar
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"\n[RTX4090-DryRun] GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print(f"[RTX4090-DryRun] Config: batch=128, accum=1, workers=12, epochs=20")

    # Auto-ajustar batch si VRAM < 20 GB (no es RTX 4090)
    if vram_gb < 20:
        cfg.EXPERT1_BATCH_SIZE = 48
        cfg.EXPERT1_ACCUMULATION_STEPS = 2
        cfg.EXPERT1_NUM_WORKERS = 8
        print(f"[RTX4090-DryRun] WARNING: VRAM={vram_gb:.1f}GB < 20GB, "
              f"fallback a batch=48, accum=2")

# torch.compile — aplicar al modelo después de creación
_original_train = None

def _apply_torch_compile_if_available():
    """Intenta habilitar torch.compile si PyTorch >= 2.0 y CUDA disponible."""
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        if major >= 8:  # Ampere+ (Ada = 8.9)
            return True
    return False

USE_COMPILE = _apply_torch_compile_if_available()
if USE_COMPILE:
    print(f"[RTX4090-DryRun] torch.compile: ENABLED (compute capability {torch.cuda.get_device_capability(0)})")
else:
    print(f"[RTX4090-DryRun] torch.compile: DISABLED")

# ── Import y ejecución ─────────────────────────────────────────────────
from fase2.train_expert1_ddp import train, HybridDeepVision
import fase2.train_expert1_ddp as train_module

# Monkey-patch para inyectar torch.compile en el modelo
_original_hdv_init = HybridDeepVision.__init__

def _patched_init(self, *args, **kwargs):
    _original_hdv_init(self, *args, **kwargs)
    # El compile se aplica después de .to(device) en train()

if USE_COMPILE:
    # Patch the train function to compile the model
    _orig_wrap = train_module.wrap_model_ddp

    def _wrap_with_compile(model, device, **kwargs):
        print(f"[RTX4090-DryRun] Applying torch.compile() to model...")
        t0 = time.time()
        model = torch.compile(model, mode="reduce-overhead")
        print(f"[RTX4090-DryRun] torch.compile() setup: {time.time()-t0:.1f}s")
        return _orig_wrap(model, device, **kwargs)

    train_module.wrap_model_ddp = _wrap_with_compile

# ── Ejecutar dry-run ───────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()
    train(dry_run=True)
    elapsed = time.time() - t_start
    print(f"\n[RTX4090-DryRun] Total dry-run time: {elapsed:.1f}s")

    # Reporte de VRAM
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated(0) / 1e6
        peak_gb = peak_mb / 1000
        print(f"[RTX4090-DryRun] Peak GPU memory: {peak_gb:.2f} GB")
        print(f"[RTX4090-DryRun] Status: {'PASS' if peak_gb < 20 else 'WARN: >20GB'}")

# Paso 4.1 — Entrenamiento End-to-End de Backbones: Plan de Implementación

> **Para el agente ejecutor:** implementar exactamente lo que está aquí. No interpretar, no reorganizar clases enteras. Ediciones quirúrgicas. Leer cada archivo antes de editarlo.

**Goal:** Crear el pipeline de entrenamiento end-to-end de los 4 backbones (`fase1_train_pipeline.py`) con soporte `--dry-run`, sin ejecutar entrenamiento real en producción.

**Architecture:**
- Tarea proxy: clasificación de dominio médico (5 clases = 5 expertos, expert_id como label)
- Pipeline: todos los 5 datasets combinados → backbone (sin congelar) + LinearHead(d_model→5) → CrossEntropyLoss
- Checkpoint: solo pesos del backbone (sin head) en `checkpoints/backbone_0X_<nombre>/backbone.pth`
- Después del entrenamiento, `fase1_pipeline.py` (Paso 4.2) carga ese checkpoint y lo congela

**Tech Stack:** PyTorch, timm, HuggingFace transformers (para CvT-13), AdamW + CosineAnnealingLR

---

## Restricciones del proyecto (NO violar)
- PROHIBIDO pesos preentrenados (timm/HuggingFace/torchvision)
- PROHIBIDO transformaciones de oclusión (RandomErasing, Cutout, etc.)
- Ediciones quirúrgicas — no reestructurar módulos completos
- `proyecto_moe.md` es la fuente de verdad

## Mapeo: archivos a tocar

| Acción | Archivo |
|--------|---------|
| **Modificar** | `src/pipeline/fase1/fase1_config.py` |
| **Modificar** | `src/pipeline/fase1/backbone_loader.py` |
| **Modificar** | `src/pipeline/fase1/backbone_cvt13.py` |
| **Modificar** | `src/pipeline/fase1/backbone_densenet.py` |
| **Modificar** | `src/pipeline/fase1/fase1_pipeline.py` |
| **Crear** | `src/pipeline/fase1/backbone_trainer.py` |
| **Crear** | `src/pipeline/fase1/fase1_train_pipeline.py` |

---

## Task 1: Añadir constantes de entrenamiento a `fase1_config.py`

**File:** `src/pipeline/fase1/fase1_config.py`

Añadir al final del archivo (después de `BACKBONE_META_KEYS`):

```python
# ── Entrenamiento end-to-end de backbones (Paso 4.1) ──────────
# Hiperparámetros para la tarea proxy de clasificación de dominio
TRAIN_EPOCHS = 20            # épocas — suficiente para proxy de dominio
TRAIN_LR = 3e-4              # AdamW default
TRAIN_WEIGHT_DECAY = 0.01    # L2 regularización
TRAIN_WARMUP_EPOCHS = 2      # épocas de warm-up lineal del LR
TRAIN_BATCH_SIZE = 64        # igual que DEFAULT_BATCH_SIZE
TRAIN_WORKERS = 4            # igual que DEFAULT_WORKERS
TRAIN_GRAD_CLIP = 1.0        # gradient clipping max norm

# Mapeo backbone_name → subdirectorio en checkpoints/
BACKBONE_TO_CHECKPOINT_DIR = {
    "vit_tiny_patch16_224":        "backbone_01_vit_tiny",
    "cvt_13":                      "backbone_02_cvt13",
    "swin_tiny_patch4_window7_224": "backbone_03_swin_tiny",
    "densenet121_custom":          "backbone_04_densenet121",
}

# Nombre del archivo de checkpoint dentro del subdirectorio
BACKBONE_CHECKPOINT_FILENAME = "backbone.pth"
```

---

## Task 2: Añadir soporte de checkpoint y modelo entrenable en `backbone_loader.py`

**File:** `src/pipeline/fase1/backbone_loader.py`

**Acción:** Añadir DOS funciones nuevas al final del archivo (después de `load_frozen_backbone`).

### Función 1: `load_trainable_backbone`

```python
def load_trainable_backbone(backbone_name="vit_tiny_patch16_224", device="cuda"):
    """
    Construye un backbone en modo entrenamiento (sin congelar parámetros).

    Usado exclusivamente por fase1_train_pipeline.py (Paso 4.1).
    Para extracción de embeddings (Paso 4.2), usar load_frozen_backbone().

    Para CvT-13 y DenseNet llama directamente a las funciones build_*
    para evitar los interceptores de timm que congelan automáticamente.

    Returns:
        model   — backbone en model.train(), requires_grad=True en todos los params
        d_model — dimensión del embedding de salida
    """
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(
            f"Backbone '{backbone_name}' no reconocido.\n"
            f"Opciones válidas: {list(BACKBONE_CONFIGS.keys())}"
        )

    expected_d = BACKBONE_CONFIGS[backbone_name]["d_model"]
    log.info("[Backbone/train] Construyendo entrenable: %s", backbone_name)

    if backbone_name == "cvt_13":
        # Bypass del interceptor timm (que congela). Usar build_cvt13_trainable directamente.
        from backbone_cvt13 import build_cvt13_trainable
        model = build_cvt13_trainable(device=device)

    elif backbone_name == "densenet121_custom":
        # Bypass del interceptor timm (que congela). Usar build_densenet directamente.
        from backbone_densenet import build_densenet
        model = build_densenet(in_channels=3, embed_dim=1024, growth_rate=32,
                               block_config=(6, 12, 24, 16))
        model.to(device)

    else:
        # ViT-Tiny, Swin-Tiny: timm.create_model estándar (sin congelar)
        model = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
        )
        model.to(device)

    # Asegurar modo train y requires_grad=True
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    # Verificar que al menos un parámetro es entrenable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("[Backbone/train] Parámetros entrenables: %s / %s",
             f"{trainable:,}", f"{total:,}")

    # Forward dummy para verificar d_model
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    model.train()

    actual_d = out.shape[1]
    if actual_d != expected_d:
        log.warning(
            "[Backbone/train] d_model real (%d) difiere del esperado (%d).",
            actual_d, expected_d,
        )

    log.info("[Backbone/train] d_model: %d | listo para entrenamiento ✓", actual_d)
    return model, actual_d
```

### Función 2: `load_frozen_backbone_from_checkpoint`

```python
def load_frozen_backbone_from_checkpoint(backbone_name, checkpoint_path,
                                          device="cuda"):
    """
    Carga un backbone ya entrenado desde checkpoint y lo congela para extracción.

    Usado por fase1_pipeline.py (Paso 4.2) cuando el checkpoint del Paso 4.1 existe.

    El checkpoint tiene la estructura:
        {
            "backbone_name": str,
            "epoch": int,
            "val_acc": float,
            "state_dict": OrderedDict,
        }

    Returns:
        model   — backbone en .eval() con todos los params congelados
        d_model — dimensión del embedding verificada empíricamente
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"[Backbone] Checkpoint no encontrado: {ckpt_path}\n"
            f"Ejecuta primero: python src/pipeline/fase1/fase1_train_pipeline.py "
            f"--backbone {backbone_name}"
        )

    log.info("[Backbone] Cargando checkpoint: %s", ckpt_path)

    # Construir arquitectura desde cero
    model, d_model = load_trainable_backbone(backbone_name, device)

    # Cargar pesos del checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    stored_name = ckpt.get("backbone_name", "desconocido")
    if stored_name != backbone_name:
        log.warning(
            "[Backbone] Checkpoint es de '%s', se solicitó '%s'.",
            stored_name, backbone_name,
        )
    model.load_state_dict(ckpt["state_dict"])
    val_acc = ckpt.get("val_acc", 0.0)
    epoch = ckpt.get("epoch", -1)
    log.info("[Backbone] Checkpoint cargado: epoch=%d, val_acc=%.4f", epoch, val_acc)

    # Congelar (mismo comportamiento que load_frozen_backbone)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    log.info("[Backbone] Backbone cargado desde checkpoint y congelado ✓")
    return model, d_model
```

**IMPORTANTE:** Añadir `from pathlib import Path` al bloque de imports de `backbone_loader.py` si no existe ya.

---

## Task 3: Añadir `build_cvt13_trainable` en `backbone_cvt13.py`

**File:** `src/pipeline/fase1/backbone_cvt13.py`

Añadir después de la función `build_cvt13` (antes de `_register_cvt13_interceptor`):

```python
def build_cvt13_trainable(
    device: str = "cuda",
) -> CvT13Wrapper:
    """
    Construye CvT-13 desde cero en modo ENTRENABLE (sin congelar parámetros).

    Usado exclusivamente por backbone_loader.load_trainable_backbone() durante
    el entrenamiento end-to-end del Paso 4.1. Para extracción (Paso 4.2), usar
    build_cvt13() que devuelve el modelo congelado.

    Returns:
        CvT13Wrapper en modo train, todos los parámetros con requires_grad=True.
    """
    from transformers import CvtConfig, CvtModel

    log.info("[CvT-13/train] Construyendo desde cero (entrenable)")

    config = CvtConfig(
        num_channels=3,
        patch_sizes=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        embed_dim=[64, 192, 384],
        num_heads=[1, 3, 6],
        depth=[1, 2, 10],
        mlp_ratio=[4.0, 4.0, 4.0],
        attention_drop_rate=[0.0, 0.0, 0.0],
        drop_rate=[0.0, 0.0, 0.0],
        drop_path_rate=[0.0, 0.0, 0.1],
        qkv_bias=[True, True, True],
        cls_token=[False, False, True],
    )

    hf_model = CvtModel(config)
    wrapper = CvT13Wrapper(hf_model)

    # NO congelar — modo entrenamiento
    for param in wrapper.parameters():
        param.requires_grad = True
    wrapper.train()
    wrapper.to(device)

    total = sum(p.numel() for p in wrapper.parameters())
    log.info("[CvT-13/train] Parámetros entrenables: %s ✓", f"{total:,}")
    return wrapper
```

---

## Task 4: Crear `backbone_trainer.py`

**File:** `src/pipeline/fase1/backbone_trainer.py` (NUEVO)

```python
"""
backbone_trainer.py — Entrenamiento End-to-End del Backbone
===========================================================

Responsabilidad única: recibir un backbone entrenable y DataLoaders,
entrenar con CrossEntropyLoss (clasificación de dominio proxy),
y guardar únicamente los pesos del backbone (sin la cabeza lineal).

No gestiona argumentos CLI ni construye datasets — eso lo hace
fase1_train_pipeline.py.
"""

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from fase1_config import (
    TRAIN_GRAD_CLIP,
    TRAIN_WARMUP_EPOCHS,
)

log = logging.getLogger("fase1")


# ══════════════════════════════════════════════════════════════
#  Cabeza lineal para clasificación de dominio proxy
# ══════════════════════════════════════════════════════════════

class LinearHead(nn.Module):
    """
    Cabeza de clasificación lineal sobre el CLS token del backbone.
    Se descarta después del entrenamiento — solo se guardan los pesos
    del backbone.
    """
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, d_model] → logits: [B, n_classes]"""
        return self.fc(z)


# ══════════════════════════════════════════════════════════════
#  Utilidades de checkpoint
# ══════════════════════════════════════════════════════════════

def backbone_checkpoint_exists(checkpoint_path: str) -> bool:
    """Verifica si el checkpoint del backbone existe en disco."""
    return Path(checkpoint_path).exists()


def save_backbone_checkpoint(
    backbone: nn.Module,
    checkpoint_path: str,
    backbone_name: str,
    epoch: int,
    val_acc: float,
) -> None:
    """
    Guarda únicamente los pesos del backbone (sin la cabeza lineal).

    Estructura del checkpoint:
        {
            "backbone_name": str,
            "epoch": int,
            "val_acc": float,
            "state_dict": OrderedDict,
        }
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "backbone_name": backbone_name,
        "epoch": epoch,
        "val_acc": val_acc,
        "state_dict": backbone.state_dict(),
    }
    torch.save(ckpt, path)
    size_mb = path.stat().st_size / 1e6
    log.info(
        "[Trainer] Checkpoint guardado: %s (%.1f MB) | epoch=%d val_acc=%.4f",
        path, size_mb, epoch, val_acc,
    )


# ══════════════════════════════════════════════════════════════
#  Warm-up lineal del LR
# ══════════════════════════════════════════════════════════════

def _warmup_lr_scale(epoch: int, warmup_epochs: int) -> float:
    """Escala lineal del LR durante warm-up. Retorna 1.0 una vez terminado."""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / warmup_epochs)


# ══════════════════════════════════════════════════════════════
#  Bucle de entrenamiento — una época
# ══════════════════════════════════════════════════════════════

def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """
    Entrena una época completa.

    Returns:
        avg_loss — CrossEntropyLoss promedio sobre el epoch
        accuracy — accuracy de dominio [0, 1]
    """
    backbone.train()
    head.train()

    total_loss = 0.0
    correct = 0
    total = 0
    t_start = time.time()

    for batch_idx, (imgs, expert_ids, _names) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = expert_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        z = backbone(imgs)            # [B, d_model]
        logits = head(z)              # [B, N_EXPERTS_DOMAIN]
        loss = criterion(logits, labels)

        loss.backward()

        # Gradient clipping
        if TRAIN_GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()),
                TRAIN_GRAD_CLIP,
            )

        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        if batch_idx % 100 == 0:
            elapsed = time.time() - t_start
            speed = (batch_idx + 1) * loader.batch_size / elapsed if elapsed > 0 else 0
            log.info(
                "[Trainer] Epoch %d/%d | Batch %4d/%d | loss=%.4f | %.0f img/s",
                epoch + 1, total_epochs,
                batch_idx, len(loader),
                loss.item(), speed,
            )

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════
#  Bucle de validación
# ══════════════════════════════════════════════════════════════

def validate(
    backbone: nn.Module,
    head: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evalúa el modelo sobre el split de validación.

    Returns:
        avg_loss, accuracy
    """
    backbone.eval()
    head.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for imgs, expert_ids, _names in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = expert_ids.to(device, non_blocking=True)

            z = backbone(imgs)
            logits = head(z)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════
#  Entrenador completo
# ══════════════════════════════════════════════════════════════

def train_backbone(
    backbone: nn.Module,
    d_model: int,
    backbone_name: str,
    train_loader,
    val_loader,
    device: torch.device,
    checkpoint_path: str,
    cfg: dict,
) -> dict:
    """
    Entrena backbone + LinearHead end-to-end para clasificación de dominio.

    Args:
        backbone      — modelo entrenable (de load_trainable_backbone)
        d_model       — dimensión del CLS token del backbone
        backbone_name — nombre del backbone (para el checkpoint)
        train_loader  — DataLoader del split train (mode="embedding")
        val_loader    — DataLoader del split val (mode="embedding")
        device        — torch.device
        checkpoint_path — ruta donde guardar backbone.pth
        cfg           — dict con: epochs, lr, weight_decay, warmup_epochs

    Returns:
        dict con métricas: best_val_acc, best_epoch, train_history, val_history
    """
    from config import N_EXPERTS_DOMAIN

    epochs = cfg.get("epochs", 20)
    lr = cfg.get("lr", 3e-4)
    weight_decay = cfg.get("weight_decay", 0.01)
    warmup_epochs = cfg.get("warmup_epochs", TRAIN_WARMUP_EPOCHS)

    head = LinearHead(d_model, N_EXPERTS_DOMAIN).to(device)
    criterion = nn.CrossEntropyLoss()

    all_params = list(backbone.parameters()) + list(head.parameters())
    optimizer = AdamW(all_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_acc = 0.0
    best_epoch = 0
    train_history = []
    val_history = []

    log.info("=" * 60)
    log.info("[Trainer] Inicio entrenamiento backbone: %s", backbone_name)
    log.info("[Trainer] epochs=%d | lr=%.2e | wd=%.4f | warmup=%d",
             epochs, lr, weight_decay, warmup_epochs)
    log.info("[Trainer] checkpoint → %s", checkpoint_path)
    log.info("=" * 60)

    t_total = time.time()

    for epoch in range(epochs):
        # ── Warm-up lineal del LR ──
        if epoch < warmup_epochs:
            scale = _warmup_lr_scale(epoch, warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * scale
            log.debug("[Trainer] Warm-up epoch %d: lr=%.2e", epoch, lr * scale)

        # ── Train ──
        t_ep = time.time()
        train_loss, train_acc = train_one_epoch(
            backbone, head, train_loader, optimizer, criterion, device,
            epoch, epochs,
        )

        # ── Validate ──
        val_loss, val_acc = validate(backbone, head, val_loader, criterion, device)

        # ── Scheduler step (después del warm-up) ──
        if epoch >= warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed_ep = time.time() - t_ep

        log.info(
            "[Trainer] Epoch %3d/%d | train_loss=%.4f acc=%.4f | "
            "val_loss=%.4f acc=%.4f | lr=%.2e | %.1fs",
            epoch + 1, epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            current_lr, elapsed_ep,
        )

        train_history.append({"epoch": epoch + 1, "loss": train_loss, "acc": train_acc})
        val_history.append({"epoch": epoch + 1, "loss": val_loss, "acc": val_acc})

        # ── Guardar mejor checkpoint ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_backbone_checkpoint(
                backbone, checkpoint_path, backbone_name, epoch + 1, val_acc,
            )
            log.info(
                "[Trainer] ★ Nuevo mejor val_acc=%.4f (epoch %d) — checkpoint guardado",
                val_acc, epoch + 1,
            )

    total_time = time.time() - t_total
    log.info("=" * 60)
    log.info("[Trainer] Entrenamiento completado en %.1f min", total_time / 60)
    log.info("[Trainer] Mejor val_acc=%.4f (epoch %d)", best_val_acc, best_epoch)
    log.info("=" * 60)

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_time_s": total_time,
        "train_history": train_history,
        "val_history": val_history,
    }
```

---

## Task 5: Crear `fase1_train_pipeline.py`

**File:** `src/pipeline/fase1/fase1_train_pipeline.py` (NUEVO)

Seguir exactamente el mismo patrón de `fase1_pipeline.py`:
- path setup idéntico
- mismo bloque de imports
- mismos argumentos de datasets (copiar los `add_argument` de datasets)
- `--dry-run`: instanciar modelo + head, UN forward pass con un batch, imprimir config y salir
- `--force`: reentrenar aunque checkpoint exista
- idempotencia: saltar si checkpoint existe y `--force` no activo

```python
"""
fase1_train_pipeline.py — Orquestador del Paso 4.1: Entrenamiento End-to-End
============================================================================

Entrena cada backbone desde cero sobre los 5 datasets de dominio usando
clasificación de dominio (expert_id) como tarea proxy.

Después del entrenamiento, el backbone congelado se usa en Paso 4.2
(fase1_pipeline.py) para extraer CLS tokens.

Uso:
    # Verificar sin entrenar:
    python src/pipeline/fase1/fase1_train_pipeline.py --backbone vit_tiny_patch16_224 --dry-run

    # Entrenar (producción):
    python src/pipeline/fase1/fase1_train_pipeline.py --backbone vit_tiny_patch16_224
"""

import os
import sys
import argparse
import time
import logging
import datetime
import json
from pathlib import Path

import pandas as pd

# ── Path setup (IDÉNTICO a fase1_pipeline.py) ──────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _THIS_DIR.parent
_SRC_DIR = _PIPELINE_DIR.parent
_PROJECT_ROOT = _SRC_DIR.parent

for _p in [str(_THIS_DIR), str(_PIPELINE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Imports de fase1 ───────────────────────────────────────────
from fase1_config import (
    BACKBONE_CONFIGS,
    BACKBONE_TO_CHECKPOINT_DIR,
    BACKBONE_CHECKPOINT_FILENAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_WORKERS,
    PANCREAS_FOLD,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_WEIGHT_DECAY,
    TRAIN_WARMUP_EPOCHS,
    TRAIN_BATCH_SIZE,
    TRAIN_WORKERS,
)
import backbone_cvt13   # noqa: F401 — registrar interceptor timm
import backbone_densenet  # noqa: F401 — registrar interceptor timm
from backbone_loader import load_trainable_backbone
from backbone_trainer import train_backbone, backbone_checkpoint_exists
from dataset_builder import build_datasets

# ── Imports de pipeline global ──────────────────────────────────
from config import EXPERT_IDS, N_EXPERTS_DOMAIN
from logging_utils import setup_logging

import torch
from torch.utils.data import DataLoader

log = logging.getLogger("fase1")


# ══════════════════════════════════════════════════════════════
#  Ruta canónica del checkpoint para un backbone dado
# ══════════════════════════════════════════════════════════════

def _get_checkpoint_path(backbone_name: str, checkpoint_base_dir: str) -> str:
    """
    Devuelve la ruta completa del archivo backbone.pth para un backbone dado.

    Ejemplo:
        backbone_name = "vit_tiny_patch16_224"
        → checkpoints/backbone_01_vit_tiny/backbone.pth
    """
    subdir = BACKBONE_TO_CHECKPOINT_DIR[backbone_name]
    return str(Path(checkpoint_base_dir) / subdir / BACKBONE_CHECKPOINT_FILENAME)


# ══════════════════════════════════════════════════════════════
#  Detección de dispositivo (idéntico a fase1_pipeline.py)
# ══════════════════════════════════════════════════════════════

def _detect_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("[Setup] GPU: %s (%.1f GB VRAM)", name, vram)
        return device, "gpu"
    else:
        n = os.cpu_count() or 4
        n_th = max(1, n // 2)
        torch.set_num_threads(n_th)
        try:
            torch.set_num_interop_threads(max(1, n_th // 2))
        except RuntimeError:
            pass
        log.info("[Setup] CPU: %d cores → torch threads: %d", n, n_th)
        return torch.device("cpu"), "cpu"


# ══════════════════════════════════════════════════════════════
#  Dry-run
# ══════════════════════════════════════════════════════════════

def _print_dry_run_summary(args):
    """
    Modo simulación: instanciar modelo + head, un forward pass con un batch sintético,
    imprimir configuración completa. NO entrena, NO modifica disco.
    """
    import torch.nn as nn

    log.info("=" * 60)
    log.info("[DRY-RUN] Paso 4.1 — Entrenamiento End-to-End de Backbones")
    log.info("=" * 60)

    device, device_type = _detect_device()

    # 1. Configuración
    checkpoint_path = _get_checkpoint_path(args.backbone, args.checkpoint_dir)
    bb_cfg = BACKBONE_CONFIGS[args.backbone]
    log.info("")
    log.info("┌─ Configuración de entrenamiento ────────────────────────┐")
    log.info("  Backbone         : %s", args.backbone)
    log.info("  d_model          : %d", bb_cfg["d_model"])
    log.info("  VRAM estimada    : %.1f GB", bb_cfg["vram_gb"])
    log.info("  Epochs           : %d", args.epochs)
    log.info("  LR               : %.2e", args.lr)
    log.info("  Weight decay     : %.4f", args.weight_decay)
    log.info("  Warmup epochs    : %d", args.warmup_epochs)
    log.info("  Batch size       : %d", args.batch_size)
    log.info("  Workers          : %d", args.workers)
    log.info("  Dispositivo      : %s (%s)", device, device_type)
    log.info("  Checkpoint → %s", checkpoint_path)
    ckpt_exists = backbone_checkpoint_exists(checkpoint_path)
    log.info("  Checkpoint existe: %s", "SÍ ✓" if ckpt_exists else "NO (se crearía)")
    if ckpt_exists and not args.force:
        log.info("  → Con --force=False, se SALTARÍA el entrenamiento.")
    log.info("└────────────────────────────────────────────────────────┘")

    # 2. Backbone + Head
    log.info("")
    log.info("┌─ Modelo ───────────────────────────────────────────────┐")
    backbone, d_model = load_trainable_backbone(args.backbone, device)
    head = nn.Linear(d_model, N_EXPERTS_DOMAIN).to(device)
    n_bb = sum(p.numel() for p in backbone.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    log.info("  Backbone params  : %s", f"{n_bb:,}")
    log.info("  Head params      : %s  [Linear(%d→%d)]",
             f"{n_head:,}", d_model, N_EXPERTS_DOMAIN)
    log.info("  Total params     : %s", f"{n_bb + n_head:,}")
    log.info("└────────────────────────────────────────────────────────┘")

    # 3. Forward pass sintético (UN batch)
    log.info("")
    log.info("┌─ Forward pass sintético (1 batch) ─────────────────────┐")
    dummy_imgs = torch.randn(min(args.batch_size, 4), 3, 224, 224, device=device)
    dummy_labels = torch.randint(0, N_EXPERTS_DOMAIN,
                                  (min(args.batch_size, 4),), device=device)
    backbone.eval()
    head.eval()
    with torch.no_grad():
        z = backbone(dummy_imgs)
        logits = head(z)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, dummy_labels)
    log.info("  Input  : %s", list(dummy_imgs.shape))
    log.info("  CLS z  : %s", list(z.shape))
    log.info("  Logits : %s  (N_EXPERTS_DOMAIN=%d)", list(logits.shape), N_EXPERTS_DOMAIN)
    log.info("  Loss   : %.4f (esperado ~%.4f para clase aleatoria)",
             loss.item(), torch.log(torch.tensor(float(N_EXPERTS_DOMAIN))).item())
    log.info("  Forward pass OK ✓")
    log.info("└────────────────────────────────────────────────────────┘")

    # 4. Siguiente paso
    log.info("")
    log.info("[DRY-RUN] Para entrenar en producción:")
    log.info("  python src/pipeline/fase1/fase1_train_pipeline.py "
             "--backbone %s", args.backbone)
    log.info("[DRY-RUN] Fin. Sin cambios en disco.")


# ══════════════════════════════════════════════════════════════
#  Guard Clauses (reusar de fase1_pipeline.py)
# ══════════════════════════════════════════════════════════════

def _check_fase0_artifacts(cfg):
    """Verifica artefactos de Fase 0 necesarios. Igual que en fase1_pipeline.py."""
    checks = []
    for key in ("nih_train_list", "nih_val_list", "nih_test_list"):
        path = cfg.get(key)
        if path and not Path(path).exists():
            checks.append(f"  ✗ {key} = {path}")
    for key in ("isic_train_csv", "isic_val_csv", "isic_test_csv"):
        path = cfg.get(key)
        if path and not Path(path).exists():
            checks.append(f"  ✗ {key} = {path}")
    oa_root = cfg.get("oa_root")
    if oa_root:
        for split in ("train", "val", "test"):
            sd = Path(oa_root) / split
            if not sd.exists():
                checks.append(f"  ✗ oa_root/{split} no existe")
    panc_csv = cfg.get("pancreas_splits_csv")
    if panc_csv and not Path(panc_csv).exists():
        checks.append(f"  ✗ pancreas_splits_csv = {panc_csv}")
    if checks:
        log.error("Artefactos de Fase 0 faltantes:\n%s", "\n".join(checks))
        sys.exit(1)
    log.info("[Guard] Artefactos de Fase 0 verificados ✓")


# ══════════════════════════════════════════════════════════════
#  Generar reporte Markdown de entrenamiento
# ══════════════════════════════════════════════════════════════

def _generate_train_report(report_data: dict, output_path: str) -> None:
    """
    Genera fase1_train_report.md con métricas del entrenamiento.
    """
    lines = [
        "# Fase 1 — Reporte de Entrenamiento de Backbone",
        "",
        f"**Fecha:** {report_data['timestamp']}",
        f"**Backbone:** `{report_data['backbone']}`",
        f"**Dispositivo:** {report_data['device']}",
        f"**Tiempo total:** {report_data['total_time_s']:.1f}s "
        f"({report_data['total_time_s']/60:.1f} min)",
        "",
    ]

    if report_data.get("skipped"):
        lines += [
            "## Estado: OMITIDO (idempotencia)",
            "",
            "El checkpoint ya existía y `--force` no fue proporcionado.",
            "",
        ]
    else:
        lines += [
            "## Resultados",
            "",
            f"- **Mejor val_acc:** {report_data['best_val_acc']:.4f}",
            f"- **Mejor epoch:** {report_data['best_epoch']}",
            f"- **Checkpoint:** `{report_data['checkpoint_path']}`",
            "",
            "## Historia de Entrenamiento",
            "",
            "| Epoch | train_loss | train_acc | val_loss | val_acc |",
            "|-------|------------|-----------|----------|---------|",
        ]
        train_h = report_data.get("train_history", [])
        val_h = report_data.get("val_history", [])
        for t, v in zip(train_h, val_h):
            lines.append(
                f"| {t['epoch']:5d} | {t['loss']:10.4f} | {t['acc']:9.4f} "
                f"| {v['loss']:8.4f} | {v['acc']:7.4f} |"
            )
        lines.append("")

    lines += ["---", "*Generado automáticamente por fase1_train_pipeline.py*"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    log.info("[Reporte] fase1_train_report.md escrito: %s", output_path)


# ══════════════════════════════════════════════════════════════
#  Pipeline Principal
# ══════════════════════════════════════════════════════════════

def main(args):
    t_start = time.time()
    setup_logging(args.checkpoint_dir, phase_name="fase1_train")

    log.info("=" * 60)
    log.info("PASO 4.1 — Entrenamiento End-to-End del Backbone")
    log.info("=" * 60)
    log.info("Backbone: %s | epochs=%d | lr=%.2e",
             args.backbone, args.epochs, args.lr)

    checkpoint_path = _get_checkpoint_path(args.backbone, args.checkpoint_dir)

    report_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": args.backbone,
        "device": "pendiente",
        "checkpoint_path": checkpoint_path,
        "skipped": False,
        "total_time_s": 0.0,
        "best_val_acc": 0.0,
        "best_epoch": 0,
        "train_history": [],
        "val_history": [],
    }

    # ── 0. Idempotencia ──
    if not args.force and backbone_checkpoint_exists(checkpoint_path):
        log.info(
            "[SKIP] Checkpoint ya existe para '%s' en '%s'. "
            "Usa --force para reentrenar.",
            args.backbone, checkpoint_path,
        )
        report_data["skipped"] = True
        report_data["device"] = "N/A (omitido)"
        report_data["total_time_s"] = time.time() - t_start
        report_path = Path(args.checkpoint_dir) / BACKBONE_TO_CHECKPOINT_DIR[args.backbone] / "fase1_train_report.md"
        _generate_train_report(report_data, str(report_path))
        return

    # ── 1. Guard clause ──
    cfg = {
        "chest_csv": args.chest_csv, "chest_imgs": args.chest_imgs,
        "nih_train_list": args.nih_train_list, "nih_val_list": args.nih_val_list,
        "nih_test_list": args.nih_test_list, "chest_view_filter": args.chest_view_filter,
        "chest_bbox_csv": args.chest_bbox_csv,
        "isic_train_csv": args.isic_train_csv, "isic_val_csv": args.isic_val_csv,
        "isic_test_csv": args.isic_test_csv, "isic_imgs": args.isic_imgs,
        "oa_root": args.oa_root,
        "luna_patches_dir": args.luna_patches_dir, "luna_csv": args.luna_csv,
        "pancreas_splits_csv": args.pancreas_splits_csv,
        "pancreas_nii_dir": args.pancreas_nii_dir,
        "pancreas_fold": args.pancreas_fold,
        "pancreas_roi_strategy": args.pancreas_roi_strategy,
        "output_dir": args.checkpoint_dir,
    }
    _check_fase0_artifacts(cfg)

    # ── 2. Setup de dispositivo ──
    device, device_type = _detect_device()
    report_data["device"] = str(device)

    # ── 3. Backbone entrenable ──
    backbone, d_model = load_trainable_backbone(args.backbone, device)

    # ── 4. Datasets (mode="embedding" — misma interfaz, sin augmentaciones) ──
    log.info("[Setup] Construyendo datasets...")
    train_ds, val_ds, _test_ds = build_datasets(cfg)
    log.info("[Dataset] train=%s  val=%s", f"{len(train_ds):,}", f"{len(val_ds):,}")

    # ── 5. DataLoaders ──
    n_workers = args.workers if device_type == "gpu" else min(args.workers, max(1, (os.cpu_count() or 4) // 2 - 1))
    _dl_kw = dict(
        batch_size=args.batch_size,
        shuffle=True,           # shuffle=True para entrenamiento
        num_workers=n_workers,
        pin_memory=(device_type == "gpu"),
        prefetch_factor=2 if n_workers > 0 else None,
        persistent_workers=(n_workers > 0),
        drop_last=True,         # evitar batches incompletos en última iteración
    )
    train_loader = DataLoader(train_ds, **_dl_kw)
    val_loader = DataLoader(val_ds, **{**_dl_kw, "shuffle": False, "drop_last": False})

    # ── 6. Entrenamiento ──
    train_cfg = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
    }
    metrics = train_backbone(
        backbone=backbone,
        d_model=d_model,
        backbone_name=args.backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_path=checkpoint_path,
        cfg=train_cfg,
    )

    # ── 7. Reporte ──
    report_data.update({
        "total_time_s": time.time() - t_start,
        "best_val_acc": metrics["best_val_acc"],
        "best_epoch": metrics["best_epoch"],
        "train_history": metrics["train_history"],
        "val_history": metrics["val_history"],
    })
    report_path = Path(args.checkpoint_dir) / BACKBONE_TO_CHECKPOINT_DIR[args.backbone] / "fase1_train_report.md"
    _generate_train_report(report_data, str(report_path))

    log.info("[Paso 4.1] Completado. Siguiente: fase1_pipeline.py --backbone %s "
             "--checkpoint_path %s", args.backbone, checkpoint_path)


# ══════════════════════════════════════════════════════════════
#  Argparse
# ══════════════════════════════════════════════════════════════

def _build_parser():
    _DS = _PROJECT_ROOT / "datasets"

    def _default_if_exists(rel_path):
        full = _DS / rel_path
        return str(full) if full.exists() else None

    parser = argparse.ArgumentParser(
        description="PASO 4.1 — Entrenamiento end-to-end de backbones MoE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Verificar sin entrenar:\n"
            "  python fase1_train_pipeline.py --backbone vit_tiny_patch16_224 --dry-run\n\n"
            "  # Entrenar:\n"
            "  python fase1_train_pipeline.py --backbone vit_tiny_patch16_224\n"
        ),
    )

    # ── Backbone ──
    parser.add_argument("--backbone", default="vit_tiny_patch16_224",
                        choices=list(BACKBONE_CONFIGS.keys()))
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=TRAIN_WORKERS)

    # ── Entrenamiento ──
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=TRAIN_LR)
    parser.add_argument("--weight_decay", type=float, default=TRAIN_WEIGHT_DECAY)
    parser.add_argument("--warmup_epochs", type=int, default=TRAIN_WARMUP_EPOCHS)

    # ── Salida ──
    parser.add_argument("--checkpoint_dir", default="./checkpoints",
                        help="Directorio base de checkpoints (default: ./checkpoints)")

    # ── Control ──
    parser.add_argument("--force", action="store_true", default=False,
                        help="Reentrenar aunque el checkpoint ya exista")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        dest="dry_run",
                        help="Verificar configuración + forward pass sin entrenar")

    # ── Datasets (IDÉNTICO a fase1_pipeline.py) ──
    parser.add_argument("--chest_csv",
                        default=_default_if_exists("nih_chest_xrays/Data_Entry_2017.csv"))
    parser.add_argument("--chest_imgs",
                        default=_default_if_exists("nih_chest_xrays/all_images"))
    parser.add_argument("--nih_train_list",
                        default=_default_if_exists("nih_chest_xrays/splits/nih_train_list.txt"))
    parser.add_argument("--nih_val_list",
                        default=_default_if_exists("nih_chest_xrays/splits/nih_val_list.txt"))
    parser.add_argument("--nih_test_list",
                        default=_default_if_exists("nih_chest_xrays/splits/nih_test_list.txt"))
    parser.add_argument("--chest_view_filter", default=None, choices=["PA", "AP"])
    parser.add_argument("--chest_bbox_csv",
                        default=_default_if_exists("nih_chest_xrays/BBox_List_2017.csv"))
    parser.add_argument("--isic_train_csv",
                        default=_default_if_exists("isic_2019/splits/isic_train.csv"))
    parser.add_argument("--isic_val_csv",
                        default=_default_if_exists("isic_2019/splits/isic_val.csv"))
    parser.add_argument("--isic_test_csv",
                        default=_default_if_exists("isic_2019/splits/isic_test.csv"))
    parser.add_argument("--isic_imgs",
                        default=_default_if_exists("isic_2019/ISIC_2019_Training_Input"))
    parser.add_argument("--oa_root",
                        default=_default_if_exists("osteoarthritis/oa_splits"))
    parser.add_argument("--luna_patches_dir",
                        default=_default_if_exists("luna_lung_cancer/patches"))
    parser.add_argument("--luna_csv",
                        default=_default_if_exists("luna_lung_cancer/candidates_V2/candidates_V2.csv"))
    parser.add_argument("--pancreas_splits_csv",
                        default=_default_if_exists("pancreas_splits.csv"))
    parser.add_argument("--pancreas_nii_dir",
                        default=_default_if_exists("zenodo_13715870"))
    parser.add_argument("--pancreas_fold", type=int, default=PANCREAS_FOLD)
    parser.add_argument("--pancreas_roi_strategy", default="A", choices=["A", "B"])

    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    if _args.dry_run:
        setup_logging(_args.checkpoint_dir, phase_name="fase1_train")
        _print_dry_run_summary(_args)
        sys.exit(0)

    main(_args)
```

---

## Task 6: Actualizar `fase1_pipeline.py` — soporte de checkpoint entrenado

**File:** `src/pipeline/fase1/fase1_pipeline.py`

### 6a: Añadir import en el bloque de imports
Después de `from backbone_loader import load_frozen_backbone`, añadir:
```python
from backbone_loader import load_frozen_backbone, load_frozen_backbone_from_checkpoint
from fase1_config import BACKBONE_TO_CHECKPOINT_DIR, BACKBONE_CHECKPOINT_FILENAME
```

### 6b: Añadir argumento CLI `--checkpoint_path`
En `_build_parser()`, después del argumento `--force`, añadir:
```python
parser.add_argument(
    "--checkpoint_path",
    default=None,
    help=(
        "Ruta al checkpoint backbone.pth generado por Paso 4.1 "
        "(fase1_train_pipeline.py). Si no se proporciona, se usa "
        "el backbone con pesos aleatorios (ADVERTENCIA: embeddings no significativos)."
    ),
)
```

### 6c: Actualizar el paso 4 de `main()` — carga del backbone
Reemplazar:
```python
    # ── 4. Cargar backbone congelado ──
    log.info("[Setup] Cargando backbone '%s'...", args.backbone)
    model, d_model = load_frozen_backbone(args.backbone, device)
```
Con:
```python
    # ── 4. Cargar backbone congelado ──
    # Paso 4.1 entrenó el backbone y guardó el checkpoint.
    # Si no se proporciona --checkpoint_path, intenta el path canónico.
    if args.checkpoint_path is None:
        # Intentar path canónico según BACKBONE_TO_CHECKPOINT_DIR
        _subdir = BACKBONE_TO_CHECKPOINT_DIR.get(args.backbone, "")
        _canonical = Path("checkpoints") / _subdir / BACKBONE_CHECKPOINT_FILENAME
        if _canonical.exists():
            args.checkpoint_path = str(_canonical)
            log.info("[Setup] Checkpoint encontrado automáticamente: %s", args.checkpoint_path)
        else:
            log.warning(
                "[Setup] Checkpoint no encontrado en '%s'. "
                "El backbone se inicializará con pesos ALEATORIOS. "
                "Los embeddings NO serán significativos para routing. "
                "Ejecuta primero: python src/pipeline/fase1/fase1_train_pipeline.py "
                "--backbone %s",
                _canonical, args.backbone,
            )

    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        log.info("[Setup] Cargando backbone desde checkpoint: %s", args.checkpoint_path)
        model, d_model = load_frozen_backbone_from_checkpoint(
            args.backbone, args.checkpoint_path, device
        )
    else:
        log.info("[Setup] Cargando backbone '%s' (pesos aleatorios)...", args.backbone)
        model, d_model = load_frozen_backbone(args.backbone, device)
```

---

## Verificación final (dry-run)

Una vez implementadas todas las tasks, verificar que el dry-run funciona:

```bash
cd /mnt/hdd/datasets/carlos_andres_ferro/proyecto_2

# Debe imprimir configuración + forward pass OK + salir sin errores
python src/pipeline/fase1/fase1_train_pipeline.py \
    --backbone vit_tiny_patch16_224 \
    --dry-run
```

Salida esperada (mínimo):
- `[DRY-RUN] Paso 4.1 — Entrenamiento End-to-End de Backbones`
- Tabla de configuración con epochs, lr, d_model, checkpoint_path
- `Forward pass OK ✓`
- `[DRY-RUN] Fin. Sin cambios en disco.`
- Exit code 0

```bash
# También verificar que fase1_pipeline.py (Paso 4.2) sigue funcionando en dry-run
python src/pipeline/fase1/fase1_pipeline.py \
    --backbone vit_tiny_patch16_224 \
    --dry-run
```

---

## Notas para el agente implementador

1. **Leer cada archivo ANTES de editarlo** — obligatorio
2. Para CvT-13: el interceptor `_patched_create_model` en `backbone_cvt13.py` congela params. Para la función `build_cvt13_trainable`, simplemente copiar la lógica de `build_cvt13()` pero sin `param.requires_grad = False` ni `wrapper.eval()`
3. Para DenseNet: `build_densenet()` en `backbone_densenet.py` NO congela por sí mismo — el congelamiento lo hace el interceptor `_patched_create_model`. Entonces `build_densenet()` directamente ya devuelve un modelo entrenable
4. El interceptor de timm para CvT-13 y DenseNet congela al pasar por `timm.create_model()`. `load_trainable_backbone()` debe llamar las funciones `build_*` directamente, bypasseando los interceptores
5. Verificar que el import de `Path` existe en `backbone_loader.py` (ya debería existir via `from pathlib import Path`)
6. En `fase1_pipeline.py`, la línea de import existente dice `from backbone_loader import load_frozen_backbone` — REEMPLAZAR esa línea con la nueva (no duplicar)

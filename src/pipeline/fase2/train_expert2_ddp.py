"""
Script de entrenamiento DDP para Expert 2 — ConvNeXt-Small sobre ISIC 2019.

Versión multi-GPU de train_expert2.py usando DistributedDataParallel (DDP).
Funciona transparentemente en modo single-GPU si solo hay 1 GPU disponible.

Pipeline trifásico con descongelamiento progresivo:

    Fase 1 (épocas  1-5):   Solo head, backbone congelado.
                             AdamW + CosineAnnealingLR.
    Fase 2 (épocas  6-20):  Fine-tuning diferencial (últimos 2 stages + head).
                             AdamW diferencial + CosineAnnealingWarmRestarts.
    Fase 3 (épocas 21-40):  Full fine-tuning + early stopping.
                             AdamW diferencial + CosineAnnealingWarmRestarts.

Cada fase crea su propio optimizador, scheduler y wrapper DDP desde cero.
El modelo se guarda solo cuando val_f1_macro mejora (checkpoint best.pt).

Loss: FocalLossMultiClass(gamma=2.0, weight=class_weights, label_smoothing=0.1)
Métricas: F1-macro (principal), accuracy, BMCA, AUC-ROC macro OVR.
Augmentaciones de batch: CutMix (p=0.3) y MixUp (p=0.2) mutuamente excluyentes.

Lanzamiento:
    # Multi-GPU (2× Titan Xp):
    torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert2_ddp.py

    # Single-GPU:
    torchrun --nproc_per_node=1 src/pipeline/fase2/train_expert2_ddp.py

    # Dry-run:
    torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert2_ddp.py --dry-run

Dependencias:
    - src/pipeline/fase2/models/expert2_convnext_small.py: Expert2ConvNeXtSmall
    - src/pipeline/fase2/dataloader_expert2.py: build_dataloaders_expert2 (datasets)
    - src/pipeline/fase2/expert2_config.py: hiperparámetros por fase
    - src/pipeline/fase2/losses.py: FocalLossMultiClass
    - src/pipeline/datasets/isic.py: ISICDataset, cutmix_data, mixup_data
    - src/pipeline/fase2/ddp_utils.py: utilidades DDP
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

# ── Configurar paths ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from fase2.models.expert2_convnext_small import Expert2ConvNeXtSmall
from fase2.losses import FocalLossMultiClass
from datasets.isic import ISICDataset, cutmix_data, mixup_data
from fase2.expert2_config import (
    # General
    EXPERT2_NUM_CLASSES,
    EXPERT2_BATCH_SIZE,
    EXPERT2_ACCUMULATION_STEPS,
    EXPERT2_LABEL_SMOOTHING,
    EXPERT2_CHECKPOINT_DIR,
    EXPERT2_CHECKPOINT_NAME,
    EXPERT2_TOTAL_EPOCHS,
    # Fase 1
    EXPERT2_PHASE1_EPOCHS,
    EXPERT2_PHASE1_LR,
    EXPERT2_PHASE1_WD,
    EXPERT2_PHASE1_ETA_MIN,
    # Fase 2
    EXPERT2_PHASE2_EPOCHS,
    EXPERT2_PHASE2_HEAD_LR,
    EXPERT2_PHASE2_BACKBONE_LR,
    EXPERT2_PHASE2_WD,
    EXPERT2_PHASE2_T0,
    EXPERT2_PHASE2_T_MULT,
    EXPERT2_PHASE2_ETA_MIN,
    # Fase 3
    EXPERT2_PHASE3_EPOCHS,
    EXPERT2_PHASE3_HEAD_LR,
    EXPERT2_PHASE3_BACKBONE_LR,
    EXPERT2_PHASE3_WD,
    EXPERT2_PHASE3_T0,
    EXPERT2_PHASE3_T_MULT,
    EXPERT2_PHASE3_ETA_MIN,
    EXPERT2_EARLY_STOPPING_PATIENCE,
)
from fase2.ddp_utils import (
    setup_ddp,
    cleanup_ddp,
    wrap_model_ddp,
    get_ddp_dataloader,
    is_main_process,
    save_checkpoint_ddp,
    load_checkpoint_ddp,
    get_rank,
    get_world_size,
    get_model_state_dict,
    get_unwrapped_model,
    is_ddp_initialized,
)

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("expert2_train_ddp")

# ── Rutas de salida ────────────────────────────────────────────────────
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "expert_01_convnext_small" / "best.pt"
_TRAINING_LOG_PATH = (
    _CHECKPOINT_DIR / "expert_01_convnext_small" / "expert2_ddp_training_log.json"
)

# ── Constantes de entrenamiento ────────────────────────────────────────
_SEED = 42
_MIN_DELTA = 0.001  # Mejora mínima para considerar progreso en early stopping
_GRAD_CLIP_NORM = 1.0

# ── Rutas de datos ISIC 2019 ──────────────────────────────────────────
_ISIC_IMG_DIR = _PROJECT_ROOT / "datasets" / "isic_2019" / "ISIC_2019_Training_Input"
_ISIC_CACHE_DIR = (
    _PROJECT_ROOT / "datasets" / "isic_2019" / "ISIC_2019_Training_Input_preprocessed"
)
_ISIC_TRAIN_CSV = _PROJECT_ROOT / "datasets" / "isic_2019" / "splits" / "isic_train.csv"
_ISIC_VAL_CSV = _PROJECT_ROOT / "datasets" / "isic_2019" / "splits" / "isic_val.csv"
_ISIC_TEST_CSV = _PROJECT_ROOT / "datasets" / "isic_2019" / "splits" / "isic_test.csv"


def set_seed(seed: int = _SEED) -> None:
    """Fija todas las semillas para reproducibilidad.

    En modo DDP, cada proceso recibe seed + rank para diversidad de datos
    (el DistributedSampler ya maneja la partición, pero esto asegura que
    las augmentations sean diferentes por GPU).
    """
    effective_seed = seed + get_rank()
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if is_main_process():
        log.info(
            f"[INFO] [Expert2] [Seed] Semillas fijadas a {effective_seed} "
            f"(base={seed} + rank={get_rank()})"
        )


def _log_vram(tag: str = "") -> None:
    """Imprime uso actual de VRAM si hay GPU disponible (solo rank 0)."""
    if torch.cuda.is_available() and is_main_process():
        dev = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(dev) / 1e9
        reserved = torch.cuda.memory_reserved(dev) / 1e9
        log.info(
            f"[INFO] [Expert2] [VRAM{' ' + tag if tag else ''}] "
            f"GPU {dev}: Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB"
        )


class EarlyStoppingF1:
    """Early stopping por val_f1_macro (maximizar) con patience configurable."""

    def __init__(self, patience: int, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float = -float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, val_f1_macro: float) -> bool:
        """Evalúa si el entrenamiento debe detenerse."""
        if val_f1_macro > self.best_score + self.min_delta:
            self.best_score = val_f1_macro
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False


@contextmanager
def ddp_no_sync(model: nn.Module, active: bool) -> Iterator[None]:
    """Context manager para model.no_sync() en pasos intermedios de accumulation.

    DDP sincroniza gradientes en cada .backward(). Con gradient accumulation,
    solo necesitamos sincronizar en el último paso del bloque de acumulación.
    model.no_sync() evita la comunicación allreduce en los pasos intermedios.

    Args:
        model: modelo (posiblemente envuelto en DDP).
        active: True para activar no_sync (pasos intermedios),
                False para paso final (sincronización normal).
    """
    if active and is_ddp_initialized() and hasattr(model, "no_sync"):
        with model.no_sync():
            yield
    else:
        yield


def _build_datasets(
    max_samples: int | None,
) -> dict[str, ISICDataset | Subset | torch.Tensor]:
    """Construye los datasets ISIC 2019 (sin DataLoader) para aplicar DDP samplers.

    Separa la creación de datasets de la creación de DataLoaders para que
    DDP pueda inyectar su DistributedSampler en el DataLoader de train.
    """
    img_dir = _ISIC_IMG_DIR

    # Resolver cache_dir
    effective_cache_dir: Path | None = None
    if _ISIC_CACHE_DIR.exists():
        effective_cache_dir = _ISIC_CACHE_DIR

    # Verificar que los archivos existen
    for label, path in [
        ("img_dir", img_dir),
        ("train_csv", _ISIC_TRAIN_CSV),
        ("val_csv", _ISIC_VAL_CSV),
        ("test_csv", _ISIC_TEST_CSV),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"[Expert2/DataLoader] {label} no encontrado: {path}"
            )

    # Cargar DataFrames
    train_df = pd.read_csv(_ISIC_TRAIN_CSV)
    val_df = pd.read_csv(_ISIC_VAL_CSV)
    test_df = pd.read_csv(_ISIC_TEST_CSV)

    # Limitar muestras para dry-run ANTES de crear datasets
    if max_samples is not None:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max_samples)
        test_df = test_df.head(max_samples)

    # Crear datasets (ISICDataset maneja transforms internamente)
    train_ds = ISICDataset(
        img_dir=img_dir,
        cache_dir=effective_cache_dir,
        split_df=train_df,
        mode="expert",
        split="train",
        apply_bcn_crop=True,
    )

    val_ds = ISICDataset(
        img_dir=img_dir,
        cache_dir=effective_cache_dir,
        split_df=val_df,
        mode="expert",
        split="val",
        apply_bcn_crop=True,
    )

    test_ds = ISICDataset(
        img_dir=img_dir,
        cache_dir=effective_cache_dir,
        split_df=test_df,
        mode="expert",
        split="test",
        apply_bcn_crop=True,
    )

    # Obtener class_weights
    if hasattr(train_ds, "class_weights") and train_ds.class_weights is not None:
        class_weights = train_ds.class_weights
    else:
        # Fallback: computar pesos por inverse-frequency
        labels = [train_ds[i][1] for i in range(len(train_ds))]
        n_classes = ISICDataset.N_TRAIN_CLS
        counts = np.bincount(labels, minlength=n_classes)
        class_weights = torch.tensor(
            len(train_ds) / (n_classes * np.maximum(counts, 1)),
            dtype=torch.float32,
        )

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "class_weights": class_weights,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int,
    use_fp16: bool,
    dry_run: bool = False,
    cutmix_prob: float = 0.3,
    mixup_prob: float = 0.2,
) -> float:
    """Ejecuta una época de entrenamiento con DDP + gradient accumulation + FP16.

    Incluye CutMix/MixUp batch-level augmentation, gradient clipping,
    y model.no_sync() en pasos intermedios de accumulation.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, (imgs, labels, _stems) in enumerate(loader):
        if dry_run and batch_idx >= 2:
            break

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        # ── Determinar si es paso intermedio (no_sync) o final (sync) ──
        is_accumulation_step = ((batch_idx + 1) % accumulation_steps) != 0

        # ── Selección de augmentación de batch (mutuamente excluyentes) ──
        r = np.random.random()
        use_cutmix = r < cutmix_prob
        use_mixup = (not use_cutmix) and (r < cutmix_prob + mixup_prob)

        with ddp_no_sync(model, active=is_accumulation_step):
            with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
                if use_cutmix:
                    imgs, y_a, y_b, lam = cutmix_data(imgs, labels)
                    logits = model(imgs)
                    loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(
                        logits, y_b
                    )
                elif use_mixup:
                    imgs, y_a, y_b, lam = mixup_data(imgs, labels)
                    logits = model(imgs)
                    loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(
                        logits, y_b
                    )
                else:
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

        # ── Optimizer step: optimizer.step() → scaler.update() ─────
        if not is_accumulation_step:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        n_batches += 1

        if dry_run and is_main_process():
            log.info(
                f"  [INFO] [Expert2] [Train batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"logits={list(logits.shape)} | "
                f"loss={loss.item() * accumulation_steps:.4f}"
            )

    # Flush de gradientes residuales
    if n_batches % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_fp16: bool,
    dry_run: bool = False,
) -> dict[str, float]:
    """Ejecuta validación y calcula métricas multiclase.

    Métricas:
        - val_loss: FocalLossMultiClass promedio.
        - val_acc: accuracy top-1.
        - val_f1_macro: F1-score macro (sklearn).
        - val_bmca: Balanced Multi-Class Accuracy.
        - val_auc: AUC-ROC macro one-vs-rest (8 clases).

    Usa np.nanmean() para promediar AUC por clase evitando NaN en clases
    sin positivos (bug conocido #2).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch_idx, (imgs, labels, _stems) in enumerate(loader):
        if dry_run and batch_idx >= 1:
            break

        imgs = imgs.to(device, non_blocking=True)
        labels_dev = labels.long().to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(imgs)
            loss = criterion(logits, labels_dev)

        total_loss += loss.item()
        n_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

        if dry_run and is_main_process():
            log.info(
                f"  [INFO] [Expert2] [Val batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"logits={list(logits.shape)} | "
                f"loss={loss.item():.4f}"
            )

    avg_loss = total_loss / max(n_batches, 1)

    # ── Métricas ───────────────────────────────────────────────────
    all_logits_t = torch.cat(all_logits, dim=0)  # [N, num_classes]
    all_probs = torch.softmax(all_logits_t, dim=1).numpy()
    all_preds = all_probs.argmax(axis=1)
    all_labels_np = torch.cat(all_labels, dim=0).numpy()

    # Accuracy top-1
    acc = float(np.mean(all_preds == all_labels_np))

    # F1 Macro
    f1_macro = float(
        f1_score(all_labels_np, all_preds, average="macro", zero_division=0)
    )

    # BMCA = Balanced Multi-Class Accuracy
    bmca = float(balanced_accuracy_score(all_labels_np, all_preds))

    # AUC-ROC macro OVR — usa np.nanmean para evitar NaN (bug conocido #2)
    try:
        auc = float(
            roc_auc_score(
                all_labels_np,
                all_probs[:, :EXPERT2_NUM_CLASSES],
                multi_class="ovr",
                average="macro",
                labels=list(range(EXPERT2_NUM_CLASSES)),
            )
        )
    except ValueError:
        auc = 0.0
        if is_main_process():
            log.warning(
                "[INFO] [Expert2] [Val] AUC-ROC no computable "
                "(clases insuficientes) → AUC=0.0"
            )

    return {
        "val_loss": avg_loss,
        "val_acc": acc,
        "val_f1_macro": f1_macro,
        "val_bmca": bmca,
        "val_auc": auc,
    }


def _run_phase(
    phase_num: int,
    phase_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    train_sampler: object | None,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    use_fp16: bool,
    num_epochs: int,
    global_epoch_offset: int,
    best_f1_macro: float,
    training_log: list[dict],
    early_stopping: EarlyStoppingF1 | None,
    dry_run: bool,
    cutmix_prob: float = 0.3,
    mixup_prob: float = 0.2,
) -> float:
    """Ejecuta una fase completa de entrenamiento con soporte DDP.

    Diferencias clave vs. versión sin DDP:
    - train_sampler.set_epoch(epoch) al inicio de cada época.
    - Solo rank=0 hace logging, checkpointing, y escritura de métricas.
    - Gradient accumulation con model.no_sync() en pasos intermedios.
    - Orden correcto: optimizer.step() → scaler.update() → scheduler.step()
      (bug conocido #1: scheduler nunca se llama antes de optimizer.step()).
    """
    max_epochs = 1 if dry_run else num_epochs
    raw_model = get_unwrapped_model(model)

    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info(f"  [INFO] [Expert2] FASE {phase_num}: {phase_name}")
        log.info(
            f"  Épocas: {max_epochs} "
            f"(global {global_epoch_offset + 1}-{global_epoch_offset + max_epochs})"
        )
        log.info(f"  Params entrenables: {raw_model.count_parameters():,}")
        for i, pg in enumerate(optimizer.param_groups):
            log.info(
                f"  Param group {i}: lr={pg['lr']:.2e}, "
                f"wd={pg.get('weight_decay', 0):.1e}"
            )
        world = get_world_size()
        log.info(
            f"  DDP: world_size={world}, "
            f"batch_per_gpu={train_loader.batch_size}, "
            f"batch_total={train_loader.batch_size * world}"
        )
        log.info(f"{'=' * 70}\n")

    for epoch_local in range(max_epochs):
        epoch_global = global_epoch_offset + epoch_local + 1
        epoch_start = time.time()

        # ── Actualizar epoch en DistributedSampler para shuffle correcto ──
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch_global)

        # ── Train ──────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            accumulation_steps=EXPERT2_ACCUMULATION_STEPS,
            use_fp16=use_fp16,
            dry_run=dry_run,
            cutmix_prob=cutmix_prob,
            mixup_prob=mixup_prob,
        )

        # ── Validation (solo rank=0 necesita métricas, pero todos ejecutan) ──
        val_results = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_fp16=use_fp16,
            dry_run=dry_run,
        )

        # ── Scheduler step DESPUÉS de optimizer.step() (bug conocido #1) ──
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Extraer métricas ───────────────────────────────────────
        epoch_time = time.time() - epoch_start
        val_loss = val_results["val_loss"]
        val_acc = val_results["val_acc"]
        val_f1_macro = val_results["val_f1_macro"]
        val_bmca = val_results["val_bmca"]
        val_auc = val_results["val_auc"]

        is_best = val_f1_macro > best_f1_macro + _MIN_DELTA

        # ── Logging (solo rank=0) ──────────────────────────────────
        if is_main_process():
            log.info(
                f"[INFO] [Expert2] [Epoch {epoch_global:3d}/{EXPERT2_TOTAL_EPOCHS} | F{phase_num}] "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | val_f1_macro={val_f1_macro:.4f} | "
                f"val_bmca={val_bmca:.4f} | val_auc={val_auc:.4f} | "
                f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
                f"{' ★ BEST' if is_best else ''}"
            )
            _log_vram(f"epoch-{epoch_global}")

        # ── Guardar log de métricas (solo rank=0) ──────────────────
        if is_main_process():
            epoch_log: dict = {
                "epoch": epoch_global,
                "phase": phase_num,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1_macro,
                "val_bmca": val_bmca,
                "val_auc": val_auc,
                "lr": current_lr,
                "epoch_time_s": round(epoch_time, 1),
                "is_best": is_best,
                "world_size": get_world_size(),
            }
            training_log.append(epoch_log)

        # ── Guardar mejor checkpoint (solo rank=0) ─────────────────
        if is_best:
            best_f1_macro = val_f1_macro
            checkpoint = {
                "epoch": epoch_global,
                "phase": phase_num,
                "model_state_dict": get_model_state_dict(model),
                "val_f1_macro": val_f1_macro,
                "val_bmca": val_bmca,
                "val_auc": val_auc,
                "val_loss": val_loss,
                "config": {
                    "phase1_lr": EXPERT2_PHASE1_LR,
                    "phase2_head_lr": EXPERT2_PHASE2_HEAD_LR,
                    "phase2_backbone_lr": EXPERT2_PHASE2_BACKBONE_LR,
                    "phase3_head_lr": EXPERT2_PHASE3_HEAD_LR,
                    "phase3_backbone_lr": EXPERT2_PHASE3_BACKBONE_LR,
                    "label_smoothing": EXPERT2_LABEL_SMOOTHING,
                    "batch_size": EXPERT2_BATCH_SIZE,
                    "accumulation_steps": EXPERT2_ACCUMULATION_STEPS,
                    "seed": _SEED,
                    "world_size": get_world_size(),
                },
            }
            if not dry_run:
                save_checkpoint_ddp(checkpoint, _CHECKPOINT_PATH)

        # ── Guardar training log (solo rank=0) ─────────────────────
        if is_main_process() and not dry_run:
            with open(_TRAINING_LOG_PATH, "w") as f:
                json.dump(training_log, f, indent=2)

        # ── Early stopping (solo rank=0 decide, todos los procesos paran) ──
        if early_stopping is not None and not dry_run:
            should_stop = False
            if is_main_process():
                should_stop = early_stopping.step(val_f1_macro)

            # Broadcast la decisión de early stopping a todos los procesos
            if is_ddp_initialized():
                stop_tensor = torch.tensor([1 if should_stop else 0], dtype=torch.int32)
                if torch.cuda.is_available():
                    stop_tensor = stop_tensor.to(device)
                torch.distributed.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                if is_main_process():
                    log.info(
                        f"\n[INFO] [Expert2] [EarlyStopping] Detenido en época {epoch_global}. "
                        f"val_f1_macro no mejoró en "
                        f"{early_stopping.patience} épocas. "
                        f"Mejor val_f1_macro: {best_f1_macro:.4f}"
                    )
                break

    # ── Resumen de fase ────────────────────────────────────────────
    if is_main_process():
        phase_logs = [e for e in training_log if e["phase"] == phase_num]
        if phase_logs:
            best_epoch_log = max(phase_logs, key=lambda x: x["val_f1_macro"])
            log.info(
                f"\n[INFO] [Expert2] [Fase {phase_num} resumen] "
                f"Mejor época: {best_epoch_log['epoch']} | "
                f"val_f1_macro={best_epoch_log['val_f1_macro']:.4f} | "
                f"val_bmca={best_epoch_log['val_bmca']:.4f} | "
                f"val_auc={best_epoch_log['val_auc']:.4f}"
            )

    return best_f1_macro


def train(
    dry_run: bool = False,
    batch_per_gpu: int | None = None,
) -> None:
    """Función principal de entrenamiento trifásico del Expert 2 con DDP.

    Orquesta las 3 fases secuencialmente, pasando el modelo entrenado
    de una fase a la siguiente con nuevo optimizador, scheduler y wrapper DDP.

    Args:
        dry_run: si True, ejecuta 2 batches de train y 1 de val por fase.
        batch_per_gpu: override del batch size por GPU. Si None, se calcula
            automáticamente como EXPERT2_BATCH_SIZE // world_size.
    """
    # ── Inicializar DDP ────────────────────────────────────────────
    setup_ddp()

    set_seed(_SEED)

    # ── Dispositivo ────────────────────────────────────────────────
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main_process():
        log.info(f"[INFO] [Expert2] Dispositivo: {device}")
        log.info(f"[INFO] [Expert2] World size: {get_world_size()}")
        if device.type == "cpu":
            log.warning(
                "[INFO] [Expert2] Entrenando en CPU — será lento. "
                "Se recomienda GPU con >= 12 GB VRAM."
            )

    # ── Configuración ──────────────────────────────────────────────
    world_size = get_world_size()

    if batch_per_gpu is None:
        effective_batch_per_gpu = EXPERT2_BATCH_SIZE // world_size
    else:
        effective_batch_per_gpu = batch_per_gpu

    use_fp16 = device.type == "cuda"

    if is_main_process():
        log.info(
            f"[INFO] [Expert2] DDP batch: {effective_batch_per_gpu}/gpu × {world_size} GPUs "
            f"× {EXPERT2_ACCUMULATION_STEPS} accum = "
            f"{effective_batch_per_gpu * world_size * EXPERT2_ACCUMULATION_STEPS} "
            f"efectivo"
        )
        if not use_fp16:
            log.info(
                "[INFO] [Expert2] FP16 desactivado (no hay GPU). Usando FP32 en CPU."
            )
        if dry_run:
            log.info(
                "[INFO] [Expert2] === MODO DRY-RUN === (2 batches train + 1 batch val)"
            )

    # ── Modelo ─────────────────────────────────────────────────────
    model = Expert2ConvNeXtSmall(
        num_classes=EXPERT2_NUM_CLASSES,
        pretrained=True,
    ).to(device)

    if is_main_process():
        n_params_total = model.count_all_parameters()
        log.info(
            f"[INFO] [Expert2] Modelo ConvNeXt-Small creado: "
            f"{n_params_total:,} parámetros totales"
        )
        _log_vram("post-model")

    # ── Datasets (sin DataLoader, para aplicar DDP sampler) ────────
    num_workers_base = 0 if dry_run else max(1, os.cpu_count() // (2 * world_size))

    datasets = _build_datasets(
        max_samples=64 if dry_run else None,
    )

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]
    class_weights = datasets["class_weights"]

    # ── DataLoaders con DDP ────────────────────────────────────────
    # Train: con DistributedSampler (reemplaza WeightedRandomSampler del original)
    # Nota: DDP usa DistributedSampler en lugar de WeightedRandomSampler.
    # El balanceo de clases se maneja mediante class_weights en la loss.
    train_loader, train_sampler = get_ddp_dataloader(
        dataset=train_ds,
        batch_size=effective_batch_per_gpu,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers_base,
    )

    # Val/Test: SIN DistributedSampler (todos ven todos los datos)
    val_loader = DataLoader(
        val_ds,
        batch_size=effective_batch_per_gpu,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers_base,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers_base > 0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=effective_batch_per_gpu,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers_base,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers_base > 0,
    )

    if is_main_process():
        log.info(
            f"[INFO] [Expert2] [DataLoader] Train: {len(train_ds):,} samples, "
            f"batch_per_gpu={effective_batch_per_gpu}"
        )
        log.info(
            f"[INFO] [Expert2] [DataLoader] Val: {len(val_ds):,} | "
            f"Test: {len(test_ds):,}"
        )

    # ── Loss ───────────────────────────────────────────────────────
    class_weights = class_weights.to(device)
    criterion = FocalLossMultiClass(
        gamma=2.0,
        weight=class_weights,
        label_smoothing=EXPERT2_LABEL_SMOOTHING,
    )
    if is_main_process():
        log.info(
            f"[INFO] [Expert2] Loss: FocalLossMultiClass("
            f"gamma=2.0, weight=class_weights[{class_weights.shape[0]}], "
            f"label_smoothing={EXPERT2_LABEL_SMOOTHING})"
        )

    # ── GradScaler para FP16 (compartido entre fases) ──────────────
    scaler = GradScaler(device=device.type, enabled=use_fp16)

    # ── Directorio de checkpoints ──────────────────────────────────
    if is_main_process():
        _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Estado global ──────────────────────────────────────────────
    best_f1_macro: float = -float("inf")
    training_log: list[dict] = []

    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info(
            "  [INFO] [Expert2] INICIO DE ENTRENAMIENTO — Expert 2 DDP "
            "(ConvNeXt-Small / ISIC 2019)"
        )
        log.info(
            f"  Total épocas: {EXPERT2_TOTAL_EPOCHS} | "
            f"Batch efectivo: {effective_batch_per_gpu}×{world_size}×"
            f"{EXPERT2_ACCUMULATION_STEPS}="
            f"{effective_batch_per_gpu * world_size * EXPERT2_ACCUMULATION_STEPS}"
        )
        log.info(
            f"  FP16: {use_fp16} | Accumulation: {EXPERT2_ACCUMULATION_STEPS} | "
            f"Grad clip: {_GRAD_CLIP_NORM} | GPUs: {world_size}"
        )
        log.info(
            f"  3 fases: head-only({EXPERT2_PHASE1_EPOCHS}) → "
            f"partial({EXPERT2_PHASE2_EPOCHS}) → full({EXPERT2_PHASE3_EPOCHS})"
        )
        log.info(f"{'=' * 70}\n")

    # ================================================================
    # FASE 1: Solo head, backbone congelado
    # ================================================================
    model.freeze_backbone()

    if is_main_process():
        log.info(
            f"[INFO] [Expert2] [Fase 1] freeze_backbone() → "
            f"{model.count_parameters():,} params entrenables (solo head)"
        )

    # En Fase 1 el backbone está congelado → find_unused_parameters=True
    model_ddp = wrap_model_ddp(model, device, find_unused_parameters=True)

    optimizer_p1 = torch.optim.AdamW(
        model.get_head_params(),
        lr=EXPERT2_PHASE1_LR,
        weight_decay=EXPERT2_PHASE1_WD,
    )
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1,
        T_max=EXPERT2_PHASE1_EPOCHS,
        eta_min=EXPERT2_PHASE1_ETA_MIN,
    )

    best_f1_macro = _run_phase(
        phase_num=1,
        phase_name="Head-only (backbone congelado)",
        model=model_ddp,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_p1,
        scheduler=scheduler_p1,
        scaler=scaler,
        device=device,
        use_fp16=use_fp16,
        num_epochs=EXPERT2_PHASE1_EPOCHS,
        global_epoch_offset=0,
        best_f1_macro=best_f1_macro,
        training_log=training_log,
        early_stopping=None,
        dry_run=dry_run,
    )

    # ================================================================
    # FASE 2: Fine-tuning diferencial (últimos 2 stages + head)
    # ================================================================
    # Desempaquetar DDP, cambiar freeze, re-envolver
    model = get_unwrapped_model(model_ddp)
    model.unfreeze_last_stages(n=2)

    if is_main_process():
        log.info(
            f"[INFO] [Expert2] [Fase 2] unfreeze_last_stages(n=2) → "
            f"{model.count_parameters():,} params entrenables"
        )

    # Parcialmente congelado → find_unused_parameters=True
    model_ddp = wrap_model_ddp(model, device, find_unused_parameters=True)

    optimizer_p2 = torch.optim.AdamW(
        [
            {"params": model.get_head_params(), "lr": EXPERT2_PHASE2_HEAD_LR},
            {"params": model.get_backbone_params(), "lr": EXPERT2_PHASE2_BACKBONE_LR},
        ],
        weight_decay=EXPERT2_PHASE2_WD,
    )
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_p2,
        T_0=EXPERT2_PHASE2_T0,
        T_mult=EXPERT2_PHASE2_T_MULT,
        eta_min=EXPERT2_PHASE2_ETA_MIN,
    )

    best_f1_macro = _run_phase(
        phase_num=2,
        phase_name="Fine-tuning diferencial (últimos 2 stages + head)",
        model=model_ddp,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_p2,
        scheduler=scheduler_p2,
        scaler=scaler,
        device=device,
        use_fp16=use_fp16,
        num_epochs=EXPERT2_PHASE2_EPOCHS,
        global_epoch_offset=EXPERT2_PHASE1_EPOCHS,
        best_f1_macro=best_f1_macro,
        training_log=training_log,
        early_stopping=None,
        dry_run=dry_run,
    )

    # ================================================================
    # FASE 3: Full fine-tuning + early stopping
    # ================================================================
    model = get_unwrapped_model(model_ddp)
    model.unfreeze_all()

    if is_main_process():
        log.info(
            f"[INFO] [Expert2] [Fase 3] unfreeze_all() → "
            f"{model.count_parameters():,} params entrenables (todo descongelado)"
        )

    # Todo descongelado → find_unused_parameters=False
    model_ddp = wrap_model_ddp(model, device, find_unused_parameters=False)

    optimizer_p3 = torch.optim.AdamW(
        [
            {"params": model.get_head_params(), "lr": EXPERT2_PHASE3_HEAD_LR},
            {"params": model.get_backbone_params(), "lr": EXPERT2_PHASE3_BACKBONE_LR},
        ],
        weight_decay=EXPERT2_PHASE3_WD,
    )
    scheduler_p3 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_p3,
        T_0=EXPERT2_PHASE3_T0,
        T_mult=EXPERT2_PHASE3_T_MULT,
        eta_min=EXPERT2_PHASE3_ETA_MIN,
    )

    early_stopping = EarlyStoppingF1(
        patience=EXPERT2_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    if is_main_process():
        log.info(
            f"[INFO] [Expert2] [Fase 3] EarlyStopping: monitor=val_f1_macro, "
            f"patience={EXPERT2_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
        )

    best_f1_macro = _run_phase(
        phase_num=3,
        phase_name="Full fine-tuning + early stopping",
        model=model_ddp,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_p3,
        scheduler=scheduler_p3,
        scaler=scaler,
        device=device,
        use_fp16=use_fp16,
        num_epochs=EXPERT2_PHASE3_EPOCHS,
        global_epoch_offset=EXPERT2_PHASE1_EPOCHS + EXPERT2_PHASE2_EPOCHS,
        best_f1_macro=best_f1_macro,
        training_log=training_log,
        early_stopping=early_stopping,
        dry_run=dry_run,
    )

    # ── Resumen final (solo rank=0) ────────────────────────────────
    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info(
            "  [INFO] [Expert2] ENTRENAMIENTO FINALIZADO — Expert 2 DDP "
            "(ConvNeXt-Small / ISIC 2019)"
        )
        log.info(f"  Mejor val_f1_macro: {best_f1_macro:.4f}")
        if training_log:
            best_epoch = max(training_log, key=lambda x: x["val_f1_macro"])
            log.info(
                f"  Mejor época: {best_epoch['epoch']} (fase {best_epoch['phase']}) | "
                f"F1-macro: {best_epoch['val_f1_macro']:.4f} | "
                f"BMCA: {best_epoch['val_bmca']:.4f} | "
                f"AUC: {best_epoch['val_auc']:.4f}"
            )
        if not dry_run:
            log.info(f"  Checkpoint: {_CHECKPOINT_PATH}")
            log.info(f"  Training log: {_TRAINING_LOG_PATH}")
        log.info(f"{'=' * 70}")

        if dry_run:
            log.info(
                "\n[INFO] [Expert2] [DRY-RUN] Pipeline verificado exitosamente. "
                "Ejecuta sin --dry-run para entrenar."
            )

    # ── Cleanup DDP ────────────────────────────────────────────────
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Entrenamiento Expert 2 DDP — ConvNeXt-Small / ISIC 2019 (3 fases). "
            "Usar con torchrun para multi-GPU."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Ejecuta 2 batches de train y 1 de val para verificar el pipeline "
            "sin entrenar"
        ),
    )
    parser.add_argument(
        "--batch-per-gpu",
        type=int,
        default=None,
        help=(
            "Override del batch size por GPU. Default: EXPERT2_BATCH_SIZE // world_size"
        ),
    )
    args = parser.parse_args()
    train(
        dry_run=args.dry_run,
        batch_per_gpu=args.batch_per_gpu,
    )

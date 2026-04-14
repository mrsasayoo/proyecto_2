"""
Script de entrenamiento DDP para Expert 1 — ConvNeXt-Tiny sobre NIH ChestXray14.

Versión multi-GPU de train_expert1.py usando DistributedDataParallel (DDP).
Funciona transparentemente en modo single-GPU si solo hay 1 GPU disponible.

Pipeline LP-FT (Linear Probing → Fine-Tuning):

    Fase 1 (LP, 5 épocas):  Backbone congelado, solo head + domain_conv entrenables.
                             AdamW(lr=1e-3) sin scheduler.
    Fase 2 (FT, 30 épocas): Todo descongelado.
                             AdamW(lr=1e-4) + CosineAnnealingLR + early stopping.

Evaluación final con TTA (Test-Time Augmentation):
    Promedia logits de test original + test con HorizontalFlip determinista.
    Reporta AUC-ROC por clase (14 patologías) y macro AUC.

Lanzamiento (usa torchrun para detectar GPUs automáticamente):
    # Multi-GPU (2× Titan Xp):
    torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert1_ddp.py

    # Single-GPU (fallback transparente):
    torchrun --nproc_per_node=1 src/pipeline/fase2/train_expert1_ddp.py

    # O con el script wrapper:
    bash run_expert.sh 1

    # Dry-run:
    torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert1_ddp.py --dry-run

Nota técnica sobre térmica y batch balanceado:
    Con la configuración original (1 GPU, batch_size=32, accumulation=4):
        - GPU 0: 100% carga → 84°C, batch efectivo=128
        - GPU 1: 0% carga → idle

    Con DDP en 2× Titan Xp (batch_per_gpu = 32 // 2 = 16):
        - GPU 0: ~50% carga → ~65-70°C estimado
        - GPU 1: ~50% carga → ~65-70°C estimado
        - Batch efectivo total: 16 × 2 GPUs × 4 acumulación = 128 (idéntico)

    En la práctica, con 12 GB VRAM por Titan Xp y ConvNeXt-Tiny en FP16
    (imágenes 224×224 RGB), el consumo de VRAM por batch de 16 es ~500 MiB.
    Se puede subir a batch_per_gpu=24-28 para aprovechar mejor los 12 GB
    de VRAM sin saturar temperatura (estimado ~72-76°C). Para esto,
    ajustar accumulation_steps a 2 para mantener batch efectivo ~112-128:
        batch_per_gpu=24, accum=2 → efectivo = 24 × 2 GPUs × 2 = 96
        batch_per_gpu=28, accum=2 → efectivo = 28 × 2 GPUs × 2 = 112

    Recomendación: empezar con los defaults (16 per GPU) y monitorizar
    nvidia-smi. Si la temperatura se estabiliza bajo 75°C, probar
    --batch-per-gpu 24 para reducir el tiempo de entrenamiento ~30%.

Dependencias:
    - src/pipeline/fase2/models/expert1_convnext.py: Expert1ConvNeXtTiny
    - src/pipeline/fase2/dataloader_expert1.py: build_expert1_dataloaders (datasets)
    - src/pipeline/fase2/expert1_config.py: hiperparámetros LP-FT
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
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# ── Configurar paths ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from config import CHEST_PATHOLOGIES
from fase2.models.expert1_convnext import Expert1ConvNeXtTiny
from fase2.dataloader_expert1 import (
    build_expert1_dataloaders,
    _build_train_transform,
    _build_val_transform,
    _build_flip_transform,
)
from fase2.expert1_config import (
    EXPERT1_LP_EPOCHS,
    EXPERT1_FT_EPOCHS,
    EXPERT1_LP_LR,
    EXPERT1_FT_LR,
    EXPERT1_WEIGHT_DECAY,
    EXPERT1_DROPOUT_FC,
    EXPERT1_BATCH_SIZE,
    EXPERT1_NUM_WORKERS,
    EXPERT1_ACCUMULATION_STEPS,
    EXPERT1_FP16,
    EXPERT1_NUM_CLASSES,
    EXPERT1_EARLY_STOPPING_PATIENCE,
    EXPERT1_CONFIG_SUMMARY,
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

# ── Imports del dataset (para construir datasets sin DataLoader) ───────
from datasets.chest import ChestXray14Dataset
from torch.utils.data import Subset

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("expert1_train_ddp")

# ── Rutas de salida ────────────────────────────────────────────────────
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "expert_00_convnext_tiny" / "best.pt"
_TRAINING_LOG_PATH = (
    _CHECKPOINT_DIR / "expert_00_convnext_tiny" / "expert1_ddp_training_log.json"
)

# ── Constantes de entrenamiento ────────────────────────────────────────
_SEED = 42
_MIN_DELTA = 0.001  # Mejora mínima para considerar progreso en early stopping
_TOTAL_EPOCHS = EXPERT1_LP_EPOCHS + EXPERT1_FT_EPOCHS


# ── Rutas por defecto del dataset ──────────────────────────────────────


def get_default_paths(project_root: Path | None = None) -> dict[str, Path]:
    """Devuelve las rutas por defecto del dataset NIH ChestXray14."""
    root = project_root or _PROJECT_ROOT
    base = root / "datasets" / "nih_chest_xrays"
    splits = base / "splits"
    return {
        "csv_path": base / "Data_Entry_2017.csv",
        "images_dir": base / "all_images",
        "train_split": splits / "nih_train_list.txt",
        "val_split": splits / "nih_val_list.txt",
        "test_split": splits / "nih_test_list.txt",
    }


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
            f"[Seed] Semillas fijadas a {effective_seed} "
            f"(base={seed} + rank={get_rank()})"
        )


def _log_vram(tag: str = "") -> None:
    """Imprime uso actual de VRAM si hay GPU disponible (solo rank 0)."""
    if torch.cuda.is_available() and is_main_process():
        dev = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(dev) / 1e9
        reserved = torch.cuda.memory_reserved(dev) / 1e9
        log.info(
            f"[VRAM{' ' + tag if tag else ''}] "
            f"GPU {dev}: Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB"
        )


class EarlyStoppingAUC:
    """Early stopping por val_macro_auc (maximizar) con patience configurable."""

    def __init__(self, patience: int, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float = -float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, val_macro_auc: float) -> bool:
        """Evalúa si el entrenamiento debe detenerse."""
        if val_macro_auc > self.best_score + self.min_delta:
            self.best_score = val_macro_auc
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
    model.no_sync() evita la comunicación allreduce en los pasos intermedios,
    reduciendo overhead de red ~(accumulation_steps - 1) / accumulation_steps.

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
) -> float:
    """Ejecuta una época de entrenamiento con DDP + gradient accumulation + FP16.

    Usa model.no_sync() en los pasos intermedios de accumulation para
    evitar comunicación allreduce redundante entre GPUs.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, (imgs, labels, _stems) in enumerate(loader):
        if dry_run and batch_idx >= 2:
            break

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Determinar si es paso intermedio (no_sync) o final (sync) ──
        is_accumulation_step = ((batch_idx + 1) % accumulation_steps) != 0

        with ddp_no_sync(model, active=is_accumulation_step):
            with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

        if not is_accumulation_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        n_batches += 1

        if dry_run and is_main_process():
            log.info(
                f"  [Train batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"logits={list(logits.shape)} | "
                f"loss={loss.item() * accumulation_steps:.4f}"
            )

    # Flush de gradientes residuales
    if n_batches % accumulation_steps != 0:
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
) -> dict[str, float | list[float]]:
    """Ejecuta validación y calcula métricas multilabel.

    En modo DDP, la validación se ejecuta en todos los procesos pero las
    métricas se computan solo en rank=0 (el DataLoader de val no usa
    DistributedSampler para evitar pérdida de samples por padding).
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
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(imgs)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

        if dry_run and is_main_process():
            log.info(
                f"  [Val batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"logits={list(logits.shape)} | "
                f"loss={loss.item():.4f}"
            )

    avg_loss = total_loss / max(n_batches, 1)

    all_logits_t = torch.cat(all_logits, dim=0)  # [N, 14]
    all_labels_np = torch.cat(all_labels, dim=0).numpy().astype(int)  # [N, 14]
    probs = torch.sigmoid(all_logits_t).numpy()  # [N, 14]

    # AUC-ROC por clase (skip clases con una sola label en este split)
    auc_per_class: list[float] = []
    for c in range(EXPERT1_NUM_CLASSES):
        try:
            auc_per_class.append(roc_auc_score(all_labels_np[:, c], probs[:, c]))
        except ValueError:
            auc_per_class.append(0.0)
    macro_auc = float(np.mean(auc_per_class))

    return {
        "val_loss": avg_loss,
        "val_macro_auc": macro_auc,
        "val_auc_per_class": auc_per_class,
    }


@torch.no_grad()
def eval_with_tta(
    model: nn.Module,
    dl_orig: DataLoader,
    dl_flip: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluación con TTA: promedia logits de original + HorizontalFlip."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for (x_orig, labels_orig, _s1), (x_flip, _labels_flip, _s2) in zip(
        dl_orig, dl_flip
    ):
        x_orig = x_orig.to(device, non_blocking=True)
        x_flip = x_flip.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits_orig = model(x_orig)
            logits_flip = model(x_flip)

        tta_logits = (logits_orig + logits_flip) / 2.0
        all_logits.append(tta_logits.cpu())
        all_labels.append(labels_orig.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def _build_datasets(
    paths: dict[str, Path],
    model_mean: tuple[float, ...],
    model_std: tuple[float, ...],
    use_cache: bool,
    max_samples: int | None,
) -> dict[str, ChestXray14Dataset | Subset | torch.Tensor]:
    """Construye los datasets (sin DataLoader) para poder aplicar DDP samplers.

    Separa la creación de datasets de la creación de DataLoaders para que
    DDP pueda inyectar su DistributedSampler en el DataLoader de train.
    """
    csv_path = str(paths["csv_path"])
    images_dir = str(paths["images_dir"])

    if max_samples is not None:
        use_cache = False

    train_tfm = _build_train_transform(model_mean, model_std)
    val_tfm = _build_val_transform(model_mean, model_std)
    flip_tfm = _build_flip_transform(model_mean, model_std)

    train_ds = ChestXray14Dataset(
        csv_path=csv_path,
        img_dir=images_dir,
        file_list=str(paths["train_split"]),
        transform=train_tfm,
        mode="expert",
        split="train",
        use_cache=use_cache,
    )

    val_ds = ChestXray14Dataset(
        csv_path=csv_path,
        img_dir=images_dir,
        file_list=str(paths["val_split"]),
        transform=val_tfm,
        mode="expert",
        split="val",
        use_cache=use_cache,
    )

    test_ds = ChestXray14Dataset(
        csv_path=csv_path,
        img_dir=images_dir,
        file_list=str(paths["test_split"]),
        transform=val_tfm,
        mode="expert",
        split="test",
        use_cache=use_cache,
    )

    test_flip_ds = ChestXray14Dataset(
        csv_path=csv_path,
        img_dir=images_dir,
        file_list=str(paths["test_split"]),
        transform=flip_tfm,
        mode="expert",
        split="test",
        use_cache=use_cache,
    )

    # Extraer pos_weight ANTES de hacer Subset
    pos_weight = train_ds.class_weights
    if pos_weight is None:
        raise RuntimeError(
            "[Expert1/DataLoader] train_ds.class_weights es None. "
            "Verifica que mode='expert' esté configurado correctamente."
        )

    # Limitar muestras si max_samples activo (dry-run)
    if max_samples is not None:
        n_train = min(max_samples, len(train_ds))
        n_val = min(max_samples, len(val_ds))
        n_test = min(max_samples, len(test_ds))
        n_test_flip = min(max_samples, len(test_flip_ds))
        train_ds = Subset(train_ds, list(range(n_train)))
        val_ds = Subset(val_ds, list(range(n_val)))
        test_ds = Subset(test_ds, list(range(n_test)))
        test_flip_ds = Subset(test_flip_ds, list(range(n_test_flip)))

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "test_flip_ds": test_flip_ds,
        "pos_weight": pos_weight,
    }


def _run_phase(
    phase_name: str,
    phase_tag: str,
    model: nn.Module,
    train_loader: DataLoader,
    train_sampler,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: GradScaler,
    device: torch.device,
    use_fp16: bool,
    num_epochs: int,
    global_epoch_offset: int,
    best_macro_auc: float,
    training_log: list[dict],
    early_stopping: EarlyStoppingAUC | None,
    dry_run: bool,
) -> float:
    """Ejecuta una fase completa de entrenamiento (LP o FT) con soporte DDP.

    Diferencias clave vs. versión sin DDP:
    - train_sampler.set_epoch(epoch) al inicio de cada época para shuffle correcto.
    - Solo rank=0 hace logging, checkpointing, y escritura de métricas.
    - Gradient accumulation con model.no_sync() en pasos intermedios.
    """
    max_epochs = 1 if dry_run else num_epochs
    raw_model = get_unwrapped_model(model)

    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info(f"  {phase_tag}: {phase_name}")
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
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_global)

        # ── Train ──────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            accumulation_steps=EXPERT1_ACCUMULATION_STEPS,
            use_fp16=use_fp16,
            dry_run=dry_run,
        )

        # ── Validation (solo en rank=0, loader sin DistributedSampler) ──
        val_results = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_fp16=use_fp16,
            dry_run=dry_run,
        )

        # ── Scheduler step (solo FT) ──────────────────────────────
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Extraer métricas ───────────────────────────────────────
        epoch_time = time.time() - epoch_start
        val_loss = val_results["val_loss"]
        val_macro_auc = val_results["val_macro_auc"]

        is_best = val_macro_auc > best_macro_auc + _MIN_DELTA

        # ── Logging (solo rank=0) ──────────────────────────────────
        if is_main_process():
            log.info(
                f"[Epoch {epoch_global:3d}/{_TOTAL_EPOCHS} | {phase_tag}] "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_macro_auc={val_macro_auc:.4f} | "
                f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
                f"{' ★ BEST' if is_best else ''}"
            )
            _log_vram(f"epoch-{epoch_global}")

        # ── Guardar log de métricas (solo rank=0) ──────────────────
        if is_main_process():
            epoch_log: dict = {
                "epoch": epoch_global,
                "phase": phase_tag,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_auc": val_macro_auc,
                "val_auc_per_class": val_results["val_auc_per_class"],
                "lr": current_lr,
                "epoch_time_s": round(epoch_time, 1),
                "is_best": is_best,
                "world_size": get_world_size(),
            }
            training_log.append(epoch_log)

        # ── Guardar mejor checkpoint (solo rank=0) ─────────────────
        if is_best:
            best_macro_auc = val_macro_auc
            checkpoint = {
                "epoch": epoch_global,
                "phase": phase_tag,
                "model_state_dict": get_model_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_auc": val_macro_auc,
                "val_loss": val_loss,
                "val_auc_per_class": val_results["val_auc_per_class"],
                "config": {
                    "lp_lr": EXPERT1_LP_LR,
                    "ft_lr": EXPERT1_FT_LR,
                    "weight_decay": EXPERT1_WEIGHT_DECAY,
                    "dropout_fc": EXPERT1_DROPOUT_FC,
                    "batch_size": EXPERT1_BATCH_SIZE,
                    "accumulation_steps": EXPERT1_ACCUMULATION_STEPS,
                    "fp16": EXPERT1_FP16,
                    "lp_epochs": EXPERT1_LP_EPOCHS,
                    "ft_epochs": EXPERT1_FT_EPOCHS,
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
                should_stop = early_stopping.step(val_macro_auc)

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
                        f"\n[EarlyStopping] Detenido en época {epoch_global}. "
                        f"val_macro_auc no mejoró en "
                        f"{early_stopping.patience} épocas. "
                        f"Mejor val_macro_auc: {best_macro_auc:.4f}"
                    )
                break

    # ── Resumen de fase ────────────────────────────────────────────
    if is_main_process():
        phase_logs = [e for e in training_log if e["phase"] == phase_tag]
        if phase_logs:
            best_epoch_log = max(phase_logs, key=lambda x: x["val_macro_auc"])
            log.info(
                f"\n[{phase_tag} resumen] Mejor época: {best_epoch_log['epoch']} | "
                f"val_macro_auc={best_epoch_log['val_macro_auc']:.4f}"
            )

    return best_macro_auc


def train(
    dry_run: bool = False,
    data_root: str | None = None,
    batch_per_gpu: int | None = None,
) -> None:
    """Función principal de entrenamiento LP-FT del Expert 1 con DDP.

    Args:
        dry_run: si True, ejecuta 2 batches de train y 1 de val.
        data_root: ruta raíz del proyecto. Si None, se auto-detecta.
        batch_per_gpu: override del batch size por GPU. Si None, se calcula
            automáticamente como EXPERT1_BATCH_SIZE // world_size.
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
        log.info(f"[Expert1] Dispositivo: {device}")
        log.info(f"[Expert1] World size: {get_world_size()}")
        if device.type == "cpu":
            log.warning(
                "[Expert1] Entrenando en CPU — será lento. "
                "Se recomienda GPU con >= 8 GB VRAM."
            )

    # ── Configuración ──────────────────────────────────────────────
    world_size = get_world_size()

    # Batch por GPU: dividir el batch total entre las GPUs para mantener
    # el mismo batch efectivo (batch_per_gpu * world_size * accum ≈ original)
    if batch_per_gpu is None:
        effective_batch_per_gpu = EXPERT1_BATCH_SIZE // world_size
    else:
        effective_batch_per_gpu = batch_per_gpu

    if is_main_process():
        log.info(f"[Expert1] Config: {EXPERT1_CONFIG_SUMMARY}")
        log.info(
            f"[Expert1] DDP batch: {effective_batch_per_gpu}/gpu × {world_size} GPUs "
            f"× {EXPERT1_ACCUMULATION_STEPS} accum = "
            f"{effective_batch_per_gpu * world_size * EXPERT1_ACCUMULATION_STEPS} "
            f"efectivo"
        )
        if dry_run:
            log.info("[Expert1] === MODO DRY-RUN === (2 batches train + 1 batch val)")

    use_fp16 = EXPERT1_FP16 and device.type == "cuda"
    if is_main_process() and not use_fp16 and EXPERT1_FP16:
        log.info("[Expert1] FP16 desactivado (no hay GPU). Usando FP32 en CPU.")

    # ── Modelo ─────────────────────────────────────────────────────
    model = Expert1ConvNeXtTiny(
        dropout_fc=EXPERT1_DROPOUT_FC,
        num_classes=EXPERT1_NUM_CLASSES,
        pretrained=True,
    ).to(device)

    if is_main_process():
        n_params_total = model.count_all_parameters()
        log.info(
            f"[Expert1] Modelo ConvNeXt-Tiny creado: "
            f"{n_params_total:,} parámetros totales"
        )
        _log_vram("post-model")

    # ── Datasets (sin DataLoader, para aplicar DDP sampler) ────────
    num_workers_base = 0 if dry_run else EXPERT1_NUM_WORKERS
    project_root = Path(data_root) if data_root else _PROJECT_ROOT
    paths = get_default_paths(project_root)

    datasets = _build_datasets(
        paths=paths,
        model_mean=model.model_mean,
        model_std=model.model_std,
        use_cache=True,
        max_samples=64 if dry_run else None,
    )

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]
    test_flip_ds = datasets["test_flip_ds"]
    pos_weight = datasets["pos_weight"]

    # ── DataLoaders con DDP ────────────────────────────────────────
    # Train: con DistributedSampler
    train_loader, train_sampler = get_ddp_dataloader(
        dataset=train_ds,
        batch_size=effective_batch_per_gpu,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers_base,
    )

    # Val/Test: SIN DistributedSampler (todos ven todos los datos)
    # Esto es intencional: en evaluación queremos métricas sobre el dataset completo.
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

    test_flip_loader = DataLoader(
        test_flip_ds,
        batch_size=effective_batch_per_gpu,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers_base,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers_base > 0,
    )

    if is_main_process():
        log.info(
            f"[Expert1/DataLoader] Train: {len(train_ds):,} samples, "
            f"batch_per_gpu={effective_batch_per_gpu}"
        )
        log.info(
            f"[Expert1/DataLoader] Val: {len(val_ds):,} | "
            f"Test: {len(test_ds):,} | Test flip: {len(test_flip_ds):,}"
        )

    # ── Loss ───────────────────────────────────────────────────────
    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if is_main_process():
        log.info(
            f"[Expert1] Loss: BCEWithLogitsLoss(pos_weight shape={pos_weight.shape})"
        )

    # ── GradScaler para FP16 (compartido entre fases) ──────────────
    scaler = GradScaler(device=device.type, enabled=use_fp16)

    # ── Directorio de checkpoints ──────────────────────────────────
    if is_main_process():
        _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Estado global ──────────────────────────────────────────────
    best_macro_auc: float = -float("inf")
    training_log: list[dict] = []

    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info(
            "  INICIO DE ENTRENAMIENTO — Expert 1 DDP (ConvNeXt-Tiny / ChestXray14)"
        )
        log.info(
            f"  LP: {EXPERT1_LP_EPOCHS} épocas (LR={EXPERT1_LP_LR}) | "
            f"FT: {EXPERT1_FT_EPOCHS} épocas (LR={EXPERT1_FT_LR})"
        )
        log.info(
            f"  Batch efectivo: "
            f"{effective_batch_per_gpu}×{world_size}×"
            f"{EXPERT1_ACCUMULATION_STEPS}="
            f"{effective_batch_per_gpu * world_size * EXPERT1_ACCUMULATION_STEPS}"
        )
        log.info(
            f"  FP16: {use_fp16} | Accumulation: {EXPERT1_ACCUMULATION_STEPS} | "
            f"GPUs: {world_size}"
        )
        log.info(f"{'=' * 70}\n")

    # ================================================================
    # FASE 1: Linear Probing (backbone congelado)
    # ================================================================
    model.freeze_backbone()

    if is_main_process():
        log.info(
            f"[LP] freeze_backbone() -> "
            f"{model.count_parameters():,} params entrenables (head + domain_conv)"
        )

    # En LP el backbone está congelado, así que DDP necesita
    # find_unused_parameters=True para los parámetros congelados.
    model_ddp = wrap_model_ddp(model, device, find_unused_parameters=True)

    optimizer_lp = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=EXPERT1_LP_LR,
        weight_decay=EXPERT1_WEIGHT_DECAY,
    )

    best_macro_auc = _run_phase(
        phase_name="Linear Probing (backbone congelado)",
        phase_tag="LP",
        model=model_ddp,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_lp,
        scheduler=None,  # Sin scheduler en LP
        scaler=scaler,
        device=device,
        use_fp16=use_fp16,
        num_epochs=EXPERT1_LP_EPOCHS,
        global_epoch_offset=0,
        best_macro_auc=best_macro_auc,
        training_log=training_log,
        early_stopping=None,  # Sin early stopping en LP
        dry_run=dry_run,
    )

    # ================================================================
    # FASE 2: Fine-Tuning (todo descongelado)
    # ================================================================
    # Desempaquetar DDP para cambiar el estado de freeze del modelo,
    # luego re-envolver. DDP no permite cambiar parámetros in-place.
    model = get_unwrapped_model(model_ddp)
    model.unfreeze_backbone()

    if is_main_process():
        log.info(
            f"[FT] unfreeze_backbone() -> "
            f"{model.count_parameters():,} params entrenables (todo descongelado)"
        )

    # Re-envolver con find_unused_parameters=False (todo descongelado)
    model_ddp = wrap_model_ddp(model, device, find_unused_parameters=False)

    optimizer_ft = torch.optim.AdamW(
        model.parameters(),
        lr=EXPERT1_FT_LR,
        weight_decay=EXPERT1_WEIGHT_DECAY,
    )
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft,
        T_max=EXPERT1_FT_EPOCHS,
    )

    early_stopping = EarlyStoppingAUC(
        patience=EXPERT1_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    if is_main_process():
        log.info(
            f"[FT] EarlyStopping: monitor=val_macro_auc, "
            f"patience={EXPERT1_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
        )

    best_macro_auc = _run_phase(
        phase_name="Fine-Tuning (todo descongelado + early stopping)",
        phase_tag="FT",
        model=model_ddp,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_ft,
        scheduler=scheduler_ft,
        scaler=scaler,
        device=device,
        use_fp16=use_fp16,
        num_epochs=EXPERT1_FT_EPOCHS,
        global_epoch_offset=EXPERT1_LP_EPOCHS,
        best_macro_auc=best_macro_auc,
        training_log=training_log,
        early_stopping=early_stopping,
        dry_run=dry_run,
    )

    # ================================================================
    # EVALUACIÓN FINAL CON TTA (solo rank=0)
    # ================================================================
    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info("  EVALUACIÓN FINAL — Test set con TTA (original + HorizontalFlip)")
        log.info(f"{'=' * 70}\n")

        # Desempaquetar DDP para evaluación
        model = get_unwrapped_model(model_ddp)

        # Cargar mejor checkpoint para evaluación
        if _CHECKPOINT_PATH.exists() and not dry_run:
            ckpt = load_checkpoint_ddp(_CHECKPOINT_PATH, map_location=device)
            if ckpt is not None:
                model.load_state_dict(ckpt["model_state_dict"])
                log.info(
                    f"[TTA] Cargado mejor checkpoint: época {ckpt['epoch']} "
                    f"(val_macro_auc={ckpt['val_macro_auc']:.4f})"
                )
        else:
            log.info(
                "[TTA] Usando modelo del final del entrenamiento (no hay checkpoint)"
            )

        tta_logits, tta_labels = eval_with_tta(
            model=model,
            dl_orig=test_loader,
            dl_flip=test_flip_loader,
            device=device,
            use_fp16=use_fp16,
        )

        # Calcular métricas TTA
        tta_probs = torch.sigmoid(tta_logits).numpy()
        tta_labels_np = tta_labels.numpy().astype(int)

        test_auc_per_class: list[float] = []
        for c in range(EXPERT1_NUM_CLASSES):
            try:
                auc_c = roc_auc_score(tta_labels_np[:, c], tta_probs[:, c])
            except ValueError:
                auc_c = 0.0
            test_auc_per_class.append(auc_c)

        test_macro_auc = float(np.mean(test_auc_per_class))

        # Reportar AUC por clase
        log.info("[TTA] AUC-ROC por clase:")
        for i, (name, auc_val) in enumerate(zip(CHEST_PATHOLOGIES, test_auc_per_class)):
            log.info(f"  [{i:2d}] {name:20s}: {auc_val:.4f}")
        log.info(f"  {'─' * 30}")
        log.info(f"  Macro AUC (TTA): {test_macro_auc:.4f}")

        # Agregar resultados TTA al training log
        tta_results = {
            "test_macro_auc_tta": test_macro_auc,
            "test_auc_per_class_tta": {
                name: auc_val
                for name, auc_val in zip(CHEST_PATHOLOGIES, test_auc_per_class)
            },
        }
        if training_log:
            training_log.append({"evaluation": "TTA", **tta_results})

        # Guardar training log final
        if not dry_run:
            with open(_TRAINING_LOG_PATH, "w") as f:
                json.dump(training_log, f, indent=2)

        # Resumen final
        log.info(f"\n{'=' * 70}")
        log.info(
            "  ENTRENAMIENTO FINALIZADO — Expert 1 DDP (ConvNeXt-Tiny / ChestXray14)"
        )
        log.info(f"  Mejor val_macro_auc: {best_macro_auc:.4f}")
        log.info(f"  Test macro AUC (TTA): {test_macro_auc:.4f}")
        if training_log:
            epoch_logs = [e for e in training_log if "epoch" in e]
            if epoch_logs:
                best_epoch = max(epoch_logs, key=lambda x: x["val_macro_auc"])
                log.info(
                    f"  Mejor época: {best_epoch['epoch']} "
                    f"({best_epoch['phase']}) | "
                    f"val_macro_auc: {best_epoch['val_macro_auc']:.4f}"
                )
        if not dry_run:
            log.info(f"  Checkpoint: {_CHECKPOINT_PATH}")
            log.info(f"  Training log: {_TRAINING_LOG_PATH}")
        log.info(f"{'=' * 70}")

        if dry_run:
            log.info(
                "\n[DRY-RUN] Pipeline verificado exitosamente. "
                "Ejecuta sin --dry-run para entrenar."
            )

    # ── Cleanup DDP ────────────────────────────────────────────────
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Entrenamiento Expert 1 DDP — ConvNeXt-Tiny / ChestXray14 (LP-FT). "
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
        "--data-root",
        type=str,
        default=None,
        help="Ruta raíz del proyecto (default: auto-detectada)",
    )
    parser.add_argument(
        "--batch-per-gpu",
        type=int,
        default=None,
        help=(
            "Override del batch size por GPU. Default: EXPERT1_BATCH_SIZE // world_size "
            "(32//2=16 con 2 GPUs). Subir a 24-28 para mejor aprovechamiento de "
            "VRAM si la temperatura se mantiene bajo 75°C."
        ),
    )
    args = parser.parse_args()
    train(
        dry_run=args.dry_run,
        data_root=args.data_root,
        batch_per_gpu=args.batch_per_gpu,
    )

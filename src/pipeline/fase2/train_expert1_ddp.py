"""
Script de entrenamiento DDP para Expert 1 — Hybrid-Deep-Vision sobre NIH ChestXray14.

Versión multi-GPU de train_expert1.py usando DistributedDataParallel (DDP).
Funciona transparentemente en modo single-GPU si solo hay 1 GPU disponible.

Entrenamiento directo desde cero (sin LP-FT):
    AdamW(lr=1e-3, wd=1e-4) + CosineAnnealingLR + early stopping.
    50 épocas máximo con patience=10.

Evaluación final con TTA (Test-Time Augmentation):
    Promedia probabilidades de test original + test con HorizontalFlip.
    Reporta AUC-ROC por clase (14 patologías) y macro AUC.

Lanzamiento (usa torchrun para detectar GPUs automáticamente):
    # Multi-GPU (2× Titan Xp):
    torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert1_ddp.py

    # Single-GPU (fallback transparente):
    torchrun --nproc_per_node=1 src/pipeline/fase2/train_expert1_ddp.py

    # Dry-run:
    torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert1_ddp.py --dry-run

Dependencias:
    - src/pipeline/fase2/models/expert1_convnext.py: HybridDeepVision
    - src/pipeline/fase2/dataloader_expert1.py: build_expert1_dataloaders (datasets)
    - src/pipeline/fase2/expert1_config.py: hiperparámetros de entrenamiento
    - src/pipeline/fase2/ddp_utils.py: utilidades DDP
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

# ── Variables NCCL (deben definirse ANTES de importar torch/NCCL) ──────
os.environ.setdefault(
    "NCCL_TIMEOUT", "1800000"
)  # 30 min (previene timeout en epoch 9+)
os.environ.setdefault("NCCL_IB_DISABLE", "1")  # deshabilita InfiniBand si no hay
os.environ.setdefault("NCCL_P2P_DISABLE", "1")  # fuerza PCIe (más estable con Titan Xp)
os.environ.setdefault(
    "NCCL_BLOCKING_WAIT", "1"
)  # timeout bloqueante (más limpio para debug)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# ── Configurar paths ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from config import CHEST_PATHOLOGIES
from fase2.models.expert1_convnext import HybridDeepVision
from fase2.dataloader_expert1 import (
    build_expert1_dataloaders,
    _build_train_transform,
    _build_val_transform,
    _build_flip_transform,
    _load_dataset_stats,
)
from fase2.expert1_config import (
    EXPERT1_EPOCHS,
    EXPERT1_LR,
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
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "expert_01_hybrid_deep_vision" / "best.pt"
_TRAINING_LOG_PATH = (
    _CHECKPOINT_DIR / "expert_01_hybrid_deep_vision" / "expert1_ddp_training_log.json"
)

# ── Constantes de entrenamiento ────────────────────────────────────────
_SEED = 42
_MIN_DELTA = 0.001  # Mejora mínima para considerar progreso en early stopping
_TOTAL_EPOCHS = EXPERT1_EPOCHS

# ── Ruta por defecto a stats.json ─────────────────────────────────────
_DEFAULT_STATS_PATH = (
    _PROJECT_ROOT / "datasets" / "nih_chest_xrays" / "preprocessed" / "stats.json"
)


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
    max_grad_norm: float = 1.0,
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
                probs = model(imgs)
                loss = criterion(probs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

        if not is_accumulation_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        n_batches += 1

        if dry_run and is_main_process():
            log.info(
                f"  [Train batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"probs={list(probs.shape)} | "
                f"loss={loss.item() * accumulation_steps:.4f}"
            )

    # Flush de gradientes residuales
    if n_batches % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
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

    El modelo produce probabilidades (post-sigmoid), por lo que se usan
    directamente para calcular AUC-ROC.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch_idx, (imgs, labels, _stems) in enumerate(loader):
        if dry_run and batch_idx >= 1:
            break

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            probs = model(imgs)
            loss = criterion(probs, labels)

        total_loss += loss.item()
        n_batches += 1

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

        if dry_run and is_main_process():
            log.info(
                f"  [Val batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"probs={list(probs.shape)} | "
                f"loss={loss.item():.4f}"
            )

    avg_loss = total_loss / max(n_batches, 1)

    all_probs_np = torch.cat(all_probs, dim=0).numpy()  # [N, 14]
    all_labels_np = torch.cat(all_labels, dim=0).numpy().astype(int)  # [N, 14]

    # AUC-ROC por clase (skip clases sin positivos o sin negativos en este split)
    auc_per_class: list[float] = []
    for c in range(EXPERT1_NUM_CLASSES):
        y_true_c = all_labels_np[:, c]
        if y_true_c.sum() == 0 or y_true_c.sum() == len(y_true_c):
            auc_per_class.append(float("nan"))
        else:
            try:
                auc_per_class.append(roc_auc_score(y_true_c, all_probs_np[:, c]))
            except ValueError:
                auc_per_class.append(float("nan"))

    n_valid = sum(1 for a in auc_per_class if not np.isnan(a))
    if n_valid < EXPERT1_NUM_CLASSES and is_main_process():
        log.warning(
            f"[Val] AUC calculado sobre {n_valid}/{EXPERT1_NUM_CLASSES} clases "
            f"({EXPERT1_NUM_CLASSES - n_valid} sin positivos o sin negativos)"
        )
    macro_auc = float(np.nanmean(auc_per_class)) if n_valid > 0 else 0.0

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
    """Evaluación con TTA: promedia probabilidades de original + HorizontalFlip."""
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for (x_orig, labels_orig, _s1), (x_flip, _labels_flip, _s2) in zip(
        dl_orig, dl_flip
    ):
        x_orig = x_orig.to(device, non_blocking=True)
        x_flip = x_flip.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            probs_orig = model(x_orig)
            probs_flip = model(x_flip)

        tta_probs = (probs_orig + probs_flip) / 2.0
        all_probs.append(tta_probs.cpu())
        all_labels.append(labels_orig.cpu())

    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


def _build_datasets(
    paths: dict[str, Path],
    dataset_mean: list[float],
    dataset_std: list[float],
    use_cache: bool,
    max_samples: int | None,
) -> dict[str, ChestXray14Dataset | Subset | torch.Tensor]:
    """Construye los datasets (sin DataLoader) para poder aplicar DDP samplers.

    Separa la creación de datasets de la creación de DataLoaders para que
    DDP pueda inyectar su DistributedSampler en el DataLoader de train.
    """
    csv_path = str(paths["csv_path"])
    images_dir = str(paths["images_dir"])
    preprocessed_dir = str(paths["images_dir"].parent / "preprocessed")

    if max_samples is not None:
        use_cache = False

    train_tfm = _build_train_transform(dataset_mean, dataset_std)
    val_tfm = _build_val_transform(dataset_mean, dataset_std)
    flip_tfm = _build_flip_transform(dataset_mean, dataset_std)

    train_ds = ChestXray14Dataset(
        csv_path=csv_path,
        img_dir=images_dir,
        file_list=str(paths["train_split"]),
        preprocessed_dir=preprocessed_dir,
        transform=train_tfm,
        mode="expert",
        split="train",
        use_cache=use_cache,
    )

    val_ds = ChestXray14Dataset(
        csv_path=csv_path,
        img_dir=images_dir,
        file_list=str(paths["val_split"]),
        preprocessed_dir=preprocessed_dir,
        transform=val_tfm,
        mode="expert",
        split="val",
        use_cache=use_cache,
    )

    test_ds = ChestXray14Dataset(
        csv_path=csv_path,
        img_dir=images_dir,
        file_list=str(paths["test_split"]),
        preprocessed_dir=preprocessed_dir,
        transform=val_tfm,
        mode="expert",
        split="test",
        use_cache=use_cache,
    )

    test_flip_ds = ChestXray14Dataset(
        csv_path=csv_path,
        img_dir=images_dir,
        file_list=str(paths["test_split"]),
        preprocessed_dir=preprocessed_dir,
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


def _run_training(
    model: nn.Module,
    train_loader: DataLoader,
    train_sampler: object,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    use_fp16: bool,
    num_epochs: int,
    start_epoch: int,
    best_macro_auc: float,
    training_log: list[dict],
    early_stopping: EarlyStoppingAUC,
    dry_run: bool,
) -> float:
    """Ejecuta el loop de entrenamiento completo con soporte DDP.

    Args:
        start_epoch: epoch number to start from (for resume support).
        best_macro_auc: best metric so far (for resume support).

    Returns:
        Best val_macro_auc achieved during training.
    """
    max_epochs = 1 if dry_run else num_epochs
    raw_model = get_unwrapped_model(model)

    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info("  ENTRENAMIENTO: Hybrid-Deep-Vision desde cero")
        log.info(
            f"  Épocas: {max_epochs} "
            f"(desde {start_epoch + 1} hasta {start_epoch + max_epochs})"
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
        epoch_global = start_epoch + epoch_local + 1
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
            accumulation_steps=EXPERT1_ACCUMULATION_STEPS,
            use_fp16=use_fp16,
            dry_run=dry_run,
        )

        # ── Validation ─────────────────────────────────────────────
        val_results = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_fp16=use_fp16,
            dry_run=dry_run,
        )

        # ── Scheduler step ─────────────────────────────────────────
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
                f"[Epoch {epoch_global:3d}/{_TOTAL_EPOCHS}] "
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
                "model_state_dict": get_model_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_auc": val_macro_auc,
                "val_loss": val_loss,
                "val_auc_per_class": val_results["val_auc_per_class"],
                "config": {
                    "lr": EXPERT1_LR,
                    "weight_decay": EXPERT1_WEIGHT_DECAY,
                    "dropout_fc": EXPERT1_DROPOUT_FC,
                    "batch_size": EXPERT1_BATCH_SIZE,
                    "accumulation_steps": EXPERT1_ACCUMULATION_STEPS,
                    "fp16": EXPERT1_FP16,
                    "epochs": EXPERT1_EPOCHS,
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

        # ── Early stopping ─────────────────────────────────────────
        if not dry_run:
            should_stop = False
            if is_main_process():
                should_stop = early_stopping.step(val_macro_auc)

            # Broadcast la decisión de early stopping a todos los procesos
            if is_ddp_initialized():
                stop_tensor = torch.tensor([1 if should_stop else 0], dtype=torch.int32)
                if torch.cuda.is_available():
                    stop_tensor = stop_tensor.to(device)
                    torch.cuda.synchronize()
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

    # ── Resumen ────────────────────────────────────────────────────
    if is_main_process() and training_log:
        best_epoch_log = max(training_log, key=lambda x: x.get("val_macro_auc", 0))
        if "epoch" in best_epoch_log:
            log.info(
                f"\n[Resumen] Mejor época: {best_epoch_log['epoch']} | "
                f"val_macro_auc={best_epoch_log['val_macro_auc']:.4f}"
            )

    return best_macro_auc


def train(
    dry_run: bool = False,
    data_root: str | None = None,
    batch_per_gpu: int | None = None,
    resume: str | None = None,
) -> None:
    """Función principal de entrenamiento del Expert 1 con DDP.

    Args:
        dry_run: si True, ejecuta 2 batches de train y 1 de val.
        data_root: ruta raíz del proyecto. Si None, se auto-detecta.
        batch_per_gpu: override del batch size por GPU. Si None, se calcula
            automáticamente como EXPERT1_BATCH_SIZE // world_size.
        resume: path al checkpoint para reanudar entrenamiento.
    """
    # ── Inicializar DDP ────────────────────────────────────────────
    # Auto-detectar backend: NCCL para GPU, Gloo para CPU
    setup_ddp(backend="auto")

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
    model = HybridDeepVision(
        dropout_fc=EXPERT1_DROPOUT_FC,
        num_classes=EXPERT1_NUM_CLASSES,
    ).to(device)

    if is_main_process():
        n_params_total = model.count_all_parameters()
        log.info(
            f"[Expert1] Modelo Hybrid-Deep-Vision creado: "
            f"{n_params_total:,} parámetros totales"
        )
        _log_vram("post-model")

    # ── Cargar estadísticas del dataset desde stats.json ───────────
    project_root = Path(data_root) if data_root else _PROJECT_ROOT
    stats_path = (
        project_root / "datasets" / "nih_chest_xrays" / "preprocessed" / "stats.json"
    )
    dataset_mean, dataset_std = _load_dataset_stats(stats_path)

    # ── Datasets (sin DataLoader, para aplicar DDP sampler) ────────
    num_workers_base = 0 if dry_run else EXPERT1_NUM_WORKERS
    paths = get_default_paths(project_root)

    datasets = _build_datasets(
        paths=paths,
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
        use_cache=True,
        max_samples=64 if dry_run else None,
    )

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]
    test_flip_ds = datasets["test_flip_ds"]
    pos_weight = datasets["pos_weight"]

    # ── DataLoaders con DDP ────────────────────────────────────────
    _dl_extra: dict = {}
    if num_workers_base > 0:
        _dl_extra["persistent_workers"] = True
        _dl_extra["prefetch_factor"] = 2
    train_loader, train_sampler = get_ddp_dataloader(
        dataset=train_ds,
        batch_size=effective_batch_per_gpu,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers_base,
        pin_memory=torch.cuda.is_available(),
        **_dl_extra,
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
        **({"prefetch_factor": 2} if num_workers_base > 0 else {}),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=effective_batch_per_gpu,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers_base,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers_base > 0,
        **({"prefetch_factor": 2} if num_workers_base > 0 else {}),
    )

    test_flip_loader = DataLoader(
        test_flip_ds,
        batch_size=effective_batch_per_gpu,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers_base,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers_base > 0,
        **({"prefetch_factor": 2} if num_workers_base > 0 else {}),
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
    # El modelo produce logits crudos (sin sigmoid). Usamos Focal Loss
    # que aplica sigmoid internamente y reduce el peso de ejemplos fáciles.
    # Cap pos_weight a 50 para evitar overflow/gradientes explosivos
    # (Hernia tenía ~538, peligroso incluso en FP32).
    pos_weight = pos_weight.clamp(max=50.0)
    pos_weight = pos_weight.to(device)
    criterion = FocalLoss(alpha=pos_weight, gamma=2.0, reduction="mean")

    if is_main_process():
        log.info(
            f"[Expert1] Loss: FocalLoss(alpha shape={pos_weight.shape}, gamma=2.0)"
        )

    # ── GradScaler para FP16 ───────────────────────────────────────
    scaler = GradScaler(device=device.type, enabled=use_fp16)

    # ── Directorio de checkpoints ──────────────────────────────────
    if is_main_process():
        _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Estado global ──────────────────────────────────────────────
    best_macro_auc: float = -float("inf")
    training_log: list[dict] = []
    resume_epoch: int = 0

    # ── Reanudación desde checkpoint ───────────────────────────────
    if resume is not None:
        resume_path = Path(resume)
        if resume_path.exists():
            ckpt = load_checkpoint_ddp(resume_path, map_location=device)
            if ckpt is not None:
                model.load_state_dict(ckpt["model_state_dict"])
                resume_epoch = ckpt.get("epoch", 0)
                best_macro_auc = ckpt.get("val_macro_auc", -float("inf"))
                if is_main_process():
                    log.info(
                        f"[RESUME] Reanudando desde época {resume_epoch}, "
                        f"best_metric={best_macro_auc:.4f}"
                    )
        else:
            if is_main_process():
                log.warning(
                    f"[RESUME] Checkpoint no encontrado: {resume_path}. "
                    "Iniciando desde cero."
                )

    # ── Envolver con DDP ───────────────────────────────────────────
    model_ddp = wrap_model_ddp(model, device, find_unused_parameters=False)

    # ── Optimizer ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=EXPERT1_LR,
        weight_decay=EXPERT1_WEIGHT_DECAY,
    )

    # ── Scheduler ──────────────────────────────────────────────────
    warnings.filterwarnings(
        "ignore",
        message=r"Detected call of.*lr_scheduler",
        category=UserWarning,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EXPERT1_EPOCHS,
        last_epoch=-1,
    )

    # ── Early stopping ─────────────────────────────────────────────
    early_stopping = EarlyStoppingAUC(
        patience=EXPERT1_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    if is_main_process():
        log.info(
            f"[Expert1] EarlyStopping: monitor=val_macro_auc, "
            f"patience={EXPERT1_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
        )

    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info(
            "  INICIO DE ENTRENAMIENTO — Expert 1 DDP "
            "(Hybrid-Deep-Vision / ChestXray14)"
        )
        log.info(
            f"  Épocas: {EXPERT1_EPOCHS} | LR={EXPERT1_LR} | WD={EXPERT1_WEIGHT_DECAY}"
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

    # ── Entrenamiento ──────────────────────────────────────────────
    remaining_epochs = EXPERT1_EPOCHS - resume_epoch
    best_macro_auc = _run_training(
        model=model_ddp,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        use_fp16=use_fp16,
        num_epochs=remaining_epochs,
        start_epoch=resume_epoch,
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

        tta_probs, tta_labels = eval_with_tta(
            model=model,
            dl_orig=test_loader,
            dl_flip=test_flip_loader,
            device=device,
            use_fp16=use_fp16,
        )

        # Calcular métricas TTA (probs ya son probabilidades post-sigmoid)
        tta_probs_np = tta_probs.numpy()
        tta_labels_np = tta_labels.numpy().astype(int)

        test_auc_per_class: list[float] = []
        for c in range(EXPERT1_NUM_CLASSES):
            y_true_c = tta_labels_np[:, c]
            if y_true_c.sum() == 0 or y_true_c.sum() == len(y_true_c):
                auc_c = float("nan")
            else:
                try:
                    auc_c = roc_auc_score(y_true_c, tta_probs_np[:, c])
                except ValueError:
                    auc_c = float("nan")
            test_auc_per_class.append(auc_c)

        test_macro_auc = float(np.nanmean(test_auc_per_class))

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
            "  ENTRENAMIENTO FINALIZADO — Expert 1 DDP "
            "(Hybrid-Deep-Vision / ChestXray14)"
        )
        log.info(f"  Mejor val_macro_auc: {best_macro_auc:.4f}")
        log.info(f"  Test macro AUC (TTA): {test_macro_auc:.4f}")
        if training_log:
            epoch_logs = [e for e in training_log if "epoch" in e]
            if epoch_logs:
                best_epoch = max(epoch_logs, key=lambda x: x["val_macro_auc"])
                log.info(
                    f"  Mejor época: {best_epoch['epoch']} | "
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


class FocalLoss(nn.Module):
    """Binary Focal Loss for multi-label classification over raw logits.

    Implements FL(pt) = -alpha * (1 - pt)^gamma * log(pt) where pt is the
    predicted probability of the correct class.  Built on top of
    ``F.binary_cross_entropy_with_logits`` for numerical stability (log-sum-exp
    trick), with the focal modulating factor applied multiplicatively.

    Args:
        alpha: Tensor [num_classes] with per-class positive weights (e.g.
            n_neg / n_pos).  Applied only to positive samples, matching the
            ``pos_weight`` semantics of ``BCEWithLogitsLoss``.
        gamma: Focusing parameter.  ``gamma=0`` recovers standard weighted BCE.
            Default: ``2.0``.
        reduction: ``"mean"`` (default) or ``"none"``.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: [B, C] raw scores (pre-sigmoid) from the model.
            targets: [B, C] binary labels (0 or 1).

        Returns:
            Scalar loss (if reduction="mean") or [B, C] tensor (if "none").
        """
        # Numerically stable BCE per element (no reduction yet).
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        # Probabilities for the focal modulating factor.
        probs = torch.sigmoid(logits)
        # pt = p when target=1, (1-p) when target=0
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)

        # Focal modulating factor: (1 - pt)^gamma
        focal_weight = (1.0 - pt).pow(self.gamma)

        loss = focal_weight * bce

        # Per-class alpha weighting (positive samples only, like pos_weight).
        if self.alpha is not None:
            # alpha_t: alpha for positives, 1.0 for negatives
            alpha_t = targets * self.alpha + (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Entrenamiento Expert 1 DDP — Hybrid-Deep-Vision / ChestXray14. "
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
            "Override del batch size por GPU. Default: EXPERT1_BATCH_SIZE // world_size."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path al checkpoint para reanudar entrenamiento. "
            "Ej: checkpoints/expert_01_hybrid_deep_vision/best.pt"
        ),
    )
    args = parser.parse_args()
    train(
        dry_run=args.dry_run,
        data_root=args.data_root,
        batch_per_gpu=args.batch_per_gpu,
        resume=args.resume,
    )

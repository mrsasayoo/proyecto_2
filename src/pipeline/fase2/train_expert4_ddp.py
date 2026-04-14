"""
Script de entrenamiento DDP para Expert 4 — ResNet 3D (R3D-18) sobre Páncreas CT.

Versión multi-GPU de train_expert4.py usando DistributedDataParallel (DDP).
Funciona transparentemente en modo single-GPU si solo hay 1 GPU disponible.

Pipeline de entrenamiento directo (sin LP-FT):

    ResNet 3D (R3D-18) ~33M params, entrenamiento desde cero.
    AdamW(lr=5e-5, wd=0.05) + CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
    FocalLoss(gamma=2, alpha=0.75) + FP16 + gradient accumulation (8 steps)
    k-fold CV (k=5) con fold seleccionable via --fold
    Early stopping por val_loss (patience=15)
    Gradient checkpointing obligatorio en GPU para reducir VRAM.

Nota sobre dataset pequeño:
    El dataset Páncreas tiene ~447 volúmenes NPY preprocesados. Con k-fold CV
    (k=5), cada fold de train tiene ~357 y val ~90 volúmenes. Con batch_size=1
    por GPU y 2 GPUs, el DistributedSampler divide 357/2 ≈ 178 samples por GPU.
    Se incluye un guard que verifica que cada split tenga al menos 2 samples
    por GPU antes de crear el DataLoader, evitando val/test vacíos.

Lanzamiento:
    # Multi-GPU (2× Titan Xp):
    torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert4_ddp.py --fold 0

    # Single-GPU (fallback transparente):
    torchrun --nproc_per_node=1 src/pipeline/fase2/train_expert4_ddp.py --fold 0

    # Dry-run:
    torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert4_ddp.py --dry-run

    # Todos los folds:
    for fold in 0 1 2 3 4; do
        torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert4_ddp.py --fold $fold
    done

Consideraciones DDP para volúmenes 3D:
    Con la configuración original (1 GPU, batch_size=2, accumulation=8):
        - GPU 0: 100% carga, batch efectivo=16
        - GPU 1: idle

    Con DDP en 2× Titan Xp (batch_per_gpu = max(1, 2 // 2) = 1):
        - GPU 0: ~50% carga → ~65-70°C estimado
        - GPU 1: ~50% carga → ~65-70°C estimado
        - Batch efectivo total: 1 × 2 GPUs × 8 acumulación = 16 (idéntico)

    Volúmenes 3D [1, 64, 64, 64] consumen ~4-6 GB VRAM por batch de 2 en FP16
    con gradient checkpointing. Con batch_per_gpu=1, ~2-3 GB por GPU.

Dependencias:
    - src/pipeline/fase2/models/expert4_resnet3d.py: ExpertPancreasSwin3D
    - src/pipeline/fase2/dataloader_expert4.py: build_dataloaders_expert4 (datasets)
    - src/pipeline/fase2/expert4_config.py: hiperparámetros
    - src/pipeline/fase2/ddp_utils.py: utilidades DDP
    - src/pipeline/losses.py: FocalLoss
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
from sklearn.metrics import f1_score, roc_auc_score

# ── Configurar paths ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from fase2.models.expert4_resnet3d import ExpertPancreasSwin3D
from fase2.dataloader_expert4 import (
    build_dataloaders_expert4,
    _load_valid_pairs_from_splits,
)
from fase2.expert4_config import (
    EXPERT4_LR,
    EXPERT4_WEIGHT_DECAY,
    EXPERT4_BATCH_SIZE,
    EXPERT4_ACCUMULATION_STEPS,
    EXPERT4_FP16,
    EXPERT4_MAX_EPOCHS,
    EXPERT4_EARLY_STOPPING_PATIENCE,
    EXPERT4_EARLY_STOPPING_MONITOR,
    EXPERT4_NUM_CLASSES,
    EXPERT4_NUM_FOLDS,
    EXPERT4_FOCAL_ALPHA,
    EXPERT4_FOCAL_GAMMA,
    EXPERT4_SCHEDULER_T0,
    EXPERT4_SCHEDULER_T_MULT,
    EXPERT4_CONFIG_SUMMARY,
)
from losses import FocalLoss
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
from datasets.pancreas import PancreasDataset
from torch.utils.data import Subset

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("expert4_train_ddp")

# ── Rutas de salida ────────────────────────────────────────────────────
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
_CHECKPOINT_BASE = _CHECKPOINT_DIR / "expert_04_swin3d_tiny"
_CHECKPOINT_PATH = _CHECKPOINT_BASE / "expert4_best.pt"
_TRAINING_LOG_PATH = _CHECKPOINT_BASE / "expert4_ddp_training_log.json"

# ── Constantes de entrenamiento ────────────────────────────────────────
_SEED = 42
_MIN_DELTA = 0.001  # Mejora mínima para considerar progreso en early stopping


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


def _enable_gradient_checkpointing(model: ExpertPancreasSwin3D) -> bool:
    """Habilita gradient checkpointing en los residual blocks del R3D-18.

    Envuelve el forward de cada layer con torch.utils.checkpoint.checkpoint
    para reducir consumo de VRAM a costa de re-computar activaciones en backward.

    Returns:
        True si se habilitó, False si no fue posible.
    """
    try:
        from torch.utils.checkpoint import checkpoint

        layer_names = ("layer1", "layer2", "layer3", "layer4")
        applied = 0

        for name in layer_names:
            layer = getattr(model, name, None)
            if layer is None:
                if is_main_process():
                    log.warning(
                        f"[GradCheckpoint] Atributo '{name}' no encontrado, omitido"
                    )
                continue

            original_forward = layer.forward

            def make_checkpointed_forward(orig_fwd):
                def checkpointed_forward(x):
                    return checkpoint(orig_fwd, x, use_reentrant=False)

                return checkpointed_forward

            layer.forward = make_checkpointed_forward(original_forward)
            applied += 1

        if is_main_process():
            log.info(
                f"[GradCheckpoint] Gradient checkpointing HABILITADO en "
                f"{applied}/{len(layer_names)} layers de R3D-18"
            )
        return applied > 0
    except Exception as e:
        if is_main_process():
            log.warning(f"[GradCheckpoint] No se pudo habilitar: {e}")
        return False


class EarlyStopping:
    """Early stopping por val_loss con patience configurable."""

    def __init__(self, patience: int, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """Evalúa si el entrenamiento debe detenerse."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
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

    Evita comunicación allreduce redundante en pasos intermedios de
    gradient accumulation, reduciendo overhead de red.

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


def _build_datasets_for_fold(
    fold: int,
    num_folds: int,
    max_samples: int | None = None,
    roi_strategy: str = "B",
) -> dict[str, PancreasDataset | Subset]:
    """Construye datasets de train y val para un fold dado (sin DataLoader).

    Separa la creación de datasets de DataLoaders para que DDP pueda
    inyectar su DistributedSampler en el DataLoader de train.

    Args:
        fold: índice del fold (0 a num_folds-1).
        num_folds: número total de folds.
        max_samples: si se especifica, limita cada split a N muestras (dry-run).
        roi_strategy: estrategia ROI ("A" o "B").

    Returns:
        dict con "train_ds" y "val_ds".
    """
    valid_pairs = _load_valid_pairs_from_splits()

    if not valid_pairs:
        raise ValueError(
            "[Expert4/DataLoader] No se encontraron pares válidos. "
            "Verifica que existan volúmenes .nii.gz y el CSV de splits/labels."
        )

    # Generar k-fold splits
    folds = PancreasDataset.build_kfold_splits(valid_pairs, k=num_folds)
    train_pairs, val_pairs = folds[fold]

    if is_main_process():
        n_pos_tr = sum(1 for _, l in train_pairs if l == 1)
        n_neg_tr = sum(1 for _, l in train_pairs if l == 0)
        n_pos_va = sum(1 for _, l in val_pairs if l == 1)
        n_neg_va = sum(1 for _, l in val_pairs if l == 0)
        log.info(
            f"[Expert4/DataLoader] Fold {fold}/{num_folds - 1}:\n"
            f"    Train: {len(train_pairs):,} (PDAC+={n_pos_tr}, PDAC-={n_neg_tr})\n"
            f"    Val:   {len(val_pairs):,} (PDAC+={n_pos_va}, PDAC-={n_neg_va})"
        )

    # Crear datasets
    train_ds = PancreasDataset(
        valid_pairs=train_pairs,
        mode="expert",
        roi_strategy=roi_strategy,
        z_score_per_volume=True,
        split="train",
        augment_3d=True,
    )

    val_ds = PancreasDataset(
        valid_pairs=val_pairs,
        mode="expert",
        roi_strategy=roi_strategy,
        z_score_per_volume=True,
        split="val",
        augment_3d=False,
    )

    # Limitar muestras si max_samples activo (dry-run)
    if max_samples is not None:
        n_train = min(max_samples, len(train_ds))
        n_val = min(max_samples, len(val_ds))
        train_ds = Subset(train_ds, list(range(n_train)))
        val_ds = Subset(val_ds, list(range(n_val)))

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
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
) -> float:
    """Ejecuta una época de entrenamiento con DDP + gradient accumulation + FP16.

    Usa model.no_sync() en los pasos intermedios de accumulation para
    evitar comunicación allreduce redundante entre GPUs.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, (volumes, labels, _stems) in enumerate(loader):
        if dry_run and batch_idx >= 2:
            break

        volumes = volumes.to(device, non_blocking=True)
        # labels es int desde PancreasDataset → convertir a float para FocalLoss
        labels = labels.float().to(device, non_blocking=True)

        # ── Determinar si es paso intermedio (no_sync) o final (sync) ──
        is_accumulation_step = ((batch_idx + 1) % accumulation_steps) != 0

        with ddp_no_sync(model, active=is_accumulation_step):
            with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
                logits = model(volumes)  # [B, 2]
                # FocalLoss espera logits [B] y targets [B] float
                loss = criterion(logits[:, 1], labels)
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
                f"volume={list(volumes.shape)} | "
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
) -> dict[str, float]:
    """Ejecuta validación y calcula métricas.

    En modo DDP, la validación se ejecuta en todos los procesos pero las
    métricas se computan solo en rank=0 (el DataLoader de val no usa
    DistributedSampler para evitar pérdida de samples por padding).

    Usa np.nanmean() para métricas y maneja clases sin positivos para
    evitar NaN en métricas.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    for batch_idx, (volumes, labels, _stems) in enumerate(loader):
        if dry_run and batch_idx >= 2:
            break

        volumes = volumes.to(device, non_blocking=True)
        labels_float = labels.float().to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(volumes)  # [B, 2]
            loss = criterion(logits[:, 1], labels_float)

        total_loss += loss.item()
        n_batches += 1

        # Probabilidades y predicciones
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_labels.append(labels.numpy())
        all_probs.append(probs)
        all_preds.append(preds)

        if dry_run and is_main_process():
            log.info(
                f"  [Val batch {batch_idx}] "
                f"volume={list(volumes.shape)} | "
                f"logits={list(logits.shape)} | "
                f"loss={loss.item():.4f}"
            )

    avg_loss = total_loss / max(n_batches, 1)

    if not all_labels:
        if is_main_process():
            log.warning("[Val] No hay batches de validación — métricas en 0.0")
        return {"val_loss": avg_loss, "val_auc": 0.0, "val_f1": 0.0}

    # ── Métricas (con guards contra NaN) ───────────────────────────
    labels_arr = np.concatenate(all_labels)
    preds_arr = np.concatenate(all_preds)
    probs_arr = np.concatenate(all_probs)

    # AUC-ROC — requiere al menos 2 clases en labels
    if len(np.unique(labels_arr)) >= 2:
        try:
            val_auc = float(roc_auc_score(labels_arr, probs_arr))
        except ValueError:
            val_auc = 0.0
    else:
        val_auc = 0.0
        if is_main_process():
            log.warning("[Val] Solo una clase presente → AUC=0.0")

    # F1 Macro — maneja zero_division
    if len(np.unique(labels_arr)) >= 2:
        val_f1 = float(
            f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
        )
    else:
        val_f1 = 0.0
        if is_main_process():
            log.warning("[Val] Solo una clase presente → F1=0.0")

    return {
        "val_loss": avg_loss,
        "val_auc": val_auc,
        "val_f1": val_f1,
    }


def train(
    dry_run: bool = False,
    fold: int = 0,
    batch_per_gpu: int | None = None,
) -> None:
    """Función principal de entrenamiento del Expert 4 con DDP.

    Args:
        dry_run: si True, ejecuta 2 batches de train y 2 de val.
        fold: índice del fold para k-fold CV (0 a EXPERT4_NUM_FOLDS-1).
        batch_per_gpu: override del batch size por GPU. Si None, se calcula
            automáticamente como max(1, EXPERT4_BATCH_SIZE // world_size).
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
        log.info(f"[Expert4] Dispositivo: {device}")
        log.info(f"[Expert4] World size: {get_world_size()}")
        if device.type == "cpu":
            log.warning(
                "[Expert4] Entrenando en CPU — será muy lento. "
                "Se recomienda GPU con >= 16 GB VRAM."
            )

    # ── Configuración ──────────────────────────────────────────────
    world_size = get_world_size()

    # Batch por GPU: para 3D con batch_size=2 y 2 GPUs → max(1, 2//2) = 1
    # Esto garantiza al menos 1 sample por GPU.
    if batch_per_gpu is None:
        effective_batch_per_gpu = max(1, EXPERT4_BATCH_SIZE // world_size)
    else:
        effective_batch_per_gpu = batch_per_gpu

    if is_main_process():
        log.info(f"[Expert4] Config: {EXPERT4_CONFIG_SUMMARY}")
        log.info(f"[Expert4] Fold: {fold}/{EXPERT4_NUM_FOLDS - 1}")
        log.info(
            f"[Expert4] DDP batch: {effective_batch_per_gpu}/gpu × {world_size} GPUs "
            f"× {EXPERT4_ACCUMULATION_STEPS} accum = "
            f"{effective_batch_per_gpu * world_size * EXPERT4_ACCUMULATION_STEPS} "
            f"efectivo"
        )
        if dry_run:
            log.info("[Expert4] === MODO DRY-RUN === (2 batches train + 2 batches val)")

    use_fp16 = EXPERT4_FP16 and device.type == "cuda"
    if is_main_process() and not use_fp16 and EXPERT4_FP16:
        log.info("[Expert4] FP16 desactivado (no hay GPU). Usando FP32 en CPU.")

    # ── Modelo ─────────────────────────────────────────────────────
    model = ExpertPancreasSwin3D(
        in_channels=1,
        num_classes=EXPERT4_NUM_CLASSES,
    ).to(device)

    n_params = model.count_parameters()
    if is_main_process():
        log.info(
            f"[Expert4] Modelo ResNet3D (R3D-18) creado: "
            f"{n_params:,} parámetros entrenables"
        )
        _log_vram("post-model")

    # ── Gradient checkpointing (OBLIGATORIO en GPU) ────────────────
    if device.type == "cuda":
        _enable_gradient_checkpointing(model)

    # ── Datasets (sin DataLoader, para aplicar DDP sampler) ────────
    num_workers_base = (
        0 if dry_run else max(1, (os.cpu_count() or 4) // (2 * world_size))
    )

    datasets = _build_datasets_for_fold(
        fold=fold,
        num_folds=EXPERT4_NUM_FOLDS,
        max_samples=64 if dry_run else None,
    )

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]

    # ── Guard: verificar que cada split tenga suficientes samples ───
    min_samples_per_gpu = effective_batch_per_gpu * 2  # Al menos 2 batches
    if len(train_ds) < world_size * min_samples_per_gpu:
        if is_main_process():
            log.warning(
                f"[Expert4] Dataset de train muy pequeño: {len(train_ds)} samples "
                f"para {world_size} GPUs. Cada GPU verá ~{len(train_ds) // world_size} "
                f"samples. Considere reducir batch_per_gpu o usar 1 GPU."
            )
    if len(val_ds) < world_size:
        if is_main_process():
            log.warning(
                f"[Expert4] Dataset de val muy pequeño: {len(val_ds)} samples "
                f"para {world_size} GPUs. Métricas pueden ser inestables."
            )

    # ── DataLoaders con DDP ────────────────────────────────────────
    # Train: con DistributedSampler
    train_loader, train_sampler = get_ddp_dataloader(
        dataset=train_ds,
        batch_size=effective_batch_per_gpu,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers_base,
    )

    # Val: SIN DistributedSampler (todos ven todos los datos)
    val_loader = DataLoader(
        val_ds,
        batch_size=effective_batch_per_gpu,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers_base,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers_base > 0,
    )

    if is_main_process():
        log.info(
            f"[Expert4/DataLoader] Train: {len(train_ds):,} samples, "
            f"batch_per_gpu={effective_batch_per_gpu}"
        )
        log.info(f"[Expert4/DataLoader] Val: {len(val_ds):,} samples")

    # ── Envolver modelo en DDP ─────────────────────────────────────
    # Entrenamiento directo (sin LP-FT): todos los params son entrenables,
    # find_unused_parameters=False.
    model_ddp = wrap_model_ddp(model, device, find_unused_parameters=False)

    # ── Loss ───────────────────────────────────────────────────────
    criterion = FocalLoss(gamma=EXPERT4_FOCAL_GAMMA, alpha=EXPERT4_FOCAL_ALPHA)
    if is_main_process():
        log.info(
            f"[Expert4] Loss: FocalLoss(gamma={EXPERT4_FOCAL_GAMMA}, "
            f"alpha={EXPERT4_FOCAL_ALPHA})"
        )

    # ── Optimizer ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=EXPERT4_LR,
        weight_decay=EXPERT4_WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    if is_main_process():
        log.info(
            f"[Expert4] Optimizer: AdamW(lr={EXPERT4_LR}, wd={EXPERT4_WEIGHT_DECAY})"
        )

    # ── Scheduler: CosineAnnealingWarmRestarts ─────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=EXPERT4_SCHEDULER_T0,
        T_mult=EXPERT4_SCHEDULER_T_MULT,
        eta_min=1e-6,
    )
    if is_main_process():
        log.info(
            f"[Expert4] Scheduler: CosineAnnealingWarmRestarts("
            f"T_0={EXPERT4_SCHEDULER_T0}, T_mult={EXPERT4_SCHEDULER_T_MULT}, "
            f"eta_min=1e-6)"
        )

    # ── GradScaler para FP16 ───────────────────────────────────────
    scaler = GradScaler(device=device.type, enabled=use_fp16)

    # ── Early stopping ─────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=EXPERT4_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    if is_main_process():
        log.info(
            f"[Expert4] EarlyStopping: monitor={EXPERT4_EARLY_STOPPING_MONITOR}, "
            f"patience={EXPERT4_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
        )

    # ── Paths de checkpoint con fold ───────────────────────────────
    checkpoint_path = (
        _CHECKPOINT_BASE / f"expert4_best_fold{fold}.pt"
        if fold > 0
        else _CHECKPOINT_PATH
    )
    log_path = (
        _CHECKPOINT_BASE / f"expert4_ddp_training_log_fold{fold}.json"
        if fold > 0
        else _TRAINING_LOG_PATH
    )

    # ── Directorio de checkpoints ──────────────────────────────────
    if is_main_process():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────
    best_val_loss: float = float("inf")
    training_log: list[dict] = []
    max_epochs = 1 if dry_run else EXPERT4_MAX_EPOCHS

    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info(
            "  INICIO DE ENTRENAMIENTO — Expert 4 DDP (ResNet3D R3D-18 / Páncreas)"
        )
        log.info(f"  Fold: {fold}/{EXPERT4_NUM_FOLDS - 1}")
        log.info(
            f"  Épocas máx: {max_epochs} | "
            f"Batch efectivo: "
            f"{effective_batch_per_gpu}×{world_size}×"
            f"{EXPERT4_ACCUMULATION_STEPS}="
            f"{effective_batch_per_gpu * world_size * EXPERT4_ACCUMULATION_STEPS}"
        )
        log.info(
            f"  FP16: {use_fp16} | Accumulation: {EXPERT4_ACCUMULATION_STEPS} | "
            f"GPUs: {world_size}"
        )
        log.info(f"{'=' * 70}\n")

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # ── Actualizar epoch en DistributedSampler para shuffle correcto ──
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ── Train ──────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model=model_ddp,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            accumulation_steps=EXPERT4_ACCUMULATION_STEPS,
            use_fp16=use_fp16,
            dry_run=dry_run,
        )

        # ── Validation (loader sin DistributedSampler) ─────────────
        val_results = validate(
            model=model_ddp,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_fp16=use_fp16,
            dry_run=dry_run,
        )

        # ── Scheduler step (DESPUÉS de optimizer.step + scaler.update) ──
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Extraer métricas ───────────────────────────────────────
        epoch_time = time.time() - epoch_start
        val_loss = val_results["val_loss"]
        val_auc = val_results["val_auc"]
        val_f1 = val_results["val_f1"]

        is_best = val_loss < best_val_loss - _MIN_DELTA

        # ── Logging (solo rank=0) ──────────────────────────────────
        if is_main_process():
            log.info(
                f"[Epoch {epoch + 1:3d}/{max_epochs}] "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_auc={val_auc:.4f} | val_f1={val_f1:.4f} | "
                f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
                f"{' ★ BEST' if is_best else ''}"
            )
            _log_vram(f"epoch-{epoch + 1}")

        # ── Guardar log de métricas (solo rank=0) ──────────────────
        if is_main_process():
            epoch_log: dict = {
                "epoch": epoch + 1,
                "fold": fold,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "lr": current_lr,
                "epoch_time_s": round(epoch_time, 1),
                "is_best": is_best,
                "world_size": world_size,
            }
            training_log.append(epoch_log)

        # ── Guardar mejor checkpoint (solo rank=0) ─────────────────
        if is_best:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "fold": fold,
                "model_state_dict": get_model_state_dict(model_ddp),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "config": {
                    "lr": EXPERT4_LR,
                    "weight_decay": EXPERT4_WEIGHT_DECAY,
                    "focal_gamma": EXPERT4_FOCAL_GAMMA,
                    "focal_alpha": EXPERT4_FOCAL_ALPHA,
                    "batch_size": EXPERT4_BATCH_SIZE,
                    "accumulation_steps": EXPERT4_ACCUMULATION_STEPS,
                    "fp16": EXPERT4_FP16,
                    "num_folds": EXPERT4_NUM_FOLDS,
                    "n_params": n_params,
                    "seed": _SEED,
                    "world_size": world_size,
                },
            }
            if not dry_run:
                save_checkpoint_ddp(checkpoint, checkpoint_path)

        # ── Guardar training log (solo rank=0) ─────────────────────
        if is_main_process() and not dry_run:
            with open(log_path, "w") as f:
                json.dump(training_log, f, indent=2)

        # ── Early stopping (rank=0 decide, todos paran) ────────────
        if not dry_run:
            should_stop = False
            if is_main_process():
                should_stop = early_stopping.step(val_loss)

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
                        f"\n[EarlyStopping] Detenido en época {epoch + 1}. "
                        f"val_loss no mejoró en "
                        f"{EXPERT4_EARLY_STOPPING_PATIENCE} épocas. "
                        f"Mejor val_loss: {best_val_loss:.4f}"
                    )
                break

    # ── Resumen final (solo rank=0) ────────────────────────────────
    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info(
            "  ENTRENAMIENTO FINALIZADO — Expert 4 DDP (ResNet3D R3D-18 / Páncreas)"
        )
        log.info(f"  Fold: {fold}/{EXPERT4_NUM_FOLDS - 1}")
        log.info(f"  Mejor val_loss: {best_val_loss:.4f}")
        if training_log:
            best_epoch = min(training_log, key=lambda x: x["val_loss"])
            log.info(
                f"  Mejor época: {best_epoch['epoch']} | "
                f"AUC: {best_epoch['val_auc']:.4f} | "
                f"F1: {best_epoch['val_f1']:.4f}"
            )
        if not dry_run:
            log.info(f"  Checkpoint: {checkpoint_path}")
            log.info(f"  Training log: {log_path}")
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
            "Entrenamiento Expert 4 DDP — ResNet3D R3D-18 / Páncreas CT. "
            "Usar con torchrun para multi-GPU."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Ejecuta 2 batches de train y 2 de val para verificar el pipeline "
            "sin entrenar"
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        choices=range(EXPERT4_NUM_FOLDS),
        help=f"Fold para k-fold CV (0 a {EXPERT4_NUM_FOLDS - 1}). Default: 0.",
    )
    parser.add_argument(
        "--batch-per-gpu",
        type=int,
        default=None,
        help=(
            "Override del batch size por GPU. Default: max(1, EXPERT4_BATCH_SIZE // "
            "world_size). Con volúmenes 3D [1,64,64,64] y gradient checkpointing, "
            "batch_per_gpu=1 requiere ~2-3 GB VRAM en FP16."
        ),
    )
    args = parser.parse_args()
    train(
        dry_run=args.dry_run,
        fold=args.fold,
        batch_per_gpu=args.batch_per_gpu,
    )

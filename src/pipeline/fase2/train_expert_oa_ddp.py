"""
Script de entrenamiento DDP para Expert OA — EfficientNet-B3 sobre Osteoarthritis Knee.

Versión multi-GPU de train_expert_oa.py usando DistributedDataParallel (DDP).
Funciona transparentemente en modo single-GPU si solo hay 1 GPU disponible.

Pipeline de fase única (sin LP-FT, sin freeze/unfreeze):

    Adam diferencial: lr_backbone=5e-5 / lr_head=5e-4
    CosineAnnealingLR(T_max=30, eta_min=1e-6)
    Early stopping por val_f1_macro (patience=10)
    Máximo 30 épocas

Loss: CrossEntropyLoss(weight=class_weights)
Métricas: F1-macro (principal), accuracy.
Sin CutMix/MixUp (prohibido por diseño del experto OA).

Lanzamiento:
    # Multi-GPU (2× Titan Xp):
    torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert_oa_ddp.py

    # Single-GPU:
    torchrun --nproc_per_node=1 src/pipeline/fase2/train_expert_oa_ddp.py

    # Dry-run:
    torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert_oa_ddp.py --dry-run

Dependencias:
    - src/pipeline/fase2/models/expert_oa_efficientnet_b3.py: ExpertOAEfficientNetB3
    - src/pipeline/fase2/dataloader_expert_oa.py: get_oa_dataloaders (datasets)
    - src/pipeline/fase2/expert_oa_config.py: hiperparámetros
    - src/pipeline/datasets/osteoarthritis.py: OAKneeDataset
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
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

# ── Configurar paths ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))
# OAKneeDataset.__getitem__ hace "from transform_domain import apply_clahe"
# que vive en src/pipeline/fase1/
_FASE1_ROOT = _PIPELINE_ROOT / "fase1"
if str(_FASE1_ROOT) not in sys.path:
    sys.path.insert(0, str(_FASE1_ROOT))

from fase2.models.expert_oa_efficientnet_b3 import ExpertOAEfficientNetB3
from fase2.expert_oa_config import (
    EXPERT_OA_LR_BACKBONE,
    EXPERT_OA_LR_HEAD,
    EXPERT_OA_WEIGHT_DECAY,
    EXPERT_OA_DROPOUT_FC,
    EXPERT_OA_BATCH_SIZE,
    EXPERT_OA_ACCUMULATION_STEPS,
    EXPERT_OA_FP16,
    EXPERT_OA_MAX_EPOCHS,
    EXPERT_OA_EARLY_STOPPING_PATIENCE,
    EXPERT_OA_EARLY_STOPPING_MONITOR,
    EXPERT_OA_NUM_CLASSES,
    EXPERT_OA_SCHEDULER_T_MAX,
    EXPERT_OA_SCHEDULER_ETA_MIN,
    EXPERT_OA_CONFIG_SUMMARY,
    EXPERT_OA_IMG_SIZE,
)
from datasets.osteoarthritis import OAKneeDataset
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
log = logging.getLogger("expert_oa_train_ddp")

# ── Rutas de salida ────────────────────────────────────────────────────
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "expert_02_efficientnet_b3" / "best.pt"
_TRAINING_LOG_PATH = (
    _CHECKPOINT_DIR / "expert_02_efficientnet_b3" / "expert_oa_ddp_training_log.json"
)

# ── Constantes de entrenamiento ────────────────────────────────────────
_SEED = 42
_MIN_DELTA = 0.001  # Mejora mínima para considerar progreso en early stopping

# ── Rutas de datos OA ─────────────────────────────────────────────────
_OA_ROOT_DIR = _PROJECT_ROOT / "datasets" / "osteoarthritis" / "oa_splits"


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
            f"[INFO] [ExpertOA] [Seed] Semillas fijadas a {effective_seed} "
            f"(base={seed} + rank={get_rank()})"
        )


def _log_vram(tag: str = "") -> None:
    """Imprime uso actual de VRAM si hay GPU disponible (solo rank 0)."""
    if torch.cuda.is_available() and is_main_process():
        dev = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(dev) / 1e9
        reserved = torch.cuda.memory_reserved(dev) / 1e9
        log.info(
            f"[INFO] [ExpertOA] [VRAM{' ' + tag if tag else ''}] "
            f"GPU {dev}: Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB"
        )


class EarlyStopping:
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
) -> dict[str, OAKneeDataset | Subset | torch.Tensor]:
    """Construye los datasets OA Knee (sin DataLoader) para aplicar DDP samplers.

    OAKneeDataset maneja transforms internamente:
      - CLAHE antes del resize
      - Augmentation online en train
      - Base transform (ToTensor + Normalize) para val/test
    """
    root_dir = _OA_ROOT_DIR

    if not root_dir.exists():
        raise FileNotFoundError(
            f"[ExpertOA/DataLoader] Directorio raíz no encontrado: {root_dir}"
        )

    img_size = EXPERT_OA_IMG_SIZE

    # Crear datasets
    train_ds = OAKneeDataset(
        root_dir=root_dir,
        split="train",
        img_size=img_size,
        mode="expert",
    )

    val_ds = OAKneeDataset(
        root_dir=root_dir,
        split="val",
        img_size=img_size,
        mode="expert",
    )

    test_ds = OAKneeDataset(
        root_dir=root_dir,
        split="test",
        img_size=img_size,
        mode="expert",
    )

    # Obtener class_weights
    if hasattr(train_ds, "class_weights") and train_ds.class_weights is not None:
        class_weights = train_ds.class_weights
    else:
        # Fallback: pesos uniformes
        class_weights = torch.ones(EXPERT_OA_NUM_CLASSES, dtype=torch.float32)

    # Limitar muestras si max_samples activo (dry-run)
    if max_samples is not None:
        n_train = min(max_samples, len(train_ds))
        n_val = min(max_samples, len(val_ds))
        n_test = min(max_samples, len(test_ds))
        train_ds = Subset(train_ds, list(range(n_train)))
        val_ds = Subset(val_ds, list(range(n_val)))
        test_ds = Subset(test_ds, list(range(n_test)))

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
) -> float:
    """Ejecuta una época de entrenamiento con DDP + gradient accumulation + FP16.

    Sin CutMix/MixUp (prohibido para Expert OA por diseño).
    Usa model.no_sync() en pasos intermedios de accumulation.

    Orden correcto: optimizer.step() → scaler.update() (bug conocido #1).
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, (imgs, labels, _img_names) in enumerate(loader):
        if dry_run and batch_idx >= 2:
            break

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        # ── Determinar si es paso intermedio (no_sync) o final (sync) ──
        is_accumulation_step = ((batch_idx + 1) % accumulation_steps) != 0

        with ddp_no_sync(model, active=is_accumulation_step):
            with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
                logits = model(imgs)  # [B, 5]
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

        # ── Optimizer step: optimizer.step() → scaler.update() ─────
        if not is_accumulation_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        n_batches += 1

        if dry_run and is_main_process():
            log.info(
                f"  [INFO] [ExpertOA] [Train batch {batch_idx}] "
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
) -> dict[str, float]:
    """Ejecuta validación y calcula métricas.

    Métricas:
        - val_loss: CrossEntropyLoss promedio
        - val_f1_macro: F1-score macro (métrica principal)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_labels: list[int] = []
    all_preds: list[int] = []

    for batch_idx, (imgs, labels, _img_names) in enumerate(loader):
        if dry_run and batch_idx >= 2:
            break

        imgs = imgs.to(device, non_blocking=True)
        labels_dev = labels.long().to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(imgs)  # [B, 5]
            loss = criterion(logits, labels_dev)

        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

        if dry_run and is_main_process():
            log.info(
                f"  [INFO] [ExpertOA] [Val batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"logits={list(logits.shape)} | "
                f"loss={loss.item():.4f}"
            )

    avg_loss = total_loss / max(n_batches, 1)

    # ── Métricas ───────────────────────────────────────────────────
    f1_macro = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    return {
        "val_loss": avg_loss,
        "val_f1_macro": f1_macro,
    }


def train(
    dry_run: bool = False,
    batch_per_gpu: int | None = None,
) -> None:
    """Función principal de entrenamiento del Expert OA con DDP.

    Fase única (sin LP-FT): Adam diferencial con CosineAnnealingLR + early stopping.

    Args:
        dry_run: si True, ejecuta 2 batches de train y 2 de val.
        batch_per_gpu: override del batch size por GPU. Si None, se calcula
            automáticamente como EXPERT_OA_BATCH_SIZE // world_size.
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
        log.info(f"[INFO] [ExpertOA] Dispositivo: {device}")
        log.info(f"[INFO] [ExpertOA] World size: {get_world_size()}")
        if device.type == "cpu":
            log.warning(
                "[INFO] [ExpertOA] Entrenando en CPU — será lento. "
                "Se recomienda GPU con >= 8 GB VRAM."
            )

    # ── Configuración ──────────────────────────────────────────────
    world_size = get_world_size()

    if batch_per_gpu is None:
        effective_batch_per_gpu = EXPERT_OA_BATCH_SIZE // world_size
    else:
        effective_batch_per_gpu = batch_per_gpu

    use_fp16 = EXPERT_OA_FP16 and device.type == "cuda"

    if is_main_process():
        log.info(f"[INFO] [ExpertOA] Config: {EXPERT_OA_CONFIG_SUMMARY}")
        log.info(
            f"[INFO] [ExpertOA] DDP batch: {effective_batch_per_gpu}/gpu × {world_size} GPUs "
            f"× {EXPERT_OA_ACCUMULATION_STEPS} accum = "
            f"{effective_batch_per_gpu * world_size * EXPERT_OA_ACCUMULATION_STEPS} "
            f"efectivo"
        )
        if not use_fp16 and EXPERT_OA_FP16:
            log.info(
                "[INFO] [ExpertOA] FP16 desactivado (no hay GPU). Usando FP32 en CPU."
            )
        if dry_run:
            log.info(
                "[INFO] [ExpertOA] === MODO DRY-RUN === (2 batches train + 2 batches val)"
            )

    # ── Modelo ─────────────────────────────────────────────────────
    # En DDP, solo rank=0 descarga los pesos pretrained (timm).
    # Los demás ranks esperan con barrier() y luego cargan desde caché.
    if is_main_process():
        model = ExpertOAEfficientNetB3(
            num_classes=EXPERT_OA_NUM_CLASSES,
            dropout=EXPERT_OA_DROPOUT_FC,
            pretrained=True,
        ).to(device)
    if is_ddp_initialized():
        torch.distributed.barrier()
    if not is_main_process():
        model = ExpertOAEfficientNetB3(
            num_classes=EXPERT_OA_NUM_CLASSES,
            dropout=EXPERT_OA_DROPOUT_FC,
            pretrained=True,
        ).to(device)

    if is_main_process():
        n_params = model.count_parameters()
        log.info(
            f"[INFO] [ExpertOA] Modelo EfficientNet-B3 creado: "
            f"{n_params:,} parámetros entrenables"
        )
        _log_vram("post-model")

    # ── Datasets (sin DataLoader, para aplicar DDP sampler) ────────
    num_workers_base = 0 if dry_run else max(1, os.cpu_count() // (2 * world_size))

    datasets = _build_datasets(
        max_samples=64 if dry_run else None,
    )

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    class_weights = datasets["class_weights"]

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
            f"[INFO] [ExpertOA] [DataLoader] Train: {len(train_ds):,} samples, "
            f"batch_per_gpu={effective_batch_per_gpu}"
        )
        log.info(f"[INFO] [ExpertOA] [DataLoader] Val: {len(val_ds):,}")

    # ── Loss ───────────────────────────────────────────────────────
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if is_main_process():
        log.info(
            f"[INFO] [ExpertOA] Loss: CrossEntropyLoss("
            f"weight=class_weights[{class_weights.shape[0]}])"
        )

    # ── Optimizer (Adam diferencial) ──────────────────────────────
    optimizer = torch.optim.Adam(
        [
            {"params": model.get_backbone_params(), "lr": EXPERT_OA_LR_BACKBONE},
            {"params": model.get_head_params(), "lr": EXPERT_OA_LR_HEAD},
        ],
        weight_decay=EXPERT_OA_WEIGHT_DECAY,
    )
    if is_main_process():
        log.info(
            f"[INFO] [ExpertOA] Optimizer: Adam diferencial "
            f"(backbone_lr={EXPERT_OA_LR_BACKBONE}, head_lr={EXPERT_OA_LR_HEAD}, "
            f"wd={EXPERT_OA_WEIGHT_DECAY})"
        )

    # ── Scheduler ──────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EXPERT_OA_SCHEDULER_T_MAX,
        eta_min=EXPERT_OA_SCHEDULER_ETA_MIN,
    )
    # Fix: evitar falso positivo "lr_scheduler.step() before optimizer.step()".
    # En dry-run con FP16, GradScaler puede saltar optimizer.step() si detecta
    # inf/nan en los gradientes del primer mini-batch, dejando _opt_called=False.
    # Al crear el scheduler, su constructor ya computó el LR de epoch 0 vía
    # step() interno, así que el primer scheduler.step() explícito es correcto.
    # Marcar _opt_called=True no altera el schedule de LR ni el estado del optimizer.
    optimizer._opt_called = True
    if is_main_process():
        log.info(
            f"[INFO] [ExpertOA] Scheduler: CosineAnnealingLR"
            f"(T_max={EXPERT_OA_SCHEDULER_T_MAX}, eta_min={EXPERT_OA_SCHEDULER_ETA_MIN})"
        )

    # ── GradScaler para FP16 ───────────────────────────────────────
    scaler = GradScaler(device=device.type, enabled=use_fp16)

    # ── Wrapping DDP ───────────────────────────────────────────────
    # Expert OA no usa freeze/unfreeze → todos los parámetros participan
    # → find_unused_parameters=False (más eficiente)
    model_ddp = wrap_model_ddp(model, device, find_unused_parameters=False)

    # ── Early stopping ─────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=EXPERT_OA_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    if is_main_process():
        log.info(
            f"[INFO] [ExpertOA] EarlyStopping: monitor={EXPERT_OA_EARLY_STOPPING_MONITOR}, "
            f"patience={EXPERT_OA_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
        )

    # ── Directorio de checkpoints ──────────────────────────────────
    if is_main_process():
        _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────
    best_f1_macro: float = -float("inf")
    training_log: list[dict] = []
    max_epochs = 2 if dry_run else EXPERT_OA_MAX_EPOCHS

    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info(
            "  [INFO] [ExpertOA] INICIO DE ENTRENAMIENTO — Expert OA DDP "
            "(EfficientNet-B3 / OA Knee)"
        )
        log.info(
            f"  Épocas max: {max_epochs} | Batch efectivo: "
            f"{effective_batch_per_gpu}×{world_size}×{EXPERT_OA_ACCUMULATION_STEPS}="
            f"{effective_batch_per_gpu * world_size * EXPERT_OA_ACCUMULATION_STEPS}"
        )
        log.info(
            f"  FP16: {use_fp16} | Accumulation: {EXPERT_OA_ACCUMULATION_STEPS} | "
            f"GPUs: {world_size}"
        )
        log.info(f"{'=' * 70}\n")

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # ── Actualizar epoch en DistributedSampler ─────────────────
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
            accumulation_steps=EXPERT_OA_ACCUMULATION_STEPS,
            use_fp16=use_fp16,
            dry_run=dry_run,
        )

        # ── Validation ─────────────────────────────────────────────
        val_results = validate(
            model=model_ddp,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_fp16=use_fp16,
            dry_run=dry_run,
        )

        # ── Scheduler step DESPUÉS de optimizer.step() (bug conocido #1) ──
        scheduler.step()
        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]

        # ── Extraer métricas ───────────────────────────────────────
        epoch_time = time.time() - epoch_start
        val_loss = val_results["val_loss"]
        val_f1_macro = val_results["val_f1_macro"]

        is_best = val_f1_macro > best_f1_macro + _MIN_DELTA

        # ── Logging (solo rank=0) ──────────────────────────────────
        if is_main_process():
            log.info(
                f"[INFO] [ExpertOA] [Epoch {epoch + 1:3d}/{max_epochs}] "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_f1_macro={val_f1_macro:.4f} | "
                f"lr_bb={current_lr_backbone:.2e} lr_hd={current_lr_head:.2e} | "
                f"time={epoch_time:.1f}s"
                f"{' ★ BEST' if is_best else ''}"
            )
            _log_vram(f"epoch-{epoch + 1}")

        # ── Guardar log de métricas (solo rank=0) ──────────────────
        if is_main_process():
            epoch_log: dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1_macro": val_f1_macro,
                "lr_backbone": current_lr_backbone,
                "lr_head": current_lr_head,
                "epoch_time_s": round(epoch_time, 1),
                "is_best": is_best,
                "world_size": get_world_size(),
            }
            training_log.append(epoch_log)

        # ── Guardar mejor checkpoint (solo rank=0) ─────────────────
        if is_best:
            best_f1_macro = val_f1_macro
            n_params = get_unwrapped_model(model_ddp).count_parameters()
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": get_model_state_dict(model_ddp),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_f1_macro": val_f1_macro,
                "config": {
                    "lr_backbone": EXPERT_OA_LR_BACKBONE,
                    "lr_head": EXPERT_OA_LR_HEAD,
                    "weight_decay": EXPERT_OA_WEIGHT_DECAY,
                    "dropout_fc": EXPERT_OA_DROPOUT_FC,
                    "batch_size": EXPERT_OA_BATCH_SIZE,
                    "accumulation_steps": EXPERT_OA_ACCUMULATION_STEPS,
                    "fp16": EXPERT_OA_FP16,
                    "num_classes": EXPERT_OA_NUM_CLASSES,
                    "scheduler_t_max": EXPERT_OA_SCHEDULER_T_MAX,
                    "scheduler_eta_min": EXPERT_OA_SCHEDULER_ETA_MIN,
                    "n_params": n_params,
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
        if not dry_run:
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
                        f"\n[INFO] [ExpertOA] [EarlyStopping] Detenido en época {epoch + 1}. "
                        f"val_f1_macro no mejoró en "
                        f"{EXPERT_OA_EARLY_STOPPING_PATIENCE} épocas. "
                        f"Mejor val_f1_macro: {best_f1_macro:.4f}"
                    )
                break

    # ── Resumen final (solo rank=0) ────────────────────────────────
    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info(
            "  [INFO] [ExpertOA] ENTRENAMIENTO FINALIZADO — Expert OA DDP "
            "(EfficientNet-B3 / OA Knee)"
        )
        log.info(f"  Mejor val_f1_macro: {best_f1_macro:.4f}")
        if training_log:
            best_epoch = max(training_log, key=lambda x: x["val_f1_macro"])
            log.info(
                f"  Mejor época: {best_epoch['epoch']} | "
                f"F1-macro: {best_epoch['val_f1_macro']:.4f}"
            )
        if not dry_run:
            log.info(f"  Checkpoint: {_CHECKPOINT_PATH}")
            log.info(f"  Training log: {_TRAINING_LOG_PATH}")
        log.info(f"{'=' * 70}")

        if dry_run:
            log.info(
                "\n[INFO] [ExpertOA] [DRY-RUN] Pipeline verificado exitosamente. "
                "Ejecuta sin --dry-run para entrenar."
            )

    # ── Cleanup DDP ────────────────────────────────────────────────
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Entrenamiento Expert OA DDP — EfficientNet-B3 / Osteoarthritis Knee. "
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
        "--batch-per-gpu",
        type=int,
        default=None,
        help=(
            "Override del batch size por GPU. Default: EXPERT_OA_BATCH_SIZE // world_size"
        ),
    )
    args = parser.parse_args()
    train(
        dry_run=args.dry_run,
        batch_per_gpu=args.batch_per_gpu,
    )

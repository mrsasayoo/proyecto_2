"""
Script de entrenamiento DDP para Expert 3 — DenseNet 3D sobre LUNA16.

Versión multi-GPU de train_expert3.py usando DistributedDataParallel (DDP).
Funciona transparentemente en modo single-GPU si solo hay 1 GPU disponible.

Pipeline de entrenamiento (sin LP-FT, entrenamiento from-scratch):

    Fase única (hasta 100 épocas):
        DenseNet 3D (~6.7M params) entrenado desde cero.
        AdamW(lr=3e-4) + CosineAnnealingWarmRestarts(T_0=15, T_mult=2).
        FocalLoss(gamma=2, alpha=0.85) con label_smoothing=0.05.
        Early stopping por val_loss (patience=20).

Manejo del desbalance de clases:
    - Ratio neg:pos ≈ 10:1 en dataset base.
    - FocalLoss(alpha=0.85) pondera la clase positiva (nódulos, minoritaria).
    - Label smoothing 0.05: {0,1} → {0.025, 0.975}.
    - SpatialDropout3d(0.15) + Dropout FC(0.4) + weight_decay=0.03.

Datos 3D y VRAM:
    Patches de forma (1, 64, 64, 64) — volúmenes CT monocanal.
    Con 2× Titan Xp (12 GB), batch_per_gpu = EXPERT3_BATCH_SIZE // world_size
    = 8 // 2 = 4. Batch efectivo = 4 × 2 GPUs × 4 accum = 32 (idéntico al original).
    FP16 habilitado — obligatorio para 12 GB VRAM con volúmenes 3D.

Lanzamiento (usa torchrun para detectar GPUs automáticamente):
    # Multi-GPU (2× Titan Xp):
    torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert3_ddp.py

    # Single-GPU (fallback transparente):
    torchrun --nproc_per_node=1 src/pipeline/fase2/train_expert3_ddp.py

    # Dry-run:
    torchrun --nproc_per_node=2 src/pipeline/fase2/train_expert3_ddp.py --dry-run

    # O con el script wrapper:
    bash run_expert.sh 3

Dependencias:
    - src/pipeline/fase2/models/expert3_densenet3d.py: Expert3MC318
    - src/pipeline/fase2/dataloader_expert3.py: LUNA16ExpertDataset, _load_label_map, etc.
    - src/pipeline/fase2/expert3_config.py: hiperparámetros
    - src/pipeline/fase2/losses.py: FocalLoss
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
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# ── Configurar paths ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from fase2.models.expert3_densenet3d import Expert3MC318
from fase2.dataloader_expert3 import (
    LUNA16ExpertDataset,
    _resolve_csv_path,
    _load_label_map,
)
from fase2.expert3_config import (
    EXPERT3_LR,
    EXPERT3_WEIGHT_DECAY,
    EXPERT3_FOCAL_GAMMA,
    EXPERT3_FOCAL_ALPHA,
    EXPERT3_LABEL_SMOOTHING,
    EXPERT3_DROPOUT_FC,
    EXPERT3_SPATIAL_DROPOUT_3D,
    EXPERT3_BATCH_SIZE,
    EXPERT3_ACCUMULATION_STEPS,
    EXPERT3_FP16,
    EXPERT3_MAX_EPOCHS,
    EXPERT3_EARLY_STOPPING_PATIENCE,
    EXPERT3_EARLY_STOPPING_MONITOR,
    EXPERT3_CONFIG_SUMMARY,
)
from fase2.losses import FocalLoss
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
log = logging.getLogger("expert3_train_ddp")

# ── Rutas de salida ────────────────────────────────────────────────────
_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "expert_03_densenet3d" / "best.pt"
_TRAINING_LOG_PATH = (
    _CHECKPOINT_DIR / "expert_03_densenet3d" / "expert3_ddp_training_log.json"
)

# ── Rutas de datos ─────────────────────────────────────────────────────
_PATCHES_BASE = PROJECT_ROOT / "datasets" / "luna_lung_cancer" / "patches"

# ── Constantes de entrenamiento ────────────────────────────────────────
_SEED = 42
_MIN_DELTA = 0.001  # Mejora mínima para considerar progreso en early stopping


# ═══════════════════════════════════════════════════════════════════════
# Utilidades
# ═══════════════════════════════════════════════════════════════════════


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
            f"[INFO] [Expert3] [Seed] Semillas fijadas a {effective_seed} "
            f"(base={seed} + rank={get_rank()})"
        )


def _log_vram(tag: str = "") -> None:
    """Imprime uso actual de VRAM si hay GPU disponible (solo rank 0)."""
    if torch.cuda.is_available() and is_main_process():
        dev = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(dev) / 1e9
        reserved = torch.cuda.memory_reserved(dev) / 1e9
        log.info(
            f"[INFO] [Expert3] [VRAM{' ' + tag if tag else ''}] "
            f"GPU {dev}: Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB"
        )


def _enable_gradient_checkpointing(model: Expert3MC318) -> bool:
    """Habilita gradient checkpointing en los dense blocks del DenseNet 3D.

    DenseNet 3D no tiene gradient_checkpointing_enable() nativo.
    Se aplica manualmente usando torch.utils.checkpoint en cada dense block,
    liberando activaciones intermedias y recalculándolas en el backward.

    Returns:
        True si se habilitó, False si no fue posible.
    """
    try:
        from torch.utils.checkpoint import checkpoint

        if not hasattr(model, "dense_blocks") or not model.dense_blocks:
            if is_main_process():
                log.warning(
                    "[INFO] [Expert3] [GradCheckpoint] Modelo sin atributo "
                    "'dense_blocks'. Gradient checkpointing no aplicable."
                )
            return False

        checkpointed_count = 0
        for _idx, block in enumerate(model.dense_blocks):

            def make_checkpointed_forward(original_fwd):
                def checkpointed_forward(x):
                    return checkpoint(original_fwd, x, use_reentrant=False)

                return checkpointed_forward

            block.forward = make_checkpointed_forward(block.forward)
            checkpointed_count += 1

        if is_main_process():
            log.info(
                f"[INFO] [Expert3] [GradCheckpoint] Gradient checkpointing HABILITADO "
                f"en {checkpointed_count} dense blocks "
                f"(dense_blocks[0..{checkpointed_count - 1}])"
            )
        return True
    except Exception as e:
        if is_main_process():
            log.warning(f"[INFO] [Expert3] [GradCheckpoint] No se pudo habilitar: {e}")
        return False


class EarlyStopping:
    """Early stopping por val_loss (minimizar) con patience configurable.

    Detiene el entrenamiento si val_loss no mejora (delta > min_delta)
    durante 'patience' épocas consecutivas.
    """

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


# ═══════════════════════════════════════════════════════════════════════
# Training y Validación
# ═══════════════════════════════════════════════════════════════════════


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

    Orden correcto del paso de optimización (bug fix vs expert1_ddp):
        1. optimizer.step()  — aplica gradientes
        2. scaler.update()   — ajusta escala FP16
        3. (scheduler.step() se llama fuera, después de la época completa)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, (volumes, labels, _stems) in enumerate(loader):
        if dry_run and batch_idx >= 2:
            break

        volumes = volumes.to(device, non_blocking=True)
        # labels es int desde el dataset → convertir a float para FocalLoss
        labels = labels.float().to(device, non_blocking=True)

        # ── Determinar si es paso intermedio (no_sync) o final (sync) ──
        is_accumulation_step = ((batch_idx + 1) % accumulation_steps) != 0

        with ddp_no_sync(model, active=is_accumulation_step):
            with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
                logits = model(volumes)  # [B, 2]
                # FocalLoss espera logits [B] y targets [B] float
                # Usamos logits[:, 1] (logit de clase positiva)
                # Aplicar label smoothing: {0,1} → {0.025, 0.975}
                labels_smooth = (
                    labels * (1 - EXPERT3_LABEL_SMOOTHING)
                    + EXPERT3_LABEL_SMOOTHING / 2.0
                )
                loss = criterion(logits[:, 1], labels_smooth)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

        # ── Optimizer step: optimizer.step() → scaler.update() ─────────
        if not is_accumulation_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        n_batches += 1

        if dry_run and is_main_process():
            log.info(
                f"[INFO] [Expert3]   [Train batch {batch_idx}] "
                f"volume={list(volumes.shape)} | "
                f"logits={list(logits.shape)} | "
                f"loss={loss.item() * accumulation_steps:.4f}"
            )

    # Flush de gradientes residuales si el último bloque no completó accumulation
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
) -> dict[str, float | list[list[int]]]:
    """Ejecuta validación y calcula métricas binarias.

    Métricas calculadas:
        - val_loss: loss promedio
        - val_f1_macro: F1 macro (promedio de F1 por clase)
        - val_auc: AUC-ROC binario (con protección contra NaN)
        - confusion_matrix: [[TN, FP], [FN, TP]]

    En modo DDP, la validación se ejecuta en todos los procesos pero las
    métricas se computan solo en rank=0 (val_loader sin DistributedSampler).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_labels: list[int] = []
    all_probs: list[float] = []
    all_preds: list[int] = []

    for batch_idx, (volumes, labels, _stems) in enumerate(loader):
        if dry_run and batch_idx >= 1:
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

        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

        if dry_run and is_main_process():
            log.info(
                f"[INFO] [Expert3]   [Val batch {batch_idx}] "
                f"volume={list(volumes.shape)} | "
                f"logits={list(logits.shape)} | "
                f"loss={loss.item():.4f}"
            )

    avg_loss = total_loss / max(n_batches, 1)

    # ── Métricas (con protección contra NaN — bug fix) ─────────────
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)

    # F1 Macro (safe: zero_division=0 maneja clases sin muestras)
    unique_labels = np.unique(labels_arr)
    if len(unique_labels) >= 2:
        f1_macro = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    else:
        f1_macro = 0.0
        if is_main_process():
            log.warning(
                "[INFO] [Expert3] [Val] Solo una clase presente en labels → F1=0.0"
            )

    # AUC-ROC (safe: verificar ambas clases presentes para evitar ValueError)
    if len(unique_labels) >= 2:
        try:
            auc = float(roc_auc_score(labels_arr, probs_arr))
        except ValueError:
            auc = 0.0
    else:
        auc = 0.0
        if is_main_process():
            log.warning("[INFO] [Expert3] [Val] Solo una clase presente → AUC=0.0")

    # Confusion matrix
    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])

    return {
        "val_loss": avg_loss,
        "val_f1_macro": float(f1_macro),
        "val_auc": auc,
        "confusion_matrix": cm.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════
# Datasets (sin DataLoader — para inyectar DistributedSampler)
# ═══════════════════════════════════════════════════════════════════════


def _build_datasets(
    patches_base: Path,
    candidates_csv: Path | str | None,
    max_samples: int | None,
) -> dict[str, LUNA16ExpertDataset | Subset]:
    """Construye los datasets (sin DataLoader) para poder aplicar DDP samplers.

    Separa la creación de datasets de la creación de DataLoaders para que
    DDP pueda inyectar su DistributedSampler en el DataLoader de train.

    Args:
        patches_base: directorio raíz con subcarpetas {train, val, test}/.
        candidates_csv: ruta al CSV de candidatos (None para auto-detect).
        max_samples: limitar muestras por split (para dry-run).

    Returns:
        Dict con train_ds, val_ds, test_ds.
    """
    csv_path = _resolve_csv_path(candidates_csv)
    label_map = _load_label_map(csv_path)

    # Verificar directorios de splits
    for split_name in ("train", "val", "test"):
        split_dir = patches_base / split_name
        if not split_dir.exists():
            raise FileNotFoundError(
                f"[Expert3/DataLoader] Directorio no encontrado: {split_dir}"
            )

    train_ds = LUNA16ExpertDataset(
        patches_dir=patches_base / "train",
        label_map=label_map,
        split="train",
        augment_3d=True,
    )

    val_ds = LUNA16ExpertDataset(
        patches_dir=patches_base / "val",
        label_map=label_map,
        split="val",
        augment_3d=False,
    )

    test_ds = LUNA16ExpertDataset(
        patches_dir=patches_base / "test",
        label_map=label_map,
        split="test",
        augment_3d=False,
    )

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
    }


# ═══════════════════════════════════════════════════════════════════════
# Entrenamiento principal
# ═══════════════════════════════════════════════════════════════════════


def train(
    dry_run: bool = False,
    data_root: str | None = None,
    batch_per_gpu: int | None = None,
    gradient_checkpointing: bool = False,
) -> None:
    """Función principal de entrenamiento del Expert 3 con DDP.

    Expert 3 es entrenado from-scratch (DenseNet 3D), por lo que no hay
    pipeline LP-FT. Se usa una fase única de entrenamiento con early stopping.

    Args:
        dry_run: si True, ejecuta 2 batches de train y 1 de val.
        data_root: ruta raíz del proyecto. Si None, se auto-detecta.
        batch_per_gpu: override del batch size por GPU. Si None, se calcula
            automáticamente como EXPERT3_BATCH_SIZE // world_size.
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
        log.info(f"[INFO] [Expert3] Dispositivo: {device}")
        log.info(f"[INFO] [Expert3] World size: {get_world_size()}")
        if device.type == "cpu":
            log.warning(
                "[INFO] [Expert3] Entrenando en CPU — será muy lento. "
                "Se recomienda GPU con >= 12 GB VRAM."
            )

    # ── Configuración ──────────────────────────────────────────────
    world_size = get_world_size()

    # Batch por GPU: dividir el batch total entre las GPUs para mantener
    # el mismo batch efectivo (batch_per_gpu * world_size * accum ≈ original)
    # EXPERT3_BATCH_SIZE=4, con 2 GPUs → 2 per gpu.
    # Batch efectivo = 2 × 2 × 8 = 32 (idéntico al original: 4 × 8 = 32)
    if batch_per_gpu is None:
        effective_batch_per_gpu = max(1, EXPERT3_BATCH_SIZE // world_size)
    else:
        effective_batch_per_gpu = batch_per_gpu

    if is_main_process():
        log.info(f"[INFO] [Expert3] Config: {EXPERT3_CONFIG_SUMMARY}")
        log.info(
            f"[INFO] [Expert3] DDP batch: {effective_batch_per_gpu}/gpu "
            f"x {world_size} GPUs x {EXPERT3_ACCUMULATION_STEPS} accum = "
            f"{effective_batch_per_gpu * world_size * EXPERT3_ACCUMULATION_STEPS} "
            f"efectivo"
        )
        if dry_run:
            log.info(
                "[INFO] [Expert3] === MODO DRY-RUN === (2 batches train + 1 batch val)"
            )

    use_fp16 = EXPERT3_FP16 and device.type == "cuda"
    if is_main_process() and not use_fp16 and EXPERT3_FP16:
        log.info("[INFO] [Expert3] FP16 desactivado (no hay GPU). Usando FP32 en CPU.")

    # ── Modelo ─────────────────────────────────────────────────────
    model = Expert3MC318(
        spatial_dropout_p=EXPERT3_SPATIAL_DROPOUT_3D,
        fc_dropout_p=EXPERT3_DROPOUT_FC,
        num_classes=2,
    ).to(device)

    if is_main_process():
        n_params = model.count_parameters()
        n_params_total = model.count_all_parameters()
        log.info(
            f"[INFO] [Expert3] Modelo DenseNet3D creado: "
            f"{n_params:,} entrenables / {n_params_total:,} totales"
        )
        _log_vram("post-model")

    # ── Gradient checkpointing (opcional — solo si se pide con --gradient-checkpointing) ─
    if gradient_checkpointing and device.type == "cuda":
        _enable_gradient_checkpointing(model)
    elif is_main_process():
        log.info("[INFO] [Expert3] Gradient checkpointing DESACTIVADO (default)")

    # ── Envolver en DDP ────────────────────────────────────────────
    # Expert 3 es from-scratch → todos los params participan → find_unused=False
    model_ddp = wrap_model_ddp(model, device, find_unused_parameters=False)

    # ── Datasets (sin DataLoader, para aplicar DDP sampler) ────────
    num_workers_base = 0 if dry_run else max(1, os.cpu_count() // (2 * world_size))

    project_root = Path(data_root) if data_root else PROJECT_ROOT
    patches_base = project_root / "datasets" / "luna_lung_cancer" / "patches"

    datasets = _build_datasets(
        patches_base=patches_base,
        candidates_csv=None,
        max_samples=64 if dry_run else None,
    )

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]

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
    pin_mem = torch.cuda.is_available()
    val_loader = DataLoader(
        val_ds,
        batch_size=effective_batch_per_gpu,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers_base,
        pin_memory=pin_mem,
        persistent_workers=num_workers_base > 0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=effective_batch_per_gpu,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers_base,
        pin_memory=pin_mem,
        persistent_workers=num_workers_base > 0,
    )

    if is_main_process():
        log.info(
            f"[INFO] [Expert3] [DataLoader] Train: {len(train_ds):,} samples, "
            f"batch_per_gpu={effective_batch_per_gpu}"
        )
        log.info(
            f"[INFO] [Expert3] [DataLoader] Val: {len(val_ds):,} | "
            f"Test: {len(test_ds):,}"
        )

    # ── Loss ───────────────────────────────────────────────────────
    criterion = FocalLoss(gamma=EXPERT3_FOCAL_GAMMA, alpha=EXPERT3_FOCAL_ALPHA)
    if is_main_process():
        log.info(
            f"[INFO] [Expert3] Loss: FocalLoss(gamma={EXPERT3_FOCAL_GAMMA}, "
            f"alpha={EXPERT3_FOCAL_ALPHA})"
        )

    # ── Optimizer ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=EXPERT3_LR,
        weight_decay=EXPERT3_WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    if is_main_process():
        log.info(
            f"[INFO] [Expert3] Optimizer: AdamW(lr={EXPERT3_LR}, "
            f"wd={EXPERT3_WEIGHT_DECAY})"
        )

    # ── Scheduler ──────────────────────────────────────────────────
    # CosineAnnealingWarmRestarts: ciclos de coseno con reinicios
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,
        T_mult=2,
        eta_min=1e-6,
    )
    if is_main_process():
        log.info(
            "[INFO] [Expert3] Scheduler: CosineAnnealingWarmRestarts"
            "(T_0=15, T_mult=2, eta_min=1e-6)"
        )

    # ── GradScaler para FP16 ───────────────────────────────────────
    scaler = GradScaler(device=device.type, enabled=use_fp16)

    # ── Early stopping ─────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=EXPERT3_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    if is_main_process():
        log.info(
            f"[INFO] [Expert3] EarlyStopping: monitor={EXPERT3_EARLY_STOPPING_MONITOR}, "
            f"patience={EXPERT3_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
        )

    # ── Directorio de checkpoints ──────────────────────────────────
    if is_main_process():
        _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Estado global ──────────────────────────────────────────────
    best_val_loss: float = float("inf")
    training_log: list[dict] = []
    max_epochs = 2 if dry_run else EXPERT3_MAX_EPOCHS

    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info("  INICIO DE ENTRENAMIENTO — Expert 3 DDP (DenseNet3D / LUNA16)")
        log.info(
            f"  Epocas max: {max_epochs} | Batch efectivo: "
            f"{effective_batch_per_gpu}x{world_size}x"
            f"{EXPERT3_ACCUMULATION_STEPS}="
            f"{effective_batch_per_gpu * world_size * EXPERT3_ACCUMULATION_STEPS}"
        )
        log.info(
            f"  FP16: {use_fp16} | Accumulation: {EXPERT3_ACCUMULATION_STEPS} | "
            f"GPUs: {world_size}"
        )
        log.info(f"{'=' * 70}\n")

    # ══════════════════════════════════════════════════════════════
    # Training loop (fase única — from-scratch, sin LP-FT)
    # ══════════════════════════════════════════════════════════════
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
            accumulation_steps=EXPERT3_ACCUMULATION_STEPS,
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

        # ── Scheduler step (DESPUÉS de optimizer.step + scaler.update) ──
        # Bug fix: el scheduler NUNCA debe ejecutarse antes del primer
        # optimizer.step(). Aquí se ejecuta después de cada época completa,
        # que ya incluye N pasos de optimizer.
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Extraer métricas ───────────────────────────────────────
        epoch_time = time.time() - epoch_start
        val_loss = val_results["val_loss"]
        val_f1 = val_results["val_f1_macro"]
        val_auc = val_results["val_auc"]
        cm = val_results["confusion_matrix"]

        is_best = val_loss < best_val_loss - _MIN_DELTA

        # ── Logging (solo rank=0) ──────────────────────────────────
        if is_main_process():
            log.info(
                f"[INFO] [Expert3] [Epoch {epoch + 1:3d}/{max_epochs}] "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_f1_macro={val_f1:.4f} | val_auc={val_auc:.4f} | "
                f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
                f"{' ★ BEST' if is_best else ''}"
            )
            log.info(
                f"[INFO] [Expert3]          Confusion Matrix: "
                f"TN={cm[0][0]:>5} FP={cm[0][1]:>5} | "
                f"FN={cm[1][0]:>5} TP={cm[1][1]:>5}"
            )
            _log_vram(f"epoch-{epoch + 1}")

        # ── Guardar log de métricas (solo rank=0) ──────────────────
        if is_main_process():
            epoch_log: dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1_macro": val_f1,
                "val_auc": val_auc,
                "confusion_matrix": cm,
                "lr": current_lr,
                "epoch_time_s": round(epoch_time, 1),
                "is_best": is_best,
                "world_size": get_world_size(),
            }
            training_log.append(epoch_log)

        # ── Guardar mejor checkpoint (solo rank=0) ─────────────────
        if is_best:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": get_model_state_dict(model_ddp),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_f1_macro": val_f1,
                "val_auc": val_auc,
                "config": {
                    "lr": EXPERT3_LR,
                    "weight_decay": EXPERT3_WEIGHT_DECAY,
                    "focal_gamma": EXPERT3_FOCAL_GAMMA,
                    "focal_alpha": EXPERT3_FOCAL_ALPHA,
                    "label_smoothing": EXPERT3_LABEL_SMOOTHING,
                    "dropout_fc": EXPERT3_DROPOUT_FC,
                    "spatial_dropout_3d": EXPERT3_SPATIAL_DROPOUT_3D,
                    "batch_size": EXPERT3_BATCH_SIZE,
                    "accumulation_steps": EXPERT3_ACCUMULATION_STEPS,
                    "fp16": EXPERT3_FP16,
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

        # ── Early stopping (rank=0 decide, broadcast a todos) ──────
        # Barrier: sincronizar todos los ranks antes del broadcast.
        # Sin esto, rank 1 puede avanzar a la siguiente época (allreduce
        # de gradientes) mientras rank 0 aún está en el broadcast,
        # causando NCCL watchdog timeout por operaciones colectivas
        # divergentes entre ranks.
        if is_ddp_initialized():
            torch.distributed.barrier()

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
                        f"\n[INFO] [Expert3] [EarlyStopping] Detenido en epoca "
                        f"{epoch + 1}. val_loss no mejoro en "
                        f"{early_stopping.patience} epocas. "
                        f"Mejor val_loss: {best_val_loss:.4f}"
                    )
                break

    # ══════════════════════════════════════════════════════════════
    # EVALUACIÓN FINAL EN TEST SET (solo rank=0)
    # ══════════════════════════════════════════════════════════════
    if is_main_process():
        log.info(f"\n{'=' * 70}")
        log.info("  EVALUACION FINAL — Test set")
        log.info(f"{'=' * 70}\n")

        # Desempaquetar DDP para evaluación
        eval_model = get_unwrapped_model(model_ddp)

        # Cargar mejor checkpoint para evaluación
        if _CHECKPOINT_PATH.exists():
            ckpt = load_checkpoint_ddp(_CHECKPOINT_PATH, map_location=device)
            if ckpt is not None:
                eval_model.load_state_dict(ckpt["model_state_dict"])
                log.info(
                    f"[INFO] [Expert3] [Test] Cargado mejor checkpoint: "
                    f"epoca {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})"
                )
            else:
                log.warning(
                    "[INFO] [Expert3] [Test] Checkpoint existe pero no se pudo "
                    "cargar. Usando modelo del final del entrenamiento."
                )
        else:
            log.warning(
                "[INFO] [Expert3] [Test] No se encontró checkpoint en "
                f"{_CHECKPOINT_PATH}. Usando modelo del final del "
                "entrenamiento (resultados pueden no reflejar el mejor modelo)."
            )

        test_results = validate(
            model=eval_model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            use_fp16=use_fp16,
            dry_run=dry_run,
        )

        test_loss = test_results["val_loss"]
        test_f1 = test_results["val_f1_macro"]
        test_auc = test_results["val_auc"]
        test_cm = test_results["confusion_matrix"]

        log.info(
            f"[INFO] [Expert3] [Test] loss={test_loss:.4f} | "
            f"f1_macro={test_f1:.4f} | auc={test_auc:.4f}"
        )
        log.info(
            f"[INFO] [Expert3] [Test] Confusion Matrix: "
            f"TN={test_cm[0][0]:>5} FP={test_cm[0][1]:>5} | "
            f"FN={test_cm[1][0]:>5} TP={test_cm[1][1]:>5}"
        )

        # Agregar resultados test al training log
        if training_log:
            test_log = {
                "evaluation": "test",
                "test_loss": test_loss,
                "test_f1_macro": test_f1,
                "test_auc": test_auc,
                "test_confusion_matrix": test_cm,
            }
            training_log.append(test_log)

        # Guardar training log final
        if not dry_run:
            with open(_TRAINING_LOG_PATH, "w") as f:
                json.dump(training_log, f, indent=2)

        # ── Resumen final ──────────────────────────────────────────
        log.info(f"\n{'=' * 70}")
        log.info("  ENTRENAMIENTO FINALIZADO — Expert 3 DDP (DenseNet3D / LUNA16)")
        log.info(f"  Mejor val_loss: {best_val_loss:.4f}")
        log.info(
            f"  Test: loss={test_loss:.4f} | F1={test_f1:.4f} | AUC={test_auc:.4f}"
        )
        if training_log:
            epoch_logs = [e for e in training_log if "epoch" in e]
            if epoch_logs:
                best_epoch = min(epoch_logs, key=lambda x: x["val_loss"])
                log.info(
                    f"  Mejor epoca: {best_epoch['epoch']} | "
                    f"val_loss: {best_epoch['val_loss']:.4f} | "
                    f"F1: {best_epoch['val_f1_macro']:.4f} | "
                    f"AUC: {best_epoch['val_auc']:.4f}"
                )
        if not dry_run:
            log.info(f"  Checkpoint: {_CHECKPOINT_PATH}")
            log.info(f"  Training log: {_TRAINING_LOG_PATH}")
        log.info(f"{'=' * 70}")

        if dry_run:
            log.info(
                "\n[INFO] [Expert3] [DRY-RUN] Pipeline verificado exitosamente. "
                "Ejecuta sin --dry-run para entrenar."
            )

    # ── Cleanup DDP ────────────────────────────────────────────────
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Entrenamiento Expert 3 DDP — DenseNet3D / LUNA16. "
            "Usar con torchrun para multi-GPU."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Ejecuta 2 batches de train y 1 de val para verificar el pipeline "
            "sin entrenar (max_samples=64, max_epochs=2)"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Ruta raiz del proyecto (default: auto-detectada)",
    )
    parser.add_argument(
        "--batch-per-gpu",
        type=int,
        default=None,
        help=(
            "Override del batch size por GPU. Default: EXPERT3_BATCH_SIZE // world_size "
            "(4//2=2 con 2 GPUs). Con volumenes 3D de 64^3, 2 per GPU consume ~3-4 GB "
            "VRAM en FP16 — hay margen para subir a 3-4 si la temperatura lo permite."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Habilitar gradient checkpointing en dense blocks (reduce VRAM, "
            "aumenta tiempo). Usar solo si hay OOM con batch actual."
        ),
    )
    args = parser.parse_args()
    train(
        dry_run=args.dry_run,
        data_root=args.data_root,
        batch_per_gpu=args.batch_per_gpu,
        gradient_checkpointing=args.gradient_checkpointing,
    )

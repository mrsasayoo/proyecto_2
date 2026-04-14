"""
Script de entrenamiento para Expert 5 — Res-U-Net Autoencoder con FiLM
Domain Conditioning (Fase 3).

Reemplaza al CAE simple con una arquitectura más potente que incorpora
skip connections, bloques residuales pre-activación y FiLM conditioning
por dominio médico.

Pipeline completo:
    1. Carga hiperparámetros desde expert6_resunet_config.py
    2. Construye modelo ConditionedResUNetAE(in_ch=3, base_ch=64, n_domains=6)
    3. Dataset sin etiquetas — autoencoder (reconstrucción)
    4. Loss: MSE + 0.1 * L1 (ReconstructionLoss del propio módulo)
    5. Optimizador: Adam lr=1e-3 con CosineAnnealingLR T_max=50
    6. Early stopping por val_loss (patience=10)
    7. FP32 estricto — sin AMP, sin autocast, sin GradScaler
    8. Checkpoints reanudables (mejor modelo + último)
    9. Métricas de entrenamiento en CSV (logs/expert5/train_metrics.csv)
   10. Umbrales OOD post-entrenamiento (percentil 50 y 99 sobre val set)

Uso:
    # Dry-run: verifica el pipeline sin entrenar
    python src/pipeline/fase3/train_expert5.py --dry-run

    # Entrenamiento completo
    python src/pipeline/fase3/train_expert5.py

    # Sobrescribir épocas
    python src/pipeline/fase3/train_expert5.py --epochs 30

Dependencias:
    - src/pipeline/fase3/models/expert6_resunet.py: ConditionedResUNetAE, ReconstructionLoss
    - src/pipeline/datasets/cae.py: MultimodalCAEDataset
    - src/pipeline/fase3/expert6_resunet_config.py: hiperparámetros
    - datasets/cae_splits.csv: split multi-modal

Reanudable: si existe un checkpoint en checkpoints/expert5/last.pt,
lo carga y continúa desde la época guardada.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Configurar paths ──────────────────────────────────────────────────
# Se añaden dos directorios al sys.path:
#   1. src/pipeline/       — para resolver módulos globales (datasets, fase3, config)
#   2. src/pipeline/fase1/ — para resolver imports bare de fase1_config dentro de
#      la cadena transitiva: datasets/__init__ → luna.py → fase1/transform_3d.py
#      → from fase1_config import ...
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
_FASE1_ROOT = _PIPELINE_ROOT / "fase1"

for _p in [str(_PIPELINE_ROOT), str(_FASE1_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fase3.models.expert6_resunet import ConditionedResUNetAE, ReconstructionLoss
from fase3.expert6_resunet_config import (
    EXPERT6_BASE_CH,
    EXPERT6_BATCH_SIZE,
    EXPERT6_DROPOUT,
    EXPERT6_EARLY_STOPPING_PATIENCE,
    EXPERT6_EMBED_DIM,
    EXPERT6_IMG_SIZE,
    EXPERT6_IN_CHANNELS,
    EXPERT6_LOSS_LAMBDA_L1,
    EXPERT6_LR,
    EXPERT6_MAX_EPOCHS,
    EXPERT6_N_DOMAINS,
    EXPERT6_OOD_THRESHOLD_PERCENTILE_LEVE,
    EXPERT6_OOD_THRESHOLD_PERCENTILE_OOD,
    EXPERT6_T_MAX,
    EXPERT6_WEIGHT_DECAY,
    EXPERT6_CONFIG_SUMMARY,
)

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("expert5_resunet_train")

# ── Rutas de salida ───────────────────────────────────────────────────
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints" / "expert5"
_BEST_CKPT_PATH = _CHECKPOINT_DIR / "best.pt"
_LAST_CKPT_PATH = _CHECKPOINT_DIR / "last.pt"
_OOD_THRESHOLDS_PATH = _CHECKPOINT_DIR / "ood_thresholds.json"

_LOG_DIR = _PROJECT_ROOT / "logs" / "expert5"
_METRICS_CSV_PATH = _LOG_DIR / "train_metrics.csv"

# ── CSV path ──────────────────────────────────────────────────────────
_CAE_CSV = _PROJECT_ROOT / "datasets" / "cae_splits.csv"

# ── Constantes ────────────────────────────────────────────────────────
_SEED = 42
_MIN_DELTA = 0.0001
_DEFAULT_DOMAIN_ID = 5  # Unknown — fallback si no hay expert_id


# ══════════════════════════════════════════════════════════════════════
# Dataset con domain_id
# ══════════════════════════════════════════════════════════════════════


class CAEWithDomainDataset(Dataset):
    """Wrapper sobre MultimodalCAEDataset que también devuelve domain_id.

    Reutiliza la lógica de carga de MultimodalCAEDataset (2D images,
    LUNA patches, Pancreas NIfTI) pero añade el domain_id del CSV
    (columna ``expert_id``) para el FiLM conditioning del
    ConditionedResUNetAE.

    Returns:
        Tupla (img_tensor [3,224,224], domain_id int, path_str).
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        img_size: int = 224,
        project_root: str | None = None,
    ) -> None:
        from datasets.cae import MultimodalCAEDataset

        self._base_ds = MultimodalCAEDataset(
            csv_path=csv_path,
            split=split,
            img_size=img_size,
            project_root=project_root,
        )

        # Leer domain_ids del CSV (columna expert_id si existe)
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)

        has_domain_col = "domain_id" in df.columns
        has_expert_col = "expert_id" in df.columns

        if has_domain_col:
            self._domain_ids = df["domain_id"].values.astype(np.int64)
            log.info(f"[Dataset] Usando columna 'domain_id' del CSV")
        elif has_expert_col:
            self._domain_ids = df["expert_id"].values.astype(np.int64)
            log.info(f"[Dataset] Usando columna 'expert_id' como domain_id")
        else:
            self._domain_ids = np.full(len(df), _DEFAULT_DOMAIN_ID, dtype=np.int64)
            log.warning(
                f"[Dataset] No existe 'domain_id' ni 'expert_id' en CSV. "
                f"Usando default={_DEFAULT_DOMAIN_ID} (Unknown) para todo."
            )

        # Verificar que los domain_ids están en rango [0, n_domains)
        unique_ids = np.unique(self._domain_ids)
        log.info(
            f"[Dataset] Split '{split}': {len(self._domain_ids):,} muestras | "
            f"Domain IDs únicos: {sorted(unique_ids.tolist())}"
        )

    def __len__(self) -> int:
        return len(self._base_ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        img, path_str = self._base_ds[idx]
        domain_id = int(self._domain_ids[idx])
        return img, domain_id, path_str


# ══════════════════════════════════════════════════════════════════════
# Utilidades
# ══════════════════════════════════════════════════════════════════════


def set_seed(seed: int = _SEED) -> None:
    """Fija todas las semillas para reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info(f"[Seed] Semillas fijadas a {seed} (numpy, torch, cuda, cudnn)")


def _log_vram(tag: str = "") -> None:
    """Imprime uso actual de VRAM si hay GPU disponible."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        log.info(
            f"[VRAM{' ' + tag if tag else ''}] "
            f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB"
        )


class EarlyStopping:
    """Early stopping por val_loss con patience configurable.

    Detiene el entrenamiento si val_loss no mejora (delta > min_delta)
    durante ``patience`` épocas consecutivas.
    """

    def __init__(self, patience: int, min_delta: float = 0.0001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_val: float = float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """Retorna True si el entrenamiento debe detenerse."""
        if val_loss < self.best_val - self.min_delta:
            self.best_val = val_loss
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False


def _write_metrics_header(path: Path) -> None:
    """Escribe el header del CSV de métricas si no existe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_mse",
                    "lr",
                    "epoch_time_s",
                    "is_best",
                ]
            )


def _append_metrics_row(path: Path, row: dict) -> None:
    """Añade una fila al CSV de métricas."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["epoch"],
                f"{row['train_loss']:.6f}",
                f"{row['val_loss']:.6f}",
                f"{row['val_mse']:.6f}",
                f"{row['lr']:.2e}",
                f"{row['epoch_time_s']:.1f}",
                row["is_best"],
            ]
        )


# ══════════════════════════════════════════════════════════════════════
# DataLoaders
# ══════════════════════════════════════════════════════════════════════


def get_dataloaders(
    csv_path: str,
    project_root: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
    split: str = "train",
) -> DataLoader:
    """Construye un DataLoader para el split dado."""
    ds = CAEWithDomainDataset(
        csv_path=csv_path,
        split=split,
        img_size=img_size,
        project_root=project_root,
    )
    shuffle = split == "train"
    drop_last = split == "train"
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    return loader


# ══════════════════════════════════════════════════════════════════════
# Train / Validate
# ══════════════════════════════════════════════════════════════════════


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: ReconstructionLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dry_run: bool = False,
) -> float:
    """Ejecuta una época de entrenamiento en FP32.

    Returns:
        Loss promedio de la época.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (imgs, domain_ids, _paths) in enumerate(loader):
        if dry_run and batch_idx >= 2:
            break

        imgs = imgs.to(device, non_blocking=True)
        domain_ids = domain_ids.to(device, non_blocking=True)

        # Forward FP32 — sin autocast
        x_hat, _z = model(imgs, domain_ids)
        loss = loss_fn(x_hat, imgs)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if dry_run:
            log.info(
                f"  [Train batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"recon={list(x_hat.shape)} | "
                f"domain_ids={domain_ids.tolist()[:4]}... | "
                f"loss={loss.item():.6f}"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: ReconstructionLoss,
    device: torch.device,
    dry_run: bool = False,
) -> dict[str, float]:
    """Ejecuta validación.

    Returns:
        dict con keys: val_loss, val_mse.
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0

    for batch_idx, (imgs, domain_ids, _paths) in enumerate(loader):
        if dry_run and batch_idx >= 2:
            break

        imgs = imgs.to(device, non_blocking=True)
        domain_ids = domain_ids.to(device, non_blocking=True)

        x_hat, _z = model(imgs, domain_ids)
        mse_val = nn.functional.mse_loss(x_hat, imgs)
        loss = loss_fn(x_hat, imgs)

        total_loss += loss.item()
        total_mse += mse_val.item()
        n_batches += 1

        if dry_run:
            log.info(
                f"  [Val batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"recon={list(x_hat.shape)} | "
                f"mse={mse_val.item():.6f} | "
                f"loss={loss.item():.6f}"
            )

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_mse": total_mse / max(n_batches, 1),
    }


# ══════════════════════════════════════════════════════════════════════
# OOD Thresholds
# ══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_ood_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dry_run: bool = False,
) -> dict[str, float]:
    """Calcula umbrales OOD (percentil 50 y 99) sobre el val set.

    Usa ``model.reconstruction_error()`` que retorna MSE + 0.1*L1 por
    sample (tensor [B]).

    Returns:
        dict con threshold_p50 y threshold_p99.
    """
    model.eval()
    all_errors: list[torch.Tensor] = []

    for batch_idx, (imgs, domain_ids, _paths) in enumerate(loader):
        if dry_run and batch_idx >= 2:
            break

        imgs = imgs.to(device, non_blocking=True)
        domain_ids = domain_ids.to(device, non_blocking=True)

        errors = model.reconstruction_error(imgs, domain_ids)  # [B]
        all_errors.append(errors.cpu())

    all_errors_np = torch.cat(all_errors).numpy()

    p50 = float(np.percentile(all_errors_np, EXPERT6_OOD_THRESHOLD_PERCENTILE_LEVE))
    p99 = float(np.percentile(all_errors_np, EXPERT6_OOD_THRESHOLD_PERCENTILE_OOD))

    log.info(
        f"[OOD] Umbrales calculados sobre {len(all_errors_np):,} muestras: "
        f"p{EXPERT6_OOD_THRESHOLD_PERCENTILE_LEVE}={p50:.6f} | "
        f"p{EXPERT6_OOD_THRESHOLD_PERCENTILE_OOD}={p99:.6f}"
    )

    return {
        "threshold_p50": p50,
        "threshold_p99": p99,
        "n_samples": len(all_errors_np),
        "percentile_leve": EXPERT6_OOD_THRESHOLD_PERCENTILE_LEVE,
        "percentile_ood": EXPERT6_OOD_THRESHOLD_PERCENTILE_OOD,
        "error_mean": float(all_errors_np.mean()),
        "error_std": float(all_errors_np.std()),
    }


# ══════════════════════════════════════════════════════════════════════
# Checkpoint save / load
# ══════════════════════════════════════════════════════════════════════


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    val_loss: float,
    val_mse: float,
    best_val_loss: float,
    early_stop_counter: int,
) -> None:
    """Guarda checkpoint completo para reanudación."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "val_mse": val_mse,
        "best_val_loss": best_val_loss,
        "early_stop_counter": early_stop_counter,
        "config": {
            "base_ch": EXPERT6_BASE_CH,
            "n_domains": EXPERT6_N_DOMAINS,
            "embed_dim": EXPERT6_EMBED_DIM,
            "dropout": EXPERT6_DROPOUT,
            "lr": EXPERT6_LR,
            "weight_decay": EXPERT6_WEIGHT_DECAY,
            "batch_size": EXPERT6_BATCH_SIZE,
            "img_size": EXPERT6_IMG_SIZE,
            "in_channels": EXPERT6_IN_CHANNELS,
            "lambda_l1": EXPERT6_LOSS_LAMBDA_L1,
            "seed": _SEED,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, float, int]:
    """Carga checkpoint y retorna (start_epoch, best_val_loss, early_stop_counter).

    Returns:
        Tupla (start_epoch, best_val_loss, early_stop_counter).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt["epoch"]  # ya entrenada, empezar en la siguiente
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    early_stop_counter = ckpt.get("early_stop_counter", 0)

    log.info(
        f"[Resume] Checkpoint cargado de '{path.name}': "
        f"epoch={start_epoch} | best_val_loss={best_val_loss:.6f} | "
        f"early_stop_counter={early_stop_counter}"
    )
    return start_epoch, best_val_loss, early_stop_counter


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def train(dry_run: bool = False, epochs_override: int | None = None) -> None:
    """Función principal de entrenamiento del Expert 5 (Res-U-Net + FiLM).

    Args:
        dry_run: si True, ejecuta 2 batches × 2 épocas para verificación.
        epochs_override: si se proporciona, sobrescribe EXPERT6_MAX_EPOCHS.
    """
    set_seed(_SEED)

    # ── Dispositivo ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Expert5] Dispositivo: {device}")
    if device.type == "cpu":
        log.warning(
            "[Expert5] Entrenando en CPU — será lento. "
            "Se recomienda GPU con >= 10 GB VRAM."
        )

    # ── Configuración ─────────────────────────────────────────────
    log.info(f"[Expert5] Config: {EXPERT6_CONFIG_SUMMARY}")
    if dry_run:
        log.info("[Expert5] === MODO DRY-RUN === (2 batches × 2 épocas)")

    max_epochs: int
    if dry_run:
        max_epochs = 2
    elif epochs_override is not None:
        max_epochs = epochs_override
    else:
        max_epochs = EXPERT6_MAX_EPOCHS

    # ── Modelo ────────────────────────────────────────────────────
    model = ConditionedResUNetAE(
        in_ch=EXPERT6_IN_CHANNELS,
        base_ch=EXPERT6_BASE_CH,
        dropout=EXPERT6_DROPOUT,
        n_domains=EXPERT6_N_DOMAINS,
        embed_dim=EXPERT6_EMBED_DIM,
    ).to(device)

    n_params = model.count_parameters()
    log.info(
        f"[Expert5] Modelo ConditionedResUNetAE creado: "
        f"{n_params:,} parámetros entrenables ({n_params / 1e6:.2f}M)"
    )
    _log_vram("post-model")

    # ── DataLoaders ───────────────────────────────────────────────
    num_workers = 0 if dry_run else 4
    # Dry-run usa batch_size=4 para evitar OOM en CPU
    batch_size = 4 if (dry_run and device.type == "cpu") else EXPERT6_BATCH_SIZE

    train_loader = get_dataloaders(
        csv_path=str(_CAE_CSV),
        project_root=str(_PROJECT_ROOT),
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=EXPERT6_IMG_SIZE,
        split="train",
    )
    val_loader = get_dataloaders(
        csv_path=str(_CAE_CSV),
        project_root=str(_PROJECT_ROOT),
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=EXPERT6_IMG_SIZE,
        split="val",
    )

    log.info(
        f"[Expert5] DataLoaders: train={len(train_loader.dataset):,} | "
        f"val={len(val_loader.dataset):,} | "
        f"batch_size={batch_size} | num_workers={num_workers}"
    )

    # ── Loss ──────────────────────────────────────────────────────
    loss_fn = ReconstructionLoss(l1_weight=EXPERT6_LOSS_LAMBDA_L1)
    log.info(f"[Expert5] Loss: MSE + {EXPERT6_LOSS_LAMBDA_L1} * L1")

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=EXPERT6_LR,
        weight_decay=EXPERT6_WEIGHT_DECAY,
    )
    log.info(f"[Expert5] Optimizer: Adam(lr={EXPERT6_LR}, wd={EXPERT6_WEIGHT_DECAY})")

    # ── Scheduler ─────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EXPERT6_T_MAX,
    )
    log.info(f"[Expert5] Scheduler: CosineAnnealingLR(T_max={EXPERT6_T_MAX})")

    # ── Early stopping ────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=EXPERT6_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    log.info(
        f"[Expert5] EarlyStopping: monitor=val_loss, "
        f"patience={EXPERT6_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
    )

    # ── Crear directorios de salida ───────────────────────────────
    if not dry_run:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Reanudar desde checkpoint si existe ───────────────────────
    start_epoch = 0
    best_val_loss = float("inf")

    if _LAST_CKPT_PATH.exists() and not dry_run:
        start_epoch, best_val_loss, es_counter = _load_checkpoint(
            _LAST_CKPT_PATH,
            model,
            optimizer,
            scheduler,
            device,
        )
        early_stopping.best_val = best_val_loss
        early_stopping.counter = es_counter
        log.info(f"[Resume] Continuando desde epoch {start_epoch + 1}/{max_epochs}")
    elif not dry_run:
        # Inicializar CSV de métricas solo si empezamos de cero
        _write_metrics_header(_METRICS_CSV_PATH)

    # ── Training loop ─────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info(f"  INICIO DE ENTRENAMIENTO — Expert 5 (Res-U-Net + FiLM)")
    log.info(
        f"  Épocas: {start_epoch + 1}..{max_epochs} | "
        f"Batch: {EXPERT6_BATCH_SIZE} | FP32 | "
        f"base_ch: {EXPERT6_BASE_CH} | n_domains: {EXPERT6_N_DOMAINS}"
    )
    log.info(f"{'=' * 70}\n")

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()

        # ── Train ─────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            dry_run=dry_run,
        )

        # ── Validation ────────────────────────────────────────────
        val_results = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            dry_run=dry_run,
        )

        # ── Scheduler step ────────────────────────────────────────
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        val_loss = val_results["val_loss"]
        val_mse = val_results["val_mse"]

        # ── Log de época ──────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        is_best = val_loss < best_val_loss - _MIN_DELTA

        log.info(
            f"[Epoch {epoch + 1:3d}/{max_epochs}] "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val_mse={val_mse:.6f} | "
            f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
            f"{' ★ BEST' if is_best else ''}"
        )
        _log_vram(f"epoch-{epoch + 1}")

        # ── Guardar métricas ──────────────────────────────────────
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mse": val_mse,
            "lr": current_lr,
            "epoch_time_s": round(epoch_time, 1),
            "is_best": is_best,
        }

        if not dry_run:
            _append_metrics_row(_METRICS_CSV_PATH, epoch_row)

        # ── Guardar mejor checkpoint ──────────────────────────────
        if is_best:
            best_val_loss = val_loss
            if not dry_run:
                _save_checkpoint(
                    _BEST_CKPT_PATH,
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    val_loss=val_loss,
                    val_mse=val_mse,
                    best_val_loss=best_val_loss,
                    early_stop_counter=0,
                )
                log.info(f"  → Mejor checkpoint guardado: {_BEST_CKPT_PATH}")

        # ── Guardar último checkpoint (siempre, para reanudación) ─
        if not dry_run:
            _save_checkpoint(
                _LAST_CKPT_PATH,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                val_loss=val_loss,
                val_mse=val_mse,
                best_val_loss=best_val_loss,
                early_stop_counter=early_stopping.counter,
            )

        # ── Early stopping ────────────────────────────────────────
        if not dry_run:
            if early_stopping.step(val_loss):
                log.info(
                    f"\n[EarlyStopping] Detenido en época {epoch + 1}. "
                    f"val_loss no mejoró en "
                    f"{EXPERT6_EARLY_STOPPING_PATIENCE} épocas. "
                    f"Mejor val_loss: {best_val_loss:.6f}"
                )
                break

    # ── Umbrales OOD ──────────────────────────────────────────────
    log.info(f"\n[OOD] Calculando umbrales sobre validation set...")

    # Cargar mejor modelo para calcular umbrales
    if _BEST_CKPT_PATH.exists() and not dry_run:
        ckpt = torch.load(_BEST_CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"[OOD] Modelo cargado desde best checkpoint para umbrales")

    ood_thresholds = compute_ood_thresholds(
        model=model,
        loader=val_loader,
        device=device,
        dry_run=dry_run,
    )

    if not dry_run:
        with open(_OOD_THRESHOLDS_PATH, "w") as f:
            json.dump(ood_thresholds, f, indent=2)
        log.info(f"[OOD] Umbrales guardados: {_OOD_THRESHOLDS_PATH}")
    else:
        log.info(f"[OOD] Umbrales (dry-run, no guardados): {ood_thresholds}")

    # ── Resumen final ─────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info(f"  ENTRENAMIENTO FINALIZADO — Expert 5 (Res-U-Net + FiLM)")
    log.info(f"  Mejor val_loss: {best_val_loss:.6f}")
    log.info(
        f"  OOD thresholds: "
        f"p{EXPERT6_OOD_THRESHOLD_PERCENTILE_LEVE}="
        f"{ood_thresholds['threshold_p50']:.6f} | "
        f"p{EXPERT6_OOD_THRESHOLD_PERCENTILE_OOD}="
        f"{ood_thresholds['threshold_p99']:.6f}"
    )
    if not dry_run:
        log.info(f"  Best checkpoint: {_BEST_CKPT_PATH}")
        log.info(f"  Last checkpoint: {_LAST_CKPT_PATH}")
        log.info(f"  Métricas CSV: {_METRICS_CSV_PATH}")
        log.info(f"  OOD thresholds: {_OOD_THRESHOLDS_PATH}")
    log.info(f"{'=' * 70}")

    if dry_run:
        log.info(
            "\n[DRY-RUN] Pipeline verificado exitosamente. "
            "Ejecuta sin --dry-run para entrenar."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Entrenamiento Expert 5 — Res-U-Net Autoencoder con FiLM "
            "Domain Conditioning (Fase 3)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Ejecuta 2 batches × 2 épocas para verificar el pipeline "
            "sin entrenar el modelo completo"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(f"Número de épocas (sobrescribe config). Default: {EXPERT6_MAX_EPOCHS}"),
    )
    args = parser.parse_args()
    train(dry_run=args.dry_run, epochs_override=args.epochs)

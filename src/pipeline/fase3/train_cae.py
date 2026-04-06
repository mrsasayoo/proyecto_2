"""
Script de entrenamiento para Expert 5 — CAE multimodal (Fase 3).

Pipeline completo:
    1. Carga hiperparámetros desde expert5_cae_config.py (fuente de verdad)
    2. Construye modelo ConvAutoEncoder 2D desde cero (weights=None)
    3. Entrena con MSE + lambda*L1 en FP32 (sin autocast ni GradScaler)
    4. Early stopping por val_mse (patience=15)
    5. Guarda checkpoints y log de métricas

Uso:
    # Dry-run: verifica el pipeline sin entrenar
    python src/pipeline/fase3/train_cae.py --dry-run

    # Entrenamiento completo
    python src/pipeline/fase3/train_cae.py

Dependencias:
    - src/pipeline/fase3/models/expert5_cae.py: ConvAutoEncoder
    - src/pipeline/fase3/dataloader_cae.py: get_cae_dataloaders
    - src/pipeline/fase3/expert5_cae_config.py: hiperparámetros
    - datasets/cae_splits.csv: split multi-modal
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Configurar paths ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from fase3.models.expert5_cae import ConvAutoEncoder
from fase3.dataloader_cae import get_cae_dataloaders
from fase3.expert5_cae_config import (
    EXPERT5_LR,
    EXPERT5_WEIGHT_DECAY,
    EXPERT5_BATCH_SIZE,
    EXPERT5_ACCUMULATION_STEPS,
    EXPERT5_FP16,
    EXPERT5_MAX_EPOCHS,
    EXPERT5_EARLY_STOPPING_PATIENCE,
    EXPERT5_EARLY_STOPPING_MONITOR,
    EXPERT5_LATENT_DIM,
    EXPERT5_IMG_SIZE,
    EXPERT5_IN_CHANNELS,
    EXPERT5_LOSS_LAMBDA_L1,
    EXPERT5_SCHEDULER_FACTOR,
    EXPERT5_SCHEDULER_PATIENCE,
    EXPERT5_CONFIG_SUMMARY,
)

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cae_train")

# ── Rutas de salida ────────────────────────────────────────────────────
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "expert_05_cae" / "cae_best.pt"
_TRAINING_LOG_PATH = _CHECKPOINT_DIR / "expert_05_cae" / "cae_training_log.json"

# ── CSV path ───────────────────────────────────────────────────────────
_CAE_CSV = _PROJECT_ROOT / "datasets" / "cae_splits.csv"

# ── Constantes de entrenamiento ────────────────────────────────────────
_SEED = 42
_MIN_DELTA = 0.0001  # Mejora mínima para considerar progreso (MSE es escala pequeña)


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
    """
    Early stopping por val_mse con patience configurable.

    Detiene el entrenamiento si val_mse no mejora (delta > min_delta)
    durante 'patience' épocas consecutivas.
    """

    def __init__(self, patience: int, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_mse: float) -> bool:
        """
        Evalúa si el entrenamiento debe detenerse.

        Args:
            val_mse: MSE de validación de la época actual.

        Returns:
            True si se debe detener, False si se debe continuar.
        """
        if val_mse < self.best_val - self.min_delta:
            self.best_val = val_mse
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False


def train_one_epoch(
    model: nn.Module,
    loader,
    mse_criterion: nn.Module,
    l1_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_l1: float,
    dry_run: bool = False,
) -> float:
    """
    Ejecuta una época de entrenamiento del CAE en FP32.

    Loss = MSE(recon, input) + lambda_l1 * L1(recon, input)

    Args:
        model: ConvAutoEncoder
        loader: DataLoader de train
        mse_criterion: nn.MSELoss
        l1_criterion: nn.L1Loss
        optimizer: Adam
        device: dispositivo (cuda/cpu)
        lambda_l1: peso del término L1
        dry_run: si True, ejecuta solo 2 batches

    Returns:
        Loss promedio de la época
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (imgs, _paths) in enumerate(loader):
        if dry_run and batch_idx >= 2:
            break

        imgs = imgs.to(device, non_blocking=True)

        # ── Forward FP32 (sin autocast) ────────────────────────────
        recon, z = model(imgs)
        loss = mse_criterion(recon, imgs) + lambda_l1 * l1_criterion(recon, imgs)

        # ── Backward ───────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if dry_run:
            log.info(
                f"  [Train batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"recon={list(recon.shape)} | "
                f"latent={list(z.shape)} | "
                f"loss={loss.item():.6f}"
            )

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    mse_criterion: nn.Module,
    l1_criterion: nn.Module,
    device: torch.device,
    lambda_l1: float,
    dry_run: bool = False,
) -> dict:
    """
    Ejecuta validación del CAE.

    Métricas:
        - val_loss: MSE + lambda*L1 (loss total)
        - val_mse: MSE puro (métrica principal para early stopping)

    Returns:
        dict con keys: val_loss, val_mse
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0

    for batch_idx, (imgs, _paths) in enumerate(loader):
        if dry_run and batch_idx >= 2:
            break

        imgs = imgs.to(device, non_blocking=True)

        recon, z = model(imgs)
        mse_val = mse_criterion(recon, imgs)
        loss = mse_val + lambda_l1 * l1_criterion(recon, imgs)

        total_loss += loss.item()
        total_mse += mse_val.item()
        n_batches += 1

        if dry_run:
            log.info(
                f"  [Val batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"recon={list(recon.shape)} | "
                f"mse={mse_val.item():.6f} | "
                f"loss={loss.item():.6f}"
            )

    avg_loss = total_loss / max(n_batches, 1)
    avg_mse = total_mse / max(n_batches, 1)

    return {
        "val_loss": avg_loss,
        "val_mse": avg_mse,
    }


def train(dry_run: bool = False) -> None:
    """
    Función principal de entrenamiento del Expert 5 (CAE).

    Args:
        dry_run: si True, ejecuta 2 batches de train y 2 de val para verificar
                 el pipeline sin entrenar el modelo completo.
    """
    set_seed(_SEED)

    # ── Dispositivo ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[CAE] Dispositivo: {device}")
    if device.type == "cpu":
        log.warning(
            "[CAE] Entrenando en CPU — será lento. Se recomienda GPU con >= 4 GB VRAM."
        )

    # ── Configuración ──────────────────────────────────────────────
    log.info(f"[CAE] Config: {EXPERT5_CONFIG_SUMMARY}")
    if dry_run:
        log.info("[CAE] === MODO DRY-RUN === (2 batches train + 2 batches val)")

    # FP32 obligatorio para el CAE (no FP16)
    if EXPERT5_FP16:
        log.warning(
            "[CAE] EXPERT5_FP16=True en config, pero se ignora. "
            "El CAE siempre usa FP32 para precisión en MSE."
        )

    # ── Modelo ─────────────────────────────────────────────────────
    model = ConvAutoEncoder(
        in_channels=EXPERT5_IN_CHANNELS,
        latent_dim=EXPERT5_LATENT_DIM,
        img_size=EXPERT5_IMG_SIZE,
    ).to(device)

    n_params = model.count_parameters()
    log.info(
        f"[CAE] Modelo ConvAutoEncoder creado: {n_params:,} parámetros entrenables"
    )
    _log_vram("post-model")

    # ── DataLoaders ────────────────────────────────────────────────
    num_workers = 0 if dry_run else 4
    train_loader, val_loader = get_cae_dataloaders(
        csv_path=str(_CAE_CSV),
        project_root=str(_PROJECT_ROOT),
        batch_size=EXPERT5_BATCH_SIZE,
        num_workers=num_workers,
        img_size=EXPERT5_IMG_SIZE,
    )

    # ── Loss ───────────────────────────────────────────────────────
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    log.info(f"[CAE] Loss: MSE + {EXPERT5_LOSS_LAMBDA_L1} * L1")

    # ── Optimizer ──────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=EXPERT5_LR,
        weight_decay=EXPERT5_WEIGHT_DECAY,
    )
    log.info(f"[CAE] Optimizer: Adam(lr={EXPERT5_LR}, wd={EXPERT5_WEIGHT_DECAY})")

    # ── Scheduler ──────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=EXPERT5_SCHEDULER_FACTOR,
        patience=EXPERT5_SCHEDULER_PATIENCE,
    )
    log.info(
        f"[CAE] Scheduler: ReduceLROnPlateau("
        f"factor={EXPERT5_SCHEDULER_FACTOR}, patience={EXPERT5_SCHEDULER_PATIENCE})"
    )

    # ── Early stopping ─────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=EXPERT5_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    log.info(
        f"[CAE] EarlyStopping: monitor={EXPERT5_EARLY_STOPPING_MONITOR}, "
        f"patience={EXPERT5_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
    )

    # ── Directorio de checkpoints ──────────────────────────────────
    _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────
    best_val_mse = float("inf")
    training_log = []
    max_epochs = 1 if dry_run else EXPERT5_MAX_EPOCHS

    log.info(f"\n{'=' * 70}")
    log.info(f"  INICIO DE ENTRENAMIENTO — Expert 5 (CAE multimodal)")
    log.info(
        f"  Épocas máx: {max_epochs} | Batch: {EXPERT5_BATCH_SIZE} | "
        f"FP32 | Latent dim: {EXPERT5_LATENT_DIM}"
    )
    log.info(f"{'=' * 70}\n")

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # ── Train ──────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            mse_criterion=mse_criterion,
            l1_criterion=l1_criterion,
            optimizer=optimizer,
            device=device,
            lambda_l1=EXPERT5_LOSS_LAMBDA_L1,
            dry_run=dry_run,
        )

        # ── Validation ─────────────────────────────────────────────
        val_results = validate(
            model=model,
            loader=val_loader,
            mse_criterion=mse_criterion,
            l1_criterion=l1_criterion,
            device=device,
            lambda_l1=EXPERT5_LOSS_LAMBDA_L1,
            dry_run=dry_run,
        )

        # ── Scheduler step (ReduceLROnPlateau usa val_mse) ─────────
        val_mse = val_results["val_mse"]
        val_loss = val_results["val_loss"]
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Log de época ───────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        is_best = val_mse < best_val_mse - _MIN_DELTA

        log.info(
            f"[Epoch {epoch + 1:3d}/{max_epochs}] "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val_mse={val_mse:.6f} | "
            f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
            f"{' ★ BEST' if is_best else ''}"
        )
        _log_vram(f"epoch-{epoch + 1}")

        # ── Guardar log de métricas ────────────────────────────────
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mse": val_mse,
            "lr": current_lr,
            "epoch_time_s": round(epoch_time, 1),
            "is_best": is_best,
        }
        training_log.append(epoch_log)

        # ── Guardar mejor checkpoint ───────────────────────────────
        if is_best:
            best_val_mse = val_mse
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_mse": val_mse,
                "val_loss": val_loss,
                "config": {
                    "lr": EXPERT5_LR,
                    "weight_decay": EXPERT5_WEIGHT_DECAY,
                    "batch_size": EXPERT5_BATCH_SIZE,
                    "latent_dim": EXPERT5_LATENT_DIM,
                    "img_size": EXPERT5_IMG_SIZE,
                    "in_channels": EXPERT5_IN_CHANNELS,
                    "lambda_l1": EXPERT5_LOSS_LAMBDA_L1,
                    "n_params": n_params,
                    "seed": _SEED,
                },
            }
            if not dry_run:
                torch.save(checkpoint, _CHECKPOINT_PATH)
                log.info(f"  → Checkpoint guardado: {_CHECKPOINT_PATH}")

        # ── Guardar training log ───────────────────────────────────
        if not dry_run:
            with open(_TRAINING_LOG_PATH, "w") as f:
                json.dump(training_log, f, indent=2)

        # ── Early stopping ─────────────────────────────────────────
        if not dry_run:
            if early_stopping.step(val_mse):
                log.info(
                    f"\n[EarlyStopping] Detenido en época {epoch + 1}. "
                    f"val_mse no mejoró en {EXPERT5_EARLY_STOPPING_PATIENCE} épocas. "
                    f"Mejor val_mse: {best_val_mse:.6f}"
                )
                break

    # ── Resumen final ──────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info(f"  ENTRENAMIENTO FINALIZADO — Expert 5 (CAE multimodal)")
    log.info(f"  Mejor val_mse: {best_val_mse:.6f}")
    if training_log:
        best_epoch = min(training_log, key=lambda x: x["val_mse"])
        log.info(
            f"  Mejor época: {best_epoch['epoch']} | "
            f"val_mse: {best_epoch['val_mse']:.6f} | "
            f"val_loss: {best_epoch['val_loss']:.6f}"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento Expert 5 — CAE multimodal (Fase 3)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta 2 batches de train y 2 de val para verificar el pipeline sin entrenar",
    )
    args = parser.parse_args()
    train(dry_run=args.dry_run)

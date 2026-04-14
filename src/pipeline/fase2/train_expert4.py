"""
Script de entrenamiento para Expert 4 — ResNet 3D (R3D-18) sobre Páncreas CT.

Pipeline completo:
    1. Carga hiperparámetros desde expert4_config.py (fuente de verdad)
    2. Construye modelo ResNet 3D adaptado (1 canal, 2 clases)
    3. Entrena con FocalLoss(gamma=2, alpha=0.75) + FP16 + gradient accumulation
    4. k-fold CV (k=5) con fold seleccionable via --fold
    5. Early stopping por val_loss (patience=15)
    6. Guarda checkpoints y log de métricas

Uso:
    # Dry-run: verifica el pipeline sin entrenar (fold=0)
    python src/pipeline/fase2/train_expert4.py --dry-run

    # Entrenamiento completo del fold 0
    python src/pipeline/fase2/train_expert4.py --fold 0

    # Entrenamiento de todos los folds
    for fold in 0 1 2 3 4; do
        python src/pipeline/fase2/train_expert4.py --fold $fold
    done

Dependencias:
    - src/pipeline/fase2/models/expert4_resnet3d.py: ExpertPancreasSwin3D
    - src/pipeline/fase2/dataloader_expert4.py: build_dataloaders_expert4
    - src/pipeline/fase2/expert4_config.py: hiperparámetros
    - src/pipeline/losses.py: FocalLoss
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
from torch.amp import GradScaler
from sklearn.metrics import f1_score, roc_auc_score

# ── Configurar paths ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from fase2.models.expert4_resnet3d import ExpertPancreasSwin3D
from fase2.dataloader_expert4 import build_dataloaders_expert4
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

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("expert4_train")

# ── Rutas de salida ────────────────────────────────────────────────────
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "expert_04_swin3d_tiny" / "expert4_best.pt"
_TRAINING_LOG_PATH = (
    _CHECKPOINT_DIR / "expert_04_swin3d_tiny" / "expert4_training_log.json"
)

# ── Constantes de entrenamiento ────────────────────────────────────────
_SEED = 42
_MIN_DELTA = 0.001  # Mejora mínima para considerar progreso en early stopping


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


def _enable_gradient_checkpointing(model: ExpertPancreasSwin3D) -> bool:
    """
    Habilita gradient checkpointing en los residual blocks del R3D-18.

    ExpertPancreasResNet3D expone layer1..layer4, cada uno un nn.Sequential
    de BasicBlock (Conv3D 3x3x3). Se envuelve el forward de cada layer con
    torch.utils.checkpoint.checkpoint para reducir el consumo de VRAM a
    costa de re-computar activaciones durante el backward pass.

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

        log.info(
            f"[GradCheckpoint] Gradient checkpointing HABILITADO en "
            f"{applied}/{len(layer_names)} layers de R3D-18"
        )
        return applied > 0
    except Exception as e:
        log.warning(f"[GradCheckpoint] No se pudo habilitar: {e}")
        return False


class EarlyStopping:
    """
    Early stopping por val_loss con patience configurable.

    Detiene el entrenamiento si val_loss no mejora (delta > min_delta)
    durante 'patience' épocas consecutivas.
    """

    def __init__(self, patience: int, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Evalúa si el entrenamiento debe detenerse.

        Returns:
            True si se debe detener, False si se debe continuar.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
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
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int,
    use_fp16: bool,
    dry_run: bool = False,
) -> float:
    """
    Ejecuta una época de entrenamiento con gradient accumulation y FP16.

    Returns:
        Loss promedio de la época.
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

        # ── Forward con autocast FP16 ──────────────────────────────
        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(volumes)  # [B, 2]
            # FocalLoss espera logits [B] y targets [B] float
            # Usamos logits[:, 1] (logit de clase positiva = PDAC+)
            loss = criterion(logits[:, 1], labels)
            # Normalizar loss por accumulation_steps
            loss = loss / accumulation_steps

        # ── Backward con GradScaler ────────────────────────────────
        scaler.scale(loss).backward()

        # ── Optimizer step cada accumulation_steps batches ─────────
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps  # Revertir normalización
        n_batches += 1

        if dry_run:
            log.info(
                f"  [Train batch {batch_idx}] "
                f"volume={list(volumes.shape)} | "
                f"logits={list(logits.shape)} | "
                f"loss={loss.item() * accumulation_steps:.4f}"
            )

    # Flush de gradientes residuales si el último bloque no completó accumulation_steps
    if n_batches % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_fp16: bool,
    dry_run: bool = False,
) -> dict:
    """
    Ejecuta validación y calcula métricas.

    Returns:
        dict con keys: val_loss, val_auc, val_f1
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_labels = []
    all_probs = []
    all_preds = []

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

        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

        if dry_run:
            log.info(
                f"  [Val batch {batch_idx}] "
                f"volume={list(volumes.shape)} | "
                f"logits={list(logits.shape)} | "
                f"loss={loss.item():.4f}"
            )

    avg_loss = total_loss / max(n_batches, 1)

    # ── Métricas ───────────────────────────────────────────────────
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)

    # AUC-ROC (métrica primaria — objetivo > 0.85)
    if len(np.unique(labels_arr)) >= 2:
        try:
            val_auc = roc_auc_score(labels_arr, probs_arr)
        except ValueError:
            val_auc = 0.0
    else:
        val_auc = 0.0
        log.warning("[Val] Solo una clase presente → AUC=0.0")

    # F1 Macro
    if len(np.unique(labels_arr)) >= 2:
        val_f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    else:
        val_f1 = 0.0
        log.warning("[Val] Solo una clase presente → F1=0.0")

    return {
        "val_loss": avg_loss,
        "val_auc": val_auc,
        "val_f1": val_f1,
    }


def train(dry_run: bool = False, fold: int = 0) -> None:
    """
    Función principal de entrenamiento del Expert 4.

    Args:
        dry_run: si True, ejecuta 2 batches de train y 2 de val para verificar
                 el pipeline sin entrenar el modelo completo.
        fold: índice del fold para k-fold CV (0 a EXPERT4_NUM_FOLDS-1).
    """
    set_seed(_SEED)

    # ── Dispositivo ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Expert4] Dispositivo: {device}")
    if device.type == "cpu":
        log.warning(
            "[Expert4] Entrenando en CPU — será muy lento. "
            "Se recomienda GPU con >= 16 GB VRAM."
        )

    # ── Configuración ──────────────────────────────────────────────
    log.info(f"[Expert4] Config: {EXPERT4_CONFIG_SUMMARY}")
    log.info(f"[Expert4] Fold: {fold}/{EXPERT4_NUM_FOLDS - 1}")
    if dry_run:
        log.info("[Expert4] === MODO DRY-RUN === (2 batches train + 2 batches val)")

    use_fp16 = EXPERT4_FP16 and device.type == "cuda"
    if not use_fp16 and EXPERT4_FP16:
        log.info("[Expert4] FP16 desactivado (no hay GPU). Usando FP32 en CPU.")

    # ── Modelo ─────────────────────────────────────────────────────
    model = ExpertPancreasSwin3D(
        in_channels=1,
        num_classes=EXPERT4_NUM_CLASSES,
    ).to(device)

    n_params = model.count_parameters()
    log.info(
        f"[Expert4] Modelo ResNet3D (R3D-18) creado: {n_params:,} parámetros entrenables"
    )
    _log_vram("post-model")

    # ── Gradient checkpointing (OBLIGATORIO) ───────────────────────
    if device.type == "cuda":
        _enable_gradient_checkpointing(model)

    # ── DataLoaders ────────────────────────────────────────────────
    num_workers = 0 if dry_run else 2
    train_loader, val_loader = build_dataloaders_expert4(
        fold=fold,
        batch_size=EXPERT4_BATCH_SIZE,
        num_workers=num_workers,
    )

    # ── Loss ───────────────────────────────────────────────────────
    criterion = FocalLoss(gamma=EXPERT4_FOCAL_GAMMA, alpha=EXPERT4_FOCAL_ALPHA)
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
    log.info(f"[Expert4] Optimizer: AdamW(lr={EXPERT4_LR}, wd={EXPERT4_WEIGHT_DECAY})")

    # ── Scheduler: CosineAnnealingWarmRestarts ─────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=EXPERT4_SCHEDULER_T0,
        T_mult=EXPERT4_SCHEDULER_T_MULT,
        eta_min=1e-6,
    )
    log.info(
        f"[Expert4] Scheduler: CosineAnnealingWarmRestarts("
        f"T_0={EXPERT4_SCHEDULER_T0}, T_mult={EXPERT4_SCHEDULER_T_MULT}, eta_min=1e-6)"
    )

    # ── GradScaler para FP16 ───────────────────────────────────────
    scaler = GradScaler(device=device.type, enabled=use_fp16)

    # ── Early stopping ─────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=EXPERT4_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    log.info(
        f"[Expert4] EarlyStopping: monitor={EXPERT4_EARLY_STOPPING_MONITOR}, "
        f"patience={EXPERT4_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
    )

    # ── Paths de checkpoint con fold ───────────────────────────────
    checkpoint_path = (
        _CHECKPOINT_PATH.parent / f"expert4_best_fold{fold}.pt"
        if fold > 0
        else _CHECKPOINT_PATH
    )
    log_path = (
        _TRAINING_LOG_PATH.parent / f"expert4_training_log_fold{fold}.json"
        if fold > 0
        else _TRAINING_LOG_PATH
    )

    # ── Training loop ──────────────────────────────────────────────
    best_val_loss = float("inf")
    training_log = []
    max_epochs = 1 if dry_run else EXPERT4_MAX_EPOCHS

    log.info(f"\n{'=' * 70}")
    log.info(f"  INICIO DE ENTRENAMIENTO — Expert 4 (ResNet3D R3D-18 / Páncreas)")
    log.info(f"  Fold: {fold}/{EXPERT4_NUM_FOLDS - 1}")
    log.info(
        f"  Épocas máx: {max_epochs} | Batch efectivo: "
        f"{EXPERT4_BATCH_SIZE}x{EXPERT4_ACCUMULATION_STEPS}="
        f"{EXPERT4_BATCH_SIZE * EXPERT4_ACCUMULATION_STEPS}"
    )
    log.info(f"  FP16: {use_fp16} | Accumulation: {EXPERT4_ACCUMULATION_STEPS}")
    log.info(f"{'=' * 70}\n")

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # ── Train ──────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            accumulation_steps=EXPERT4_ACCUMULATION_STEPS,
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

        # ── Log de época ───────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        val_loss = val_results["val_loss"]
        val_auc = val_results["val_auc"]
        val_f1 = val_results["val_f1"]

        is_best = val_loss < best_val_loss - _MIN_DELTA

        log.info(
            f"[Epoch {epoch + 1:3d}/{max_epochs}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f} | val_f1={val_f1:.4f} | "
            f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
            f"{' * BEST' if is_best else ''}"
        )
        _log_vram(f"epoch-{epoch + 1}")

        # ── Guardar log de métricas ────────────────────────────────
        epoch_log = {
            "epoch": epoch + 1,
            "fold": fold,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_f1": val_f1,
            "lr": current_lr,
            "epoch_time_s": round(epoch_time, 1),
            "is_best": is_best,
        }
        training_log.append(epoch_log)

        # ── Guardar mejor checkpoint ───────────────────────────────
        if is_best:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "fold": fold,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_f1": val_auc,  # Consistencia con otros expertos: key "val_f1" contiene AUC
                "val_auc": val_auc,
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
                },
            }
            if not dry_run:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
                log.info(f"  -> Checkpoint guardado: {checkpoint_path}")

        # ── Guardar training log ───────────────────────────────────
        if not dry_run:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(training_log, f, indent=2)

        # ── Early stopping ─────────────────────────────────────────
        if not dry_run:
            if early_stopping.step(val_loss):
                log.info(
                    f"\n[EarlyStopping] Detenido en época {epoch + 1}. "
                    f"val_loss no mejoró en {EXPERT4_EARLY_STOPPING_PATIENCE} épocas. "
                    f"Mejor val_loss: {best_val_loss:.4f}"
                )
                break

    # ── Resumen final ──────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info(f"  ENTRENAMIENTO FINALIZADO — Expert 4 (ResNet3D R3D-18 / Páncreas)")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento Expert 4 — ResNet3D R3D-18 / Páncreas CT"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta 2 batches de train y 2 de val para verificar el pipeline",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        choices=range(EXPERT4_NUM_FOLDS),
        help=f"Fold para k-fold CV (0 a {EXPERT4_NUM_FOLDS - 1}). Default: 0.",
    )
    args = parser.parse_args()
    train(dry_run=args.dry_run, fold=args.fold)

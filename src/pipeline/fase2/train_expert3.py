"""
Script de entrenamiento para Expert 3 — DenseNet 3D sobre LUNA16.

Pipeline completo:
    1. Carga hiperparámetros desde expert3_config.py (fuente de verdad)
    2. Construye modelo DenseNet 3D adaptado (1 canal, 2 clases)
    3. Entrena con FocalLoss(gamma=2, alpha=0.85) + FP16 + gradient accumulation
    4. Early stopping por val_loss (patience=20)
    5. Guarda checkpoints y log de métricas

Uso:
    # Dry-run: verifica el pipeline sin entrenar
    python src/pipeline/fase2/train_expert3.py --dry-run

    # Entrenamiento completo
    python src/pipeline/fase2/train_expert3.py

Dependencias:
    - src/pipeline/fase2/models/expert3_densenet3d.py: Expert3MC318 (alias de Expert3DenseNet3D)
    - src/pipeline/fase2/dataloader_expert3.py: build_dataloaders_expert3
    - src/pipeline/fase2/expert3_config.py: hiperparámetros
    - src/pipeline/losses.py: FocalLoss
"""

import sys
import os
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler  # torch >= 2.10
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# ── Configurar paths ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from fase2.models.expert3_densenet3d import Expert3MC318
from fase2.dataloader_expert3 import build_dataloaders_expert3
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

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("expert3_train")

# ── Rutas de salida ────────────────────────────────────────────────────
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "expert_03_densenet3d" / "expert3_best.pt"
_TRAINING_LOG_PATH = (
    _CHECKPOINT_DIR / "expert_03_densenet3d" / "expert3_training_log.json"
)

# ── Constantes de entrenamiento ────────────────────────────────────────
_SEED = 42
_MIN_DELTA = 0.001  # Mejora mínima para considerar como progreso en early stopping


def set_seed(seed: int = _SEED) -> None:
    """Fija todas las semillas para reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Determinismo en cuDNN (puede reducir velocidad ~10%)
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


def _enable_gradient_checkpointing(model: Expert3MC318) -> bool:
    """
    Intenta habilitar gradient checkpointing en el modelo DenseNet 3D.

    DenseNet 3D no tiene gradient_checkpointing_enable() nativo.
    Se implementa manualmente usando torch.utils.checkpoint en los
    dense blocks del modelo (model.dense_blocks: nn.ModuleList de _DenseBlock).

    Cada _DenseBlock contiene N _DenseLayers con conectividad densa.
    El checkpointing se aplica al forward completo de cada bloque,
    liberando activaciones intermedias y recalculándolas en el backward.

    Returns:
        True si se habilitó, False si no fue posible.
    """
    try:
        from torch.utils.checkpoint import checkpoint

        # Verificar que el modelo tiene dense_blocks (estructura DenseNet 3D)
        if not hasattr(model, "dense_blocks") or not model.dense_blocks:
            log.warning(
                "[GradCheckpoint] Modelo sin atributo 'dense_blocks'. "
                "Gradient checkpointing no aplicable."
            )
            return False

        # Aplicar checkpointing a cada _DenseBlock
        checkpointed_count = 0
        for idx, block in enumerate(model.dense_blocks):
            # Creamos un forward con checkpointing por bloque
            def make_checkpointed_forward(original_fwd):
                def checkpointed_forward(x):
                    return checkpoint(original_fwd, x, use_reentrant=False)

                return checkpointed_forward

            block.forward = make_checkpointed_forward(block.forward)
            checkpointed_count += 1

        log.info(
            f"[GradCheckpoint] Gradient checkpointing HABILITADO en "
            f"{checkpointed_count} dense blocks (dense_blocks[0..{checkpointed_count - 1}])"
        )
        return True
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

        Args:
            val_loss: loss de validación de la época actual.

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

    Args:
        model: modelo a entrenar
        loader: DataLoader de train
        criterion: función de pérdida (FocalLoss)
        optimizer: optimizador (AdamW)
        scaler: GradScaler para FP16
        device: dispositivo (cuda/cpu)
        accumulation_steps: pasos de acumulación de gradientes
        use_fp16: usar mixed precision
        dry_run: si True, ejecuta solo 2 batches

    Returns:
        Loss promedio de la época
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

        # ── Forward con autocast FP16 ──────────────────────────────
        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(volumes)  # [B, 2]
            # FocalLoss espera logits [B] y targets [B] float
            # Usamos logits[:, 1] (logit de clase positiva)
            # Aplicar label smoothing: {0,1} → {0.025, 0.975}
            labels_smooth = (
                labels * (1 - EXPERT3_LABEL_SMOOTHING) + EXPERT3_LABEL_SMOOTHING / 2.0
            )
            loss = criterion(logits[:, 1], labels_smooth)
            # Normalizar loss por accumulation_steps
            loss = loss / accumulation_steps

        # ── Backward con GradScaler ────────────────────────────────
        scaler.scale(loss).backward()

        # ── Optimizer step cada accumulation_steps batches ─────────
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps  # Revertir la normalización
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
        dict con keys: val_loss, val_f1_macro, val_auc, confusion_matrix
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_labels = []
    all_probs = []
    all_preds = []

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

    # F1 Macro
    if len(np.unique(labels_arr)) >= 2:
        f1_macro = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    else:
        f1_macro = 0.0
        log.warning("[Val] Solo una clase presente en labels → F1=0.0")

    # AUC-ROC
    if len(np.unique(labels_arr)) >= 2:
        try:
            auc = roc_auc_score(labels_arr, probs_arr)
        except ValueError:
            auc = 0.0
    else:
        auc = 0.0
        log.warning("[Val] Solo una clase presente → AUC=0.0")

    # Confusion matrix
    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])

    return {
        "val_loss": avg_loss,
        "val_f1_macro": f1_macro,
        "val_auc": auc,
        "confusion_matrix": cm.tolist(),
    }


def train(dry_run: bool = False) -> None:
    """
    Función principal de entrenamiento del Expert 3.

    Args:
        dry_run: si True, ejecuta 2 batches de train y 1 de val para verificar
                 el pipeline sin entrenar el modelo completo.
    """
    set_seed(_SEED)

    # ── Dispositivo ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Expert3] Dispositivo: {device}")
    if device.type == "cpu":
        log.warning(
            "[Expert3] ⚠ Entrenando en CPU — será muy lento. "
            "Se recomienda GPU con >= 12 GB VRAM."
        )

    # ── Configuración ──────────────────────────────────────────────
    log.info(f"[Expert3] Config: {EXPERT3_CONFIG_SUMMARY}")
    if dry_run:
        log.info("[Expert3] === MODO DRY-RUN === (2 batches train + 1 batch val)")

    use_fp16 = EXPERT3_FP16 and device.type == "cuda"
    if not use_fp16 and EXPERT3_FP16:
        log.info("[Expert3] FP16 desactivado (no hay GPU). Usando FP32 en CPU.")

    # ── Modelo ─────────────────────────────────────────────────────
    model = Expert3MC318(
        spatial_dropout_p=EXPERT3_SPATIAL_DROPOUT_3D,
        fc_dropout_p=EXPERT3_DROPOUT_FC,
        num_classes=2,
    ).to(device)

    n_params = model.count_parameters()
    log.info(f"[Expert3] Modelo DenseNet3D creado: {n_params:,} parámetros entrenables")
    _log_vram("post-model")

    # ── Gradient checkpointing ─────────────────────────────────────
    if device.type == "cuda":
        _enable_gradient_checkpointing(model)

    # ── DataLoaders ────────────────────────────────────────────────
    # Usar num_workers=0 en dry-run para evitar overhead de multiprocessing
    num_workers = 0 if dry_run else 4
    train_loader, val_loader, _test_loader = build_dataloaders_expert3(
        batch_size=EXPERT3_BATCH_SIZE,
        num_workers=num_workers,
    )

    # ── Loss ───────────────────────────────────────────────────────
    criterion = FocalLoss(gamma=EXPERT3_FOCAL_GAMMA, alpha=EXPERT3_FOCAL_ALPHA)
    log.info(
        f"[Expert3] Loss: FocalLoss(gamma={EXPERT3_FOCAL_GAMMA}, "
        f"alpha={EXPERT3_FOCAL_ALPHA})"
    )

    # ── Optimizer ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=EXPERT3_LR,
        weight_decay=EXPERT3_WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    log.info(f"[Expert3] Optimizer: AdamW(lr={EXPERT3_LR}, wd={EXPERT3_WEIGHT_DECAY})")

    # ── Scheduler ──────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,
        T_mult=2,
        eta_min=1e-6,
    )
    log.info(
        "[Expert3] Scheduler: CosineAnnealingWarmRestarts(T_0=15, T_mult=2, eta_min=1e-6)"
    )

    # ── GradScaler para FP16 ───────────────────────────────────────
    scaler = GradScaler(device=device.type, enabled=use_fp16)

    # ── Early stopping ─────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=EXPERT3_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    log.info(
        f"[Expert3] EarlyStopping: monitor={EXPERT3_EARLY_STOPPING_MONITOR}, "
        f"patience={EXPERT3_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
    )

    # ── Directorio de checkpoints ──────────────────────────────────
    _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────
    best_val_loss = float("inf")
    training_log = []
    max_epochs = 1 if dry_run else EXPERT3_MAX_EPOCHS

    log.info(f"\n{'=' * 70}")
    log.info(f"  INICIO DE ENTRENAMIENTO — Expert 3 (DenseNet3D / LUNA16)")
    log.info(
        f"  Épocas máx: {max_epochs} | Batch efectivo: "
        f"{EXPERT3_BATCH_SIZE}×{EXPERT3_ACCUMULATION_STEPS}="
        f"{EXPERT3_BATCH_SIZE * EXPERT3_ACCUMULATION_STEPS}"
    )
    log.info(f"  FP16: {use_fp16} | Accumulation: {EXPERT3_ACCUMULATION_STEPS}")
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
            accumulation_steps=EXPERT3_ACCUMULATION_STEPS,
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
        val_f1 = val_results["val_f1_macro"]
        val_auc = val_results["val_auc"]
        cm = val_results["confusion_matrix"]

        is_best = val_loss < best_val_loss - _MIN_DELTA

        log.info(
            f"[Epoch {epoch + 1:3d}/{max_epochs}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_f1_macro={val_f1:.4f} | val_auc={val_auc:.4f} | "
            f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
            f"{' ★ BEST' if is_best else ''}"
        )
        log.info(
            f"         Confusion Matrix: "
            f"TN={cm[0][0]:>5} FP={cm[0][1]:>5} | "
            f"FN={cm[1][0]:>5} TP={cm[1][1]:>5}"
        )
        _log_vram(f"epoch-{epoch + 1}")

        # ── Guardar log de métricas ────────────────────────────────
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1_macro": val_f1,
            "val_auc": val_auc,
            "confusion_matrix": cm,
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
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_f1": val_f1,
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
            if early_stopping.step(val_loss):
                log.info(
                    f"\n[EarlyStopping] Detenido en época {epoch + 1}. "
                    f"val_loss no mejoró en {EXPERT3_EARLY_STOPPING_PATIENCE} épocas. "
                    f"Mejor val_loss: {best_val_loss:.4f}"
                )
                break

    # ── Resumen final ──────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info(f"  ENTRENAMIENTO FINALIZADO — Expert 3 (DenseNet3D / LUNA16)")
    log.info(f"  Mejor val_loss: {best_val_loss:.4f}")
    if training_log:
        best_epoch = min(training_log, key=lambda x: x["val_loss"])
        log.info(
            f"  Mejor época: {best_epoch['epoch']} | "
            f"F1 Macro: {best_epoch['val_f1_macro']:.4f} | "
            f"AUC: {best_epoch['val_auc']:.4f}"
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
        description="Entrenamiento Expert 3 — DenseNet3D / LUNA16"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta 2 batches de train y 1 de val para verificar el pipeline sin entrenar",
    )
    args = parser.parse_args()
    train(dry_run=args.dry_run)

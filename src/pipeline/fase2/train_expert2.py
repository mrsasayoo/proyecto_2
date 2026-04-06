"""
Script de entrenamiento para Expert 2 — EfficientNet-B3 sobre ISIC 2019.

Pipeline completo:
    1. Carga hiperparámetros desde expert2_config.py (fuente de verdad)
    2. Construye modelo EfficientNet-B3 desde cero (9 clases: 8 train + 1 UNK)
    3. Entrena con CrossEntropyLoss(weight=class_weights) + FP16 + gradient accumulation
    4. Early stopping por val_loss (patience=10)
    5. Guarda checkpoints y log de métricas

Uso:
    # Dry-run: verifica el pipeline sin entrenar
    python src/pipeline/fase2/train_expert2.py --dry-run

    # Entrenamiento completo
    python src/pipeline/fase2/train_expert2.py

Dependencias:
    - src/pipeline/fase2/models/expert2_efficientnet.py: Expert2EfficientNetB3
    - src/pipeline/fase2/dataloader_expert2.py: build_dataloaders_expert2
    - src/pipeline/fase2/expert2_config.py: hiperparámetros
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# ── Configurar paths ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from fase2.models.expert2_efficientnet import Expert2EfficientNetB3
from fase2.dataloader_expert2 import build_dataloaders_expert2
from fase2.expert2_config import (
    EXPERT2_LR,
    EXPERT2_WEIGHT_DECAY,
    EXPERT2_DROPOUT_FC,
    EXPERT2_BATCH_SIZE,
    EXPERT2_ACCUMULATION_STEPS,
    EXPERT2_FP16,
    EXPERT2_MAX_EPOCHS,
    EXPERT2_EARLY_STOPPING_PATIENCE,
    EXPERT2_EARLY_STOPPING_MONITOR,
    EXPERT2_CONFIG_SUMMARY,
)

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("expert2_train")

# ── Rutas de salida ────────────────────────────────────────────────────
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "expert_01_efficientnet_b3" / "expert2_best.pt"
_TRAINING_LOG_PATH = (
    _CHECKPOINT_DIR / "expert_01_efficientnet_b3" / "expert2_training_log.json"
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
        criterion: función de pérdida (CrossEntropyLoss)
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

    for batch_idx, (imgs, labels, _stems) in enumerate(loader):
        if dry_run and batch_idx >= 2:
            break

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        # ── Forward con autocast FP16 ──────────────────────────────
        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(imgs)  # [B, 9]
            loss = criterion(logits, labels)
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
                f"imgs={list(imgs.shape)} | "
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

    Métricas:
        - val_loss: CrossEntropyLoss promedio
        - val_bmca: Balanced Multi-Class Accuracy (sklearn balanced_accuracy_score)
        - val_auc: AUC-ROC macro one-vs-rest (sobre las 8 clases de entrenamiento)

    Returns:
        dict con keys: val_loss, val_bmca, val_auc
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_logits = []
    all_labels = []

    for batch_idx, (imgs, labels, _stems) in enumerate(loader):
        if dry_run and batch_idx >= 1:
            break

        imgs = imgs.to(device, non_blocking=True)
        labels_dev = labels.long().to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(imgs)  # [B, 9]
            loss = criterion(logits, labels_dev)

        total_loss += loss.item()
        n_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

        if dry_run:
            log.info(
                f"  [Val batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"logits={list(logits.shape)} | "
                f"loss={loss.item():.4f}"
            )

    avg_loss = total_loss / max(n_batches, 1)

    # ── Métricas ───────────────────────────────────────────────────
    all_logits = torch.cat(all_logits, dim=0)  # [N, 9]
    all_probs = torch.softmax(all_logits, dim=1).numpy()  # [N, 9]
    all_preds = all_probs.argmax(axis=1)  # [N]
    all_labels_np = torch.cat(all_labels, dim=0).numpy()  # [N]

    # BMCA = Balanced Multi-Class Accuracy (mean per-class accuracy)
    bmca = balanced_accuracy_score(all_labels_np, all_preds)

    # AUC-ROC per class (one-vs-rest, only for the 8 train classes)
    try:
        auc = roc_auc_score(
            all_labels_np,
            all_probs[:, :8],
            multi_class="ovr",
            average="macro",
            labels=list(range(8)),
        )
    except ValueError:
        auc = 0.0
        log.warning("[Val] AUC-ROC no computable (clases insuficientes) → AUC=0.0")

    return {
        "val_loss": avg_loss,
        "val_bmca": bmca,
        "val_auc": auc,
    }


def train(dry_run: bool = False) -> None:
    """
    Función principal de entrenamiento del Expert 2.

    Args:
        dry_run: si True, ejecuta 2 batches de train y 1 de val para verificar
                 el pipeline sin entrenar el modelo completo.
    """
    set_seed(_SEED)

    # ── Dispositivo ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Expert2] Dispositivo: {device}")
    if device.type == "cpu":
        log.warning(
            "[Expert2] ⚠ Entrenando en CPU — será muy lento. "
            "Se recomienda GPU con >= 12 GB VRAM."
        )

    # ── Configuración ──────────────────────────────────────────────
    log.info(f"[Expert2] Config: {EXPERT2_CONFIG_SUMMARY}")
    if dry_run:
        log.info("[Expert2] === MODO DRY-RUN === (2 batches train + 1 batch val)")

    use_fp16 = EXPERT2_FP16 and device.type == "cuda"
    if not use_fp16 and EXPERT2_FP16:
        log.info("[Expert2] FP16 desactivado (no hay GPU). Usando FP32 en CPU.")

    # ── Modelo ─────────────────────────────────────────────────────
    model = Expert2EfficientNetB3(
        fc_dropout_p=EXPERT2_DROPOUT_FC,
        num_classes=9,
    ).to(device)

    n_params = model.count_parameters()
    log.info(
        f"[Expert2] Modelo EfficientNet-B3 creado: {n_params:,} parámetros entrenables"
    )
    _log_vram("post-model")

    # ── DataLoaders ────────────────────────────────────────────────
    # Usar num_workers=0 en dry-run para evitar overhead de multiprocessing
    num_workers = 0 if dry_run else 4
    train_loader, val_loader, _test_loader, class_weights = build_dataloaders_expert2(
        batch_size=EXPERT2_BATCH_SIZE,
        num_workers=num_workers,
    )
    class_weights = class_weights.to(device)

    # ── Loss ───────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    log.info(
        f"[Expert2] Loss: CrossEntropyLoss(weight=class_weights[{class_weights.shape[0]}])"
    )

    # ── Optimizer ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=EXPERT2_LR,
        weight_decay=EXPERT2_WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    log.info(f"[Expert2] Optimizer: AdamW(lr={EXPERT2_LR}, wd={EXPERT2_WEIGHT_DECAY})")

    # ── Scheduler ──────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,
        T_mult=2,
        eta_min=1e-6,
    )
    log.info(
        "[Expert2] Scheduler: CosineAnnealingWarmRestarts(T_0=15, T_mult=2, eta_min=1e-6)"
    )

    # ── GradScaler para FP16 ───────────────────────────────────────
    scaler = GradScaler(device=device.type, enabled=use_fp16)

    # ── Early stopping ─────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=EXPERT2_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    log.info(
        f"[Expert2] EarlyStopping: monitor={EXPERT2_EARLY_STOPPING_MONITOR}, "
        f"patience={EXPERT2_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
    )

    # ── Directorio de checkpoints ──────────────────────────────────
    _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────
    best_val_loss = float("inf")
    training_log = []
    max_epochs = 1 if dry_run else EXPERT2_MAX_EPOCHS

    log.info(f"\n{'=' * 70}")
    log.info(f"  INICIO DE ENTRENAMIENTO — Expert 2 (EfficientNet-B3 / ISIC 2019)")
    log.info(
        f"  Épocas máx: {max_epochs} | Batch efectivo: "
        f"{EXPERT2_BATCH_SIZE}×{EXPERT2_ACCUMULATION_STEPS}="
        f"{EXPERT2_BATCH_SIZE * EXPERT2_ACCUMULATION_STEPS}"
    )
    log.info(f"  FP16: {use_fp16} | Accumulation: {EXPERT2_ACCUMULATION_STEPS}")
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
            accumulation_steps=EXPERT2_ACCUMULATION_STEPS,
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
        val_bmca = val_results["val_bmca"]
        val_auc = val_results["val_auc"]

        is_best = val_loss < best_val_loss - _MIN_DELTA

        log.info(
            f"[Epoch {epoch + 1:3d}/{max_epochs}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_bmca={val_bmca:.4f} | val_auc={val_auc:.4f} | "
            f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
            f"{' ★ BEST' if is_best else ''}"
        )
        _log_vram(f"epoch-{epoch + 1}")

        # ── Guardar log de métricas ────────────────────────────────
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_bmca": val_bmca,
            "val_auc": val_auc,
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
                "val_f1": val_bmca,  # "val_f1" key for consistency with Expert 3
                "val_auc": val_auc,
                "config": {
                    "lr": EXPERT2_LR,
                    "weight_decay": EXPERT2_WEIGHT_DECAY,
                    "dropout_fc": EXPERT2_DROPOUT_FC,
                    "batch_size": EXPERT2_BATCH_SIZE,
                    "accumulation_steps": EXPERT2_ACCUMULATION_STEPS,
                    "fp16": EXPERT2_FP16,
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
                    f"val_loss no mejoró en {EXPERT2_EARLY_STOPPING_PATIENCE} épocas. "
                    f"Mejor val_loss: {best_val_loss:.4f}"
                )
                break

    # ── Resumen final ──────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info(f"  ENTRENAMIENTO FINALIZADO — Expert 2 (EfficientNet-B3 / ISIC 2019)")
    log.info(f"  Mejor val_loss: {best_val_loss:.4f}")
    if training_log:
        best_epoch = min(training_log, key=lambda x: x["val_loss"])
        log.info(
            f"  Mejor época: {best_epoch['epoch']} | "
            f"BMCA: {best_epoch['val_bmca']:.4f} | "
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
        description="Entrenamiento Expert 2 — EfficientNet-B3 / ISIC 2019"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta 2 batches de train y 1 de val para verificar el pipeline sin entrenar",
    )
    args = parser.parse_args()
    train(dry_run=args.dry_run)

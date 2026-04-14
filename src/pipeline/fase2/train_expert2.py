"""
Script de entrenamiento para Expert 2 — ConvNeXt-Small sobre ISIC 2019.

Pipeline trifásico con descongelamiento progresivo:

    Fase 1 (épocas  1-5):   Solo head, backbone congelado.
                             AdamW + CosineAnnealingLR.
    Fase 2 (épocas  6-20):  Fine-tuning diferencial (últimos 2 stages + head).
                             AdamW diferencial + CosineAnnealingWarmRestarts.
    Fase 3 (épocas 21-40):  Full fine-tuning + early stopping.
                             AdamW diferencial + CosineAnnealingWarmRestarts.

Cada fase crea su propio optimizador y scheduler desde cero.
El modelo se guarda solo cuando val_f1_macro mejora (checkpoint expert2_best.pt).

Uso:
    # Dry-run: verifica el pipeline sin entrenar
    python src/pipeline/fase2/train_expert2.py --dry-run

    # Entrenamiento completo
    python src/pipeline/fase2/train_expert2.py

Dependencias:
    - src/pipeline/fase2/models/expert2_convnext_small.py: Expert2ConvNeXtSmall
    - src/pipeline/fase2/dataloader_expert2.py: build_dataloaders_expert2
    - src/pipeline/fase2/expert2_config.py: hiperparámetros por fase
"""

import os
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
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

# ── Configurar paths ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from fase2.models.expert2_convnext_small import Expert2ConvNeXtSmall
from fase2.dataloader_expert2 import build_dataloaders_expert2
from fase2.losses import FocalLossMultiClass
from datasets.isic import cutmix_data, mixup_data
from fase2.expert2_config import (
    # General
    EXPERT2_NUM_CLASSES,
    EXPERT2_BATCH_SIZE,
    EXPERT2_ACCUMULATION_STEPS,
    EXPERT2_LABEL_SMOOTHING,
    EXPERT2_CHECKPOINT_DIR,
    EXPERT2_CHECKPOINT_NAME,
    EXPERT2_TOTAL_EPOCHS,
    # Fase 1
    EXPERT2_PHASE1_EPOCHS,
    EXPERT2_PHASE1_LR,
    EXPERT2_PHASE1_WD,
    EXPERT2_PHASE1_ETA_MIN,
    # Fase 2
    EXPERT2_PHASE2_EPOCHS,
    EXPERT2_PHASE2_HEAD_LR,
    EXPERT2_PHASE2_BACKBONE_LR,
    EXPERT2_PHASE2_WD,
    EXPERT2_PHASE2_T0,
    EXPERT2_PHASE2_T_MULT,
    EXPERT2_PHASE2_ETA_MIN,
    # Fase 3
    EXPERT2_PHASE3_EPOCHS,
    EXPERT2_PHASE3_HEAD_LR,
    EXPERT2_PHASE3_BACKBONE_LR,
    EXPERT2_PHASE3_WD,
    EXPERT2_PHASE3_T0,
    EXPERT2_PHASE3_T_MULT,
    EXPERT2_PHASE3_ETA_MIN,
    EXPERT2_EARLY_STOPPING_PATIENCE,
)

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("expert2_train")

# ── Rutas de salida ────────────────────────────────────────────────────
_CHECKPOINT_PATH = Path(os.path.join(EXPERT2_CHECKPOINT_DIR, EXPERT2_CHECKPOINT_NAME))
# Si la ruta es relativa, hacerla absoluta respecto al proyecto
if not _CHECKPOINT_PATH.is_absolute():
    _CHECKPOINT_PATH = _PROJECT_ROOT / _CHECKPOINT_PATH
_TRAINING_LOG_PATH = _CHECKPOINT_PATH.parent / "expert2_training_log.json"

# ── Constantes de entrenamiento ────────────────────────────────────────
_SEED = 42
_MIN_DELTA = 0.001  # Mejora mínima para considerar progreso
_GRAD_CLIP_NORM = 1.0


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


class EarlyStoppingF1:
    """
    Early stopping por val_f1_macro (maximizar) con patience configurable.

    Detiene el entrenamiento si val_f1_macro no mejora (delta > min_delta)
    durante 'patience' épocas consecutivas.
    """

    def __init__(self, patience: int, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_f1_macro: float) -> bool:
        """
        Evalúa si el entrenamiento debe detenerse.

        Args:
            val_f1_macro: F1-macro de validación de la época actual.

        Returns:
            True si se debe detener, False si se debe continuar.
        """
        if val_f1_macro > self.best_score + self.min_delta:
            self.best_score = val_f1_macro
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
    cutmix_prob: float = 0.3,
    mixup_prob: float = 0.2,
) -> float:
    """
    Ejecuta una época de entrenamiento con FP16, gradient accumulation,
    gradient clipping y CutMix/MixUp batch-level augmentation.

    Args:
        model: modelo a entrenar.
        loader: DataLoader de train.
        criterion: función de pérdida (FocalLossMultiClass).
        optimizer: optimizador (AdamW).
        scaler: GradScaler para FP16.
        device: dispositivo (cuda/cpu).
        accumulation_steps: pasos de acumulación de gradientes.
        use_fp16: usar mixed precision.
        dry_run: si True, ejecuta solo 2 batches.
        cutmix_prob: probabilidad de aplicar CutMix (default 0.3).
        mixup_prob: probabilidad de aplicar MixUp (default 0.2).

    Returns:
        Loss promedio de la época.
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

        # ── Selección de augmentación de batch (mutuamente excluyentes) ──
        r = np.random.random()
        use_cutmix = r < cutmix_prob
        use_mixup = (not use_cutmix) and (r < cutmix_prob + mixup_prob)

        # ── Forward con autocast FP16 ──────────────────────────────
        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            if use_cutmix:
                imgs, y_a, y_b, lam = cutmix_data(imgs, labels)
                logits = model(imgs)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(
                    logits, y_b
                )
            elif use_mixup:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels)
                logits = model(imgs)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(
                    logits, y_b
                )
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
            loss = loss / accumulation_steps

        # ── Backward con GradScaler ────────────────────────────────
        scaler.scale(loss).backward()

        # ── Optimizer step cada accumulation_steps batches ─────────
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
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
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
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
        - val_loss: CrossEntropyLoss promedio.
        - val_acc: accuracy top-1.
        - val_f1_macro: F1-score macro (sklearn).
        - val_bmca: Balanced Multi-Class Accuracy (sklearn balanced_accuracy_score).
        - val_auc: AUC-ROC macro one-vs-rest (sobre las 8 clases de entrenamiento).

    Returns:
        dict con keys: val_loss, val_acc, val_f1_macro, val_bmca, val_auc.
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
            logits = model(imgs)
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
    all_logits_t = torch.cat(all_logits, dim=0)  # [N, num_classes]
    all_probs = torch.softmax(all_logits_t, dim=1).numpy()
    all_preds = all_probs.argmax(axis=1)
    all_labels_np = torch.cat(all_labels, dim=0).numpy()

    # Accuracy top-1
    acc = float(np.mean(all_preds == all_labels_np))

    # F1 Macro
    f1_macro = float(
        f1_score(all_labels_np, all_preds, average="macro", zero_division=0)
    )

    # BMCA = Balanced Multi-Class Accuracy (mean per-class accuracy)
    bmca = float(balanced_accuracy_score(all_labels_np, all_preds))

    # AUC-ROC per class (one-vs-rest, sobre las 8 clases de entrenamiento)
    try:
        auc = float(
            roc_auc_score(
                all_labels_np,
                all_probs[:, :EXPERT2_NUM_CLASSES],
                multi_class="ovr",
                average="macro",
                labels=list(range(EXPERT2_NUM_CLASSES)),
            )
        )
    except ValueError:
        auc = 0.0
        log.warning("[Val] AUC-ROC no computable (clases insuficientes) → AUC=0.0")

    return {
        "val_loss": avg_loss,
        "val_acc": acc,
        "val_f1_macro": f1_macro,
        "val_bmca": bmca,
        "val_auc": auc,
    }


def _run_phase(
    phase_num: int,
    phase_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    use_fp16: bool,
    num_epochs: int,
    global_epoch_offset: int,
    best_f1_macro: float,
    training_log: list,
    early_stopping: EarlyStoppingF1 | None,
    dry_run: bool,
    cutmix_prob: float = 0.3,
    mixup_prob: float = 0.2,
) -> float:
    """
    Ejecuta una fase completa de entrenamiento.

    Args:
        phase_num: número de fase (1, 2, 3).
        phase_name: nombre descriptivo de la fase.
        model: modelo a entrenar.
        train_loader: DataLoader de train.
        val_loader: DataLoader de validación.
        criterion: función de pérdida.
        optimizer: optimizador de la fase (no reutilizar entre fases).
        scheduler: scheduler de la fase.
        scaler: GradScaler compartido.
        device: dispositivo.
        use_fp16: usar mixed precision.
        num_epochs: número de épocas de esta fase.
        global_epoch_offset: época global antes de empezar esta fase.
        best_f1_macro: mejor F1-macro hasta ahora (para checkpoint).
        training_log: lista de logs acumulada.
        early_stopping: objeto EarlyStoppingF1 o None si no aplica.
        dry_run: modo dry-run.

    Returns:
        Mejor val_f1_macro alcanzado (actualizado si mejoró).
    """
    max_epochs = 1 if dry_run else num_epochs

    log.info(f"\n{'=' * 70}")
    log.info(f"  FASE {phase_num}: {phase_name}")
    log.info(
        f"  Épocas: {max_epochs} (global {global_epoch_offset + 1}"
        f"-{global_epoch_offset + max_epochs})"
    )
    log.info(f"  Params entrenables: {model.count_parameters():,}")
    for i, pg in enumerate(optimizer.param_groups):
        log.info(
            f"  Param group {i}: lr={pg['lr']:.2e}, wd={pg.get('weight_decay', 0):.1e}"
        )
    log.info(f"{'=' * 70}\n")

    for epoch_local in range(max_epochs):
        epoch_global = global_epoch_offset + epoch_local + 1
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
            cutmix_prob=cutmix_prob,
            mixup_prob=mixup_prob,
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
        val_acc = val_results["val_acc"]
        val_f1_macro = val_results["val_f1_macro"]
        val_bmca = val_results["val_bmca"]
        val_auc = val_results["val_auc"]

        is_best = val_f1_macro > best_f1_macro

        log.info(
            f"[Epoch {epoch_global:3d}/{EXPERT2_TOTAL_EPOCHS} | F{phase_num}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_f1_macro={val_f1_macro:.4f} | "
            f"val_bmca={val_bmca:.4f} | val_auc={val_auc:.4f} | "
            f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
            f"{' ★ BEST' if is_best else ''}"
        )
        _log_vram(f"epoch-{epoch_global}")

        # ── Guardar log de métricas ────────────────────────────────
        epoch_log = {
            "epoch": epoch_global,
            "phase": phase_num,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1_macro,
            "val_bmca": val_bmca,
            "val_auc": val_auc,
            "lr": current_lr,
            "epoch_time_s": round(epoch_time, 1),
            "is_best": is_best,
        }
        training_log.append(epoch_log)

        # ── Guardar mejor checkpoint (por val_f1_macro) ────────────
        if is_best:
            best_f1_macro = val_f1_macro
            checkpoint = {
                "epoch": epoch_global,
                "phase": phase_num,
                "model_state_dict": model.state_dict(),
                "val_f1_macro": val_f1_macro,
                "val_bmca": val_bmca,
                "val_auc": val_auc,
                "val_loss": val_loss,
            }
            if not dry_run:
                torch.save(checkpoint, _CHECKPOINT_PATH)
                log.info(f"  → Checkpoint guardado: {_CHECKPOINT_PATH}")

        # ── Guardar training log ───────────────────────────────────
        if not dry_run:
            with open(_TRAINING_LOG_PATH, "w") as f:
                json.dump(training_log, f, indent=2)

        # ── Early stopping (solo si se proporcionó) ────────────────
        if early_stopping is not None and not dry_run:
            if early_stopping.step(val_f1_macro):
                log.info(
                    f"\n[EarlyStopping] Detenido en época {epoch_global}. "
                    f"val_f1_macro no mejoró en {early_stopping.patience} épocas. "
                    f"Mejor val_f1_macro: {best_f1_macro:.4f}"
                )
                break

    # ── Resumen de fase ────────────────────────────────────────────
    phase_logs = [e for e in training_log if e["phase"] == phase_num]
    if phase_logs:
        best_epoch_log = max(phase_logs, key=lambda x: x["val_f1_macro"])
        log.info(
            f"\n[Fase {phase_num} resumen] Mejor época: {best_epoch_log['epoch']} | "
            f"val_f1_macro={best_epoch_log['val_f1_macro']:.4f} | "
            f"val_bmca={best_epoch_log['val_bmca']:.4f} | "
            f"val_auc={best_epoch_log['val_auc']:.4f}"
        )

    return best_f1_macro


def train(dry_run: bool = False) -> None:
    """
    Función principal de entrenamiento trifásico del Expert 2.

    Orquesta las 3 fases secuencialmente, pasando el modelo entrenado
    de una fase a la siguiente con nuevo optimizador y scheduler.

    Args:
        dry_run: si True, ejecuta 2 batches de train y 1 de val por fase
                 para verificar el pipeline sin entrenar.
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

    use_fp16 = device.type == "cuda"
    if not use_fp16:
        log.info("[Expert2] FP16 desactivado (no hay GPU). Usando FP32 en CPU.")

    if dry_run:
        log.info("[Expert2] === MODO DRY-RUN === (2 batches train + 1 batch val)")

    # ── Modelo ─────────────────────────────────────────────────────
    model = Expert2ConvNeXtSmall(
        num_classes=EXPERT2_NUM_CLASSES,
        pretrained=True,
    ).to(device)

    total_params = model.count_all_parameters()
    log.info(
        f"[Expert2] Modelo ConvNeXt-Small creado: {total_params:,} parámetros totales"
    )
    _log_vram("post-model")

    # ── DataLoaders ────────────────────────────────────────────────
    num_workers = 0 if dry_run else 4
    train_loader, val_loader, _test_loader, class_weights = build_dataloaders_expert2(
        batch_size=EXPERT2_BATCH_SIZE,
        num_workers=num_workers,
    )
    class_weights = class_weights.to(device)

    # ── Loss (compartida entre las 3 fases) ────────────────────────
    criterion = FocalLossMultiClass(
        gamma=2.0,
        weight=class_weights,
        label_smoothing=EXPERT2_LABEL_SMOOTHING,
    )
    log.info(
        f"[Expert2] Loss: FocalLossMultiClass("
        f"gamma=2.0, weight=class_weights[{class_weights.shape[0]}], "
        f"label_smoothing={EXPERT2_LABEL_SMOOTHING})"
    )

    # ── GradScaler para FP16 (compartido entre fases) ──────────────
    scaler = GradScaler(device=device.type, enabled=use_fp16)

    # ── Directorio de checkpoints ──────────────────────────────────
    _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Estado global ──────────────────────────────────────────────
    best_f1_macro = -float("inf")
    training_log: list[dict] = []

    log.info(f"\n{'=' * 70}")
    log.info(f"  INICIO DE ENTRENAMIENTO — Expert 2 (ConvNeXt-Small / ISIC 2019)")
    log.info(
        f"  Total épocas: {EXPERT2_TOTAL_EPOCHS} | Batch efectivo: "
        f"{EXPERT2_BATCH_SIZE}×{EXPERT2_ACCUMULATION_STEPS}="
        f"{EXPERT2_BATCH_SIZE * EXPERT2_ACCUMULATION_STEPS}"
    )
    log.info(
        f"  FP16: {use_fp16} | Accumulation: {EXPERT2_ACCUMULATION_STEPS} | "
        f"Grad clip: {_GRAD_CLIP_NORM}"
    )
    log.info(
        f"  3 fases: head-only({EXPERT2_PHASE1_EPOCHS}) → "
        f"partial({EXPERT2_PHASE2_EPOCHS}) → full({EXPERT2_PHASE3_EPOCHS})"
    )
    log.info(f"{'=' * 70}\n")

    # ================================================================
    # FASE 1: Solo head, backbone congelado
    # ================================================================
    model.freeze_backbone()
    log.info(
        f"[Fase 1] freeze_backbone() → "
        f"{model.count_parameters():,} params entrenables (solo head)"
    )

    optimizer_p1 = torch.optim.AdamW(
        model.get_head_params(),
        lr=EXPERT2_PHASE1_LR,
        weight_decay=EXPERT2_PHASE1_WD,
    )
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1,
        T_max=EXPERT2_PHASE1_EPOCHS,
        eta_min=EXPERT2_PHASE1_ETA_MIN,
    )

    best_f1_macro = _run_phase(
        phase_num=1,
        phase_name="Head-only (backbone congelado)",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_p1,
        scheduler=scheduler_p1,
        scaler=scaler,
        device=device,
        use_fp16=use_fp16,
        num_epochs=EXPERT2_PHASE1_EPOCHS,
        global_epoch_offset=0,
        best_f1_macro=best_f1_macro,
        training_log=training_log,
        early_stopping=None,
        dry_run=dry_run,
    )

    # ================================================================
    # FASE 2: Fine-tuning diferencial (últimos 2 stages + head)
    # ================================================================
    model.unfreeze_last_stages(n=2)
    log.info(
        f"[Fase 2] unfreeze_last_stages(n=2) → "
        f"{model.count_parameters():,} params entrenables"
    )

    optimizer_p2 = torch.optim.AdamW(
        [
            {"params": model.get_head_params(), "lr": EXPERT2_PHASE2_HEAD_LR},
            {"params": model.get_backbone_params(), "lr": EXPERT2_PHASE2_BACKBONE_LR},
        ],
        weight_decay=EXPERT2_PHASE2_WD,
    )
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_p2,
        T_0=EXPERT2_PHASE2_T0,
        T_mult=EXPERT2_PHASE2_T_MULT,
        eta_min=EXPERT2_PHASE2_ETA_MIN,
    )

    best_f1_macro = _run_phase(
        phase_num=2,
        phase_name="Fine-tuning diferencial (últimos 2 stages + head)",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_p2,
        scheduler=scheduler_p2,
        scaler=scaler,
        device=device,
        use_fp16=use_fp16,
        num_epochs=EXPERT2_PHASE2_EPOCHS,
        global_epoch_offset=EXPERT2_PHASE1_EPOCHS,
        best_f1_macro=best_f1_macro,
        training_log=training_log,
        early_stopping=None,
        dry_run=dry_run,
    )

    # ================================================================
    # FASE 3: Full fine-tuning + early stopping
    # ================================================================
    model.unfreeze_all()
    log.info(
        f"[Fase 3] unfreeze_all() → "
        f"{model.count_parameters():,} params entrenables (todo descongelado)"
    )

    optimizer_p3 = torch.optim.AdamW(
        [
            {"params": model.get_head_params(), "lr": EXPERT2_PHASE3_HEAD_LR},
            {"params": model.get_backbone_params(), "lr": EXPERT2_PHASE3_BACKBONE_LR},
        ],
        weight_decay=EXPERT2_PHASE3_WD,
    )
    scheduler_p3 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_p3,
        T_0=EXPERT2_PHASE3_T0,
        T_mult=EXPERT2_PHASE3_T_MULT,
        eta_min=EXPERT2_PHASE3_ETA_MIN,
    )

    early_stopping = EarlyStoppingF1(
        patience=EXPERT2_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    log.info(
        f"[Fase 3] EarlyStopping: monitor=val_f1_macro, "
        f"patience={EXPERT2_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
    )

    best_f1_macro = _run_phase(
        phase_num=3,
        phase_name="Full fine-tuning + early stopping",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_p3,
        scheduler=scheduler_p3,
        scaler=scaler,
        device=device,
        use_fp16=use_fp16,
        num_epochs=EXPERT2_PHASE3_EPOCHS,
        global_epoch_offset=EXPERT2_PHASE1_EPOCHS + EXPERT2_PHASE2_EPOCHS,
        best_f1_macro=best_f1_macro,
        training_log=training_log,
        early_stopping=early_stopping,
        dry_run=dry_run,
    )

    # ── Resumen final ──────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info(f"  ENTRENAMIENTO FINALIZADO — Expert 2 (ConvNeXt-Small / ISIC 2019)")
    log.info(f"  Mejor val_f1_macro: {best_f1_macro:.4f}")
    if training_log:
        best_epoch = max(training_log, key=lambda x: x["val_f1_macro"])
        log.info(
            f"  Mejor época: {best_epoch['epoch']} (fase {best_epoch['phase']}) | "
            f"F1-macro: {best_epoch['val_f1_macro']:.4f} | "
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
        description="Entrenamiento Expert 2 — ConvNeXt-Small / ISIC 2019 (3 fases)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta 2 batches de train y 1 de val para verificar el pipeline",
    )
    args = parser.parse_args()
    train(dry_run=args.dry_run)

"""
Script de entrenamiento para Expert OA — EfficientNet-B0 sobre Osteoarthritis Knee.

Pipeline completo:
    1. Carga hiperparámetros desde expert_oa_config.py (fuente de verdad)
    2. Construye modelo EfficientNet-B0 pretrained (5 clases KL 0-4)
    3. Entrena con CrossEntropyLoss(weight=class_weights) + FP16 + gradient accumulation
    4. Adam diferencial: lr_backbone=5e-5 / lr_head=5e-4
    5. CosineAnnealingLR scheduler (T_max=30, eta_min=1e-6)
    6. Checkpoint por val_f1_macro máximo
    7. Early stopping por val_f1_macro (patience=10)
    8. Métricas: F1-macro (principal), QWK (complementaria)

Uso:
    # Dry-run: verifica el pipeline sin entrenar
    python src/pipeline/fase2/train_expert_oa.py --dry-run

    # Entrenamiento completo
    python src/pipeline/fase2/train_expert_oa.py

Dependencias:
    - src/pipeline/fase2/models/expert_oa_vgg16bn.py: ExpertOAEfficientNetB0
    - src/pipeline/fase2/dataloader_expert_oa.py: get_oa_dataloaders
    - src/pipeline/fase2/expert_oa_config.py: hiperparámetros
    - src/pipeline/datasets/osteoarthritis.py: OAKneeDataset (compute_qwk)
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
from sklearn.metrics import cohen_kappa_score, f1_score

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

from fase2.models.expert_oa_vgg16bn import ExpertOAEfficientNetB0
from fase2.dataloader_expert_oa import get_oa_dataloaders
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
)
from datasets.osteoarthritis import OAKneeDataset

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("expert_oa_train")

# ── Rutas de salida ────────────────────────────────────────────────────
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints" / "expert_02_vgg16_bn"
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "expert_oa_best.pt"
_TRAINING_LOG_PATH = _CHECKPOINT_DIR / "expert_oa_training_log.json"

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
    Early stopping por val_f1_macro con patience configurable.

    Detiene el entrenamiento si val_f1_macro no mejora (delta > min_delta)
    durante 'patience' épocas consecutivas. Modo 'max': mayor es mejor.
    """

    def __init__(self, patience: int, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_f1_macro: float) -> bool:
        """
        Evalua si el entrenamiento debe detenerse.

        Args:
            val_f1_macro: F1-macro de validacion de la epoca actual.

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
) -> float:
    """
    Ejecuta una epoca de entrenamiento con gradient accumulation y FP16.

    Args:
        model: modelo a entrenar (ExpertOAEfficientNetB0)
        loader: DataLoader de train
        criterion: funcion de perdida (CrossEntropyLoss)
        optimizer: optimizador (Adam diferencial)
        scaler: GradScaler para FP16
        device: dispositivo (cuda/cpu)
        accumulation_steps: pasos de acumulacion de gradientes
        use_fp16: usar mixed precision
        dry_run: si True, ejecuta solo 2 batches

    Returns:
        Loss promedio de la epoca
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

        # ── Forward con autocast FP16 ──────────────────────────────
        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(imgs)  # [B, 5]
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

        total_loss += loss.item() * accumulation_steps  # Revertir la normalizacion
        n_batches += 1

        if dry_run:
            log.info(
                f"  [Train batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"logits={list(logits.shape)} | "
                f"loss={loss.item() * accumulation_steps:.4f}"
            )

    # Flush de gradientes residuales si el ultimo bloque no completo accumulation_steps
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
    Ejecuta validacion y calcula metricas.

    Metricas:
        - val_loss: CrossEntropyLoss promedio
        - val_f1_macro: F1-score macro (metrica principal)
        - val_qwk: Quadratic Weighted Kappa (metrica complementaria)

    Returns:
        dict con keys: val_loss, val_f1_macro, val_qwk
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

        if dry_run:
            log.info(
                f"  [Val batch {batch_idx}] "
                f"imgs={list(imgs.shape)} | "
                f"logits={list(logits.shape)} | "
                f"loss={loss.item():.4f}"
            )

    avg_loss = total_loss / max(n_batches, 1)

    # ── Metricas ───────────────────────────────────────────────────
    y_true_np = np.array(all_labels)
    y_pred_np = np.array(all_preds)

    # F1-macro — metrica principal
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # QWK — metrica complementaria ordinal
    qwk = OAKneeDataset.compute_qwk(y_true_np, y_pred_np)

    return {
        "val_loss": avg_loss,
        "val_f1_macro": f1_macro,
        "val_qwk": qwk,
    }


def train(dry_run: bool = False) -> None:
    """
    Funcion principal de entrenamiento del Expert OA.

    Args:
        dry_run: si True, ejecuta 2 batches de train y 2 de val para verificar
                 el pipeline sin entrenar el modelo completo.
    """
    set_seed(_SEED)

    # ── Dispositivo ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[ExpertOA] Dispositivo: {device}")
    if device.type == "cpu":
        log.warning(
            "[ExpertOA] ⚠ Entrenando en CPU — sera muy lento. "
            "Se recomienda GPU con >= 8 GB VRAM."
        )

    # ── Configuracion ──────────────────────────────────────────────
    log.info(f"[ExpertOA] Config: {EXPERT_OA_CONFIG_SUMMARY}")
    if dry_run:
        log.info("[ExpertOA] === MODO DRY-RUN === (2 batches train + 2 batches val)")

    use_fp16 = EXPERT_OA_FP16 and device.type == "cuda"
    if not use_fp16 and EXPERT_OA_FP16:
        log.info("[ExpertOA] FP16 desactivado (no hay GPU). Usando FP32 en CPU.")

    # ── Modelo ─────────────────────────────────────────────────────
    model = ExpertOAEfficientNetB0(
        num_classes=EXPERT_OA_NUM_CLASSES,
        dropout=EXPERT_OA_DROPOUT_FC,
    ).to(device)

    n_params = model.count_parameters()
    log.info(
        f"[ExpertOA] Modelo EfficientNet-B0 creado: {n_params:,} parametros entrenables"
    )
    _log_vram("post-model")

    # ── DataLoaders ────────────────────────────────────────────────
    # Usar num_workers=0 en dry-run para evitar overhead de multiprocessing
    num_workers = 0 if dry_run else 4
    train_loader, val_loader, _test_loader, class_weights = get_oa_dataloaders(
        batch_size=EXPERT_OA_BATCH_SIZE,
        num_workers=num_workers,
    )
    class_weights = class_weights.to(device)

    # ── Loss ───────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    log.info(
        f"[ExpertOA] Loss: CrossEntropyLoss(weight=class_weights[{class_weights.shape[0]}])"
    )

    # ── Optimizer (Adam diferencial) ──────────────────────────────
    optimizer = torch.optim.Adam(
        [
            {"params": model.get_backbone_params(), "lr": EXPERT_OA_LR_BACKBONE},
            {"params": model.get_head_params(), "lr": EXPERT_OA_LR_HEAD},
        ],
        weight_decay=EXPERT_OA_WEIGHT_DECAY,
    )
    log.info(
        f"[ExpertOA] Optimizer: Adam diferencial "
        f"(backbone_lr={EXPERT_OA_LR_BACKBONE}, head_lr={EXPERT_OA_LR_HEAD}, "
        f"wd={EXPERT_OA_WEIGHT_DECAY})"
    )

    # ── Scheduler ──────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EXPERT_OA_SCHEDULER_T_MAX,
        eta_min=EXPERT_OA_SCHEDULER_ETA_MIN,
    )
    log.info(
        f"[ExpertOA] Scheduler: CosineAnnealingLR"
        f"(T_max={EXPERT_OA_SCHEDULER_T_MAX}, eta_min={EXPERT_OA_SCHEDULER_ETA_MIN})"
    )

    # ── GradScaler para FP16 ───────────────────────────────────────
    scaler = GradScaler(device=device.type, enabled=use_fp16)

    # ── Early stopping ─────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=EXPERT_OA_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    log.info(
        f"[ExpertOA] EarlyStopping: monitor={EXPERT_OA_EARLY_STOPPING_MONITOR}, "
        f"patience={EXPERT_OA_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
    )

    # ── Training loop ──────────────────────────────────────────────
    best_f1_macro = -float("inf")
    training_log: list[dict] = []
    max_epochs = 1 if dry_run else EXPERT_OA_MAX_EPOCHS

    log.info(f"\n{'=' * 70}")
    log.info("  INICIO DE ENTRENAMIENTO — Expert OA (EfficientNet-B0 / OA Knee)")
    log.info(
        f"  Epocas max: {max_epochs} | Batch efectivo: "
        f"{EXPERT_OA_BATCH_SIZE}x{EXPERT_OA_ACCUMULATION_STEPS}="
        f"{EXPERT_OA_BATCH_SIZE * EXPERT_OA_ACCUMULATION_STEPS}"
    )
    log.info(f"  FP16: {use_fp16} | Accumulation: {EXPERT_OA_ACCUMULATION_STEPS}")
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
            accumulation_steps=EXPERT_OA_ACCUMULATION_STEPS,
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
        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]

        # ── Log de epoca ───────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        val_loss = val_results["val_loss"]
        val_f1_macro = val_results["val_f1_macro"]
        val_qwk = val_results["val_qwk"]

        is_best = val_f1_macro > best_f1_macro

        log.info(
            f"[Epoch {epoch + 1:3d}/{max_epochs}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_f1_macro={val_f1_macro:.4f} | val_qwk={val_qwk:.4f} | "
            f"lr_bb={current_lr_backbone:.2e} lr_hd={current_lr_head:.2e} | "
            f"time={epoch_time:.1f}s"
            f"{' ★ BEST' if is_best else ''}"
        )
        _log_vram(f"epoch-{epoch + 1}")

        # ── Guardar log de metricas ────────────────────────────────
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1_macro": val_f1_macro,
            "val_qwk": val_qwk,
            "lr_backbone": current_lr_backbone,
            "lr_head": current_lr_head,
            "epoch_time_s": round(epoch_time, 1),
            "is_best": is_best,
        }
        training_log.append(epoch_log)

        # ── Guardar mejor checkpoint (por val_f1_macro) ────────────
        if is_best:
            best_f1_macro = val_f1_macro
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_f1_macro": val_f1_macro,
                "val_qwk": val_qwk,
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
                },
            }
            if not dry_run:
                _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, _CHECKPOINT_PATH)
                log.info(f"  → Checkpoint guardado: {_CHECKPOINT_PATH}")

        # ── Guardar training log ───────────────────────────────────
        if not dry_run:
            _TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_TRAINING_LOG_PATH, "w") as f:
                json.dump(training_log, f, indent=2)

        # ── Early stopping (por val_f1_macro) ──────────────────────
        if not dry_run:
            if early_stopping.step(val_f1_macro):
                log.info(
                    f"\n[EarlyStopping] Detenido en epoca {epoch + 1}. "
                    f"val_f1_macro no mejoro en {EXPERT_OA_EARLY_STOPPING_PATIENCE} epocas. "
                    f"Mejor val_f1_macro: {best_f1_macro:.4f}"
                )
                break

    # ── Resumen final ──────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info("  ENTRENAMIENTO FINALIZADO — Expert OA (EfficientNet-B0 / OA Knee)")
    log.info(f"  Mejor val_f1_macro: {best_f1_macro:.4f}")
    if training_log:
        best_epoch = max(training_log, key=lambda x: x["val_f1_macro"])
        log.info(
            f"  Mejor epoca: {best_epoch['epoch']} | "
            f"F1-macro: {best_epoch['val_f1_macro']:.4f} | "
            f"QWK: {best_epoch['val_qwk']:.4f}"
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
        description="Entrenamiento Expert OA — EfficientNet-B0 / Osteoarthritis Knee"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta 2 batches de train y 2 de val para verificar el pipeline sin entrenar",
    )
    args = parser.parse_args()
    train(dry_run=args.dry_run)

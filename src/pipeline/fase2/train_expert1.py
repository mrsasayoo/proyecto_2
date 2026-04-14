"""
Script de entrenamiento para Expert 1 — ConvNeXt-Tiny sobre NIH ChestXray14.

Pipeline LP-FT (Linear Probing → Fine-Tuning):

    Fase 1 (LP, 5 épocas):  Backbone congelado, solo head + domain_conv entrenables.
                             AdamW(lr=1e-3) sin scheduler.
    Fase 2 (FT, 30 épocas): Todo descongelado.
                             AdamW(lr=1e-4) + CosineAnnealingLR + early stopping.

Evaluación final con TTA (Test-Time Augmentation):
    Promedia logits de test original + test con HorizontalFlip determinista.
    Reporta AUC-ROC por clase (14 patologías) y macro AUC.

Uso:
    # Dry-run: verifica el pipeline sin entrenar
    python src/pipeline/fase2/train_expert1.py --dry-run

    # Entrenamiento completo
    python src/pipeline/fase2/train_expert1.py

    # Con data-root explícito
    python src/pipeline/fase2/train_expert1.py --data-root /ruta/al/proyecto

Dependencias:
    - src/pipeline/fase2/models/expert1_convnext.py: Expert1ConvNeXtTiny
    - src/pipeline/fase2/dataloader_expert1.py: build_expert1_dataloaders
    - src/pipeline/fase2/expert1_config.py: hiperparámetros LP-FT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from sklearn.metrics import roc_auc_score

# ── Configurar paths ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # proyecto_2/
_PIPELINE_ROOT = _PROJECT_ROOT / "src" / "pipeline"
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from config import CHEST_PATHOLOGIES
from fase2.models.expert1_convnext import Expert1ConvNeXtTiny
from fase2.dataloader_expert1 import build_expert1_dataloaders
from fase2.expert1_config import (
    EXPERT1_LP_EPOCHS,
    EXPERT1_FT_EPOCHS,
    EXPERT1_LP_LR,
    EXPERT1_FT_LR,
    EXPERT1_WEIGHT_DECAY,
    EXPERT1_DROPOUT_FC,
    EXPERT1_BATCH_SIZE,
    EXPERT1_NUM_WORKERS,
    EXPERT1_ACCUMULATION_STEPS,
    EXPERT1_FP16,
    EXPERT1_NUM_CLASSES,
    EXPERT1_EARLY_STOPPING_PATIENCE,
    EXPERT1_CONFIG_SUMMARY,
)

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("expert1_train")

# ── Rutas de salida ────────────────────────────────────────────────────
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "expert_00_convnext_tiny" / "expert1_best.pt"
_TRAINING_LOG_PATH = (
    _CHECKPOINT_DIR / "expert_00_convnext_tiny" / "expert1_training_log.json"
)

# ── Constantes de entrenamiento ────────────────────────────────────────
_SEED = 42
_MIN_DELTA = 0.001  # Mejora mínima para considerar progreso en early stopping
_TOTAL_EPOCHS = EXPERT1_LP_EPOCHS + EXPERT1_FT_EPOCHS


# ── Rutas por defecto del dataset ──────────────────────────────────────


def get_default_paths(project_root: Path | None = None) -> dict[str, Path]:
    """Devuelve las rutas por defecto del dataset NIH ChestXray14.

    Args:
        project_root: raíz del proyecto. Si None, usa _PROJECT_ROOT auto-detectado.

    Returns:
        dict con claves csv_path, images_dir, train_split, val_split, test_split.
    """
    root = project_root or _PROJECT_ROOT
    base = root / "datasets" / "nih_chest_xrays"
    splits = base / "splits"
    return {
        "csv_path": base / "Data_Entry_2017.csv",
        "images_dir": base / "all_images",
        "train_split": splits / "nih_train_list.txt",
        "val_split": splits / "nih_val_list.txt",
        "test_split": splits / "nih_test_list.txt",
    }


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


class EarlyStoppingAUC:
    """Early stopping por val_macro_auc (maximizar) con patience configurable.

    Detiene el entrenamiento si val_macro_auc no mejora (delta > min_delta)
    durante 'patience' épocas consecutivas.
    """

    def __init__(self, patience: int, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float = -float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, val_macro_auc: float) -> bool:
        """Evalúa si el entrenamiento debe detenerse.

        Args:
            val_macro_auc: macro AUC de validación de la época actual.

        Returns:
            True si se debe detener, False si se debe continuar.
        """
        if val_macro_auc > self.best_score + self.min_delta:
            self.best_score = val_macro_auc
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int,
    use_fp16: bool,
    dry_run: bool = False,
) -> float:
    """Ejecuta una época de entrenamiento con gradient accumulation y FP16.

    Args:
        model: modelo a entrenar.
        loader: DataLoader de train.
        criterion: función de pérdida (BCEWithLogitsLoss).
        optimizer: optimizador (AdamW).
        scaler: GradScaler para FP16.
        device: dispositivo (cuda/cpu).
        accumulation_steps: pasos de acumulación de gradientes.
        use_fp16: usar mixed precision.
        dry_run: si True, ejecuta solo 2 batches.

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
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
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

    # Flush de gradientes residuales
    if n_batches % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_fp16: bool,
    dry_run: bool = False,
) -> dict[str, float | list[float]]:
    """Ejecuta validación y calcula métricas multilabel.

    Métricas:
        - val_loss: BCEWithLogitsLoss promedio.
        - val_macro_auc: media de AUC-ROC por clase (14 clases).
        - val_auc_per_class: lista de 14 AUC-ROC individuales.

    Returns:
        dict con val_loss, val_macro_auc, val_auc_per_class.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch_idx, (imgs, labels, _stems) in enumerate(loader):
        if dry_run and batch_idx >= 1:
            break

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(imgs)
            loss = criterion(logits, labels)

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

    all_logits_t = torch.cat(all_logits, dim=0)  # [N, 14]
    all_labels_np = torch.cat(all_labels, dim=0).numpy().astype(int)  # [N, 14]
    probs = torch.sigmoid(all_logits_t).numpy()  # [N, 14]

    # AUC-ROC por clase (skip clases con una sola label en este split)
    auc_per_class: list[float] = []
    for c in range(EXPERT1_NUM_CLASSES):
        try:
            auc_per_class.append(roc_auc_score(all_labels_np[:, c], probs[:, c]))
        except ValueError:
            auc_per_class.append(0.0)
    macro_auc = float(np.mean(auc_per_class))

    return {
        "val_loss": avg_loss,
        "val_macro_auc": macro_auc,
        "val_auc_per_class": auc_per_class,
    }


@torch.no_grad()
def eval_with_tta(
    model: nn.Module,
    dl_orig: torch.utils.data.DataLoader,
    dl_flip: torch.utils.data.DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluación con TTA: promedia logits de original + HorizontalFlip.

    Args:
        model: modelo entrenado.
        dl_orig: DataLoader de test sin augmentation.
        dl_flip: DataLoader de test con HorizontalFlip(p=1.0).
        device: dispositivo.
        use_fp16: usar mixed precision.

    Returns:
        (tta_logits, labels) — tensores CPU, shape [N, 14] cada uno.
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for (x_orig, labels_orig, _s1), (x_flip, _labels_flip, _s2) in zip(
        dl_orig, dl_flip
    ):
        x_orig = x_orig.to(device, non_blocking=True)
        x_flip = x_flip.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            logits_orig = model(x_orig)
            logits_flip = model(x_flip)

        tta_logits = (logits_orig + logits_flip) / 2.0
        all_logits.append(tta_logits.cpu())
        all_labels.append(labels_orig.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def _run_phase(
    phase_name: str,
    phase_tag: str,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: GradScaler,
    device: torch.device,
    use_fp16: bool,
    num_epochs: int,
    global_epoch_offset: int,
    best_macro_auc: float,
    training_log: list[dict],
    early_stopping: EarlyStoppingAUC | None,
    dry_run: bool,
) -> float:
    """Ejecuta una fase completa de entrenamiento (LP o FT).

    Args:
        phase_name: nombre descriptivo (para logs).
        phase_tag: etiqueta corta ("LP" o "FT").
        model: modelo a entrenar.
        train_loader: DataLoader de train.
        val_loader: DataLoader de validación.
        criterion: función de pérdida.
        optimizer: optimizador de la fase.
        scheduler: scheduler (None para LP).
        scaler: GradScaler compartido.
        device: dispositivo.
        use_fp16: usar mixed precision.
        num_epochs: épocas de esta fase.
        global_epoch_offset: época global antes de empezar esta fase.
        best_macro_auc: mejor macro AUC hasta ahora (para checkpoint).
        training_log: lista de logs acumulada (in-place).
        early_stopping: early stopper o None si no aplica.
        dry_run: modo dry-run.

    Returns:
        Mejor val_macro_auc alcanzado (actualizado si mejoró).
    """
    max_epochs = 1 if dry_run else num_epochs

    log.info(f"\n{'=' * 70}")
    log.info(f"  {phase_tag}: {phase_name}")
    log.info(
        f"  Épocas: {max_epochs} "
        f"(global {global_epoch_offset + 1}-{global_epoch_offset + max_epochs})"
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
            accumulation_steps=EXPERT1_ACCUMULATION_STEPS,
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

        # ── Scheduler step (solo FT) ──────────────────────────────
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Extraer métricas ───────────────────────────────────────
        epoch_time = time.time() - epoch_start
        val_loss = val_results["val_loss"]
        val_macro_auc = val_results["val_macro_auc"]

        is_best = val_macro_auc > best_macro_auc + _MIN_DELTA

        log.info(
            f"[Epoch {epoch_global:3d}/{_TOTAL_EPOCHS} | {phase_tag}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_macro_auc={val_macro_auc:.4f} | "
            f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
            f"{' ★ BEST' if is_best else ''}"
        )
        _log_vram(f"epoch-{epoch_global}")

        # ── Guardar log de métricas ────────────────────────────────
        epoch_log: dict = {
            "epoch": epoch_global,
            "phase": phase_tag,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auc": val_macro_auc,
            "val_auc_per_class": val_results["val_auc_per_class"],
            "lr": current_lr,
            "epoch_time_s": round(epoch_time, 1),
            "is_best": is_best,
        }
        training_log.append(epoch_log)

        # ── Guardar mejor checkpoint (por val_macro_auc, solo en FT) ──
        if is_best:
            best_macro_auc = val_macro_auc
            checkpoint = {
                "epoch": epoch_global,
                "phase": phase_tag,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_auc": val_macro_auc,
                "val_loss": val_loss,
                "val_auc_per_class": val_results["val_auc_per_class"],
                "config": {
                    "lp_lr": EXPERT1_LP_LR,
                    "ft_lr": EXPERT1_FT_LR,
                    "weight_decay": EXPERT1_WEIGHT_DECAY,
                    "dropout_fc": EXPERT1_DROPOUT_FC,
                    "batch_size": EXPERT1_BATCH_SIZE,
                    "accumulation_steps": EXPERT1_ACCUMULATION_STEPS,
                    "fp16": EXPERT1_FP16,
                    "lp_epochs": EXPERT1_LP_EPOCHS,
                    "ft_epochs": EXPERT1_FT_EPOCHS,
                    "seed": _SEED,
                },
            }
            if not dry_run:
                torch.save(checkpoint, _CHECKPOINT_PATH)
                log.info(f"  -> Checkpoint guardado: {_CHECKPOINT_PATH}")

        # ── Guardar training log ───────────────────────────────────
        if not dry_run:
            with open(_TRAINING_LOG_PATH, "w") as f:
                json.dump(training_log, f, indent=2)

        # ── Early stopping (solo si se proporcionó) ────────────────
        if early_stopping is not None and not dry_run:
            if early_stopping.step(val_macro_auc):
                log.info(
                    f"\n[EarlyStopping] Detenido en época {epoch_global}. "
                    f"val_macro_auc no mejoró en "
                    f"{early_stopping.patience} épocas. "
                    f"Mejor val_macro_auc: {best_macro_auc:.4f}"
                )
                break

    # ── Resumen de fase ────────────────────────────────────────────
    phase_logs = [e for e in training_log if e["phase"] == phase_tag]
    if phase_logs:
        best_epoch_log = max(phase_logs, key=lambda x: x["val_macro_auc"])
        log.info(
            f"\n[{phase_tag} resumen] Mejor época: {best_epoch_log['epoch']} | "
            f"val_macro_auc={best_epoch_log['val_macro_auc']:.4f}"
        )

    return best_macro_auc


def train(dry_run: bool = False, data_root: str | None = None) -> None:
    """Función principal de entrenamiento LP-FT del Expert 1.

    Orquesta las 2 fases secuencialmente: Linear Probing y Fine-Tuning.
    Al final, evalúa el mejor modelo con TTA sobre el test set y reporta
    AUC-ROC por clase y macro AUC.

    Args:
        dry_run: si True, ejecuta 2 batches de train y 1 de val para verificar
                 el pipeline sin entrenar.
        data_root: ruta raíz del proyecto. Si None, se auto-detecta.
    """
    set_seed(_SEED)

    # ── Dispositivo ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Expert1] Dispositivo: {device}")
    if device.type == "cpu":
        log.warning(
            "[Expert1] Entrenando en CPU — será lento. "
            "Se recomienda GPU con >= 8 GB VRAM."
        )

    # ── Configuración ──────────────────────────────────────────────
    log.info(f"[Expert1] Config: {EXPERT1_CONFIG_SUMMARY}")
    if dry_run:
        log.info("[Expert1] === MODO DRY-RUN === (2 batches train + 1 batch val)")

    use_fp16 = EXPERT1_FP16 and device.type == "cuda"
    if not use_fp16 and EXPERT1_FP16:
        log.info("[Expert1] FP16 desactivado (no hay GPU). Usando FP32 en CPU.")

    # ── Modelo ─────────────────────────────────────────────────────
    model = Expert1ConvNeXtTiny(
        dropout_fc=EXPERT1_DROPOUT_FC,
        num_classes=EXPERT1_NUM_CLASSES,
        pretrained=True,
    ).to(device)

    n_params_total = model.count_all_parameters()
    log.info(
        f"[Expert1] Modelo ConvNeXt-Tiny creado: {n_params_total:,} parámetros totales"
    )
    _log_vram("post-model")

    # ── DataLoaders ────────────────────────────────────────────────
    num_workers = 0 if dry_run else EXPERT1_NUM_WORKERS

    project_root = Path(data_root) if data_root else _PROJECT_ROOT
    paths = get_default_paths(project_root)

    loaders = build_expert1_dataloaders(
        csv_path=str(paths["csv_path"]),
        images_dir=str(paths["images_dir"]),
        train_split_file=str(paths["train_split"]),
        val_split_file=str(paths["val_split"]),
        test_split_file=str(paths["test_split"]),
        model_mean=model.model_mean,
        model_std=model.model_std,
        batch_size=EXPERT1_BATCH_SIZE,
        num_workers=num_workers,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    test_flip_loader = loaders["test_flip"]
    pos_weight = loaders["pos_weight"]

    # ── Loss ───────────────────────────────────────────────────────
    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    log.info(f"[Expert1] Loss: BCEWithLogitsLoss(pos_weight shape={pos_weight.shape})")

    # ── GradScaler para FP16 (compartido entre fases) ──────────────
    scaler = GradScaler(device=device.type, enabled=use_fp16)

    # ── Directorio de checkpoints ──────────────────────────────────
    _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Estado global ──────────────────────────────────────────────
    best_macro_auc: float = -float("inf")
    training_log: list[dict] = []

    log.info(f"\n{'=' * 70}")
    log.info("  INICIO DE ENTRENAMIENTO — Expert 1 (ConvNeXt-Tiny / ChestXray14)")
    log.info(
        f"  LP: {EXPERT1_LP_EPOCHS} épocas (LR={EXPERT1_LP_LR}) | "
        f"FT: {EXPERT1_FT_EPOCHS} épocas (LR={EXPERT1_FT_LR})"
    )
    log.info(
        f"  Batch efectivo: "
        f"{EXPERT1_BATCH_SIZE}x{EXPERT1_ACCUMULATION_STEPS}="
        f"{EXPERT1_BATCH_SIZE * EXPERT1_ACCUMULATION_STEPS}"
    )
    log.info(f"  FP16: {use_fp16} | Accumulation: {EXPERT1_ACCUMULATION_STEPS}")
    log.info(f"{'=' * 70}\n")

    # ================================================================
    # FASE 1: Linear Probing (backbone congelado)
    # ================================================================
    model.freeze_backbone()
    log.info(
        f"[LP] freeze_backbone() -> "
        f"{model.count_parameters():,} params entrenables (head + domain_conv)"
    )

    optimizer_lp = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=EXPERT1_LP_LR,
        weight_decay=EXPERT1_WEIGHT_DECAY,
    )

    best_macro_auc = _run_phase(
        phase_name="Linear Probing (backbone congelado)",
        phase_tag="LP",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_lp,
        scheduler=None,  # Sin scheduler en LP
        scaler=scaler,
        device=device,
        use_fp16=use_fp16,
        num_epochs=EXPERT1_LP_EPOCHS,
        global_epoch_offset=0,
        best_macro_auc=best_macro_auc,
        training_log=training_log,
        early_stopping=None,  # Sin early stopping en LP
        dry_run=dry_run,
    )

    # ================================================================
    # FASE 2: Fine-Tuning (todo descongelado)
    # ================================================================
    model.unfreeze_backbone()
    log.info(
        f"[FT] unfreeze_backbone() -> "
        f"{model.count_parameters():,} params entrenables (todo descongelado)"
    )

    optimizer_ft = torch.optim.AdamW(
        model.parameters(),
        lr=EXPERT1_FT_LR,
        weight_decay=EXPERT1_WEIGHT_DECAY,
    )
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft,
        T_max=EXPERT1_FT_EPOCHS,
    )

    early_stopping = EarlyStoppingAUC(
        patience=EXPERT1_EARLY_STOPPING_PATIENCE,
        min_delta=_MIN_DELTA,
    )
    log.info(
        f"[FT] EarlyStopping: monitor=val_macro_auc, "
        f"patience={EXPERT1_EARLY_STOPPING_PATIENCE}, min_delta={_MIN_DELTA}"
    )

    best_macro_auc = _run_phase(
        phase_name="Fine-Tuning (todo descongelado + early stopping)",
        phase_tag="FT",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_ft,
        scheduler=scheduler_ft,
        scaler=scaler,
        device=device,
        use_fp16=use_fp16,
        num_epochs=EXPERT1_FT_EPOCHS,
        global_epoch_offset=EXPERT1_LP_EPOCHS,
        best_macro_auc=best_macro_auc,
        training_log=training_log,
        early_stopping=early_stopping,
        dry_run=dry_run,
    )

    # ================================================================
    # EVALUACIÓN FINAL CON TTA
    # ================================================================
    log.info(f"\n{'=' * 70}")
    log.info("  EVALUACIÓN FINAL — Test set con TTA (original + HorizontalFlip)")
    log.info(f"{'=' * 70}\n")

    # Cargar mejor checkpoint para evaluación
    if _CHECKPOINT_PATH.exists() and not dry_run:
        ckpt = torch.load(_CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(
            f"[TTA] Cargado mejor checkpoint: época {ckpt['epoch']} "
            f"(val_macro_auc={ckpt['val_macro_auc']:.4f})"
        )
    else:
        log.info("[TTA] Usando modelo del final del entrenamiento (no hay checkpoint)")

    tta_logits, tta_labels = eval_with_tta(
        model=model,
        dl_orig=test_loader,
        dl_flip=test_flip_loader,
        device=device,
        use_fp16=use_fp16,
    )

    # Calcular métricas TTA
    tta_probs = torch.sigmoid(tta_logits).numpy()
    tta_labels_np = tta_labels.numpy().astype(int)

    test_auc_per_class: list[float] = []
    for c in range(EXPERT1_NUM_CLASSES):
        try:
            auc_c = roc_auc_score(tta_labels_np[:, c], tta_probs[:, c])
        except ValueError:
            auc_c = 0.0
        test_auc_per_class.append(auc_c)

    test_macro_auc = float(np.mean(test_auc_per_class))

    # Reportar AUC por clase
    log.info("[TTA] AUC-ROC por clase:")
    for i, (name, auc_val) in enumerate(zip(CHEST_PATHOLOGIES, test_auc_per_class)):
        log.info(f"  [{i:2d}] {name:20s}: {auc_val:.4f}")
    log.info(f"  {'─' * 30}")
    log.info(f"  Macro AUC (TTA): {test_macro_auc:.4f}")

    # Agregar resultados TTA al training log
    tta_results = {
        "test_macro_auc_tta": test_macro_auc,
        "test_auc_per_class_tta": {
            name: auc_val
            for name, auc_val in zip(CHEST_PATHOLOGIES, test_auc_per_class)
        },
    }
    if training_log:
        training_log.append({"evaluation": "TTA", **tta_results})

    # Guardar training log final
    if not dry_run:
        with open(_TRAINING_LOG_PATH, "w") as f:
            json.dump(training_log, f, indent=2)

    # ── Resumen final ──────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info("  ENTRENAMIENTO FINALIZADO — Expert 1 (ConvNeXt-Tiny / ChestXray14)")
    log.info(f"  Mejor val_macro_auc: {best_macro_auc:.4f}")
    log.info(f"  Test macro AUC (TTA): {test_macro_auc:.4f}")
    if training_log:
        epoch_logs = [e for e in training_log if "epoch" in e]
        if epoch_logs:
            best_epoch = max(epoch_logs, key=lambda x: x["val_macro_auc"])
            log.info(
                f"  Mejor época: {best_epoch['epoch']} "
                f"({best_epoch['phase']}) | "
                f"val_macro_auc: {best_epoch['val_macro_auc']:.4f}"
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
        description="Entrenamiento Expert 1 — ConvNeXt-Tiny / ChestXray14 (LP-FT)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Ejecuta 2 batches de train y 1 de val para verificar el pipeline "
            "sin entrenar"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Ruta raíz del proyecto (default: auto-detectada)",
    )
    args = parser.parse_args()
    train(dry_run=args.dry_run, data_root=args.data_root)

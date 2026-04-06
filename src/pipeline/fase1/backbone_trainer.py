"""
backbone_trainer.py — Entrenamiento End-to-End del Backbone
===========================================================

Responsabilidad única: recibir un backbone entrenable y DataLoaders,
entrenar con CrossEntropyLoss (clasificación de dominio proxy),
y guardar únicamente los pesos del backbone (sin la cabeza lineal).

No gestiona argumentos CLI ni construye datasets — eso lo hace
fase1_train_pipeline.py.
"""

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from fase1_config import (
    TRAIN_GRAD_CLIP,
    TRAIN_WARMUP_EPOCHS,
)

log = logging.getLogger("fase1")


# ══════════════════════════════════════════════════════════════
#  Cabeza lineal para clasificación de dominio proxy
# ══════════════════════════════════════════════════════════════


class LinearHead(nn.Module):
    """
    Cabeza de clasificación lineal sobre el CLS token del backbone.
    Se descarta después del entrenamiento — solo se guardan los pesos
    del backbone.
    """

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, d_model] → logits: [B, n_classes]"""
        return self.fc(z)


# ══════════════════════════════════════════════════════════════
#  Utilidades de checkpoint
# ══════════════════════════════════════════════════════════════


def backbone_checkpoint_exists(checkpoint_path: str) -> bool:
    """Verifica si el checkpoint del backbone existe en disco."""
    return Path(checkpoint_path).exists()


def save_backbone_checkpoint(
    backbone: nn.Module,
    checkpoint_path: str,
    backbone_name: str,
    epoch: int,
    val_acc: float,
) -> None:
    """
    Guarda únicamente los pesos del backbone (sin la cabeza lineal).

    Estructura del checkpoint:
        {
            "backbone_name": str,
            "epoch": int,
            "val_acc": float,
            "state_dict": OrderedDict,
        }
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "backbone_name": backbone_name,
        "epoch": epoch,
        "val_acc": val_acc,
        "state_dict": backbone.state_dict(),
    }
    torch.save(ckpt, path)
    size_mb = path.stat().st_size / 1e6
    log.info(
        "[Trainer] Checkpoint guardado: %s (%.1f MB) | epoch=%d val_acc=%.4f",
        path,
        size_mb,
        epoch,
        val_acc,
    )


# ══════════════════════════════════════════════════════════════
#  Warm-up lineal del LR
# ══════════════════════════════════════════════════════════════


def _warmup_lr_scale(epoch: int, warmup_epochs: int) -> float:
    """Escala lineal del LR durante warm-up. Retorna 1.0 una vez terminado."""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / warmup_epochs)


# ══════════════════════════════════════════════════════════════
#  Bucle de entrenamiento — una época
# ══════════════════════════════════════════════════════════════


def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """
    Entrena una época completa.

    Returns:
        avg_loss — CrossEntropyLoss promedio sobre el epoch
        accuracy — accuracy de dominio [0, 1]
    """
    backbone.train()
    head.train()

    total_loss = 0.0
    correct = 0
    total = 0
    t_start = time.time()

    for batch_idx, (imgs, expert_ids, _names) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = expert_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        z = backbone(imgs)  # [B, d_model]
        logits = head(z)  # [B, N_EXPERTS_DOMAIN]
        loss = criterion(logits, labels)

        loss.backward()

        # Gradient clipping
        if TRAIN_GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()),
                TRAIN_GRAD_CLIP,
            )

        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        if batch_idx % 100 == 0:
            elapsed = time.time() - t_start
            speed = (batch_idx + 1) * loader.batch_size / elapsed if elapsed > 0 else 0
            log.info(
                "[Trainer] Epoch %d/%d | Batch %4d/%d | loss=%.4f | %.0f img/s",
                epoch + 1,
                total_epochs,
                batch_idx,
                len(loader),
                loss.item(),
                speed,
            )

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════
#  Bucle de validación
# ══════════════════════════════════════════════════════════════


def validate(
    backbone: nn.Module,
    head: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evalúa el modelo sobre el split de validación.

    Returns:
        avg_loss, accuracy
    """
    backbone.eval()
    head.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for imgs, expert_ids, _names in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = expert_ids.to(device, non_blocking=True)

            z = backbone(imgs)
            logits = head(z)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════
#  Entrenador completo
# ══════════════════════════════════════════════════════════════


def train_backbone(
    backbone: nn.Module,
    d_model: int,
    backbone_name: str,
    train_loader,
    val_loader,
    device: torch.device,
    checkpoint_path: str,
    cfg: dict,
) -> dict:
    """
    Entrena backbone + LinearHead end-to-end para clasificación de dominio.

    Args:
        backbone      — modelo entrenable (de load_trainable_backbone)
        d_model       — dimensión del CLS token del backbone
        backbone_name — nombre del backbone (para el checkpoint)
        train_loader  — DataLoader del split train (mode="embedding")
        val_loader    — DataLoader del split val (mode="embedding")
        device        — torch.device
        checkpoint_path — ruta donde guardar backbone.pth
        cfg           — dict con: epochs, lr, weight_decay, warmup_epochs

    Returns:
        dict con métricas: best_val_acc, best_epoch, train_history, val_history
    """
    from config import N_EXPERTS_DOMAIN

    epochs = cfg.get("epochs", 20)
    lr = cfg.get("lr", 3e-4)
    weight_decay = cfg.get("weight_decay", 0.01)
    warmup_epochs = cfg.get("warmup_epochs", TRAIN_WARMUP_EPOCHS)

    head = LinearHead(d_model, N_EXPERTS_DOMAIN).to(device)
    criterion = nn.CrossEntropyLoss()

    all_params = list(backbone.parameters()) + list(head.parameters())
    optimizer = AdamW(all_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=lr * 0.01
    )

    best_val_acc = 0.0
    best_epoch = 0
    train_history = []
    val_history = []

    log.info("=" * 60)
    log.info("[Trainer] Inicio entrenamiento backbone: %s", backbone_name)
    log.info(
        "[Trainer] epochs=%d | lr=%.2e | wd=%.4f | warmup=%d",
        epochs,
        lr,
        weight_decay,
        warmup_epochs,
    )
    log.info("[Trainer] checkpoint → %s", checkpoint_path)
    log.info("=" * 60)

    t_total = time.time()

    for epoch in range(epochs):
        # ── Warm-up lineal del LR ──
        if epoch < warmup_epochs:
            scale = _warmup_lr_scale(epoch, warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * scale
            log.debug("[Trainer] Warm-up epoch %d: lr=%.2e", epoch, lr * scale)

        # ── Train ──
        t_ep = time.time()
        train_loss, train_acc = train_one_epoch(
            backbone,
            head,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            epochs,
        )

        # ── Validate ──
        val_loss, val_acc = validate(backbone, head, val_loader, criterion, device)

        # ── Scheduler step (después del warm-up) ──
        if epoch >= warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed_ep = time.time() - t_ep

        log.info(
            "[Trainer] Epoch %3d/%d | train_loss=%.4f acc=%.4f | "
            "val_loss=%.4f acc=%.4f | lr=%.2e | %.1fs",
            epoch + 1,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
            elapsed_ep,
        )

        train_history.append({"epoch": epoch + 1, "loss": train_loss, "acc": train_acc})
        val_history.append({"epoch": epoch + 1, "loss": val_loss, "acc": val_acc})

        # ── Guardar mejor checkpoint ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_backbone_checkpoint(
                backbone,
                checkpoint_path,
                backbone_name,
                epoch + 1,
                val_acc,
            )
            log.info(
                "[Trainer] ★ Nuevo mejor val_acc=%.4f (epoch %d) — checkpoint guardado",
                val_acc,
                epoch + 1,
            )

    total_time = time.time() - t_total
    log.info("=" * 60)
    log.info("[Trainer] Entrenamiento completado en %.1f min", total_time / 60)
    log.info("[Trainer] Mejor val_acc=%.4f (epoch %d)", best_val_acc, best_epoch)
    log.info("=" * 60)

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_time_s": total_time,
        "train_history": train_history,
        "val_history": val_history,
    }

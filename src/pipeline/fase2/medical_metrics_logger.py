"""Centralized metrics logger for the MoE medical system.

Compatible with PyTorch DDP — all ``torchmetrics`` objects synchronize
automatically across ranks when ``.compute()`` is called at epoch end.

Supported tasks: ``multilabel``, ``multiclass``, ``binary``, ``segmentation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    MulticlassCohenKappa,
)
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import JaccardIndex


TaskType = Literal["multilabel", "multiclass", "binary", "segmentation"]


def _build_classification_metrics(
    task: TaskType,
    num_classes: int,
    include_kappa: bool = False,
) -> dict[str, Any]:
    """Return a dict of torchmetric objects for the given classification task."""
    metrics: dict[str, Any] = {}

    if task == "multilabel":
        metrics["AUC"] = MultilabelAUROC(num_labels=num_classes, average="macro")
        metrics["AUPRC"] = MultilabelAveragePrecision(
            num_labels=num_classes, average="macro"
        )
        metrics["F1"] = MultilabelF1Score(num_labels=num_classes, average="macro")
    elif task == "multiclass":
        metrics["AUC"] = MulticlassAUROC(num_classes=num_classes, average="macro")
        metrics["AUPRC"] = MulticlassAveragePrecision(
            num_classes=num_classes, average="macro"
        )
        metrics["F1"] = MulticlassF1Score(num_classes=num_classes, average="macro")
        if include_kappa:
            metrics["Kappa"] = MulticlassCohenKappa(
                num_classes=num_classes, weights="quadratic"
            )
    elif task == "binary":
        metrics["AUC"] = BinaryAUROC()
        metrics["AUPRC"] = BinaryAveragePrecision()
        metrics["F1"] = BinaryF1Score()

    return metrics


def _build_segmentation_metrics(num_classes: int) -> dict[str, Any]:
    """Return Dice and IoU metrics for segmentation tasks."""
    return {
        "Dice": DiceScore(num_classes=num_classes, average="macro"),
        "IoU": JaccardIndex(
            task="multiclass", num_classes=num_classes, average="macro"
        ),
    }


@dataclass
class MedicalMetricsLogger:
    """Accumulates and logs epoch-level metrics with DDP support.

    Args:
        task: One of ``"multilabel"``, ``"multiclass"``, ``"binary"``,
              ``"segmentation"``.
        num_classes: Number of target classes / labels / segmentation classes.
        device: Torch device where metrics tensors live (must match model device).
        include_kappa: Add Quadratic Weighted Kappa (only for ``"multiclass"``).
    """

    task: TaskType
    num_classes: int
    device: torch.device | str = "cuda"
    include_kappa: bool = False

    # — internal state (post-init) —
    _metrics: MetricCollection = field(init=False, repr=False)
    _loss_sum: float = field(init=False, default=0.0)
    _loss_count: int = field(init=False, default=0)
    _grad_norm_sum: float = field(init=False, default=0.0)
    _grad_norm_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.task == "segmentation":
            raw = _build_segmentation_metrics(self.num_classes)
        else:
            raw = _build_classification_metrics(
                self.task, self.num_classes, self.include_kappa
            )
        self._metrics = MetricCollection(raw).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss: float | torch.Tensor | None = None,
    ) -> None:
        """Accumulate predictions for the current epoch.

        Call once per batch inside the training / validation loop.
        ``preds`` and ``targets`` shapes depend on the task:

        - multilabel: ``(B, C)`` floats (logits or probabilities) / ``(B, C)`` ints
        - multiclass: ``(B, C)`` floats / ``(B,)`` ints
        - binary: ``(B,)`` or ``(B, 1)`` floats / same shape ints
        - segmentation: ``(B, C, *spatial)`` / ``(B, *spatial)`` ints
        """
        self._metrics.update(preds, targets)
        if loss is not None:
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            self._loss_sum += loss_val
            self._loss_count += 1

    def update_grad_norm(self, model: nn.Module) -> None:
        """Compute and accumulate the L2 gradient norm of ``model``.

        Call **after** ``loss.backward()`` and **before** ``optimizer.step()``
        (or after, depending on whether the optimizer zeroes grads).
        """
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        self._grad_norm_sum += total_norm_sq**0.5
        self._grad_norm_count += 1

    def compute_and_log(
        self,
        epoch: int,
        phase: str,
        rank: int = 0,
    ) -> dict[str, float]:
        """Compute final epoch metrics, print on rank 0, return dict.

        In DDP, ``torchmetrics`` automatically all-reduces internal state
        when ``.compute()`` is called, so every rank gets the same values.
        Only rank 0 prints the formatted line.
        """
        results: dict[str, float] = {}

        # Loss (local average — not synced across ranks on purpose;
        # each rank sees roughly the same loss with shuffled data).
        if self._loss_count > 0:
            results["Loss"] = self._loss_sum / self._loss_count

        # torchmetrics (synced across ranks)
        computed = self._metrics.compute()
        for name, value in computed.items():
            results[name] = value.item()

        # Gradient norm
        if self._grad_norm_count > 0:
            results["GradNorm"] = self._grad_norm_sum / self._grad_norm_count

        # Pretty-print on rank 0
        if rank == 0:
            phase_tag = phase.upper().rjust(5)
            parts: list[str] = []
            for key in (
                "Loss",
                "AUC",
                "AUPRC",
                "F1",
                "Kappa",
                "Dice",
                "IoU",
                "GradNorm",
            ):
                if key in results:
                    parts.append(f"{key}: {results[key]:.4f}")
            if "GradNorm" not in results and phase.upper() == "VAL":
                parts.append("GradNorm: N/A")
            line = " | ".join(parts)
            print(f"[Epoch {epoch:02d} | {phase_tag}] {line}")

        return results

    def reset(self) -> None:
        """Reset all metric accumulators. Call between epochs."""
        self._metrics.reset()
        self._loss_sum = 0.0
        self._loss_count = 0
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0


# ======================================================================
# USAGE EXAMPLE (DDP training script)
# ======================================================================
#
# """
# import os
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from medical_metrics_logger import MedicalMetricsLogger
#
#
# def train_one_epoch(model, dataloader, optimizer, criterion, logger, epoch, rank):
#     model.train()
#     logger.reset()
#
#     for images, labels in dataloader:
#         images, labels = images.cuda(), labels.cuda()
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#
#         # Accumulate grad norm AFTER backward, BEFORE step
#         logger.update_grad_norm(model.module)
#
#         optimizer.step()
#
#         # Accumulate predictions (use sigmoid/softmax as needed)
#         with torch.no_grad():
#             preds = torch.sigmoid(outputs)  # multilabel / binary
#             logger.update(preds, labels, loss=loss)
#
#     # Compute & log at epoch end — torchmetrics syncs across ranks
#     train_metrics = logger.compute_and_log(epoch, "train", rank=rank)
#     return train_metrics
#
#
# @torch.no_grad()
# def validate(model, dataloader, criterion, logger, epoch, rank):
#     model.eval()
#     logger.reset()
#
#     for images, labels in dataloader:
#         images, labels = images.cuda(), labels.cuda()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         preds = torch.sigmoid(outputs)
#         logger.update(preds, labels, loss=loss)
#
#     val_metrics = logger.compute_and_log(epoch, "val", rank=rank)
#     return val_metrics
#
#
# def main():
#     dist.init_process_group("nccl")
#     rank = dist.get_rank()
#     torch.cuda.set_device(rank)
#
#     model = MyModel().cuda()
#     model = DDP(model, device_ids=[rank])
#
#     # Example: Expert 1 — ChestXray14, multilabel, 14 classes
#     train_logger = MedicalMetricsLogger(
#         task="multilabel",
#         num_classes=14,
#         device=torch.device(f"cuda:{rank}"),
#     )
#     val_logger = MedicalMetricsLogger(
#         task="multilabel",
#         num_classes=14,
#         device=torch.device(f"cuda:{rank}"),
#     )
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
#     criterion = torch.nn.BCEWithLogitsLoss()
#
#     for epoch in range(1, 51):
#         train_metrics = train_one_epoch(
#             model, train_loader, optimizer, criterion, train_logger, epoch, rank
#         )
#         val_metrics = validate(
#             model, val_loader, criterion, val_logger, epoch, rank
#         )
#
#     dist.destroy_process_group()
#
#
# if __name__ == "__main__":
#     main()
# """

"""
test_evaluator.py
-----------------
Evaluador batch para el Paso 9 del proyecto MoE.
Itera sobre los test sets de los 5 expertos de dominio,
calcula métricas F1 macro, AUC-ROC, routing accuracy y load balance,
y guarda artefactos JSON + matrices de confusión.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.pipeline.fase6.fase6_config import (
    RESULTS_DIR,
    FIGURES_DIR,
    EVAL_BATCH_SIZE,
    EVAL_NUM_WORKERS,
    EXPERT_NAMES,
    N_EXPERTS_DOMAIN,
    F1_THRESHOLD_2D,
    F1_THRESHOLD_3D,
    F1_FULL_2D,
    F1_FULL_3D,
    ROUTING_ACCURACY_MIN,
    LOAD_BALANCE_MAX_RATIO,
)

logger = logging.getLogger(__name__)


class TestEvaluator:
    """
    Evaluador batch para el sistema MoE.

    Args:
        inference_engine: InferenceEngine instance (ya en eval mode)
        device: torch.device
        dry_run: si True, solo valida setup sin iterar datos reales
    """

    def __init__(
        self,
        inference_engine,
        device: Optional[torch.device] = None,
        dry_run: bool = False,
    ):
        self.engine = inference_engine
        self.device = device or torch.device("cpu")
        self.dry_run = dry_run

        # Crear directorios de salida
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    def evaluate_expert(
        self,
        expert_idx: int,
        dataloader: DataLoader,
        expert_name: str,
        num_classes: int,
        is_multilabel: bool = False,
    ) -> dict:
        """
        Evalúa un experto específico sobre su test set.

        Returns dict con métricas:
            f1_macro, auc_roc, routing_accuracy, n_samples,
            predicted_expert_counts (dict: expert_id → count),
            confusion_matrix (list of lists)
        """
        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Skipping real evaluation for expert {expert_idx} ({expert_name})"
            )
            return self._dry_run_metrics(expert_idx, expert_name, num_classes)

        all_labels = []
        all_preds = []
        all_probs = []
        all_expert_ids_used = []

        logger.info(
            f"Evaluating expert {expert_idx} ({expert_name}) — {len(dataloader)} batches"
        )

        for batch_idx, batch in enumerate(dataloader):
            # Datasets devuelven (x, label) o (x, label, meta)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, labels = batch[0], batch[1]
            else:
                logger.warning(f"Unexpected batch format at idx {batch_idx}, skipping")
                continue

            x = x.to(self.device)
            labels = (
                labels.cpu().numpy()
                if isinstance(labels, torch.Tensor)
                else np.array(labels)
            )

            result = self.engine(x)

            # Recopilar expert_ids usados
            all_expert_ids_used.extend(result["expert_ids"])

            # Extraer logits y predicciones
            for i, logit in enumerate(result["logits"]):
                if is_multilabel:
                    prob = torch.sigmoid(logit).squeeze(0).cpu().numpy()
                    pred = (prob > 0.5).astype(int)
                else:
                    prob = torch.softmax(logit, dim=-1).squeeze(0).cpu().numpy()
                    pred = int(prob.argmax())

                all_probs.append(prob)
                all_preds.append(pred)

            all_labels.extend(
                labels.tolist() if hasattr(labels, "tolist") else list(labels)
            )

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # Métricas
        f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

        try:
            if is_multilabel:
                auc = float(
                    roc_auc_score(
                        all_labels, all_probs, average="macro", multi_class="ovr"
                    )
                )
            elif num_classes == 2:
                auc = float(roc_auc_score(all_labels, all_probs[:, 1]))
            else:
                auc = float(
                    roc_auc_score(
                        all_labels, all_probs, average="macro", multi_class="ovr"
                    )
                )
        except Exception as e:
            logger.warning(f"AUC-ROC computation failed: {e}")
            auc = float("nan")

        # Routing accuracy: fracción de muestras ruteadas al experto correcto
        routing_acc = float(
            sum(1 for eid in all_expert_ids_used if eid == expert_idx)
            / max(len(all_expert_ids_used), 1)
        )

        # Load balance: contar asignaciones por experto
        expert_counts = {}
        for eid in all_expert_ids_used:
            expert_counts[str(eid)] = expert_counts.get(str(eid), 0) + 1

        # Confusion matrix (solo para clasificación single-label)
        cm = (
            confusion_matrix(all_labels, all_preds).tolist()
            if not is_multilabel
            else []
        )

        metrics = {
            "expert_idx": expert_idx,
            "expert_name": expert_name,
            "n_samples": int(len(all_labels)),
            "f1_macro": f1,
            "auc_roc": auc,
            "routing_accuracy": routing_acc,
            "predicted_expert_counts": expert_counts,
            "confusion_matrix": cm,
            "is_multilabel": is_multilabel,
        }

        # Guardar JSON por experto
        out_path = Path(RESULTS_DIR) / f"test_metrics_expert_{expert_idx}.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics for expert {expert_idx} → {out_path}")

        return metrics

    def _dry_run_metrics(
        self, expert_idx: int, expert_name: str, num_classes: int
    ) -> dict:
        """Returns dummy metrics for dry-run validation."""
        metrics = {
            "expert_idx": expert_idx,
            "expert_name": expert_name,
            "n_samples": 0,
            "f1_macro": 0.0,
            "auc_roc": 0.0,
            "routing_accuracy": 0.0,
            "predicted_expert_counts": {},
            "confusion_matrix": [],
            "is_multilabel": expert_idx == 0,  # chest is multilabel
            "dry_run": True,
        }
        out_path = Path(RESULTS_DIR) / f"test_metrics_expert_{expert_idx}.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(
            f"[DRY-RUN] Saved dummy metrics for expert {expert_idx} → {out_path}"
        )
        return metrics

    def compute_load_balance(self, all_expert_metrics: list) -> dict:
        """
        Calcula load balance global: max_count / min_count ratio.
        Objetivo: ratio < LOAD_BALANCE_MAX_RATIO (1.30).
        """
        global_counts = {str(i): 0 for i in range(N_EXPERTS_DOMAIN)}

        for metrics in all_expert_metrics:
            for eid_str, cnt in metrics.get("predicted_expert_counts", {}).items():
                global_counts[eid_str] = global_counts.get(eid_str, 0) + cnt

        counts = [v for v in global_counts.values() if v > 0]
        if len(counts) < 2:
            ratio = 1.0
        else:
            ratio = float(max(counts) / min(counts))

        lb_report = {
            "global_expert_counts": global_counts,
            "max_min_ratio": ratio,
            "threshold": LOAD_BALANCE_MAX_RATIO,
            "passes": ratio < LOAD_BALANCE_MAX_RATIO,
        }

        out_path = Path(RESULTS_DIR) / "load_balance_test.json"
        with open(out_path, "w") as f:
            json.dump(lb_report, f, indent=2)
        logger.info(
            f"Load balance ratio: {ratio:.3f} (threshold: {LOAD_BALANCE_MAX_RATIO})"
        )

        return lb_report

    def compute_summary(self, all_expert_metrics: list, lb_report: dict) -> dict:
        """
        Calcula y guarda el resumen global de métricas.
        """
        experts_2d = [0, 1, 2]  # chest, isic, oa_knee
        experts_3d = [3, 4]  # luna, pancreas

        f1_2d = [
            m["f1_macro"] for m in all_expert_metrics if m["expert_idx"] in experts_2d
        ]
        f1_3d = [
            m["f1_macro"] for m in all_expert_metrics if m["expert_idx"] in experts_3d
        ]
        routing_accs = [m["routing_accuracy"] for m in all_expert_metrics]

        summary = {
            "f1_macro_2d_mean": float(np.mean(f1_2d)) if f1_2d else 0.0,
            "f1_macro_3d_mean": float(np.mean(f1_3d)) if f1_3d else 0.0,
            "routing_accuracy_mean": float(np.mean(routing_accs))
            if routing_accs
            else 0.0,
            "load_balance_ratio": lb_report["max_min_ratio"],
            "passes_2d_acceptable": float(np.mean(f1_2d)) >= F1_THRESHOLD_2D
            if f1_2d
            else False,
            "passes_2d_full": float(np.mean(f1_2d)) >= F1_FULL_2D if f1_2d else False,
            "passes_3d_acceptable": float(np.mean(f1_3d)) >= F1_THRESHOLD_3D
            if f1_3d
            else False,
            "passes_3d_full": float(np.mean(f1_3d)) >= F1_FULL_3D if f1_3d else False,
            "passes_routing": float(np.mean(routing_accs)) >= ROUTING_ACCURACY_MIN
            if routing_accs
            else False,
            "passes_load_balance": lb_report["passes"],
            "per_expert": all_expert_metrics,
        }

        out_path = Path(RESULTS_DIR) / "test_metrics_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved → {out_path}")

        return summary

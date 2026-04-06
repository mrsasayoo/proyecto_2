"""
ood_detector.py
---------------
Detección de Out-of-Distribution (OOD) para el sistema MoE.

Flujo:
  1. Calibración: usar val set in-distribution para encontrar percentil 95 de entropía
  2. OOD AUROC: evaluar umbral calibrado sobre test set con muestras OOD conocidas
  3. Guardar umbral en ENTROPY_THRESHOLD_PATH (pickle) para uso en InferenceEngine
"""

import json
import logging
import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.pipeline.fase6.fase6_config import (
    ENTROPY_THRESHOLD_PATH,
    OOD_ENTROPY_PERCENTILE,
    OOD_AUROC_MIN,
    RESULTS_DIR,
    N_EXPERTS_DOMAIN,
)

logger = logging.getLogger(__name__)


class OODDetector:
    """
    Calibra y evalúa el detector OOD basado en entropía del router.

    Args:
        inference_engine: InferenceEngine (sin umbral fijo o con umbral inicial)
        device: torch.device
        dry_run: si True, usa umbral por defecto sin iterar datos
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

    def calibrate_threshold(self, val_dataloader: DataLoader) -> float:
        """
        Calibra el umbral de entropía usando el percentil OOD_ENTROPY_PERCENTILE
        del val set in-distribution.

        Returns: entropy_threshold (float)
        """
        if self.dry_run:
            threshold = math.log(N_EXPERTS_DOMAIN) / 2.0
            logger.info(f"[DRY-RUN] Using default entropy threshold: {threshold:.4f}")
            self._save_threshold(threshold)
            return threshold

        logger.info("Calibrating OOD entropy threshold on val set...")
        all_entropies = []

        for batch in val_dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)
            result = self.engine(x)
            entropies = result["entropy"].cpu().numpy()
            all_entropies.extend(entropies.tolist())

        threshold = float(np.percentile(all_entropies, OOD_ENTROPY_PERCENTILE))
        logger.info(
            f"Entropy threshold calibrated at {OOD_ENTROPY_PERCENTILE}th percentile: {threshold:.4f} "
            f"(n={len(all_entropies)} val samples)"
        )

        self._save_threshold(threshold)
        return threshold

    def _save_threshold(self, threshold: float) -> None:
        path = Path(ENTROPY_THRESHOLD_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(threshold, f)
        logger.info(f"Entropy threshold saved → {path}")

    def compute_ood_auroc(
        self,
        in_dist_dataloader: DataLoader,
        ood_dataloader: DataLoader,
    ) -> dict:
        """
        Computa AUROC OOD: in-distribution (label=0) vs OOD (label=1).

        Args:
            in_dist_dataloader: DataLoader con muestras in-distribution
            ood_dataloader: DataLoader con muestras OOD (CAE dataset como proxy)

        Returns dict con auroc, threshold, passes
        """
        if self.dry_run:
            result = {
                "ood_auroc": 0.0,
                "entropy_threshold": self.engine.entropy_threshold,
                "n_in_dist": 0,
                "n_ood": 0,
                "passes": False,
                "threshold_min": OOD_AUROC_MIN,
                "dry_run": True,
            }
            self._save_report(result)
            return result

        logger.info("Computing OOD AUROC...")
        entropies = []
        labels = []  # 0 = in-dist, 1 = OOD

        # In-distribution samples
        for batch in in_dist_dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
            result = self.engine(x)
            entropies.extend(result["entropy"].cpu().numpy().tolist())
            labels.extend([0] * x.shape[0])

        n_in = len([l for l in labels if l == 0])

        # OOD samples
        for batch in ood_dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
            result = self.engine(x)
            entropies.extend(result["entropy"].cpu().numpy().tolist())
            labels.extend([1] * x.shape[0])

        n_ood = len([l for l in labels if l == 1])

        entropies = np.array(entropies)
        labels = np.array(labels)

        try:
            auroc = float(roc_auc_score(labels, entropies))
        except Exception as e:
            logger.warning(f"OOD AUROC computation failed: {e}")
            auroc = float("nan")

        report = {
            "ood_auroc": auroc,
            "entropy_threshold": float(self.engine.entropy_threshold),
            "n_in_dist": n_in,
            "n_ood": n_ood,
            "passes": auroc >= OOD_AUROC_MIN,
            "threshold_min": OOD_AUROC_MIN,
        }

        logger.info(
            f"OOD AUROC: {auroc:.4f} (threshold: {OOD_AUROC_MIN}) → {'PASS' if report['passes'] else 'FAIL'}"
        )
        self._save_report(report)
        return report

    def _save_report(self, report: dict) -> None:
        out_path = Path(RESULTS_DIR) / "ood_auroc_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"OOD report saved → {out_path}")

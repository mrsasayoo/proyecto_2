"""
ood_detector.py
---------------
Detección de Out-of-Distribution (OOD) para el sistema MoE.

Flujo:
  1. Calibración: usar val set in-distribution para encontrar percentil 95 de entropía
  2. Calibración CAE: calcular θ_leve y θ_OOD a partir del MSE de reconstrucción
  3. OOD AUROC: evaluar umbral calibrado sobre test set con muestras OOD conocidas
  4. Guardar umbrales en OOD_THRESHOLDS_JSON_PATH (JSON) para uso en InferenceEngine

Umbrales CAE (feedback loop de 3 pasos):
  - θ_leve  = percentil 50 del MSE in-distribution → debajo = imagen limpia
  - θ_OOD   = percentil 95 del MSE in-distribution → encima = OOD absoluto
  - Entre ambos: denoising con re-routing
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
    OOD_THRESHOLDS_JSON_PATH,
    OOD_ENTROPY_PERCENTILE,
    CAE_MSE_PERCENTILE_LEVE,
    CAE_MSE_PERCENTILE_OOD,
    OOD_AUROC_MIN,
    RESULTS_DIR,
    N_EXPERTS_DOMAIN,
    CAE_EXPERT_IDX,
)

logger = logging.getLogger(__name__)


class OODDetector:
    """
    Calibra y evalúa el detector OOD basado en entropía del router y
    umbrales de reconstrucción del CAE.

    Calibra tres umbrales:
      - entropy_threshold: percentil 95 de entropía del router sobre val set
      - cae_theta_leve: percentil 50 del MSE de reconstrucción (in-dist)
      - cae_theta_ood: percentil 95 del MSE de reconstrucción (in-dist)

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
        """Guarda el umbral de entropía en pickle (backward compat) y en JSON."""
        # Pickle (backward compat)
        pkl_path = Path(ENTROPY_THRESHOLD_PATH)
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(threshold, f)
        logger.info(f"Entropy threshold saved (pkl) → {pkl_path}")

        # JSON consolidado — actualiza sin sobreescribir claves CAE existentes
        self._update_json_thresholds({"entropy_threshold": threshold})

    # ------------------------------------------------------------------
    # CAE threshold calibration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def calibrate_cae_thresholds(
        self,
        val_dataloader: DataLoader,
        cae_model: Optional[torch.nn.Module] = None,
    ) -> dict[str, float]:
        """
        Calibra los umbrales del CAE (θ_leve y θ_OOD) usando la distribución
        de MSE de reconstrucción sobre el val set in-distribution.

        Si no se pasa ``cae_model`` explícitamente, se extrae del
        MoESystem a través de ``self.engine.moe.experts[CAE_EXPERT_IDX]``.

        Percentiles usados (definidos en fase6_config):
          - θ_leve = percentil 50  (CAE_MSE_PERCENTILE_LEVE)
          - θ_OOD  = percentil 95  (CAE_MSE_PERCENTILE_OOD)

        Los umbrales se guardan en OOD_THRESHOLDS_JSON_PATH bajo las claves
        ``"cae_theta_leve"`` y ``"cae_theta_ood"``.

        Args:
            val_dataloader: DataLoader con muestras in-distribution del val set.
            cae_model: ConvAutoEncoder ya cargado. Si None, se usa el experto
                CAE del MoESystem.

        Returns:
            dict con ``"cae_theta_leve"`` y ``"cae_theta_ood"``.
        """
        if cae_model is None:
            cae_model = self.engine.moe.experts[CAE_EXPERT_IDX]

        # Asegurar FP32 y eval
        cae_model = cae_model.float().to(self.device)
        cae_model.eval()

        if self.dry_run:
            # Valores placeholder para dry-run
            thresholds = {
                "cae_theta_leve": 0.01,
                "cae_theta_ood": 0.05,
            }
            logger.info(
                "[DRY-RUN] Using default CAE thresholds: "
                f"θ_leve={thresholds['cae_theta_leve']:.4f}, "
                f"θ_OOD={thresholds['cae_theta_ood']:.4f}"
            )
            self._update_json_thresholds(thresholds)
            return thresholds

        logger.info("Calibrating CAE reconstruction thresholds on val set...")
        all_mse: list[float] = []

        for batch in val_dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device).float()  # FP32

            # reconstruction_error devuelve tensor [B] con MSE por sample
            mse_per_sample = cae_model.reconstruction_error(x)
            all_mse.extend(mse_per_sample.cpu().tolist())

        all_mse_arr = np.array(all_mse)

        theta_leve = float(np.percentile(all_mse_arr, CAE_MSE_PERCENTILE_LEVE))
        theta_ood = float(np.percentile(all_mse_arr, CAE_MSE_PERCENTILE_OOD))

        logger.info(
            f"CAE thresholds calibrated (n={len(all_mse)} samples): "
            f"θ_leve (p{CAE_MSE_PERCENTILE_LEVE})={theta_leve:.6f}, "
            f"θ_OOD (p{CAE_MSE_PERCENTILE_OOD})={theta_ood:.6f}"
        )
        logger.info(
            f"MSE distribution — min={all_mse_arr.min():.6f}, "
            f"median={np.median(all_mse_arr):.6f}, "
            f"mean={all_mse_arr.mean():.6f}, "
            f"max={all_mse_arr.max():.6f}"
        )

        thresholds = {
            "cae_theta_leve": theta_leve,
            "cae_theta_ood": theta_ood,
        }
        self._update_json_thresholds(thresholds)
        return thresholds

    # ------------------------------------------------------------------
    # JSON persistence (consolidado)
    # ------------------------------------------------------------------

    def _update_json_thresholds(self, new_values: dict[str, float]) -> None:
        """
        Actualiza (merge) el archivo JSON de umbrales OOD con ``new_values``.

        Si el archivo ya existe, carga los valores previos y les hace merge
        con los nuevos, preservando claves no tocadas. Si no existe, lo crea.
        """
        json_path = Path(OOD_THRESHOLDS_JSON_PATH)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        existing: dict = {}
        if json_path.exists():
            with open(json_path, "r") as f:
                existing = json.load(f)

        existing.update(new_values)

        with open(json_path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info(
            f"OOD thresholds JSON updated → {json_path} (keys: {list(existing.keys())})"
        )

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

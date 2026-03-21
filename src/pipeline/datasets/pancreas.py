"""
Pancreas PANORAMA / PDAC — Experto 4: clasificación binaria de volúmenes CT abdominales.

Hallazgos implementados:
  H1 → etiquetas en repo GitHub separado: PanoramaLabelLoader con cruce
        de case_ids y fijación del hash del commit para reproducibilidad
  H2 → páncreas ocupa ~1% del volumen: PancreasROIExtractor con 3 estrategias;
        resize naïve destruye la señal diagnóstica
  H3 → clip HU [-100, 400] para abdomen (NO usar [-1000, 400] de LUNA16);
        z-score por volumen para bias multicéntrico (Radboudumc/MSD/NIH)
  Item-3  → lectura .nii.gz con SimpleITK/nibabel
  Item-6  → gradient checkpointing batch_size 1–2 con 12 GB VRAM
  Item-7  → FocalLoss(alpha=0.75, gamma=2) — desbalance ~2.3:1 a 3:1
  Item-8  → k-fold CV (k=5) dado el tamaño limitado (~281 volúmenes)
  Item-9  → auditoría de fuentes (Radboudumc/MSD/NIH) + z-score por volumen
"""

import re
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import SimpleITK as sitk

from config import EXPERT_IDS, HU_ABDOMEN_CLIP, HU_LUNG_CLIP
from preprocessing import normalize_hu, resize_volume_3d, volume_to_vit_input

log = logging.getLogger("fase0")


class PanoramaLabelLoader:
    """
    H1 — Carga las etiquetas PDAC del repositorio GitHub separado.

    PROBLEMA CRÍTICO: el ZIP de Zenodo solo contiene volúmenes CT sin etiquetas.
    Las etiquetas binarias (PDAC+ / PDAC−) están en:
        https://github.com/DIAGNijmegen/panorama_labels

    Procedimiento obligatorio antes de cualquier entrenamiento:
        git clone https://github.com/DIAGNijmegen/panorama_labels.git
        cd panorama_labels
        git checkout <hash_del_commit>   # fijar versión exacta
    """

    LABELS_REPO_URL = "https://github.com/DIAGNijmegen/panorama_labels"

    @staticmethod
    def load_labels(labels_repo_dir: str,
                    expected_commit: str = None) -> dict:
        """
        H1/Item-1 — Carga las etiquetas del repositorio clonado.

        Returns:
            dict {case_id: label_int}  — label_int ∈ {0, 1}
            0 = PDAC negativo, 1 = PDAC positivo
        """
        repo_dir = Path(labels_repo_dir)
        if not repo_dir.exists():
            log.error(
                f"[Pancreas/H1] Repositorio de etiquetas NO encontrado: '{repo_dir}'.\n"
                f"    El ZIP de Zenodo NO incluye etiquetas. Pasos obligatorios:\n"
                f"        git clone {PanoramaLabelLoader.LABELS_REPO_URL}\n"
                f"        cd panorama_labels\n"
                f"        git checkout <hash_del_commit>   # fija reproducibilidad\n"
                f"    Sin este paso el dataset no es utilizable para clasificación."
            )
            return {}

        # Verificar hash del commit actual
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_dir), capture_output=True, text=True, timeout=10
            )
            current_hash = result.stdout.strip()
            if expected_commit:
                if current_hash.startswith(expected_commit):
                    log.info(f"[Pancreas/H1] Commit verificado: {current_hash[:12]} ✓")
                else:
                    log.error(
                        f"[Pancreas/H1] ¡Hash incorrecto! "
                        f"Esperado: {expected_commit[:12]} | "
                        f"Actual: {current_hash[:12]}. "
                        f"Ejecuta: cd {repo_dir} && git checkout {expected_commit}"
                    )
            else:
                log.warning(
                    f"[Pancreas/H1] Commit actual: {current_hash[:12]} — "
                    f"sin expected_commit especificado. "
                    f"Registra este hash en tu README para reproducibilidad: "
                    f"expected_commit='{current_hash[:12]}'"
                )
        except Exception as e:
            log.warning(f"[Pancreas/H1] No se pudo verificar hash del commit: {e}")

        # Buscar archivo de etiquetas (JSON o CSV)
        labels = {}
        label_files = list(repo_dir.glob("*.json")) + list(repo_dir.glob("*.csv"))

        if not label_files:
            log.error(f"[Pancreas/H1] Sin archivos .json/.csv en '{repo_dir}'. "
                      f"¿El repositorio está correctamente clonado?")
            return {}

        log.info(f"[Pancreas/H1] Archivos de etiquetas encontrados: "
                 f"{[f.name for f in label_files]}")

        for lf in label_files:
            try:
                if lf.suffix == ".json":
                    import json as _json
                    with open(lf) as f:
                        data = _json.load(f)
                    if isinstance(data, dict):
                        for case_id, info in data.items():
                            if isinstance(info, dict):
                                label = int(info.get("label",
                                             info.get("has_pdac", 0)))
                            else:
                                label = int(info)
                            labels[str(case_id)] = label
                elif lf.suffix == ".csv":
                    df = pd.read_csv(lf)
                    for col in ["case_id", "id", "case"]:
                        if col in df.columns:
                            id_col = col
                            break
                    else:
                        log.warning(f"[Pancreas/H1] CSV '{lf.name}' sin columna "
                                    f"case_id/id/case reconocida.")
                        continue
                    for lbl_col in ["label", "has_pdac", "pdac", "class"]:
                        if lbl_col in df.columns:
                            break
                    else:
                        log.warning(f"[Pancreas/H1] CSV '{lf.name}' sin columna "
                                    f"label/has_pdac/pdac/class reconocida.")
                        continue
                    for _, row in df.iterrows():
                        case_id = str(row[id_col])
                        label   = int(bool(row[lbl_col]))
                        labels[case_id] = label
            except Exception as e:
                log.warning(f"[Pancreas/H1] Error leyendo '{lf}': {e}")

        n_pos = sum(1 for v in labels.values() if v == 1)
        n_neg = sum(1 for v in labels.values() if v == 0)
        log.info(
            f"[Pancreas/H1] Etiquetas cargadas: {len(labels):,} casos | "
            f"PDAC+ (positivo): {n_pos:,} | "
            f"PDAC− (negativo): {n_neg:,} | "
            f"ratio neg/pos: {n_neg/max(n_pos,1):.1f}:1"
        )
        return labels

    @staticmethod
    def cross_match(labels: dict,
                    nii_paths: list,
                    require_both: bool = True) -> list:
        """
        H1/Item-1+2 — Cruza los case_ids del repositorio con los .nii.gz del ZIP.

        FIX — sufijo nnU-Net: los archivos del ZIP tienen el canal codificado
        al final, e.g. "100298_00001_0000.nii.gz". Se normaliza eliminando
        el sufijo _XXXX (4 digitos) antes del match.

        Returns:
            lista de (nii_path, label_int) — solo casos con etiqueta confirmada
        """
        valid_pairs   = []
        no_label      = []
        no_volume     = []

        normalized_index: dict = {}
        for nii_path in nii_paths:
            stem = Path(nii_path).name
            for ext in [".nii.gz", ".nii"]:
                if stem.endswith(ext):
                    stem = stem[:-len(ext)]
                    break
            normalized = re.sub(r'_\d{4}$', '', stem)
            if normalized not in normalized_index:
                normalized_index[normalized] = str(nii_path)

        for norm_id, nii_path in normalized_index.items():
            label = labels.get(norm_id)
            if label is not None:
                valid_pairs.append((nii_path, label))
            else:
                no_label.append(norm_id)

        for case_id in labels:
            if case_id not in normalized_index:
                no_volume.append(case_id)

        n_pos = sum(1 for _, l in valid_pairs if l == 1)
        n_neg = sum(1 for _, l in valid_pairs if l == 0)

        log.info(
            f"[Pancreas/H1] Cross-match completado:\n"
            f"    Pares válidos (volumen + etiqueta) : {len(valid_pairs):,}\n"
            f"    PDAC+ (positivo)                  : {n_pos:,}\n"
            f"    PDAC− (negativo)                  : {n_neg:,}\n"
            f"    Ratio neg/pos                     : {n_neg/max(n_pos,1):.1f}:1\n"
            f"    Volúmenes sin etiqueta             : {len(no_label):,} "
            f"{'(excluidos)' if require_both else '(incluidos sin label)'}\n"
            f"    Etiquetas sin volumen en disco     : {len(no_volume):,}"
        )
        if no_label:
            log.warning(f"[Pancreas/H1] {len(no_label)} volúmenes sin etiqueta "
                        f"— primeros 5: {no_label[:5]}. "
                        f"¿El repositorio está actualizado?")
        if no_volume:
            log.warning(f"[Pancreas/H1] {len(no_volume)} etiquetas sin volumen en disco "
                        f"— primeros 5: {no_volume[:5]}. "
                        f"¿Faltan batches del ZIP de Zenodo?")
        if len(valid_pairs) < 50:
            log.warning(f"[Pancreas/H1] Solo {len(valid_pairs)} pares válidos. "
                        f"Dataset muy pequeño — usar k-fold CV (k=5).")

        return valid_pairs


class PancreasROIExtractor:
    """
    H2 — Extrae la región de interés del páncreas antes del resize.

    PROBLEMA CRÍTICO: el páncreas ocupa ~0.5–2% del volumen CT abdominal.
    Con resize naïve a 64×64×64: el páncreas queda en ~5 vóxeles → señal destruida.

    Tres estrategias:
      Opción A — Resize completo (SOLO para FASE 0 / routing)
      Opción B — Recorte de abdomen superior Z[120:220] [RECOMENDADA para FASE 2]
      Opción C — Segmentar con nnU-Net → recortar ROI [SOTA, requiere nnU-Net externo]
    """

    PANCREAS_Z_MIN = 120
    PANCREAS_Z_MAX = 220

    @staticmethod
    def extract_option_a(volume: np.ndarray,
                         target: tuple = (64, 64, 64)) -> np.ndarray:
        """H2/Opción A — Resize completo. SOLO para FASE 0 (embedding/routing)."""
        t = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        t = nn.functional.interpolate(t, size=target,
                                       mode="trilinear", align_corners=False)
        return t.squeeze().numpy()

    @staticmethod
    def extract_option_b(volume: np.ndarray,
                         z_min: int = None,
                         z_max: int = None,
                         target: tuple = (64, 64, 64)) -> np.ndarray:
        """H2/Opción B — Recorte Z fijo + resize. Mejora ratio páncreas/volumen ~3×."""
        z_min = z_min if z_min is not None else PancreasROIExtractor.PANCREAS_Z_MIN
        z_max = z_max if z_max is not None else PancreasROIExtractor.PANCREAS_Z_MAX
        d = volume.shape[0]
        z_min = max(0, min(z_min, d - 1))
        z_max = max(z_min + 1, min(z_max, d))
        cropped = volume[z_min:z_max, :, :]
        t = torch.from_numpy(cropped).float().unsqueeze(0).unsqueeze(0)
        t = nn.functional.interpolate(t, size=target,
                                       mode="trilinear", align_corners=False)
        return t.squeeze().numpy()

    @staticmethod
    def verify_pancreas_z_range(nii_paths: list,
                                 segmentation_paths: list = None) -> None:
        """H2 — Verifica que el rango Z predeterminado cubre el páncreas."""
        import random
        n_check = min(5, len(nii_paths))
        log.info(f"[Pancreas/H2] Verificando rango Z en {n_check} volúmenes...")

        for nii_path in random.sample(nii_paths, n_check):
            try:
                image = sitk.ReadImage(str(nii_path))
                arr   = sitk.GetArrayFromImage(image)
                d, h, w = arr.shape
                spacing = image.GetSpacing()
                log.info(f"    {Path(nii_path).name}: shape=[{d},{h},{w}] | "
                         f"spacing={spacing[2]:.2f}×{spacing[1]:.2f}×{spacing[0]:.2f} mm")
                if d < PancreasROIExtractor.PANCREAS_Z_MAX:
                    log.warning(f"    ⚠ Volumen con solo {d} slices — "
                                f"PANCREAS_Z_MAX={PancreasROIExtractor.PANCREAS_Z_MAX} "
                                f"excede la dimensión. Ajustar PancreasROIExtractor.PANCREAS_Z_MAX.")
            except Exception as e:
                log.warning(f"    Error leyendo '{nii_path}': {e}")

        if segmentation_paths:
            log.info("[Pancreas/H2] Calculando rango Z real del páncreas desde segmentaciones...")
            z_mins, z_maxs = [], []
            for seg_path in random.sample(segmentation_paths,
                                          min(5, len(segmentation_paths))):
                try:
                    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))
                    pancreas_z = np.where(seg > 0)[0]
                    if len(pancreas_z):
                        z_mins.append(int(pancreas_z.min()))
                        z_maxs.append(int(pancreas_z.max()))
                        log.info(f"    {Path(seg_path).name}: "
                                 f"páncreas Z=[{pancreas_z.min()},{pancreas_z.max()}]")
                except Exception as e:
                    log.warning(f"    Error leyendo segmentación '{seg_path}': {e}")
            if z_mins:
                log.info(
                    f"[Pancreas/H2] Rango Z del páncreas en muestra:\n"
                    f"    Z mínimo: {min(z_mins)} (actual PANCREAS_Z_MIN={PancreasROIExtractor.PANCREAS_Z_MIN})\n"
                    f"    Z máximo: {max(z_maxs)} (actual PANCREAS_Z_MAX={PancreasROIExtractor.PANCREAS_Z_MAX})\n"
                    f"    ⚠ Si los valores difieren, actualizar PancreasROIExtractor.PANCREAS_Z_MIN/MAX."
                )
        else:
            log.info("[Pancreas/H2] Sin segmentaciones disponibles — "
                     "verificar rango Z manualmente con un visualizador DICOM/NIfTI.")


class PancreasDataset(Dataset):
    """
    Pancreatic Cancer CT 3D (PANORAMA / PDAC) — Clasificación binaria.

    Dos modos:
      mode="embedding" → FASE 0: (img_2d, expert_id=4, case_stem)
      mode="expert"    → FASE 2: (volume_3d, label_binary, case_stem)
    """

    def __init__(self, valid_pairs: list,
                 mode: str = "embedding",
                 roi_strategy: str = "B",
                 z_score_per_volume: bool = True):
        assert mode in ("embedding", "expert"), \
            f"[Pancreas] mode debe ser 'embedding' o 'expert', recibido: '{mode}'"
        assert roi_strategy in ("A", "B"), \
            f"[Pancreas] roi_strategy debe ser 'A' o 'B', recibido: '{roi_strategy}'."

        self.expert_id        = EXPERT_IDS["pancreas"]
        self.mode             = mode
        self.roi_strategy     = roi_strategy
        self.z_score_norm     = z_score_per_volume
        self.samples          = valid_pairs

        n_total = len(self.samples)
        n_pos   = sum(1 for _, l in self.samples if l == 1)
        n_neg   = sum(1 for _, l in self.samples if l == 0)

        log.info(
            f"[Pancreas] Dataset inicializado: {n_total:,} volúmenes | "
            f"mode='{mode}' | roi_strategy='{roi_strategy}' | "
            f"z_score_per_volume={z_score_per_volume}"
        )

        if n_total == 0:
            log.error("[Pancreas] ¡Sin muestras! Ejecuta PanoramaLabelLoader primero.")
            return

        ratio = n_neg / max(n_pos, 1)
        log.info(
            f"[Pancreas] Item-2 — PDAC+ (positivo): {n_pos:,} | "
            f"PDAC− (negativo): {n_neg:,} | ratio: {ratio:.1f}:1"
        )

        # ── H3/Item-4: clip HU abdominal ─────────────────────────────────────
        log.info(
            f"[Pancreas] H3/Item-4 — HU clip: {HU_ABDOMEN_CLIP} (abdomen).\n"
            f"    ⚠ NO usar HU_LUNG_CLIP {HU_LUNG_CLIP} de LUNA16.\n"
            f"    Parénquima pancreático: +30 a +150 HU (fase portal-venosa).\n"
            f"    Tumor PDAC (hipodenso): -20 a +80 HU."
        )

        # ── H2/Item-5: ROI strategy ───────────────────────────────────────────
        log.info(
            f"[Pancreas] H2/Item-5 — Estrategia ROI: Opción {roi_strategy}\n"
            + {
                "A": (
                    "    Opción A — Resize completo a 64³.\n"
                    "    ⚠ Solo válida para FASE 0 (routing/embedding).\n"
                    "    Páncreas quedará representado por ~5 vóxeles."
                ),
                "B": (
                    f"    Opción B — Recorte Z[{PancreasROIExtractor.PANCREAS_Z_MIN}:"
                    f"{PancreasROIExtractor.PANCREAS_Z_MAX}] + resize 64³.\n"
                    f"    Mejora el ratio páncreas/volumen ~3×."
                ),
            }[roi_strategy]
        )

        # ── H3/Item-9: bias multicéntrico + z-score ───────────────────────────
        self._source_audit()
        if z_score_per_volume:
            log.info(
                "[Pancreas] H3/Item-9 — z-score por volumen ACTIVO.\n"
                "    Normaliza media=0, std=1 dentro de cada volumen antes del clip HU.\n"
                "    Compensa diferencias sistemáticas entre Radboudumc/MSD/NIH."
            )

        # ── Item-6: gradient checkpointing ───────────────────────────────────
        log.info(
            "[Pancreas] Item-6 — Gradient checkpointing OBLIGATORIO con 12 GB VRAM.\n"
            "    Un CT abdominal 512×512×300 float32 = ~300 MB por volumen.\n"
            "    Configuración obligatoria: batch_size=1–2 + FP16 + checkpoint."
        )

        # ── H4/Item-7: FocalLoss ──────────────────────────────────────────────
        log.info(
            f"[Pancreas] H4/Item-7 — Loss: FocalLoss(alpha=0.75, gamma=2).\n"
            f"    Desbalance ~{ratio:.1f}:1 + asimetría de costo clínico.\n"
            f"    alpha=0.75 (vs 0.25 de LUNA16): más peso a PDAC+ → penaliza más FN."
        )

        # ── H5/Item-8: k-fold CV ─────────────────────────────────────────────
        log.info(
            f"[Pancreas] H5/Item-8 — k-fold CV (k=5) OBLIGATORIO con {n_total:,} volúmenes.\n"
            f"    Split fijo 80/20 → val ≈ {n_total//5} muestras → σ(AUC) ≈ ±0.05\n"
            f"    k-fold (k=5): σ reducida ~{0.05/5**0.5:.3f}"
        )

    @staticmethod
    def build_kfold_splits(valid_pairs: list,
                           k: int = 5,
                           random_state: int = 42) -> list:
        """H5/Item-8 — Genera k folds estratificados para cross-validation."""
        from sklearn.model_selection import StratifiedKFold

        paths  = [p for p, _ in valid_pairs]
        labels = [l for _, l in valid_pairs]

        skf   = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        folds = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
            train_pairs = [valid_pairs[i] for i in train_idx]
            val_pairs   = [valid_pairs[i] for i in val_idx]

            n_pos_tr = sum(1 for _, l in train_pairs if l == 1)
            n_neg_tr = sum(1 for _, l in train_pairs if l == 0)
            n_pos_va = sum(1 for _, l in val_pairs   if l == 1)
            n_neg_va = sum(1 for _, l in val_pairs   if l == 0)

            log.info(
                f"[Pancreas/kfold] Fold {fold_idx+1}/{k}: "
                f"train={len(train_pairs):,} (pos={n_pos_tr}, neg={n_neg_tr}) | "
                f"val={len(val_pairs):,} (pos={n_pos_va}, neg={n_neg_va})"
            )
            folds.append((train_pairs, val_pairs))

        return folds

    @staticmethod
    def check_trivial_convergence(logits: torch.Tensor,
                                  labels: torch.Tensor,
                                  threshold: float = 0.4) -> dict:
        """H4/Item-7 — Detecta si el modelo converge al mínimo trivial (siempre negativo)."""
        with torch.no_grad():
            probs = torch.sigmoid(logits.float()).cpu().numpy()
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

        pos_mask = labels_np == 1
        neg_mask = labels_np == 0

        prob_pos_mean = float(probs[pos_mask].mean()) if pos_mask.any() else 0.0
        prob_neg_mean = float(probs[neg_mask].mean()) if neg_mask.any() else 0.0
        frac_pred_pos = float((probs > 0.5).mean())

        convergence_risk = (prob_pos_mean < threshold) or (frac_pred_pos < 0.05)

        if convergence_risk:
            log.error(
                f"[Pancreas] H4 — ⚠ RIESGO DE CONVERGENCIA TRIVIAL:\n"
                f"    prob_pos_mean = {prob_pos_mean:.4f} (esperado ≥ {threshold})\n"
                f"    frac_pred_pos = {frac_pred_pos:.4f} (esperado ≥ 0.05)\n"
                f"    El modelo probablemente predice siempre PDAC− (negativo).\n"
                f"    Acciones:\n"
                f"      1. Verificar FocalLoss(alpha=0.75, gamma=2) está activa\n"
                f"      2. Aumentar LR o reducir weight decay\n"
                f"      3. Verificar ROI strategy (H2)\n"
                f"      4. Verificar z_score_per_volume=True"
            )
        else:
            log.info(
                f"[Pancreas] H4 — Convergencia no trivial ✓ | "
                f"prob_pos={prob_pos_mean:.4f} | prob_neg={prob_neg_mean:.4f} | "
                f"frac_pred_pos={frac_pred_pos:.4f}"
            )

        return {
            "prob_pos_mean":    prob_pos_mean,
            "prob_neg_mean":    prob_neg_mean,
            "frac_pred_pos":    frac_pred_pos,
            "convergence_risk": convergence_risk,
        }

    def _source_audit(self):
        """H6/Item-9 — Infiere la fuente de cada caso por naming convention."""
        stems  = [Path(p).name.split(".nii")[0] for p, _ in self.samples]
        n_msd  = sum(1 for s in stems if "Task07" in s or "msd" in s.lower())
        n_nih  = sum(1 for s in stems if "pancreas" in s.lower() and
                     not any(x in s for x in ["panc_", "PANC_"]))
        n_radb = len(stems) - n_msd - n_nih
        total  = max(len(stems), 1)

        log.info(
            f"[Pancreas] H6/Item-9 — Auditoría de fuentes (heurística por naming):\n"
            f"    Radboudumc + UMCG (~Holanda, protocolo variable): "
            f"{n_radb:>4} ({100*n_radb/total:.0f}%)\n"
            f"    MSD Task07 (~NYC, ~50% PDAC)                    : "
            f"{n_msd:>4} ({100*n_msd/total:.0f}%)\n"
            f"    NIH Pancreas-CT (todos negativos)               : "
            f"{n_nih:>4} ({100*n_nih/total:.0f}%)\n"
            f"    ⚠ NIH = 100% negativos → puede inflar accuracy por bias de dominio.\n"
            f"    Contramedidas activas:\n"
            f"      z_score_per_volume = {'activo ✓' if self.z_score_norm else 'INACTIVO ⚠'}\n"
            f"      FocalLoss(alpha=0.75) → concentra aprendizaje en casos difíciles"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        nii_path, label = self.samples[idx]
        case_stem       = Path(nii_path).name.split(".nii")[0]

        try:
            image  = sitk.ReadImage(str(nii_path))
            volume = sitk.GetArrayFromImage(image).astype(np.float32)
        except Exception as e:
            log.warning(f"[Pancreas] Error leyendo '{nii_path}': {e}. "
                        f"Reemplazando con tensor cero.")
            if self.mode == "embedding":
                return torch.zeros(3, 224, 224), self.expert_id, case_stem
            else:
                return torch.zeros(1, 64, 64, 64), label, case_stem

        lo, hi = HU_ABDOMEN_CLIP

        if self.z_score_norm:
            volume = np.clip(volume, lo, hi)
            mean_v = volume.mean()
            std_v  = volume.std()
            if std_v > 1e-6:
                volume = (volume - mean_v) / std_v
                volume = np.clip(volume, -3, 3)
                volume = (volume - (-3)) / 6.0
            else:
                volume = np.zeros_like(volume)
        else:
            volume = normalize_hu(volume, lo, hi)

        if volume.max() == volume.min():
            log.warning(f"[Pancreas] Volumen '{case_stem}' es constante tras normalización.")

        if self.mode == "embedding":
            volume_t = resize_volume_3d(volume, target=(64, 64, 64))
            img      = volume_to_vit_input(volume_t)
            return img, self.expert_id, case_stem
        else:
            if self.roi_strategy == "B":
                roi = PancreasROIExtractor.extract_option_b(volume)
            else:
                roi = PancreasROIExtractor.extract_option_a(volume)
            volume_t = torch.from_numpy(roi).float().unsqueeze(0)
            return volume_t, label, case_stem


class _LegacyPancreasDataset(Dataset):
    """Fallback para .npy pre-procesados (compatibilidad hacia atrás)."""

    def __init__(self, patches_dir):
        self.patches_dir = Path(patches_dir)
        self.expert_id   = EXPERT_IDS["pancreas"]
        self.samples     = list(self.patches_dir.glob("*.npy"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pf = self.samples[idx]
        try:
            v = np.load(pf)
        except Exception:
            return torch.zeros(3, 224, 224), self.expert_id, pf.stem
        lo, hi = HU_ABDOMEN_CLIP
        v = normalize_hu(v, lo, hi)
        vt = resize_volume_3d(v)
        return volume_to_vit_input(vt), self.expert_id, pf.stem

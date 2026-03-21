"""
LUNA16 / LIDC-IDRI — Experto 3: parches 3D de candidatos a nódulo pulmonar.

Hallazgos implementados:
  H1 → tarea: parches 64×64×64, NO volúmenes completos; tensor [B,1,64,64,64]
  H2 → desbalance ~490:1: FocalLoss implementada; submuestreo como alternativa
  H3 → conversión world→vóxel correcta: LUNA16PatchExtractor con validación
  H4 → normalización HU clip[-1000,400]
  H5 → gradient checkpointing documentado (obligatorio con 12 GB VRAM)
  H6 → CPM y FROC (métrica oficial), NO AUC-ROC
  Item-2  → candidates_V2.csv vs V1 obsoleto
  Item-4  → verificación HU
  Item-5  → ratio class=1/class=0
  Item-7  → gradient checkpointing
  Item-8  → FROC/CPM
  Item-9  → spacing variable
"""

import os
import logging
import random
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import SimpleITK as sitk

from ..config import EXPERT_IDS, HU_LUNG_CLIP
from ..preprocessing import normalize_hu, resize_volume_3d, volume_to_vit_input

log = logging.getLogger("fase0")


# ── Constantes HU para LUNA16 ─────────────────────────────────────────────
# H4 — El rango [-1000, 400] HU no es arbitrario: cada valor tiene justificación clínica.
#
# Tejido pulmonar          : -900 a -500 HU (aire + tejido alveolar)
# Nódulos sólidos          : -100 a +100 HU (mismo rango que tejido blando)
# Nódulos part-sólidos     :  -700 a +100 HU (combinan ambos)
# Pared torácica / músculo :  +30 a +80 HU
# Hueso                    :  +400 a +1800 HU  ← EXCLUIDO
# Metales / artefactos     : >+1800 HU         ← EXCLUIDO
#
# Por qué se excluye el hueso (>400 HU):
#   El hueso domina la dinámica de la red si no se elimina.
#
# Por qué -1000 HU como mínimo (no un valor más alto):
#   -1000 HU es la densidad del aire puro. Preservarlo asegura que el fondo
#   del parche tenga siempre el mismo valor absoluto = 0.0 tras la normalización.


def verify_hu_normalization(patches_dir: str, n_sample: int = 3) -> None:
    """
    H4/Item-4 — Verifica visualmente que los parches están correctamente normalizados.

    Un parche bien normalizado tiene:
      - min ≈ 0.0 (aire = -1000 HU tras el clip)
      - max ≈ 1.0 (tejido muy denso = 400 HU)
      - mean entre 0.0 y 0.3 (predomina el aire en el contexto pulmonar)
    """
    patches = list(Path(patches_dir).glob("*.npy"))
    if not patches:
        log.warning(f"[LUNA16/verify_hu] Sin parches .npy en '{patches_dir}'")
        return
    sample = random.sample(patches, min(n_sample, len(patches)))
    log.info(f"[LUNA16/verify_hu] H4 — Verificando normalización HU en {len(sample)} parches:")
    for p in sample:
        arr = np.load(p)
        log.info(
            f"    {p.name}: min={arr.min():.4f} | max={arr.max():.4f} | "
            f"mean={arr.mean():.4f} | shape={arr.shape}"
        )
        if arr.min() < -0.01:
            log.error(f"    ⚠ min < 0 → parche probablemente en HU crudo (no normalizado). "
                      f"Aplica HU_LUNG_CLIP=(-1000,400) y normaliza a [0,1].")
        if arr.max() > 1.01:
            log.error(f"    ⚠ max > 1 → parche en HU crudo. "
                      f"Normalización: (HU - (-1000)) / (400 - (-1000)).")
        if arr.mean() > 0.5:
            log.warning(f"    ⚠ mean > 0.5 → densidad inusualmente alta. "
                        f"¿El clip HU usa el rango correcto {HU_LUNG_CLIP}?")


class LUNA16FROCEvaluator:
    """
    H6 — Interfaz para calcular CPM y curva FROC oficial de LUNA16.

    CPM (Competition Performance Metric) — métrica oficial:
      Promedio de sensitividad en los 7 puntos FP/scan del challenge:
        FP_THRESHOLDS = {1/8, 1/4, 1/2, 1, 2, 4, 8}
      Punto clínicamente más relevante: sensitividad @ 1 FP/scan.

    Referencia: Setio et al., Medical Image Analysis 2017 (arXiv:1612.08012)
    """

    FP_THRESHOLDS = [1/8, 1/4, 1/2, 1, 2, 4, 8]

    def __init__(self, seriesuid_list: list,
                 annotations_csv: str,
                 candidates_v2_csv: str):
        self.seriesuid_list    = seriesuid_list
        self.annotations_csv   = annotations_csv
        self.candidates_v2_csv = candidates_v2_csv

        self.annotations = pd.read_csv(annotations_csv)
        self.candidates  = pd.read_csv(candidates_v2_csv)

        n_nodules   = len(self.annotations)
        n_positives = (self.candidates["class"] == 1).sum()
        sensitivity_ceil = n_positives / max(n_nodules, 1)

        log.info(
            f"[FROC] Evaluador LUNA16 inicializado:\n"
            f"    Nódulos en annotations.csv : {n_nodules}\n"
            f"    Candidatos positivos (V2)  : {n_positives}\n"
            f"    Techo teórico sensitividad : {sensitivity_ceil:.4f} "
            f"({sensitivity_ceil*100:.1f}%) — {n_nodules - n_positives} nódulos sin candidato\n"
            f"    FP_THRESHOLDS (CPM)        : {self.FP_THRESHOLDS}\n"
            f"    Punto clínico clave        : sensitividad @ 1 FP/scan"
        )

    def write_submission(self, predictions: dict, output_csv: str) -> None:
        """
        H6 — Genera el CSV de submission en el formato requerido por
        noduleCADEvaluationLUNA16.py.
        """
        rows = []
        for i, row in self.candidates.iterrows():
            stem = f"candidate_{i:06d}"
            prob = predictions.get(stem, 0.0)
            rows.append({
                "seriesuid":   row["seriesuid"],
                "coordX":      row["coordX"],
                "coordY":      row["coordY"],
                "coordZ":      row["coordZ"],
                "probability": float(prob),
            })
        df_out = pd.DataFrame(rows)
        df_out.to_csv(output_csv, index=False)
        log.info(
            f"[FROC] Submission CSV escrito: {output_csv} | "
            f"{len(df_out):,} candidatos | "
            f"prob media: {df_out['probability'].mean():.4f} | "
            f"prob máx: {df_out['probability'].max():.4f}"
        )

    def run_froc_script(self, submission_csv: str,
                        evaluation_script: str,
                        output_dir: str) -> None:
        """H6 — Ejecuta noduleCADEvaluationLUNA16.py con subprocess."""
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "python", evaluation_script,
            "-s", submission_csv,
            "-a", self.annotations_csv,
            "-o", output_dir,
        ]
        log.info(f"[FROC] Ejecutando: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                log.info(f"[FROC] Evaluación FROC completada ✓ → resultados en '{output_dir}'")
                for line in result.stdout.split("\n"):
                    if "CPM" in line or "cpm" in line.lower() or "sensitivity" in line.lower():
                        log.info(f"[FROC] {line.strip()}")
            else:
                log.error(f"[FROC] Error en el script FROC (returncode={result.returncode}):\n"
                          f"{result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            log.error("[FROC] El script FROC tardó más de 5 minutos. "
                      "Verifica que el submission CSV no esté vacío.")
        except FileNotFoundError:
            log.error(f"[FROC] Script no encontrado: '{evaluation_script}'. "
                      f"Extrae noduleCADEvaluationLUNA16.py del ZIP del challenge LUNA16.")

    @staticmethod
    def log_cpm_interpretation(cpm_value: float) -> None:
        """H6 — Interpreta el CPM en el contexto del challenge LUNA16."""
        if cpm_value >= 0.90:
            level = "EXCELENTE — nivel SOTA (ensemble de sistemas)"
        elif cpm_value >= 0.80:
            level = "BUENO — nivel sistema individual SOTA (~0.831)"
        elif cpm_value >= 0.70:
            level = "ACEPTABLE — objetivo mínimo del proyecto"
        elif cpm_value >= 0.50:
            level = "BAJO — el modelo detecta nódulos pero con muchos FP"
        else:
            level = "MUY BAJO — revisar FocalLoss, conversión world→vóxel y normalización HU"
        log.info(
            f"[FROC] CPM = {cpm_value:.4f} → {level}\n"
            f"    Referencia: LUNA16 sistema individual SOTA ≈ 0.831\n"
            f"    Referencia: LUNA16 ensemble SOTA ≈ 0.946\n"
            f"    Punto clave: sensitividad @ 1 FP/scan (objetivo >0.60)"
        )


class LUNA16PatchExtractor:
    """
    H3 — Extrae parches 3D centrados en coordenadas world del candidates_V2.csv.

    PROBLEMA CRÍTICO: las coordenadas en candidates_V2.csv están en milímetros
    en el espacio del mundo (world coordinates), NO en índices de vóxel.
    """

    @staticmethod
    def world_to_voxel(coord_world: np.ndarray,
                       origin: np.ndarray,
                       spacing: np.ndarray,
                       direction: np.ndarray) -> np.ndarray:
        """H3 — Conversión world → vóxel correcta."""
        coord_shifted = np.array(coord_world) - np.array(origin)
        coord_voxel   = np.linalg.solve(
            np.array(direction).reshape(3, 3) * np.array(spacing),
            coord_shifted
        )
        return np.round(coord_voxel[::-1]).astype(int)   # [iz, iy, ix]

    @staticmethod
    def extract(mhd_path: str,
                coord_world: list,
                patch_size: int = 64,
                clip_hu: tuple = (-1000, 400)) -> np.ndarray:
        """
        H3 — Lee un volumen .mhd y extrae un parche centrado en coord_world.
        Normalización HU incluida (clip + escala a [0,1]).
        """
        image   = sitk.ReadImage(mhd_path)
        origin  = np.array(image.GetOrigin())
        spacing = np.array(image.GetSpacing())
        direc   = np.array(image.GetDirection())
        array   = sitk.GetArrayFromImage(image).astype(np.float32)

        iz, iy, ix = LUNA16PatchExtractor.world_to_voxel(
            coord_world, origin, spacing, direc)

        half = patch_size // 2
        z1, z2 = max(0, iz - half), min(array.shape[0], iz + half)
        y1, y2 = max(0, iy - half), min(array.shape[1], iy + half)
        x1, x2 = max(0, ix - half), min(array.shape[2], ix + half)

        patch = array[z1:z2, y1:y2, x1:x2]

        if patch.shape != (patch_size, patch_size, patch_size):
            pad_z = patch_size - patch.shape[0]
            pad_y = patch_size - patch.shape[1]
            pad_x = patch_size - patch.shape[2]
            patch = np.pad(patch,
                           ((0, pad_z), (0, pad_y), (0, pad_x)),
                           constant_values=clip_hu[0])

        return normalize_hu(patch, clip_hu[0], clip_hu[1])

    @staticmethod
    def validate_extraction(mhd_paths: list,
                            candidates_df,
                            n_positives: int = 10) -> bool:
        """
        H3/Item-3 — Valida la conversión world→vóxel en n candidatos class=1.
        OBLIGATORIO ejecutar antes de extraer el dataset completo.
        """
        positives = candidates_df[candidates_df["class"] == 1].head(n_positives)
        if len(positives) == 0:
            log.error("[LUNA16/validate] Sin candidatos positivos en el CSV. "
                      "¿Estás usando candidates_V2.csv?")
            return False

        mhd_map  = {Path(p).stem: p for p in mhd_paths}
        all_ok   = True

        log.info(f"[LUNA16/validate] Validando conversión world→vóxel "
                 f"en {len(positives)} candidatos positivos...")

        for i, (_, row) in enumerate(positives.iterrows()):
            mhd_path = mhd_map.get(row["seriesuid"])
            if mhd_path is None:
                log.warning(f"  [{i+1}] seriesuid '{row['seriesuid']}' no encontrado "
                            f"en la lista de .mhd")
                continue
            try:
                patch = LUNA16PatchExtractor.extract(
                    mhd_path, [row["coordX"], row["coordY"], row["coordZ"]])
                mean_val = patch.mean()
                status   = "✓" if mean_val > 0.1 else "⚠ POSIBLE ERROR"
                log.info(f"  [{i+1}] {Path(mhd_path).stem[:30]:30s} | "
                         f"coord=({row['coordX']:.1f},{row['coordY']:.1f},{row['coordZ']:.1f}) | "
                         f"patch_mean={mean_val:.3f} {status}")
                if mean_val <= 0.1:
                    all_ok = False
                    log.error(f"  [{i+1}] Parche con intensidad media muy baja ({mean_val:.3f}). "
                              f"El parche es casi todo aire — posible error en la conversión "
                              f"world→vóxel. Verificar origen, spacing y direction del header .mhd.")
            except Exception as e:
                log.error(f"  [{i+1}] Error extrayendo parche: {e}")
                all_ok = False

        if all_ok:
            log.info("[LUNA16/validate] Conversión world→vóxel VÁLIDA en todos los candidatos ✓")
        else:
            log.error("[LUNA16/validate] ¡VALIDACIÓN FALLIDA! Revisar la conversión "
                      "world→vóxel antes de extraer el dataset completo.")
        return all_ok


class LUNA16Dataset(Dataset):
    """
    LUNA16 — Clasificación de PARCHES 3D de candidatos a nódulo.

    Dos modos:
      mode="embedding" → FASE 0: (img_2d, expert_id=3, patch_stem)
      mode="expert"    → FASE 2: (volume_3d, label_binary, patch_stem)
    """

    SENSITIVITY_CEILING = 1120 / 1186   # ≈ 0.9443

    def __init__(self, patches_dir: str, candidates_csv: str, mode: str = "embedding"):
        assert mode in ("embedding", "expert"), \
            f"[LUNA16] mode debe ser 'embedding' o 'expert', recibido: '{mode}'"

        self.patches_dir = Path(patches_dir)
        self.expert_id   = EXPERT_IDS["luna"]
        self.mode        = mode

        if not self.patches_dir.exists():
            log.error(f"[LUNA16] Directorio de parches no encontrado: {self.patches_dir}")
            self.samples = []
            return

        # ── Item-2 / H7: verificar V2 vs V1 ─────────────────────────────────
        csv_path = Path(candidates_csv)
        csv_lower = csv_path.name.lower()
        if "candidates_v2" in csv_lower or "_v2" in csv_lower:
            log.debug(f"[LUNA16] H7/Item-2 — candidates_V2.csv verificado ✓")
        elif csv_path.name == "candidates.csv" or "candidates.csv" == csv_path.name:
            log.error(
                f"[LUNA16] H7/Item-2 — ¡CSV OBSOLETO! Se detectó 'candidates.csv' (V1). "
                f"V1 es el conjunto original del ISBI 2016, explícitamente deprecado. "
                f"candidates_V2.csv detecta 1,162/1,186 nódulos (+24 que V1). "
                f"Usa: --luna_csv /ruta/candidates_V2/candidates_V2.csv "
                f"Impacto: usar V1 reduce artificialmente el techo de sensitividad alcanzable."
            )
        else:
            log.warning(
                f"[LUNA16] H7/Item-2 — El CSV '{csv_path.name}' no parece ser candidates_V2.csv. "
                f"Verifica que estás usando V2 y no V1 (obsoleto). "
                f"V2 detecta 24 nódulos más que V1 y es el estándar del challenge."
            )

        df = pd.read_csv(candidates_csv)
        log.info(f"[LUNA16] CSV cargado: {len(df):,} candidatos totales | "
                 f"columnas: {list(df.columns)}")

        # ── Item-5: verificar ratio class=1 / class=0 ─────────────────────────
        n_pos_csv = (df["class"] == 1).sum()
        n_neg_csv = (df["class"] == 0).sum()
        ratio_csv = n_neg_csv / max(n_pos_csv, 1)
        log.info(
            f"[LUNA16] Item-5 — Distribución en CSV: "
            f"positivos={n_pos_csv:,} | negativos={n_neg_csv:,} | "
            f"ratio={ratio_csv:.0f}:1 (esperado ~490:1)"
        )
        if ratio_csv < 100:
            log.warning(f"[LUNA16] Ratio neg/pos={ratio_csv:.0f}:1 inusualmente bajo. "
                        f"¿El CSV está completo?")
        elif ratio_csv > 1000:
            log.warning(f"[LUNA16] Ratio neg/pos={ratio_csv:.0f}:1 extremo. "
                        f"FocalLoss obligatoria — BCELoss convergirá al mínimo trivial.")

        # ── Cargar parches disponibles en disco ───────────────────────────────
        self.samples = []
        missing = 0
        for _, row in df.iterrows():
            patch_file = self.patches_dir / f"candidate_{row.name:06d}.npy"
            if patch_file.exists():
                self.samples.append((patch_file, int(row["class"])))
            else:
                missing += 1

        n_pos = sum(1 for _, c in self.samples if c == 1)
        n_neg = sum(1 for _, c in self.samples if c == 0)
        ratio = n_neg / max(n_pos, 1)

        log.info(
            f"[LUNA16] {len(self.samples):,} parches cargados del disco | "
            f"positivos: {n_pos:,} | negativos: {n_neg:,} | ratio: {ratio:.0f}:1"
        )
        if missing > 0:
            log.warning(
                f"[LUNA16] {missing:,} parches del CSV no encontrados en disco '{self.patches_dir}'. "
                f"Ejecuta LUNA16PatchExtractor.extract() para generarlos."
            )
        if n_pos == 0:
            log.error("[LUNA16] ¡Cero candidatos positivos cargados! "
                      "El modelo no podrá aprender nódulos. Verifica el CSV y los .npy.")

        # ── H4/Item-4: normalización HU ──────────────────────────────────────
        log.info(
            "[LUNA16] H4/Item-4 — Normalización HU clip[-1000, 400] → [0, 1]:\n"
            f"    HU_LUNG_CLIP = {HU_LUNG_CLIP} — constante exportada para FASE 2.\n"
            "    Justificación del rango:\n"
            "      -1000 HU = aire puro (fondo del parche, padding consistente)\n"
            "       -900 a -500 HU = tejido pulmonar\n"
            "       -100 a +100 HU = nódulos sólidos (lo que queremos detectar)\n"
            "       +400 HU = límite: hueso EXCLUIDO — dominaría la dinámica de la red\n"
            f"    Verificar normalización con: verify_hu_normalization(patches_dir)"
        )

        # ── H2: log de FocalLoss ──────────────────────────────────────────────
        log.info(
            "[LUNA16] H2 — Loss obligatoria para FASE 2: FocalLoss(gamma=2, alpha=0.25). "
            "BCELoss estándar convergirá a predecir siempre class=0 (accuracy >99.7% trivial). "
            "Alternativa complementaria: submuestreo de negativos (ratio máx. 1:20)."
        )

        # ── H5/Item-7: gradient checkpointing ────────────────────────────────
        log.info(
            "[LUNA16] H5/Item-7 — Gradient checkpointing OBLIGATORIO con 12 GB VRAM.\n"
            "    Cálculo de VRAM por batch:\n"
            "      Entrada  : 1 parche 64³ float32 = 1 MB → batch_size=4 → 4 MB\n"
            "      Activaciones red 3D (5+ capas) sin checkpointing: 4-8 GB por muestra\n"
            "      Con FP16 + checkpointing: ~1-2 GB por muestra → batch_size=4 viable\n"
            "    ⚠ Las 3 condiciones deben estar activas SIMULTÁNEAMENTE:\n"
            "      1. FP16 (torch.cuda.amp.autocast)\n"
            "      2. gradient checkpointing (torch.utils.checkpoint)\n"
            "      3. batch_size ≤ 4"
        )

        # ── H6/Item-8: FROC/CPM ──────────────────────────────────────────────
        log.info(
            f"[LUNA16] H6/Item-8 — Métrica oficial: CPM (Competition Performance Metric).\n"
            f"    ⚠ AUC-ROC NO es la métrica correcta.\n"
            f"    CPM = promedio de sensitividad en los 7 puntos FP/scan:\n"
            f"      {LUNA16FROCEvaluator.FP_THRESHOLDS}\n"
            f"    Techo teórico: {self.SENSITIVITY_CEILING:.4f} ({self.SENSITIVITY_CEILING*100:.1f}%)\n"
            f"    — 66 nódulos sin candidato en V2: nunca detectables."
        )

        # ── Item-9: verificar spacing ─────────────────────────────────────────
        self._verify_spacing_sample()

    def _verify_spacing_sample(self):
        """Item-9 — Verifica dimensiones de parches en disco."""
        n_check = min(5, len(self.samples))
        if n_check == 0:
            return
        shapes = set()
        for patch_file, _ in random.sample(self.samples, n_check):
            try:
                arr = np.load(patch_file)
                shapes.add(arr.shape)
            except Exception:
                pass
        if len(shapes) == 1 and next(iter(shapes)) == (64, 64, 64):
            log.debug("[LUNA16] Item-9 — Todos los parches muestreados son [64,64,64] ✓")
        elif len(shapes) == 1:
            log.warning(f"[LUNA16] Item-9 — Parches con forma {next(iter(shapes))} "
                        f"— se esperaba (64,64,64). Verifica la extracción.")
        else:
            log.error(f"[LUNA16] Item-9 — Formas inconsistentes en los parches: {shapes}. "
                      f"Todos los .npy deben ser [64,64,64].")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch_file, label = self.samples[idx]
        try:
            volume = np.load(patch_file)
        except Exception as e:
            log.warning(f"[LUNA16] Error cargando '{patch_file}': {e}. "
                        f"Reemplazando con tensor cero.")
            if self.mode == "embedding":
                return torch.zeros(3, 224, 224), self.expert_id, patch_file.stem
            else:
                return torch.zeros(1, 64, 64, 64), label, patch_file.stem

        if volume.max() == volume.min():
            log.warning(f"[LUNA16] Parche '{patch_file.name}' es constante "
                        f"(valor={volume.max():.3f}). Posible error en la extracción "
                        f"world→vóxel o parche fuera del volumen.")

        if self.mode == "embedding":
            volume_t = resize_volume_3d(volume)
            img      = volume_to_vit_input(volume_t)
            return img, self.expert_id, patch_file.stem
        else:
            volume_t = torch.from_numpy(volume).float().unsqueeze(0)
            return volume_t, label, patch_file.stem

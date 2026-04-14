"""
NIH ChestXray14 — Experto 0: Multi-label, 14 patologías.

Pipeline modernizado: cv2 + CLAHE + multistage_resize (preload en RAM)
+ Albumentations online (augmentación por batch).

Hallazgos implementados:
  H1 → modo expert: vector multi-label [14] + BCEWithLogitsLoss
  H2 → verificación cruzada de Patient ID entre splits (sin leakage)
  H3 → warning de ruido NLP + benchmark AUC en logs
  H4 → filtro opcional por View Position (PA/AP)
  H5 → load_bbox_index() para validar heatmaps del dashboard
  H6 → pos_weight automático + FocalLossMultiLabel disponible
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CHEST_BBOX_CLASSES, CHEST_PATHOLOGIES, EXPERT_IDS, N_CHEST_CLASSES

log = logging.getLogger("fase0")


# ── Constantes ────────────────────────────────────────────────────────
FINDING_LABELS: list[str] = list(CHEST_PATHOLOGIES)  # alias público (14 etiquetas)
TARGET_SIZE: int = 224


# ── Utilidades de preprocesamiento offline ────────────────────────────


def _multistage_resize(img: np.ndarray, target: int = TARGET_SIZE) -> np.ndarray:
    """Halvings iterativos con INTER_AREA, nunca factor >2× en un paso.

    Evita aliasing al reducir imágenes de alta resolución (e.g. 1024→224)
    haciendo pasos intermedios de ½ hasta que el lado menor esté a ≤2×
    del target, y luego un resize final exacto.
    """
    h, w = img.shape[:2]
    while min(h, w) > target * 2:
        h, w = h // 2, w // 2
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)


# CLAHE singleton — reutilizado por todas las instancias del dataset.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ── Dataset ───────────────────────────────────────────────────────────


class ChestXray14Dataset(Dataset):
    """NIH ChestXray14 — Multi-label, 14 patologías.

    Dos modos de uso:
      mode="embedding" → FASE 0: devuelve (img, expert_id=0, img_name)
      mode="expert"    → FASE 2: devuelve (img, label_vector_14, img_name)

    Pipeline:
      Offline (_preload): cv2.imread grayscale → CLAHE → multistage_resize → cache RAM
      Online (__getitem__): GRAY2RGB → Albumentations transform → tensor
    """

    FINDING_LABELS: list[str] = list(CHEST_PATHOLOGIES)

    def __init__(
        self,
        csv_path: str | Path,
        img_dir: str | Path,
        file_list: str | Path,
        transform: Any = None,
        mode: str = "embedding",
        split: str = "train",
        use_cache: bool = True,
        patient_ids_other: set[int] | None = None,
        filter_view: str | None = None,
        # ── parámetros legacy (ignorados, mantienen compatibilidad) ───
        aug_transform: Any = None,
    ) -> None:
        assert mode in ("embedding", "expert"), (
            f"[Chest] mode debe ser 'embedding' o 'expert', recibido: '{mode}'"
        )

        self.img_dir = Path(img_dir)
        self.transform = transform
        self.split = split
        self.expert_id: int = EXPERT_IDS["chest"]
        self.mode = mode
        self.filter_view = filter_view
        self.use_cache = use_cache
        self._cache: dict[int, np.ndarray] = {}

        # ── H2+H3: Verificación de columnas del CSV ───────────────────
        df_full = pd.read_csv(csv_path)
        log.debug(f"[Chest] CSV cargado: {len(df_full):,} filas en {csv_path}")

        required_cols = {
            "Image Index",
            "Finding Labels",
            "Patient ID",
            "View Position",
            "Follow-up #",
        }
        missing_cols = required_cols - set(df_full.columns)
        if missing_cols:
            log.error(
                f"[Chest] Columnas faltantes en el CSV: {missing_cols}. "
                f"Verifica que sea Data_Entry_2017.csv oficial."
            )
        else:
            log.debug(f"[Chest] Columnas verificadas: {sorted(required_cols)} ✓")

        # ── H2: Aplicar split oficial (por Patient ID, no aleatorio) ──
        with open(file_list) as f:
            valid_files = set(f.read().splitlines())
        log.debug(
            f"[Chest] file_list '{Path(file_list).name}': {len(valid_files):,} nombres"
        )

        self.df = df_full[df_full["Image Index"].isin(valid_files)].reset_index(
            drop=True
        )
        n_before_view = len(self.df)
        n_missing_csv = len(valid_files) - n_before_view

        if n_before_view == 0:
            log.error(
                "[Chest] ¡Dataset vacío tras aplicar file_list! "
                "Verifica que chest_csv y chest_train_list sean del mismo release."
            )
        elif n_missing_csv > 0:
            log.warning(
                f"[Chest] {n_missing_csv:,} nombres del file_list no están en el CSV "
                f"(aceptable en subsets). Imágenes cargadas: {n_before_view:,}"
            )
        else:
            log.info(
                f"[Chest] {n_before_view:,} imágenes cargadas (split '{Path(file_list).name}')"
            )

        # ── H4: Filtro por View Position (PA vs AP) ───────────────────
        if "View Position" in self.df.columns:
            view_counts = self.df["View Position"].value_counts()
            log.info(
                f"[Chest] Distribución View Position ANTES del filtro: "
                + " | ".join(f"{v}: {c:,}" for v, c in view_counts.items())
            )

            if filter_view is not None:
                n_antes = len(self.df)
                self.df = self.df[self.df["View Position"] == filter_view].reset_index(
                    drop=True
                )
                n_filtradas = n_antes - len(self.df)
                log.info(
                    f"[Chest] H4 — Filtro View Position='{filter_view}' aplicado: "
                    f"{n_filtradas:,} imágenes eliminadas → "
                    f"{len(self.df):,} imágenes restantes."
                )
                log.info(
                    f"[Chest] H4 — Documenta este filtro en el Reporte Técnico "
                    f"(sección 3 — decisiones de preprocesado)."
                )
                if len(self.df) == 0:
                    log.error(
                        f"[Chest] ¡Dataset vacío tras filtrar View Position='{filter_view}'! "
                        f"Verifica que la columna tenga ese valor. "
                        f"Valores disponibles: {list(view_counts.index)}"
                    )
            else:
                if "AP" in view_counts and "PA" in view_counts:
                    ap_pct = 100 * view_counts.get("AP", 0) / n_before_view
                    log.warning(
                        f"[Chest] H4 — {ap_pct:.1f}% de imágenes son vistas AP (pacientes "
                        f"encamados). Las AP tienen corazón aparentemente más grande y mayor "
                        f"distorsión geométrica — sesgo de dominio potencial. "
                        f"Para estudios controlados: usa filter_view='PA' o "
                        f"--chest_view_filter PA en CLI."
                    )

        n_matched = len(self.df)

        # ── H2: Verificación cruzada de Patient ID (leakage) ──────────
        self.patient_ids: set[int] = set(self.df["Patient ID"].unique())
        log.info(f"[Chest] Pacientes únicos en este split: {len(self.patient_ids):,}")

        if patient_ids_other is not None:
            leaked = self.patient_ids & patient_ids_other
            if leaked:
                log.error(
                    f"[Chest] ¡DATA LEAKAGE! {len(leaked)} Patient IDs aparecen en AMBOS "
                    f"splits. Primeros 5: {list(leaked)[:5]}. "
                    f"Usa EXCLUSIVAMENTE train_val_list.txt y test_list.txt oficiales."
                )
            else:
                log.info(
                    f"[Chest] H2 — Verificación de leakage: 0 Patient IDs compartidos ✓"
                )
        else:
            log.warning(
                "[Chest] patient_ids_other=None — verificación de leakage H2 OMITIDA. "
                "Pasa el set de Patient IDs del split complementario para validar."
            )

        # ── H2: Estadísticas de seguimiento longitudinal ──────────────
        if "Follow-up #" in self.df.columns:
            followup_count = (self.df["Follow-up #"] > 0).sum()
            pct_followup = 100 * followup_count / n_matched if n_matched > 0 else 0
            log.info(
                f"[Chest] H2 — Imágenes con Follow-up # > 0 (visitas repetidas): "
                f"{followup_count:,} ({pct_followup:.1f}%) — "
                f"refuerza la importancia de separar por Patient ID en el split."
            )

        # ── Archivos en disco ─────────────────────────────────────────
        try:
            on_disk = set(os.listdir(str(self.img_dir)))
        except FileNotFoundError:
            log.error(
                f"[Chest] Directorio de imágenes NO encontrado: '{self.img_dir}'. "
                f"Verifica que --chest_imgs apunte a la carpeta all_images/."
            )
            on_disk = set()
        missing_disk = [r for r in self.df["Image Index"] if r not in on_disk]
        if missing_disk:
            log.warning(
                f"[Chest] {len(missing_disk):,} archivos no encontrados en disco "
                f"'{self.img_dir}'. Primeros 5: {missing_disk[:5]}"
            )

        # ── H1 + H6: Modo expert — vector multi-label + pesos de clase
        self.class_weights: torch.Tensor | None = None
        if self.mode == "expert":
            log.info(
                "[Chest] Modo 'expert': preparando vectores multi-label de 14 patologías."
            )
            log.info(
                "[Chest] H1  — Loss obligatoria: BCEWithLogitsLoss (NO CrossEntropyLoss)."
            )
            log.info(
                "[Chest] H6  — Alternativa: FocalLossMultiLabel(gamma=2.0) — "
                "útil para Hernia/Pneumonia. Justificar en reporte si se usa."
            )
            log.info(
                "[Chest] H6  — ⚠ NUNCA usar Accuracy como métrica principal. "
                "Usar: AUC-ROC por clase + F1 Macro + AUPRC. "
                "Benchmark: Wang et al. ResNet-50 ≈ 0.745, CheXNet DenseNet-121 ≈ 0.841 AUC macro."
            )
            log.info(
                "[Chest] H3  — Etiquetas generadas por NLP (~>90% precisión). "
                "AUC > 0.85 significativo → revisar confounding antes de reportar."
            )

            self.df = self.df.copy()
            self.df["label_vector"] = self.df["Finding Labels"].apply(
                self._parse_finding_labels
            )

            all_labels = np.stack(self.df["label_vector"].values)  # [N, 14]
            prevalence = all_labels.mean(axis=0)
            log.info("[Chest] Prevalencia por patología:")
            for name, prev in zip(CHEST_PATHOLOGIES, prevalence):
                bbox_tag = " [BBox✓]" if name in CHEST_BBOX_CLASSES else ""
                bar = "█" * max(1, int(prev * 30))
                log.info(f"    {name:<22}: {prev:.3f}  {bar}{bbox_tag}")

            n_pos = all_labels.sum(axis=0).clip(min=1)
            n_neg = len(all_labels) - n_pos
            self.class_weights = torch.tensor(n_neg / n_pos, dtype=torch.float32)
            log.info(
                f"[Chest] H6  — pos_weight BCEWithLogitsLoss — "
                f"min: {self.class_weights.min():.1f} ({CHEST_PATHOLOGIES[self.class_weights.argmin()]}) | "
                f"max: {self.class_weights.max():.1f} ({CHEST_PATHOLOGIES[self.class_weights.argmax()]})"
            )
            log.debug(
                "[Chest] pos_weight por clase: "
                + " | ".join(
                    f"{n}:{w:.1f}"
                    for n, w in zip(CHEST_PATHOLOGIES, self.class_weights.tolist())
                )
            )

            no_finding_only = self.df["Finding Labels"].str.strip() == "No Finding"
            nf_count = no_finding_only.sum()
            log.info(
                f"[Chest] Imágenes 'No Finding' (vector todo-ceros): "
                f"{nf_count:,} ({100 * nf_count / n_matched:.1f}%) — "
                f"la clase dominante que inflaría Accuracy."
            )

        # ── Preload en RAM (pipeline offline) ─────────────────────────
        if self.use_cache:
            self._preload()

    # ──────────────────────────────────────────────────────────────────
    # Pipeline offline: preload con cv2
    # ──────────────────────────────────────────────────────────────────

    def _preload(self) -> None:
        """Carga todas las imágenes en RAM: cv2 grayscale → CLAHE → multistage_resize.

        Almacena arrays uint8 (224×224, 1 canal) en ``self._cache``.
        Ejecutado una sola vez al instanciar el dataset.
        """
        n = len(self.df)
        log.info(f"[Chest] _preload: cargando {n:,} imágenes en RAM (cv2 + CLAHE)...")
        n_ok = 0
        n_err = 0
        for idx in range(n):
            img_name: str = self.df.loc[idx, "Image Index"]
            img_path = str(self.img_dir / img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(
                        f"cv2.imread returned None for '{img_path}'"
                    )
                # CLAHE en alta resolución (antes del resize)
                img = _CLAHE.apply(img)
                # Multistage resize al target
                img = _multistage_resize(img, target=TARGET_SIZE)
                self._cache[idx] = img
                n_ok += 1
            except Exception as e:
                if n_err < 5:
                    log.warning(f"[Chest] _preload error idx={idx} '{img_name}': {e}")
                n_err += 1
                # Placeholder: array negro del tamaño correcto
                self._cache[idx] = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
        log.info(
            f"[Chest] _preload completado: {n_ok:,} OK, {n_err:,} errores. "
            f"RAM ≈ {n * TARGET_SIZE * TARGET_SIZE / 1024**2:.0f} MB (uint8 gray)"
        )

    # ──────────────────────────────────────────────────────────────────
    # Métodos estáticos preservados (H1, H2, H5)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_finding_labels(label_str: str) -> np.ndarray:
        """H1 — Convierte "Cardiomegaly|Effusion" → vector binario float32 [14].

        "No Finding" o strings con solo etiquetas no en CHEST_PATHOLOGIES → todo-ceros.
        """
        vec = np.zeros(N_CHEST_CLASSES, dtype=np.float32)
        labels = [lbl.strip() for lbl in label_str.split("|")]
        for label in labels:
            if label in CHEST_PATHOLOGIES:
                vec[CHEST_PATHOLOGIES.index(label)] = 1.0
        return vec

    @staticmethod
    def get_patient_ids_from_file_list(csv_path: str, file_list: str) -> set[int]:
        """H2 — Extrae Patient IDs de un split sin construir el dataset completo."""
        df = pd.read_csv(csv_path, usecols=["Image Index", "Patient ID"])
        with open(file_list) as f:
            valid = set(f.read().splitlines())
        return set(df[df["Image Index"].isin(valid)]["Patient ID"].unique())

    @staticmethod
    def load_bbox_index(bbox_csv_path: str) -> dict[str, list[dict[str, Any]]]:
        """H5 — Carga BBox_List_2017.csv como índice {image_name: [bbox_list]}.

        Solo 8 de 14 patologías tienen BBoxes (ver CHEST_BBOX_CLASSES).
        ~1,000 imágenes (~0.9% del total) tienen anotaciones.
        """
        df = pd.read_csv(bbox_csv_path)
        idx: dict[str, list[dict[str, Any]]] = {}
        for _, row in df.iterrows():
            img = row["Image Index"]
            if img not in idx:
                idx[img] = []
            idx[img].append(
                {
                    "label": row["Finding Label"],
                    "x": int(row["Bbox [x"]),
                    "y": int(row["y"]),
                    "w": int(row["w"]),
                    "h": int(row["h]"]),
                }
            )
        log.info(
            f"[Chest] H5 — BBox index cargado: {len(idx):,} imágenes con anotaciones | "
            f"clases cubiertas: {sorted(CHEST_BBOX_CLASSES)}"
        )
        return idx

    # ──────────────────────────────────────────────────────────────────
    # __len__ / __getitem__
    # ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Any, str]:
        img_name: str = self.df.loc[idx, "Image Index"]

        # ── Obtener imagen preprocesada (grayscale uint8 224×224) ─────
        if self.use_cache:
            gray = self._cache[idx]
        else:
            # Fallback: carga en disco sin preload (sin cache en RAM)
            gray = self._load_single(idx, img_name)

        # ── Convertir a RGB (3 canales) para Albumentations / modelo ──
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # (224, 224, 3) uint8

        # ── Aplicar transform (Albumentations Compose o None) ─────────
        if self.transform is not None:
            augmented = self.transform(image=img_rgb)
            img_tensor: torch.Tensor = augmented["image"]
        else:
            # Fallback sin transform: HWC uint8 → CHW float32 [0, 1]
            img_tensor = torch.from_numpy(
                img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        # ── Retorno según modo ────────────────────────────────────────
        if self.mode == "embedding":
            return img_tensor, self.expert_id, img_name
        else:
            label_vec = torch.tensor(
                self.df.loc[idx, "label_vector"], dtype=torch.float32
            )
            return img_tensor, label_vec, img_name

    def _load_single(self, idx: int, img_name: str) -> np.ndarray:
        """Carga y preprocesa una imagen sin usar cache (fallback use_cache=False).

        Aplica el mismo pipeline offline: cv2 grayscale → CLAHE → multistage_resize.
        """
        img_path = str(self.img_dir / img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"cv2.imread returned None for '{img_path}'")
            img = _CLAHE.apply(img)
            img = _multistage_resize(img, target=TARGET_SIZE)
            return img
        except Exception as e:
            log.warning(
                f"[Chest] Error cargando '{img_path}': {e}. "
                f"Reemplazando con imagen cero."
            )
            return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)


# ── Alias para compatibilidad ─────────────────────────────────────────
NIHChestDataset = ChestXray14Dataset

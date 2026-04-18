"""
NIH ChestXray14 — Experto 0: Multi-label, 14 patologías.

Pipeline refactorizado: lee imágenes preprocesadas desde disco (.npy float32)
+ Albumentations online (augmentación por batch).

Las imágenes deben preprocesarse offline con ``pre_chestxray14.py`` antes de
usar este dataset.  Cada split vive en
``datasets/nih_chest_xrays/preprocessed/{train,val,test}/`` con archivos
``.npy`` (256×256, float32, [0,1]) y un ``metadata.csv``.

Hallazgos implementados:
  H1 → modo expert: vector multi-label [14] + BCEWithLogitsLoss
  H5 → load_bbox_index() para validar heatmaps del dashboard
  H6 → pos_weight automático + FocalLossMultiLabel disponible
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CHEST_BBOX_CLASSES, CHEST_PATHOLOGIES, EXPERT_IDS, N_CHEST_CLASSES

log = logging.getLogger("fase0")


# ── Constantes ────────────────────────────────────────────────────────
FINDING_LABELS: list[str] = list(CHEST_PATHOLOGIES)  # alias público (14 etiquetas)
TARGET_SIZE: int = 256


# ── Dataset ───────────────────────────────────────────────────────────


class ChestXray14Dataset(Dataset):
    """NIH ChestXray14 — Multi-label, 14 patologías.

    Lee imágenes **ya preprocesadas** (CLAHE + resize) desde disco como
    archivos ``.npy`` float32 256×256.  No realiza ningún preprocesamiento
    pesado en RAM.

    Dos modos de uso:
      mode="embedding" → FASE 0: devuelve (img, expert_id=0, img_name)
      mode="expert"    → FASE 2: devuelve (img, label_vector_14, img_name)

    Constructor::

        ChestXray14Dataset(preprocessed_dir="datasets/nih_chest_xrays/preprocessed",
                           split="train",
                           transform=albumentations_compose)
    """

    FINDING_LABELS: list[str] = list(CHEST_PATHOLOGIES)

    def __init__(
        self,
        preprocessed_dir: str | Path,
        split: str = "train",
        mode: str = "expert",
        transform: Any = None,
        use_cache: bool = True,
        patient_ids_other: set[int] | None = None,
    ) -> None:
        assert mode in ("embedding", "expert"), (
            f"[Chest] mode debe ser 'embedding' o 'expert', recibido: '{mode}'"
        )
        assert split in ("train", "val", "test"), (
            f"[Chest] split debe ser 'train', 'val' o 'test', recibido: '{split}'"
        )

        self.split = split
        self.transform = transform
        self.expert_id: int = EXPERT_IDS["chest"]
        self.mode = mode

        # ── Validar directorio preprocesado ───────────────────────────
        split_dir = Path(preprocessed_dir) / split
        metadata_path = split_dir / "metadata.csv"

        if not split_dir.is_dir():
            raise RuntimeError(
                f"[Chest] Directorio preprocesado no encontrado: '{split_dir}'. "
                f"Ejecuta pre_chestxray14.py primero para generar las imágenes "
                f"preprocesadas."
            )
        if not metadata_path.is_file():
            raise RuntimeError(
                f"[Chest] metadata.csv no encontrado en '{split_dir}'. "
                f"Ejecuta pre_chestxray14.py primero."
            )

        self.split_dir = split_dir

        # ── Cargar metadata ───────────────────────────────────────────
        self.df = pd.read_csv(metadata_path)
        log.info(
            f"[Chest] Metadata cargada: {len(self.df):,} imágenes "
            f"desde '{metadata_path}'"
        )

        required_cols = {"filename", "patient_id", "label_vector"}
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise RuntimeError(
                f"[Chest] Columnas faltantes en metadata.csv: {missing_cols}. "
                f"Regenera con pre_chestxray14.py."
            )

        if len(self.df) == 0:
            raise RuntimeError(
                f"[Chest] El split '{split}' no tiene imágenes en '{split_dir}'. "
                f"Ejecuta pre_chestxray14.py primero."
            )

        # Validar que los .npy existen (muestreo rápido)
        sample_npy = self._to_npy_name(self.df.loc[0, "filename"])
        sample_file = self.split_dir / sample_npy
        if not sample_file.is_file():
            raise RuntimeError(
                f"[Chest] Archivo .npy no encontrado: '{sample_file}'. "
                f"Ejecuta pre_chestxray14.py primero."
            )

        # ── Parsear label_vectors ─────────────────────────────────────
        self.df["_label_vec"] = self.df["label_vector"].apply(
            lambda s: np.array(json.loads(s), dtype=np.float32)
        )

        # ── Patient IDs ───────────────────────────────────────────────
        self.patient_ids: set[int] = set(self.df["patient_id"].unique())
        log.info(
            f"[Chest] Pacientes únicos en split '{split}': {len(self.patient_ids):,}"
        )

        if patient_ids_other is not None:
            leaked = self.patient_ids & patient_ids_other
            if leaked:
                log.error(
                    f"[Chest] ¡DATA LEAKAGE! {len(leaked)} Patient IDs en ambos splits. "
                    f"Primeros 5: {list(leaked)[:5]}"
                )
            else:
                log.info("[Chest] Verificación de leakage: 0 Patient IDs compartidos ✓")

        # ── Modo expert: estadísticas y pos_weight ────────────────────
        self.class_weights: torch.Tensor | None = None
        if self.mode == "expert":
            all_labels = np.stack(self.df["_label_vec"].values)  # [N, 14]
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
                f"[Chest] pos_weight — "
                f"min: {self.class_weights.min():.1f} ({CHEST_PATHOLOGIES[self.class_weights.argmin()]}) | "
                f"max: {self.class_weights.max():.1f} ({CHEST_PATHOLOGIES[self.class_weights.argmax()]})"
            )

        log.info(
            f"[Chest] Dataset listo: {len(self.df):,} imágenes, split='{split}', mode='{mode}'"
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
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_npy_name(filename: str) -> str:
        """Ensure filename has .npy extension (handles both 'img.png' and 'img.npy')."""
        from pathlib import Path as _P

        return _P(filename).stem + ".npy"

    # ──────────────────────────────────────────────────────────────────
    # __len__ / __getitem__
    # ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Any, str]:
        row = self.df.iloc[idx]
        img_name: str = row["filename"]

        # ── Cargar imagen preprocesada (.npy float32, 256×256) ────────
        npy_name = self._to_npy_name(img_name)
        npy_path = self.split_dir / npy_name
        try:
            img = np.load(npy_path)  # (256, 256) float32 [0, 1]
        except Exception as e:
            log.warning(
                f"[Chest] Error cargando '{npy_path}': {e}. Usando imagen cero."
            )
            img = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)

        # ── Aplicar transform (Albumentations Compose o None) ─────────
        # Albumentations espera HWC, así que expandimos a (256, 256, 1)
        if self.transform is not None:
            img_hwc = img[:, :, np.newaxis]  # (256, 256, 1)
            augmented = self.transform(image=img_hwc)
            img_tensor: torch.Tensor = augmented["image"]
        else:
            # Fallback sin transform: (256, 256) → (1, 256, 256) float32
            img_tensor = torch.from_numpy(img[np.newaxis, :, :])

        # ── Retorno según modo ────────────────────────────────────────
        if self.mode == "embedding":
            return img_tensor, self.expert_id, img_name
        else:
            label_vec = torch.from_numpy(row["_label_vec"])
            return img_tensor, label_vec, img_name


# ── Alias para compatibilidad ─────────────────────────────────────────
NIHChestDataset = ChestXray14Dataset

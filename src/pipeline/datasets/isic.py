"""
ISIC 2019 — Experto 2: Multiclase, 8 clases (ConvNeXt-Small).

Pipeline actualizado con la solución ganadora de ISIC 2020:
  - DullRazor + Color Constancy online + augmentaciones basadas en la literatura
  - CutMix/MixUp en el training loop (batch level)
  - FocalLossMultiClass reemplaza CrossEntropyLoss

Hallazgos implementados:
  H1 → multiclase: FocalLossMultiClass, 8 neuronas softmax, pesos de clase
  H2 → split por lesion_id sin leakage (build_lesion_split)
  H3 → bias 3 fuentes: _source_audit() + apply_circular_crop() para BCN_20000
  H4 → deduplicación MD5: elimina duplicados silenciosos ISIC 2018/2019

Transforms online:
  Train: RandomCrop(224) → flips → rotation 360° → ColorJitter →
         RandomGamma → ShadesOfGray(p=0.5) → CoarseDropout → ToTensor → Normalize
  Val:   CenterCrop(224) → ToTensor → Normalize
  Emb:   Resize(224x224) → ToTensor → Normalize
"""

import logging
import hashlib
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd

from config import EXPERT_IDS

try:
    from fase1.transform_domain import apply_circular_crop

    _HAS_BCN_CROP = True
except ImportError:
    _HAS_BCN_CROP = False
    apply_circular_crop = None  # type: ignore[assignment]

# ── Importar DullRazor + resize para fallback on-the-fly ───────────────
try:
    from fase0.pre_isic import remove_hair_dullrazor, resize_shorter_side

    _HAS_PREPROCESS = True
except ImportError:
    _HAS_PREPROCESS = False
    remove_hair_dullrazor = None  # type: ignore[assignment]
    resize_shorter_side = None  # type: ignore[assignment]

log = logging.getLogger("fase0")


# ── Custom transforms ──────────────────────────────────────────────────


class RandomGamma:
    """Corrección gamma aleatoria: I' = I^γ con γ ∈ gamma_range."""

    def __init__(self, gamma_range: tuple[float, float] = (0.7, 1.5), p: float = 0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() < self.p:
            gamma = np.random.uniform(*self.gamma_range)
            return transforms.functional.adjust_gamma(img, gamma)
        return img


class ShadesOfGray:
    """Color Constancy: normalización del iluminante por norma de Minkowski (p=6).
    Aplica corrección von Kries diagonal para simular luz blanca canónica.
    Usar con p=0.5 online para ConvNeXt-Small (>20M params)."""

    def __init__(self, power: int = 6):
        self.power = power

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img, dtype=np.float32)
        illuminant = np.mean(arr**self.power, axis=(0, 1)) ** (1.0 / self.power)
        illuminant = illuminant / (np.mean(illuminant) + 1e-6)
        corrected = arr / (illuminant + 1e-6)
        return Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8))


class CoarseDropout:
    """Elimina 1–3 parches rectangulares aleatorios (CutOut / CoarseDropout).
    Opera sobre PIL Image ANTES de ToTensor."""

    def __init__(
        self,
        n_holes_range: tuple[int, int] = (1, 3),
        size_range: tuple[int, int] = (32, 96),
        fill_value: tuple[int, int, int] = (
            int(0.485 * 255),
            int(0.456 * 255),
            int(0.406 * 255),
        ),
        p: float = 0.5,
    ):
        self.n_holes_range = n_holes_range
        self.size_range = size_range
        self.fill_value = fill_value
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() >= self.p:
            return img
        arr = np.array(img)
        h, w = arr.shape[:2]
        n_holes = np.random.randint(self.n_holes_range[0], self.n_holes_range[1] + 1)
        for _ in range(n_holes):
            hole_h = np.random.randint(self.size_range[0], self.size_range[1] + 1)
            hole_w = np.random.randint(self.size_range[0], self.size_range[1] + 1)
            y = np.random.randint(0, max(1, h - hole_h))
            x = np.random.randint(0, max(1, w - hole_w))
            arr[y : y + hole_h, x : x + hole_w] = self.fill_value
        return Image.fromarray(arr)


# ── Transform pipelines ────────────────────────────────────────────────

TARGET_SIZE = 224  # debe coincidir con EXPERT2_IMG_SIZE

TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(TARGET_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=360,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        RandomGamma(gamma_range=(0.7, 1.5), p=0.5),
        transforms.RandomApply([ShadesOfGray(power=6)], p=0.5),
        CoarseDropout(n_holes_range=(1, 3), size_range=(32, 96), p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TRANSFORM_VAL = transforms.Compose(
    [
        transforms.CenterCrop(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Para modo "embedding" (Fase 0) — sin augmentación
TRANSFORM_EMBEDDING = transforms.Compose(
    [
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ── CutMix / MixUp helpers ────────────────────────────────────────────


def cutmix_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix: recorta región rectangular de imagen A y la pega en imagen B.
    Returns: mixed_x, y_a, y_b, lam
    Usar en training loop como: loss = lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    rand_index = torch.randperm(batch_size, device=x.device)
    y_a = y
    y_b = y[rand_index]
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    x = x.clone()
    x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (W * H)
    return x, y_a, y_b, lam


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp: combina linealmente dos imágenes y sus etiquetas.
    Returns: mixed_x, y_a, y_b, lam
    Usar en training loop como: loss = lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    rand_index = torch.randperm(batch_size, device=x.device)
    y_a = y
    y_b = y[rand_index]
    mixed_x = lam * x + (1.0 - lam) * x[rand_index]
    return mixed_x, y_a, y_b, lam


# ── ISICDataset ────────────────────────────────────────────────────────


class ISICDataset(Dataset):
    """
    ISIC 2019 — Multiclase, 8 clases (ConvNeXt-Small).

    Dos modos:
      mode="embedding" → FASE 0: (img, expert_id=1, img_name)
      mode="expert"    → FASE 2: (img, class_label_int, img_name)

    Soporta cache_dir: si las imágenes preprocesadas existen ahí
    (DullRazor + resize shorter_side=224), se cargan directamente.
    Si no, se cargan las originales con resize a shorter_side=224.
    """

    CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    N_TRAIN_CLS = 8
    KNOWN_DUPLICATES = {"ISIC_0067980", "ISIC_0069013"}
    _error_count = 0

    @staticmethod
    def _resize_shorter_side(img: Image.Image, target_size: int) -> Image.Image:
        """Resize manteniendo aspecto: lado corto = target_size."""
        w, h = img.size
        scale = target_size / min(w, h)
        new_w, new_h = math.ceil(w * scale), math.ceil(h * scale)
        return img.resize((new_w, new_h), Image.LANCZOS)

    def _resolve_image_path(self, img_name: str) -> tuple[Path | None, bool]:
        """Resuelve la ruta real de una imagen probando todas las convenciones de nombre.

        Cadena de resolución (primera que exista gana):
          1.  cache_dir / {img_name}_pp_{TARGET_SIZE}.jpg   (cache preprocesado exacto)
          2.  cache_dir / {img_name}.jpg                    (original copiado al cache)
          3.  img_dir   / {img_name}.jpg                    (original exacto)
          4.  img_dir   / {img_name}_pp_{TARGET_SIZE}.jpg   (pp en img_dir — cubre
                  el caso donde img_dir apunta al directorio preprocesado)

          Si img_name termina en '_downsampled', repetir 1-4 con base_name
          (sin el sufijo _downsampled):
          5.  cache_dir / {base}_pp_{TARGET_SIZE}.jpg
          6.  cache_dir / {base}.jpg
          7.  img_dir   / {base}.jpg
          8.  img_dir   / {base}_pp_{TARGET_SIZE}.jpg

          Último recurso — glob:
          9.  glob en cache_dir y img_dir buscando {img_name}* y {base}*
              (cubre extensiones .jpeg/.png y variantes no anticipadas).

        Returns:
            (ruta, is_cache): ruta al archivo encontrado y si proviene de cache
                              o tiene naming _pp_ (→ no necesita preproceso).
                              (None, False) si no se encuentra.
        """
        pp_suffix = f"_pp_{TARGET_SIZE}"

        # Candidatos: (directorio, nombre_archivo, es_cache_o_pp)
        candidates: list[tuple[Path | None, str, bool]] = [
            (self.cache_dir, f"{img_name}{pp_suffix}.jpg", True),  # 1
            (self.cache_dir, f"{img_name}.jpg", True),  # 2
            (self.img_dir, f"{img_name}.jpg", False),  # 3
            (self.img_dir, f"{img_name}{pp_suffix}.jpg", True),  # 4
        ]

        # Variantes sin _downsampled
        if img_name.endswith("_downsampled"):
            base = img_name[: -len("_downsampled")]
            candidates += [
                (self.cache_dir, f"{base}{pp_suffix}.jpg", True),  # 5
                (self.cache_dir, f"{base}.jpg", True),  # 6
                (self.img_dir, f"{base}.jpg", False),  # 7
                (self.img_dir, f"{base}{pp_suffix}.jpg", True),  # 8
            ]

        for directory, filename, is_cached in candidates:
            if directory is None:
                continue
            path = directory / filename
            if path.exists():
                return path, is_cached

        # ── Último recurso: glob en ambos directorios ──────────────────
        #   Cubre extensiones alternativas (.jpeg, .png) y variantes de
        #   naming no anticipadas en el dataset.
        base_id = (
            img_name.split("_downsampled")[0]
            if "_downsampled" in img_name
            else img_name
        )
        glob_patterns = [f"{img_name}*"]
        if base_id != img_name:
            glob_patterns.append(f"{base_id}*")

        for directory in (self.cache_dir, self.img_dir):
            if directory is None:
                continue
            for pattern in glob_patterns:
                matches = sorted(directory.glob(pattern))
                for match in matches:
                    if match.is_file() and match.suffix.lower() in (
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".tiff",
                    ):
                        is_pp = pp_suffix in match.name
                        return match, is_pp

        return None, False

    @staticmethod
    def get_weighted_sampler(split_df: pd.DataFrame, classes: list[str]):
        """
        WeightedRandomSampler para compensar desbalance NV>>DF/VASC en FASE 2.
        """
        from torch.utils.data import WeightedRandomSampler

        # Soporta CSV con columna label_idx (entero) o columnas one-hot por clase
        if "label_idx" in split_df.columns:
            labels = split_df["label_idx"].values.astype(int)
        else:
            labels = split_df[classes].values.argmax(axis=1)
        counts = np.bincount(labels, minlength=len(classes)).astype(float)
        counts = np.maximum(counts, 1)
        w_cls = 1.0 / counts
        w_samp = w_cls[labels]
        return WeightedRandomSampler(
            weights=torch.tensor(w_samp, dtype=torch.float32),
            num_samples=len(w_samp),
            replacement=True,
        )

    def __init__(
        self,
        img_dir: str | Path,
        split_df: pd.DataFrame,
        mode: str = "embedding",
        split: str = "train",
        cache_dir: str | Path | None = None,
        apply_bcn_crop: bool = True,
        **kwargs,  # absorbe params extra de callers (ej. transform_standard)
    ):
        if kwargs:
            log.debug(
                "[ISIC] __init__ ignoró kwargs extras: %s",
                list(kwargs.keys()),
            )

        assert mode in ("embedding", "expert"), (
            f"[ISIC] mode debe ser 'embedding' o 'expert', recibido: '{mode}'"
        )

        self.img_dir = Path(img_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.expert_id = EXPERT_IDS["isic"]
        self.mode = mode
        self.split = split
        self.apply_bcn_crop = apply_bcn_crop
        self.df = split_df.reset_index(drop=True)
        self.class_weights = None

        # Seleccionar transform según modo y split
        if mode == "embedding":
            self.transform = TRANSFORM_EMBEDDING
        elif split == "train":
            self.transform = TRANSFORM_TRAIN
        else:
            self.transform = TRANSFORM_VAL

        log.info(
            f"[ISIC] Dataset: {len(self.df):,} imgs | mode='{mode}' | "
            f"split='{split}' | cache_dir={self.cache_dir} | "
            f"apply_bcn_crop={apply_bcn_crop}"
        )

        # Verificar etiquetas
        if "label_idx" in self.df.columns:
            bad = (
                (self.df["label_idx"] < 0) | (self.df["label_idx"] >= len(self.CLASSES))
            ).sum()
            if bad:
                log.error(
                    f"[ISIC] {bad} filas con label_idx fuera del rango [0, {len(self.CLASSES) - 1}]. "
                    f"ISIC es MULTICLASE con {len(self.CLASSES)} clases."
                )
            else:
                log.debug(
                    "[ISIC] Ground truth label_idx ✓ (rango 0-%d)",
                    len(self.CLASSES) - 1,
                )
        elif all(c in self.df.columns for c in self.CLASSES):
            bad = (self.df[self.CLASSES].sum(axis=1) != 1).sum()
            if bad:
                log.error(
                    f"[ISIC] {bad} filas sin exactamente un 1.0. "
                    f"ISIC es MULTICLASE (no multi-label)."
                )
            else:
                log.debug("[ISIC] Ground truth one-hot ✓")

        # H3/Item-6 — verificar accesibilidad de TODAS las imágenes del split
        #   Chequea cache_dir y img_dir; reporta cuáles NO se encuentran en ninguno.
        ds_names = self.df["image"][
            self.df["image"].str.endswith("_downsampled")
        ].tolist()
        if ds_names:
            pct = 100 * len(ds_names) / len(self.df)
            log.info(
                f"[ISIC] H3/Item-6 — {len(ds_names):,} ({pct:.1f}%) imgs _downsampled (MSK)."
            )

        # Verificar que CADA imagen del split es accesible (cache o img_dir)
        missing_all: list[str] = []
        for img_n in self.df["image"]:
            path, _ = self._resolve_image_path(img_n)
            if path is None:
                missing_all.append(img_n)

        if missing_all:
            log.warning(
                f"[ISIC] Item-6 — {len(missing_all)} imgs NO encontradas en cache NI "
                f"img_dir (se usará fallback on-the-fly o último recurso). "
                f"Primeros 5: {missing_all[:5]}"
            )
        else:
            log.info(
                f"[ISIC] Item-6 — Todas las {len(self.df):,} imgs del split "
                f"verificadas en disco ✓ (cache o img_dir)"
            )

        # H3/Item-7 — source audit (HAM/BCN/MSK)
        self._source_audit()

        # Preparar modo expert
        if mode == "expert":
            log.info(
                "[ISIC] Loss: FocalLossMultiClass con pesos (gamma=2.0, label_smoothing=0.1)."
            )
            log.info("[ISIC] 8 neuronas softmax. Métrica: BMCA (no Accuracy global).")
            log.info(
                "[ISIC] Augmentation pipeline:\n"
                "    Train: RandomCrop(224) → flips(0.5) → rotation(360°) → "
                "ColorJitter → RandomGamma → ShadesOfGray(p=0.5) → CoarseDropout\n"
                "    + CutMix(p=0.3) / MixUp(p=0.2) a nivel de batch\n"
                "    Usar get_weighted_sampler() en DataLoader de FASE 2."
            )

            self.df = self.df.copy()
            # Soporta CSV con columna label_idx (entero) o columnas one-hot por clase
            if "label_idx" in self.df.columns:
                self.df["class_label"] = self.df["label_idx"].astype(int)
            else:
                self.df["class_label"] = self.df[self.CLASSES].values.argmax(axis=1)

            dist = self.df["class_label"].value_counts().sort_index()
            total = len(self.df)
            log.info("[ISIC] Distribución de clases:")
            for i, cls in enumerate(self.CLASSES):
                count = dist.get(i, 0)
                bar = "█" * max(1, int(30 * count / total))
                log.info(
                    f"    {cls:<5}(idx {i}): {count:>6,} ({100 * count / total:.1f}%) {bar}"
                )

            counts = np.array(
                [dist.get(i, 1) for i in range(self.N_TRAIN_CLS)], dtype=float
            )
            weights = total / (self.N_TRAIN_CLS * counts)
            self.class_weights = torch.tensor(weights, dtype=torch.float32)
            log.info(
                f"[ISIC] class_weights[{self.N_TRAIN_CLS}]: "
                f"min={self.class_weights.min():.2f}({self.CLASSES[self.class_weights.argmin()]}) | "
                f"max={self.class_weights.max():.2f}({self.CLASSES[self.class_weights.argmax()]})"
            )

            # Verificar filas sin etiqueta válida
            if "label_idx" in self.df.columns:
                invalid = (
                    (self.df["class_label"] < 0)
                    | (self.df["class_label"] >= self.N_TRAIN_CLS)
                ).sum()
                if invalid:
                    log.warning(
                        f"[ISIC] {invalid} filas con class_label fuera de [0, {self.N_TRAIN_CLS - 1}]."
                    )
            else:
                all_zero = (self.df[self.CLASSES].sum(axis=1) == 0).sum()
                if all_zero:
                    log.warning(
                        f"[ISIC] {all_zero} filas sum=0 → argmax asignará clase 0 (MEL)."
                    )

    def _source_audit(self) -> None:
        """H3/Item-7 — Infiere fuente por naming e imprime estadísticas de bias de dominio."""
        names = self.df["image"]
        msk_mask = names.str.endswith("_downsampled")
        try:
            numeric_ids = names.str.extract(r"ISIC_(\d+)")[0].astype(float)
            ham_mask = (~msk_mask) & (numeric_ids <= 67977)
            bcn_mask = (~msk_mask) & (numeric_ids > 67977)
        except Exception:
            ham_mask = pd.Series([False] * len(names), index=names.index)
            bcn_mask = ~msk_mask & ~ham_mask

        n_ham, n_bcn, n_msk = ham_mask.sum(), bcn_mask.sum(), msk_mask.sum()
        total = len(names)
        log.info(
            f"[ISIC] H3/Item-7 — Auditoría de fuentes (heurística por ID range):\n"
            f"    HAM10000 (Viena, 600×450, recortado)      : {n_ham:>5,} ({100 * n_ham / total:.1f}%)\n"
            f"    BCN_20000 (Barcelona, 1024×1024, dermoscop): {n_bcn:>5,} ({100 * n_bcn / total:.1f}%)\n"
            f"    MSK (NYC, variable, _downsampled)          : {n_msk:>5,} ({100 * n_msk / total:.1f}%)\n"
            f"    Contramedida activa: apply_circular_crop() (BCN) + ShadesOfGray(p=0.5).\n"
            f"    ⚠ Para inspección visual: samplea 5-10 imgs de cada fuente antes de FASE 2."
        )

    @staticmethod
    def build_lesion_split(
        gt_csv,
        img_dir=None,
        metadata_csv=None,
        frac_train: float = 0.8,
        random_state: int = 42,
    ):
        """
        H2 + H4 — Split por lesion_id + deduplicación MD5.
        """
        df_gt = pd.read_csv(gt_csv)
        log.info(f"[ISIC/split] GT CSV: {len(df_gt):,} filas")

        # H4: eliminar duplicados conocidos por ID
        before = len(df_gt)
        df_gt = df_gt[~df_gt["image"].isin(ISICDataset.KNOWN_DUPLICATES)].reset_index(
            drop=True
        )
        if before - len(df_gt):
            log.info(
                f"[ISIC/split] H4 — {before - len(df_gt)} KNOWN_DUPLICATES eliminados."
            )

        # H4: deduplicación por hash MD5 de archivos reales
        if img_dir is not None:
            img_path_obj = Path(img_dir)
            log.info(
                f"[ISIC/split] H4 — Calculando hashes MD5 de {len(df_gt):,} imgs "
                f"(puede tardar ~1-2 min)..."
            )
            hashes: dict[str, str] = {}
            dup_ids: set[str] = set()
            missing = 0
            for _, row in df_gt.iterrows():
                fpath = img_path_obj / f"{row['image']}.jpg"
                if not fpath.exists():
                    missing += 1
                    continue
                try:
                    md5 = hashlib.md5(fpath.read_bytes()).hexdigest()
                    if md5 in hashes:
                        dup_ids.add(row["image"])
                        log.debug(
                            f"[ISIC/split] H4 — Dup: '{row['image']}' == '{hashes[md5]}'"
                        )
                    else:
                        hashes[md5] = row["image"]
                except Exception as e:
                    log.warning(f"[ISIC/split] H4 — No se pudo hashear '{fpath}': {e}")
            if missing:
                log.warning(
                    f"[ISIC/split] H4 — {missing} archivos no encontrados para hashing."
                )
            if dup_ids:
                df_gt = df_gt[~df_gt["image"].isin(dup_ids)].reset_index(drop=True)
                log.info(
                    f"[ISIC/split] H4 — {len(dup_ids)} duplicados MD5 eliminados. "
                    f"Quedan {len(df_gt):,} imgs únicas."
                )
            else:
                log.info("[ISIC/split] H4 — Sin duplicados MD5 detectados ✓")
        else:
            log.warning(
                "[ISIC/split] H4 — img_dir=None: dedup MD5 OMITIDA. "
                "Pasa --isic_imgs a build_lesion_split para deduplicación completa."
            )

        # H2: split por lesion_id
        if metadata_csv is not None:
            df_meta = pd.read_csv(metadata_csv, usecols=["image", "lesion_id"])
            df_gt = df_gt.merge(df_meta, on="image", how="left")
            n_with = df_gt["lesion_id"].notna().sum()
            n_wo = df_gt["lesion_id"].isna().sum()
            log.info(
                f"[ISIC/split] H2 — lesion_id: {n_with:,} con ID | {n_wo:,} sin ID (→ únicos)"
            )

            mask_na = df_gt["lesion_id"].isna()
            df_gt.loc[mask_na, "lesion_id"] = df_gt.loc[mask_na, "image"].apply(
                lambda x: f"_unique_{x}"
            )

            unique_lesions = df_gt["lesion_id"].unique()
            rng = np.random.RandomState(random_state)
            rng.shuffle(unique_lesions)
            n_tr = int(len(unique_lesions) * frac_train)
            train_lesions = set(unique_lesions[:n_tr])
            val_lesions = set(unique_lesions[n_tr:])

            train_df = df_gt[df_gt["lesion_id"].isin(train_lesions)].copy()
            val_df = df_gt[df_gt["lesion_id"].isin(val_lesions)].copy()

            shared = set(train_df["lesion_id"]) & set(val_df["lesion_id"])
            if shared:
                log.error(
                    f"[ISIC/split] H2 — ¡LEAKAGE! {len(shared)} lesion_ids compartidos."
                )
            else:
                log.info(
                    f"[ISIC/split] H2 — Split por lesion_id ✓ | "
                    f"train: {len(train_lesions):,} lesiones ({len(train_df):,} imgs) | "
                    f"val: {len(val_lesions):,} lesiones ({len(val_df):,} imgs)"
                )
        else:
            log.warning(
                "[ISIC/split] H2 — metadata_csv=None: split aleatorio por imagen. "
                "Pasa --isic_metadata para split correcto por lesion_id."
            )
            idx_all = np.arange(len(df_gt))
            rng = np.random.RandomState(random_state)
            rng.shuffle(idx_all)
            n_tr = int(len(idx_all) * frac_train)
            train_df = df_gt.iloc[idx_all[:n_tr]].copy()
            val_df = df_gt.iloc[idx_all[n_tr:]].copy()

        log.info(
            f"[ISIC/split] Final — train: {len(train_df):,} | val: {len(val_df):,}"
        )
        return train_df, val_df

    def __len__(self) -> int:
        return len(self.df)

    def _try_open_image(self, path: Path) -> Image.Image | None:
        """Intenta abrir una imagen; retorna None si falla."""
        if path.exists():
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                return None
        return None

    def _preprocess_on_the_fly(self, img: Image.Image) -> Image.Image:
        """Aplica DullRazor + resize_shorter_side on-the-fly (misma pipeline de fase0)."""
        arr = np.array(img)
        if _HAS_PREPROCESS:
            try:
                arr = remove_hair_dullrazor(arr)
            except Exception:
                pass  # Si falla hair removal, seguimos con la imagen original
            arr = resize_shorter_side(arr, TARGET_SIZE)
        else:
            # Fallback PIL: solo resize
            img = self._resize_shorter_side(img, TARGET_SIZE)
            return img
        return Image.fromarray(arr)

    def __getitem__(self, idx: int):
        img_name = self.df.loc[idx, "image"]

        # ── Resolución unificada de ruta ───────────────────────────────
        #   _resolve_image_path prueba todas las convenciones de nombres
        #   (original, preprocesado, con/sin _downsampled) en cache_dir
        #   e img_dir. Retorna (path, is_cached) o (None, False).
        #
        # Cuando la imagen se carga desde un archivo original (no _pp_)
        # se aplica DullRazor + resize on-the-fly para replicar la
        # pipeline de preprocesamiento de fase0.

        img = None
        loaded_from_cache = False

        resolved_path, is_cached = self._resolve_image_path(img_name)
        if resolved_path is not None:
            img = self._try_open_image(resolved_path)
            if img is not None:
                loaded_from_cache = is_cached

        # --- Último recurso: tensor cero ---
        if img is None:
            ISICDataset._error_count += 1
            if ISICDataset._error_count <= 20:
                extra = ""
                if resolved_path is not None:
                    extra = f" (archivo encontrado en {resolved_path} pero no se pudo abrir)"
                log.error(
                    f"[ISIC] TODAS las rutas de carga fallaron para '{img_name}'.{extra} "
                    f"Paths intentados: cache={self.cache_dir}, img_dir={self.img_dir}. "
                    f"Retornando tensor cero (error {ISICDataset._error_count}/20 logged)."
                )
            dummy = torch.zeros(3, TARGET_SIZE, TARGET_SIZE)
            if self.mode == "embedding":
                return dummy, self.expert_id, img_name
            label = (
                int(self.df.loc[idx, "class_label"])
                if "class_label" in self.df.columns
                else 0
            )
            return dummy, label, img_name

        # --- Preprocesamiento on-the-fly para imágenes NO cacheadas ---
        # Si la imagen se cargó desde img_dir (original), aplicar DullRazor +
        # resize para replicar la pipeline de fase0 que produjo los caches.
        if not loaded_from_cache:
            img = self._preprocess_on_the_fly(img)

        # H3/Item-8 — eliminar bordes negros BCN_20000 (solo si no viene de cache)
        if self.apply_bcn_crop and not loaded_from_cache and _HAS_BCN_CROP:
            img = apply_circular_crop(img)

        # Resize a shorter_side >= TARGET_SIZE para que RandomCrop(224) funcione.
        # Se aplica siempre (cache o no): las imgs cacheadas con el bug de
        # truncamiento int() pueden tener lado corto = 223 en vez de 224.
        if self.mode != "embedding" and min(img.size) < TARGET_SIZE:
            img = self._resize_shorter_side(img, TARGET_SIZE)

        if self.mode == "embedding":
            return self.transform(img), self.expert_id, img_name
        else:
            return self.transform(img), int(self.df.loc[idx, "class_label"]), img_name

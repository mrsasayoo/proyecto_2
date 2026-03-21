"""
ISIC 2019 — Experto 1: Multiclase, 9 clases (8 en train + UNK solo en test).

Hallazgos implementados:
  H1 → multiclase: CrossEntropyLoss, 9 neuronas softmax, pesos de clase
  H2 → split por lesion_id sin leakage (build_lesion_split)
  H3 → bias 3 fuentes: _source_audit() + apply_circular_crop() para BCN_20000
  H4 → deduplicación MD5: elimina duplicados silenciosos ISIC 2018/2019
  H5 → UNK: validación todo-ceros en train, slot 8 para OOD en inferencia
  H6 → augmentation diferenciado: estándar vs agresivo (Gessert et al.) por clase
"""

import logging
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd

from config import EXPERT_IDS, IMAGENET_MEAN, IMAGENET_STD
from preprocessing import apply_circular_crop

log = logging.getLogger("fase0")


class ISICDataset(Dataset):
    """
    ISIC 2019 — Multiclase, 9 clases (8 en train + UNK solo en test).

    Dos modos:
      mode="embedding" → FASE 0: (img, expert_id=1, img_name)
      mode="expert"    → FASE 2: (img, class_label_int, img_name)
    """

    CLASSES       = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    N_CLASSES     = 9
    N_TRAIN_CLS   = 8
    MINORITY_IDX  = {3, 5, 6, 7}       # AK=3, DF=5, VASC=6, SCC=7
    MINORITY_NAMES= {"AK", "DF", "VASC", "SCC"}
    KNOWN_DUPLICATES = {"ISIC_0067980", "ISIC_0069013"}

    @staticmethod
    def build_isic_transforms(img_size=224):
        """
        H6 — Tres pipelines de transform:
          "embedding" : sin augmentation (FASE 0).
          "standard"  : augmentation moderado para clases mayoritarias (FASE 2).
          "minority"  : augmentation agresivo Gessert et al. 2020 (FASE 2).
        """
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        embedding_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])

        standard_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])

        minority_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation((90,  90))],  p=0.33),
            transforms.RandomApply([transforms.RandomRotation((180, 180))], p=0.33),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                    scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ])

        return {"embedding": embedding_tf,
                "standard":  standard_tf,
                "minority":  minority_tf}

    @staticmethod
    def get_weighted_sampler(split_df, classes):
        """
        H6 — WeightedRandomSampler para compensar desbalance NV>>DF/VASC en FASE 2.
        """
        from torch.utils.data import WeightedRandomSampler
        labels  = split_df[classes].values.argmax(axis=1)
        counts  = np.bincount(labels, minlength=len(classes)).astype(float)
        counts  = np.maximum(counts, 1)
        w_cls   = 1.0 / counts
        w_samp  = w_cls[labels]
        return WeightedRandomSampler(
            weights=torch.tensor(w_samp, dtype=torch.float32),
            num_samples=len(w_samp), replacement=True)

    def __init__(self, img_dir, split_df, mode="embedding",
                 transform_standard=None, transform_minority=None,
                 apply_bcn_crop=True):
        assert mode in ("embedding", "expert"), \
            f"[ISIC] mode debe ser 'embedding' o 'expert', recibido: '{mode}'"

        self.img_dir        = Path(img_dir)
        self.expert_id      = EXPERT_IDS["isic"]
        self.mode           = mode
        self.apply_bcn_crop = apply_bcn_crop
        self.df             = split_df.reset_index(drop=True)
        self.class_weights  = None

        tfs = self.build_isic_transforms()
        if mode == "embedding":
            self.transform = tfs["embedding"]
        else:
            self.transform_standard = transform_standard or tfs["standard"]
            self.transform_minority = transform_minority or tfs["minority"]
            self.transform = self.transform_standard

        log.info(f"[ISIC] Dataset: {len(self.df):,} imgs | mode='{mode}' | "
                 f"apply_bcn_crop={apply_bcn_crop}")

        # H1 — verificar one-hot
        if all(c in self.df.columns for c in self.CLASSES):
            bad = (self.df[self.CLASSES].sum(axis=1) != 1).sum()
            if bad:
                log.error(f"[ISIC] H1 — {bad} filas sin exactamente un 1.0. "
                          f"ISIC es MULTICLASE (no multi-label). row_sum=0=UNK; row_sum>1=error CSV.")
            else:
                log.debug("[ISIC] H1 — Ground truth one-hot ✓")

        # H5 — verificar UNK = todo-ceros en train
        if "UNK" in self.df.columns:
            n_unk = (self.df["UNK"] != 0).sum()
            if n_unk:
                log.warning(f"[ISIC] H5 — {n_unk} filas con UNK≠0. "
                            f"UNK solo debería estar en el test set oficial.")
            else:
                log.info("[ISIC] H5 — UNK=0 en train ✓ | "
                         "Slot 8 (softmax) disponible para OOD en inferencia → Experto 5")

        # H3/Item-6 — detectar _downsampled y verificar en disco
        ds_names = self.df["image"][self.df["image"].str.endswith("_downsampled")].tolist()
        if ds_names:
            pct = 100 * len(ds_names) / len(self.df)
            log.warning(f"[ISIC] H3/Item-6 — {len(ds_names):,} ({pct:.1f}%) imgs _downsampled (MSK). "
                        f"Verificando en disco...")
            missing_ds = [n for n in ds_names if not (self.img_dir / f"{n}.jpg").exists()]
            if missing_ds:
                log.error(f"[ISIC] Item-6 — {len(missing_ds)} archivos _downsampled NO en disco. "
                          f"Primeros 5: {missing_ds[:5]}")
            else:
                log.info(f"[ISIC] Item-6 — Todos los _downsampled verificados en disco ✓")
        else:
            log.debug("[ISIC] Item-6 — Sin imágenes _downsampled en este split.")

        # H3/Item-7 — source audit (HAM/BCN/MSK)
        self._source_audit()

        # H6/Item-9 — preparar modo expert
        if mode == "expert":
            log.info("[ISIC] H1  — Loss: CrossEntropyLoss con pesos (NO BCEWithLogitsLoss).")
            log.info("[ISIC] H1  — 9 neuronas softmax. Metrica: BMCA (no Accuracy global).")
            log.info("[ISIC] H6/Item-9 — Augmentation diferenciado:\n"
                     "    Mayoría: flip H, rot±30°, ColorJitter moderado\n"
                     "    Minoría (DF/VASC/SCC/AK): flip H+V, rot 0/90/180/270°, "
                     "ColorJitter agresivo, RandomAffine, RandomErasing (Cutout)\n"
                     "    Usar get_weighted_sampler() en DataLoader de FASE 2.")

            self.df = self.df.copy()
            self.df["class_label"] = self.df[self.CLASSES].values.argmax(axis=1)
            self.df["is_minority"]  = self.df["class_label"].isin(self.MINORITY_IDX)

            dist  = self.df["class_label"].value_counts().sort_index()
            total = len(self.df)
            log.info("[ISIC] Distribución de clases:")
            for i, cls in enumerate(self.CLASSES):
                count = dist.get(i, 0)
                bar   = "█" * max(1, int(30 * count / total))
                tag   = " ⚠ MINORÍA" if i in self.MINORITY_IDX else ""
                log.info(f"    {cls:<5}(idx {i}): {count:>6,} ({100*count/total:.1f}%) {bar}{tag}")

            counts  = np.array([dist.get(i, 1) for i in range(self.N_TRAIN_CLS)], dtype=float)
            weights = total / (self.N_TRAIN_CLS * counts)
            weights_full = np.append(weights, 1.0)
            self.class_weights = torch.tensor(weights_full, dtype=torch.float32)
            log.info(f"[ISIC] H1 — class_weights: "
                     f"min={self.class_weights.min():.2f}({self.CLASSES[self.class_weights.argmin()]}) | "
                     f"max={self.class_weights.max():.2f}({self.CLASSES[self.class_weights.argmax()]})")

            all_zero = (self.df[self.CLASSES].sum(axis=1) == 0).sum()
            if all_zero:
                log.warning(f"[ISIC] {all_zero} filas sum=0 → argmax asignará clase 0 (MEL).")

    def _source_audit(self):
        """H3/Item-7 — Infiere fuente por naming e imprime estadísticas de bias de dominio."""
        names    = self.df["image"]
        msk_mask = names.str.endswith("_downsampled")
        try:
            numeric_ids = names.str.extract(r"ISIC_(\d+)")[0].astype(float)
            ham_mask    = (~msk_mask) & (numeric_ids <= 67977)
            bcn_mask    = (~msk_mask) & (numeric_ids > 67977)
        except Exception:
            ham_mask = pd.Series([False] * len(names), index=names.index)
            bcn_mask = ~msk_mask & ~ham_mask

        n_ham, n_bcn, n_msk = ham_mask.sum(), bcn_mask.sum(), msk_mask.sum()
        total = len(names)
        log.info(
            f"[ISIC] H3/Item-7 — Auditoría de fuentes (heurística por ID range):\n"
            f"    HAM10000 (Viena, 600×450, recortado)      : {n_ham:>5,} ({100*n_ham/total:.1f}%)\n"
            f"    BCN_20000 (Barcelona, 1024×1024, dermoscop): {n_bcn:>5,} ({100*n_bcn/total:.1f}%)\n"
            f"    MSK (NYC, variable, _downsampled)          : {n_msk:>5,} ({100*n_msk/total:.1f}%)\n"
            f"    Contramedida activa: apply_circular_crop() (BCN) + ColorJitter (todas).\n"
            f"    ⚠ Para inspección visual: samplea 5-10 imgs de cada fuente antes de FASE 2."
        )

    @staticmethod
    def build_lesion_split(gt_csv, img_dir=None, metadata_csv=None,
                           frac_train=0.8, random_state=42):
        """
        H2 + H4 — Split por lesion_id + deduplicación MD5.
        """
        df_gt = pd.read_csv(gt_csv)
        log.info(f"[ISIC/split] GT CSV: {len(df_gt):,} filas")

        # H4: eliminar duplicados conocidos por ID
        before = len(df_gt)
        df_gt  = df_gt[~df_gt["image"].isin(ISICDataset.KNOWN_DUPLICATES)].reset_index(drop=True)
        if before - len(df_gt):
            log.info(f"[ISIC/split] H4 — {before-len(df_gt)} KNOWN_DUPLICATES eliminados.")

        # H4: deduplicación por hash MD5 de archivos reales
        if img_dir is not None:
            img_path_obj = Path(img_dir)
            log.info(f"[ISIC/split] H4 — Calculando hashes MD5 de {len(df_gt):,} imgs "
                     f"(puede tardar ~1-2 min)...")
            hashes, dup_ids, missing = {}, set(), 0
            for _, row in df_gt.iterrows():
                fpath = img_path_obj / f"{row['image']}.jpg"
                if not fpath.exists():
                    missing += 1
                    continue
                try:
                    md5 = hashlib.md5(fpath.read_bytes()).hexdigest()
                    if md5 in hashes:
                        dup_ids.add(row["image"])
                        log.debug(f"[ISIC/split] H4 — Dup: '{row['image']}' == '{hashes[md5]}'")
                    else:
                        hashes[md5] = row["image"]
                except Exception as e:
                    log.warning(f"[ISIC/split] H4 — No se pudo hashear '{fpath}': {e}")
            if missing:
                log.warning(f"[ISIC/split] H4 — {missing} archivos no encontrados para hashing.")
            if dup_ids:
                df_gt = df_gt[~df_gt["image"].isin(dup_ids)].reset_index(drop=True)
                log.info(f"[ISIC/split] H4 — {len(dup_ids)} duplicados MD5 eliminados. "
                         f"Quedan {len(df_gt):,} imgs únicas.")
            else:
                log.info("[ISIC/split] H4 — Sin duplicados MD5 detectados ✓")
        else:
            log.warning("[ISIC/split] H4 — img_dir=None: dedup MD5 OMITIDA. "
                        "Pasa --isic_imgs a build_lesion_split para deduplicación completa.")

        # H2: split por lesion_id
        if metadata_csv is not None:
            df_meta = pd.read_csv(metadata_csv, usecols=["image", "lesion_id"])
            df_gt   = df_gt.merge(df_meta, on="image", how="left")
            n_with  = df_gt["lesion_id"].notna().sum()
            n_wo    = df_gt["lesion_id"].isna().sum()
            log.info(f"[ISIC/split] H2 — lesion_id: {n_with:,} con ID | {n_wo:,} sin ID (→ únicos)")

            mask_na = df_gt["lesion_id"].isna()
            df_gt.loc[mask_na, "lesion_id"] = df_gt.loc[mask_na, "image"].apply(
                lambda x: f"_unique_{x}")

            unique_lesions = df_gt["lesion_id"].unique()
            rng = np.random.RandomState(random_state)
            rng.shuffle(unique_lesions)
            n_tr          = int(len(unique_lesions) * frac_train)
            train_lesions = set(unique_lesions[:n_tr])
            val_lesions   = set(unique_lesions[n_tr:])

            train_df = df_gt[df_gt["lesion_id"].isin(train_lesions)].copy()
            val_df   = df_gt[df_gt["lesion_id"].isin(val_lesions)].copy()

            shared = set(train_df["lesion_id"]) & set(val_df["lesion_id"])
            if shared:
                log.error(f"[ISIC/split] H2 — ¡LEAKAGE! {len(shared)} lesion_ids compartidos.")
            else:
                log.info(f"[ISIC/split] H2 — Split por lesion_id ✓ | "
                         f"train: {len(train_lesions):,} lesiones ({len(train_df):,} imgs) | "
                         f"val: {len(val_lesions):,} lesiones ({len(val_df):,} imgs)")
        else:
            log.warning("[ISIC/split] H2 — metadata_csv=None: split aleatorio por imagen. "
                        "Pasa --isic_metadata para split correcto por lesion_id.")
            idx_all = np.arange(len(df_gt))
            rng     = np.random.RandomState(random_state)
            rng.shuffle(idx_all)
            n_tr     = int(len(idx_all) * frac_train)
            train_df = df_gt.iloc[idx_all[:n_tr]].copy()
            val_df   = df_gt.iloc[idx_all[n_tr:]].copy()

        # H5: verificar UNK=0 en ambos splits
        for sname, sdf in [("train", train_df), ("val", val_df)]:
            if "UNK" in sdf.columns:
                n_unk = (sdf["UNK"] != 0).sum()
                if n_unk:
                    log.warning(f"[ISIC/split] H5 — {n_unk} filas UNK≠0 en '{sname}'. "
                                f"UNK solo debería estar en el test set oficial.")
                else:
                    log.debug(f"[ISIC/split] H5 — UNK=0 en '{sname}' ✓")

        log.info(f"[ISIC/split] Final — train: {len(train_df):,} | val: {len(val_df):,}")
        return train_df, val_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image"]
        img_path = self.img_dir / f"{img_name}.jpg"
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning(f"[ISIC] Error abriendo '{img_path}': {e}. Reemplazando con tensor cero.")
            dummy = torch.zeros(3, 224, 224)
            if self.mode == "embedding":
                return dummy, self.expert_id, img_name
            label = int(self.df.loc[idx, "class_label"]) if "class_label" in self.df.columns else 0
            return dummy, label, img_name

        # H3/Item-8 — eliminar bordes negros BCN_20000 (idempotente para HAM/MSK)
        if self.apply_bcn_crop:
            img = apply_circular_crop(img)

        if self.mode == "embedding":
            return self.transform(img), self.expert_id, img_name
        else:
            is_minority = bool(self.df.loc[idx, "is_minority"])
            tf  = self.transform_minority if is_minority else self.transform_standard
            return tf(img), int(self.df.loc[idx, "class_label"]), img_name

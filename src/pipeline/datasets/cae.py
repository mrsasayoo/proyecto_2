"""
MultimodalCAEDataset — Dataset multi-modal para entrenamiento del CAE (Expert 5).

Carga imágenes de 5 dominios: Chest (NIH), ISIC, OA, LUNA16, Páncreas.
Todas las imágenes se normalizan a [3, 224, 224] float32.

Para datos 3D:
  - LUNA16 (.npy patches [64,64,64]): slice central axial -> replicate 3ch -> resize [3,224,224]
  - Páncreas (.nii.gz): HU clip [-150,250] -> slice central -> replicate 3ch -> resize [3,224,224]

Schema de cae_splits.csv:
  ruta_imagen       — path relativo desde project_root
  dataset_origen    — nih | isic | oa | luna | pancreas
  split             — train | val | test
  expert_id         — 0..4
  tipo_dato         — 2d_image | 3d_patch_npy | 3d_volume_nifti
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

log = logging.getLogger("cae_dataset")

# ── ImageNet stats para normalización ─────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# ── HU clip para páncreas (abdomen) ──────────────────────────────────
_HU_ABDOMEN_LO = -150
_HU_ABDOMEN_HI = 250


class MultimodalCAEDataset(Dataset):
    """
    Dataset multi-modal para entrenamiento del CAE.
    Carga imágenes de 5 dominios: Chest, ISIC, OA, LUNA16, Páncreas.
    Todas las imágenes se normalizan a [3, 224, 224] float32.

    Para datos 3D:
    - LUNA16 (.npy patches [1,64,64,64] o [64,64,64]):
        slice central axial -> [64,64] -> replicate 3ch -> resize [3,224,224]
    - Páncreas (.nii.gz):
        HU clip [-150, 250] -> slice central del volumen -> replicate 3ch -> resize [3,224,224]

    Args:
        csv_path: ruta a cae_splits.csv
        split: "train", "val" o "test"
        img_size: tamaño de salida (default=224)
        project_root: raíz del proyecto para resolver paths relativos
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        img_size: int = 224,
        project_root: str = None,
    ):
        assert split in ("train", "val", "test"), (
            f"[CAE] split debe ser 'train', 'val' o 'test', recibido: '{split}'"
        )

        self.split = split
        self.img_size = img_size
        self.project_root = Path(project_root) if project_root else Path.cwd()

        # ── Cargar CSV y filtrar por split ─────────────────────────────
        df = pd.read_csv(csv_path)
        required_cols = {"ruta_imagen", "dataset_origen", "split", "tipo_dato"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"[CAE] Columnas faltantes en CSV: {missing_cols}. "
                f"Columnas encontradas: {list(df.columns)}"
            )

        self.df = df[df["split"] == split].reset_index(drop=True)
        log.info(f"[CAE] Split '{split}': {len(self.df):,} muestras | CSV: {csv_path}")

        # Log distribución por dataset
        if len(self.df) > 0:
            counts = self.df["dataset_origen"].value_counts()
            dist_str = " | ".join(f"{k}: {v:,}" for k, v in counts.items())
            log.info(f"[CAE] Distribución: {dist_str}")

        # ── Transforms 2D ─────────────────────────────────────────────
        if split == "train":
            self.transform_2d = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ]
            )
        else:
            self.transform_2d = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ]
            )

        # ── Transform para slices 3D convertidos a PIL ────────────────
        # Sin augmentation de color/flip (no tiene sentido clínico en CT slices)
        self.transform_3d_slice = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_2d_image(self, abs_path: Path) -> torch.Tensor:
        """Carga una imagen 2D nativa (PNG, JPG) y aplica transforms."""
        img = Image.open(abs_path).convert("RGB")
        return self.transform_2d(img)

    def _load_luna_patch(self, abs_path: Path) -> torch.Tensor:
        """
        Carga un parche LUNA16 .npy y extrae el slice central axial.

        Shape esperado: [64,64,64] o [1,64,64,64].
        Slice central: arr[D//2, :, :] -> [H, W]
        Normalizar a [0,1], convertir a PIL RGB, resize a [3,224,224].
        """
        arr = np.load(abs_path).astype(np.float32)

        # Eliminar dimensión de canal si existe: [1,D,H,W] -> [D,H,W]
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 4:
            # Shape [1,64,64,64] con canal no-unitario: tomar primer canal
            arr = arr[0]

        if arr.ndim != 3:
            log.warning(
                f"[CAE/LUNA] Shape inesperado {arr.shape} en '{abs_path.name}'. "
                f"Retornando tensor cero."
            )
            return torch.zeros(3, self.img_size, self.img_size)

        # Slice central axial
        central_idx = arr.shape[0] // 2
        slice_2d = arr[central_idx, :, :]  # [H, W]

        # Min-max normalización sobre el slice a [0, 1]
        s_min, s_max = slice_2d.min(), slice_2d.max()
        if s_max - s_min > 1e-6:
            slice_2d = (slice_2d - s_min) / (s_max - s_min)
        else:
            slice_2d = np.zeros_like(slice_2d)

        # Convertir a uint8 para PIL
        slice_uint8 = (slice_2d * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(slice_uint8, mode="L").convert("RGB")

        return self.transform_3d_slice(pil_img)

    def _load_pancreas_nifti(self, abs_path: Path) -> torch.Tensor:
        """
        Carga un volumen NIfTI de páncreas y extrae el slice central.

        HU clip [-150, 250] -> normalizar a [0, 1] -> slice central Z
        -> convertir a PIL RGB -> resize a [3, 224, 224].
        """
        import nibabel as nib

        nii = nib.load(str(abs_path))
        vol = nii.get_fdata().astype(np.float32)

        # HU clip [-150, 250] y normalizar a [0, 1]
        vol = np.clip(vol, _HU_ABDOMEN_LO, _HU_ABDOMEN_HI)
        hu_range = _HU_ABDOMEN_HI - _HU_ABDOMEN_LO
        vol = (vol - _HU_ABDOMEN_LO) / hu_range

        # Slice central: vol tiene shape [H, W, D] en nibabel
        # Tomar slice central del último eje (axial)
        central_idx = vol.shape[2] // 2
        slice_2d = vol[:, :, central_idx]  # [H, W]

        # Convertir a uint8 para PIL
        slice_uint8 = (slice_2d * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(slice_uint8, mode="L").convert("RGB")

        return self.transform_3d_slice(pil_img)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            (img_tensor [3, 224, 224], path_str)
            Sin label — el autoencoder no necesita clase.
        """
        row = self.df.iloc[idx]
        rel_path = row["ruta_imagen"]
        tipo_dato = row["tipo_dato"]
        abs_path = self.project_root / rel_path

        try:
            if not abs_path.exists():
                log.warning(
                    f"[CAE] Archivo no encontrado: '{abs_path}'. Skipeando con tensor cero."
                )
                return torch.zeros(3, self.img_size, self.img_size), str(rel_path)

            if tipo_dato == "2d_image":
                img = self._load_2d_image(abs_path)
            elif tipo_dato == "3d_patch_npy":
                img = self._load_luna_patch(abs_path)
            elif tipo_dato == "3d_volume_nifti":
                img = self._load_pancreas_nifti(abs_path)
            else:
                log.warning(
                    f"[CAE] tipo_dato desconocido '{tipo_dato}' para '{rel_path}'. "
                    f"Retornando tensor cero."
                )
                return torch.zeros(3, self.img_size, self.img_size), str(rel_path)

            return img, str(rel_path)

        except Exception as e:
            log.warning(
                f"[CAE] Error cargando '{rel_path}' (tipo={tipo_dato}): {e}. "
                f"Retornando tensor cero."
            )
            return torch.zeros(3, self.img_size, self.img_size), str(rel_path)

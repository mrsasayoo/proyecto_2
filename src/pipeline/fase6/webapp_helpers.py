"""
webapp_helpers.py
-----------------
Helpers para paso11_webapp.py:
- Metadatos de expertos
- Contador de load balance en memoria
- Preprocesamiento de imagen para la webapp
- Utilidades de carga de ablation results
- Mock inference para dry-run
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# =====================================================================
# 1. EXPERT_METADATA
# =====================================================================
EXPERT_METADATA: dict[int, dict] = {
    0: {
        "name": "Experto 1 — Chest X-Ray",
        "architecture": "ConvNeXt-Tiny",
        "dataset": "NIH ChestXray14",
        "n_classes": 14,
        "modality": "2D",
        "class_names": [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ],
    },
    1: {
        "name": "Experto 2 — Dermatología (ISIC)",
        "architecture": "EfficientNet-B3",
        "dataset": "ISIC 2019",
        "n_classes": 9,
        "modality": "2D",
        "class_names": ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"],
    },
    2: {
        "name": "Experto 3 — OA Knee",
        "architecture": "EfficientNet-B0",
        "dataset": "Osteoarthritis Knee",
        "n_classes": 5,
        "modality": "2D",
        "class_names": [
            "Normal (KL0)",
            "Dudoso (KL1)",
            "Leve (KL2)",
            "Moderado (KL3)",
            "Severo (KL4)",
        ],
    },
    3: {
        "name": "Experto 4 — Nódulos Pulmonares (LUNA16)",
        "architecture": "MC3-18",
        "dataset": "LUNA16",
        "n_classes": 2,
        "modality": "3D",
        "class_names": ["Benigno", "Maligno"],
    },
    4: {
        "name": "Experto 5 — Páncreas (MSD)",
        "architecture": "Swin3D-Tiny",
        "dataset": "Medical Segmentation Decathlon",
        "n_classes": 2,
        "modality": "3D",
        "class_names": ["Normal", "PDAC"],
    },
    5: {
        "name": "Experto 6 — CAE / OOD",
        "architecture": "ConvAutoEncoder 2D",
        "dataset": "5 datasets combinados",
        "n_classes": 0,
        "modality": "2D",
        "class_names": [],
    },
}

# Placeholder cuando ablation_results.json no existe
ABLATION_PLACEHOLDER: dict = {
    "note": "ablation_results.json no disponible — entrenamiento pendiente",
    "results": {
        "TopK Router": {"routing_accuracy": None, "latency_ms": None},
        "Soft Router": {"routing_accuracy": None, "latency_ms": None},
        "Noisy TopK Router": {"routing_accuracy": None, "latency_ms": None},
        "Linear Router (ViT)": {"routing_accuracy": None, "latency_ms": None},
    },
}


# =====================================================================
# 2. LoadBalanceCounter
# =====================================================================
class LoadBalanceCounter:
    """Contador en memoria de uso de expertos (se resetea al reiniciar el servicio)."""

    def __init__(self, n_experts: int = 6):
        self.n_experts = n_experts
        self.counts: dict[int, int] = {i: 0 for i in range(n_experts)}

    def update(self, expert_ids: list[int]) -> None:
        """Registra qué expertos fueron usados en este batch."""
        for eid in expert_ids:
            if 0 <= eid < self.n_experts:
                self.counts[eid] += 1

    def get_counts(self) -> dict[int, int]:
        return dict(self.counts)

    def get_frequencies(self) -> dict[int, float]:
        """Frecuencias relativas f_i (suma=1 si hay conteos)."""
        total = sum(self.counts.values())
        if total == 0:
            return {i: 0.0 for i in range(self.n_experts)}
        return {i: c / total for i, c in self.counts.items()}

    def get_max_min_ratio(self) -> float:
        """max(f_i) / min(f_i) — 0.0 si no hay datos."""
        freqs = list(self.get_frequencies().values())
        nonzero = [f for f in freqs if f > 0]
        if len(nonzero) < 2:
            return 0.0
        return max(nonzero) / min(nonzero)

    def reset(self) -> None:
        self.counts = {i: 0 for i in range(self.n_experts)}


# =====================================================================
# 3. preprocess_image_for_webapp
# =====================================================================
def preprocess_image_for_webapp(
    image_input,  # str | Path | np.ndarray | PIL.Image
    target_size: tuple[int, int] = (224, 224),
) -> tuple[torch.Tensor, dict]:
    """
    Preprocesa imagen para la webapp.

    IMPORTANTE: La detección de modalidad (2D/3D) se hace SOLO por rank del tensor.
    No se usa el nombre del archivo para routing — solo para carga.

    Soporta:
      - str / Path: carga desde disco (PNG/JPEG con PIL, NIfTI con SimpleITK)
      - np.ndarray: convierte directamente
      - PIL.Image: convierte a numpy y procesa

    Returns:
        (tensor, metadata_dict) donde tensor es [1,C,H,W] (2D) o [1,C,D,H,W] (3D)
        metadata_dict: {"original_shape", "adapted_shape", "modality", "is_nifti"}
    """
    is_nifti = False
    original_shape: tuple = ()

    # --- Carga desde disco ---
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        suffix = path.suffix.lower()

        if suffix in (".nii",) or path.name.endswith(".nii.gz"):
            # NIfTI — cargar con SimpleITK
            try:
                import SimpleITK as sitk
            except ImportError as e:
                logger.error("SimpleITK no disponible para cargar NIfTI: %s", e)
                dummy = torch.zeros(1, 1, 64, 64, 64)
                meta = {
                    "original_shape": (),
                    "adapted_shape": tuple(dummy.shape),
                    "modality": "3D",
                    "is_nifti": True,
                    "error": f"SimpleITK no disponible: {e}",
                }
                return dummy, meta

            img_sitk = sitk.ReadImage(str(path))
            arr = sitk.GetArrayFromImage(img_sitk)  # [D, H, W] or [D, H, W, C]
            is_nifti = True
            original_shape = arr.shape
            image_input = arr
        else:
            # PNG / JPEG — cargar con PIL
            try:
                from PIL import Image
            except ImportError as e:
                logger.error("PIL no disponible: %s", e)
                dummy = torch.zeros(1, 3, *target_size)
                meta = {
                    "original_shape": (),
                    "adapted_shape": tuple(dummy.shape),
                    "modality": "2D",
                    "is_nifti": False,
                    "error": f"PIL no disponible: {e}",
                }
                return dummy, meta

            pil_img = Image.open(str(path)).convert("RGB")
            arr = np.array(pil_img)
            original_shape = arr.shape
            image_input = arr

    # --- PIL.Image a numpy ---
    try:
        from PIL import Image as _PILImage

        if isinstance(image_input, _PILImage.Image):
            image_input = np.array(image_input.convert("RGB"))
            if not original_shape:
                original_shape = image_input.shape
    except ImportError:
        pass  # PIL no disponible, no es PIL.Image

    # --- numpy → torch ---
    if isinstance(image_input, np.ndarray):
        if not original_shape:
            original_shape = image_input.shape
        arr = image_input.astype(np.float32)

        # Detectar modalidad SOLO por rank del array
        if arr.ndim == 2:
            # Grayscale 2D [H, W] → [1, 3, H, W] (replicar a 3ch)
            arr = np.stack([arr, arr, arr], axis=0)  # [3, H, W]
            tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, 3, H, W]
        elif arr.ndim == 3:
            if arr.shape[-1] in (1, 3, 4):
                # [H, W, C] — imagen 2D con canales
                if arr.shape[-1] == 4:
                    arr = arr[:, :, :3]  # drop alpha
                if arr.shape[-1] == 1:
                    arr = np.concatenate([arr, arr, arr], axis=-1)
                arr = arr.transpose(2, 0, 1)  # [C, H, W]
                tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, C, H, W]
            else:
                # [D, H, W] — volumen 3D sin canal
                tensor = (
                    torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                )  # [1, 1, D, H, W]
        elif arr.ndim == 4:
            # [D, H, W, C] — volumen 3D con canales
            arr = arr.transpose(3, 0, 1, 2)  # [C, D, H, W]
            tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, C, D, H, W]
        else:
            logger.warning("Array con ndim=%d no soportado, tratando como 2D", arr.ndim)
            arr_2d = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr
            tensor = torch.from_numpy(arr_2d).float().unsqueeze(0).unsqueeze(0)
    elif isinstance(image_input, torch.Tensor):
        if not original_shape:
            original_shape = tuple(image_input.shape)
        tensor = image_input.float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # [C,H,W] → [1,C,H,W]
        elif tensor.ndim == 4:
            pass  # ya [B,C,H,W] o necesita check
        elif tensor.ndim == 5:
            pass  # ya [B,C,D,H,W]
    else:
        logger.error("Tipo de entrada no soportado: %s", type(image_input))
        dummy = torch.zeros(1, 3, *target_size)
        return dummy, {
            "original_shape": (),
            "adapted_shape": tuple(dummy.shape),
            "modality": "2D",
            "is_nifti": False,
            "error": f"Tipo no soportado: {type(image_input)}",
        }

    # --- Detección de modalidad por rank del tensor ---
    if tensor.ndim == 4:
        modality = "2D"
        # Resize 2D al target_size
        try:
            import torch.nn.functional as F

            tensor = F.interpolate(
                tensor, size=target_size, mode="bilinear", align_corners=False
            )
        except Exception as e:
            logger.warning("Resize 2D fallido: %s", e)

        # Normalización ImageNet para 2D
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # Solo normalizar si tiene 3 canales
        if tensor.shape[1] == 3:
            tensor = (tensor - mean) / std

    elif tensor.ndim == 5:
        modality = "3D"
        # Clipping HU para 3D: [-1000, 400] → norm [0, 1]
        tensor = tensor.clamp(-1000.0, 400.0)
        tensor = (tensor - (-1000.0)) / (400.0 - (-1000.0))
    else:
        modality = "2D"
        logger.warning("Tensor con ndim=%d inesperado, asumiendo 2D", tensor.ndim)

    adapted_shape = tuple(tensor.shape)

    metadata = {
        "original_shape": original_shape,
        "adapted_shape": adapted_shape,
        "modality": modality,
        "is_nifti": is_nifti,
    }

    logger.debug(
        "Preprocesado: %s → %s (modality=%s, nifti=%s)",
        original_shape,
        adapted_shape,
        modality,
        is_nifti,
    )
    return tensor, metadata


# =====================================================================
# 4. load_ablation_results
# =====================================================================
def load_ablation_results(path: Optional[str] = None) -> dict:
    """
    Carga ablation_results.json si existe.
    Si path es None o el archivo no existe, retorna ABLATION_PLACEHOLDER.
    """
    if path is None:
        logger.info(
            "No se proporcionó ruta de ablation_results.json — usando placeholder"
        )
        return dict(ABLATION_PLACEHOLDER)

    p = Path(path)
    if not p.exists():
        logger.warning(
            "ablation_results.json no encontrado en %s — usando placeholder", p
        )
        return dict(ABLATION_PLACEHOLDER)

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("ablation_results.json cargado desde %s", p)
        return data
    except Exception as e:
        logger.warning(
            "Error cargando ablation_results.json: %s — usando placeholder", e
        )
        return dict(ABLATION_PLACEHOLDER)


# =====================================================================
# 5. create_mock_inference_result
# =====================================================================
def create_mock_inference_result(n_experts: int = 5) -> dict:
    """Crea un resultado de inferencia simulado (dry-run / testing)."""
    gates = torch.softmax(torch.randn(1, n_experts), dim=-1)
    entropy = -(gates * torch.log(gates + 1e-8)).sum(dim=-1)
    expert_id = int(gates.argmax(dim=-1).item())

    # Asegurar que el expert_id existe en EXPERT_METADATA
    n_classes = EXPERT_METADATA.get(expert_id, {}).get("n_classes", 1)
    if n_classes == 0:
        n_classes = 1  # CAE no tiene clases, usar 1 como dummy

    return {
        "logits": [torch.randn(1, n_classes)],
        "expert_ids": [expert_id],
        "gates": gates,
        "entropy": entropy,
        "is_ood": bool((entropy > 0.8).item()),
        "_is_mock": True,
    }


# =====================================================================
# 6. format_confidence
# =====================================================================
def format_confidence(logits: torch.Tensor, expert_id: int) -> tuple[str, float]:
    """
    Retorna (label_str, confidence_pct).

    Para multilabel (Chest, expert_id=0) usa sigmoid + top-k.
    Para multiclase (ISIC, OA, LUNA, Pancreas) usa softmax + argmax.
    Para CAE (expert_id=5) retorna ("OOD/CAE", 0.0).
    """
    if expert_id == 5:
        return "OOD/CAE", 0.0

    meta = EXPERT_METADATA.get(expert_id)
    if meta is None:
        return "Desconocido", 0.0

    class_names = meta["class_names"]

    if logits.ndim > 2:
        logits = logits.squeeze()
    if logits.ndim == 0:
        return class_names[0] if class_names else "N/A", 0.0

    logits_1d = logits.squeeze(0) if logits.ndim == 2 else logits

    if expert_id == 0:
        # Multilabel — sigmoid + top-k
        probs = torch.sigmoid(logits_1d)
        topk_vals, topk_idx = probs.topk(min(3, len(class_names)))
        labels = []
        for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            if idx < len(class_names):
                labels.append(f"{class_names[idx]}({val * 100:.1f}%)")
        label_str = ", ".join(labels) if labels else "Sin hallazgo"
        confidence = float(topk_vals[0].item()) * 100.0
    else:
        # Multiclase — softmax + argmax
        probs = torch.softmax(logits_1d, dim=-1)
        confidence_val, pred_idx = probs.max(dim=-1)
        idx = int(pred_idx.item())
        if idx < len(class_names):
            label_str = class_names[idx]
        else:
            label_str = f"Clase {idx}"
        confidence = float(confidence_val.item()) * 100.0

    return label_str, confidence

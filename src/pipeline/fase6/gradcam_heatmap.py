"""
gradcam_heatmap.py
------------------
Generación de Grad-CAM y Attention Rollout para el Paso 9.

Expertos 2D (0, 1, 2): Grad-CAM sobre última capa conv del backbone
Experto 3D Swin3D (3): Attention Rollout (aproximado con Grad-CAM 3D)
Experto 3D MC3 (4 - LUNA): Grad-CAM 3D

Produce figuras PNG en FIGURES_DIR para el reporte.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.pipeline.fase6.fase6_config import FIGURES_DIR

logger = logging.getLogger(__name__)


class GradCAMExtractor:
    """
    Extractor de Grad-CAM para un módulo target dentro de un experto.

    Args:
        model: nn.Module — experto a analizar
        target_layer: nn.Module — capa sobre la que calcular Grad-CAM
        expert_name: str — para nombrar los archivos de salida
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        expert_name: str,
    ):
        self.model = model
        self.target_layer = target_layer
        self.expert_name = expert_name
        self._activations = None
        self._gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compute(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Computa el mapa Grad-CAM.

        Args:
            x: [1, C, H, W] o [1, C, D, H, W]
            class_idx: índice de clase target. Si None, usa la clase predicha.

        Returns:
            cam: np.ndarray de shape [H, W] (normalizado 0-1) para 2D
                 np.ndarray de shape [D, H, W] para 3D
        """
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        if class_idx is None:
            class_idx = int(logits.squeeze().argmax().item())

        self.model.zero_grad()
        score = logits[0, class_idx] if logits.ndim > 1 else logits[0]
        score.backward()

        if self._gradients is None or self._activations is None:
            logger.warning(f"Grad-CAM hooks did not fire for {self.expert_name}")
            return np.zeros((1, 1))

        # Pool gradients over spatial dims
        # 2D: [B, C, H, W] → pool over H, W
        # 3D: [B, C, D, H, W] → pool over D, H, W
        grads = self._gradients  # [B, C, ...]
        acts = self._activations  # [B, C, ...]

        n_spatial = grads.ndim - 2  # 2 for 2D, 3 for 3D
        pool_dims = tuple(range(2, 2 + n_spatial))
        weights = grads.mean(dim=pool_dims, keepdim=True)  # [B, C, 1, ...1]

        cam = (weights * acts).sum(dim=1, keepdim=False)  # [B, ...]
        cam = cam.squeeze(0).cpu().numpy()  # spatial dims only
        cam = np.maximum(cam, 0)

        # Normalize
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def save_overlay(
        self,
        original_image: np.ndarray,
        cam: np.ndarray,
        filename: str,
        title: str = "",
    ) -> None:
        """
        Guarda overlay de Grad-CAM sobre imagen original como PNG.
        Solo funciona para CAM 2D (H, W).
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from PIL import Image
            import cv2
        except ImportError as e:
            logger.warning(f"Cannot save Grad-CAM overlay: {e}")
            return

        Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

        # Resize CAM to match image
        if cam.ndim == 2:
            h, w = original_image.shape[:2]
            cam_resized = cv2.resize(cam, (w, h))

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(
                original_image, cmap="gray" if original_image.ndim == 2 else None
            )
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(
                original_image, cmap="gray" if original_image.ndim == 2 else None
            )
            axes[1].imshow(cam_resized, cmap="jet", alpha=0.5)
            axes[1].set_title(f"Grad-CAM {title}")
            axes[1].axis("off")

            out_path = Path(FIGURES_DIR) / filename
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Grad-CAM overlay saved → {out_path}")
        else:
            logger.warning(
                f"3D CAM shape {cam.shape} — skipping overlay (not supported yet)"
            )


def generate_gradcam_samples(
    moe_system: nn.Module,
    expert_idx: int,
    expert_name: str,
    sample_images: torch.Tensor,
    dry_run: bool = False,
) -> None:
    """
    Genera y guarda muestras de Grad-CAM para un experto dado.

    Args:
        moe_system: MoESystem
        expert_idx: índice del experto (0-4)
        expert_name: nombre legible
        sample_images: [N, C, H, W] o [N, C, D, H, W] — hasta 8 imágenes de muestra
        dry_run: si True, skip
    """
    if dry_run:
        logger.info(
            f"[DRY-RUN] Skipping Grad-CAM for expert {expert_idx} ({expert_name})"
        )
        return

    expert = moe_system.experts[expert_idx]

    # Encontrar última capa convolucional del backbone
    target_layer = _find_last_conv(expert)
    if target_layer is None:
        logger.warning(
            f"Could not find conv layer for expert {expert_idx} — skipping Grad-CAM"
        )
        return

    extractor = GradCAMExtractor(
        model=expert,
        target_layer=target_layer,
        expert_name=expert_name,
    )

    n_samples = min(sample_images.shape[0], 4)

    try:
        for i in range(n_samples):
            xi = sample_images[i : i + 1]
            cam = extractor.compute(xi)

            # Convertir tensor a numpy para overlay
            img_np = xi.squeeze().cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                img_np = img_np.transpose(1, 2, 0)  # CHW → HWC
                if img_np.shape[-1] == 1:
                    img_np = img_np.squeeze(-1)

            extractor.save_overlay(
                original_image=img_np,
                cam=cam,
                filename=f"gradcam_samples_{expert_name}_sample{i}.png",
                title=f"{expert_name} sample {i}",
            )
    finally:
        extractor.remove_hooks()

    logger.info(f"Grad-CAM generation complete for expert {expert_name}")


def _find_last_conv(module: nn.Module) -> Optional[nn.Module]:
    """Encuentra la última capa Conv2d o Conv3d en el módulo."""
    last_conv = None
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            last_conv = m
    return last_conv

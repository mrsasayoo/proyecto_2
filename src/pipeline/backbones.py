"""
Carga y congelamiento de backbones ViT/Swin/CvT para extracción de embeddings.

Incluye el patch de CvT-13 para compatibilidad con timm >= 0.9.x.
"""

import sys
import os
import logging

import torch
import timm

from config import BACKBONE_CONFIGS

log = logging.getLogger("fase0")

# === PATCH CvT-13 — inyectado para compatibilidad con timm >= 0.9.x ===
# timm no tiene cvt_13 en versiones recientes. Este patch intercepta la llamada
# a timm.create_model('cvt_13', ...) y la redirige a CvT13Wrapper (HuggingFace).

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))

_original_timm_create = timm.create_model


def _patched_create_model(model_name, *args, **kwargs):
    if model_name == 'cvt_13':
        log.info("[Backbone/patch] cvt_13 interceptado → CvT13Wrapper (HuggingFace)")
        from cvt13_backbone import build_cvt13
        _device = kwargs.get('device', 'cpu')
        _pretrained = kwargs.get('pretrained', True)
        return build_cvt13(pretrained=_pretrained, device=_device)
    return _original_timm_create(model_name, *args, **kwargs)


timm.create_model = _patched_create_model
# === FIN PATCH ===


def load_frozen_backbone(backbone_name="vit_tiny_patch16_224", device="cuda"):
    """
    Carga un backbone ViT/Swin/CvT preentrenado (timm) y lo congela completamente.
    Solo se usa para forward pass — sin gradientes.

    Retorna:
      model   — backbone listo para inferencia, en `device`
      d_model — dimensión del CLS token (varía según backbone)
    """
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(
            f"Backbone '{backbone_name}' no reconocido.\n"
            f"Opciones válidas: {list(BACKBONE_CONFIGS.keys())}"
        )

    expected_d = BACKBONE_CONFIGS[backbone_name]["d_model"]
    vram_est   = BACKBONE_CONFIGS[backbone_name]["vram_gb"]
    log.info(f"[Backbone] Seleccionado  : {backbone_name}")
    log.info(f"[Backbone] d_model esp.  : {expected_d}")
    log.info(f"[Backbone] VRAM estimada : ~{vram_est} GB")

    model = timm.create_model(
        backbone_name,
        pretrained=True,
        num_classes=0          # elimina la cabeza de clasificación → devuelve CLS token
    )

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)

    # Verificar que el congelamiento fue efectivo
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        log.error(f"[Backbone] ¡{len(trainable)} parámetros con requires_grad=True tras el congelamiento! "
                  f"Los embeddings NO serán reproducibles. Primeros 3: {trainable[:3]}")
    else:
        log.debug(f"[Backbone] Congelamiento verificado: 0 parámetros entrenables.")

    # Verificar d_model con un forward pass dummy
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)

    actual_d = out.shape[1]
    if actual_d != expected_d:
        log.warning(f"[Backbone] d_model real ({actual_d}) difiere del esperado ({expected_d}). "
                    f"Actualiza BACKBONE_CONFIGS si agregaste un backbone nuevo.")
    else:
        log.debug(f"[Backbone] d_model verificado: {actual_d} ✓")

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"[Backbone] d_model real (CLS dim) : {actual_d}")
    log.info(f"[Backbone] Parámetros congelados  : {total_params:,}")

    return model, actual_d

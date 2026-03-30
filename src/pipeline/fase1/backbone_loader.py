"""
backbone_loader.py — Carga y Congelamiento del Backbone
=======================================================

Responsabilidad única: recibir un nombre de backbone y devolver un
modelo en modo inferencia pura, con todos los parámetros congelados
y la dimensión de salida verificada empíricamente.

Los backbones se construyen desde cero (pesos aleatorios) para cumplir
con el requisito del proyecto de no usar pesos preentrenados.

No sabe nada de CvT-13 específicamente — la compatibilidad se activa
importando backbone_cvt13 antes de llamar a este módulo.
"""

import logging

import torch
import timm

from fase1_config import BACKBONE_CONFIGS

log = logging.getLogger("fase1")


def load_frozen_backbone(backbone_name="vit_tiny_patch16_224", device="cuda"):
    """
    Carga un backbone ViT/Swin/CvT desde cero (sin pesos preentrenados) y lo congela.

    Returns:
        model   — backbone listo para inferencia en `device`
        d_model — dimensión del CLS token verificada empíricamente
    """
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(
            f"Backbone '{backbone_name}' no reconocido.\n"
            f"Opciones válidas: {list(BACKBONE_CONFIGS.keys())}"
        )

    expected_d = BACKBONE_CONFIGS[backbone_name]["d_model"]
    vram_est = BACKBONE_CONFIGS[backbone_name]["vram_gb"]
    log.info("[Backbone] Seleccionado  : %s", backbone_name)
    log.info("[Backbone] d_model esp.  : %d", expected_d)
    log.info("[Backbone] VRAM estimada : ~%.1f GB", vram_est)

    # Construir modelo desde cero — sin pesos preentrenados (requisito del proyecto)
    model = timm.create_model(
        backbone_name,
        pretrained=False,
        num_classes=0,  # elimina cabeza de clasificación → devuelve CLS token
    )

    # Congelar todos los parámetros
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)

    # Verificar que ningún parámetro quedó entrenable
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        log.error(
            "[Backbone] ¡%d parámetros con requires_grad=True tras congelamiento! "
            "Primeros 3: %s",
            len(trainable),
            trainable[:3],
        )
    else:
        log.debug("[Backbone] Congelamiento verificado: 0 parámetros entrenables.")

    # Forward dummy para verificar d_model
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)

    actual_d = out.shape[1]
    if actual_d != expected_d:
        log.warning(
            "[Backbone] d_model real (%d) difiere del esperado (%d). "
            "Actualiza BACKBONE_CONFIGS si agregaste un backbone nuevo.",
            actual_d,
            expected_d,
        )
    else:
        log.debug("[Backbone] d_model verificado: %d ✓", actual_d)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("[Backbone] d_model real (CLS dim) : %d", actual_d)
    log.info("[Backbone] Parámetros congelados  : %s", f"{total_params:,}")

    return model, actual_d

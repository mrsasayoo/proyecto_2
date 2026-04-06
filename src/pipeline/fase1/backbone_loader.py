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
from pathlib import Path

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

    # OPT-4: torch.compile mejora throughput ~10-30% en CPU (disponible desde PyTorch 2.0)
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log.info("[Backbone] torch.compile() activado (mode=reduce-overhead)")
        except Exception as e:
            log.warning("[Backbone] torch.compile() no disponible: %s", e)

    return model, actual_d


def load_trainable_backbone(backbone_name="vit_tiny_patch16_224", device="cuda"):
    """
    Construye un backbone en modo entrenamiento (sin congelar parámetros).

    Usado exclusivamente por fase1_train_pipeline.py (Paso 4.1).
    Para extracción de embeddings (Paso 4.2), usar load_frozen_backbone().

    Para CvT-13 y DenseNet llama directamente a las funciones build_*
    para evitar los interceptores de timm que congelan automáticamente.

    Returns:
        model   — backbone en model.train(), requires_grad=True en todos los params
        d_model — dimensión del embedding de salida
    """
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(
            f"Backbone '{backbone_name}' no reconocido.\n"
            f"Opciones válidas: {list(BACKBONE_CONFIGS.keys())}"
        )

    expected_d = BACKBONE_CONFIGS[backbone_name]["d_model"]
    log.info("[Backbone/train] Construyendo entrenable: %s", backbone_name)

    if backbone_name == "cvt_13":
        # Bypass del interceptor timm (que congela). Usar build_cvt13_trainable directamente.
        from backbone_cvt13 import build_cvt13_trainable

        model = build_cvt13_trainable(device=device)

    elif backbone_name == "densenet121_custom":
        # Bypass del interceptor timm (que congela). Usar build_densenet directamente.
        from backbone_densenet import build_densenet

        model = build_densenet(
            in_channels=3, embed_dim=1024, growth_rate=32, block_config=(6, 12, 24, 16)
        )
        model.to(device)

    else:
        # ViT-Tiny, Swin-Tiny: timm.create_model estándar (sin congelar)
        model = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
        )
        model.to(device)

    # Asegurar modo train y requires_grad=True
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    # Verificar que al menos un parámetro es entrenable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "[Backbone/train] Parámetros entrenables: %s / %s",
        f"{trainable:,}",
        f"{total:,}",
    )

    # Forward dummy para verificar d_model
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    model.train()

    actual_d = out.shape[1]
    if actual_d != expected_d:
        log.warning(
            "[Backbone/train] d_model real (%d) difiere del esperado (%d).",
            actual_d,
            expected_d,
        )

    log.info("[Backbone/train] d_model: %d | listo para entrenamiento ✓", actual_d)
    return model, actual_d


def load_frozen_backbone_from_checkpoint(backbone_name, checkpoint_path, device="cuda"):
    """
    Carga un backbone ya entrenado desde checkpoint y lo congela para extracción.

    Usado por fase1_pipeline.py (Paso 4.2) cuando el checkpoint del Paso 4.1 existe.

    El checkpoint tiene la estructura:
        {
            "backbone_name": str,
            "epoch": int,
            "val_acc": float,
            "state_dict": OrderedDict,
        }

    Returns:
        model   — backbone en .eval() con todos los params congelados
        d_model — dimensión del embedding verificada empíricamente
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"[Backbone] Checkpoint no encontrado: {ckpt_path}\n"
            f"Ejecuta primero: python src/pipeline/fase1/fase1_train_pipeline.py "
            f"--backbone {backbone_name}"
        )

    log.info("[Backbone] Cargando checkpoint: %s", ckpt_path)

    # Construir arquitectura desde cero
    model, d_model = load_trainable_backbone(backbone_name, device)

    # Cargar pesos del checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    stored_name = ckpt.get("backbone_name", "desconocido")
    if stored_name != backbone_name:
        log.warning(
            "[Backbone] Checkpoint es de '%s', se solicitó '%s'.",
            stored_name,
            backbone_name,
        )
    model.load_state_dict(ckpt["state_dict"])
    val_acc = ckpt.get("val_acc", 0.0)
    epoch = ckpt.get("epoch", -1)
    log.info("[Backbone] Checkpoint cargado: epoch=%d, val_acc=%.4f", epoch, val_acc)

    # Congelar (mismo comportamiento que load_frozen_backbone)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    log.info("[Backbone] Backbone cargado desde checkpoint y congelado ✓")
    return model, d_model

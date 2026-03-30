"""
backbone_cvt13.py — Soporte Específico de CvT-13 (autocontenido)
=================================================================

Resuelve la incompatibilidad de CvT-13 con timm >= 0.9.x registrando
un interceptor transparente sobre timm.create_model.

Autocontenido: define CvT13Wrapper y build_cvt13 directamente.
scripts/cvt13_backbone.py puede marcarse como obsoleto — este módulo
es la implementación canónica en Fase 1.

Al importar este módulo, el registro se activa automáticamente
(idempotente — seguro para múltiples importaciones).

El modelo se construye siempre desde cero con pesos aleatorios
(requisito del proyecto: sin pesos preentrenados).

Referencia:
    Wu et al., "CvT: Introducing Convolutions to Vision Transformers",
    ICCV 2021. https://arxiv.org/abs/2103.14899
"""

import logging

import timm
import torch
import torch.nn as nn

log = logging.getLogger("fase1")

_CVT13_REGISTERED = False


class CvT13Wrapper(nn.Module):
    """
    Wrapper sobre transformers.CvtModel que expone la misma interfaz
    que un modelo timm con num_classes=0:
      - forward(x) devuelve [B, d_model] directamente.
      - Extrae el CLS token de la última etapa (etapa 3, dim=384).
    """

    D_MODEL = 384  # dimensión de salida del CLS token en CvT-13 etapa 3

    def __init__(self, hf_model):
        super().__init__()
        self.cvt = hf_model
        # Proyección identidad: permite override sin cambiar la interfaz
        self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float32 [B, 3, H, W] normalizado con ImageNet stats.
        Returns:
            z: float32 [B, 384] — CLS token de la etapa 3 de CvT-13.
        """
        outputs = self.cvt(pixel_values=x)
        # CvT-13 solo tiene CLS token en la etapa 3 (última).
        if hasattr(outputs, "cls_token_value") and outputs.cls_token_value is not None:
            cls = outputs.cls_token_value.squeeze(1)  # [B, 384]
        else:
            log.warning(
                "[CvT-13] cls_token_value is None — fallback a last_hidden_state[:, 0]"
            )
            cls = outputs.last_hidden_state[:, 0, :]  # fallback: primer token
        return self.proj(cls)  # [B, 384]


def build_cvt13(
    pretrained: bool = True,
    device: str = "cuda",
    hf_model_name: str = "microsoft/cvt-13",
) -> CvT13Wrapper:
    """
    Construye CvT-13 desde cero para extracción de embeddings (congelado, eval).

    NOTA: El argumento `pretrained` se mantiene por compatibilidad de interfaz
    pero se ignora. El modelo SIEMPRE se inicializa con pesos aleatorios
    (requisito del proyecto: sin pesos preentrenados).

    Args:
        pretrained   : ignorado — se conserva solo por compatibilidad de interfaz.
        device       : "cuda" o "cpu".
        hf_model_name: ignorado — la arquitectura se define localmente.

    Returns:
        CvT13Wrapper en modo eval, todos los parámetros congelados.
    """
    from transformers import CvtConfig, CvtModel

    if pretrained:
        log.warning(
            "[CvT-13] pretrained=True fue solicitado pero se ignora. "
            "El modelo se construye desde cero (requisito del proyecto)."
        )

    log.info("[CvT-13] Construyendo desde cero con pesos aleatorios")
    log.info("[CvT-13] device=%s", device)

    # Arquitectura CvT-13 definida localmente — sin descargar nada
    config = CvtConfig(
        num_channels=3,
        patch_sizes=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        embed_dim=[64, 192, 384],
        num_heads=[1, 3, 6],
        depth=[1, 2, 10],
        mlp_ratio=[4.0, 4.0, 4.0],
        attention_drop_rate=[0.0, 0.0, 0.0],
        drop_rate=[0.0, 0.0, 0.0],
        drop_path_rate=[0.0, 0.0, 0.1],
        qkv_bias=[True, True, True],
        cls_token=[False, False, True],
    )

    # Inicialización con pesos aleatorios — sin checkpoint externo
    hf_model = CvtModel(config)
    log.info("[CvT-13] Modelo construido con pesos aleatorios ✓")

    wrapper = CvT13Wrapper(hf_model)

    for param in wrapper.parameters():
        param.requires_grad = False

    wrapper.eval()
    wrapper.to(device)

    # Verificación con dummy forward
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = wrapper(dummy)

    assert out.shape == (1, CvT13Wrapper.D_MODEL), (
        f"[CvT-13] d_model inesperado: esperado (1, {CvT13Wrapper.D_MODEL}), "
        f"obtenido {tuple(out.shape)}"
    )

    total_params = sum(p.numel() for p in wrapper.parameters())
    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    log.info("[CvT-13] d_model verificado: %d ✓", CvT13Wrapper.D_MODEL)
    log.info("[CvT-13] Parámetros totales  : %s", f"{total_params:,}")
    log.info("[CvT-13] Parámetros entrenab.: %d (debe ser 0 en Fase 1)", trainable)

    if trainable > 0:
        raise RuntimeError(
            f"[CvT-13] ¡{trainable} parámetros con requires_grad=True! "
            "Los embeddings NO serían reproducibles."
        )

    return wrapper


def _register_cvt13_interceptor():
    """Registra el interceptor en timm.create_model (idempotente)."""
    global _CVT13_REGISTERED
    if _CVT13_REGISTERED:
        return

    _original_create = timm.create_model

    def _patched_create_model(model_name, *args, **kwargs):
        if model_name == "cvt_13":
            log.info("[CvT-13/patch] cvt_13 interceptado → CvT13Wrapper (HuggingFace)")
            _device = kwargs.get("device", "cpu")
            _pretrained = kwargs.get("pretrained", True)
            return build_cvt13(pretrained=_pretrained, device=_device)
        return _original_create(model_name, *args, **kwargs)

    timm.create_model = _patched_create_model
    _CVT13_REGISTERED = True
    log.debug("[CvT-13/patch] Interceptor registrado en timm.create_model")


# Activar al importar este módulo
_register_cvt13_interceptor()

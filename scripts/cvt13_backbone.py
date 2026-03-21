"""
cvt13_backbone.py
=================
CvT-13 (Microsoft, ICCV 2021) como backbone extractor compatible con la
interfaz usada en fase0_extract_embeddings.py.

Usa el modelo oficial de HuggingFace (microsoft/cvt-13) vía transformers,
que contiene los pesos del paper original entrenados en ImageNet-1K.

Interfaz pública:
    model = build_cvt13(pretrained=True, device="cuda")
    z = model(imgs)          # imgs: [B, 3, 224, 224] → z: [B, 384]

Compatible con:
    BACKBONE_CONFIGS["cvt_13"] = {"d_model": 384, "vram_gb": 3.0}

Referencia:
    Wu et al., "CvT: Introducing Convolutions to Vision Transformers",
    ICCV 2021. https://arxiv.org/abs/2103.14899
    HuggingFace: https://huggingface.co/microsoft/cvt-13
"""

import logging
import torch
import torch.nn as nn

log = logging.getLogger("cvt13_backbone")


class CvT13Wrapper(nn.Module):
    """
    Wrapper sobre transformers.CvtModel que:
      1. Expone la misma interfaz que un modelo timm con num_classes=0
         (forward devuelve [B, d_model] directamente).
      2. Extrae el CLS token de la última etapa (sequence_output[:, 0, :]).
      3. Proyecta de d_model_interno (384) a d_model_salida (384) — identidad,
         pero explícita para facilitar cambios futuros.

    La salida del CLS token de CvT-13 ya tiene dimensión 384 en la etapa 3,
    lo que coincide con el d_model esperado en BACKBONE_CONFIGS["cvt_13"].
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
            x: tensor float32 [B, 3, H, W] normalizado con ImageNet stats
               (mismo preprocesado que vit_tiny y swin_tiny en fase0)
        Returns:
            z: tensor float32 [B, 384] — CLS token de la etapa 3 de CvT-13
        """
        # CvtModel de HuggingFace devuelve un objeto con:
        #   .last_hidden_state: [B, N_tokens, 384]  (N incluye CLS en etapa 3)
        #   .cls_token_value:   [B, 1, 384]         (disponible directamente)
        outputs = self.cvt(pixel_values=x)

        # Extraer CLS token:
        # En CvT-13 el CLS solo existe en la etapa 3 (última).
        # HuggingFace lo expone en cls_token_value si está disponible,
        # o como primer token de last_hidden_state.
        if hasattr(outputs, "cls_token_value") and outputs.cls_token_value is not None:
            cls = outputs.cls_token_value.squeeze(1)      # [B, 384]
        else:
            # fallback: primer token de la secuencia de la última etapa
            cls = outputs.last_hidden_state[:, 0, :]      # [B, 384]

        return self.proj(cls)                              # [B, 384]


def build_cvt13(pretrained: bool = True,
                device: str = "cuda",
                hf_model_name: str = "microsoft/cvt-13") -> CvT13Wrapper:
    """
    Construye CvT-13 listo para extracción de embeddings (congelado, eval).

    Args:
        pretrained   : si True, descarga pesos oficiales de microsoft/cvt-13
        device       : "cuda" o "cpu"
        hf_model_name: nombre del modelo en HuggingFace Hub

    Returns:
        model: CvT13Wrapper en modo eval, todos los parámetros congelados,
               en el device especificado.

    Uso en fase0_extract_embeddings.py (reemplaza timm.create_model):
        from scripts.cvt13_backbone import build_cvt13
        model, d_model = build_cvt13(pretrained=True, device=device), 384
    """
    from transformers import CvtModel

    log.info(f"[CvT-13] Cargando desde HuggingFace: {hf_model_name}")
    log.info(f"[CvT-13] pretrained={pretrained} | device={device}")

    if pretrained:
        hf_model = CvtModel.from_pretrained(hf_model_name)
        log.info(f"[CvT-13] Pesos oficiales descargados ✓")
    else:
        from transformers import CvtConfig
        config   = CvtConfig.from_pretrained(hf_model_name)
        hf_model = CvtModel(config)
        log.info(f"[CvT-13] Modelo aleatorio (pretrained=False)")

    wrapper = CvT13Wrapper(hf_model)

    # Congelar TODOS los parámetros — FASE 0 es extracción pura
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
    trainable    = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    log.info(f"[CvT-13] d_model verificado: {CvT13Wrapper.D_MODEL} ✓")
    log.info(f"[CvT-13] Parámetros totales  : {total_params:,}")
    log.info(f"[CvT-13] Parámetros entrenab.: {trainable} (debe ser 0 en FASE 0)")

    if trainable > 0:
        raise RuntimeError(
            f"[CvT-13] ¡{trainable} parámetros con requires_grad=True! "
            "Los embeddings NO serían reproducibles."
        )

    return wrapper


# ─── Mini-test de sanidad ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"\n🔬 Test de sanidad CvT-13 en {device}")

    model = build_cvt13(pretrained=True, device=device)

    # Batch de 4 imágenes dummy
    x = __import__("torch").randn(4, 3, 224, 224, device=device)
    with __import__("torch").no_grad():
        z = model(x)

    print(f"  Input  shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(z.shape)}  ← debe ser (4, 384)")
    print(f"  Norma L2 media: {z.norm(dim=1).mean().item():.3f}")
    assert z.shape == (4, 384), f"❌ Shape incorrecto: {z.shape}"
    print("  ✅ CvT-13 OK — listo para fase0_extract_embeddings.py")
    sys.exit(0)

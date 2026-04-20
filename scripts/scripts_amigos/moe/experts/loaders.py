"""
Loaders nativos para los 5 expertos del MoE.

Cada `build_expN(...)` devuelve un `nn.Module` con los pesos entrenados cargados,
listo para `forward()` con la vista nativa del experto.

Fuentes de los pesos:
  - exp1 CXR14: HuggingFace Hub `mitgar14/moe-medical-experts/exp1v21/*`.
  - exp2 ISIC:  local `models/expert3_isic_best.pth.zip` (EfficientNet-B3 torchvision, 8c).
  - exp3 Osteo: local `models/expert2_osteo_best.pth.zip` (EfficientNet-B0 torchvision, 5c KL).
  - exp4 LUNA:  local `models/LUNA-LIDCIDRI_best.pt.zip` (DenseNet-3D custom reproducida).
  - exp5 Pancreas: local `models/exp5_best.pth.zip` (r3d_18 torchvision, 2c binary).

Todos los loaders validan strict_load=True y levantan RuntimeError si hay missing o
unexpected keys. `load_all_experts(...)` devuelve un dict {label: (model, extras)}
donde `extras` incluye thresholds por clase, metricas del checkpoint y notas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b3
from torchvision.models.video import r3d_18

from moe.experts.archs.cxr_v21_wrapper import CXRExpertSingleHead
from moe.experts.archs.densenet3d_luna import DenseNet3DLUNA


DEFAULT_REPO_ID = "mitgar14/moe-medical-experts"
DEFAULT_CKPT_DIR = Path("models")


def _strict_load(model: nn.Module, state_dict: dict) -> None:
    res = model.load_state_dict(state_dict, strict=True)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(
            f"strict load fallo: missing={res.missing_keys[:5]} "
            f"unexpected={res.unexpected_keys[:5]}"
        )


def _remap_convnextv2_keys(sd: dict) -> dict:
    """
    Remapea claves timm legacy (flat `stem_0`, `stages_0`) al naming moderno nested
    (`stem.0`, `stages.0`) que usa timm >=1.0. Checkpoints v20/v21 se entrenaron con
    timm legacy; el modelo construido con timm 1.0.26 usa el naming nested.
    """
    import re

    new_sd = {}
    for k, v in sd.items():
        nk = re.sub(r"backbone\.stem_(\d+)", r"backbone.stem.\1", k)
        nk = re.sub(r"backbone\.stages_(\d+)", r"backbone.stages.\1", nk)
        nk = re.sub(r"backbone\.downsample_(\d+)", r"backbone.downsample.\1", nk)
        new_sd[nk] = v
    return new_sd


def build_exp1_cxr14_v21(
    repo_id: str = DEFAULT_REPO_ID,
    hf_token: str | None = None,
    use_ema: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """exp1 CXR14 v21: ConvNeXt-V2 Base 384 + head(Dropout, Linear(1024, 6)).

    Descarga el checkpoint EMA desde HF Hub. Devuelve (modelo, extras) donde extras
    incluye thresholds por clase y summary del experimento.
    """
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename="exp1v21/ckpt_exp1v21_f0_best.pt",
        token=hf_token,
    )
    thresholds_path = hf_hub_download(
        repo_id=repo_id,
        filename="exp1v21/exp1v21_f0_thresholds.json",
        token=hf_token,
    )
    summary_path = hf_hub_download(
        repo_id=repo_id,
        filename="exp1v21/exp1v21_summary.json",
        token=hf_token,
    )
    oof_meta_path = hf_hub_download(
        repo_id=repo_id,
        filename="exp1v21/oof_val_fold0_meta.json",
        token=hf_token,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_key = "ema_state_dict" if (use_ema and "ema_state_dict" in ckpt) else "model_state_dict"
    sd = ckpt[sd_key]

    model = CXRExpertSingleHead(num_classes=6)
    _strict_load(model, sd)
    model.eval()

    thresholds = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    oof_meta = json.loads(Path(oof_meta_path).read_text(encoding="utf-8"))

    # El summary selecciona el tau optimo por fold; la prevalencia train viene del OOF meta
    tau_optimo = float(summary["fold_results"]["0"].get("tau", 0.5))
    prevalence_train = oof_meta["prevalence_train"]

    extras = {
        "source": f"hf://{repo_id}/exp1v21/",
        "sd_key_used": sd_key,
        "epoch": ckpt.get("epoch"),
        "phase": ckpt.get("phase"),
        "best_val_metric": ckpt.get("best_val_metric"),
        "class_names": summary["classes"],
        "thresholds": thresholds,
        "tau_logit_adjustment": tau_optimo,
        "prevalence_train": prevalence_train,
        "auroc_val": summary["fold_results"]["0"].get("auroc"),
        "f1_opt_val": summary["fold_results"]["0"].get("f1_opt"),
    }
    return model, extras


def build_exp2_isic(
    ckpt_path: str | Path = DEFAULT_CKPT_DIR / "expert3_isic_best.pth.zip",
) -> tuple[nn.Module, dict[str, Any]]:
    """exp2 ISIC 2019: EfficientNet-B3 torchvision + classifier(Dropout, Linear(1536, 8))."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["model_state"]

    model = efficientnet_b3(num_classes=8)
    _strict_load(model, sd)
    model.eval()

    extras = {
        "source": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "f1_macro_val": ckpt.get("f1_macro"),
        "f1_per_class_val": ckpt.get("f1_per_class"),
        "class_names": ckpt.get("labels"),
    }
    return model, extras


def build_exp3_osteo(
    ckpt_path: str | Path = DEFAULT_CKPT_DIR / "expert2_osteo_best.pth.zip",
) -> tuple[nn.Module, dict[str, Any]]:
    """exp3 Osteoarthritis Knee: EfficientNet-B0 torchvision + classifier(Dropout, Linear(1280, 5)).

    Importante: el checkpoint REAL tiene 5 clases KL
    (Normal/Dudoso/Leve/Moderado/Severo), NO 3 como figuraba en el spec previo.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["model_state"]

    model = efficientnet_b0(num_classes=5)
    _strict_load(model, sd)
    model.eval()

    extras = {
        "source": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "f1_macro_val": ckpt.get("f1_macro"),
        "f1_per_class_val": ckpt.get("f1_per_class"),
        "class_names": ckpt.get("labels"),
        "thresholds": ckpt.get("thresholds"),
    }
    return model, extras


def build_exp4_luna(
    ckpt_path: str | Path = DEFAULT_CKPT_DIR / "LUNA-LIDCIDRI_best.pt.zip",
) -> tuple[nn.Module, dict[str, Any]]:
    """exp4 LUNA16 + LIDC-IDRI: DenseNet-3D custom reproducida desde shapes.

    block_config=(4,8,16,12), growth_rate=32, bn_size=4, compression=0.5, init=64.
    Input esperado: (B, 1, 64, 64, 64). Output logits (B, 2).
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    model = DenseNet3DLUNA(
        num_classes=2,
        num_init_features=64,
        growth_rate=32,
        bn_size=4,
        block_config=(4, 8, 16, 12),
        compression=0.5,
        dropout_fc=0.4,
    )
    _strict_load(model, sd)
    model.eval()

    extras = {
        "source": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "val_f1_macro": ckpt.get("val_f1_macro"),
        "val_auc": ckpt.get("val_auc"),
        "config": ckpt.get("config"),
    }
    return model, extras


def build_exp5_pancreas(
    ckpt_path: str | Path = DEFAULT_CKPT_DIR / "exp5_best.pth.zip",
) -> tuple[nn.Module, dict[str, Any]]:
    """exp5 PANORAMA Pancreas: torchvision r3d_18 con fc(2, 512). Binary.

    Input esperado por el checkpoint: (B, 3, 64, 64, 64) — el preprocessor del
    experto replica 1→3 canales antes del forward.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["model_state"]

    model = r3d_18(num_classes=2)
    _strict_load(model, sd)
    model.eval()

    extras = {
        "source": str(ckpt_path),
        "arch": ckpt.get("arch"),
        "epoch": ckpt.get("epoch"),
        "best_f1": ckpt.get("best_f1"),
        "input_shape": ckpt.get("input_shape"),
        "model_input": ckpt.get("model_input"),
        "tipo": ckpt.get("tipo"),
        "dataset": ckpt.get("dataset"),
    }
    return model, extras


def load_all_experts(
    use_hf_for_exp1: bool = True,
    hf_token: str | None = None,
    strict: bool = True,
) -> dict[int, tuple[nn.Module, dict[str, Any]]]:
    """Carga los 5 expertos con pesos reales. Devuelve dict por label.

    Si `strict=False` y un experto falla, registra el error en extras y devuelve
    (None, extras_con_error). Si `strict=True`, re-levanta la excepcion.
    """
    experts: dict[int, tuple[nn.Module | None, dict[str, Any]]] = {}
    builders = [
        (0, "exp1_cxr14_v21", lambda: build_exp1_cxr14_v21(hf_token=hf_token)) if use_hf_for_exp1
        else (0, "exp1_cxr14_v21_local", lambda: (_raise("exp1 requiere HF"),)),
        (1, "exp2_isic", build_exp2_isic),
        (2, "exp3_osteo", build_exp3_osteo),
        (3, "exp4_luna", build_exp4_luna),
        (4, "exp5_pancreas", build_exp5_pancreas),
    ]
    for entry in builders:
        label, name, fn = entry
        try:
            model, extras = fn()
            extras["builder"] = name
            experts[label] = (model, extras)
        except Exception as e:
            if strict:
                raise
            experts[label] = (None, {"builder": name, "error": str(e)})
    return experts


def _raise(msg: str):  # pragma: no cover
    raise RuntimeError(msg)

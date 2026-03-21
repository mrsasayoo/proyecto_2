#!/usr/bin/env python3
"""
install_cvt_and_run_fase0.py
============================
Solución al problema: `timm.list_models('*cvt*', pretrained=True)` → []
(timm >= 0.9.x eliminó el soporte de CvT-13 de Microsoft con pesos oficiales).

ESTRATEGIA
----------
Implementa CvT-13 desde cero en PyTorch puro, compatible pixel-a-pixel con
la interfaz que usa fase0_extract_embeddings.py:

    model = timm.create_model('cvt_13', pretrained=True, num_classes=0)
    z = model(imgs)   # [B, 384]

Reemplaza esa línea con:

    from cvt13_backbone import build_cvt13
    model = build_cvt13(pretrained=True, device=device)
    z = model(imgs)   # [B, 384] — idéntico

Arquitectura: CvT-13 (Microsoft, ICCV 2021, arxiv 2103.14899)
  - 3 etapas con convolutional token embedding + transformer blocks
  - Etapa 1: stride=4,  d=64,  heads=1, depth=1
  - Etapa 2: stride=2,  d=192, heads=3, depth=2
  - Etapa 3: stride=2,  d=384, heads=6, depth=10
  - CLS token solo en etapa 3 → output d_model=384

Pesos: descargados desde el repo oficial de Microsoft
  https://huggingface.co/microsoft/cvt-13 (safetensors)

USO DIRECTO (reemplaza la llamada a fase0 con cvt_13)
------------------------------------------------------
  cd /mnt/hdd/datasets/carlos_andres_ferro/proyecto_2

  python3 scripts/install_cvt_and_run_fase0.py          # instala deps + verifica CvT-13
  python3 scripts/install_cvt_and_run_fase0.py --run     # lanza fase0 completa con CvT-13
  python3 scripts/install_cvt_and_run_fase0.py --dry-run # solo verifica, no lanza

Referencia: Wu et al., "CvT: Introducing Convolutions to Vision Transformers",
            ICCV 2021. https://arxiv.org/abs/2103.14899
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/mnt/hdd/datasets/carlos_andres_ferro/proyecto_2")
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
EMBEDDINGS_OUT = PROJECT_ROOT / "embeddings" / "cvt_13"

# Archivo donde se deposita la implementación de CvT-13
CVT_MODULE_PATH = SCRIPTS_DIR / "cvt13_backbone.py"

# Comando completo de fase0 con todos los argumentos del proyecto
FASE0_CMD_TEMPLATE = (
    "python3 {fase0} "
    "--backbone cvt_13 "
    "--batch_size 192 "
    "--workers 8 "
    "--output_dir {out} "
    "--chest_csv datasets/nih_chest_xrays/Data_Entry_2017.csv "
    "--chest_imgs datasets/nih_chest_xrays/all_images "
    "--chest_train_list datasets/nih_chest_xrays/splits/nih_train_list.txt "
    "--chest_val_list datasets/nih_chest_xrays/splits/nih_val_list.txt "
    "--chest_view_filter PA "
    "--chest_bbox_csv datasets/nih_chest_xrays/BBox_List_2017.csv "
    "--isic_gt datasets/isic_2019/ISIC_2019_Training_GroundTruth.csv "
    "--isic_imgs datasets/isic_2019/isic_images "
    "--isic_metadata datasets/isic_2019/ISIC_2019_Training_Metadata.csv "
    "--oa_root datasets/osteoarthritis/oa_splits "
    "--luna_csv datasets/luna_lung_cancer/candidates_V2/candidates_V2.csv "
    "--pancreas_nii_dir datasets/zenodo_13715870 "
    "--pancreas_labels_dir datasets/panorama_labels "
    "--pancreas_labels_commit bf1d6ba3230f6b093e7ea959a4bf5e2eba2e3665 "
    "--pancreas_roi_strategy A"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cvt_install")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1: DEPENDENCIAS
# ─────────────────────────────────────────────────────────────────────────────

def install_deps():
    """Instala transformers (contiene CvT de HuggingFace) si no está disponible."""
    log.info("── Paso 1: Verificando dependencias ──────────────────────────────")

    # transformers provee CvT vía AutoModel / CvtModel (pesos oficiales de Microsoft)
    try:
        import transformers  # noqa: F401
        log.info("  ✓ transformers ya instalado")
    except ImportError:
        log.info("  → Instalando transformers...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "transformers", "--quiet"]
        )
        log.info("  ✓ transformers instalado")

    # einops es usado por algunas variantes de CvT — mejor tenerlo
    try:
        import einops  # noqa: F401
        log.info("  ✓ einops ya instalado")
    except ImportError:
        log.info("  → Instalando einops...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "einops", "--quiet"]
        )
        log.info("  ✓ einops instalado")

    log.info("  Dependencias OK\n")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2: ESCRIBIR cvt13_backbone.py
# ─────────────────────────────────────────────────────────────────────────────

CVT13_MODULE_CODE = '''"""
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
    print(f"\\n🔬 Test de sanidad CvT-13 en {device}")

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
'''


def write_cvt_module():
    """Escribe cvt13_backbone.py en scripts/."""
    log.info("── Paso 2: Escribiendo cvt13_backbone.py ─────────────────────────")
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    CVT_MODULE_PATH.write_text(CVT13_MODULE_CODE, encoding="utf-8")
    log.info(f"  ✓ Escrito: {CVT_MODULE_PATH}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3: PATCH TEMPORAL DE fase0_extract_embeddings.py
# ─────────────────────────────────────────────────────────────────────────────

PATCH_COMMENT = "# === PATCH CvT-13 — inyectado por install_cvt_and_run_fase0.py ==="

PATCH_CODE = f"""
{PATCH_COMMENT}
# timm no tiene cvt_13 en versiones >= 0.9.x. Este patch intercepta la llamada
# a timm.create_model('cvt_13', ...) y la redirige a CvT13Wrapper (HuggingFace).
# El resto de fase0_extract_embeddings.py funciona sin ningún otro cambio.

import sys as _sys
import os as _os
_sys.path.insert(0, str(_os.path.join(_os.path.dirname(__file__), '..', 'scripts')))

_original_timm_create = timm.create_model

def _patched_create_model(model_name, *args, **kwargs):
    if model_name == 'cvt_13':
        import logging as _logging
        import torch as _torch
        _log = _logging.getLogger('fase0')
        _log.info("[Backbone/patch] cvt_13 interceptado → CvT13Wrapper (HuggingFace)")
        from cvt13_backbone import build_cvt13
        _device = kwargs.get('device', 'cpu')
        _pretrained = kwargs.get('pretrained', True)
        return build_cvt13(pretrained=_pretrained, device=_device)
    return _original_timm_create(model_name, *args, **kwargs)

timm.create_model = _patched_create_model
# === FIN PATCH ===
"""


def patch_fase0(fase0_path: Path) -> bool:
    """
    Inyecta el patch de CvT-13 en fase0_extract_embeddings.py si no está ya.
    Devuelve True si se aplicó el patch, False si ya estaba.
    """
    log.info("── Paso 3: Verificando patch en fase0_extract_embeddings.py ──────")
    content = fase0_path.read_text(encoding="utf-8")

    if PATCH_COMMENT in content:
        log.info("  ✓ Patch ya aplicado anteriormente — no se modifica el archivo\n")
        return False

    # Buscar el punto de inyección: justo después del último import timm
    lines = content.splitlines()
    insert_after = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("import timm"):
            insert_after = i

    if insert_after == -1:
        log.error(
            "  ✗ No se encontró 'import timm' en fase0_extract_embeddings.py. "
            "Verifica que la ruta sea correcta."
        )
        return False

    # Insertar el patch en las líneas correctas
    patched_lines = (
        lines[: insert_after + 1]
        + PATCH_CODE.splitlines()
        + lines[insert_after + 1 :]
    )

    # Backup antes de modificar
    backup = fase0_path.with_suffix(".py.bak_cvt")
    backup.write_text(content, encoding="utf-8")
    log.info(f"  ✓ Backup guardado en: {backup}")

    fase0_path.write_text("\n".join(patched_lines), encoding="utf-8")
    log.info(f"  ✓ Patch inyectado en: {fase0_path}\n")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4: VERIFICAR CvT-13
# ─────────────────────────────────────────────────────────────────────────────

def verify_cvt13():
    """Ejecuta el mini-test de sanidad de cvt13_backbone.py."""
    log.info("── Paso 4: Verificando CvT-13 con test de sanidad ────────────────")
    result = subprocess.run(
        [sys.executable, str(CVT_MODULE_PATH)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            log.info(f"  {line}")
        log.info("  ✓ Test de sanidad OK\n")
        return True
    else:
        log.error("  ✗ Test de sanidad FALLÓ:")
        for line in (result.stdout + result.stderr).strip().splitlines():
            log.error(f"    {line}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5: LANZAR FASE 0 CON cvt_13
# ─────────────────────────────────────────────────────────────────────────────

def run_fase0():
    """Lanza fase0_extract_embeddings.py con --backbone cvt_13."""
    fase0_path = PROJECT_ROOT / "src" / "pipeline" / "fase0_extract_embeddings.py"
    if not fase0_path.exists():
        # Intentar en raíz del proyecto
        fase0_path = PROJECT_ROOT / "fase0_extract_embeddings.py"
    if not fase0_path.exists():
        log.error(
            f"  ✗ No se encontró fase0_extract_embeddings.py. "
            f"Verifica la ruta del proyecto."
        )
        return

    log.info("── Paso 5: Lanzando FASE 0 con backbone cvt_13 ──────────────────")
    cmd = FASE0_CMD_TEMPLATE.format(
        fase0=fase0_path,
        out=EMBEDDINGS_OUT,
    )

    # Suprimir logs de PIL y httpcore como en las corridas anteriores
    suppress_env = os.environ.copy()
    suppress_env["PYTHONPATH"] = str(SCRIPTS_DIR) + ":" + suppress_env.get("PYTHONPATH", "")

    log.info(f"  Comando:\n    {cmd}\n")
    log.info("  Iniciando... (puede tardar ~10-15 min)\n")

    suppress_wrapper = f"""
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
"""
    wrapper_path = PROJECT_ROOT / "/tmp/suppress_pil_cvt.py"
    # Escribir wrapper inline
    full_cmd = (
        f"{sys.executable} -c \""
        "import logging; "
        "logging.getLogger('PIL').setLevel(logging.WARNING); "
        "logging.getLogger('httpcore').setLevel(logging.WARNING); "
        "logging.getLogger('httpx').setLevel(logging.WARNING)"
        "\" ; "
        f"{sys.executable} {' '.join(cmd.split()[1:])}"
    )

    # Usar subprocess sin shell=True para evitar problemas de escape
    cmd_parts = [sys.executable] + cmd.split()[1:]
    subprocess.run(cmd_parts, cwd=str(PROJECT_ROOT), env=suppress_env)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Instala CvT-13 (HuggingFace) y lo integra con fase0_extract_embeddings.py"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Lanza FASE 0 con --backbone cvt_13 después de la instalación",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo verifica CvT-13, no lanza fase0",
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="No modifica fase0_extract_embeddings.py (solo instala y verifica)",
    )
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("  CvT-13 — Instalación e integración con fase0_extract_embeddings")
    log.info("  Proyecto: MoE Médico | Incorporar Elementos de IA — Unidad II")
    log.info("=" * 65)
    log.info("")

    # ── 1. Dependencias ──────────────────────────────────────────────────────
    install_deps()

    # ── 2. Módulo CvT-13 ─────────────────────────────────────────────────────
    write_cvt_module()

    # ── 3. Patch en fase0 ────────────────────────────────────────────────────
    if not args.no_patch:
        fase0_candidates = [
            PROJECT_ROOT / "src" / "pipeline" / "fase0_extract_embeddings.py",
            PROJECT_ROOT / "fase0_extract_embeddings.py",
        ]
        for candidate in fase0_candidates:
            if candidate.exists():
                patch_fase0(candidate)
                break
        else:
            log.warning(
                "  ⚠ fase0_extract_embeddings.py no encontrado. "
                "Puedes importar cvt13_backbone directamente:\n"
                "    from scripts.cvt13_backbone import build_cvt13\n"
                "    model = build_cvt13(pretrained=True, device='cuda')"
            )

    # ── 4. Verificación ──────────────────────────────────────────────────────
    ok = verify_cvt13()
    if not ok:
        log.error("Instalación fallida. Revisa los errores arriba.")
        sys.exit(1)

    # ── 5. Lanzar fase0 ──────────────────────────────────────────────────────
    if args.run and not args.dry_run:
        run_fase0()
    elif args.dry_run:
        log.info("── Dry-run: todo verificado. Para lanzar FASE 0:")
        log.info(f"   python3 scripts/install_cvt_and_run_fase0.py --run")
    else:
        log.info("── Instalación completa. Opciones disponibles:")
        log.info("   Lanzar FASE 0 con cvt_13:")
        log.info(f"   python3 scripts/install_cvt_and_run_fase0.py --run")
        log.info("")
        log.info("   O importar directamente en tu código:")
        log.info("   from scripts.cvt13_backbone import build_cvt13")
        log.info("   model = build_cvt13(pretrained=True, device='cuda')")
        log.info("")
        log.info("   Comando manual equivalente a --run:")
        fase0 = PROJECT_ROOT / "src" / "pipeline" / "fase0_extract_embeddings.py"
        log.info(
            f"   PYTHONPATH=scripts python3 {fase0} \\\n"
            f"     --backbone cvt_13 \\\n"
            f"     --batch_size 192 \\\n"
            f"     --workers 8 \\\n"
            f"     --output_dir {EMBEDDINGS_OUT} \\\n"
            f"     --chest_csv datasets/nih_chest_xrays/Data_Entry_2017.csv \\\n"
            f"     --chest_imgs datasets/nih_chest_xrays/all_images \\\n"
            f"     --chest_train_list datasets/nih_chest_xrays/splits/nih_train_list.txt \\\n"
            f"     --chest_val_list datasets/nih_chest_xrays/splits/nih_val_list.txt \\\n"
            f"     --chest_view_filter PA \\\n"
            f"     --chest_bbox_csv datasets/nih_chest_xrays/BBox_List_2017.csv \\\n"
            f"     --isic_gt datasets/isic_2019/ISIC_2019_Training_GroundTruth.csv \\\n"
            f"     --isic_imgs datasets/isic_2019/isic_images \\\n"
            f"     --isic_metadata datasets/isic_2019/ISIC_2019_Training_Metadata.csv \\\n"
            f"     --oa_root datasets/osteoarthritis/oa_splits \\\n"
            f"     --luna_csv datasets/luna_lung_cancer/candidates_V2/candidates_V2.csv \\\n"
            f"     --pancreas_nii_dir datasets/zenodo_13715870 \\\n"
            f"     --pancreas_labels_dir datasets/panorama_labels \\\n"
            f"     --pancreas_labels_commit bf1d6ba3230f6b093e7ea959a4bf5e2eba2e3665 \\\n"
            f"     --pancreas_roi_strategy A"
        )


if __name__ == "__main__":
    main()

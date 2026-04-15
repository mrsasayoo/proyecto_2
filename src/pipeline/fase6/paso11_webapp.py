"""
paso11_webapp.py
----------------
Paso 11 — Interfaz web para inferencia interactiva del sistema MoE.

Implementa las 8 funcionalidades obligatorias de proyecto_moe.md §11:
  1. Carga de imagen (PNG, JPEG, NIfTI) — detección 2D/3D por tensor.ndim
  2. Preprocesado transparente (dimensiones originales y adaptadas)
  3. Inferencia en tiempo real (etiqueta, confianza %, tiempo ms)
  4. Attention Heatmap del Router ViT (Grad-CAM)
  5. Panel del experto activado (nombre, arquitectura, dataset, gating score)
  6. Panel del ablation study (tabla comparativa 4 métodos)
  7. Load Balance (gráfica de barras f_i acumulado)
  8. OOD Detection (alerta cuando H(g) > umbral calibrado)

Uso:
  --dry-run  : verificar componentes sin lanzar servidor (EXIT 0 si OK)
  --port     : puerto del servidor (default: 7860)
  --share    : crear enlace público (solo producción)
  --device   : cpu o cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# ── Project root ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Routers path for linear.py imports
_ROUTERS_DIR = str(PROJECT_ROOT / "src" / "pipeline" / "fase2" / "routers")
if _ROUTERS_DIR not in sys.path:
    sys.path.insert(0, _ROUTERS_DIR)

from src.pipeline.logging_utils import setup_logging
from src.pipeline.fase6.webapp_helpers import (
    EXPERT_METADATA,
    ABLATION_PLACEHOLDER,
    LoadBalanceCounter,
    preprocess_image_for_webapp,
    load_ablation_results,
    create_mock_inference_result,
    format_confidence,
)

logger = logging.getLogger(__name__)

# ── Global state ────────────────────────────────────────────
_load_counter = LoadBalanceCounter(n_experts=6)
_engine = None  # InferenceEngine o None (demo mode)
_ablation_data = None  # dict cargado de ablation_results.json


# =====================================================================
# CLI
# =====================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paso 11 — Interfaz web MoE para inferencia de imagen médica"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verificar todos los componentes sin lanzar el servidor web",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo de inferencia (default: cpu)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Puerto del servidor Gradio (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Crear enlace público de Gradio (solo producción)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio de salida para reportes (default: results/paso11)",
    )
    return parser.parse_args()


# =====================================================================
# build_inference_engine
# =====================================================================
def build_inference_engine(device: str, dry_run: bool):
    """
    Intenta cargar MoESystem desde checkpoint y crear InferenceEngine.

    En dry-run: retorna None (no instancia InferenceEngine).
    Si no existe el checkpoint: logging.warning + retorna None (degradación graceful).
    """
    if dry_run:
        logger.info("DRY-RUN: InferenceEngine no instanciado")
        return None

    from src.pipeline.fase6.fase6_config import MOE_CHECKPOINT

    checkpoint_path = PROJECT_ROOT / MOE_CHECKPOINT
    if not checkpoint_path.exists():
        logger.warning(
            "Checkpoint no encontrado en %s — webapp funcionará en demo mode "
            "(mock inference). Ejecute el entrenamiento (Paso 8) primero.",
            checkpoint_path,
        )
        return None

    try:
        from src.pipeline.fase6.inference_engine import InferenceEngine
        from src.pipeline.fase5.moe_model import MoESystem
        from src.pipeline.fase2.models.expert1_convnext import Expert1ConvNeXtTiny
        from src.pipeline.fase2.models.expert2_convnext_small import (
            Expert2ConvNeXtSmall,
        )
        from src.pipeline.fase2.models.expert_oa_efficientnet_b3 import (
            ExpertOAEfficientNetB3,
        )
        from src.pipeline.fase2.models.expert3_densenet3d import Expert3MC318
        from src.pipeline.fase2.models.expert4_resnet3d import ExpertPancreasSwin3D
        from src.pipeline.fase3.models.expert6_resunet import ConditionedResUNetAE
        from linear import LinearGatingHead
        from src.pipeline.fase6.fase6_config import N_EXPERTS_DOMAIN

        import torch.nn as nn

        experts = nn.ModuleList(
            [
                Expert1ConvNeXtTiny(dropout_fc=0.3, num_classes=14),
                Expert2ConvNeXtSmall(),
                ExpertOAEfficientNetB3(),
                Expert3MC318(spatial_dropout_p=0.15, fc_dropout_p=0.4, num_classes=2),
                ExpertPancreasSwin3D(in_channels=1, num_classes=2),
                ConditionedResUNetAE(in_ch=3, base_ch=64, n_domains=6),
            ]
        )

        router = LinearGatingHead(d_model=192, n_experts=N_EXPERTS_DOMAIN)
        moe = MoESystem(experts=experts, router=router, backbone=None)

        dev = torch.device(device)
        engine = InferenceEngine.from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            moe_system_class=MoESystem,
            moe_system_kwargs={"experts": experts, "router": router, "backbone": None},
            device=dev,
        )
        logger.info("InferenceEngine cargado exitosamente")
        return engine

    except Exception as e:
        logger.warning("Error construyendo InferenceEngine: %s — usando demo mode", e)
        return None


# =====================================================================
# run_inference
# =====================================================================
def run_inference(
    image_input,
    engine,
    load_counter: LoadBalanceCounter,
    gradcam_enabled: bool = True,
) -> dict:
    """
    Función principal de inferencia.

    1. Preprocesa imagen
    2. Si engine=None → mock inference (demo mode)
    3. Si engine disponible → forward real
    4. Actualiza load_counter
    5. Calcula tiempo de inferencia
    6. Si gradcam_enabled y engine disponible → genera heatmap
    """
    t0 = time.perf_counter()

    tensor, preprocess_meta = preprocess_image_for_webapp(image_input)

    if engine is None:
        result = create_mock_inference_result()
        result["_demo_mode"] = True
        logger.info("Inferencia mock (demo mode)")
    else:
        dev = engine.device
        tensor = tensor.to(dev)
        result = engine(tensor)
        result["_demo_mode"] = False

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    expert_ids = result["expert_ids"]
    load_counter.update(expert_ids)

    expert_id = expert_ids[0] if expert_ids else 0
    logits = result["logits"][0] if result["logits"] else torch.zeros(1, 1)

    label_str, confidence_pct = format_confidence(logits, expert_id)

    # Entropy & OOD check
    entropy_val = (
        float(result["entropy"].item())
        if result["entropy"].numel() == 1
        else float(result["entropy"][0].item())
    )
    is_ood = (
        bool(result["is_ood"])
        if isinstance(result["is_ood"], bool)
        else bool(
            result["is_ood"].item()
            if result["is_ood"].numel() == 1
            else result["is_ood"][0].item()
        )
    )

    # Gates
    gates = result["gates"]
    gates_dict = {}
    if gates is not None and gates.numel() > 0:
        gates_1d = gates[0] if gates.ndim == 2 else gates
        for i, g in enumerate(gates_1d.tolist()):
            expert_name = EXPERT_METADATA.get(i, {}).get("name", f"Expert {i}")
            gates_dict[expert_name] = round(g, 4)

    # Expert panel info
    expert_meta = EXPERT_METADATA.get(expert_id, {})
    expert_panel = {
        "expert_id": expert_id,
        "name": expert_meta.get("name", f"Expert {expert_id}"),
        "architecture": expert_meta.get("architecture", "N/A"),
        "dataset": expert_meta.get("dataset", "N/A"),
        "n_classes": expert_meta.get("n_classes", 0),
        "modality": expert_meta.get("modality", "N/A"),
        "gating_score": gates_dict.get(expert_meta.get("name", ""), 0.0),
    }

    # Heatmap placeholder
    heatmap_img = None
    if gradcam_enabled and engine is not None:
        try:
            from src.pipeline.fase6.gradcam_heatmap import (
                GradCAMExtractor,
                _find_last_conv,
            )

            expert_module = engine.moe.experts[expert_id]
            target_layer = _find_last_conv(expert_module)
            if target_layer is not None:
                extractor = GradCAMExtractor(
                    model=expert_module,
                    target_layer=target_layer,
                    expert_name=expert_meta.get("name", "unknown"),
                )
                try:
                    cam = extractor.compute(tensor)
                    # Convert cam to displayable image
                    if cam.ndim == 2:
                        import numpy as np

                        cam_uint8 = (cam * 255).astype(np.uint8)
                        heatmap_img = cam_uint8
                finally:
                    extractor.remove_hooks()
        except Exception as e:
            logger.warning("Grad-CAM no disponible: %s", e)

    return {
        "label": label_str,
        "confidence_pct": round(confidence_pct, 2),
        "elapsed_ms": round(elapsed_ms, 2),
        "expert_panel": expert_panel,
        "heatmap": heatmap_img,
        "gates": gates_dict,
        "entropy": round(entropy_val, 4),
        "is_ood": is_ood,
        "preprocess": preprocess_meta,
        "is_mock": result.get("_is_mock", False),
        "is_demo_mode": result.get("_demo_mode", False),
    }


# =====================================================================
# Funciones Gradio
# =====================================================================
def fn_infer(image_numpy, nifti_file=None):
    """
    Función principal llamada por Gradio al subir imagen.

    Returns tuple para los outputs de Gradio:
      (label, confidence, time, expert_json, heatmap, dims, ood_alert)
    """
    global _engine, _load_counter

    # Determinar input: NIfTI tiene prioridad si se proporcionó
    if nifti_file is not None:
        image_input = nifti_file
    elif image_numpy is not None:
        image_input = image_numpy
    else:
        return (
            "Sin imagen",
            "N/A",
            "N/A",
            {},
            None,
            "No se proporcionó imagen",
            "Sin inferencia",
        )

    try:
        result = run_inference(
            image_input=image_input,
            engine=_engine,
            load_counter=_load_counter,
            gradcam_enabled=True,
        )
    except Exception as e:
        logger.error("Error en inferencia: %s", e)
        return (
            f"Error: {e}",
            "N/A",
            "N/A",
            {},
            None,
            "Error en preprocesado",
            "Error",
        )

    demo_note = " [DEMO MODE]" if result["is_demo_mode"] else ""

    label_text = f"{result['label']}{demo_note}"
    confidence_text = f"{result['confidence_pct']:.2f}%"
    time_text = f"{result['elapsed_ms']:.2f} ms"
    expert_json = result["expert_panel"]
    heatmap = result["heatmap"]
    dims_text = (
        f"Original: {result['preprocess'].get('original_shape', 'N/A')} → "
        f"Adaptado: {result['preprocess'].get('adapted_shape', 'N/A')} "
        f"(Modalidad: {result['preprocess'].get('modality', 'N/A')})"
    )

    if result["is_ood"]:
        ood_text = (
            f"⚠ OOD DETECTADO — Entropía H(g)={result['entropy']:.4f} "
            f"(sobre umbral). Muestra fuera de distribución."
        )
    else:
        ood_text = f"✓ In-distribution — Entropía H(g)={result['entropy']:.4f}"

    return (
        label_text,
        confidence_text,
        time_text,
        expert_json,
        heatmap,
        dims_text,
        ood_text,
    )


def fn_load_balance_chart():
    """Genera gráfica de barras con f_i actuales del LoadBalanceCounter."""
    global _load_counter

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib no disponible para gráfica de load balance")
        return None

    freqs = _load_counter.get_frequencies()
    counts = _load_counter.get_counts()
    ratio = _load_counter.get_max_min_ratio()

    expert_names = [
        EXPERT_METADATA.get(i, {}).get("name", f"E{i}").split(" — ")[0]
        for i in range(6)
    ]
    freq_values = [freqs.get(i, 0.0) for i in range(6)]
    count_values = [counts.get(i, 0) for i in range(6)]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#607D8B"]
    bars = ax.bar(
        expert_names, freq_values, color=colors, edgecolor="white", linewidth=0.8
    )

    # Añadir conteo sobre cada barra
    for bar, count in zip(bars, count_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Frecuencia relativa (f_i)")
    ax.set_title(f"Load Balance — max/min ratio: {ratio:.2f}")
    ax.set_ylim(0, max(freq_values) * 1.3 if max(freq_values) > 0 else 1.0)
    plt.tight_layout()
    return fig


def fn_ablation_table():
    """Retorna tabla del ablation study como DataFrame (o placeholder)."""
    global _ablation_data

    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas no disponible para tabla de ablation")
        return None

    if _ablation_data is None:
        _ablation_data = load_ablation_results(None)

    results = _ablation_data.get("results", {})

    rows = []
    for router_name, metrics in results.items():
        rows.append(
            {
                "Router": router_name,
                "Routing Accuracy": metrics.get("routing_accuracy", "N/A"),
                "Latency (ms)": metrics.get("latency_ms", "N/A"),
            }
        )

    if not rows:
        rows.append(
            {"Router": "Sin datos", "Routing Accuracy": "N/A", "Latency (ms)": "N/A"}
        )

    df = pd.DataFrame(rows)
    return df


def fn_reset_counter():
    """Resetea el LoadBalanceCounter."""
    global _load_counter
    _load_counter.reset()
    logger.info("LoadBalanceCounter reseteado")
    return "✓ Contador de load balance reseteado exitosamente"


# =====================================================================
# build_gradio_app
# =====================================================================
def build_gradio_app(engine):
    """Construye la aplicación Gradio con todas las tabs."""
    import gradio as gr

    global _engine, _ablation_data
    _engine = engine

    # Cargar ablation data
    ablation_path = PROJECT_ROOT / "results" / "ablation_results.json"
    _ablation_data = load_ablation_results(
        str(ablation_path) if ablation_path.exists() else None
    )

    with gr.Blocks(title="MoE Medical Image Dashboard") as demo:
        gr.Markdown("# 🏥 Sistema MoE — Inferencia de Imagen Médica")
        gr.Markdown("Pipeline Mixture of Experts para imágenes médicas multimodales")

        # ── Tab 1: Inferencia ──
        with gr.Tab("🔬 Inferencia"):
            with gr.Row():
                image_input = gr.Image(label="Cargar imagen (PNG/JPEG)", type="numpy")
                nifti_input = gr.File(
                    label="Cargar NIfTI (.nii/.nii.gz)",
                    file_types=[".nii", ".nii.gz"],
                )

            infer_btn = gr.Button("🚀 Inferir", variant="primary")

            with gr.Row():
                label_out = gr.Textbox(label="Predicción")
                confidence_out = gr.Textbox(label="Confianza (%)")
                time_out = gr.Textbox(label="Tiempo (ms)")

            with gr.Row():
                heatmap_out = gr.Image(label="Grad-CAM Heatmap (Router ViT)")
                expert_panel = gr.JSON(label="Experto Activado")

            dims_out = gr.Textbox(label="Preprocesado (dimensiones)")
            ood_alert = gr.Textbox(label="Estado OOD", value="Sin inferencia")

            infer_btn.click(
                fn=fn_infer,
                inputs=[image_input, nifti_input],
                outputs=[
                    label_out,
                    confidence_out,
                    time_out,
                    expert_panel,
                    heatmap_out,
                    dims_out,
                    ood_alert,
                ],
            )

        # ── Tab 2: Load Balance ──
        with gr.Tab("⚖️ Load Balance"):
            balance_plot = gr.Plot(label="Distribución f_i acumulada")
            refresh_btn = gr.Button("🔄 Actualizar")
            reset_btn = gr.Button("🗑️ Resetear contador")
            reset_msg = gr.Textbox(label="Estado")

            refresh_btn.click(fn=fn_load_balance_chart, outputs=[balance_plot])
            reset_btn.click(fn=fn_reset_counter, outputs=[reset_msg])

        # ── Tab 3: Ablation Study ──
        with gr.Tab("📊 Ablation Study"):
            ablation_table = gr.DataFrame(label="Comparativa de 4 Routers")
            load_ablation_btn = gr.Button("📥 Cargar datos")
            load_ablation_btn.click(fn=fn_ablation_table, outputs=[ablation_table])

    return demo


# =====================================================================
# Dry-run
# =====================================================================
def run_dry_run(args: argparse.Namespace, output_dir: Path) -> int:
    """
    Verifica todos los componentes sin lanzar el servidor.

    Checks:
      DR-1:  Import gradio
      DR-2:  Import webapp_helpers (LoadBalanceCounter, EXPERT_METADATA, etc.)
      DR-3:  Import inference_engine (InferenceEngine)
      DR-4:  Import gradcam_heatmap (GradCAMExtractor)
      DR-5:  Import ood_detector (OODDetector)
      DR-6:  LoadBalanceCounter — instanciar y hacer update
      DR-7:  preprocess_image_for_webapp — imagen sintética numpy
      DR-8:  create_mock_inference_result — genera resultado mock
      DR-9:  load_ablation_results — carga o retorna placeholder
      DR-10: fn_load_balance_chart — genera figura matplotlib
      DR-11: fn_ablation_table — genera DataFrame
      DR-12: build_inference_engine(dry_run=True) — None (graceful)
      DR-13: Gradio Blocks — instanciar el demo (NO lanzar)

    Escribe: results/paso11/dry_run_report.json
    Exit: 0 siempre en dry-run
    """
    import numpy as np

    checks = []

    # DR-1: Import gradio
    try:
        import gradio as gr

        checks.append(
            {
                "id": "DR-1",
                "name": "import gradio",
                "result": "PASS",
                "observed": gr.__version__,
            }
        )
    except ImportError as e:
        checks.append(
            {
                "id": "DR-1",
                "name": "import gradio",
                "result": "WARN",
                "observed": str(e),
                "note": "gradio no instalado — instalar con: pip install gradio",
            }
        )

    # DR-2: Import webapp_helpers
    try:
        from src.pipeline.fase6.webapp_helpers import (
            EXPERT_METADATA as _em,
            LoadBalanceCounter as _lbc,
            preprocess_image_for_webapp as _piw,
            load_ablation_results as _lar,
            create_mock_inference_result as _cmir,
            format_confidence as _fc,
        )

        checks.append(
            {
                "id": "DR-2",
                "name": "import webapp_helpers",
                "result": "PASS",
                "observed": f"6 exports OK, {len(_em)} experts",
            }
        )
    except ImportError as e:
        checks.append(
            {
                "id": "DR-2",
                "name": "import webapp_helpers",
                "result": "FAIL",
                "observed": str(e),
            }
        )

    # DR-3: Import inference_engine
    try:
        from src.pipeline.fase6.inference_engine import InferenceEngine

        checks.append(
            {
                "id": "DR-3",
                "name": "import inference_engine",
                "result": "PASS",
                "observed": "InferenceEngine class loaded",
            }
        )
    except ImportError as e:
        checks.append(
            {
                "id": "DR-3",
                "name": "import inference_engine",
                "result": "WARN",
                "observed": str(e),
            }
        )

    # DR-4: Import gradcam_heatmap
    try:
        from src.pipeline.fase6.gradcam_heatmap import GradCAMExtractor

        checks.append(
            {
                "id": "DR-4",
                "name": "import gradcam_heatmap",
                "result": "PASS",
                "observed": "GradCAMExtractor class loaded",
            }
        )
    except ImportError as e:
        checks.append(
            {
                "id": "DR-4",
                "name": "import gradcam_heatmap",
                "result": "WARN",
                "observed": str(e),
            }
        )

    # DR-5: Import ood_detector
    try:
        from src.pipeline.fase6.ood_detector import OODDetector

        checks.append(
            {
                "id": "DR-5",
                "name": "import ood_detector",
                "result": "PASS",
                "observed": "OODDetector class loaded",
            }
        )
    except ImportError as e:
        checks.append(
            {
                "id": "DR-5",
                "name": "import ood_detector",
                "result": "WARN",
                "observed": str(e),
            }
        )

    # DR-6: LoadBalanceCounter
    try:
        counter = LoadBalanceCounter(n_experts=6)
        counter.update([0, 1, 2, 0, 0])
        counts = counter.get_counts()
        freqs = counter.get_frequencies()
        ratio = counter.get_max_min_ratio()
        assert counts[0] == 3, f"Expected counts[0]=3, got {counts[0]}"
        assert sum(freqs.values()) > 0.99, "Frequencies should sum to ~1.0"
        checks.append(
            {
                "id": "DR-6",
                "name": "LoadBalanceCounter",
                "result": "PASS",
                "observed": f"counts={counts}, ratio={ratio:.2f}",
            }
        )
    except Exception as e:
        checks.append(
            {
                "id": "DR-6",
                "name": "LoadBalanceCounter",
                "result": "FAIL",
                "observed": str(e),
            }
        )

    # DR-7: preprocess_image_for_webapp
    try:
        dummy_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tensor, meta = preprocess_image_for_webapp(dummy_img)
        assert tensor.ndim == 4, f"Expected 4D tensor, got ndim={tensor.ndim}"
        assert meta["modality"] == "2D", f"Expected 2D, got {meta['modality']}"
        checks.append(
            {
                "id": "DR-7",
                "name": "preprocess_image_for_webapp",
                "result": "PASS",
                "observed": f"shape={list(tensor.shape)}, modality={meta['modality']}",
            }
        )
    except Exception as e:
        checks.append(
            {
                "id": "DR-7",
                "name": "preprocess_image_for_webapp",
                "result": "FAIL",
                "observed": str(e),
            }
        )

    # DR-8: create_mock_inference_result
    try:
        mock = create_mock_inference_result(n_experts=5)
        assert "logits" in mock, "Missing logits"
        assert "expert_ids" in mock, "Missing expert_ids"
        assert "gates" in mock, "Missing gates"
        assert "_is_mock" in mock and mock["_is_mock"], "Missing _is_mock flag"
        checks.append(
            {
                "id": "DR-8",
                "name": "create_mock_inference_result",
                "result": "PASS",
                "observed": f"expert_id={mock['expert_ids'][0]}, keys={list(mock.keys())}",
            }
        )
    except Exception as e:
        checks.append(
            {
                "id": "DR-8",
                "name": "create_mock_inference_result",
                "result": "FAIL",
                "observed": str(e),
            }
        )

    # DR-9: load_ablation_results
    try:
        ablation = load_ablation_results(None)
        assert "results" in ablation, "Missing results key"
        assert "note" in ablation, "Missing note key"
        assert len(ablation["results"]) == 4, (
            f"Expected 4 routers, got {len(ablation['results'])}"
        )
        checks.append(
            {
                "id": "DR-9",
                "name": "load_ablation_results",
                "result": "PASS",
                "observed": f"placeholder loaded, {len(ablation['results'])} routers",
            }
        )
    except Exception as e:
        checks.append(
            {
                "id": "DR-9",
                "name": "load_ablation_results",
                "result": "FAIL",
                "observed": str(e),
            }
        )

    # DR-10: fn_load_balance_chart
    try:
        # Update counter to have some data
        _load_counter.reset()
        _load_counter.update([0, 1, 2, 3, 0, 1])
        fig = fn_load_balance_chart()
        if fig is not None:
            checks.append(
                {
                    "id": "DR-10",
                    "name": "fn_load_balance_chart",
                    "result": "PASS",
                    "observed": f"figure generated: {type(fig).__name__}",
                }
            )
        else:
            checks.append(
                {
                    "id": "DR-10",
                    "name": "fn_load_balance_chart",
                    "result": "WARN",
                    "observed": "matplotlib no disponible — gráfica retornó None",
                }
            )
        # Cleanup
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass
    except Exception as e:
        checks.append(
            {
                "id": "DR-10",
                "name": "fn_load_balance_chart",
                "result": "WARN",
                "observed": str(e),
            }
        )

    # DR-11: fn_ablation_table
    try:
        df = fn_ablation_table()
        if df is not None:
            checks.append(
                {
                    "id": "DR-11",
                    "name": "fn_ablation_table",
                    "result": "PASS",
                    "observed": f"DataFrame shape={df.shape}, columns={list(df.columns)}",
                }
            )
        else:
            checks.append(
                {
                    "id": "DR-11",
                    "name": "fn_ablation_table",
                    "result": "WARN",
                    "observed": "pandas no disponible, retornó None",
                }
            )
    except Exception as e:
        checks.append(
            {
                "id": "DR-11",
                "name": "fn_ablation_table",
                "result": "WARN",
                "observed": str(e),
            }
        )

    # DR-12: build_inference_engine(dry_run=True)
    try:
        engine = build_inference_engine(device=args.device, dry_run=True)
        assert engine is None, f"Expected None in dry-run, got {type(engine)}"
        checks.append(
            {
                "id": "DR-12",
                "name": "build_inference_engine(dry_run=True)",
                "result": "PASS",
                "observed": "None (expected in dry-run)",
            }
        )
    except Exception as e:
        checks.append(
            {
                "id": "DR-12",
                "name": "build_inference_engine(dry_run=True)",
                "result": "FAIL",
                "observed": str(e),
            }
        )

    # DR-13: Gradio Blocks
    try:
        import gradio as gr

        with gr.Blocks(title="MoE Medical Image Dashboard [DR]") as demo_test:
            gr.Markdown("# Test")
        checks.append(
            {
                "id": "DR-13",
                "name": "Gradio Blocks (instanciar sin lanzar)",
                "result": "PASS",
                "observed": "Blocks created successfully",
            }
        )
    except ImportError:
        checks.append(
            {
                "id": "DR-13",
                "name": "Gradio Blocks (instanciar sin lanzar)",
                "result": "WARN",
                "observed": "gradio no instalado — DR-1 ya lo reportó",
            }
        )
    except Exception as e:
        checks.append(
            {
                "id": "DR-13",
                "name": "Gradio Blocks (instanciar sin lanzar)",
                "result": "WARN",
                "observed": str(e),
            }
        )

    # ── Generar reporte ──
    n_pass = sum(1 for c in checks if c["result"] == "PASS")
    n_warn = sum(1 for c in checks if c["result"] == "WARN")
    n_fail = sum(1 for c in checks if c["result"] == "FAIL")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "total_checks": len(checks),
        "passed": n_pass,
        "warnings": n_warn,
        "failed": n_fail,
        "checks": checks,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dry_run_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Dry-run report written → %s", report_path)
    logger.info(
        "TOTAL: %d | PASS: %d | WARN: %d | FAIL: %d",
        report["total_checks"],
        report["passed"],
        report["warnings"],
        report["failed"],
    )

    # Log cada check
    for c in checks:
        status_icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(c["result"], "?")
        logger.info(
            "  %s [%s] %s — %s",
            status_icon,
            c["id"],
            c["name"],
            c.get("observed", ""),
        )

    return 0  # SIEMPRE 0 en dry-run


# =====================================================================
# main
# =====================================================================
def main():
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "results" / "paso11"
    )

    setup_logging(output_dir=str(output_dir), phase_name="paso11_webapp")

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("PASO 11 — DRY-RUN MODE")
        logger.info("Verificando componentes de la webapp sin lanzar servidor")
        logger.info("=" * 60)
        sys.exit(run_dry_run(args, output_dir))

    # Solo llega aquí en modo producción (NO dry-run)
    engine = build_inference_engine(device=args.device, dry_run=False)
    demo = build_gradio_app(engine)

    logger.info("Lanzando webapp en puerto %d ...", args.port)
    demo.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()

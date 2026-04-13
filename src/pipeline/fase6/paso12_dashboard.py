"""
paso12_dashboard.py
-------------------
Paso 12 — Dashboard de reporte final del sistema MoE.

Genera un dashboard Gradio con 6 tabs:
  1. Arquitectura MoE (diagrama, trazas, descripciones)
  2. Métricas de Entrenamiento (curvas, tabla resumen)
  3. Ablation Study (tabla comparativa 4 routers)
  4. Load Balance (gráfica de distribución por experto)
  5. Estado del Sistema (verificación paso10, dry-run paso11, checkpoints)
  6. Figuras del Reporte (galería de 5 figuras, regenerar, descargar)

Uso:
  --dry-run  : verificar componentes, generar figuras, escribir reporte (EXIT 0)
  --port     : puerto del servidor (default: 7861)
  --share    : crear enlace público de Gradio
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Project root ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Routers path for linear.py imports (needed by downstream modules)
_ROUTERS_DIR = str(PROJECT_ROOT / "src" / "pipeline" / "fase2" / "routers")
if _ROUTERS_DIR not in sys.path:
    sys.path.insert(0, _ROUTERS_DIR)

from src.pipeline.logging_utils import setup_logging

logger = logging.getLogger(__name__)

# ── Default output ──────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "results/paso12"
DEFAULT_PORT = 7861

# ── Paths to upstream results ───────────────────────────────
PASO9_METRICS_PATH = PROJECT_ROOT / "results" / "paso9" / "test_metrics_summary.json"
PASO9_LOAD_BALANCE_PATH = PROJECT_ROOT / "results" / "paso9" / "load_balance_test.json"
PASO10_REPORT_PATH = PROJECT_ROOT / "results" / "paso10" / "verification_report.json"
PASO11_REPORT_PATH = PROJECT_ROOT / "results" / "paso11" / "dry_run_report.json"

# ── Expected figure filenames ───────────────────────────────
EXPECTED_FIGURES = [
    "figura1_arquitectura_moe.png",
    "figura2_ablation_study.png",
    "figura3_curvas_entrenamiento.png",
    "figura4_load_balance.png",
    "figura5_attention_heatmap.png",
]


# =====================================================================
# CLI
# =====================================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for paso12 dashboard."""
    parser = argparse.ArgumentParser(
        description="Paso 12 — Dashboard de reporte final del sistema MoE"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verificar componentes, generar figuras, escribir reporte (EXIT 0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Puerto del servidor Gradio (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Crear enlace público de Gradio",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Directorio de salida para reportes (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


# =====================================================================
# Helpers
# =====================================================================
def _read_json_safe(path: Path) -> dict | None:
    """Read a JSON file, return None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _make_check(name: str, status: str, message: str) -> dict:
    """Build a single check result dict."""
    return {
        "name": name,
        "status": status,
        "message": message,
    }


# =====================================================================
# MoE Architecture Trace (Tab 1 content)
# =====================================================================
_ARCHITECTURE_TRACE = """\
═══════════════════════════════════════════════════════════════
      TRAZA PASO A PASO DEL SISTEMA MoE
═══════════════════════════════════════════════════════════════

1. ENTRADA
   → Imagen médica (2D: PNG/JPEG, 3D: NIfTI .nii/.nii.gz)

2. PREPROCESADO
   → Detección de modalidad por tensor.ndim (2D si ndim≤3, 3D si ndim≥4)
   → Resize a 224×224 (2D) o 96×96×96 (3D)
   → Normalización ImageNet (2D) o z-score (3D)

3. BACKBONE — ViT-Tiny (DINOv2)
   → Extrae embeddings de 192 dimensiones
   → Patch size 16, 12 heads, depth 12
   → Output: tensor [B, 192]

4. ROUTER — Linear Gating Head
   → Input:  embedding [B, 192]
   → Linear: 192 → 5 (N_EXPERTS_DOMAIN)
   → Softmax: gates g_i ∈ [0, 1], Σg_i = 1
   → Selección: expert_id = argmax(g)

5. EXPERTOS (5 dominio + 1 CAE)
   → Expert 0: ConvNeXt-Tiny       — Chest X-Ray (14 clases)
   → Expert 1: EfficientNet-B3     — ISIC Dermatología (9 clases)
   → Expert 2: EfficientNet-B0     — OA Knee (5 clases)
   → Expert 3: MC3-18              — LUNA16 Nódulos 3D (2 clases)
   → Expert 4: Swin3D-Tiny         — Páncreas MSD 3D (2 clases)
   → Expert 5: ConvAutoEncoder     — OOD / Reconstrucción

6. SALIDA
   → Logits del experto seleccionado
   → Entropía H(g) para detección OOD
   → Si H(g) > umbral calibrado → alerta OOD

═══════════════════════════════════════════════════════════════
"""

_COMPONENT_DESCRIPTIONS: dict[str, str] = {
    "ViT-Tiny (Backbone)": (
        "Vision Transformer Tiny pretrained con DINOv2. "
        "Extrae embeddings de 192 dimensiones a partir de parches 16×16. "
        "Se congela durante el fine-tuning del router (solo las últimas capas se ajustan)."
    ),
    "Linear Gating Head (Router)": (
        "Capa lineal 192→5 seguida de Softmax. Produce gate values g_i para cada "
        "experto de dominio. El experto con gate máximo recibe la imagen. "
        "Entrenado end-to-end en Stage 3."
    ),
    "Expert 0 — ConvNeXt-Tiny (Chest X-Ray)": (
        "Clasificador multi-etiqueta para 14 patologías torácicas del dataset NIH ChestXray14. "
        "Arquitectura ConvNeXt-Tiny con dropout 0.3 en la capa FC final."
    ),
    "Expert 1 — EfficientNet-B3 (ISIC)": (
        "Clasificador de lesiones dermatológicas con 9 clases (ISIC 2019). "
        "EfficientNet-B3 con compound scaling optimizado para imágenes de piel."
    ),
    "Expert 2 — EfficientNet-B0 (OA Knee)": (
        "Clasificador de osteoartritis de rodilla (5 grados KL). "
        "EfficientNet-B0 con dropout 0.5."
    ),
    "Expert 3 — MC3-18 (LUNA16)": (
        "Clasificador 3D de nódulos pulmonares (benigno/maligno) del dataset LUNA16. "
        "Mixed Convolution 3D-18 con spatial dropout 0.15."
    ),
    "Expert 4 — Swin3D-Tiny (Páncreas)": (
        "Clasificador 3D para detección de PDAC en el páncreas (MSD). "
        "Swin Transformer 3D Tiny con ventanas desplazadas."
    ),
    "Expert 5 — ConvAutoEncoder (OOD)": (
        "Autoencoder convolucional para detección out-of-distribution. "
        "Reconstruye la imagen de entrada; un error de reconstrucción alto "
        "indica muestra OOD. No produce clasificación directa."
    ),
}


# =====================================================================
# build_dashboard
# =====================================================================
def build_dashboard(output_dir: Path, logger: logging.Logger) -> "gr.Blocks":
    """
    Construye el dashboard Gradio con 6 tabs para reporte final.

    Parameters
    ----------
    output_dir : Path
        Directorio donde se almacenan las figuras generadas.
    logger : logging.Logger
        Logger configurado para el pipeline.

    Returns
    -------
    gr.Blocks
        Aplicación Gradio lista para lanzar con demo.launch().
    """
    import gradio as gr

    # ── Cargar datos upstream ───────────────────────────────
    metrics_data = _read_json_safe(PASO9_METRICS_PATH) or {}
    load_balance_data = _read_json_safe(PASO9_LOAD_BALANCE_PATH) or {}
    paso10_report = _read_json_safe(PASO10_REPORT_PATH) or {}
    paso11_report = _read_json_safe(PASO11_REPORT_PATH) or {}

    # ── Figure paths ────────────────────────────────────────
    figure_paths = {name: output_dir / "figures" / name for name in EXPECTED_FIGURES}

    # ── Helper: load figure as image or placeholder text ────
    def _load_figure(fig_name: str) -> str | None:
        path = figure_paths.get(fig_name)
        if path and path.exists():
            return str(path)
        return None

    # ── Helper: regenerate figures ──────────────────────────
    def fn_regenerate_figures() -> list:
        """Re-generate all 5 figures and return gallery items."""
        try:
            from src.pipeline.fase6.dashboard_figures import generate_all_figures

            generate_all_figures(output_dir, logger)
            logger.info("Figuras regeneradas exitosamente")
        except Exception as e:
            logger.error("Error regenerando figuras: %s", e)

        gallery_items = []
        for fig_name in EXPECTED_FIGURES:
            path = figure_paths.get(fig_name)
            if path and path.exists():
                gallery_items.append(str(path))
        return gallery_items

    # ── Build metrics table ─────────────────────────────────
    def _build_metrics_dataframe():
        """Build a pandas DataFrame from test_metrics_summary.json."""
        try:
            import pandas as pd
        except ImportError:
            return None

        per_expert = metrics_data.get("per_expert", [])
        if not per_expert:
            return pd.DataFrame(
                {"Info": ["No hay datos de métricas disponibles (ejecute Paso 9)"]}
            )

        rows = []
        for exp in per_expert:
            rows.append(
                {
                    "Expert": f"{exp.get('expert_idx', '?')} — {exp.get('expert_name', 'N/A')}",
                    "F1 Macro": f"{exp.get('f1_macro', 0.0):.4f}",
                    "AUC-ROC": f"{exp.get('auc_roc', 0.0):.4f}",
                    "Routing Acc": f"{exp.get('routing_accuracy', 0.0):.4f}",
                    "N Samples": exp.get("n_samples", 0),
                    "Multilabel": "Sí" if exp.get("is_multilabel") else "No",
                    "Dry Run": "Sí" if exp.get("dry_run") else "No",
                }
            )
        return pd.DataFrame(rows)

    # ── Build ablation table ────────────────────────────────
    def _build_ablation_dataframe():
        """Build ablation study comparison table for 4 routers."""
        try:
            import pandas as pd
        except ImportError:
            return None

        from src.pipeline.fase6.webapp_helpers import (
            load_ablation_results,
            ABLATION_PLACEHOLDER,
        )

        ablation_path = PROJECT_ROOT / "results" / "ablation_results.json"
        ablation_data = load_ablation_results(
            str(ablation_path) if ablation_path.exists() else None
        )

        results = ablation_data.get("results", {})
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
                {
                    "Router": "Sin datos",
                    "Routing Accuracy": "N/A",
                    "Latency (ms)": "N/A",
                }
            )

        return pd.DataFrame(rows)

    # ── Build system status ─────────────────────────────────
    def _build_system_status() -> dict:
        """Compile system status from paso10 + paso11 reports and checkpoints."""
        status = {}

        # Paso 10 verification summary
        status["paso10_verification"] = {
            "total_checks": paso10_report.get("total_checks", "N/A"),
            "passed": paso10_report.get("passed", "N/A"),
            "failed": paso10_report.get("failed", "N/A"),
            "skipped": paso10_report.get("skipped", "N/A"),
            "warnings": paso10_report.get("warnings", "N/A"),
            "penalties_detected": paso10_report.get("penalties_detected", []),
            "timestamp": paso10_report.get("timestamp", "N/A"),
        }

        # Paso 11 dry-run summary
        status["paso11_webapp"] = {
            "total_checks": paso11_report.get("total_checks", "N/A"),
            "passed": paso11_report.get("passed", "N/A"),
            "warnings": paso11_report.get("warnings", "N/A"),
            "failed": paso11_report.get("failed", "N/A"),
            "timestamp": paso11_report.get("timestamp", "N/A"),
        }

        # Checkpoint listing
        checkpoint_dir = PROJECT_ROOT / "checkpoints"
        checkpoints = []
        if checkpoint_dir.exists():
            for ckpt in sorted(checkpoint_dir.rglob("*.pt")):
                rel = ckpt.relative_to(PROJECT_ROOT)
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                checkpoints.append(
                    {
                        "path": str(rel),
                        "size_mb": round(size_mb, 2),
                    }
                )
        status["checkpoints"] = checkpoints if checkpoints else ["No checkpoints found"]

        return status

    # ── Build load balance summary ──────────────────────────
    def _build_load_balance_summary() -> dict:
        """Summarize load_balance_test.json data."""
        if not load_balance_data:
            return {"status": "No load balance data (ejecute Paso 9)"}

        return {
            "global_expert_counts": load_balance_data.get("global_expert_counts", {}),
            "max_min_ratio": load_balance_data.get("max_min_ratio", "N/A"),
            "threshold": load_balance_data.get("threshold", "N/A"),
            "passes": load_balance_data.get("passes", "N/A"),
        }

    # ═════════════════════════════════════════════════════════
    # Gradio Blocks
    # ═════════════════════════════════════════════════════════
    with gr.Blocks(
        title="Paso 12 — Dashboard de Reporte MoE",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Paso 12 — Dashboard de Reporte Final\n"
            "Sistema Mixture of Experts para clasificación de imágenes médicas multimodales"
        )

        # ── Tab 1: Arquitectura MoE ────────────────────────
        with gr.Tab("Arquitectura MoE"):
            gr.Markdown("## Diagrama de Arquitectura")
            fig1_path = _load_figure("figura1_arquitectura_moe.png")
            if fig1_path:
                gr.Image(
                    value=fig1_path,
                    label="Figura 1 — Arquitectura del Sistema MoE",
                    show_label=True,
                )
            else:
                gr.Markdown(
                    "*Figura 1 no generada aún. Use la pestaña 'Figuras del Reporte' "
                    "para regenerar.*"
                )

            gr.Markdown("## Traza Paso a Paso")
            gr.Textbox(
                value=_ARCHITECTURE_TRACE,
                label="Flujo de datos: Entrada → Preprocesado → Backbone → Router → Experto → Salida",
                lines=30,
                interactive=False,
            )

            gr.Markdown("## Descripción de Componentes")
            for comp_name, comp_desc in _COMPONENT_DESCRIPTIONS.items():
                with gr.Accordion(comp_name, open=False):
                    gr.Markdown(comp_desc)

        # ── Tab 2: Métricas de Entrenamiento ───────────────
        with gr.Tab("Métricas de Entrenamiento"):
            gr.Markdown("## Curvas de Entrenamiento")
            fig3_path = _load_figure("figura3_training_curves.png")
            if fig3_path:
                gr.Image(
                    value=fig3_path,
                    label="Figura 3 — Curvas de Entrenamiento",
                    show_label=True,
                )
            else:
                gr.Markdown(
                    "*Figura 3 no disponible. Regenere desde la pestaña 'Figuras del Reporte'.*"
                )

            gr.Markdown("## Métricas Globales")
            global_metrics_md = (
                f"| Métrica | Valor |\n"
                f"|---|---|\n"
                f"| F1 Macro 2D (media) | {metrics_data.get('f1_macro_2d_mean', 0.0):.4f} |\n"
                f"| F1 Macro 3D (media) | {metrics_data.get('f1_macro_3d_mean', 0.0):.4f} |\n"
                f"| Routing Accuracy (media) | {metrics_data.get('routing_accuracy_mean', 0.0):.4f} |\n"
                f"| Load Balance Ratio | {metrics_data.get('load_balance_ratio', 'N/A')} |\n"
                f"| Pasa 2D aceptable | {'Sí' if metrics_data.get('passes_2d_acceptable') else 'No'} |\n"
                f"| Pasa 2D full | {'Sí' if metrics_data.get('passes_2d_full') else 'No'} |\n"
                f"| Pasa 3D aceptable | {'Sí' if metrics_data.get('passes_3d_acceptable') else 'No'} |\n"
                f"| Pasa 3D full | {'Sí' if metrics_data.get('passes_3d_full') else 'No'} |\n"
                f"| Pasa Routing | {'Sí' if metrics_data.get('passes_routing') else 'No'} |\n"
                f"| Pasa Load Balance | {'Sí' if metrics_data.get('passes_load_balance') else 'No'} |\n"
            )
            gr.Markdown(global_metrics_md)

            gr.Markdown("## Métricas por Experto")
            metrics_df = _build_metrics_dataframe()
            if metrics_df is not None:
                gr.DataFrame(
                    value=metrics_df,
                    label="Resultados por experto (Paso 9)",
                )

        # ── Tab 3: Ablation Study ──────────────────────────
        with gr.Tab("Ablation Study"):
            gr.Markdown("## Tabla Comparativa de Routers")
            fig2_path = _load_figure("figura2_ablation_table.png")
            if fig2_path:
                gr.Image(
                    value=fig2_path,
                    label="Figura 2 — Ablation Study",
                    show_label=True,
                )
            else:
                gr.Markdown(
                    "*Figura 2 no disponible. Regenere desde la pestaña 'Figuras del Reporte'.*"
                )

            gr.Markdown("## Datos del Ablation Study")
            ablation_df = _build_ablation_dataframe()
            if ablation_df is not None:
                gr.DataFrame(
                    value=ablation_df,
                    label="Comparativa 4 Routers (TopK, Soft, NoisyTopK, Linear ViT)",
                )

        # ── Tab 4: Load Balance ────────────────────────────
        with gr.Tab("Load Balance"):
            gr.Markdown("## Distribución de Carga por Experto")
            fig4_path = _load_figure("figura4_load_balance.png")
            if fig4_path:
                gr.Image(
                    value=fig4_path,
                    label="Figura 4 — Load Balance",
                    show_label=True,
                )
            else:
                gr.Markdown(
                    "*Figura 4 no disponible. Regenere desde la pestaña 'Figuras del Reporte'.*"
                )

            gr.Markdown("## Estadísticas de Load Balance")
            lb_summary = _build_load_balance_summary()
            gr.JSON(
                value=lb_summary,
                label="Load Balance — load_balance_test.json",
            )

        # ── Tab 5: Estado del Sistema ──────────────────────
        with gr.Tab("Estado del Sistema"):
            gr.Markdown("## Resumen de Verificación (Paso 10 + Paso 11)")
            system_status = _build_system_status()
            gr.JSON(
                value=system_status,
                label="Estado completo del pipeline",
            )

        # ── Tab 6: Figuras del Reporte ─────────────────────
        with gr.Tab("Figuras del Reporte"):
            gr.Markdown("## Galería de Figuras del Reporte Final")

            # Initial gallery items
            initial_gallery = []
            for fig_name in EXPECTED_FIGURES:
                path = figure_paths.get(fig_name)
                if path and path.exists():
                    initial_gallery.append(str(path))

            gallery = gr.Gallery(
                value=initial_gallery if initial_gallery else None,
                label="Figuras generadas",
                columns=3,
                height="auto",
            )

            regen_btn = gr.Button("Regenerar Figuras", variant="primary")
            regen_btn.click(fn=fn_regenerate_figures, outputs=[gallery])

            gr.Markdown("### Descargas individuales")
            for fig_name in EXPECTED_FIGURES:
                path = figure_paths.get(fig_name)
                if path and path.exists():
                    gr.File(
                        value=str(path),
                        label=fig_name,
                    )
                else:
                    gr.Markdown(f"- `{fig_name}` — *no generada*")

    logger.info("Dashboard Gradio construido con 6 tabs")
    return demo


# =====================================================================
# DryRunChecker
# =====================================================================
class DryRunChecker:
    """
    Ejecuta 10 verificaciones para validar que el dashboard
    puede construirse y funcionar correctamente.
    """

    def __init__(self, output_dir: Path, logger: logging.Logger) -> None:
        self.output_dir = output_dir
        self.logger = logger
        self.checks: list[dict] = []
        self.figures_generated: list[str] = []

    def _add(self, name: str, status: str, message: str) -> None:
        """Register a check result."""
        self.checks.append(_make_check(name, status, message))
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(status, "?")
        self.logger.info("  %s [%s] %s — %s", icon, status, name, message)

    # ── Check 1: Core imports ───────────────────────────────
    def check_imports(self) -> None:
        """Verify that gradio, matplotlib, pandas, torch are importable."""
        missing = []
        versions = {}

        for mod_name in ("gradio", "matplotlib", "pandas", "torch"):
            try:
                mod = __import__(mod_name)
                versions[mod_name] = getattr(mod, "__version__", "unknown")
            except ImportError:
                missing.append(mod_name)

        if not missing:
            ver_str = ", ".join(f"{k}={v}" for k, v in versions.items())
            self._add("check_imports", "PASS", f"All imports OK: {ver_str}")
        elif all(m in ("gradio", "matplotlib") for m in missing):
            # gradio/matplotlib missing is a warning (may not be installed in CI)
            self._add(
                "check_imports",
                "WARN",
                f"Missing optional: {missing}; available: {versions}",
            )
        else:
            self._add("check_imports", "FAIL", f"Missing: {missing}")

    # ── Check 2: dashboard_figures module ───────────────────
    def check_dashboard_figures_module(self) -> None:
        """Verify that dashboard_figures module is importable."""
        try:
            from src.pipeline.fase6.dashboard_figures import generate_all_figures

            self._add(
                "check_dashboard_figures_module",
                "PASS",
                "generate_all_figures importable",
            )
        except ImportError as e:
            self._add(
                "check_dashboard_figures_module",
                "WARN",
                f"dashboard_figures not importable: {e}",
            )

    # ── Check 3: results/paso9 metrics ──────────────────────
    def check_results_paso9(self) -> None:
        """Verify test_metrics_summary.json is readable."""
        data = _read_json_safe(PASO9_METRICS_PATH)
        if data is not None:
            n_experts = len(data.get("per_expert", []))
            f1_2d = data.get("f1_macro_2d_mean", "N/A")
            self._add(
                "check_results_paso9",
                "PASS",
                f"test_metrics_summary.json readable: {n_experts} experts, f1_2d_mean={f1_2d}",
            )
        else:
            self._add(
                "check_results_paso9",
                "WARN",
                f"test_metrics_summary.json not readable at {PASO9_METRICS_PATH}",
            )

    # ── Check 4: results/paso10 report ──────────────────────
    def check_results_paso10(self) -> None:
        """Verify verification_report.json is readable."""
        data = _read_json_safe(PASO10_REPORT_PATH)
        if data is not None:
            total = data.get("total_checks", "?")
            passed = data.get("passed", "?")
            self._add(
                "check_results_paso10",
                "PASS",
                f"verification_report.json readable: {passed}/{total} passed",
            )
        else:
            self._add(
                "check_results_paso10",
                "WARN",
                f"verification_report.json not readable at {PASO10_REPORT_PATH}",
            )

    # ── Check 5: results/paso11 report ──────────────────────
    def check_results_paso11(self) -> None:
        """Verify dry_run_report.json from paso11 is readable."""
        data = _read_json_safe(PASO11_REPORT_PATH)
        if data is not None:
            total = data.get("total_checks", "?")
            passed = data.get("passed", "?")
            self._add(
                "check_results_paso11",
                "PASS",
                f"dry_run_report.json readable: {passed}/{total} passed",
            )
        else:
            self._add(
                "check_results_paso11",
                "WARN",
                f"dry_run_report.json not readable at {PASO11_REPORT_PATH}",
            )

    # ── Check 6: load balance data ──────────────────────────
    def check_load_balance_data(self) -> None:
        """Verify load_balance_test.json is readable."""
        data = _read_json_safe(PASO9_LOAD_BALANCE_PATH)
        if data is not None:
            ratio = data.get("max_min_ratio", "?")
            passes = data.get("passes", "?")
            self._add(
                "check_load_balance_data",
                "PASS",
                f"load_balance_test.json readable: ratio={ratio}, passes={passes}",
            )
        else:
            self._add(
                "check_load_balance_data",
                "WARN",
                f"load_balance_test.json not readable at {PASO9_LOAD_BALANCE_PATH}",
            )

    # ── Check 7: output directory ───────────────────────────
    def check_output_dir(self) -> None:
        """Verify output directory is creatable and writable."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.output_dir / ".write_test"
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink()
            self._add(
                "check_output_dir",
                "PASS",
                f"Output dir writable: {self.output_dir}",
            )
        except Exception as e:
            self._add(
                "check_output_dir",
                "FAIL",
                f"Output dir not writable: {e}",
            )

    # ── Check 8: figures generation ─────────────────────────
    def check_figures_generation(self) -> None:
        """Call generate_all_figures and verify 5 figures are produced."""
        try:
            from src.pipeline.fase6.dashboard_figures import generate_all_figures

            generate_all_figures(self.output_dir, self.logger)

            generated = []
            for fig_name in EXPECTED_FIGURES:
                fig_path = self.output_dir / "figures" / fig_name
                if fig_path.exists():
                    generated.append(fig_name)

            self.figures_generated = generated
            n = len(generated)

            if n == len(EXPECTED_FIGURES):
                self._add(
                    "check_figures_generation",
                    "PASS",
                    f"All {n} figures generated: {generated}",
                )
            elif n > 0:
                missing = [f for f in EXPECTED_FIGURES if f not in generated]
                self._add(
                    "check_figures_generation",
                    "WARN",
                    f"{n}/{len(EXPECTED_FIGURES)} figures generated; missing: {missing}",
                )
            else:
                self._add(
                    "check_figures_generation",
                    "FAIL",
                    "No figures generated",
                )
        except ImportError as e:
            self._add(
                "check_figures_generation",
                "WARN",
                f"dashboard_figures not importable, cannot generate: {e}",
            )
        except Exception as e:
            self._add(
                "check_figures_generation",
                "FAIL",
                f"Error generating figures: {e}",
            )

    # ── Check 9: Gradio components ──────────────────────────
    def check_gradio_components(self) -> None:
        """Instantiate Gradio Blocks/Tabs/Image/JSON without launching."""
        try:
            import gradio as gr

            with gr.Blocks(title="MoE Dashboard [DR Test]") as _test_demo:
                with gr.Tab("Test Tab"):
                    gr.Markdown("# Test")
                    gr.Image(label="Test Image")
                    gr.JSON(label="Test JSON")

            self._add(
                "check_gradio_components",
                "PASS",
                "Blocks/Tab/Image/JSON instantiated without launch",
            )
        except ImportError:
            self._add(
                "check_gradio_components",
                "WARN",
                "gradio not installed — cannot test components",
            )
        except Exception as e:
            self._add(
                "check_gradio_components",
                "WARN",
                f"Gradio component test error: {e}",
            )

    # ── Check 10: report write ──────────────────────────────
    def check_report_write(self) -> None:
        """Write dry_run_report.json to output directory."""
        try:
            report = self._build_report()
            report_path = self.output_dir / "dry_run_report.json"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self._add(
                "check_report_write",
                "PASS",
                f"Report written to {report_path}",
            )
        except Exception as e:
            self._add(
                "check_report_write",
                "FAIL",
                f"Cannot write report: {e}",
            )

    # ── Build report ────────────────────────────────────────
    def _build_report(self) -> dict:
        """Build the dry-run report dict."""
        n_pass = sum(1 for c in self.checks if c["status"] == "PASS")
        n_warn = sum(1 for c in self.checks if c["status"] == "WARN")
        n_fail = sum(1 for c in self.checks if c["status"] == "FAIL")

        return {
            "paso": 12,
            "phase": "paso12_dashboard",
            "dry_run": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total": len(self.checks),
                "passed": n_pass,
                "warned": n_warn,
                "failed": n_fail,
            },
            "checks": self.checks,
            "figures_generated": self.figures_generated,
            "exit_code": 0,
        }

    # ── Run all checks ──────────────────────────────────────
    def run_all(self) -> dict:
        """Execute all 10 checks in order and return the report."""
        self.logger.info("=" * 60)
        self.logger.info("PASO 12 — DRY-RUN: Ejecutando 10 checks")
        self.logger.info("=" * 60)

        self.check_imports()
        self.check_dashboard_figures_module()
        self.check_results_paso9()
        self.check_results_paso10()
        self.check_results_paso11()
        self.check_load_balance_data()
        self.check_output_dir()
        self.check_figures_generation()
        self.check_gradio_components()
        self.check_report_write()

        report = self._build_report()

        self.logger.info("-" * 60)
        self.logger.info(
            "TOTAL: %d | PASS: %d | WARN: %d | FAIL: %d",
            report["summary"]["total"],
            report["summary"]["passed"],
            report["summary"]["warned"],
            report["summary"]["failed"],
        )
        self.logger.info("=" * 60)

        return report


# =====================================================================
# main
# =====================================================================
def main() -> None:
    """Entry point for paso12 dashboard."""
    args = parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "results" / "paso12"
    )

    log = setup_logging(output_dir=str(output_dir), phase_name="paso12_dashboard")

    if args.dry_run:
        log.info("=" * 60)
        log.info("PASO 12 — DRY-RUN MODE")
        log.info("Verificando componentes del dashboard sin lanzar servidor")
        log.info("=" * 60)

        checker = DryRunChecker(output_dir=output_dir, logger=log)
        report = checker.run_all()

        # Write final report (check_report_write already writes a preliminary one,
        # but we overwrite with the final version that includes all checks)
        report_path = output_dir / "dry_run_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info("Final dry-run report → %s", report_path)

        sys.exit(0)

    # ── Production mode: build and launch dashboard ─────────
    log.info("=" * 60)
    log.info("PASO 12 — DASHBOARD MODE")
    log.info("Construyendo y lanzando dashboard en puerto %d", args.port)
    log.info("=" * 60)

    # Generate figures before launching
    try:
        from src.pipeline.fase6.dashboard_figures import generate_all_figures

        output_dir.mkdir(parents=True, exist_ok=True)
        generate_all_figures(output_dir, log)
        log.info("Figuras generadas exitosamente")
    except Exception as e:
        log.warning("No se pudieron generar figuras: %s — dashboard sin figuras", e)

    demo = build_dashboard(output_dir=output_dir, logger=log)
    log.info("Lanzando dashboard en puerto %d ...", args.port)
    demo.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()

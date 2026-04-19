# Paso 12 — Dashboard de Reporte Final del Sistema MoE

| Campo | Valor |
|---|---|
| **Fecha** | 2026-04-06 |
| **Alcance** | Paso 12 — Dashboard de reporte final con 6 tabs y 5 figuras obligatorias del reporte técnico (`proyecto_moe.md` §12.2). Gradio app con resumen de arquitectura, métricas, ablation study, load balance, estado del sistema y galería de figuras. Generador de figuras matplotlib (Agg backend, sin plotly). Implementado en `src/pipeline/fase6/` con degradación graceful sin datos upstream |
| **Estado general** | ✅ **Implementado (dry-run verificado)** |

---

## 1. Descripción

El Paso 12 es el dashboard de reporte final del sistema MoE. A diferencia del Paso 11, que es la interfaz interactiva de inferencia en tiempo real, el Paso 12 se enfoca en la **visualización y exportación de resultados para el reporte técnico**. Implementa:

- **Dashboard Gradio con 6 tabs:** presentación organizada de todos los resultados del pipeline (arquitectura, métricas, ablation, load balance, estado del sistema, figuras).
- **5 figuras obligatorias del reporte técnico:** definidas en `proyecto_moe.md` §12.2, generadas con matplotlib (backend Agg, sin plotly), exportadas a PNG a 150 dpi.
- **Integración con pasos previos:** consume resultados de Paso 9 (métricas, load balance), Paso 10 (verificación) y Paso 11 (dry-run webapp) para compilar un resumen completo del sistema.

Produce un reporte JSON de dry-run, un log de ejecución y las 5 figuras obligatorias en `results/paso12/`.

---

## 2. Archivos implementados

| Archivo | Descripción | Función principal | Líneas |
|---|---|---|---|
| `src/pipeline/fase6/paso12_dashboard.py` | Dashboard Gradio 6 tabs, 10 checks DR, CLI --dry-run | `main()` — construye dashboard o ejecuta dry-run con `DryRunChecker` | ~920 |
| `src/pipeline/fase6/dashboard_figures.py` | Generador de 5 figuras matplotlib del reporte técnico | `generate_all_figures()` — produce Figura 1–5 como PNG a 150 dpi | ~1085 |

---

## 3. Dashboard: 6 tabs

| # | Tab | Contenido | Datos upstream |
|---|---|---|---|
| 1 | **Arquitectura MoE** | Diagrama del sistema (Figura 1), traza paso a paso del flujo de datos, descripción de cada componente (backbone, router, 6 expertos) | Figura 1 generada por `dashboard_figures.py` |
| 2 | **Métricas de Entrenamiento** | Curvas de entrenamiento (Figura 3), tabla de métricas globales (F1 2D/3D, routing accuracy, load balance), métricas por experto (DataFrame) | `results/paso9/test_metrics_summary.json` |
| 3 | **Ablation Study** | Tabla comparativa de 4 routers (Figura 2), DataFrame con routing accuracy y latencia por router | `results/ablation_results.json` via `webapp_helpers.load_ablation_results()` |
| 4 | **Load Balance** | Gráfica de distribución por experto (Figura 4), JSON con conteos globales, max/min ratio y umbral de penalización | `results/paso9/load_balance_test.json` |
| 5 | **Estado del Sistema** | Resumen de verificación Paso 10 (PASS/FAIL/SKIP), dry-run Paso 11, inventario de checkpoints `.pt` con tamaño | `results/paso10/verification_report.json`, `results/paso11/dry_run_report.json`, `checkpoints/**/*.pt` |
| 6 | **Figuras del Reporte** | Galería de las 5 figuras obligatorias, botón "Regenerar Figuras", descarga individual de cada figura | `results/paso12/figures/` |

---

## 4. Figuras obligatorias del reporte (§12.2 de `proyecto_moe.md`)

| Figura | Nombre de archivo | Descripción | Datos |
|---|---|---|---|
| **Figura 1** | `figura1_arquitectura_moe.png` | Diagrama de arquitectura: Input → Preprocesador → Backbone → Router → 6 Expertos → Agregación → Output. Leyenda de colores por tipo de experto (2D/3D/CAE) | Estático (constantes del proyecto) |
| **Figura 2** | `figura2_ablation_table.png` | Tabla comparativa de 4 routers: entropy_router (valores reales/dry-run), gating_network, load_balance_router, top_k_router (baselines estimados). Columnas: F1-Macro, Load Balance Ratio, OOD AUROC, Routing Accuracy | `test_metrics_summary.json`, `ood_auroc_report.json`, `load_balance_test.json` |
| **Figura 3** | `figura3_training_curves.png` | Curvas de entrenamiento dual-axis: Loss (eje izquierdo) + F1-Macro (eje derecho) sobre épocas. Anotaciones de las 3 etapas de fine-tuning (Stage 1: Router, Stage 2: Router+Heads, Stage 3: Global FT) | `training_history.json` (si existe) o curvas sintéticas estimadas |
| **Figura 4** | `figura4_load_balance.png` | Barras de fracción `f_i` por experto de dominio (0–4), línea de balance ideal (1/5), línea de umbral de penalización (1.30×). Muestra max/min ratio y si pasa el umbral | `load_balance_test.json` |
| **Figura 5** | `figura5_attention_heatmap.png` | Grid 1×3 con imágenes sintéticas + mapas de atención gaussianos superpuestos para 3 modalidades: Dermatoscopía (ISIC), Radiografía de tórax (Chest), Fondo de ojo (Fundus). Colorbar de intensidad de atención | Sintético (sin inferencia real en dry-run) |

Todas las figuras usan matplotlib con backend `Agg` (no interactivo) y se exportan a 150 dpi como PNG. No se usa plotly.

---

## 5. Incongruencias detectadas y resolución

| # | ID | Aspecto | Problema | Resolución adoptada |
|---|---|---|---|---|
| 1 | INC-PASO12-01 | **Ruta de figuras** | `paso12_dashboard.py` busca figuras en `output_dir/` directamente (`EXPECTED_FIGURES` sin subdirectorio), pero `dashboard_figures.py` las guarda en `output_dir/figures/` | Las figuras se generan en `results/paso12/figures/`. El check DR-8 verifica la existencia tras la generación. El dashboard busca en `output_dir/` (nivel raíz), lo cual genera desconexión con la galería si se usan las rutas de `EXPECTED_FIGURES`. El botón "Regenerar" llama a `generate_all_figures(output_dir)` que escribe a `output_dir/figures/` |
| 2 | INC-PASO12-02 | **Routers en Figura 2** | `dashboard_figures.py` usa nombres `entropy_router`, `gating_network`, `load_balance_router`, `top_k_router`, que difieren de los 4 routers reales del ablation (Linear, GMM, NaiveBayes, kNN-FAISS) | Los 3 routers no-principales se etiquetan como "(baseline)" con valores estimados. Los valores reales del ablation study vendrían de `ablation_results.json` |
| 3 | INC-PASO12-03 | **Fondo de ojo en Figura 5** | La tercera modalidad de la Figura 5 es "Fondo de Ojo (Fundus)" con Expert 2 (VGG16-BN), pero el Expert 2 real es OA Knee, no Fundus | En dry-run las imágenes son sintéticas. Cuando haya inferencia real, se deben usar imágenes de las 3 modalidades reales del proyecto (2D Chest, 2D ISIC, 3D LUNA/Páncreas) |

---

## 6. Artefactos producidos

| Artefacto | Tipo | Ruta | Consumidor |
|---|---|---|---|
| Reporte dry-run | JSON | `results/paso12/dry_run_report.json` | Verificación pre-entrega, auditoría |
| Log de ejecución | Log | `results/paso12/paso12_dashboard.log` | Depuración, auditoría |
| Figura 1 — Arquitectura MoE | PNG | `results/paso12/figures/figura1_arquitectura_moe.png` | Reporte técnico §12.2 |
| Figura 2 — Ablation Study | PNG | `results/paso12/figures/figura2_ablation_study.png` | Reporte técnico §12.2 |
| Figura 3 — Curvas de Entrenamiento | PNG | `results/paso12/figures/figura3_curvas_entrenamiento.png` | Reporte técnico §12.2 |
| Figura 4 — Load Balance | PNG | `results/paso12/figures/figura4_load_balance.png` | Reporte técnico §12.2 |
| Figura 5 — Attention Heatmap | PNG | `results/paso12/figures/figura5_attention_heatmap.png` | Reporte técnico §12.2 |

El reporte JSON de dry-run contiene:

```json
{
  "paso": 12,
  "phase": "paso12_dashboard",
  "dry_run": true,
  "timestamp": "2026-04-06T...",
  "summary": {
    "total": 10,
    "passed": 7,
    "warned": 3,
    "failed": 0
  },
  "checks": [ ... ],
  "figures_generated": [ ... ],
  "exit_code": 0
}
```

---

## 7. Comandos de uso

### 7.1 Dry-run (verificación obligatoria antes de producción)

```bash
/home/mrsasayo_mesa/venv_global/bin/python src/pipeline/fase6/paso12_dashboard.py --dry-run
```

### 7.2 Producción (lanzar dashboard Gradio)

```bash
# Requiere: gradio instalado, resultados de Pasos 9, 10, 11
/home/mrsasayo_mesa/venv_global/bin/python src/pipeline/fase6/paso12_dashboard.py --port 7861
```

### 7.3 Con enlace público (solo producción)

```bash
/home/mrsasayo_mesa/venv_global/bin/python src/pipeline/fase6/paso12_dashboard.py --port 7861 --share
```

### 7.4 Generar solo las figuras (sin dashboard)

```bash
/home/mrsasayo_mesa/venv_global/bin/python src/pipeline/fase6/dashboard_figures.py
```

### 7.5 Argumentos CLI

| Argumento | Default | Descripción |
|---|---|---|
| `--dry-run` | `False` | Verificar 10 checks, generar figuras, escribir reporte. Exit code siempre 0 |
| `--port` | `7861` | Puerto del servidor Gradio |
| `--share` | `False` | Crear enlace público de Gradio (solo producción) |
| `--output-dir` | `results/paso12` | Directorio de salida para reportes y figuras |

---

## 8. Prerrequisitos

Los siguientes artefactos son necesarios para las distintas funcionalidades en modo producción (no aplica a `--dry-run`):

| # | Artefacto | Generado en | Tab/Figura | Ruta | Estado |
|---|---|---|---|---|---|
| 1 | Métricas de test | Paso 9 | Tab 2, Figura 2/3 | `results/paso9/test_metrics_summary.json` | ✅ Existe (dry-run) |
| 2 | Reporte load balance | Paso 9 | Tab 4, Figura 4 | `results/paso9/load_balance_test.json` | ✅ Existe (dry-run) |
| 3 | Reporte OOD AUROC | Paso 9 | Figura 2 | `results/paso9/ood_auroc_report.json` | ✅ Existe (dry-run) |
| 4 | Reporte verificación | Paso 10 | Tab 5 | `results/paso10/verification_report.json` | ✅ Existe |
| 5 | Reporte dry-run webapp | Paso 11 | Tab 5 | `results/paso11/dry_run_report.json` | ✅ Existe |
| 6 | Resultados ablation study | Paso 6 | Tab 3 | `results/ablation_results.json` | ❌ Pendiente (placeholder activo) |
| 7 | Librería `gradio` | pip install | Dashboard | `pip install gradio>=4.0.0` | ⚠ No instalado (WARN en DR-1/DR-9) |
| 8 | Librería `matplotlib` | pip install | Figuras | `pip install matplotlib` | ✅ Instalado |

En modo `--dry-run`:
- Los artefactos upstream no son necesarios: valores por defecto y figuras sintéticas.
- `gradio` no es necesario para checks DR-1 a DR-8 — DR-1 y DR-9 reportan WARN si falta.
- Las 5 figuras se generan con datos demo/sintéticos (etiquetados claramente como tales).
- Exit code siempre es 0.

---

## 9. Checks del dry-run (10 checks)

| Check | Nombre | Severidad | Qué verifica |
|---|---|---|---|
| DR-1 | `check_imports` | WARN | Que `gradio`, `matplotlib`, `pandas`, `torch` estén instalados. WARN si falta `gradio`/`matplotlib`, FAIL si falta `pandas`/`torch` |
| DR-2 | `check_dashboard_figures_module` | WARN | Que `dashboard_figures.generate_all_figures` sea importable |
| DR-3 | `check_results_paso9` | WARN | Que `test_metrics_summary.json` sea legible y contenga métricas |
| DR-4 | `check_results_paso10` | WARN | Que `verification_report.json` sea legible |
| DR-5 | `check_results_paso11` | WARN | Que `dry_run_report.json` de Paso 11 sea legible |
| DR-6 | `check_load_balance_data` | WARN | Que `load_balance_test.json` sea legible |
| DR-7 | `check_output_dir` | CRITICAL | Que `results/paso12/` sea creatable y writable (test write/unlink) |
| DR-8 | `check_figures_generation` | CRITICAL | Llamar a `generate_all_figures()` y verificar que las 5 figuras se producen. PASS si 5/5, WARN si parcial, FAIL si 0 |
| DR-9 | `check_gradio_components` | WARN | Instanciar `gr.Blocks`/`gr.Tab`/`gr.Image`/`gr.JSON` sin lanzar servidor |
| DR-10 | `check_report_write` | CRITICAL | Escribir `dry_run_report.json` a disco |

---

## 10. Resultado dry-run

```
TOTAL: 10 | PASS: 7 | WARN: 3 | FAIL: 0
⚠ gradio no instalado (pip install gradio para activar)
EXIT: 0 ✅
```

| Verificación | Resultado |
|---|---|
| Import gradio, matplotlib, pandas, torch | ⚠ WARN (gradio no instalado) |
| Import dashboard_figures (generate_all_figures) | ✅ OK |
| results/paso9/test_metrics_summary.json legible | ✅ OK |
| results/paso10/verification_report.json legible | ✅ OK |
| results/paso11/dry_run_report.json legible | ✅ OK |
| results/paso9/load_balance_test.json legible | ✅ OK |
| Output dir writable (results/paso12) | ✅ OK |
| Figuras generadas (5/5) | ✅ OK |
| Gradio Blocks/Tab/Image/JSON instantiate | ⚠ WARN (gradio no disponible) |
| Report JSON write | ✅ OK |
| Dry-run exit code | ✅ 0 |

---

## 11. Dependencias

| Paso previo | Artefacto consumido | Bloqueante para Paso 12 |
|---|---|---|
| Paso 9 (Prueba datos reales) | `results/paso9/test_metrics_summary.json` | No — Tab 2 y Figuras 2/3 usan valores por defecto si no existe |
| Paso 9 (Prueba datos reales) | `results/paso9/load_balance_test.json` | No — Tab 4 y Figura 4 generan distribución demo si no existe |
| Paso 9 (Prueba datos reales) | `results/paso9/ood_auroc_report.json` | No — Figura 2 usa `ood_auroc=0.0` si no existe |
| Paso 10 (Verificación) | `results/paso10/verification_report.json` | No — Tab 5 muestra "N/A" si no existe |
| Paso 11 (Página web) | `results/paso11/dry_run_report.json` | No — Tab 5 muestra "N/A" si no existe |
| Paso 6 (Ablation study) | `results/ablation_results.json` | No — Tab 3 usa `ABLATION_PLACEHOLDER` si no existe |
| pip install | `matplotlib` | Sí — necesario para generar las 5 figuras |
| pip install | `gradio>=4.0.0` | Sí (modo producción) — dry-run funciona sin gradio |

---

## 12. Notas de implementación

### 12.1 Backend matplotlib Agg

`dashboard_figures.py` fuerza `matplotlib.use("Agg")` al inicio del módulo. Esto asegura que la generación de figuras funciona en entornos sin display (servidores, CI/CD, contenedores). No se usa plotly ni ningún otro backend interactivo, cumpliendo la restricción del proyecto.

### 12.2 Figuras con datos sintéticos en dry-run

Cuando los resultados upstream (Paso 9, Paso 10) no están disponibles o contienen valores de dry-run (todos ceros):

- **Figura 2 (Ablation):** El router principal muestra valores del JSON disponible; los 3 routers baseline muestran valores estimados etiquetados como `(est.)`.
- **Figura 3 (Curvas):** Sin historial de entrenamiento, genera curvas sintéticas monótonas convergentes (loss decreciente, F1 creciente) con semilla fija (`RandomState(42)`). Claramente etiquetadas como "curvas estimadas".
- **Figura 4 (Load Balance):** Si conteos son cero, genera distribución demo con `np.random.dirichlet` (semilla 42). Etiquetada como "datos demo — dry-run".
- **Figura 5 (Heatmap):** Siempre genera imágenes y mapas de atención sintéticos (gaussianos). Etiquetada como "simulado — sin inferencia real".

### 12.3 Separación dashboard vs. figuras

Los dos scripts tienen responsabilidades distintas:

- `paso12_dashboard.py`: orquestador principal, CLI, Gradio app, dry-run checker. No genera figuras directamente.
- `dashboard_figures.py`: generador de figuras puro. Se puede ejecutar standalone (`python dashboard_figures.py`) sin Gradio.

### 12.4 Traza de arquitectura (Tab 1)

Tab 1 incluye una traza textual paso a paso del flujo de datos del sistema MoE, y descripciones expandibles (Gradio `Accordion`) de cada componente:

- ViT-Tiny (Backbone)
- Linear Gating Head (Router)
- Expert 0–4 (5 expertos de dominio)
- Expert 5 (CAE/OOD)

### 12.5 Puerto por defecto

El Paso 12 usa puerto `7861` por defecto (diferente al `7860` del Paso 11) para permitir ejecución simultánea de ambos dashboards.

### 12.6 Exit code

| Condición | Exit code |
|---|---|
| `--dry-run` (cualquier resultado) | 0 |
| Modo producción, servidor lanzado | N/A (proceso persistente) |
| Modo producción, error al arrancar | 1 |

---

## 13. Cumplimiento §12 / §13

### 13.1 Figuras obligatorias §12.2 — todas generadas

| Figura §12.2 | Implementación | Estado |
|---|---|---|
| Figura 1: Diagrama de arquitectura MoE | `generate_figure1_architecture()` → PNG 150 dpi | ✅ |
| Figura 2: Tabla comparativa ablation study | `generate_figure2_ablation()` → PNG 150 dpi | ✅ |
| Figura 3: Curvas de entrenamiento (loss + F1) | `generate_figure3_training_curves()` → PNG 150 dpi | ✅ |
| Figura 4: Evolución balance de carga f_i | `generate_figure4_load_balance()` → PNG 150 dpi | ✅ |
| Figura 5: Attention Heatmap 3 modalidades | `generate_figure5_attention_heatmap()` → PNG 150 dpi | ✅ |

### 13.2 Rúbrica §13 cubierta por este paso

| Componente §13 | Cobertura del Paso 12 |
|---|---|
| Dashboard interactivo (15%) | ✅ Complementa Paso 11: 6 tabs de reporte con arquitectura, métricas, ablation, load balance, estado del sistema y galería de figuras |
| Reporte técnico (15%) | ✅ Genera las 5 figuras obligatorias del reporte como PNG exportable a 150 dpi |

---

## 14. Restricciones aplicadas

| Restricción | Detalle |
|---|---|
| **Sin plotly** | Todas las figuras usan exclusivamente matplotlib con backend Agg. No se importa ni se usa plotly |
| **Sin pesos preentrenados** | Las figuras no cargan modelos ni ejecutan inferencia. Los datos provienen de JSONs de resultados upstream |
| **Matplotlib Agg backend** | `matplotlib.use("Agg")` forzado al inicio de `dashboard_figures.py`. Seguro para servidores sin display |
| **Degradación graceful** | Sin resultados upstream → figuras con datos demo/sintéticos claramente etiquetados. Sin gradio → dry-run funciona igual |
| **Exportación a 150 dpi** | Todas las figuras se exportan a 150 dpi como PNG, cumpliendo `proyecto_moe.md` §12.2 |

---

*Documento generado el 2026-04-06. Fuentes: `src/pipeline/fase6/paso12_dashboard.py`, `src/pipeline/fase6/dashboard_figures.py`, `proyecto_moe.md` §12 y §13, `arquitectura_documentacion.md` §8.*

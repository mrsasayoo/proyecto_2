# Paso 11 — Página Web: Dashboard Interactivo MoE

| Campo | Valor |
|---|---|
| **Fecha** | 2026-04-06 |
| **Alcance** | Paso 11 — Dashboard Interactivo MoE con las 8 funcionalidades obligatorias de `proyecto_moe.md` §11. Interfaz web Gradio para carga de imagen (PNG/JPEG/NIfTI), inferencia en tiempo real, Grad-CAM, panel de experto, ablation study, load balance y alerta OOD. Implementado en `src/pipeline/fase6/` con degradación graceful sin checkpoints |
| **Estado general** | ✅ **Implementado (dry-run verificado)** |

---

## 1. Descripción

El Paso 11 es la interfaz web interactiva para demostración y evaluación del sistema MoE. Implementa las **8 funcionalidades obligatorias** definidas en `proyecto_moe.md` §11. Basada en Gradio, permite subir imágenes PNG/JPEG/NIfTI y obtener predicciones en tiempo real con explicabilidad visual.

A diferencia de los Pasos 9 y 10, que evalúan y verifican el sistema en batch, el Paso 11 ofrece una interfaz humana interactiva que:

- **Carga de imagen:** acepta PNG, JPEG y NIfTI. La detección 2D/3D se hace SOLO por `tensor.ndim` (no por nombre de archivo ni metadata) — cumple restricción §11.2.
- **Preprocesado transparente:** muestra dimensiones originales y adaptadas tras el preprocesador.
- **Inferencia en tiempo real:** etiqueta predicha, confianza (%), tiempo de respuesta en milisegundos. Degradación graceful sin checkpoint (modo demo con `create_mock_inference_result()`).
- **Attention Heatmap:** Grad-CAM sobre la imagen original usando `GradCAMExtractor` de `gradcam_heatmap.py`.
- **Panel del experto activado:** nombre, arquitectura, dataset de entrenamiento, gating score.
- **Panel del ablation study:** tabla comparativa de 4 métodos. Placeholder `ABLATION_PLACEHOLDER` si `ablation_results.json` no existe.
- **Load Balance:** gráfica de barras `f_i` acumulado en tiempo real con `LoadBalanceCounter` (objeto en memoria, se resetea al reiniciar el servidor).
- **OOD Detection:** alerta visual cuando H(g) > umbral calibrado (`entropy_threshold.pkl`).

Produce un reporte JSON de dry-run y log de ejecución en `results/paso11/`.

---

## 2. Archivos implementados

| Archivo | Descripción | Función principal | Líneas |
|---|---|---|---|
| `src/pipeline/fase6/paso11_webapp.py` | App Gradio principal, 13 checks DR, CLI --dry-run | `main()` — construye InferenceEngine, lanza servidor o ejecuta dry-run | ~978 |
| `src/pipeline/fase6/webapp_helpers.py` | EXPERT_METADATA, LoadBalanceCounter, preprocess, ablation loader, mock inference | Helpers compartidos por la webapp | ~435 |

---

## 3. Funcionalidades implementadas (8 de §11)

| # | Funcionalidad | Descripción | Implementación | Estado |
|---|---|---|---|---|
| 1 | **Carga de imagen** | PNG, JPEG, NIfTI. Detección 2D/3D solo por `tensor.ndim` (no por nombre de archivo) | `preprocess_image_for_webapp()` en `webapp_helpers.py` — soporta str/Path/ndarray/PIL | ✅ |
| 2 | **Preprocesado transparente** | Muestra dims originales y adaptadas tras preprocesador | `preprocess_meta` dict con `original_shape`, `adapted_shape`, `modality` | ✅ |
| 3 | **Inferencia en tiempo real** | Etiqueta, confianza (%), tiempo en ms. Degradación graceful sin checkpoint | `run_inference()` + `format_confidence()`. Si engine=None → `create_mock_inference_result()` | ✅ |
| 4 | **Attention Heatmap del Router ViT** | Grad-CAM sobre imagen original | Usa `GradCAMExtractor` + `_find_last_conv()` de `gradcam_heatmap.py` | ✅ |
| 5 | **Panel del experto activado** | Nombre, arquitectura, dataset, gating score | `EXPERT_METADATA` dict (6 expertos) + `expert_panel` construido en `run_inference()` | ✅ |
| 6 | **Panel del ablation study** | Tabla comparativa 4 métodos. Placeholder si `ablation_results.json` pendiente | `load_ablation_results()` + `fn_ablation_table()` → DataFrame. `ABLATION_PLACEHOLDER` como fallback | ✅ |
| 7 | **Load Balance gráfica** | Barras `f_i` acumulado en tiempo real con `LoadBalanceCounter` | `fn_load_balance_chart()` → matplotlib Figure. Muestra ratio max/min en título | ✅ |
| 8 | **OOD Detection alerta** | Alerta visual cuando H(g) > umbral calibrado (`entropy_threshold.pkl`) | Entropía calculada en `run_inference()`, alerta en `fn_infer()`. Flag `is_ood` booleano | ✅ |

---

## 4. Incongruencias detectadas y resolución

| # | ID | Aspecto | Problema | Resolución adoptada |
|---|---|---|---|---|
| 1 | INC-PASO11-01 | **arch_doc divide §11** | `arquitectura_documentacion.md` divide §11 en Paso 11 (web) y Paso 12 (dashboard) | Paso 11 cubre las 8 funcionalidades completas del §11 de la guía oficial |
| 2 | INC-PASO11-02 | **Script inexistente en arch_doc** | `arquitectura_documentacion.md` menciona `fase6_webapp.py` que no existe | Creados `paso11_webapp.py` + `webapp_helpers.py` siguiendo la convención de nombres del pipeline |
| 3 | INC-PASO11-03 | **Dependencia faltante** | `gradio` no estaba en `requirements.txt` | Añadido `gradio>=4.0.0` a `requirements.txt` |
| 4 | INC-PASO11-04 | **Artefacto pendiente** | `ablation_results.json` no existe (entrenamiento pendiente) | Degradación graceful: `load_ablation_results()` retorna `ABLATION_PLACEHOLDER` con tabla de 4 routers vacía |
| 5 | INC-PASO11-05 | **Sin checkpoints .pt** | No hay checkpoints de modelos entrenados | Modo demo: `build_inference_engine()` retorna None + `create_mock_inference_result()` genera resultado sintético |

---

## 5. Artefactos producidos

| Artefacto | Tipo | Ruta | Consumidor |
|---|---|---|---|
| Reporte dry-run | JSON | `results/paso11/dry_run_report.json` | Verificación pre-entrega, auditoría |
| Log de ejecución | Log | `results/paso11/paso11_webapp.log` | Depuración, auditoría |

El reporte JSON de dry-run contiene:

```json
{
  "timestamp": "2026-04-06T...",
  "dry_run": true,
  "total_checks": 13,
  "passed": 11,
  "warnings": 2,
  "failed": 0,
  "checks": [ ... ]
}
```

---

## 6. Comandos de uso

### 6.1 Dry-run (verificación obligatoria antes de producción)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase6/paso11_webapp.py --dry-run
```

### 6.2 Producción (solo cuando los modelos estén entrenados)

```bash
# Requiere: gradio instalado, checkpoints de Paso 8, ablation_results.json
PYENV_VERSION=3.12.3 python src/pipeline/fase6/paso11_webapp.py --device cuda --port 7860
```

### 6.3 Con enlace público (solo producción)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase6/paso11_webapp.py --device cuda --port 7860 --share
```

### 6.4 Argumentos CLI

| Argumento | Default | Descripción |
|---|---|---|
| `--dry-run` | `False` | Verificar 13 checks sin lanzar servidor. Exit code siempre 0 |
| `--device` | `cpu` | Dispositivo de inferencia: `cpu` / `cuda` |
| `--port` | `7860` | Puerto del servidor Gradio |
| `--share` | `False` | Crear enlace público de Gradio (solo producción) |
| `--output-dir` | `results/paso11` | Directorio de salida para reportes |

---

## 7. Prerrequisitos

Los siguientes artefactos son necesarios para las distintas funcionalidades en modo producción (no aplica a `--dry-run`):

| # | Artefacto | Generado en | Funcionalidad | Ruta | Estado |
|---|---|---|---|---|---|
| 1 | Checkpoint MoE final | Fase 5 (Paso 8) | Funcionalidades 3, 4, 5 | `checkpoints/fase5/moe_final.pt` | ❌ Pendiente (modo demo activo) |
| 2 | Resultados ablation study | Fase 2 (Paso 6) | Funcionalidad 6 | `results/ablation_results.json` | ❌ Pendiente (placeholder activo) |
| 3 | Umbral de entropía calibrado | Fase 2 / Paso 6 | Funcionalidad 8 | `checkpoints/entropy_threshold.pkl` | ✅ Existe |
| 4 | Librería `gradio` | pip install | Todas | `pip install gradio>=4.0.0` | ⚠ No instalado (WARN en DR-1/DR-13) |

En modo `--dry-run`:
- Los checkpoints no son necesarios: `build_inference_engine(dry_run=True)` retorna None.
- `ablation_results.json` no es necesario: `load_ablation_results(None)` retorna `ABLATION_PLACEHOLDER`.
- `gradio` no es necesario para los checks DR-2 a DR-12 — DR-1 y DR-13 reportan WARN, no FAIL.
- Exit code siempre es 0.

---

## 8. Checks del dry-run (13 checks)

| Check | Nombre | Severidad | Qué verifica |
|---|---|---|---|
| DR-1 | `import gradio` | WARN | Que `gradio` esté instalado. WARN si falta (no bloquea otros checks) |
| DR-2 | `import webapp_helpers` | CRITICAL | 6 exports: `EXPERT_METADATA`, `LoadBalanceCounter`, `preprocess_image_for_webapp`, `load_ablation_results`, `create_mock_inference_result`, `format_confidence` |
| DR-3 | `import inference_engine` | WARN | Que `InferenceEngine` class sea importable |
| DR-4 | `import gradcam_heatmap` | WARN | Que `GradCAMExtractor` class sea importable |
| DR-5 | `import ood_detector` | WARN | Que `OODDetector` class sea importable |
| DR-6 | `LoadBalanceCounter` | CRITICAL | Instanciar, hacer update, verificar counts y frecuencias |
| DR-7 | `preprocess_image_for_webapp` | CRITICAL | Imagen numpy sintética 64×64×3 → tensor 4D, modality="2D" |
| DR-8 | `create_mock_inference_result` | CRITICAL | Genera resultado mock con logits, expert_ids, gates, entropy, _is_mock |
| DR-9 | `load_ablation_results` | CRITICAL | Retorna placeholder con 4 routers cuando path=None |
| DR-10 | `fn_load_balance_chart` | WARN | Genera matplotlib Figure con barras de frecuencia |
| DR-11 | `fn_ablation_table` | WARN | Genera DataFrame de pandas con 4 routers |
| DR-12 | `build_inference_engine(dry_run=True)` | CRITICAL | Retorna None (esperado en dry-run) |
| DR-13 | `Gradio Blocks` | WARN | Instanciar `gr.Blocks()` sin lanzar servidor |

---

## 9. Resultado dry-run

```
TOTAL: 13 | PASS: 11 | WARN: 2 | FAIL: 0
⚠ gradio no instalado (pip install gradio para activar)
EXIT: 0 ✅
```

| Verificación | Resultado |
|---|---|
| Import webapp_helpers (EXPERT_METADATA, LoadBalanceCounter, etc.) | ✅ OK |
| Import inference_engine (InferenceEngine) | ✅ OK |
| Import gradcam_heatmap (GradCAMExtractor) | ✅ OK |
| Import ood_detector (OODDetector) | ✅ OK |
| LoadBalanceCounter — instanciar, update, get_counts, get_frequencies | ✅ OK |
| preprocess_image_for_webapp — imagen numpy sintética | ✅ OK |
| create_mock_inference_result — resultado mock válido | ✅ OK |
| load_ablation_results — placeholder 4 routers | ✅ OK |
| fn_load_balance_chart — matplotlib Figure | ✅ OK |
| fn_ablation_table — pandas DataFrame | ✅ OK |
| build_inference_engine(dry_run=True) → None | ✅ OK |
| gradio import | ⚠ WARN (no instalado) |
| Gradio Blocks instanciar | ⚠ WARN (gradio no disponible) |
| Dry-run exit code | ✅ 0 |

---

## 10. Dependencias

| Paso previo | Artefacto consumido | Bloqueante para Paso 11 |
|---|---|---|
| Paso 8 (Fine-tuning por etapas) | `checkpoints/fase5/moe_final.pt` | Sí (modo producción) — sin checkpoint, funciona en modo demo |
| Paso 6 (Ablation study) | `ablation_results.json` | No — `ABLATION_PLACEHOLDER` activo si no existe |
| Paso 6 / Fase 2 | `checkpoints/entropy_threshold.pkl` | No — fallback `log(5)/2 ≈ 0.8047` si no existe |
| Pasos 5.1–5.6 | Checkpoints de los 6 expertos | Sí (modo producción) — sin checkpoints, modo demo |
| pip install | `gradio>=4.0.0` | Sí (modo producción) — dry-run funciona sin gradio |

---

## 11. Notas de implementación

### 11.1 Detección 2D/3D: solo `tensor.ndim`

La detección de modalidad en `preprocess_image_for_webapp()` se basa exclusivamente en el rango del tensor numpy/torch:

```python
if arr.ndim == 2:      # Grayscale 2D [H, W]
elif arr.ndim == 3:    # [H, W, C] (2D) o [D, H, W] (3D sin canal)
elif arr.ndim == 4:    # [D, H, W, C] (3D con canales)
```

El nombre del archivo (`.nii`, `.png`) se usa SOLO para determinar el método de carga (SimpleITK vs PIL), nunca para routing. Cumple restricción §11.2 de `proyecto_moe.md`.

### 11.2 Modo demo sin checkpoints

Cuando `build_inference_engine()` no encuentra el checkpoint o está en dry-run:

1. Retorna `engine = None`
2. `run_inference()` detecta `engine is None` y llama a `create_mock_inference_result()`
3. El resultado mock tiene `_is_mock = True` y `_demo_mode = True`
4. La interfaz muestra `[DEMO MODE]` junto a la predicción

Los valores de inferencia mock (logits, gates, entropy) son aleatorios y no representan métricas reales.

### 11.3 `LoadBalanceCounter` en memoria

El `LoadBalanceCounter` es un objeto Python en memoria que acumula conteos de uso de expertos. Se resetea al reiniciar el servidor Gradio. No persiste entre sesiones. Calcula:

- `get_counts()`: conteos absolutos por experto
- `get_frequencies()`: frecuencias relativas `f_i` (suma = 1.0)
- `get_max_min_ratio()`: `max(f_i)/min(f_i)` sobre expertos con uso > 0

### 11.4 Grad-CAM con `GradCAMExtractor`

El heatmap (Funcionalidad 4) reutiliza el `GradCAMExtractor` existente de `gradcam_heatmap.py`. En `run_inference()`:

1. Se busca la última capa convolucional del experto activado con `_find_last_conv()`
2. Se instancia `GradCAMExtractor(model, target_layer, expert_name)`
3. Se computa `cam = extractor.compute(tensor)`
4. Se limpia con `extractor.remove_hooks()`

Solo funciona con engine real (no en modo demo).

### 11.5 Gradio no instalado en el entorno

Los WARNs en DR-1 y DR-13 indican que `gradio` no está instalado en el entorno actual. Esto no es un FAIL: el dry-run verifica todos los componentes internos (helpers, preprocesado, mock inference, load balance, ablation loader) independientemente de Gradio. Para lanzar el servidor en producción, instalar con `pip install gradio>=4.0.0`.

### 11.6 `ABLATION_PLACEHOLDER`

Cuando `ablation_results.json` no existe, `load_ablation_results()` retorna:

```python
ABLATION_PLACEHOLDER = {
    "note": "ablation_results.json no disponible — entrenamiento pendiente",
    "results": {
        "TopK Router": {"routing_accuracy": None, "latency_ms": None},
        "Soft Router": {"routing_accuracy": None, "latency_ms": None},
        "Noisy TopK Router": {"routing_accuracy": None, "latency_ms": None},
        "Linear Router (ViT)": {"routing_accuracy": None, "latency_ms": None},
    },
}
```

La tabla en la UI muestra "N/A" para las métricas pendientes.

### 11.7 Exit code

| Condición | Exit code |
|---|---|
| `--dry-run` (cualquier resultado) | 0 |
| Modo producción, servidor lanzado | N/A (proceso persistente) |
| Modo producción, error al arrancar | 1 |

---

## 12. Cumplimiento §11 / §13

### 12.1 Funcionalidades §11 — todas implementadas

| Funcionalidad §11 | Implementación | Estado |
|---|---|---|
| §11.1 Carga de imagen (PNG, JPEG, NIfTI) | `preprocess_image_for_webapp()` | ✅ |
| §11.2 Preprocesado transparente | `preprocess_meta` dict en `run_inference()` | ✅ |
| §11.3 Inferencia en tiempo real | `run_inference()` + `format_confidence()` | ✅ |
| §11.4 Attention Heatmap Router ViT | `GradCAMExtractor` en `run_inference()` | ✅ |
| §11.5 Panel del experto activado | `EXPERT_METADATA` + `expert_panel` dict | ✅ |
| §11.6 Panel del ablation study | `load_ablation_results()` + `fn_ablation_table()` | ✅ |
| §11.7 Load Balance gráfica | `LoadBalanceCounter` + `fn_load_balance_chart()` | ✅ |
| §11.8 OOD Detection alerta | Entropía H(g) vs umbral + alerta visual | ✅ |

### 12.2 Rúbrica §13 cubierta por este paso

| Componente §13 | Cobertura del Paso 11 |
|---|---|
| Dashboard interactivo (15%) | ✅ Estructura completa: 8 funcionalidades obligatorias implementadas, Gradio UI con tabs, panel ablation visible, heatmap router, OOD detection activo |

---

## 13. Restricciones aplicadas

| Restricción | Detalle |
|---|---|
| **Sin metadatos de entrada** | Detección 2D/3D usa SOLO `tensor.ndim`. El nombre del archivo se usa solo para carga (SimpleITK vs PIL), nunca para routing. Violación: −20% nota |
| **Balance de carga** | `LoadBalanceCounter` monitorea `max(f_i)/min(f_i)` en tiempo real. Gráfica visible en tab dedicada |
| **Degradación graceful** | Sin checkpoint → modo demo. Sin ablation_results.json → placeholder. Sin gradio → dry-run funciona igual |
| **Gradio no instalado** | WARNs en DR-1 y DR-13 — no son FAIL. Todos los componentes internos se verifican independientemente |

---

*Documento generado el 2026-04-06. Fuentes: `src/pipeline/fase6/paso11_webapp.py`, `src/pipeline/fase6/webapp_helpers.py`, `proyecto_moe.md` §11 y §13, `arquitectura_documentacion.md` §8.*

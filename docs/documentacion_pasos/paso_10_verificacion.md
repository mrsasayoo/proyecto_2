# Paso 10 — Verificación Funcional Completa

| Campo | Valor |
|---|---|
| **Fecha** | 2026-04-05 |
| **Alcance** | Paso 10 — Verificación funcional completa del pipeline MoE. Valida todos los componentes contra los requisitos del proyecto (`proyecto_moe.md` §10 y §13), detecta violaciones de penalización (−40% load balance, −20% metadata), genera reporte JSON. Implementado en `src/pipeline/fase6/` |
| **Estado general** | ✅ **Implementado — dry-run verificado** |

---

## 1. Descripción

El Paso 10 es la verificación funcional integral del sistema MoE. A diferencia del Paso 9, que ejecuta inferencia sobre datos reales y mide métricas, el Paso 10 verifica que **todos los componentes cumplan los requisitos del proyecto** antes de la entrega final.

El paso verifica:

- **Imports y smoke tests:** que todos los módulos del pipeline (config, fase6_config, InferenceEngine, OODDetector) se importen correctamente y expongan las interfaces esperadas.
- **Restricción de no-metadata:** análisis estático del código fuente para confirmar que ningún componente usa metadatos (`.dcm`, `.nii`, `modality`) para routing — solo `tensor.ndim`.
- **Balance de carga:** lectura de `load_balance_test.json` y verificación de `max_ratio ≤ 1.30` (penalización −40%).
- **Umbrales de métricas §10:** F1 2D > 0.72, F1 3D > 0.65, Routing Accuracy > 0.80, OOD AUROC > 0.80.
- **Reproducibilidad:** existencia de `requirements.txt`, seeds=42, `README.md`, ausencia de rutas absolutas.
- **Artefactos generados:** existencia de checkpoints y JSONs de resultados producidos por pasos previos.
- **Augmentaciones prohibidas:** análisis estático para detectar `RandomVerticalFlip`, `RandomErasing`, `CutMix`, `MixUp`, `GridMask` en código activo.
- **Restricciones VRAM:** presencia de gradient checkpointing, FP16 autocast y gradient accumulation en el código fuente.
- **Arquitectura:** 5 expertos de dominio, 6 totales, al menos 4 tipos de router, backbone compartido.

Produce un reporte JSON con resultados individuales por check, resumen de penalizaciones detectadas y estadísticas globales (PASS/FAIL/SKIP) en `results/paso10/`.

---

## 2. Archivos implementados

| Archivo | Descripción | Función principal |
|---|---|---|
| `src/pipeline/fase6/paso10_verificacion.py` | Orquestador de verificación completa (CLI) | `main()` — ejecuta 9 categorías de checks (A–I), genera `verification_report.json` |

---

## 3. Checks implementados (9 categorías)

### 3.1 Categoría A — Imports & Smoke Tests (5 checks)

| Check | Severidad | Qué verifica |
|---|---|---|
| A1 | CRITICAL | `config.N_EXPERTS_DOMAIN == 5` y `config.N_EXPERTS_TOTAL == 6` |
| A2 | CRITICAL | `fase6_config` expone `F1_THRESHOLD_2D` y `LOAD_BALANCE_MAX_RATIO` |
| A3 | CRITICAL | `logging_utils.setup_logging` es callable |
| A4 | CRITICAL | `InferenceEngine` existe como clase en `inference_engine.py` |
| A5 | CRITICAL | `OODDetector` existe como clase en `ood_detector.py` |

### 3.2 Categoría B — Restricción No-Metadata (2 checks)

| Check | Severidad | Qué verifica |
|---|---|---|
| B1 | CRITICAL | Análisis estático de todo `src/`: no hay patrones de routing basado en metadata (`.dcm`, `.nii`, `metadata['modality']`). Verifica presencia positiva de referencias `ndim` |
| B2 | CRITICAL | `inference_engine.py` específicamente usa detección basada en `ndim` |

**Penalización asociada:** −20% de la nota si B1 falla.

### 3.3 Categoría C — Load Balance (2 checks)

| Check | Severidad | Qué verifica |
|---|---|---|
| C1 | CRITICAL | `results/paso9/load_balance_test.json` → `max_ratio ≤ 1.30` |
| C2 | CRITICAL | `LOAD_BALANCE_MAX_RATIO == 1.30` definido en `fase6_config.py` |

**Penalización asociada:** −40% de la nota si C1 falla (max_ratio > 1.30).

### 3.4 Categoría D — Umbrales de Métricas §10 (4 checks)

| Check | Severidad | Umbral | Fuente |
|---|---|---|---|
| D1 | CRITICAL | F1 Macro 2D > 0.72 | `test_metrics_summary.json` → `f1_macro_2d_mean` |
| D2 | CRITICAL | F1 Macro 3D > 0.65 | `test_metrics_summary.json` → `f1_macro_3d_mean` |
| D3 | CRITICAL | Routing Accuracy > 0.80 | `test_metrics_summary.json` → `routing_accuracy_mean` |
| D4 | CRITICAL | OOD AUROC > 0.80 | `ood_auroc_report.json` → `ood_auroc` |

En modo `--dry-run`, D1–D4 se marcan como SKIP (resultados con valores dummy F1=0.0).

### 3.5 Categoría E — Reproducibilidad (4 checks)

| Check | Severidad | Qué verifica |
|---|---|---|
| E1 | CRITICAL | `requirements.txt` existe y contiene `torch` |
| E2 | WARNING | No hay rutas absolutas (`/home/`, `/root/`, `/mnt/`) en código fuente (excluye `Path(__file__)` y `PROJECT_ROOT`) |
| E3 | WARNING | `SEED=42` presente en código fuente |
| E4 | WARNING | `README.md` existe en la raíz del proyecto |

### 3.6 Categoría F — Artefactos Generados (5 checks)

| Check | Severidad | Artefacto verificado |
|---|---|---|
| F1 | CRITICAL | `checkpoints/entropy_threshold.pkl` |
| F2 | CRITICAL | `results/paso9/test_metrics_summary.json` |
| F3 | CRITICAL | `results/paso9/load_balance_test.json` |
| F4 | CRITICAL | `results/paso9/ood_auroc_report.json` |
| F5 | WARNING | Al menos 1 archivo `.pt` en `checkpoints/` |

### 3.7 Categoría G — Augmentaciones Prohibidas (5 checks)

| Check | Severidad | Augmentación prohibida |
|---|---|---|
| G1 | CRITICAL | `RandomVerticalFlip` |
| G2 | CRITICAL | `RandomErasing` |
| G3 | CRITICAL | `CutMix` |
| G4 | CRITICAL | `MixUp` |
| G5 | CRITICAL | `GridMask` |

Análisis estático: busca cada patrón en todos los archivos `.py` bajo `src/`, ignorando líneas comentadas.

### 3.8 Categoría H — Restricciones VRAM (3 checks)

| Check | Severidad | Patrón buscado |
|---|---|---|
| H1 | WARNING | `gradient_checkpointing` o `use_gradient_checkpointing` |
| H2 | WARNING | `torch.cuda.amp` o `autocast` |
| H3 | WARNING | `accumulation_steps` o `grad_accum` |

### 3.9 Categoría I — Arquitectura (4 checks)

| Check | Severidad | Qué verifica |
|---|---|---|
| I1 | CRITICAL | `N_EXPERTS_DOMAIN == 5` |
| I2 | CRITICAL | `N_EXPERTS_TOTAL == 6` |
| I3 | WARNING | Al menos 4 tipos de router referenciados en código (LinearGatingHead, TopKRouter, SoftRouter, etc.) |
| I4 | WARNING | Referencias a `backbone` en `src/pipeline/fase2/` |

---

## 4. Detección de penalizaciones

El script detecta automáticamente violaciones que implican penalización en la nota del proyecto:

| Penalización | Porcentaje | Condición de activación | Check asociado |
|---|---|---|---|
| **Load Balance** | −40% | `max_ratio > 1.30` en `load_balance_test.json` | C1 = FAIL |
| **Metadata Routing** | −20% | Patrones de routing basado en metadata detectados en código fuente | B1 = FAIL |

Las penalizaciones se reportan en el campo `penalties_detected` del JSON de salida.

---

## 5. Artefactos producidos

| Artefacto | Tipo | Ruta | Consumidor |
|---|---|---|---|
| Reporte de verificación | JSON | `results/paso10/verification_report.json` | Reporte técnico, revisión pre-entrega |
| Log de ejecución | Log | `results/paso10/paso10.log` | Depuración, auditoría |

El reporte JSON contiene:

```json
{
  "timestamp": "2026-04-05T...",
  "dry_run": true,
  "total_checks": 34,
  "passed": 20,
  "failed": 3,
  "skipped": 11,
  "warnings": 1,
  "critical_failures": 2,
  "penalties_detected": [],
  "summary": "...",
  "checks": [ ... ]
}
```

En modo `--dry-run`, los checks de métricas (D1–D4) y load balance (C1) se marcan como SKIP, y el exit code es siempre 0.

---

## 6. Comandos de uso

### 6.1 Dry-run (verifica setup sin resultados reales)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase6/paso10_verificacion.py --dry-run
```

### 6.2 Verificación completa (requiere resultados de Paso 9)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase6/paso10_verificacion.py
```

### 6.3 Con directorio de salida personalizado

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase6/paso10_verificacion.py --output-dir results/paso10_custom
```

### 6.4 Argumentos CLI

| Argumento | Default | Descripción |
|---|---|---|
| `--dry-run` | `False` | Modo verificación ligera: métricas no se validan, exit code siempre 0 |
| `--device` | `cpu` | Device de ejecución: `cpu` / `cuda` |
| `--output-dir` | `results/paso10` | Directorio de salida para reporte JSON y log |

---

## 7. Prerrequisitos

Los siguientes artefactos deben existir antes de ejecutar el Paso 10 en modo producción (no aplica a `--dry-run`):

| # | Artefacto | Generado en | Ruta |
|---|---|---|---|
| 1 | Métricas de test por experto | Paso 9 | `results/paso9/test_metrics_summary.json` |
| 2 | Reporte de load balance | Paso 9 | `results/paso9/load_balance_test.json` |
| 3 | Reporte OOD AUROC | Paso 9 | `results/paso9/ood_auroc_report.json` |
| 4 | Umbral de entropía calibrado | Paso 6 / Fase 2 | `checkpoints/entropy_threshold.pkl` |
| 5 | Al menos 1 checkpoint `.pt` | Pasos 5.1–5.6 / Paso 8 | `checkpoints/**/*.pt` |
| 6 | `requirements.txt` | Manual | `requirements.txt` |
| 7 | `README.md` | Manual | `README.md` |

En modo `--dry-run`:
- Los artefactos de Paso 9 no son necesarios: checks D1–D4 y C1 se marcan SKIP.
- Los checks de análisis estático (B, E, G, H, I) se ejecutan normalmente.
- Exit code siempre es 0.

---

## 8. Hallazgos relevantes

### 8.1 Augmentaciones prohibidas (Wave 1B — corregidas)

Durante la implementación se detectaron 3 violaciones de augmentaciones prohibidas en código activo, todas corregidas:

| # | Archivo | Augmentación | Acción |
|---|---|---|---|
| 1 | `datasets/isic.py` | `RandomVerticalFlip` | Eliminado |
| 2 | `datasets/isic.py` | `RandomErasing` | Eliminado |
| 3 | `src/pipeline/fase2/dataloader_expert2.py` | `RandomVerticalFlip` | Eliminado |

Post-corrección, los checks G1 y G2 pasan exitosamente.

### 8.2 Checks de métricas en dry-run

Los checks D1–D4 se marcan como SKIP en modo `--dry-run` porque los resultados de Paso 9 contienen valores dummy (F1=0.0, routing_accuracy=0.0, ood_auroc=0.0). Estos valores provienen de la ejecución con pesos aleatorios y son meaningless — la validación real requiere los resultados de una ejecución completa del Paso 9 con checkpoint entrenado.

### 8.3 Artefactos faltantes en dry-run

Los checks F1–F4 pueden fallar en entornos donde el Paso 9 no se ha ejecutado previamente. Esto es esperado y no indica un bug — indica que el pipeline aún no ha producido los artefactos necesarios.

---

## 9. Dependencias

| Paso previo | Artefacto consumido | Bloqueante para Paso 10 |
|---|---|---|
| Paso 9 (Prueba datos reales) | `results/paso9/test_metrics_summary.json` | Sí (modo producción) |
| Paso 9 (Prueba datos reales) | `results/paso9/load_balance_test.json` | Sí (modo producción) |
| Paso 9 (Prueba datos reales) | `results/paso9/ood_auroc_report.json` | Sí (modo producción) |
| Paso 6 (Ablation study) | `checkpoints/entropy_threshold.pkl` | Sí (check F1) |
| Pasos 5.1–5.6 / Paso 8 | Checkpoints `.pt` | Sí (check F5) |

El Paso 9 debe haberse completado exitosamente para ejecutar el Paso 10 en modo producción. En `--dry-run`, los checks de análisis estático se ejecutan normalmente y los de métricas se omiten.

---

## 10. Notas de implementación

### 10.1 Análisis estático con `_grep_source_files()`

El script implementa una función de búsqueda regex sobre todos los archivos `.py` bajo `src/`. Por defecto ignora líneas comentadas (`skip_comments=True`) para evitar falsos positivos. Las categorías B (no-metadata), E (rutas absolutas, seeds), G (augmentaciones prohibidas) y H (VRAM) usan esta función.

### 10.2 Lectura segura de JSON con `_read_json_safe()`

Los artefactos JSON de Paso 9 se leen con `_read_json_safe()`, que retorna `None` si el archivo no existe o tiene formato inválido. Esto permite que el script se ejecute incluso sin artefactos previos, marcando los checks correspondientes como SKIP.

### 10.3 Detección de resultados dry-run

El script detecta automáticamente si los artefactos de Paso 9 provienen de una ejecución `--dry-run` verificando el campo `dry_run` en los JSONs. Si detecta resultados dry-run, marca los checks de métricas como SKIP independientemente del flag `--dry-run` del Paso 10.

### 10.4 Exit code

| Condición | Exit code |
|---|---|
| `--dry-run` (cualquier resultado) | 0 |
| Modo producción, 0 fallos críticos | 0 |
| Modo producción, ≥1 fallo crítico | 1 |

---

## 11. Cumplimiento §10 / §13

### 11.1 Métricas §10 validadas por este paso

| Métrica §10 | Check | Umbral | Estado dry-run |
|---|---|---|---|
| F1 Macro 2D > 0.72 | D1 | 0.72 | SKIP (dummy) |
| F1 Macro 3D > 0.65 | D2 | 0.65 | SKIP (dummy) |
| Routing Accuracy > 0.80 | D3 | 0.80 | SKIP (dummy) |
| OOD AUROC > 0.80 | D4 | 0.80 | SKIP (dummy) |
| Load Balance < 1.30 | C1, C2 | 1.30 | SKIP (C1) / PASS (C2) |

### 11.2 Rúbrica §13 cubierta por este paso

| Componente §13 | Checks asociados | Cobertura |
|---|---|---|
| Sistema MoE completo (20%) | A1–A5, I1–I4 | Verifica imports, N_EXPERTS, backbone compartido |
| Métricas de clasificación (20%) | D1–D4 | Valida umbrales F1, routing, OOD |
| Repositorio Git y reproducibilidad (5%) | E1–E4 | requirements.txt, seeds, README, no abs paths |
| Penalización Load Balance (−40%) | C1, C2 | Detecta violación max_ratio > 1.30 |
| Penalización Metadata (−20%) | B1, B2 | Análisis estático anti-metadata |

---

## 12. Dry-run verificado

| Verificación | Resultado |
|---|---|
| Importaciones de módulos (config, fase6_config, logging_utils, InferenceEngine, OODDetector) | ✅ OK |
| Análisis estático no-metadata (B1, B2) | ✅ OK |
| Load balance `LOAD_BALANCE_MAX_RATIO == 1.30` (C2) | ✅ OK |
| Métricas D1–D4 marcadas SKIP (dry-run) | ✅ OK |
| Reproducibilidad (E1–E4) | ✅ OK |
| Augmentaciones prohibidas (G1–G5) | ✅ OK (post-corrección Wave 1B) |
| VRAM patterns (H1–H3) | ✅ OK |
| Arquitectura (I1–I4) | ✅ OK |
| Reporte JSON generado en `results/paso10/verification_report.json` | ✅ OK |
| Dry-run exit code | ✅ 0 |
| Verificación con resultados reales de Paso 9 | ⏳ Pendiente (requiere ejecución completa de Paso 9) |

---

## 13. Restricciones aplicadas

| Restricción | Detalle |
|---|---|
| **Sin metadatos de entrada** | Verificado por análisis estático (B1, B2). Solo `tensor.ndim`. Violación: −20% nota |
| **Balance de carga** | `max(f_i)/min(f_i) ≤ 1.30` sobre expertos 0–4. Verificado contra JSON de Paso 9. Violación: −40% nota |
| **Augmentaciones prohibidas** | `RandomVerticalFlip`, `RandomErasing`, `CutMix`, `MixUp`, `GridMask` no presentes en código activo |
| **Reproducibilidad** | `requirements.txt` con torch, `SEED=42`, `README.md`, sin rutas absolutas |
| **Umbrales §10** | F1 2D>0.72, F1 3D>0.65, Routing>0.80, OOD AUROC>0.80 — validados contra JSONs de Paso 9 |

---

*Documento generado el 2026-04-05. Fuentes: `src/pipeline/fase6/paso10_verificacion.py`, `proyecto_moe.md` §10 y §13, `arquitectura_documentacion.md`, `src/pipeline/fase6/fase6_config.py`.*

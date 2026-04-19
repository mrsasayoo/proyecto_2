# Paso 6 — Ablation Study del Router MoE

| Campo | Valor |
|---|---|
| **Fecha** | 2026-04-05 |
| **Alcance** | Step 6 — Ablation Study del router MoE. Compara 4 estrategias de routing (Linear, GMM, NaiveBayes, kNN-FAISS) sobre embeddings pre-computados de Fase 1 para seleccionar el mejor router del gating network. No usa datos de imagen — trabaja exclusivamente sobre vectores de embedding |
| **Estado general** | ✅ **Implementado — dry-run verificado** |

---

## 1. Objetivo

Comparar **4 estrategias de routing** (Linear + Softmax, GMM, Naive Bayes, kNN-FAISS) sobre embeddings pre-computados por Fase 1 para seleccionar el mejor router de la red de gating del sistema MoE. El ablation study responde la pregunta científica central del proyecto:

> ¿Justifica el Vision Transformer su costo computacional como router frente a métodos estadísticos clásicos operando sobre los mismos embeddings?

Este paso **no procesa imágenes ni volúmenes**: trabaja exclusivamente sobre vectores de embedding `z ∈ ℝ^{d_model}` extraídos por el backbone congelado en Fase 1. Los 4 routers compiten en igualdad de condiciones sobre exactamente los mismos datos.

---

## 2. Entrada (Input)

Embeddings pre-computados por Fase 1, almacenados en el directorio `embeddings/{backbone}/`:

| Archivo | Shape | Descripción |
|---|---|---|
| `Z_train.npy` | `[N_train, d_model]` | Embeddings de entrenamiento (CLS tokens congelados) |
| `Z_val.npy` | `[N_val, d_model]` | Embeddings de validación |
| `Z_test.npy` (opcional) | `[N_test, d_model]` | Embeddings de test (si disponibles) |
| `y_train.npy` | `[N_train]` | Labels de dominio (expert_id 0–4) |
| `y_val.npy` | `[N_val]` | Labels de validación |
| `y_test.npy` (opcional) | `[N_test]` | Labels de test |
| `backbone_meta.json` | — | Metadatos: backbone name, d_model, n_train, n_val, n_test, vram_gb |

### 2.1 Parámetros clave

| Parámetro | Valor | Fuente |
|---|---|---|
| **d_model** | 192 (ViT-Tiny default) | `config.py: BACKBONE_CONFIGS` |
| **N_EXPERTS_DOMAIN** | 5 | `config.py` — solo expertos de dominio (0–4), CAE/OOD excluido |
| **N_EXPERTS_TOTAL** | 6 | `config.py` — incluye slot OOD (expert_id=5) |

### 2.2 Validaciones del loader

`embeddings_loader.py` realiza verificaciones de integridad antes de ejecutar el ablation:

1. Archivos obligatorios presentes (`Z_train.npy`, `y_train.npy`, `Z_val.npy`, `y_val.npy`)
2. `backbone_meta.json`: claves requeridas (`backbone`, `d_model`, `n_train`, `n_val`, `n_test`, `vram_gb`) y consistencia con `Z_train.shape[1]`
3. Ausencia de NaN e Inf en todos los arrays
4. Clases en `y_train` == `{0, 1, 2, 3, 4}` (las 5 clases de dominio deben estar representadas)
5. Carga opcional de `Z_test.npy` / `y_test.npy`

---

## 3. Routers evaluados

### 3.1 Router A — Linear + Softmax (baseline DL)

| Campo | Valor |
|---|---|
| **Clase** | `LinearGatingHead(d_model, N_EXPERTS_DOMAIN)` |
| **Arquitectura** | Matriz de pesos `W ∈ ℝ^{d_model × n_experts}` + bias + Softmax |
| **Tipo** | Paramétrico (gradiente) |
| **Optimizador** | Adam, LR=1e-3 |
| **Batch size** | 512 |
| **Épocas** | 50 |
| **GPU** | Sí (usa CUDA si disponible) |

**Loss total:** `CrossEntropy + L_aux` (Switch Transformer auxiliary loss)

```
L_aux = α · N · Σ f_i · P_i
```

- `α = ALPHA_L_AUX = 0.01` — coeficiente de penalización
- `N = N_EXPERTS_DOMAIN = 5` — número de expertos
- `f_i` = fracción real de muestras del batch asignadas al experto `i` (distribución empírica)
- `P_i` = probabilidad media que el router asigna al experto `i` (softmax medio sobre el batch)

L_aux penaliza la concentración de carga: si toda la carga va a un experto, `L_aux ≈ α·N`. Si la carga está perfectamente balanceada, `L_aux = α`.

El modelo retornado tiene sus pesos restaurados al mejor checkpoint de validación (best val accuracy).

### 3.2 Router B — GMM (paramétrico EM)

| Campo | Valor |
|---|---|
| **Clase** | `sklearn.mixture.GaussianMixture` |
| **Componentes** | `n_components = N_EXPERTS_DOMAIN = 5` |
| **Covarianza** | `"full"` (fallback automático a `"diag"` si EM falla) |
| **Max iteraciones** | 200 |
| **Tipo** | Paramétrico (EM sin supervisión) |
| **GPU** | No |

**Mapeo componente→experto:** el GMM se ajusta sin etiquetas (EM puro). Después del ajuste, cada componente se asigna al experto cuyas muestras son mayoría (voto mayoritario). Si dos componentes se asignan al mismo experto y otro queda sin cobertura, se aplica reasignación greedy.

**Predicción:** `predict_proba()` devuelve probabilidades de componente, que se convierten a probabilidades de experto sumando responsabilidades y normalizando.

### 3.3 Router C — Naive Bayes (MLE analítico)

| Campo | Valor |
|---|---|
| **Clase** | `sklearn.naive_bayes.GaussianNB` |
| **Tipo** | Paramétrico (MLE analítico) |
| **Iteraciones** | Ninguna — solución analítica (estimación de medias y varianzas por clase) |
| **GPU** | No |

El router más rápido del ablation. GaussianNB tiene solución cerrada (MLE de medias y varianzas por clase), sin iteraciones de convergencia. Si su accuracy es competitivo con el GMM pero su tiempo de entrenamiento es 100× menor, el tradeoff puede ser favorable.

### 3.4 Router D — kNN-FAISS (no paramétrico)

| Campo | Valor |
|---|---|
| **Índice** | `faiss.IndexFlatIP(d_model)` — inner product (coseno tras normalización L2) |
| **k** | 5 (`KNN_K` en `fase2_config.py`) |
| **Similitud** | Coseno (L2 normalización + inner product) |
| **Tipo** | No paramétrico |
| **GPU** | No |

**Normalización L2:** los embeddings se normalizan con `faiss.normalize_L2()` antes de indexar y buscar. Esto convierte el inner product en similitud coseno, que es la métrica apropiada para embeddings de transformers (la magnitud no tiene semántica clínica).

**Predicción:** votación por mayoría de los k=5 vecinos más cercanos.

**Suavizado de Laplace:** con k=5, las probabilidades de voto sin suavizar son discretas `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`, produciendo entropías artificialmente discretas. `LAPLACE_EPSILON = 0.01` hace las probabilidades continuas sin alterar las predicciones, permitiendo que el umbral OOD sea comparable con los otros routers.

---

## 4. Criterio de selección del ganador

Selección en dos pasos (implementada en `ablation_reporter.py`):

### Paso 1 — Filtro de balance de carga

Descartar routers con `load_balance > LOAD_BALANCE_THRESHOLD` (1.30):

```python
safe_routers = {k: v for k, v in results.items()
                if v["load_balance"] <= LOAD_BALANCE_THRESHOLD}
```

Un router que supera este umbral tiene riesgo de **expert collapse** (concentra la carga en pocos expertos) y penalización del 40% en la evaluación final del proyecto.

### Paso 2 — Selección por accuracy

Entre los routers que pasan el filtro de balance: seleccionar el de mayor `val_accuracy`:

```python
winner = max(safe_routers.items(), key=lambda x: x[1]["acc"])
```

### Fallback

Si **TODOS** los routers fallan el filtro de balance de carga → seleccionar el de menor ratio `load_balance` (con warning explícito en el log):

```python
winner = min(results.items(), key=lambda x: x[1]["load_balance"])
```

---

## 5. Métricas de comparación

| Métrica | Descripción | Cálculo |
|---|---|---|
| **accuracy val** | Routing accuracy sobre el set de validación | `sklearn.metrics.accuracy_score(y_val, preds)` |
| **accuracy test** | Routing accuracy sobre test (si disponible) | `accuracy_score(y_test, test_preds)` |
| **train_time_s** | Tiempo de entrenamiento/ajuste en segundos | `time.time()` delta |
| **latency_ms** | Latencia de inferencia (batch=32, promedio de 10 corridas, excluye warm-up) | `router_metrics.measure_latency()` |
| **params** | Número de parámetros del modelo | Calculado por router |
| **needs_gpu** | Si requiere GPU para inferencia | `True` solo para Linear |
| **load_balance_ratio** | `max(f_i)/min(f_i)` sobre predicciones de val (expertos 0–4) | `router_metrics.check_load_balance()` |
| **entropy_threshold_p95** | Umbral de entropía OOD calibrado (percentil 95 de H(g) sobre val) | `router_metrics.calibrate_entropy_threshold()` |

Adicionalmente, se reporta **accuracy por experto de dominio** para detectar si el router colapsa hacia un subconjunto o ignora sistemáticamente a algún experto.

---

## 6. Hiperparámetros

Todos definidos en `src/pipeline/fase2/fase2_config.py` (fuente de verdad):

| Constante | Valor | Router | Descripción |
|---|---|---|---|
| `ALPHA_L_AUX` | 0.01 | Linear | Coeficiente α de la Auxiliary Loss del Switch Transformer |
| `LINEAR_EPOCHS` | 50 | Linear | Épocas de entrenamiento |
| `LINEAR_LR` | 1e-3 | Linear | Learning rate (Adam) |
| `LINEAR_BATCH_SIZE` | 512 | Linear | Batch size |
| `GMM_COV_TYPE` | `"full"` | GMM | Tipo de covarianza (fallback automático a `"diag"`) |
| `GMM_MAX_ITER` | 200 | GMM | Iteraciones máximas del algoritmo EM |
| `KNN_K` | 5 | kNN-FAISS | Número de vecinos más cercanos |
| `LAPLACE_EPSILON` | 0.01 | kNN-FAISS | Suavizado de Laplace para probabilidades de voto |
| `LOAD_BALANCE_THRESHOLD` | 1.30 | Ablation | Umbral `max(f_i)/min(f_i)` — superar descalifica al router |
| `ENTROPY_PERCENTILE` | 95 | Ablation | Percentil para calibrar umbral OOD sobre val set |

Constantes globales en `src/pipeline/config.py`:

| Constante | Valor | Descripción |
|---|---|---|
| `N_EXPERTS_DOMAIN` | 5 | Expertos de dominio (0–4) — participan en el ablation |
| `N_EXPERTS_TOTAL` | 6 | Total incluyendo slot OOD (expert_id=5) |

---

## 7. Archivos clave

| Archivo | Descripción |
|---|---|
| `src/pipeline/fase2/fase2_pipeline.py` | Orquestador principal: CLI, flujo de ejecución, dry-run. No contiene lógica propia — solo llama funciones |
| `src/pipeline/fase2/fase2_config.py` | Constantes exclusivas de Fase 2 (hiperparámetros, umbrales). Fuente de verdad |
| `src/pipeline/fase2/embeddings_loader.py` | Carga y validación de embeddings de Fase 1. Verificaciones de integridad. Nunca escribe en disco |
| `src/pipeline/fase2/ablation_runner.py` | Ejecución secuencial de los 4 routers. Recopila resultados en dict uniforme. No escribe en disco |
| `src/pipeline/fase2/ablation_reporter.py` | Tabla comparativa, selección del ganador, guardado de modelo y `ablation_results.json` |
| `src/pipeline/fase2/router_metrics.py` | Métricas compartidas: `compute_entropy()`, `per_expert_accuracy()`, `check_load_balance()`, `calibrate_entropy_threshold()`, `measure_latency()` |
| `src/pipeline/fase2/routers/__init__.py` | Re-exporta los 4 routers para import unificado |
| `src/pipeline/fase2/routers/linear.py` | Router A — `LinearGatingHead` + `train_linear_router()`. CrossEntropy + L_aux Switch Transformer |
| `src/pipeline/fase2/routers/gmm.py` | Router B — GaussianMixture + mapeo componente→experto por voto mayoritario + reasignación greedy |
| `src/pipeline/fase2/routers/naive_bayes.py` | Router C — GaussianNB (MLE analítico). El más rápido |
| `src/pipeline/fase2/routers/knn.py` | Router D — FAISS IndexFlatIP + votación k-NN con suavizado de Laplace |
| `src/pipeline/config.py` | Constantes globales compartidas: `N_EXPERTS_DOMAIN`, `N_EXPERTS_TOTAL`, `EXPERT_IDS`, `EXPERT_NAMES`, `BACKBONE_CONFIGS` |
| `src/pipeline/logging_utils.py` | Configuración de logging (setup_logging) |

---

## 8. Output artifacts

Todos los artefactos se guardan en el directorio de embeddings (`{embeddings_dir}/`):

| Artefacto | Formato | Descripción |
|---|---|---|
| `ablation_results.json` | JSON | Comparación completa: resultados de los 4 routers + nombre del ganador + path del modelo + d_model + backbone + threshold OOD + expert_notes |
| `best_router_{type}.{ext}` | `.pt` / `.joblib` / `.faiss` | Modelo ganador serializado. Linear→`.pt` (state_dict), GMM/NB→`.joblib`, kNN→`.faiss` |
| `fase2.log` | Texto | Log de ejecución completo (tabla comparativa, métricas por experto, selección del ganador) |

### 8.1 Estructura de `ablation_results.json`

```json
{
  "results": {
    "Linear":     {"acc": ..., "acc_test": ..., "train_time_s": ..., "latency_ms": ..., "params": ..., "needs_gpu": ..., "load_balance": ..., "entropy_thresh": ...},
    "GMM":        {...},
    "NaiveBayes": {...},
    "kNN-FAISS":  {...}
  },
  "winner": "Linear",
  "best_router_path": "/path/to/best_router_linear.pt",
  "n_experts_domain": 5,
  "n_experts_total": 6,
  "d_model": 192,
  "backbone": "vit_tiny_patch16_224",
  "load_balance_threshold": 1.30,
  "entropy_threshold_winner": 0.1234,
  "note_ood": "Experto 5 (OOD) se activa en inferencia cuando H(g) >= ...",
  "expert_notes_fase2": {...}
}
```

---

## 9. Cómo ejecutar

### 9.1 Dry-run (datos sintéticos, sin guardar artefactos)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/fase2_pipeline.py --dry-run
```

El modo `--dry-run` ejecuta:

1. Genera embeddings sintéticos: `Z_train=(200, 192)`, `Z_val=(40, 192)` (40 muestras por clase en train, 8 por clase en val)
2. Ejecuta los 4 routers con parámetros reducidos (Linear: 2 épocas, GMM: 5 iteraciones)
3. Imprime la tabla comparativa completa
4. Selecciona el ganador (con el mismo criterio que la ejecución real)
5. **No guarda artefactos a disco** (ni `ablation_results.json` ni modelo ganador)
6. Exit code 0 si todo pasa correctamente

### 9.2 Ejecución real (requiere embeddings de Fase 1)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/fase2_pipeline.py \
    --embeddings ./embeddings/vit_tiny_patch16_224
```

### 9.3 Con overrides de hiperparámetros

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/fase2_pipeline.py \
    --embeddings ./embeddings/vit_tiny_patch16_224 \
    --epochs 50 \
    --knn_k 5 \
    --l_aux_alpha 0.01
```

### 9.4 Argumentos CLI

| Argumento | Default | Descripción |
|---|---|---|
| `--embeddings` | `./embeddings` | Directorio con artefactos de Fase 1. También destino de los artefactos de salida |
| `--epochs` | 50 (`LINEAR_EPOCHS`) | Épocas para el router Linear |
| `--knn_k` | 5 (`KNN_K`) | Número de vecinos para kNN-FAISS |
| `--l_aux_alpha` | 0.01 (`ALPHA_L_AUX`) | Coeficiente α de L_aux (Switch Transformer) |
| `--dry-run` | `False` | Ejecuta con embeddings sintéticos, sin guardar artefactos |

---

## 10. Dry-run verificado

| Verificación | Resultado |
|---|---|
| Importaciones de todos los módulos | ✅ OK |
| Embeddings sintéticos generados | ✅ OK — Z_train=(200, 192), Z_val=(40, 192) |
| Router Linear (2 épocas) | ✅ OK — ejecutó correctamente |
| Router GMM (5 iter) | ✅ OK — ejecutó correctamente |
| Router NaiveBayes | ✅ OK — ejecutó correctamente |
| Router kNN-FAISS | ✅ OK — ejecutó correctamente |
| Tabla comparativa impresa | ✅ OK |
| Selección del ganador | ✅ OK |
| No se guardaron artefactos a disco | ✅ Confirmado (dry-run salta guardado) |
| Dry-run exit code | ✅ 0 |
| Ejecución real | ⏳ Pendiente (requiere embeddings de Fase 1) |

---

## 11. Restricciones aplicadas

| Restricción | Detalle |
|---|---|
| **No datos de imagen** | Este paso trabaja exclusivamente sobre vectores de embedding — no accede a imágenes PNG, JPEG ni NIfTI |
| **Sin pesos preentrenados** | Los embeddings provienen de backbones entrenados from scratch en Fase 1 (sin ImageNet ni HuggingFace) |
| **FP32 implícito** | Los routers CPU-based (GMM, NaiveBayes, kNN) operan en FP32. El Linear usa FP32 también (embeddings pequeños, no requiere FP16) |
| **Sin augmentaciones de oclusión** | No aplica — no hay procesamiento de imagen en este paso |
| **Balance de carga obligatorio** | `max(f_i)/min(f_i) ≤ 1.30` — superar descalifica al router del ranking. Penalización del 40% si se viola en producción |
| **expert_id=5 excluido** | El CAE/OOD no participa en el ablation ni en el cálculo de load balance. Solo los expertos 0–4 |
| **Dry-run no-save guarantee** | En modo `--dry-run`, el pipeline no escribe `ablation_results.json` ni guarda ningún modelo a disco |
| **Embeddings loader read-only** | `embeddings_loader.py` nunca escribe en disco — es consumidor puro de artefactos de Fase 1 |
| **ablation_runner no-write** | `ablation_runner.py` no escribe ningún archivo — esa responsabilidad es exclusiva de `ablation_reporter.py` |

---

## 12. Notas

- **Step 6 es extensión:** este paso fue añadido como extensión al roadmap original de `proyecto_moe.md`. La guía define el ablation study como parte de la evaluación, pero la implementación modular en pipeline separado es decisión de diseño del proyecto.
- **Requiere Fase 1 completada:** la ejecución real necesita embeddings de Fase 1 (`Z_train.npy`, `Z_val.npy`, `backbone_meta.json`). A la fecha (2026-04-05), Fase 1 training está completada (Paso 4.1) pero la extracción de embeddings (Paso 4.2) está pendiente.
- **expert_id=5 excluido:** el CAE/OOD (expert_id=5) no participa en el routing competition. `N_EXPERTS_DOMAIN=5` (expertos 0–4), no 6. El slot OOD se activa por entropía, no por `argmax(g)`.
- **Consumido por Fase 3:** el router ganador y `ablation_results.json` son consumidos por Fase 3 (inferencia pipeline). En Fase 3 real, la cabeza de gating se expande a `LinearGatingHead(d_model, N_EXPERTS_TOTAL=6)` para incluir el slot OOD, que se entrena con `L_error` (no con labels de dominio).
- **Diseño modular:** cada módulo tiene responsabilidad única. Añadir un quinto router solo requiere modificar `ablation_runner.py` y `routers/__init__.py` — el orquestador y el reportero no necesitan cambios.
- **Path setup:** `fase2_pipeline.py` añade 3 directorios a `sys.path` para permitir ejecución como script directo (`python src/pipeline/fase2/fase2_pipeline.py`). Los imports dentro del paquete son absolutos (sin `..`).

---

*Documento generado el 2026-04-05. Fuentes: `src/pipeline/fase2/fase2_config.py`, `fase2_pipeline.py`, `ablation_runner.py`, `ablation_reporter.py`, `embeddings_loader.py`, `router_metrics.py`, `routers/linear.py`, `routers/gmm.py`, `routers/naive_bayes.py`, `routers/knn.py`, `src/pipeline/config.py`.*

# Paso 9 — Prueba con datos reales del sistema MoE

| Campo | Valor |
|---|---|
| **Fecha** | 2026-04-05 |
| **Alcance** | Paso 9 — Evaluación end-to-end del sistema MoE sobre test sets reales. Inferencia sin `expert_id` (router decide), métricas por experto, detección OOD, Grad-CAM. Implementado en `src/pipeline/fase6/`. Consume el checkpoint `moe_final.pt` de Fase 5 |
| **Estado general** | ✅ **Implementado — dry-run verificado** |

---

## 1. Descripción

El Paso 9 es la primera evaluación completa del sistema MoE en modo **inferencia real**: el router decide qué experto procesa cada imagen, sin acceso al `expert_id` ground truth. A diferencia del entrenamiento (Paso 8), donde `forward(x, expert_id)` conoce el experto correcto, aquí el sistema opera como lo haría en producción.

El paso evalúa:

- **Calidad de clasificación:** F1 Macro por experto (2D y 3D) sobre los test sets oficiales de cada dataset.
- **Calidad del routing:** Routing Accuracy — fracción de muestras que el router envía al experto correcto.
- **Balance de carga:** ratio `max(f_i)/min(f_i)` entre los 5 expertos de dominio.
- **Detección OOD:** AUROC del mecanismo de entropía como detector de muestras fuera de distribución.
- **Explicabilidad:** Grad-CAM / Attention Rollout sobre muestras representativas.

Produce un reporte de métricas, figuras de evaluación (confusion matrices, curvas ROC) y mapas de calor Grad-CAM en `results/paso9/`.

---

## 2. Archivos implementados

| Archivo | Descripción | Función principal |
|---|---|---|
| `src/pipeline/fase6/fase6_config.py` | Constantes de Paso 9: umbrales de métricas, rutas de checkpoints, parámetros de evaluación | Fuente de verdad para umbrales F1, AUROC, load balance |
| `src/pipeline/fase6/inference_engine.py` | Forward de inferencia del MoE (sin `expert_id`) | `InferenceEngine.forward(x)` — backbone→router→entropy→experto |
| `src/pipeline/fase6/test_evaluator.py` | Evaluador batch por experto | `TestEvaluator.evaluate_expert()`, `compute_load_balance()`, `compute_summary()` |
| `src/pipeline/fase6/gradcam_heatmap.py` | Generación de mapas de calor Grad-CAM / Attention Rollout | `generate_gradcam_samples()` |
| `src/pipeline/fase6/ood_detector.py` | Calibración de umbral de entropía + cálculo AUROC OOD | `OODDetector.calibrate_threshold()`, `compute_ood_auroc()` |
| `src/pipeline/fase6/fase9_test_real.py` | Orquestador principal del Paso 9 (CLI) | `main()` — ensambla MoE, crea InferenceEngine, ejecuta evaluación |

---

## 3. Pipeline de inferencia

La diferencia fundamental entre entrenamiento e inferencia es que en inferencia **no se conoce el `expert_id`**. El router decide qué experto procesa cada muestra.

### 3.1 Forward de entrenamiento (Paso 8)

```
forward(x, expert_id) → output del experto expert_id
```

El `expert_id` es ground truth — supervisado, conocido de antemano por el DataLoader mixto.

### 3.2 Forward de inferencia (Paso 9)

```
1. backbone(x) → z              # embedding del CLS token, z ∈ ℝ^d_model
2. router(z)   → gates           # softmax(W·z + b) ∈ ℝ^N_EXPERTS_DOMAIN
3. H = -Σ gates · log(gates + ε) # entropía de Shannon por muestra
4. if H > θ:                     # θ = umbral calibrado (percentil 95 sobre val set)
       expert_id = 5 (CAE)       # OOD → experto CAE
   else:
       expert_id = argmax(gates)  # experto de dominio con mayor probabilidad
5. output = experts[expert_id](x) # forward del experto seleccionado
6. return {logits, expert_ids, gates, entropy, is_ood}
```

### 3.3 Per-sample routing loop

La inferencia usa un **loop por muestra** (no batching vectorizado) en `InferenceEngine.forward()`:

```python
for i in range(batch_size):
    xi = x[i:i+1]           # [1, C, H, W] o [1, C, D, H, W]
    if is_ood[i]:
        logit_i = experts[CAE_EXPERT_IDX](xi)
    else:
        eid = int(gates[i].argmax().item())
        logit_i = experts[eid](xi)
    logits_list.append(logit_i)
```

Este loop es **intencional** — no se puede hacer batching vectorizado porque:

- Los expertos tienen dimensiones de salida heterogéneas (14 clases vs. 9 vs. 3 vs. 2 vs. reconstrucción).
- Muestras distintas del batch pueden ir a expertos distintos (2D vs. 3D).
- `torch.stack()` requiere tensores de la misma shape, imposible con salidas heterogéneas.

El resultado es una **lista de tensores**, no un tensor apilado.

---

## 4. Métricas objetivo

Umbrales extraídos de `fase6_config.py` (fuente de verdad), derivados de `proyecto_moe.md` §10 y §13:

| Métrica | Umbral Full | Umbral Aceptable | Aplica a | Constante en `fase6_config.py` |
|---|---|---|---|---|
| **F1 Macro — Expertos 2D** | > 0.72 | > 0.65 | Chest, ISIC, OA-Knee | `F1_FULL_2D=0.72`, `F1_THRESHOLD_2D=0.65` |
| **F1 Macro — Expertos 3D** | > 0.65 | > 0.58 | LUNA16, Páncreas | `F1_FULL_3D=0.65`, `F1_THRESHOLD_3D=0.58` |
| **Routing Accuracy** | > 0.80 | — | Router sobre test set completo | `ROUTING_ACCURACY_MIN=0.80` |
| **OOD AUROC** | > 0.80 | — | Entropía como detector OOD | `OOD_AUROC_MIN=0.80` |
| **Load Balance** | < 1.30 | — | `max(f_i)/min(f_i)` expertos 0–4 | `LOAD_BALANCE_MAX_RATIO=1.30` |

### 4.1 Configuración de expertos evaluados

| Expert ID | Dataset | `num_classes` | `is_multilabel` | Loss de referencia |
|---|---|---|---|---|
| 0 | NIH ChestXray14 | 14 | Sí | BCEWithLogitsLoss |
| 1 | ISIC 2019 | 9 | No | CrossEntropyLoss |
| 2 | OA Knee | 3 | No | CrossEntropyLoss |
| 3 | LUNA16 | 2 | No | FocalLoss |
| 4 | Páncreas | 2 | No | FocalLoss |

---

## 5. Artefactos producidos

| Artefacto | Tipo | Ruta | Consumidor |
|---|---|---|---|
| Métricas por experto (F1, AUC, routing acc) | JSON / log | `results/paso9/` | Reporte técnico, Fase 6 dashboard |
| Figuras de evaluación (confusion matrix, ROC) | PNG | `results/paso9/figures/` | Reporte técnico |
| Muestras Grad-CAM / Attention Rollout | PNG | `results/paso9/figures/` | Dashboard (Funcionalidad 4) |
| Reporte OOD (AUROC, umbral calibrado) | JSON / log | `results/paso9/` | Dashboard (Funcionalidad 8) |
| Reporte de load balance | JSON / log | `results/paso9/` | Verificación restricción −40% |

En modo `--dry-run` los directorios se crean pero los artefactos contienen valores sintéticos (pesos aleatorios → métricas meaningless).

---

## 6. Comandos de uso

### 6.1 Dry-run (verifica setup sin datos reales)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase6/fase9_test_real.py --dry-run
```

### 6.2 Evaluación completa (requiere checkpoint + datos)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase6/fase9_test_real.py \
    --checkpoint checkpoints/fase5/moe_final.pt
```

### 6.3 Evaluación de expertos específicos

```bash
# Solo expertos 2D
PYENV_VERSION=3.12.3 python src/pipeline/fase6/fase9_test_real.py --experts 0,1,2

# Solo expertos 3D
PYENV_VERSION=3.12.3 python src/pipeline/fase6/fase9_test_real.py --experts 3,4
```

### 6.4 Sin Grad-CAM ni OOD (solo métricas de clasificación)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase6/fase9_test_real.py --no-gradcam --no-ood
```

### 6.5 Argumentos CLI

| Argumento | Default | Descripción |
|---|---|---|
| `--dry-run` | `False` | Valida imports, configuración y estructura sin ejecutar inferencia real |
| `--checkpoint` | `checkpoints/fase5/moe_final.pt` | Ruta al checkpoint del MoESystem entrenado |
| `--experts` | `0,1,2,3,4` | Expertos a evaluar (comma-separated) |
| `--no-gradcam` | `False` | Deshabilita generación de Grad-CAM |
| `--no-ood` | `False` | Deshabilita evaluación OOD (calibración + AUROC) |
| `--device` | auto (`cuda` si disponible) | Device de ejecución: `cuda` / `cpu` |

---

## 7. Prerrequisitos

Los siguientes artefactos deben existir antes de ejecutar el Paso 9 en modo producción (no aplica a `--dry-run`):

| # | Artefacto | Generado en | Ruta |
|---|---|---|---|
| 1 | Embeddings ViT-Tiny | Paso 4.2 | `embeddings/vit_tiny_patch16_224/` |
| 2 | Expert 0 checkpoint (ConvNeXt-Tiny) | Paso 5.1 | `checkpoints/expert_00_convnext_tiny/expert1_best.pt` |
| 3 | Expert 1 checkpoint (EfficientNet-B3) | Paso 5.2 | `checkpoints/expert_01_efficientnet_b3/expert2_best.pt` |
| 4 | Expert 2 checkpoint (VGG16-BN) | Paso 5.3 | `checkpoints/expert_02_vgg16_bn/expert_oa_best.pt` |
| 5 | Expert 3 checkpoint (MC3-18) | Paso 5.4 | `checkpoints/expert_03_vivit_tiny/expert3_best.pt` |
| 6 | Expert 4 checkpoint (Swin3D-Tiny) | Paso 5.5 | `checkpoints/expert_04_swin3d_tiny/expert4_best.pt` |
| 7 | Expert 5 checkpoint (CAE) | Paso 5.6 | `checkpoints/expert_05_cae/cae_best.pt` |
| 8 | MoE final checkpoint (Fase 5) | Paso 8 | `checkpoints/fase5/moe_final.pt` |
| 9 | Umbral de entropía calibrado | Fase 2 (ablation) | `checkpoints/entropy_threshold.pkl` |
| 10 | Test sets de los 5 datasets | Pasos 1–3 | `datasets/*/` + splits |

En modo `--dry-run`:
- Los checkpoints no son necesarios: el MoESystem se instancia con pesos aleatorios.
- Los DataLoaders no se construyen: se devuelve un dict vacío.
- El umbral de entropía usa fallback: `log(N_EXPERTS_DOMAIN) / 2 ≈ 0.8047`.

---

## 8. Incongruencias detectadas y resolución

Se documentaron 5 incongruencias durante la implementación del Paso 9, todas resueltas:

| # | ID | Aspecto | Problema | Resolución adoptada |
|---|---|---|---|---|
| 1 | INC-01 | **Forward de inferencia** | `moe_model.py` (entrenamiento) usa `forward(x, expert_id)` con `expert_id` conocido. En inferencia, `expert_id` no existe | Creado `inference_engine.py` con forward autónomo: backbone→router→entropy→argmax. Brecha crítica cerrada |
| 2 | INC-02 | **Batching heterogéneo** | Los expertos tienen dimensiones de salida distintas (14, 9, 3, 2, 2, reconstrucción). `torch.stack()` falla | Loop per-sample intencional en `InferenceEngine.forward()`. Output es `list[Tensor]`, no tensor apilado |
| 3 | INC-03 | **Umbral de entropía** | `entropy_threshold.pkl` no existe hasta que Fase 2 calibra sobre val set real | Fallback a `log(N)/2` con warning. Funcional para dry-run, calibración real pendiente |
| 4 | INC-04 | **Import de `linear.py`** | `LinearGatingHead` vive en `src/pipeline/fase2/routers/linear.py` pero el import bare `from linear import ...` requiere `sys.path` explícito | `fase9_test_real.py` inserta `_ROUTERS_DIR` en `sys.path` antes del import |
| 5 | INC-05 | **Backbone `None`** | `MoESystem(backbone=None)` en dry-run — el backbone no existe como módulo separado en la implementación actual | `build_moe_system()` pasa `backbone=None`. En producción, el backbone se carga desde checkpoint de Fase 1 |

---

## 9. Dependencias

| Paso previo | Artefacto consumido | Bloqueante para Paso 9 |
|---|---|---|
| Paso 1 (Descargar datos) | Datasets en `datasets/` | Sí (modo producción) |
| Paso 2 (Extraer archivos) | Datos descomprimidos | Sí (modo producción) |
| Paso 3 (Preparar datos) | Splits 80/10/10 | Sí (modo producción) |
| Paso 4.1 (Entrenar backbones) | `checkpoints/backbone_0X_*/backbone.pth` | Sí (para backbone del MoE) |
| Paso 4.2 (Generar embeddings) | `embeddings/vit_tiny_patch16_224/` | Sí (para router) |
| Pasos 5.1–5.6 (Entrenar expertos) | Checkpoints de los 6 expertos | Sí (para MoESystem) |
| Paso 6 (Ablation study) | `ablation_results.json`, `entropy_threshold.pkl` | Sí (para umbral OOD) |
| Paso 8 (Fine-tuning por etapas) | `checkpoints/fase5/moe_final.pt` | Sí (checkpoint principal) |

Todos los pasos previos (1–8) deben haberse completado exitosamente para ejecutar el Paso 9 en modo producción. En `--dry-run`, ninguno es bloqueante.

---

## 10. Notas de implementación

### 10.1 Per-sample routing loop

El loop por muestra en `InferenceEngine.forward()` es una decisión de diseño deliberada, no una limitación temporal. En un sistema MoE con expertos heterogéneos (2D/3D, distintas `num_classes`), no es posible:

- Agrupar muestras del batch por experto y hacer forward vectorizado, porque el router decide per-sample y las dimensiones de salida difieren.
- Usar `torch.stack()` sobre los outputs, porque los tensores tienen shapes distintos (14 vs. 9 vs. 3 vs. 2 clases).

El costo es O(B) forwards individuales en lugar de O(1) forward batched. Para evaluación (sin gradientes, `@torch.no_grad()`), el overhead es aceptable.

### 10.2 Fallback del umbral de entropía

Si `checkpoints/entropy_threshold.pkl` no existe (caso normal en dry-run), `InferenceEngine._load_threshold()` usa:

```python
entropy_threshold = math.log(N_EXPERTS_DOMAIN) / 2.0  # ≈ 0.8047
```

Este valor es `log(5)/2`, mitad de la entropía máxima para 5 clases. Es un fallback razonable pero debe reemplazarse con el percentil 95 calibrado sobre el validation set real (`OOD_ENTROPY_PERCENTILE = 95` en `fase6_config.py`).

### 10.3 Backbone `None` en dry-run

El `MoESystem` se instancia con `backbone=None` tanto en dry-run como potencialmente en producción temprana. Esto implica que `self.moe.backbone(x)` fallaria si se invoca directamente. En producción, el backbone se carga desde el checkpoint de Fase 1 y se integra al MoESystem antes de la inferencia.

### 10.4 Expert 5 CAE: FP32 obligatorio

Al igual que en el entrenamiento (Paso 8), el Expert 5 (CAE) **no puede usar FP16** durante la inferencia. La reconstrucción requiere precisión completa para evitar artefactos numéricos.

### 10.5 Grad-CAM en dry-run

En modo `--dry-run`, `generate_gradcam_samples()` recibe tensores sintéticos (`torch.randn`) con las dimensiones correctas por tipo de experto:

- Expertos 2D (0, 1, 2): `[2, 3, 224, 224]`
- Expertos 3D (3, 4): `[2, 1, 64, 64, 64]`

Los mapas de calor resultantes son meaningless (pesos aleatorios) pero verifican que el pipeline de generación funciona correctamente.

### 10.6 Constantes de evaluación

Valores de `fase6_config.py` relevantes para la evaluación:

| Constante | Valor | Uso |
|---|---|---|
| `EVAL_BATCH_SIZE` | 32 | Batch size para DataLoaders de test (sin gradientes, permite batches mayores) |
| `EVAL_NUM_WORKERS` | 4 | Workers del DataLoader de evaluación |
| `OOD_ENTROPY_PERCENTILE` | 95 | Percentil para calibrar el umbral de entropía sobre val set |
| `N_EXPERTS_DOMAIN` | 5 | Expertos de dominio (0–4) |
| `N_EXPERTS_TOTAL` | 6 | Total expertos (dominio + CAE) |
| `CAE_EXPERT_IDX` | 5 | Indice del experto CAE/OOD |

---

## 11. Dry-run verificado

| Verificación | Resultado |
|---|---|
| Importaciones de todos los módulos (fase6_config, inference_engine, test_evaluator, ood_detector, gradcam_heatmap) | ✅ OK |
| MoESystem instanciado (6 expertos + 1 router, backbone=None) | ✅ OK |
| InferenceEngine creado con umbral fallback | ✅ OK |
| DataLoaders skipped (dry-run) | ✅ OK |
| Evaluación batch por experto (dry-run mode) | ✅ OK |
| Load balance computado | ✅ OK |
| OOD detector instanciado | ✅ OK |
| Grad-CAM con tensores sintéticos (2D y 3D) | ✅ OK |
| Directorios de salida creados (`results/paso9/`, `results/paso9/figures/`) | ✅ OK |
| Dry-run exit code | ✅ 0 |
| Evaluación con datos reales | ⏳ Pendiente (requiere checkpoint Paso 8 + test sets de Pasos 1–3) |

---

## 12. Restricciones aplicadas

| Restriccion | Detalle |
|---|---|
| **Sin metadatos de entrada** | Solo pixeles. Deteccion 2D/3D por rank del tensor. Violacion: −20% nota |
| **Balance de carga** | `max(f_i)/min(f_i) ≤ 1.30` sobre expertos 0–4. Violacion: −40% nota |
| **Expert 5 CAE: FP32 obligatorio** | Sin autocast — precision completa para reconstruccion |
| **Expert 5 excluido de load balance** | Solo expertos 0–4 participan en el calculo de `max(f_i)/min(f_i)` |
| **Per-sample routing** | Loop obligatorio por outputs heterogeneos — no se puede batching vectorizado |
| **Umbral de entropia calibrado** | Percentil 95 sobre val set. Fallback `log(5)/2` si no existe |

---

*Documento generado el 2026-04-05. Fuentes: `src/pipeline/fase6/fase6_config.py`, `inference_engine.py`, `fase9_test_real.py`, `test_evaluator.py`, `ood_detector.py`, `gradcam_heatmap.py`, `proyecto_moe.md` §10 y §13, `arquitectura_documentacion.md`.*

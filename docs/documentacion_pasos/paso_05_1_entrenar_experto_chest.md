# Paso 5.1 — Entrenar Experto 1 (NIH ChestXray14 / ConvNeXt-Tiny)

| Campo | Valor |
|---|---|
| **Fecha** | 2026-04-14 |
| **Alcance** | Entrenamiento del Experto 1 (clasificación multilabel de 14 patologías torácicas) con ConvNeXt-Tiny pretrained (timm) mediante estrategia LP-FT. Genera checkpoint consumido por Fase 4 (ablation) y Fase 5 (fine-tuning global) |
| **Estado general** | ✅ **Script creado y verificado con dry-run** — datos pendientes de disponibilidad para entrenamiento real |

---

## 1. Objetivo

Entrenar un modelo **ConvNeXt-Tiny pretrained** (timm: `convnext_tiny.in12k_ft_in1k`) para clasificación **multilabel** de las 14 patologías del dataset NIH ChestXray14, usando la estrategia **LP-FT (Linear Probing → Fine-Tuning)**.

Este experto es el Experto 0 del sistema MoE, responsable del dominio de radiografía de tórax. El backbone se descarga pretrained de ImageNet-12K fine-tuned en ImageNet-1K vía la librería `timm`.

---

## 2. Archivos del Experto 1

| Archivo | Descripción |
|---|---|
| `src/pipeline/fase2/expert1_config.py` | Constantes de hiperparámetros (prefijo `EXPERT1_`). Fuente de verdad para LP/FT LRs, épocas, WD, dropout, batch, accumulation, patience, etc. |
| `src/pipeline/fase2/models/expert1_convnext.py` | Clase `Expert1ConvNeXt` (alias `Expert1ConvNeXtTiny`) — ConvNeXt-Tiny via `timm.create_model(pretrained=True)` con `domain_conv` residual y métodos `freeze_backbone()` / `unfreeze_backbone()` |
| `src/pipeline/datasets/chest.py` | Clase `ChestXray14Dataset` — preprocesado offline con cv2+CLAHE+multistage_resize, augmentación online con Albumentations |
| `src/pipeline/fase2/dataloader_expert1.py` | Función `build_expert1_dataloaders()` — retorna dict con `train`, `val`, `test`, `test_flip` (TTA) y `pos_weight`. 3 pipelines Albumentations (train/val/flip-TTA) |
| `src/pipeline/fase2/train_expert1.py` | Script de entrenamiento LP-FT: 2 fases secuenciales, early stopping por `val_macro_auc`, evaluación final con TTA, soporte `--dry-run` y `--data-root` |

---

## 3. Estrategia de entrenamiento: LP-FT

El entrenamiento sigue la estrategia **LP-FT (Linear Probing → Fine-Tuning)** en 2 fases secuenciales:

| Fase | Nombre | Épocas | LR | Componentes entrenables | Scheduler | Early Stopping |
|---|---|---|---|---|---|---|
| **LP** | Linear Probing | 5 | 1×10⁻³ | Solo `head` + `domain_conv` (backbone congelado) | Ninguno | No |
| **FT** | Fine-Tuning | 30 | 1×10⁻⁴ | Todo el modelo (backbone descongelado) | CosineAnnealingLR(T_max=30) | Sí (patience=10, monitor=`val_macro_auc`) |

**Total máximo de épocas:** 35 (5 LP + 30 FT), con posibilidad de terminar antes por early stopping en la fase FT.

**Lógica LP-FT:**
1. `model.freeze_backbone()` → solo `head` + `domain_conv` tienen `requires_grad=True`
2. Entrenar 5 épocas con AdamW(lr=1e-3) sin scheduler
3. `model.unfreeze_backbone()` → todos los parámetros son entrenables
4. Entrenar hasta 30 épocas con AdamW(lr=1e-4) + CosineAnnealingLR + early stopping

---

## 4. Arquitectura del modelo

```
imgs [B, 3, 224, 224]
    │
    ▼
timm ConvNeXt-Tiny (pretrained, num_classes=0)
    │ backbone.forward_features(x)
    │   → [B, 768, H, W]
    │
    ├── domain_conv (adapter residual):
    │   Conv2d(768, 768, 3, pad=1) → BN → GELU → Conv2d(768, 768, 1) → BN
    │   feat = feat + domain_conv(feat)  ← conexión residual
    │
    ├── AdaptiveAvgPool2d(1) → [B, 768, 1, 1]
    │
    └── head:
        Flatten → Dropout(0.3) → Linear(768, 14)
    │
    ▼
logits [B, 14]  ← BCEWithLogitsLoss(pos_weight)
```

| Campo | Valor |
|---|---|
| **Clase** | `Expert1ConvNeXt` (alias: `Expert1ConvNeXtTiny`) |
| **Backbone** | `timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True, num_classes=0)` |
| **Pretraining** | ImageNet-12K → fine-tuned ImageNet-1K |
| **Parámetros totales** | ~28M |
| **Entrada** | `[B, 3, 224, 224]` — radiografía de tórax RGB, float32 |
| **Salida** | `[B, 14]` — logits crudos (antes de sigmoid) |
| **Adapter** | `domain_conv` — bloque convolucional residual post-backbone (adapta features al dominio médico) |
| **Cabeza FC** | `Flatten → Dropout(0.3) → Linear(768, 14)` |
| **Normalización** | `model_mean` y `model_std` resueltos programáticamente via `timm.data.resolve_data_config()` |

**Métodos LP-FT:**
- `freeze_backbone()` — congela todos los parámetros del backbone (`requires_grad=False`)
- `unfreeze_backbone()` — descongela todos los parámetros del backbone
- `count_parameters()` — cuenta parámetros entrenables (con `requires_grad=True`)
- `count_all_parameters()` — cuenta todos los parámetros (entrenables + congelados)

---

## 5. Configuración de entrenamiento

| Hiperparámetro | Valor | Constante en `expert1_config.py` |
|---|---|---|
| **Backbone** | `convnext_tiny.in12k_ft_in1k` | `EXPERT1_BACKBONE` |
| **Optimizador** | AdamW (uno por fase) | — |
| **LR fase LP** | 1×10⁻³ | `EXPERT1_LP_LR` |
| **LR fase FT** | 1×10⁻⁴ | `EXPERT1_FT_LR` |
| **Weight decay** | 0.05 | `EXPERT1_WEIGHT_DECAY` |
| **Dropout FC** | 0.3 | `EXPERT1_DROPOUT_FC` |
| **Batch size (real)** | 32 | `EXPERT1_BATCH_SIZE` |
| **Accumulation steps** | 4 | `EXPERT1_ACCUMULATION_STEPS` |
| **Batch efectivo** | 128 (32 × 4) | — |
| **Mixed precision (FP16)** | Sí | `EXPERT1_FP16` |
| **Épocas LP** | 5 | `EXPERT1_LP_EPOCHS` |
| **Épocas FT** | 30 | `EXPERT1_FT_EPOCHS` |
| **Scheduler LP** | Ninguno | — |
| **Scheduler FT** | CosineAnnealingLR(T_max=30) | — |
| **Early stopping patience** | 10 (solo FT) | `EXPERT1_EARLY_STOPPING_PATIENCE` |
| **Early stopping monitor** | `val_macro_auc` (maximizar) | Clase `EarlyStoppingAUC` |
| **Early stopping min_delta** | 0.001 | — |
| **Imagen** | 224×224 | `EXPERT1_IMG_SIZE` |
| **Num clases** | 14 | `EXPERT1_NUM_CLASSES` |
| **Num workers** | 4 (0 en dry-run) | `EXPERT1_NUM_WORKERS` |
| **Semilla** | 42 | `_SEED` |

---

## 6. Pipeline de preprocesado

El preprocesado se divide en 2 etapas: **offline** (al instanciar el dataset, cacheado en RAM) y **online** (en cada `__getitem__`, por batch).

### 6.1 Pipeline offline (`_preload` en `chest.py`)

| Paso | Técnica | Detalle |
|---|---|---|
| 1 | `cv2.imread(grayscale)` | Carga imagen como grayscale 8-bit |
| 2 | **CLAHE** | `clipLimit=2.0, tileGridSize=(8,8)` — realza contraste en campos pulmonares a resolución original |
| 3 | **multistage_resize(224)** | Halvings iterativos con `INTER_AREA` (nunca factor >2× por paso) hasta target, evita aliasing |
| 4 | Cache en RAM | Array `uint8 (224×224)` grayscale en `self._cache[idx]` |

**RAM estimada:** `N_imgs × 224 × 224 bytes` (e.g., ~86K train ≈ ~4.1 GB uint8 gray).

### 6.2 Pipeline online (`__getitem__` en `chest.py` + transforms en `dataloader_expert1.py`)

| Paso | Técnica | Detalle |
|---|---|---|
| 1 | `cv2.cvtColor(GRAY2RGB)` | Grayscale → 3 canales RGB `(224, 224, 3) uint8` |
| 2 | **Albumentations transform** | Pipeline completo (varía por split, ver tabla abajo) |

### 6.3 Transforms Albumentations (3 pipelines)

**Train (con augmentation):**

| Augmentación | Parámetros |
|---|---|
| `HorizontalFlip` | `p=0.5` |
| `RandomBrightnessContrast` | `brightness_limit=0.1, contrast_limit=0.1, p=0.5` |
| `RandomGamma` | `gamma_limit=(85, 115), p=0.5` |
| `GaussNoise` | `std_range=(0.01, 0.02), p=0.1` |
| `Normalize` | `mean=model_mean, std=model_std` (del backbone timm) |
| `ToTensorV2` | HWC uint8 → CHW float32 |

**Val / Test (sin augmentation):**

| Transformación | Parámetros |
|---|---|
| `Normalize` | `mean=model_mean, std=model_std` |
| `ToTensorV2` | HWC uint8 → CHW float32 |

**Test-Flip (TTA — HorizontalFlip determinista):**

| Transformación | Parámetros |
|---|---|
| `HorizontalFlip` | `p=1.0` (siempre aplica) |
| `Normalize` | `mean=model_mean, std=model_std` |
| `ToTensorV2` | HWC uint8 → CHW float32 |

> **Nota:** `model_mean` y `model_std` se obtienen del backbone pretrained via `timm.data.resolve_data_config()`. No se hardcodean estadísticas de ImageNet.

---

## 7. DataLoaders

La función `build_expert1_dataloaders()` retorna un dict con 5 claves:

| Clave | Tipo | Descripción |
|---|---|---|
| `"train"` | `DataLoader` | `shuffle=True, drop_last=True`, transform con augmentation |
| `"val"` | `DataLoader` | `shuffle=False, drop_last=False`, solo normalización |
| `"test"` | `DataLoader` | `shuffle=False, drop_last=False`, solo normalización |
| `"test_flip"` | `DataLoader` | `shuffle=False, drop_last=False`, HorizontalFlip(p=1.0) para TTA |
| `"pos_weight"` | `Tensor[14]` | `n_neg / n_pos` por clase, para `BCEWithLogitsLoss` |

Configuración común de DataLoaders:
- `pin_memory=True`
- `persistent_workers=True` (si `num_workers > 0`)
- `batch_size=32` (`EXPERT1_BATCH_SIZE`)
- `num_workers=4` (`EXPERT1_NUM_WORKERS`)

El dataset `ChestXray14Dataset` se instancia con `mode="expert"` (retorna `(img_tensor, label_vec_14, img_name)`) y `use_cache=True` (preload en RAM).

**Wrapper de compatibilidad:** `build_dataloaders_expert1()` adapta la API legacy (retorna tupla `(train, val, test, pos_weight)`) a la nueva API (dict).

---

## 8. Criterio de pérdida y métricas

### 8.1 Loss

`BCEWithLogitsLoss(pos_weight=pos_weight_tensor)` — multilabel con 14 clases independientes.

El `pos_weight_tensor` (shape `[14]`) se computa automáticamente por `ChestXray14Dataset` en modo `"expert"`: `n_neg / n_pos` por clase. Compensa el desbalance de prevalencia entre patologías (~0.2% Hernia → ~17.7% Infiltration, ~53% No Finding).

### 8.2 Métricas

| Métrica | Uso | Justificación |
|---|---|---|
| **AUC-ROC por clase** | Evaluación principal (14 AUCs individuales) | Métrica estándar para multilabel desbalanceado (robusta ante threshold) |
| **Macro AUC** | Monitor de early stopping y checkpoint | Media de las 14 AUC-ROC individuales |
| **F1 Macro** | Evaluación secundaria | Requerida por la rúbrica del proyecto |
| **AUPRC** | Evaluación complementaria | Más informativa que AUC-ROC en clases de muy baja prevalencia |
| **Accuracy** | **NUNCA** | ~54% prediciendo siempre "No Finding" — métrica engañosa para este dataset |

### 8.3 Evaluación con TTA

La evaluación final usa **Test-Time Augmentation (TTA)**:
1. Forward pass con test original (sin augmentation)
2. Forward pass con test + HorizontalFlip determinista (p=1.0)
3. Promedio de logits: `tta_logits = (logits_orig + logits_flip) / 2.0`
4. AUC-ROC por clase y macro AUC calculados sobre los logits promediados

La función `eval_with_tta()` orquesta este proceso usando los DataLoaders `test` y `test_flip`.

---

## 9. Training loop

### 9.1 `train_one_epoch()`

- Gradient accumulation: acumula gradientes cada `EXPERT1_ACCUMULATION_STEPS` (4) pasos
- Mixed precision con `torch.amp.autocast()` + `GradScaler`
- Loss escalada: `loss = loss / accumulation_steps` antes de backward
- Flush de gradientes residuales al final de la época

### 9.2 `validate()`

- `@torch.no_grad()` — sin cálculo de gradientes
- Acumula logits y labels de todos los batches
- Calcula `val_loss`, `val_macro_auc` y `val_auc_per_class` (14 AUCs)
- AUC-ROC per-class con `sklearn.metrics.roc_auc_score`; devuelve 0.0 si una clase tiene una sola label en el split

### 9.3 `_run_phase()`

Función genérica que ejecuta una fase completa (LP o FT):
- Itera épocas locales con offset global
- Llama `train_one_epoch()` + `validate()` por época
- Aplica scheduler (si se proporcionó, solo en FT)
- Guarda checkpoint del mejor modelo (por `val_macro_auc`)
- Evalúa early stopping (si se proporcionó, solo en FT)
- Guarda training log JSON incremental por época

### 9.4 Early stopping

La clase `EarlyStoppingAUC` monitorea `val_macro_auc` (maximizar):
- `patience=10` épocas sin mejora (delta > `min_delta=0.001`)
- Solo activo en la fase FT (no en LP)
- Cuando se dispara, detiene el entrenamiento y el mejor checkpoint ya está guardado

---

## 10. Checkpoint

| Campo | Valor |
|---|---|
| **Directorio** | `checkpoints/expert_00_convnext_tiny/` |
| **Archivo modelo** | `expert1_best.pt` |
| **Archivo log** | `expert1_training_log.json` |
| **Criterio de guardado** | Mejor `val_macro_auc` durante el entrenamiento (LP o FT) |

El checkpoint contiene:
- `epoch` — época global del mejor modelo
- `phase` — `"LP"` o `"FT"`
- `model_state_dict` — pesos completos del modelo
- `optimizer_state_dict` — estado del optimizador
- `val_macro_auc` — macro AUC de validación
- `val_loss` — loss de validación
- `val_auc_per_class` — lista de 14 AUCs individuales
- `config` — diccionario con todos los hiperparámetros (lp_lr, ft_lr, weight_decay, dropout_fc, batch_size, accumulation_steps, fp16, lp_epochs, ft_epochs, seed)

---

## 11. Hallazgos del dataset implementados (H1-H6)

| ID | Hallazgo | Implementación |
|---|---|---|
| **H1** | Modo expert: multilabel | Vector binario `[14]` + `BCEWithLogitsLoss` (nunca `CrossEntropyLoss`) |
| **H2** | Leakage por Patient ID | Split por `file_list` oficial del NIH + verificación cruzada de Patient IDs entre splits |
| **H3** | Ruido NLP en etiquetas | Warning en logs: etiquetas generadas por NLP (~>90% precisión, Wang et al.) |
| **H4** | View Position (PA/AP) | Filtro opcional `filter_view` + log de distribución PA/AP + warning de sesgo en AP |
| **H5** | Bounding boxes | `load_bbox_index()` para validar heatmaps (8 de 14 patologías con BBox) |
| **H6** | Desbalance de clases | `pos_weight` automático + FocalLossMultiLabel disponible como alternativa |

---

## 12. Cómo ejecutar

### 12.1 Entrenamiento completo

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/train_expert1.py
```

### 12.2 Con data-root explícito

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/train_expert1.py --data-root /ruta/al/proyecto
```

### 12.3 Dry-run (verificación del pipeline)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/train_expert1.py --dry-run
```

El modo `--dry-run` ejecuta:

1. Fijación de semilla (42)
2. Instanciación del modelo `Expert1ConvNeXtTiny` (pretrained)
3. Construcción de DataLoaders (con `num_workers=0`)
4. Fase LP: 1 época con 2 batches de train + 1 batch de val
5. Fase FT: 1 época con 2 batches de train + 1 batch de val
6. TTA sobre test set (sin cargar checkpoint)
7. Verificación de shapes, métricas y flujo completo

### 12.4 Rutas por defecto del dataset

| Ruta | Valor por defecto |
|---|---|
| CSV | `{project_root}/datasets/nih_chest_xrays/Data_Entry_2017.csv` |
| Imágenes | `{project_root}/datasets/nih_chest_xrays/all_images/` |
| Train split | `{project_root}/datasets/nih_chest_xrays/splits/nih_train_list.txt` |
| Val split | `{project_root}/datasets/nih_chest_xrays/splits/nih_val_list.txt` |
| Test split | `{project_root}/datasets/nih_chest_xrays/splits/nih_test_list.txt` |

---

## 13. Dependencias clave

| Librería | Uso |
|---|---|
| `timm` | Backbone pretrained (`convnext_tiny.in12k_ft_in1k`) + resolución de `model_mean`/`model_std` |
| `albumentations` | Transforms de augmentación (train), normalización (val/test) y TTA (flip) |
| `cv2` (OpenCV) | Carga de imágenes, CLAHE, multistage_resize, conversión grayscale→RGB |
| `torch.amp` | Mixed precision (FP16) con `autocast` + `GradScaler` |
| `sklearn.metrics` | `roc_auc_score` para AUC-ROC por clase |

---

## 14. Estado

| Verificación | Resultado |
|---|---|
| Importaciones | ✅ OK |
| Forward pass del modelo | ✅ OK — shape `(2, 14)` |
| freeze/unfreeze backbone | ✅ OK — reduce y restaura trainables |
| model_mean/model_std desde timm | ✅ OK — resueltos programáticamente |
| Dry-run | ✅ Pasa (falla solo en datos ausentes — esperado) |
| Entrenamiento real | ⏳ Pendiente de datos |

---

*Documento actualizado el 2026-04-14. Fuentes: `src/pipeline/fase2/expert1_config.py`, `models/expert1_convnext.py`, `datasets/chest.py`, `dataloader_expert1.py`, `train_expert1.py`.*

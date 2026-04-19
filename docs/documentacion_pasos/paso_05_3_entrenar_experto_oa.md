# Paso 5.3 — Entrenar Experto 2 (OA Knee / EfficientNet-B3)

| Campo | Valor |
|---|---|
| **Fecha** | 2026-04-14 |
| **Alcance** | Entrenamiento del Experto 2 (clasificación de 5 grados de osteoartritis de rodilla, escala Kellgren-Lawrence KL0–KL4) con EfficientNet-B3 pretrained (ImageNet). Genera checkpoint consumido por Fase 4 (ablation) y Fase 5 (fine-tuning global) |
| **Estado general** | ✅ **Script creado y verificado con dry-run** — pendiente de ejecutar entrenamiento completo |

---

## 1. Objetivo

Entrenar un modelo **EfficientNet-B3 pretrained** (ImageNet1K, `torchvision.models.efficientnet_b3`) para clasificación de osteoartritis de rodilla según la escala Kellgren-Lawrence completa de **5 clases** (KL0–KL4), usando **fine-tuning con Adam diferencial** (learning rates distintas para backbone y head).

Este experto es el Experto 2 del sistema MoE (`expert_id=2`), responsable del dominio de radiografía de rodilla. La métrica principal es **F1 Macro**, complementada por QWK (Quadratic Weighted Kappa) como métrica ordinal.

---

## 2. Archivos del Experto 2

| Archivo | Descripción |
|---|---|
| `src/pipeline/fase2/expert_oa_config.py` | Constantes de hiperparámetros (prefijo `EXPERT_OA_`). Fuente de verdad para LRs diferenciales, WD, dropout, batch, accumulation, patience, scheduler, etc. |
| `src/pipeline/fase2/models/expert_oa_vgg16bn.py` | Clase `ExpertOAEfficientNetB3` (alias `ExpertOAVGG16BN` por compatibilidad) — EfficientNet-B3 via `torchvision.models.efficientnet_b3(weights=IMAGENET1K_V1)` con head reemplazada: `Dropout(0.4) → Linear(1536, 5)`. Métodos `get_backbone_params()` / `get_head_params()` para Adam diferencial |
| `src/pipeline/fase2/dataloader_expert_oa.py` | Función `get_oa_dataloaders()` — retorna `(train_loader, val_loader, test_loader, class_weights_tensor)`. Transforms internos en `OAKneeDataset` |
| `src/pipeline/fase2/train_expert_oa.py` | Script de entrenamiento completo: Adam diferencial, CosineAnnealingLR, FP16, gradient accumulation, EarlyStopping por `val_f1_macro`, soporte `--dry-run` |
| `src/pipeline/datasets/osteoarthritis.py` | Clase `OAKneeDataset` — carga desde carpetas `{split}/{0,1,2,3,4}/`, transforms internos (train/val diferenciados), `class_weights` inverse-frequency, `compute_qwk()`, `evaluate_boundary_confusion()` |

### 2.1 Nota sobre nomenclatura de archivos

Los archivos conservan nombres históricos (`expert_oa_vgg16bn.py`, `checkpoints/expert_02_vgg16_bn/`) para no romper imports ni referencias en otros módulos del pipeline. La clase interna es `ExpertOAEfficientNetB3`; se exporta un alias `ExpertOAVGG16BN` para compatibilidad hacia atrás.

---

## 3. Arquitectura del modelo

```
imgs [B, 3, 224, 224]
    │
    ▼
EfficientNet-B3 (pretrained ImageNet1K)
    │
    ├── features: bloques MBConv con Squeeze-and-Excitation (compound scaling B3)
    │   → [B, 1536, 7, 7]
    │
    ├── avgpool: AdaptiveAvgPool2d(1) → [B, 1536, 1, 1]
    │
    └── classifier (HEAD reemplazada):
        Dropout(0.4) → Linear(1536, 5)
    │
    ▼
logits [B, 5]  ← CrossEntropyLoss(weight=class_weights)
```

| Campo | Valor |
|---|---|
| **Clase** | `ExpertOAEfficientNetB3` (alias: `ExpertOAVGG16BN`) |
| **Backbone** | `torchvision.models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)` |
| **Pretraining** | ImageNet-1K (transfer learning) |
| **Parámetros totales** | ~10.7M |
| **Entrada** | `[B, 3, 224, 224]` — radiografía de rodilla RGB, float32 |
| **Salida** | `[B, 5]` — logits crudos (antes de softmax) |
| **Cabeza FC** | `Dropout(0.4) → Linear(1536, 5)` |
| **Normalización** | ImageNet estándar: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |

> **Nota:** Los archivos del modelo conservan nombres históricos (`expert_oa_vgg16bn.py`, checkpoint `expert_02_vgg16_bn/`) por compatibilidad. La clase principal ahora es `ExpertOAEfficientNetB3`, exportada con alias `ExpertOAVGG16BN`.

**Métodos para Adam diferencial:**
- `get_backbone_params()` — retorna parámetros de `model.features` (backbone EfficientNet)
- `get_head_params()` — retorna parámetros de `model.classifier` (head reemplazada)
- `count_parameters()` — cuenta parámetros entrenables (con `requires_grad=True`)
- `count_all_parameters()` — cuenta todos los parámetros (entrenables + congelados)

### 3.1 Clases de salida (5 grados Kellgren-Lawrence)

| Grado KL | Clase modelo | Descripción clínica |
|---|---|---|
| KL 0 | Cls 0 — Normal | Sin signos de osteoartritis |
| KL 1 | Cls 1 — Dudoso | Cambios dudosos, posible osteofito mínimo |
| KL 2 | Cls 2 — Leve | Osteofitos definidos, posible estrechamiento del espacio articular |
| KL 3 | Cls 3 — Moderado | Osteofitos múltiples, estrechamiento moderado, esclerosis subcondral |
| KL 4 | Cls 4 — Severo | Espacio articular muy reducido, esclerosis marcada, deformidad ósea |

---

## 4. Configuración de entrenamiento

| Hiperparámetro | Valor | Constante en `expert_oa_config.py` |
|---|---|---|
| **Modelo** | EfficientNet-B3 (pretrained ImageNet) | `EXPERT_OA_MODEL_NAME` |
| **Optimizador** | Adam diferencial (2 param groups) | `EXPERT_OA_OPTIMIZER` |
| **LR backbone** | 5×10⁻⁵ | `EXPERT_OA_LR_BACKBONE` |
| **LR head** | 5×10⁻⁴ | `EXPERT_OA_LR_HEAD` |
| **Weight decay** | 1×10⁻⁴ | `EXPERT_OA_WEIGHT_DECAY` |
| **Dropout FC** | 0.4 | `EXPERT_OA_DROPOUT_FC` |
| **Batch size (real)** | 32 | `EXPERT_OA_BATCH_SIZE` |
| **Accumulation steps** | 2 | `EXPERT_OA_ACCUMULATION_STEPS` |
| **Batch efectivo** | 64 (32 × 2) | — |
| **Mixed precision (FP16)** | Sí | `EXPERT_OA_FP16` |
| **Max épocas** | 30 | `EXPERT_OA_MAX_EPOCHS` |
| **Scheduler** | CosineAnnealingLR(T_max=30, eta_min=1e-6) | `EXPERT_OA_SCHEDULER`, `EXPERT_OA_SCHEDULER_T_MAX`, `EXPERT_OA_SCHEDULER_ETA_MIN` |
| **Early stopping patience** | 10 | `EXPERT_OA_EARLY_STOPPING_PATIENCE` |
| **Early stopping monitor** | `val_f1_macro` (maximizar) | `EXPERT_OA_EARLY_STOPPING_MONITOR` |
| **Early stopping min_delta** | 0.001 | — |
| **Imagen** | 224×224 | `EXPERT_OA_IMG_SIZE` |
| **Num clases** | 5 | `EXPERT_OA_NUM_CLASSES` |
| **Num workers** | 4 (0 en dry-run) | — |
| **Semilla** | 42 | `_SEED` |

### 4.1 Adam diferencial

El optimizador usa **2 parameter groups** con learning rates distintos:

| Grupo | Parámetros | LR | Justificación |
|---|---|---|---|
| Backbone | `model.features` (~10.4M params) | 5×10⁻⁵ | LR conservador para preservar features pretrained de ImageNet |
| Head | `model.classifier` (~7.7K params) | 5×10⁻⁴ | LR 10× mayor — la head se inicializa aleatoriamente y necesita adaptarse rápido a las 5 clases KL |

Ambos grupos comparten `weight_decay=1e-4`.

---

## 5. Pipeline de preprocesado y transforms

Los transforms están definidos en `src/pipeline/datasets/osteoarthritis.py` como `TRANSFORM_TRAIN` y `TRANSFORM_VAL`, y se aplican internamente dentro de `OAKneeDataset`. No se pasan transforms externos al constructor del dataset.

### 5.1 Train transforms (con augmentación)

| Paso | Técnica | Parámetros |
|---|---|---|
| 1 | `Resize` | `(256, 256, antialias=True)` |
| 2 | `RandomCrop` | `224` — recorte aleatorio del resize 256→224 |
| 3 | `RandomHorizontalFlip` | `p=0.5` — anatomía de rodilla simétrica izquierda-derecha |
| 4 | `RandomRotation` | `degrees=15` (±15°) — variabilidad de posicionamiento del paciente |
| 5 | `ColorJitter` | `brightness=0.3, contrast=0.3` — compensa variabilidad de exposición entre equipos |
| 6 | `RandomAutocontrast` | `p=0.3` — realza contraste adaptativo en un subconjunto de imágenes |
| 7 | `ToTensor` | PIL Image → `[C, H, W]` float32 [0, 1] |
| 8 | `Normalize` | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` (ImageNet) |

### 5.2 Val / Test transforms (sin augmentación)

| Paso | Técnica | Parámetros |
|---|---|---|
| 1 | `Resize` | `(224, 224, antialias=True)` |
| 2 | `ToTensor` | PIL Image → `[C, H, W]` float32 [0, 1] |
| 3 | `Normalize` | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` (ImageNet) |

> **⚠ PROHIBIDO:** `RandomVerticalFlip` — las radiografías de rodilla tienen orientación anatómica fija (tibia abajo, fémur arriba). Un flip vertical produce una imagen anatómicamente imposible.

> **⚠ PROHIBIDO:** `RandomErasing`, `Cutout`, `CutMix`, `MixUp` — restricción del proyecto MoE.

---

## 6. DataLoaders

La función `get_oa_dataloaders()` retorna una tupla de 4 elementos:

| Elemento | Tipo | Descripción |
|---|---|---|
| `train_loader` | `DataLoader` | `shuffle=True, drop_last=True`, transform con augmentación |
| `val_loader` | `DataLoader` | `shuffle=False, drop_last=False`, solo resize + normalización |
| `test_loader` | `DataLoader` | `shuffle=False, drop_last=False`, solo resize + normalización |
| `class_weights` | `Tensor[5]` | Pesos inverse-frequency por clase, para `CrossEntropyLoss` |

Configuración común de DataLoaders:
- `pin_memory=True`
- `persistent_workers=True` (si `num_workers > 0`)
- `batch_size=32` (`EXPERT_OA_BATCH_SIZE`)
- `num_workers=4` (0 en dry-run)

El dataset `OAKneeDataset` se instancia con `mode="expert"` (retorna `(img_tensor, kl_label_int, img_name)`) y el split correspondiente. Los transforms son internos al dataset (selección automática: `TRANSFORM_TRAIN` para train, `TRANSFORM_VAL` para val/test).

**Ruta del dataset:** `datasets/osteoarthritis/oa_splits/{train,val,test}/{0,1,2,3,4}/`

---

## 7. Criterio de pérdida y métricas

### 7.1 Loss

`CrossEntropyLoss(weight=class_weights_tensor)` — clasificación multiclase con 5 clases.

El `class_weights_tensor` (shape `[5]`) se computa automáticamente por `OAKneeDataset` usando frecuencia inversa: `total / (n_classes × count_per_class)`. Compensa el desbalance entre los 5 grados KL.

### 7.2 Métricas

| Métrica | Uso | Justificación |
|---|---|---|
| **F1 Macro** | Evaluación principal + early stopping + checkpoint | Media de F1 por clase. Captura rendimiento balanceado entre las 5 clases KL. Monitor para early stopping y selección de mejor modelo |
| **QWK** | Evaluación complementaria (ordinal) | Quadratic Weighted Kappa — penaliza errores proporcionalmente al cuadrado de la distancia ordinal (Severo→Normal penaliza 16×, clase adyacente penaliza 1×). Informativa para evaluar coherencia ordinal |
| **Accuracy** | **NUNCA** | No distingue la severidad del error — clasificar Severo como Normal pesa lo mismo que Leve como Normal |

### 7.3 Umbral de aceptación

`val_f1_macro ≥ 0.72` — requerido por la rúbrica del proyecto (§10.1: F1 Macro > 0.72 para full marks en expertos 2D). Definido en `EXPERT_OA_TARGET_METRIC_THRESHOLD`.

### 7.4 Análisis de fronteras con `evaluate_boundary_confusion()`

El dataset provee una función estática que analiza la matriz de confusión 5×5 con foco en las fronteras entre clases adyacentes (KL0↔KL1, KL1↔KL2, KL2↔KL3, KL3↔KL4), tasas de error por frontera, recall por clase, y el peor error clínico (KL4 predicho como KL0).

---

## 8. Training loop

### 8.1 `train_one_epoch()`

- Gradient accumulation: acumula gradientes cada `EXPERT_OA_ACCUMULATION_STEPS` (2) pasos
- Mixed precision con `torch.amp.autocast()` + `GradScaler`
- Loss escalada: `loss = loss / accumulation_steps` antes de backward
- Flush de gradientes residuales al final de la época

### 8.2 `validate()`

- `@torch.no_grad()` — sin cálculo de gradientes
- Acumula predicciones (`argmax`) y labels de todos los batches
- Calcula `val_loss`, `val_f1_macro` (sklearn `f1_score(average='macro')`) y `val_qwk` (sklearn `cohen_kappa_score(weights='quadratic')`)

### 8.3 Training loop principal

El bucle en `train()` ejecuta una única fase de fine-tuning (no LP-FT):
- Itera hasta `EXPERT_OA_MAX_EPOCHS` (30) épocas
- Llama `train_one_epoch()` + `validate()` por época
- Aplica `scheduler.step()` cada época (CosineAnnealingLR)
- Guarda checkpoint del mejor modelo (por `val_f1_macro`)
- Evalúa early stopping (por `val_f1_macro`)
- Guarda training log JSON incremental por época

### 8.4 Early stopping

La clase `EarlyStopping` monitorea `val_f1_macro` (maximizar):
- `patience=10` épocas sin mejora (delta > `min_delta=0.001`)
- Cuando se dispara, detiene el entrenamiento y el mejor checkpoint ya está guardado

---

## 9. Checkpoint

| Campo | Valor |
|---|---|
| **Directorio** | `checkpoints/expert_02_vgg16_bn/` (nombre preservado por compatibilidad) |
| **Archivo modelo** | `expert_oa_best.pt` |
| **Archivo log** | `expert_oa_training_log.json` |
| **Criterio de guardado** | Mejor `val_f1_macro` durante el entrenamiento |

El checkpoint contiene:
- `epoch` — época del mejor modelo
- `model_state_dict` — pesos completos del modelo (EfficientNet-B3)
- `optimizer_state_dict` — estado del optimizador (Adam, 2 param groups)
- `scheduler_state_dict` — estado del CosineAnnealingLR
- `val_loss` — loss de validación
- `val_f1_macro` — F1 Macro de validación (métrica principal)
- `val_qwk` — QWK de validación (métrica complementaria)
- `config` — diccionario con todos los hiperparámetros (lr_backbone, lr_head, weight_decay, dropout_fc, batch_size, accumulation_steps, fp16, num_classes, scheduler_t_max, scheduler_eta_min, n_params, seed)

---

## 10. Cómo ejecutar

### 10.1 Entrenamiento completo

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/train_expert_oa.py
```

### 10.2 Dry-run (verificación del pipeline)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/train_expert_oa.py --dry-run
```

El modo `--dry-run` ejecuta:

1. Fijación de semilla (42)
2. Instanciación del modelo `ExpertOAEfficientNetB3` (pretrained ImageNet)
3. Construcción de DataLoaders (con `num_workers=0`)
4. 1 época con 2 batches de train + 2 batches de val
5. Cálculo de métricas (F1 Macro, QWK) sobre los batches limitados
6. Verificación de shapes, métricas y flujo completo
7. No guarda checkpoint

### 10.3 Rutas por defecto del dataset

| Ruta | Valor por defecto |
|---|---|
| Root dir | `{project_root}/datasets/osteoarthritis/oa_splits/` |
| Train split | `{root_dir}/train/{0,1,2,3,4}/` |
| Val split | `{root_dir}/val/{0,1,2,3,4}/` |
| Test split | `{root_dir}/test/{0,1,2,3,4}/` |

---

## 11. Hallazgos del dataset implementados (H1-H5)

| ID | Hallazgo | Implementación |
|---|---|---|
| **H1** | Desbalance entre clases KL | `class_weights` inverse-frequency computado por `OAKneeDataset` + `CrossEntropyLoss(weight=...)` |
| **H2** | Posible augmentation offline en el ZIP | `_audit_augmentation_offline()` detecta ratio anómalo de imágenes y hashes duplicados. Si se detecta, se ajustan transforms |
| **H3** | Sin metadatos de paciente | Split tomado tal cual de las carpetas del ZIP original (OAI). Limitación documentada: posible leakage a nivel paciente |
| **H4** | Orientación anatómica fija | `RandomVerticalFlip` prohibido — tibia abajo, fémur arriba. Solo `RandomHorizontalFlip` es seguro (rodilla izq. flipped ≈ rodilla der.) |
| **H5** | Fronteras ambiguas entre grados KL | `evaluate_boundary_confusion()` para monitorear errores por frontera adyacente. Fronteras KL0↔KL1 y KL1↔KL2 son las más difíciles (alta variabilidad inter-observador) |

---

## 12. Restricciones aplicadas

| Restricción | Detalle |
|---|---|
| **Pretrained ImageNet** | `efficientnet_b3(weights=IMAGENET1K_V1)` — transfer learning desde ImageNet-1K |
| **Sin augmentaciones de oclusión** | Prohibido `RandomErasing`, `Cutout`, `CutMix` o cualquier transformación que oculte partes de la imagen |
| **Sin RandomVerticalFlip** | Orientación anatómica fija (tibia abajo, fémur arriba) — flip vertical produce imágenes anatómicamente imposibles |
| **Nombres de archivo preservados** | `expert_oa_vgg16bn.py` y `checkpoints/expert_02_vgg16_bn/` conservan nombres históricos por compatibilidad con el pipeline |

---

## 13. Cambios respecto a la versión anterior (VGG16-BN → EfficientNet-B0 → EfficientNet-B3)

| Campo | Antes (EfficientNet-B0) | Ahora (EfficientNet-B3) |
|---|---|---|
| **Modelo** | EfficientNet-B0 pretrained | EfficientNet-B3 pretrained |
| **Parámetros** | ~5.3M | ~10.7M |
| **Feature dim** | 1280 | 1536 |
| **Cabeza** | Linear(1280, 5) | Linear(1536, 5) |
| **Optimizador** | Adam diferencial (backbone=5e-5, head=5e-4) | Adam diferencial (backbone=5e-5, head=5e-4) |
| **Scheduler** | CosineAnnealingLR(T_max=30) | CosineAnnealingLR(T_max=30) |
| **Épocas** | 30 | 30 |
| **Dropout** | 0.4 | 0.4 |
| **Accumulation** | 2 (batch efectivo: 64) | 2 (batch efectivo: 64) |
| **Métrica principal** | F1 Macro | F1 Macro |
| **Early stopping** | `val_f1_macro` | `val_f1_macro` |
| **Transforms** | Sin cambios | Sin cambios |

---

## 14. Dependencias clave

| Librería | Uso |
|---|---|
| `torchvision` | Backbone pretrained (`efficientnet_b3` + `EfficientNet_B3_Weights.IMAGENET1K_V1`), transforms (`Resize`, `RandomCrop`, `ColorJitter`, etc.) |
| `torch.amp` | Mixed precision (FP16) con `autocast` + `GradScaler` |
| `sklearn.metrics` | `f1_score(average='macro')` para F1 Macro, `cohen_kappa_score(weights='quadratic')` para QWK |
| `PIL` (Pillow) | Carga de imágenes en `OAKneeDataset` |

---

## 15. Estado

| Verificación | Resultado |
|---|---|
| Importaciones | ✅ OK |
| Forward pass del modelo | ✅ OK — shape `(2, 5)` |
| `get_backbone_params()` / `get_head_params()` | ✅ OK — parámetros separados para Adam diferencial |
| Alias `ExpertOAVGG16BN` → `ExpertOAEfficientNetB3` | ✅ OK — compatibilidad verificada |
| Conteo de parámetros | ✅ ~10.7M |
| Dry-run | ✅ Exitoso |
| Entrenamiento real | ⏳ Pendiente de ejecutar |

---

*Documento actualizado el 2026-04-14. Fuentes: `src/pipeline/fase2/expert_oa_config.py`, `models/expert_oa_vgg16bn.py`, `datasets/osteoarthritis.py`, `dataloader_expert_oa.py`, `train_expert_oa.py`.*

# Paso 5.2 — Entrenar Experto 2 (ISIC 2019 / ConvNeXt-Small)

| Campo | Valor |
|---|---|
| **Fecha** | 2026-04-14 |
| **Alcance** | Entrenamiento del Experto 2 (clasificación multiclase de 8+1 clases de lesiones cutáneas) con ConvNeXt-Small desde cero. Genera checkpoint consumido por Fase 4 (ablation) y Fase 5 (fine-tuning global) |
| **Estado general** | ✅ **Script creado y verificado con dry-run** — datos pendientes de disponibilidad para entrenamiento real |

---

## 1. Objetivo

Entrenar un modelo **ConvNeXt-Small desde cero** (sin pesos preentrenados) para clasificación **multiclase** de lesiones cutáneas del dataset ISIC 2019. Este experto es el Experto 1 del sistema MoE, responsable del dominio de dermatoscopía.

El dataset ISIC 2019 combina imágenes de 3 fuentes con bias de dominio (HAM10000, BCN_20000, MSK), lo que requiere augmentación agresiva como contramedida.

---

## 2. Archivos creados

| Archivo | Descripción |
|---|---|
| `src/pipeline/fase2/expert2_config.py` | Constantes de hiperparámetros (prefijo `EXPERT2_`). Fuente de verdad para LR, WD, dropout, batch, accumulation, patience, etc. |
| `src/pipeline/fase2/models/expert2_convnext.py` | Clase `Expert2ConvNeXtSmall` — ConvNeXt-Small via `timm.create_model('convnext_small', pretrained=False)`, cabeza adaptada a 9 neuronas con Dropout(0.3) |
| `src/pipeline/fase2/dataloader_expert2.py` | Función `build_dataloaders_expert2()` — retorna `(train_loader, val_loader, test_loader, class_weights_tensor)` |
| `src/pipeline/fase2/train_expert2.py` | Script de entrenamiento completo: bucle de épocas, EarlyStopping, checkpointing, métricas, soporte `--dry-run` |

---

## 3. Arquitectura del modelo

```
imgs [B, 3, 224, 224]
    │
    ▼
ConvNeXt-Small (weights=None, desde cero)
    │
    ├── stem: Conv2d(3, 96, 4, stride=4) → LayerNorm
    ├── stages: 4 etapas de bloques ConvNeXt
    │   → [B, 768, 7, 7]
    │
    ├── norm: LayerNorm → [B, 768]
    └── head: Dropout(0.3) → Linear(768, 9)
    │
    ▼
logits [B, 9]  ← FocalLossMultiClass(gamma=2.0, weight=class_weights, label_smoothing=0.1)
```

| Campo | Valor |
|---|---|
| **Clase** | `Expert2ConvNeXtSmall` |
| **Backbone** | `timm.create_model("convnext_small", pretrained=False, num_classes=0)` |
| **Pretraining** | Ninguno (entrenamiento desde cero — requisito del proyecto) |
| **Parámetros totales** | ~50M |
| **Entrada** | `[B, 3, 224, 224]` — imagen dermatoscópica RGB, float32 |
| **Salida** | `[B, 9]` — logits crudos (antes de softmax) |
| **Cabeza FC** | `Dropout(0.3) → Linear(768, 9)` |

> **Nota:** ConvNeXt-Small tiene feature dimension 768 (igual que ConvNeXt-Tiny pero con mayor profundidad de bloques). Los 9 slots de salida mantienen la estructura UNK del experto ISIC (ver sección 3.1).

### 3.1 Nota sobre las 9 neuronas de salida

La capa de salida tiene **9 neuronas** aunque el entrenamiento solo usa **8 clases** (MEL, NV, BCC, AK, BKL, DF, VASC, SCC). La novena neurona (índice 8) es un **slot UNK** reservado para inferencia:

| Índice | Clase | Uso |
|---|---|---|
| 0 | MEL (Melanoma) | Train + Inferencia |
| 1 | NV (Melanocytic nevus) | Train + Inferencia |
| 2 | BCC (Basal cell carcinoma) | Train + Inferencia |
| 3 | AK (Actinic keratosis) | Train + Inferencia |
| 4 | BKL (Benign keratosis) | Train + Inferencia |
| 5 | DF (Dermatofibroma) | Train + Inferencia |
| 6 | VASC (Vascular lesion) | Train + Inferencia |
| 7 | SCC (Squamous cell carcinoma) | Train + Inferencia |
| 8 | UNK (Unknown) | Solo inferencia |

El slot UNK no recibe supervisión durante el entrenamiento (`FocalLossMultiClass` con labels 0–7). Su peso en `class_weights` es 1.0. En inferencia, el softmax sobre 9 neuronas permite detectar distribuciones anómalas vía entropía alta o probabilidad concentrada en UNK.

---

## 4. Configuración de entrenamiento

| Hiperparámetro | Valor | Constante en `expert2_config.py` |
|---|---|---|
| **Optimizador** | AdamW | — |
| **Learning rate** | 3×10⁻⁴ | `EXPERT2_LR` |
| **Weight decay** | 0.05 | `EXPERT2_WEIGHT_DECAY` |
| **Dropout FC** | 0.3 | `EXPERT2_DROPOUT_FC` |
| **Batch size (real)** | 32 | `EXPERT2_BATCH_SIZE` |
| **Accumulation steps** | 4 | `EXPERT2_ACCUMULATION_STEPS` |
| **Batch efectivo** | 128 (32 × 4) | — |
| **Mixed precision (FP16)** | Sí | `EXPERT2_FP16` |
| **Max épocas** | 50 | `EXPERT2_MAX_EPOCHS` |
| **Early stopping patience** | 10 | `EXPERT2_EARLY_STOPPING_PATIENCE` |
| **Early stopping monitor** | `val_loss` | `EXPERT2_EARLY_STOPPING_MONITOR` |

---

## 5. Pipeline de preprocesado

El preprocesado se divide en dos fases: **offline** (una sola vez, script `pre_isic.py`) y **online** (por epoch, integrado en `ISICDataset`).

### 5.0 Preprocesado offline (`pre_isic.py` — Fase 0, Paso 6)

| Paso | Técnica | Justificación |
|---|---|---|
| 1 | **Auditoría** | Registra dimensiones, aspect ratio, fuente (HAM/BCN/MSK) por imagen |
| 2 | **DullRazor** (hair removal) | Eliminación de vello: grayscale → morphological closing → detección diferencia > 10 → inpainting TELEA. Reduce artefactos que confunden al modelo |
| 3 | **Resize** (lado corto=224px) | Aspect-ratio-preserving con LANCZOS. Las imágenes resultantes son rectangulares (lado corto=224, lado largo ≥ 224) |
| 4 | **Guardado en caché** | JPEG calidad 95 como `{isic_id}_pp_224.jpg` en `ISIC_2019_Training_Input_preprocessed/` |

**Notas:**
- Color Constancy NO se aplica offline (se aplica online con p=0.5 para mantener variabilidad cromática).
- Idempotente: si el archivo cacheado ya existe, se salta.
- El script genera `preprocess_report.json` con estadísticas.

### 5.1 Preprocesado online (por epoch)

| Paso | Técnica | Justificación |
|---|---|---|
| 1 | **RandomCrop(224)** | Extrae cuadrado 224×224 de la imagen cacheada (que tiene lado corto=224). Introduce variabilidad espacial |
| 2 | **RandomHorizontalFlip(p=0.5)** | Lesiones cutáneas son simétricas horizontalmente |
| 3 | **RandomVerticalFlip(p=0.5)** | No hay orientación vertical canónica |
| 4 | **RandomRotation(360°)** | Variabilidad de orientación del dermatoscopio. Rango completo 360° |
| 5 | **ColorJitter** | `brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1` — contramedida contra bias de dominio |
| 6 | **RandomGamma(p=0.5)** | Corrección gamma aleatoria γ ∈ [0.7, 1.5] — simula variaciones de exposición |
| 7 | **ShadesOfGray(p=0.5)** | Color Constancy por norma de Minkowski (p=6) — normaliza iluminante. Aplicado online para mantener variabilidad |
| 8 | **CoarseDropout(p=0.5)** | 1–3 parches rectangulares (32–96px) eliminados, relleno con mean ImageNet. Regularización contra sobreajuste a regiones |
| 9 | **ToTensor** | Conversión a tensor PyTorch |
| 10 | **Normalize** (ImageNet stats) | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |

**Pipeline train:** `RandomCrop(224) → HFlip → VFlip → Rotation(360°) → ColorJitter → RandomGamma → ShadesOfGray → CoarseDropout → ToTensor → Normalize(ImageNet)`

**Pipeline val/test:** `CenterCrop(224) → ToTensor → Normalize(ImageNet)`

> **Nota sobre caché:** Si `cache_dir` está configurado y la imagen preprocesada existe, se carga directamente (ya tiene lado corto=224). Si no existe, se carga la imagen original y se hace resize interno a shorter_side=224 antes de aplicar el transform.

### 5.2 Augmentación a nivel de batch (solo train)

CutMix y MixUp se aplican **en el training loop** (no como transform de imagen). Son mutuamente excluyentes por batch:

| Técnica | Probabilidad | Parámetros | Justificación |
|---|---|---|---|
| **CutMix** | p=0.3 | alpha=1.0 (Beta distribution) | Recorta región de imagen A y la pega en imagen B. Loss: λ·L(y_a) + (1-λ)·L(y_b). Mejora generalización en clases minoritarias |
| **MixUp** | p=0.2 | alpha=0.4 (Beta distribution) | Combina linealmente dos imágenes y sus etiquetas. Loss: λ·L(y_a) + (1-λ)·L(y_b). Regularización suave |
| **Ninguna** | p=0.5 | — | Batch normal sin mezcla |

> **Nota sobre augmentaciones de oclusión:** La restricción original de "prohibir RandomErasing, Cutout, CutMix" se ha levantado. CutMix opera a nivel de batch (mezcla entre imágenes), no elimina información — la región recortada se reemplaza con contenido de otra imagen del batch. CoarseDropout reemplaza la función de Cutout con relleno en mean ImageNet en lugar de negro.

---

## 6. Criterio de pérdida y métricas

### 6.1 Loss

`FocalLossMultiClass(gamma=2.0, weight=class_weights_tensor, label_smoothing=0.1)` — multiclase con 9 neuronas de salida.

La Focal Loss (Lin et al., ICCV 2017) reduce el gradiente de ejemplos fáciles y concentra el aprendizaje en ejemplos difíciles de clases minoritarias (DF, VASC, AK, SCC). El `class_weights_tensor` se computa automáticamente por `build_dataloaders_expert2()` usando frecuencia inversa sobre el split de entrenamiento. Compensa el desbalance severo entre clases (~53% NV → ~0.9% VASC). El peso de la clase UNK (índice 8) se fija en 1.0. El `label_smoothing=0.1` reduce overconfidence.

### 6.2 Métricas

| Métrica | Uso | Justificación |
|---|---|---|
| **BMCA** | Evaluación principal (métrica oficial ISIC 2019) | `balanced_accuracy_score` de scikit-learn. Promedio de recall por clase, compensando el desbalance automáticamente |
| **AUC-ROC por clase** | Evaluación secundaria | Capacidad discriminativa por clase, independiente del threshold |
| **Accuracy** | **NUNCA** | ~53% prediciendo siempre "NV" — métrica engañosa para este dataset |

> **Nota:** BMCA = Balanced Multi-Class Accuracy = `sklearn.metrics.balanced_accuracy_score`. Es el promedio del recall de cada clase, equivalente a la macro-average del recall. Es la métrica oficial del challenge ISIC 2019.

---

## 7. Checkpoint

| Campo | Valor |
|---|---|
| **Directorio** | `checkpoints/expert_01_efficientnet_b3/` |
| **Archivo** | `expert2_best.pt` |
| **Ruta completa** | `checkpoints/expert_01_efficientnet_b3/expert2_best.pt` |
| **Criterio de guardado** | Mejor `val_loss` durante el entrenamiento |

---

## 8. Cómo ejecutar

### 8.1 Entrenamiento real (cuando los datos estén disponibles)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/train_expert2.py
```

### 8.2 Dry-run (verificación sin datos)

```bash
PYENV_VERSION=3.12.3 python src/pipeline/fase2/train_expert2.py --dry-run
```

El modo `--dry-run` ejecuta:

1. Importación de todos los módulos (config, modelo, dataloader)
2. Instanciación del modelo `Expert2ConvNeXtSmall`
3. Forward pass sintético con un batch de 2 imágenes aleatorias `[2, 3, 224, 224]`
4. Verificación de la shape de salida `(2, 9)`
5. Conteo de parámetros
6. Falla únicamente si los datos no están disponibles en disco (comportamiento esperado)

---

## 9. Estado

| Verificación | Resultado |
|---|---|
| Importaciones | ✅ OK |
| Forward pass del modelo | ✅ OK — shape `(2, 9)` (ConvNeXt-Small con 9 salidas) |
| Conteo de parámetros | ✅ ~50M |
| Dry-run | ✅ Pasa — falla solo en datos ausentes (esperado) |
| Entrenamiento real | ⏳ Pendiente de datos |

---

## 10. Restricciones aplicadas

| Restricción | Detalle |
|---|---|
| **Sin pesos preentrenados** | `timm.create_model("convnext_small", pretrained=False)` — entrenamiento from scratch obligatorio |
| **CutMix/MixUp a nivel batch** | CutMix (p=0.3) y MixUp (p=0.2) operan a nivel de batch en el training loop, no como transforms de imagen. La región recortada en CutMix se reemplaza con contenido de otra imagen (no se elimina información) |
| **CoarseDropout como regularización** | Reemplaza la función de Cutout con relleno mean ImageNet (no negro). 1–3 parches de 32–96px |

---

## 11. Dependencias clave

| Librería | Uso |
|---|---|
| `timm` | Backbone ConvNeXt-Small (`timm.create_model("convnext_small", pretrained=False)`) |
| `torchvision` | Transforms (`RandomCrop`, `ColorJitter`, `Normalize`, etc.) |
| `torch.amp` | Mixed precision (FP16) con `autocast` + `GradScaler` |
| `sklearn.metrics` | `balanced_accuracy_score` para BMCA |
| `PIL` (Pillow) | Carga de imágenes en `ISICDataset` |

---

*Documento generado el 2026-04-05, actualizado el 2026-04-14 (migración a ConvNeXt-Small from scratch). Fuentes: `src/pipeline/fase2/expert2_config.py`, `models/expert2_convnext.py`, `dataloader_expert2.py`, `train_expert2.py`, `src/pipeline/fase0/pre_isic.py`, `src/pipeline/datasets/isic.py`, `src/pipeline/fase2/losses.py`, `proyecto_moe.md` §5.2.*

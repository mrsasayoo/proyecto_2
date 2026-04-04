# Transformaciones de Datasets — Fase 1

**Fecha:** 1 de abril de 2026
**Propósito:** Documentar el pipeline de transformación aplicado a cada dataset durante la Fase 1 del proyecto MoE para clasificación de imágenes médicas.

Este documento registra de forma exhaustiva las transformaciones aplicadas a cada uno de los cinco datasets (uno por experto) en la Fase 1 del proyecto Mixture of Experts (MoE). Para cada dataset se especifica el formato de entrada, las operaciones aplicadas con sus parámetros exactos, el orden de ejecución, las diferencias entre modos de operación y la forma final del tensor de salida. El objetivo es garantizar la reproducibilidad completa de los embeddings generados y servir como referencia canónica para cualquier regeneración futura.

---

## 1. Constantes globales

Las siguientes constantes están definidas en `fase1_config.py` y se utilizan de forma transversal en los pipelines de transformación:

| Constante | Valor | Descripción |
|---|---|---|
| `IMG_SIZE` | `224` | Tamaño final de imagen (píxeles) |
| `CLAHE_CLIP_LIMIT` | `2.0` | Límite de clip para CLAHE |
| `CLAHE_TILE_GRID` | `(8, 8)` | Grid de tiles para CLAHE |
| `GAMMA_VALUE` | `1.0` | Valor de corrección gamma (identidad) |
| `TVF_WEIGHT` | `10.0` | Peso del término TVF |
| `TVF_ITER` | `30` | Iteraciones del filtro TVF |
| `HU_LUNG_CLIP` | `(-1000, 400)` | Rango HU para pulmón (LUNA16) |
| `HU_ABDOMEN_CLIP` | `(-100, 400)` | Rango HU para abdomen (Páncreas) |
| `VOLUME_SIZE_3D` | `(64, 64, 64)` | Tamaño de volumen 3D tras resize |
| `NORMALIZE_MEAN` | `[0.485, 0.456, 0.406]` | Media ImageNet (normalización final) |
| `NORMALIZE_STD` | `[0.229, 0.224, 0.225]` | Desv. estándar ImageNet (normalización final) |

---

## 2. Pipeline por dataset

---

### Expert 0 — NIH ChestXray14 (`chest.py`)

**Dataset:** NIH ChestXray14 (radiografías de tórax)
**Formato de entrada:** PNG, resolución variable, RGB o escala de grises
**Tensor de salida:** `[3, 224, 224]`

Este pipeline utiliza directamente la función `build_2d_transform()` definida en `transform_2d.py`.

#### Tabla de transformaciones

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 1 | `CLAHE` | `clip_limit=2.0`, `tile_grid_size=(8, 8)` | Se aplica PRIMERO, a resolución original (antes de cualquier resize) |
| 2 | `Resize` | `size=224`, `interpolation=BILINEAR` | Redimensiona la imagen al tamaño estándar |
| 3 | `TVF filter` | `weight=10.0`, `iterations=30` | Filtro de variación total para suavizado |
| 4 | `GammaCorrection` | `γ=1.0` | Identidad (sin efecto), se mantiene por reproducibilidad |
| 5 | `ToTensor` | — | Convierte a tensor float en rango `[0, 1]` |
| 6 | `Normalize` | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` | Normalización con estadísticas de ImageNet |

#### Diagrama del pipeline

```
┌────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌───────────────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────────────────┐    ┌────────────┐
│ Input PNG  │ →  │ CLAHE 2.0 / 8×8  │ →  │ Resize 224  │ →  │ TVF w=10.0 n=30   │ →  │ Gamma γ=1.0  │ →  │ ToTensor │ →  │ Normalize ImageNet  │ →  │ [3,224,224]│
│ (variable) │    │ (resolución orig) │    │ (BILINEAR)  │    │ (suavizado)       │    │ (identidad)  │    │ [0,1]    │    │ mean/std ImageNet   │    │            │
└────────────┘    └──────────────────┘    └─────────────┘    └───────────────────┘    └──────────────┘    └──────────┘    └─────────────────────┘    └────────────┘
```

#### Modo de operación

Modo único. Utiliza `build_2d_transform()` directamente sin variaciones.

---

### Expert 1 — ISIC 2019 (`isic.py`)

**Dataset:** ISIC 2019 (imágenes de dermatoscopia)
**Formato de entrada:** JPG, resolución variable, RGB
**Tensor de salida:** `[3, 224, 224]`

Este dataset tiene **dos modos de operación** con pipelines distintos.

#### Modo Embedding (Fase 1 — extracción de backbone)

Utiliza el transform interno `tfs["embedding"]`. **NO** usa `build_2d_transform()`.

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 1 | `CircularCrop` | `thresh=10`, `guard=95%` | Elimina artefactos de borde oscuro circular típicos de dermatoscopia |
| 2 | `Resize` | `size=224` | Redimensiona al tamaño estándar |
| 3 | `ToTensor` | — | Convierte a tensor float en rango `[0, 1]` |
| 4 | `Normalize` | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` | Normalización con estadísticas de ImageNet |

> **Importante:** En modo embedding NO se aplican TVF, CLAHE ni Gamma.

#### Modo Expert (Fase 2+ — entrenamiento del experto)

Utiliza `_make_isic_transform()` con data augmentations adicionales específicas del dominio.

#### Diagrama del pipeline (modo embedding)

```
┌─────────────┐    ┌──────────────────────────┐    ┌─────────────┐    ┌──────────┐    ┌─────────────────────┐    ┌────────────┐
│ Input JPG   │ →  │ CircularCrop             │ →  │ Resize 224  │ →  │ ToTensor │ →  │ Normalize ImageNet  │ →  │ [3,224,224]│
│ (RGB, var.) │    │ thresh=10, guard=95%     │    │             │    │ [0,1]    │    │ mean/std ImageNet   │    │            │
└─────────────┘    └──────────────────────────┘    └─────────────┘    └──────────┘    └─────────────────────┘    └────────────┘
```

---

### Expert 2 — Osteoarthritis Knee (`osteoarthritis.py`)

**Dataset:** Osteoarthritis Initiative — Radiografías de rodilla
**Formato de entrada:** Imágenes de rayos X (varios formatos), resolución variable, escala de grises
**Tensor de salida:** `[3, 224, 224]`

El pipeline está definido **inline** dentro de `__getitem__`. **NO** usa `build_2d_transform()`.

#### Tabla de transformaciones

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 1 | `CLAHE` | `clip_limit=2.0`, `tile_grid_size=(8, 8)` | Se aplica a resolución original |
| 2 | `Resize` | `size=224`, `interpolation=BICUBIC` | **BICUBIC** (no BILINEAR como NIH) — preserva detalle óseo |
| 3 | `ToTensor` | — | Convierte a tensor float en rango `[0, 1]` |
| 4 | `Normalize` | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` | Normalización con estadísticas de ImageNet |

> **Importante:** NO se aplican TVF ni Gamma en este pipeline.

#### Diagrama del pipeline

```
┌────────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────────────────┐    ┌────────────┐
│ Input X-ray    │ →  │ CLAHE 2.0 / 8×8  │ →  │ Resize 224   │ →  │ ToTensor │ →  │ Normalize ImageNet  │ →  │ [3,224,224]│
│ (gris, var.)   │    │ (resolución orig) │    │ (BICUBIC)    │    │ [0,1]    │    │ mean/std ImageNet   │    │            │
└────────────────┘    └──────────────────┘    └──────────────┘    └──────────┘    └─────────────────────┘    └────────────┘
```

---

### Expert 3 — LUNA16 Lung Nodules (`luna.py`)

**Dataset:** LUNA16 (nódulos pulmonares, volúmenes 3D)
**Formato de entrada:** `.npy` (parches pre-extraídos de volúmenes DICOM/mhd)
**Tensor de salida:** `[3, 224, 224]`

Este pipeline opera en **dos tiempos**: extracción (offline, una sola vez) y DataLoader (en cada batch).

#### Tiempo de extracción (pre-almacenado en `.npy`)

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| E1 | `HU Clip` | `rango=(-1000, 400)` | Recorte de unidades Hounsfield para ventana pulmonar |
| E2 | `Rescale` | `[0, 1]` | Normalización al rango unitario |

Los archivos `.npy` resultantes ya contienen los parches con HU clipeado y normalizado.

#### Tiempo de DataLoader (cada batch)

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 1 | `Load .npy` | — | Carga el parche 3D (ya HU-clipeado y normalizado) |
| 2 | `resize_volume_3d` | `target=(64, 64, 64)` | Resize trilineal del volumen 3D |
| 3 | `Extracción de 3 cortes centrales` | `índice=32` | Axial, coronal y sagital en el centro del volumen |
| 4 | `Stack` | — | Apilado como `[3, 64, 64]` (pseudo-RGB) |
| 5 | `F.interpolate` | `mode=bilinear`, `target=(224, 224)` | Escalado 2D de `[3, 64, 64]` a `[3, 224, 224]` |
| 6 | `Normalize` | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` | Normalización con estadísticas de ImageNet |

#### Distribución de parches

| Split | Cantidad de archivos |
|-------|---------------------|
| `patches/train/` | 14,728 |
| `patches/val/` | 1,143 |
| `patches/test/` | 1,914 |

#### Diagrama del pipeline

```
=== Tiempo de extracción (offline, una sola vez) ===

┌──────────────────┐    ┌──────────────────────┐    ┌───────────────┐    ┌──────────────┐
│ Volumen DICOM/   │ →  │ HU Clip              │ →  │ Rescale [0,1] │ →  │ .npy patch   │
│ mhd (raw)        │    │ (-1000, 400)          │    │               │    │ (almacenado) │
└──────────────────┘    └──────────────────────┘    └───────────────┘    └──────────────┘

=== Tiempo de DataLoader (cada batch) ===

┌──────────────┐    ┌──────────────────────┐    ┌────────────────────────────┐    ┌──────────────┐    ┌────────────────────┐    ┌─────────────────────┐    ┌────────────┐
│ Load .npy    │ →  │ resize_volume_3d     │ →  │ 3 cortes centrales (idx=32)│ →  │ Stack        │ →  │ F.interpolate      │ →  │ Normalize ImageNet  │ →  │ [3,224,224]│
│ patch 3D     │    │ target=(64,64,64)    │    │ axial, coronal, sagital    │    │ [3, 64, 64]  │    │ bilinear→(224,224) │    │ mean/std ImageNet   │    │            │
└──────────────┘    └──────────────────────┘    └────────────────────────────┘    └──────────────┘    └────────────────────┘    └─────────────────────┘    └────────────┘
```

---

### Expert 4 — Pancreas CT NIfTI (`pancreas.py`)

**Dataset:** Medical Segmentation Decathlon — Páncreas (volúmenes CT)
**Formato de entrada:** `.nii.gz` (NIfTI), 557 archivos, ~93 GB en total
**Tensor de salida:** `[3, 224, 224]`

Todo el procesamiento ocurre en **tiempo de DataLoader** (no hay pre-extracción).

#### Tabla de transformaciones

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 1 | `ReadNIfTI` | — | Lectura del volumen crudo en unidades Hounsfield |
| 2 | `HU Clip` | `rango=(-100, 400)` | Ventana abdominal (`HU_ABDOMEN_CLIP`) |
| 3 | `Z-score normalization` | `por volumen` | Media y desviación estándar calculadas por volumen individual |
| 4 | `Rescale` | `[0, 1]` | Normalización al rango unitario |
| 5 | `resize_volume_3d` | `target=(64, 64, 64)` | Resize trilineal del volumen 3D |
| 6 | `Extracción de 3 cortes centrales` | `índice=32` | Axial, coronal y sagital en el centro del volumen |
| 7 | `Stack` | — | Apilado como `[3, 64, 64]` (pseudo-RGB) |
| 8 | `F.interpolate` | `mode=bilinear`, `target=(224, 224)` | Escalado 2D de `[3, 64, 64]` a `[3, 224, 224]` |
| 9 | `Normalize` | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` | Normalización con estadísticas de ImageNet |

#### Modo Expert ROI (Opción B)

En modo expert (Fase 2+), se añade un recorte axial `Z[120:220]` **ANTES** del resize 3D. Esto focaliza el volumen en la región de interés del páncreas, descartando cortes superiores e inferiores irrelevantes.

#### Diagrama del pipeline (modo embedding)

```
┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌───────────────┐    ┌──────────────────────┐    ┌────────────────────────────┐
│ .nii.gz      │ →  │ HU Clip          │ →  │ Z-score norm      │ →  │ Rescale [0,1] │ →  │ resize_volume_3d     │ →  │ 3 cortes centrales (idx=32)│ → ...
│ (ReadNIfTI)  │    │ (-100, 400)      │    │ (por volumen)     │    │               │    │ target=(64,64,64)    │    │ axial, coronal, sagital    │
└──────────────┘    └──────────────────┘    └───────────────────┘    └───────────────┘    └──────────────────────┘    └────────────────────────────┘

    ┌──────────────┐    ┌────────────────────┐    ┌─────────────────────┐    ┌────────────┐
... │ Stack        │ →  │ F.interpolate      │ →  │ Normalize ImageNet  │ →  │ [3,224,224]│
    │ [3, 64, 64]  │    │ bilinear→(224,224) │    │ mean/std ImageNet   │    │            │
    └──────────────┘    └────────────────────┘    └─────────────────────┘    └────────────┘
```

#### Diagrama del pipeline (modo expert, Opción B con ROI)

```
┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌───────────────┐    ┌─────────────────┐    ┌──────────────────────┐
│ .nii.gz      │ →  │ HU Clip          │ →  │ Z-score norm      │ →  │ Rescale [0,1] │ →  │ ROI Z[120:220]  │ →  │ resize_volume_3d     │ → ... (continúa igual)
│ (ReadNIfTI)  │    │ (-100, 400)      │    │ (por volumen)     │    │               │    │ (crop axial)    │    │ target=(64,64,64)    │
└──────────────┘    └──────────────────┘    └───────────────────┘    └───────────────┘    └─────────────────┘    └──────────────────────┘
```

---

## 3. Tabla comparativa consolidada

| Expert | Dataset | Formato entrada | CLAHE | TVF | Gamma | CircularCrop | HU Clip | Z-score | ROI crop | Interpolación resize | Tensor salida |
|--------|---------|-----------------|-------|-----|-------|--------------|---------|---------|----------|----------------------|---------------|
| 0 | NIH ChestXray14 | PNG | ✅ (2.0/8×8) | ✅ (10.0/30) | ✅ (1.0) | ❌ | ❌ | ❌ | ❌ | BILINEAR | `[3, 224, 224]` |
| 1 | ISIC 2019 | JPG | ❌ | ❌ | ❌ | ✅ (thresh=10) | ❌ | ❌ | ❌ | — | `[3, 224, 224]` |
| 2 | OA Knee | X-ray | ✅ (2.0/8×8) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | BICUBIC | `[3, 224, 224]` |
| 3 | LUNA16 | .npy (3D) | ❌* | ❌ | ❌ | ❌ | ✅ (−1000, 400)* | ❌ | ❌ | BILINEAR | `[3, 224, 224]` |
| 4 | Páncreas | .nii.gz (3D) | ❌ | ❌ | ❌ | ❌ | ✅ (−100, 400) | ✅ | Opcional (B) | BILINEAR | `[3, 224, 224]` |

> \*LUNA16: El HU clip se aplica en **tiempo de extracción** y queda pre-almacenado en los archivos `.npy`, no en tiempo de DataLoader.

---

## 4. Visualización del pipeline por dataset

### Expert 0 — NIH ChestXray14

```
[Input PNG] → [CLAHE 2.0/8×8] → [Resize 224 BILINEAR] → [TVF w=10.0/n=30] → [Gamma γ=1.0] → [ToTensor] → [Normalize ImageNet] → [3,224,224]
```

### Expert 1 — ISIC 2019 (modo embedding)

```
[Input JPG] → [CircularCrop thresh=10 guard=95%] → [Resize 224] → [ToTensor] → [Normalize ImageNet] → [3,224,224]
```

### Expert 2 — OA Knee

```
[Input X-ray] → [CLAHE 2.0/8×8] → [Resize 224 BICUBIC] → [ToTensor] → [Normalize ImageNet] → [3,224,224]
```

### Expert 3 — LUNA16

```
Extracción:   [DICOM/mhd] → [HU Clip (-1000,400)] → [Rescale 0-1] → [.npy 64³]

DataLoader:   [Load .npy] → [resize_volume_3d (64,64,64)] → [3 cortes idx=32] → [Stack 3,64,64] → [F.interpolate bilinear 224×224] → [Normalize ImageNet] → [3,224,224]
```

### Expert 4 — Páncreas

```
DataLoader:   [ReadNIfTI] → [HU Clip (-100,400)] → [Z-score/vol] → [Rescale 0-1] → [resize_volume_3d (64,64,64)] → [3 cortes idx=32] → [Stack 3,64,64] → [F.interpolate bilinear 224×224] → [Normalize ImageNet] → [3,224,224]

Con ROI (B):  [ReadNIfTI] → [HU Clip (-100,400)] → [Z-score/vol] → [Rescale 0-1] → [ROI Z[120:220]] → [resize_volume_3d (64,64,64)] → [3 cortes idx=32] → [Stack 3,64,64] → [F.interpolate bilinear 224×224] → [Normalize ImageNet] → [3,224,224]
```

---

## 5. Notas de implementación y decisiones de diseño

### CLAHE se aplica ANTES del resize

CLAHE opera sobre la resolución original de la imagen antes de cualquier redimensionamiento. Esto preserva el contraste local a la resolución nativa, donde los detalles anatómicos tienen mayor densidad estadística en el histograma. Si se aplicara después del resize, la ecualización adaptativa trabajaría sobre una imagen ya degradada, perdiendo información diagnósticamente relevante. Esta decisión corresponde a la corrección del **BUG #1**.

### Extracción de 3 cortes centrales (pseudo-RGB) para volúmenes 3D

LUNA16 y Páncreas extraen tres cortes ortogonales del centro del volumen (axial, coronal, sagital en el índice 32) y los apilan como un tensor de 3 canales. Esta estrategia de pseudo-RGB permite:

- Reutilizar backbones pre-entrenados en ImageNet (que esperan 3 canales de entrada).
- Capturar información espacial tridimensional desde tres perspectivas complementarias.
- Evitar el coste computacional de procesar el volumen 3D completo con una red 3D durante la fase de embedding.

### ISIC usa CircularCrop en lugar de CLAHE

Las imágenes de dermatoscopia presentan bordes oscuros circulares (artefacto del dermatoscopio) que distorsionarían el histograma de CLAHE. `CircularCrop` con `thresh=10` y `guard=95%` elimina estos bordes antes de cualquier procesamiento, asegurando que solo la lesión y la piel circundante participen en la normalización posterior.

### OA Knee usa BICUBIC en lugar de BILINEAR

Las radiografías de rodilla contienen detalles finos en las superficies articulares y los márgenes óseos que son críticos para la clasificación de osteoartritis. La interpolación bicúbica preserva mejor estos bordes y detalles de alta frecuencia que la bilineal, a un coste computacional marginalmente mayor.

### `build_2d_transform()` no es utilizado por OA ni ISIC

Estos dos datasets tienen requisitos específicos de dominio que no se ajustan al pipeline genérico de `build_2d_transform()`:

- **ISIC** necesita `CircularCrop` (no CLAHE) y no requiere TVF ni Gamma.
- **OA Knee** necesita interpolación BICUBIC y no requiere TVF ni Gamma. Además, define su pipeline inline en `__getitem__()`.

Por ello, ambos implementan sus transformaciones fuera de la cadena estándar.

### Deprecación de `preprocessing.py::build_2d_transform()`

Como parte de la corrección del **BUG #3**, la función `build_2d_transform()` en `preprocessing.py` ha sido marcada como **deprecated**. Ahora delega internamente a `fase1.transform_2d.build_2d_transform`, que contiene la implementación canónica y corregida. Todo código nuevo debe importar directamente desde `fase1.transform_2d`.

---

## 6. Estado de embeddings

> ⚠️ **ADVERTENCIA:** A la fecha de redacción de este documento, todos los embeddings existentes están **OBSOLETOS (STALE)** y deben ser regenerados con la bandera `--force`.

### Motivos de invalidación

| Bug | Descripción | Datasets afectados | Impacto en valores |
|-----|-------------|-------------------|--------------------|
| **BUG #1** | Corrección del orden de CLAHE (ahora se aplica antes del resize) | NIH ChestXray14 (Expert 0), OA Knee (Expert 2) | Cambia valores de embeddings |
| **BUG #2** | Corrección del guard para `test/` en LUNA16 | LUNA16 (Expert 3) | Corrección estructural, no cambia valores de embeddings |
| **BUG #3** | Deprecación de `preprocessing.py` | Todos | Corrección estructural |

### Comando de regeneración

Ejecutar para **cada backbone**:

```bash
/home/mrsasayo_mesa/venv_global/bin/python src/pipeline/fase1/fase1_pipeline.py \
    --backbone <BACKBONE_NAME> --output_dir embeddings/<BACKBONE_NAME> \
    --batch_size 256 --workers 4 --force
```

### Backbones a regenerar

| Backbone | Identificador |
|----------|---------------|
| ViT Tiny | `vit_tiny_patch16_224` |
| DenseNet-121 | `densenet121_custom` |
| Swin Tiny | `swin_tiny_patch4_window7_224` |
| CvT-13 | `cvt_13` |

---

*Documento generado como referencia canónica de las transformaciones de Fase 1. Cualquier modificación a los pipelines debe reflejarse en este documento antes de regenerar embeddings.*

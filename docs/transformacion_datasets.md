# Transformaciones de Datasets — Fase 1 y Entrenamiento de Expertos

**Fecha:** 14 de abril de 2026
**Propósito:** Documentar el pipeline de transformación aplicado a cada dataset durante la Fase 1 (backbone/embedding) y el entrenamiento de expertos del proyecto MoE para clasificación de imágenes médicas.

Este documento registra de forma exhaustiva las transformaciones aplicadas a cada uno de los cinco datasets (uno por experto) en el proyecto Mixture of Experts (MoE). Para cada dataset se especifica el formato de entrada, las operaciones aplicadas con sus parámetros exactos, el orden de ejecución, las diferencias entre modos de operación y la forma final del tensor de salida. Cada sección diferencia claramente entre el **pipeline de Fase 1 (embedding/backbone)** y el **pipeline de entrenamiento de experto ("Lista Maestra")**. El objetivo es garantizar la reproducibilidad completa de los embeddings y del entrenamiento de expertos, y servir como referencia canónica para cualquier regeneración futura.

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
| `HU_ABDOMEN_CLIP` | `(-100, 400)` | Rango HU para abdomen — Fase 1 embedding |
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

#### Tabla de transformaciones (`build_2d_transform()`)

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 1 | `CLAHE` | `clip_limit=2.0`, `tile_grid_size=(8, 8)` | Se aplica PRIMERO, a resolución original (antes de cualquier resize) |
| 2 | `Resize` | `size=224`, `interpolation=BILINEAR` | Redimensiona la imagen al tamaño estándar |
| 3 | `TVF filter` | `weight=10.0`, `iterations=30` | Filtro de variación total para suavizado |
| 4 | `ToTensor` | — | Convierte a tensor float en rango `[0, 1]` |
| 5 | `Normalize` | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` | Normalización con estadísticas de ImageNet |

> **Nota (fix INC-04, 2026-04-05):** `GammaCorrection` con `γ=1.0` fue eliminado del pipeline. Cuando `gamma=1.0` la corrección es una operación identidad (sin efecto), por lo que `build_2d_transform()` la omite (`if gamma != 1.0`). Solo se incluye cuando se pasa un valor de gamma diferente de 1.0.

#### Diagrama del pipeline

```
┌────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌───────────────────┐    ┌──────────┐    ┌─────────────────────┐    ┌────────────┐
│ Input PNG  │ →  │ CLAHE 2.0 / 8×8  │ →  │ Resize 224  │ →  │ TVF w=10.0 n=30   │ →  │ ToTensor │ →  │ Normalize ImageNet  │ →  │ [3,224,224]│
│ (variable) │    │ (resolución orig) │    │ (BILINEAR)  │    │ (suavizado)       │    │ [0,1]    │    │ mean/std ImageNet   │    │            │
└────────────┘    └──────────────────┘    └─────────────┘    └───────────────────┘    └──────────┘    └─────────────────────┘    └────────────┘
```

#### Modo de operación

- **Validación / Test (embedding):** Utiliza `build_2d_transform()` directamente sin augmentaciones.
- **Entrenamiento:** Utiliza `build_2d_aug_transform()`, que extiende el pipeline base con augmentaciones de datos entre Resize y TVF.

#### Pipeline de entrenamiento — `build_2d_aug_transform()`

La función `build_2d_aug_transform()` (definida en `transform_2d.py`) construye un pipeline aumentado para entrenamiento de chest X-ray. Es idéntico a `build_2d_transform()` pero inserta tres augmentaciones entre Resize y TVF. Las transformaciones basadas en oclusión (RandomErasing, CutMix, etc.) están **prohibidas**.

| Paso | Operación | Parámetros | Justificación clínica |
|------|-----------|------------|----------------------|
| 1 | `CLAHE` | `clip_limit=2.0`, `tile_grid_size=(8, 8)` | Resolución original, antes del resize |
| 2 | `Resize` | `size=224`, `interpolation=BILINEAR` | Tamaño estándar |
| 3 | `RandomHorizontalFlip` | `p=0.5` | El tórax tiene simetría anatómica izquierda-derecha |
| 4 | `RandomRotation` | `degrees=10` | Simula variación de posicionamiento del paciente |
| 5 | `ColorJitter` | `brightness=0.2`, `contrast=0.2` | Simula variabilidad entre diferentes equipos de rayos X |
| 6 | `TVF filter` | `weight=10.0`, `iterations=30` | Denoising preservando bordes |
| 7 | `ToTensor` | — | Convierte a tensor float en rango `[0, 1]` |
| 8 | `Normalize` | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` | Normalización ImageNet |

> **Nota:** Al igual que en `build_2d_transform()`, GammaCorrection solo se incluye si `γ ≠ 1.0`. Con el valor por defecto (`DEFAULT_GAMMA = 1.0`), no se aplica.

##### Diagrama del pipeline de entrenamiento

```
┌────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌───────────────────────────────┐    ┌───────────────────┐    ┌──────────┐    ┌─────────────────────┐    ┌────────────┐
│ Input PNG  │ →  │ CLAHE 2.0 / 8×8  │ →  │ Resize 224  │ →  │ HFlip p=0.5  │ →  │ Rot ±10°      │ →  │ ColorJitter bright=0.2 c=0.2  │ →  │ TVF w=10.0 n=30   │ →  │ ToTensor │ →  │ Normalize ImageNet  │ →  │ [3,224,224]│
│ (variable) │    │ (resolución orig) │    │ (BILINEAR)  │    │              │    │               │    │                               │    │ (suavizado)       │    │ [0,1]    │    │ mean/std ImageNet   │    │            │
└────────────┘    └──────────────────┘    └─────────────┘    └──────────────┘    └───────────────┘    └───────────────────────────────┘    └───────────────────┘    └──────────┘    └─────────────────────┘    └────────────┘
```

#### Entrenamiento del Experto — Lista Maestra (Martín — ConvNeXt-Tiny)

El pipeline de Fase 1 (embedding/backbone) ya documentado arriba permanece sin cambios. A continuación se documenta la **Lista Maestra definitiva** para el entrenamiento del Expert 0.

##### FASE 1 OFFLINE — `_preload()` (una sola vez → RAM)

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 1 | `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` | monocanal | ~26 GB vs ~80 GB en RGB |
| 2 | `clahe.apply(img)` | `clipLimit=2.0`, `tileGridSize=(8,8)` | Sobre imagen alta resolución (1024px) |
| 3 | `multistage_resize(img, target=224)` | `INTER_AREA`, halvings iterativos | 1024→~512→224, anti-aliasing |
| 4 | _(en `__getitem__`)_ `cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)` | → 3 canales | NO cacheado (ahorra 2× RAM) |

##### FASE 2 ONLINE — Solo train (Albumentations)

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 5 | `A.HorizontalFlip(p=0.5)` | — | — |
| 6 | `A.RandomBrightnessContrast(p=0.5, limits=±0.1)` | — | Simula variación inter-escáner |
| 7 | `A.RandomGamma(gamma_limit=(85, 115), p=0.5)` | — | Curvas de respuesta distintas |
| 8 | `A.GaussNoise(var_limit=(1e-4, 4e-4), p=0.1)` | — | Ruido cuántico portátiles AP |
| 9 | `A.Normalize(mean=MODEL_MEAN, std=MODEL_STD)` | — | Resueltos via `timm.data.resolve_data_config()` |
| 10 | `ToTensorV2()` | — | — |

##### Inferencia (val/test)

Pasos 4 → 9 → 10 (sin pasos 5–8).

##### TTA

Promedio `logits(original)` + `logits(HorizontalFlip)` → `eval_with_tta()`.

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

> **Nota de arquitectura:** Experto Isabella, ConvNeXt-Small desde cero (`weights=None`), ~50M params.

#### Entrenamiento del Experto — Lista Maestra (Isabella — ConvNeXt-Small)

##### FASE 1 OFFLINE (caché en disco — determinista — una sola vez)

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 1 | Auditoría del dataset | → CSV con resoluciones, fuentes y artefactos | — |
| 2 | Eliminación de vello | DullRazor o CNN-VAE | — |
| 3 | Color Constancy | Shades of Gray (p=6) | — |
| 4 | Resize aspect-ratio-preserving | lado corto = `target_size` (LANCZOS4) | — |
| 5 | Guardado en caché | `{isic_id}_cc_{size}.jpg` (calidad JPEG 95) | — |

##### FASE 2 ONLINE / ON-THE-FLY (solo train — estocástica — por epoch)

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 6 | `WeightedRandomSampler` | peso ∝ 1/freq_clase | Batch balanceado |
| 7 | `RandomCrop(target_size × target_size)` | — | Recorte cuadrado |
| 8 | `RandomHorizontalFlip(p=0.5)` + `RandomVerticalFlip(p=0.5)` | — | — |
| 9 | `RandomRotation([0°, 360°])` | bilineal, fill=reflect | — |
| 10 | `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` | — | — |
| 11 | `RandomGamma(γ ∈ [0.7, 1.5], p=0.5)` | — | — |
| 12 | `CoarseDropout / CutOut` | 1–3 parches 32–96px, p=0.5 | — |
| 13 | `CutMix(p=0.3)` o `MixUp(p=0.2)` | — | Mezcla de batch completo |
| 14 | `ToTensor()` | → [0.0, 1.0] | — |
| 15 | `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` | — | — |

##### FASE 3 INFERENCIA (val/test)

- `CenterCrop(target_size)` → `ToTensor()` → `Normalize()`
- TTA opcional: 8 variantes (flips × rotaciones 0°/90°/180°/270°)

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

> **Nota de arquitectura:** Experto Isabella, EfficientNet-B3. Cabeza: `Dropout(0.4) → Linear(1536, 5)` (B3 feature dim = 1536).

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

#### Entrenamiento del Experto — Lista Maestra (Isabella — EfficientNet-B3)

El pipeline de transforms del experto ya está validado y **NO cambia**:

| Modo | Pipeline |
|------|----------|
| **Train** | `Resize(256,256)` → `RandomCrop(224)` → `HFlip(0.5)` → `Rotation(±15°)` → `ColorJitter(0.3,0.3)` → `RandomAutocontrast(0.3)` → `ToTensor` → `Normalize(ImageNet)` |
| **Val** | `Resize(224,224)` → `ToTensor` → `Normalize(ImageNet)` |

---

### Expert 3 — LUNA16 Lung Nodules (`luna.py`)

**Dataset:** LUNA16 (nódulos pulmonares, volúmenes 3D)
**Formato de entrada:** `.npy` (parches pre-extraídos de volúmenes DICOM/mhd)
**Tensor de salida Fase 1:** `[3, 224, 224]`

> **Nota de arquitectura:** Experto Nicolás, DenseNet 3D (desde cero).

Este pipeline opera en **dos tiempos**: extracción (offline, una sola vez) y DataLoader (en cada batch).

#### Pipeline de Fase 1 (Embedding / Backbone)

##### Tiempo de extracción (pre-almacenado en `.npy`)

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| E1 | `HU Clip` | `rango=(-1000, 400)` | Recorte de unidades Hounsfield para ventana pulmonar |
| E2 | `Rescale` | `[0, 1]` | Normalización al rango unitario |

Los archivos `.npy` resultantes ya contienen los parches con HU clipeado y normalizado.

##### Tiempo de DataLoader (cada batch)

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

#### Diagrama del pipeline (Fase 1)

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

#### Entrenamiento del Experto — Lista Maestra (Nicolás — DenseNet 3D)

**Tensor de salida del experto:** `[B, 1, 64, 64, 64]` — parche CT centrado en candidato a nódulo.

##### FASE 1 OFFLINE (todo el dataset — una sola vez)

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 1 | Carga `.mhd/.raw` | → conversión a HU (slope/intercept) | — |
| 2 | Corrección píxeles fuera de FOV | −2000 → 0 | — |
| 3 | Remuestreo isotrópico | → 1×1×1 mm³ | — |
| 4 | Segmentación pulmonar | máscara 3D + dilatación morfológica | — |
| 5 | Clipping HU | → [−1000, +400] | — |
| 6 | Normalización | → [0.0, 1.0] | — |
| 7 | Zero-centering | restar media global (≈ 0.25 en LUNA16) | — |
| 8 | Extracción de parches 3D | 64³ o 128³ vóxeles → `.npy` | — |

##### FASE 2 ONLINE / ON-THE-FLY (solo entrenamiento — cada epoch)

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 9 | Oversampling de muestras positivas | ratio ~1:10 | — |
| 10 | Flips aleatorios | ejes X, Y, Z (p=0.5 c/u) | — |
| 11 | Rotaciones 3D | continuas (±15°) o discretas (90°/180°/270°) | — |
| 12 | Escalado uniforme aleatorio | [0.8, 1.2] × tamaño | — |
| 13 | Traslación aleatoria | ±3–5 mm por eje | — |
| 14 | Deformación elástica 3D | σ=1–3, α=0–5 mm | — |
| 15 | Ruido Gaussiano + ajuste de brillo/contraste | — | — |

---

### Expert 4 — Pancreas / PANORAMA (`pancreas.py`)

**Dataset:** PANORAMA Challenge (Zenodo registros 13715870, 13742336, 11034011, 10999754) — Volúmenes CT abdominales
**Formato de entrada:** `.nii.gz` (NIfTI)
**Tensor de salida Fase 1:** `[3, 224, 224]`

> **Nota de arquitectura:** Experta Luz, ResNet 3D (desde cero).

#### Pipeline de Fase 1 (Embedding / Backbone)

Todo el procesamiento ocurre en **tiempo de DataLoader** (no hay pre-extracción).

##### Tabla de transformaciones

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

##### Modo Expert ROI (Opción B)

En modo expert (Fase 2+), se añade un recorte axial `Z[120:220]` **ANTES** del resize 3D. Esto focaliza el volumen en la región de interés del páncreas, descartando cortes superiores e inferiores irrelevantes.

##### Diagrama del pipeline (modo embedding)

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

##### Diagrama del pipeline (modo expert, Opción B con ROI)

```
┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌───────────────┐    ┌─────────────────┐    ┌──────────────────────┐
│ .nii.gz      │ →  │ HU Clip          │ →  │ Z-score norm      │ →  │ Rescale [0,1] │ →  │ ROI Z[120:220]  │ →  │ resize_volume_3d     │ → ... (continúa igual)
│ (ReadNIfTI)  │    │ (-100, 400)      │    │ (por volumen)     │    │               │    │ (crop axial)    │    │ target=(64,64,64)    │
└──────────────┘    └──────────────────┘    └───────────────────┘    └───────────────┘    └─────────────────┘    └──────────────────────┘
```

#### Entrenamiento del Experto — Lista Maestra (Luz — ResNet 3D)

**Tensor de salida del experto:** `[B, 1, 64, 64, 64]` — parche CT abdominal.

##### FASE 1 OFFLINE (dataset completo — determinista — una vez)

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 1 | Carga `.nii.gz` | verificación orientación/dirección + alineación máscara | — |
| 2 | Remuestreo isotrópico | → 0.8×0.8×0.8 mm³ (Bspline img / NN máscara) | — |
| 3 | Clipping HU | → [−150, +250] para CECT abdominal/páncreas | — |
| 4 | [ETAPA COARSE] Segmentación del páncreas | nnUNet 3d_lowres → máscara ROI | — |
| 5 | Recorte al bounding box del páncreas | + margen de 20 mm por eje | — |
| 6 | Normalización Z-score | sobre foreground (percentil 0.5–99.5) | — |
| 7 | Extracción de parches 3D | 96³ o 192³ vóxeles → `.nii.gz` / `.npy` | — |

##### FASE 2 ONLINE / ON-THE-FLY (solo entrenamiento — estocástica — cada epoch)

| Paso | Operación | Parámetros | Notas |
|------|-----------|------------|-------|
| 8 | Oversampling de parches positivos (PDAC) | → 33% del batch | — |
| 9 | Flips aleatorios | ejes X, Y, Z (p=0.5 c/u) | — |
| 10 | Rotaciones 3D | ±30° en X/Y, ±180° en Z (interpolación trilineal) | — |
| 11 | Escalado uniforme aleatorio | [0.7, 1.4] × tamaño del parche | — |
| 12 | Deformación elástica 3D | σ=5–8, magnitud=50–150 vóxeles, p=0.2 | — |
| 13 | Ajuste Gamma + brillo | [0.7, 1.5] + multiplicación [0.75, 1.25] | — |
| 14 | Ruido Gaussiano aditivo | σ ∈ [0, 50 HU equivalente] | — |
| 15 | Blur axial / simulación de movimiento respiratorio | p=0.1 | — |

---

## 3. Tabla comparativa consolidada

| Expert | Dataset | Responsable | Arquitectura Experto | Formato entrada | CLAHE | TVF | Gamma | CircularCrop | HU Clip | Z-score | ROI crop | Interpolación resize | Tensor salida (Fase 1) |
|--------|---------|-------------|---------------------|-----------------|-------|-----|-------|--------------|---------|---------|----------|----------------------|------------------------|
| 0 | NIH ChestXray14 | Martín | ConvNeXt-Tiny | PNG | ✅ (2.0/8×8) | ✅ (10.0/30) | ❌ (γ=1.0, omitida) | ❌ | ❌ | ❌ | ❌ | BILINEAR | `[3, 224, 224]` |
| 1 | ISIC 2019 | Isabella | ConvNeXt-Small | JPG | ❌ | ❌ | ❌ | ✅ (thresh=10) | ❌ | ❌ | ❌ | — | `[3, 224, 224]` |
| 2 | OA Knee | Isabella | EfficientNet-B3 | X-ray | ✅ (2.0/8×8) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | BICUBIC | `[3, 224, 224]` |
| 3 | LUNA16 | Nicolás | DenseNet 3D | .npy (3D) | ❌* | ❌ | ❌ | ❌ | ✅ (−1000, 400)* | ❌ | ❌ | BILINEAR | `[3, 224, 224]` |
| 4 | PANORAMA (Páncreas) | Luz | ResNet 3D | .nii.gz (3D) | ❌ | ❌ | ❌ | ❌ | ✅ (−100, 400) | ✅ | Opcional (B) | BILINEAR | `[3, 224, 224]` |

> \*LUNA16: El HU clip se aplica en **tiempo de extracción** y queda pre-almacenado en los archivos `.npy`, no en tiempo de DataLoader.

---

## 4. Visualización del pipeline por dataset

### Expert 0 — NIH ChestXray14

```
Fase 1 Inf:   [Input PNG] → [CLAHE 2.0/8×8] → [Resize 224 BILINEAR] → [TVF w=10.0/n=30] → [ToTensor] → [Normalize ImageNet] → [3,224,224]

Fase 1 Train: [Input PNG] → [CLAHE 2.0/8×8] → [Resize 224 BILINEAR] → [HFlip p=0.5] → [Rot ±10°] → [ColorJitter b=0.2 c=0.2] → [TVF w=10.0/n=30] → [ToTensor] → [Normalize ImageNet] → [3,224,224]

Expert (Mar): [GRAYSCALE 1024px] → [CLAHE 2.0/8×8] → [multistage_resize 224] → ONLINE(train): [GRAY2RGB] → [HFlip] → [BrightContrast ±0.1] → [Gamma 85-115] → [GaussNoise] → [Normalize timm] → [ToTensorV2] → [3,224,224]
```

### Expert 1 — ISIC 2019

```
Fase 1 Emb:   [Input JPG] → [CircularCrop thresh=10 guard=95%] → [Resize 224] → [ToTensor] → [Normalize ImageNet] → [3,224,224]

Expert (Isa): [Audit] → [DullRazor] → [ColorConstancy SoG p=6] → [Resize LANCZOS4] → [cache JPG95] → ONLINE(train): [WRS] → [RandomCrop] → [HFlip+VFlip] → [Rot 0-360°] → [ColorJitter] → [Gamma] → [CutOut] → [CutMix/MixUp] → [ToTensor] → [Normalize ImageNet]
```

### Expert 2 — OA Knee

```
Fase 1:       [Input X-ray] → [CLAHE 2.0/8×8] → [Resize 224 BICUBIC] → [ToTensor] → [Normalize ImageNet] → [3,224,224]

Expert (Isa): Train: [Resize 256] → [RandomCrop 224] → [HFlip 0.5] → [Rot ±15°] → [ColorJitter 0.3,0.3] → [RandomAutocontrast 0.3] → [ToTensor] → [Normalize ImageNet]
              Val:   [Resize 224] → [ToTensor] → [Normalize ImageNet]
```

### Expert 3 — LUNA16

```
Fase 1 Ext:   [DICOM/mhd] → [HU Clip (-1000,400)] → [Rescale 0-1] → [.npy 64³]

Fase 1 DL:    [Load .npy] → [resize_volume_3d (64,64,64)] → [3 cortes idx=32] → [Stack 3,64,64] → [F.interpolate bilinear 224×224] → [Normalize ImageNet] → [3,224,224]

Expert (Nic): [.mhd/.raw] → [HU conv] → [FOV fix] → [resample 1³] → [lung seg] → [HU clip -1000,+400] → [norm 0-1] → [zero-center] → [parches 64³/128³] → ONLINE: [oversample 1:10] → [flips XYZ] → [rot 3D] → [scale 0.8–1.2] → [translate ±3-5mm] → [elastic 3D] → [noise+contrast] → [B,1,64,64,64]
```

### Expert 4 — Páncreas

```
Fase 1:       [ReadNIfTI] → [HU Clip (-100,400)] → [Z-score/vol] → [Rescale 0-1] → [resize_volume_3d (64,64,64)] → [3 cortes idx=32] → [Stack 3,64,64] → [F.interpolate bilinear 224×224] → [Normalize ImageNet] → [3,224,224]

Fase 1 ROI:   [ReadNIfTI] → [HU Clip (-100,400)] → [Z-score/vol] → [Rescale 0-1] → [ROI Z[120:220]] → [resize_volume_3d (64,64,64)] → [3 cortes idx=32] → [Stack 3,64,64] → [F.interpolate bilinear 224×224] → [Normalize ImageNet] → [3,224,224]

Expert (Luz): [.nii.gz] → [orient+align] → [resample 0.8³] → [HU clip -150,+250] → [nnUNet coarse] → [bbox+20mm] → [Z-score fg] → [parches 96³/192³] → ONLINE: [oversample 33%] → [flips XYZ] → [rot 3D] → [scale 0.7–1.4] → [elastic] → [gamma+bright] → [noise] → [blur axial] → [B,1,64,64,64]
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

### `build_2d_aug_transform()` para entrenamiento NIH

La función `build_2d_aug_transform()` extiende `build_2d_transform()` con augmentaciones de datos específicas para chest X-ray (RandomHorizontalFlip, RandomRotation, ColorJitter). Las augmentaciones se insertan entre Resize y TVF para operar sobre la imagen ya redimensionada pero antes del denoising. Las transformaciones basadas en oclusión están prohibidas por diseño.

### Deprecación de `preprocessing.py::build_2d_transform()`

Como parte de la corrección del **BUG #3**, la función `build_2d_transform()` en `preprocessing.py` ha sido marcada como **deprecated**. Ahora delega internamente a `fase1.transform_2d.build_2d_transform`, que contiene la implementación canónica y corregida. Todo código nuevo debe importar directamente desde `fase1.transform_2d`.

---

## 6. Estado de embeddings

> ⚠️ **ADVERTENCIA:** Todos los embeddings previamente generados fueron **ELIMINADOS** (directorio `embeddings/` vaciado desde BUG-C3). Deben ser regenerados desde cero con la bandera `--force`.

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

*Documento generado como referencia canónica de las transformaciones de Fase 1 y entrenamiento de expertos. Cualquier modificación a los pipelines debe reflejarse en este documento antes de regenerar embeddings o reentrenar expertos.*

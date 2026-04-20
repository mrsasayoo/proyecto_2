# Inventario Exhaustivo de Transformaciones Offline por Dataset

> Proyecto MoE Médico — Fase 0 (Preprocesamiento Offline)
>
> Generado: 2026-04-19 | Scripts fuente: `src/pipeline/fase0/`

---

## Tabla de Contenidos

1. [NIH ChestX-ray14](#1-nih-chestx-ray14)
2. [ISIC 2019](#2-isic-2019)
3. [LUNA16](#3-luna16)
4. [OsteoArthritis (Knee X-ray)](#4-osteoarthritis-knee-x-ray)
5. [Pancreas (PANORAMA/Zenodo)](#5-pancreas-panoramazenodo)
6. [Splits (todos los datasets)](#6-splits-todos-los-datasets)
7. [LUNA16 — Augmentación Offline](#7-luna16--augmentación-offline)
8. [LUNA16 — Auditoría de Patches](#8-luna16--auditoría-de-patches)
9. [Tabla Comparativa](#9-tabla-comparativa)

---

## 1. NIH ChestX-ray14

**Script:** `src/pipeline/fase0/pre_chestxray14.py` (388 líneas)

### Flujo de ejecución

```
imagen PNG original → carga grayscale → validación dimensiones → resize → CLAHE → float32 [0,1] → .npy + metadata CSV
```

### Transformaciones

| #  | Transformación | Función/Línea | Parámetros exactos | Input | Output |
|----|---------------|---------------|---------------------|-------|--------|
| 1  | Carga grayscale | `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` | — | PNG RGB/grayscale | uint8 2D array |
| 2  | Validación dimensiones | `h < MIN_DIM or w < MIN_DIM` | `MIN_DIM = 800` | uint8 array | skip si < 800px en cualquier eje |
| 3  | Resize | `cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)` | `TARGET_SIZE = 256` | uint8 (H×W) | uint8 (256×256) |
| 4  | CLAHE | `cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE).apply(img)` | `CLAHE_CLIP = 2.0`, `CLAHE_TILE = (8, 8)` | uint8 (256×256) | uint8 (256×256) |
| 5  | Normalización float32 | `img.astype(np.float32) / 255.0` | — | uint8 [0,255] | float32 [0,1] |
| 6  | Guardado | `np.save(out_path, arr)` | — | float32 (256×256) | `.npy` file |
| 7  | Metadata label | `_build_label_lookup()` parsing `Data_Entry_2017.csv` | 14 etiquetas multi-label, separadas por `\|` | CSV | vector binario 14-dim |
| 8  | Metadata CSV | escritura por split (train/val/test) | columnas: filename, sha256, 14 labels | — | `metadata_{split}.csv` |
| 9  | Stats JSON | `stats.json` con conteo por split | — | — | JSON |

### Constantes

```python
LABELS_14 = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
             "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
             "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
TARGET_SIZE = 256
MIN_DIM = 800
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
```

### Justificación

- **Grayscale:** Radiografías son monocromáticas por naturaleza.
- **Resize 256:** Balance resolución/memoria para CNN 2D; suficiente para hallazgos patológicos.
- **CLAHE:** Mejora contraste local sin saturar; clip=2.0 es conservador para evitar artefactos.
- **float32 [0,1]:** Formato estándar para redes neuronales.

### Archivos de salida

```
datasets/nih_chest_xrays/preprocessed/{split}/{filename}.npy   # float32 (256,256)
datasets/nih_chest_xrays/preprocessed/{split}/metadata_{split}.csv
datasets/nih_chest_xrays/preprocessed/stats.json
```

---

## 2. ISIC 2019

**Script:** `src/pipeline/fase0/pre_isic.py` (553 líneas)

### Flujo de ejecución

```
imagen JPEG original → DullRazor hair removal → resize (lado corto=224, aspect ratio preservado) → guardado JPEG q95
```

### Transformaciones

| #  | Transformación | Función/Línea | Parámetros exactos | Input | Output |
|----|---------------|---------------|---------------------|-------|--------|
| 1  | DullRazor hair removal | `remove_hair_dullrazor()` | kernel=`cv2.getStructuringElement(MORPH_RECT, (3,3))`, threshold=`10`, inpaintRadius=`3`, flags=`cv2.INPAINT_TELEA`, dilate iterations=`1` | RGB uint8 (H×W×3) | RGB uint8 (H×W×3) |
| 1a | — Grayscale | `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)` | — | RGB | grayscale |
| 1b | — Morphological closing | `cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)` | kernel 3×3 rect | grayscale | fondo estimado |
| 1c | — Diferencia absoluta | `cv2.absdiff(closed, gray)` | — | — | mapa de pelo |
| 1d | — Umbral binario | `cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)` | threshold=10 | — | máscara binaria |
| 1e | — Dilatación máscara | `cv2.dilate(mask, kernel, iterations=1)` | kernel 3×3, 1 iteración | — | máscara expandida |
| 1f | — Inpainting | `cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)` | radio=3, algoritmo Telea | RGB + máscara | RGB sin pelo |
| 2  | Resize aspect-ratio | `resize_shorter_side()` | `target_size=224`, interpolación `cv2.INTER_LANCZOS4` (o PIL LANCZOS fallback) | RGB (H×W×3) | RGB (new_H×new_W×3) donde min(new_H,new_W)=224 |
| 3  | Guardado JPEG | PIL `Image.save(quality=95)` | `quality=95` | RGB array | JPEG file |

### Constantes

```python
target_size = 224      # lado corto tras resize
quality = 95           # calidad JPEG de salida
# DullRazor:
kernel = (3, 3)        # morphological structuring element
threshold = 10         # diferencia de intensidad para detectar pelo
inpaintRadius = 3      # radio de inpainting Telea
```

### Transformaciones NO aplicadas offline (explícitamente documentado)

- **Color Constancy (Shades of Gray):** Definida en el script (`shades_of_gray()`, power=6, Finlayson & Trezzi 2004), pero se aplica **online con p=0.5** en el training loop para que ConvNeXt-Small vea variabilidad cromática. NO es transformación offline.

### Justificación

- **DullRazor:** Elimina artefactos de pelo que interfieren con el diagnóstico dermatológico (Lee et al., 1997).
- **Resize lado corto 224:** Mantiene aspect ratio; tamaño estándar para ImageNet pretrained models (ConvNeXt-Small).
- **LANCZOS:** Interpolación de alta calidad para downsampling.
- **JPEG q95:** Compresión mínima; preserva detalle visual necesario.

### Archivos de salida

```
datasets/isic_2019/ISIC_2019_Training_Input_preprocessed/{isic_id}_pp_224.jpg
datasets/isic_2019/ISIC_2019_Training_Input_preprocessed/preprocess_report.json
```

---

## 3. LUNA16

**Script:** `src/pipeline/fase0/pre_embeddings.py` (2164 líneas, sección LUNA)

### Flujo de ejecución

```
CT .mhd/.raw → carga SimpleITK → resampleo isotrópico 1mm → máscara pulmonar → clip HU → normalización [0,1] → extracción patch 64³ → zero-centering → guardado .npy
```

### Transformaciones

| #  | Transformación | Función/Línea | Parámetros exactos | Input | Output |
|----|---------------|---------------|---------------------|-------|--------|
| 1  | Carga CT + conversión HU | `sitk.ReadImage(mhd_path)` → `sitk.GetArrayFromImage().astype(np.float32)` | SimpleITK aplica slope/intercept automáticamente; `array[array < -1000] = -1000.0` | .mhd + .raw | float32 3D [Z,Y,X] |
| 2  | Resampleo isotrópico | `scipy_zoom(array, zoom_factors, order=1)` | `zoom_factors = (spacing[2], spacing[1], spacing[0])` → target 1×1×1 mm³, interpolación lineal (order=1) | float32 3D (spacing original) | float32 3D (1mm isotrópico) |
| 3  | Máscara pulmonar | Carga segmentación `seg-lungs-LUNA16/{uid}.mhd`, resampleo con `order=0` (NN), binarización `> 0.5`, fuera-de-pulmón → `-1000.0` | directorio `seg-lungs-LUNA16`, threshold=0.5, nearest-neighbor resample | segmentación + CT | CT enmascarado |
| 4  | Clip HU | `np.clip(array, -1000, 400)` | `HU_LUNG_CLIP = (-1000, 400)` | float32 CT | float32 CT clipped |
| 5  | Normalización min-max | `(array - (-1000)) / (400 - (-1000))` = `(array + 1000) / 1400` | rango [-1000, 400] → [0, 1] | float32 | float32 [0,1] |
| 6  | Extracción patch | Conversión world→voxel, slice 64³ con zero-pad en bordes | `PATCH_SIZE = 64`, `PATCH_HALF = 32`, pad con `constant_values=0.0` | CT 3D completo | float32 (64,64,64) |
| 7  | Zero-centering | `patch -= GLOBAL_MEAN` (aplicado en bulk post-extracción) | `GLOBAL_MEAN = 0.09921630471944809` | float32 [0,1] | float32 centrado |
| 8  | Guardado | `np.save(out_path, patch)` | — | float32 (64,64,64) | `.npy` file |

### Constantes

```python
HU_LUNG_CLIP = (-1000, 400)
PATCH_SIZE = 64
PATCH_HALF = 32
SEG_DIR_NAME = "seg-lungs-LUNA16"
RANDOM_SEED = 42
GLOBAL_MEAN = 0.09921630471944809   # media global para zero-centering
```

### Justificación

- **Isotrópico 1mm:** Estandariza resolución espacial entre CTs con spacings diferentes; crítico para que patches representen la misma región física.
- **Máscara pulmonar:** Elimina tejido extrapulmonar irrelevante (costillas, mediastino); reduce ruido en features.
- **Clip [-1000, 400]:** -1000=aire, 400=hueso denso; rango relevante para patología pulmonar.
- **Normalización [0,1]:** Estándar para redes neuronales.
- **Zero-centering:** Centra la distribución de intensidades; la media global se calcula sobre todos los patches de entrenamiento.
- **Patch 64³:** Balance resolución/memoria para nódulos (típicamente 3-30mm diámetro).

### Archivos de salida

```
datasets/luna_lung_cancer/patches/{split}/candidate_{idx}.npy   # float32 (64,64,64)
```

---

## 4. OsteoArthritis (Knee X-ray)

**Script:** `src/pipeline/fase0/pre_modelo.py` (sección `split_osteoarthritis`)

### Transformaciones offline

OsteoArthritis NO tiene un script de preprocesamiento dedicado. Las transformaciones offline se limitan a:

| #  | Transformación | Función/Línea | Parámetros exactos | Input | Output |
|----|---------------|---------------|---------------------|-------|--------|
| 1  | Organización por clases | `split_osteoarthritis()` en `pre_modelo.py` | 5 clases KL (0, 1, 2, 3, 4) como directorios | imágenes en subdirectorios por KL grade | estructura de splits |
| 2  | Deduplicación por similitud visual | fingerprint hash + comparación | `similarity_threshold=0.12`, `fingerprint_size=16` | todas las imágenes | imágenes únicas (duplicados excluidos) |

### Constantes

```python
similarity_threshold = 0.12   # umbral para considerar dos imágenes como duplicadas
fingerprint_size = 16          # tamaño del fingerprint de imagen para deduplicación
# 5 clases directas: KL grades 0, 1, 2, 3, 4
```

### Justificación

- **Sin preprocesamiento de píxeles:** Las imágenes se usan tal cual; las transformaciones de imagen se aplican online durante entrenamiento.
- **Deduplicación:** El dataset original contiene duplicados; la similitud visual evita data leakage entre splits.

### Archivos de salida

```
datasets/osteoarthritis/splits/{split}/{KL_grade}/*.png
```

---

## 5. Pancreas (PANORAMA/Zenodo)

**Script:** `src/pipeline/fase0/pre_embeddings.py` (sección Pancreas, líneas ~897-1600)

### Flujo de ejecución

```
NIfTI .nii.gz → carga SimpleITK → resampleo isotrópico 1mm (B-spline order=3) → clip HU → normalización [0,1] → resize 64³ → centroide desde máscara (label==3) → crop 48³ → guardado .npy
```

### Transformaciones

| #  | Transformación | Función/Línea | Parámetros exactos | Input | Output |
|----|---------------|---------------|---------------------|-------|--------|
| 1  | Carga NIfTI | `sitk.ReadImage(nii_path)` → `sitk.GetArrayFromImage().astype(np.float32)` | — | `*_0000.nii.gz` | float32 3D [D,H,W] |
| 2  | Resampleo isotrópico | `scipy_zoom(array, zoom_factors, order=3)` | B-spline order=3 (cúbico), target 1×1×1 mm³ | float32 3D (spacing original) | float32 3D (1mm isotrópico) |
| 3  | Clip HU | `np.clip(array, -150, 250)` | `PANCREAS_HU_MIN = -150`, `PANCREAS_HU_MAX = 250` | float32 CT | float32 CT clipped |
| 4  | Normalización min-max | `(array - (-150)) / (250 - (-150))` = `(array + 150) / 400` | `PANCREAS_HU_RANGE = 400` | float32 | float32 [0,1] |
| 5  | Resize trilineal a 64³ | `scipy_zoom(array, (target/D, target/H, target/W), order=1)` | `PANCREAS_OUTPUT_SIZE = 64`, order=1 (trilineal) | float32 3D isotrópico | float32 (64,64,64) |
| 6  | Centroide pancreático | Desde máscara panorama_labels (label==3): calcula centroide en espacio isotrópico, escala a 64³. Fallback: centro geométrico (32,32,32) | `label==3` en panorama_labels, clamp a [24, 40] | máscara NIfTI | coordenadas (cz, cy, cx) |
| 7  | Crop 48³ centrado | `array[cz-24:cz+24, cy-24:cy+24, cx-24:cx+24]` con zero-pad si necesario | `PANCREAS_CROP_SIZE = 48`, `half = 24`, clamp centroide a [24, 40] | float32 (64,64,64) | float32 (48,48,48) |
| 8  | Guardado | `np.save(out_path, crop)` | — | float32 (48,48,48) | `.npy` file |

### Constantes

```python
PANCREAS_HU_MIN = -150
PANCREAS_HU_MAX = 250
PANCREAS_HU_RANGE = 400        # = 250 - (-150)
PANCREAS_OUTPUT_SIZE = 64      # volumen completo tras resize
PANCREAS_CROP_SIZE = 48        # crop final centrado en páncreas
# Centroide:
_half_crop = 24                # = CROP_SIZE // 2
_clamp_lo = 24                 # mínimo centroide para que crop quepa
_clamp_hi = 40                 # = OUTPUT_SIZE - _half_crop
# Fallback centroide: (32, 32, 32) = centro geométrico
```

### Estrategia de centroide e invalidación de caché

El script persiste la estrategia de centroide en `centroid_strategy.txt`:
- `"mask_centroid_real"` — si se encontraron máscaras panorama_labels
- `"geometric_center_fallback"` — si no hay máscaras disponibles

Si la estrategia cambia entre ejecuciones (e.g., se añaden máscaras), se invalida el caché completo y se reprocesan todos los volúmenes.

### Justificación

- **B-spline order=3:** Interpolación cúbica para CT; mejor preservación de intensidades que lineal.
- **Clip [-150, 250]:** Rango HU relevante para tejido blando abdominal (páncreas ~30-45 HU, hígado ~40-60 HU); excluye aire y hueso.
- **Resize 64³ → crop 48³:** Estandariza volumen; el crop centrado en páncreas elimina contexto irrelevante periférico.
- **Centroide desde máscara label==3:** El label 3 en PANORAMA corresponde al páncreas; centra el crop en la región de interés real.

### Archivos de salida

```
datasets/pancreas/zenodo_13715870/preprocessed/{case_id}.npy   # float32 (48,48,48)
datasets/pancreas/zenodo_13715870/preprocessed/pancreas_preprocess_report.json
datasets/pancreas/zenodo_13715870/preprocessed/centroid_strategy.txt
```

---

## 6. Splits (todos los datasets)

**Script:** `src/pipeline/fase0/pre_modelo.py` (1233 líneas)

### Configuración global

```python
SEED = 42
# Proporción: 80% train / 10% val / 10% test
```

### Splits por dataset

| Dataset | Unidad de split | Estrategia | Estratificación | Particularidades |
|---------|----------------|------------|-----------------|-----------------|
| NIH ChestX-ray14 | `patient_id` (extraído de filename: `00012345_001.png` → `00012345`) | `StratifiedShuffleSplit` | Por etiqueta más rara | Si test oficial > 12%, reduce a 10% y mueve sobrantes al pool train_val |
| ISIC 2019 | `lesion_id` | `StratifiedShuffleSplit` | Por clase diagnóstica | Agrupa por lesion_id para evitar data leakage (múltiples imágenes de misma lesión) |
| OsteoArthritis | Imagen individual | `train_test_split` + deduplicación visual | Por KL grade | Deduplicación por fingerprint (similarity_threshold=0.12, fingerprint_size=16) |
| LUNA16 | `seriesuid` | `train_test_split` | Por label (candidato positivo/negativo) | Split a nivel de CT completo para evitar leakage entre patches del mismo paciente |
| Pancreas | `patient_id` | 10% test holdout + 5-fold `GroupKFold` en restante | Agrupado por patient_id | GroupKFold garantiza que todas las imágenes del mismo paciente estén en el mismo fold |

### CAE (Aggregated Splits)

`pre_modelo.py` también genera splits agregados para el **Convolutional Autoencoder (CAE)**:
- Combina datos de todos los datasets 2D para entrenamiento del autoencoder.

### Archivos de salida

```
datasets/{dataset}/splits/{split}_list.txt   # o similar por dataset
```

---

## 7. LUNA16 — Augmentación Offline

**Script:** `src/pipeline/fase0/create_augmented_train.py` (379 líneas)

> **NOTA:** Esta es augmentación **offline** (se ejecuta una vez y genera archivos en disco), NO augmentación on-the-fly.

### Objetivo

Reducir desbalance de clases de ~10:1 (negativo:positivo) a **2:1** mediante augmentación de patches positivos.

### Pipeline de augmentación por patch

| #  | Augmentación | Parámetros | Probabilidad |
|----|-------------|------------|--------------|
| 1  | Flip por eje | 3 ejes independientes | p=0.5 cada eje |
| 2  | Rotación 3D | ±15° en 3 planos `(1,2), (0,2), (0,1)`, `scipy_rotate(order=1, mode='nearest')` | Siempre (si \|angle\| > 0.5°) |
| 3  | Zoom | factor ∈ [0.80, 1.20], `scipy_zoom(order=1)`, crop/pad a 64³ | Siempre |
| 4  | Traslación | ±4 voxels por eje, zero-fill | Siempre |
| 5  | Deformación elástica | sigma ∈ [1.0, 3.0], alpha ∈ [0.0, 5.0], `map_coordinates(order=1, mode='nearest')` | p=0.5 |
| 6  | Ruido gaussiano | sigma ∈ [0, 25/1400] | p=0.5 |
| 7  | Brillo/contraste | scale ∈ [0.9, 1.1], offset ∈ [-20/1400, 20/1400] | Siempre |
| 8  | Blur gaussiano | sigma ∈ [0.1, 0.5] | p=0.5 |
| 9  | Clamp final | `np.clip(volume, -GLOBAL_MEAN, 1.0 - GLOBAL_MEAN)` | Siempre |

### Constantes

```python
TARGET_RATIO = 2          # ratio neg:pos deseado
GLOBAL_MEAN = 0.09921630471944809
```

### Archivos de salida

```
datasets/luna_lung_cancer/patches/train_aug/candidate_{idx}.npy
datasets/luna_lung_cancer/patches/train_aug/aug_{orig_idx}_{copy}.npy
datasets/luna_lung_cancer/patches/train_aug_manifest.csv
datasets/luna_lung_cancer/patches/train_aug_report.json
```

---

## 8. LUNA16 — Auditoría de Patches

**Script:** `src/pipeline/fase0/audit_dataset.py` (503 líneas)

### Verificaciones

| Check | Descripción | Criterio |
|-------|------------|----------|
| Shape | Todas las dimensiones correctas | `== (64, 64, 64)` |
| Dtype | Tipo de dato correcto | `float32` |
| Zero-centering | Media cercana a cero | `abs(mean) < threshold` |
| Balance | Ratio de clases aceptable | Reporta distribución pos/neg |
| Duplicados | Sin patches duplicados | Hash de contenido |
| NaN/Inf | Sin valores no finitos | `np.isfinite()` |

---

## 9. Tabla Comparativa

| Característica | NIH ChestX-ray14 | ISIC 2019 | LUNA16 | OsteoArthritis | Pancreas |
|---------------|-------------------|-----------|--------|----------------|----------|
| **Modalidad** | Radiografía 2D | Dermatoscopia 2D | CT 3D | Radiografía 2D | CT 3D |
| **Script principal** | `pre_chestxray14.py` | `pre_isic.py` | `pre_embeddings.py` | (solo splits) | `pre_embeddings.py` |
| **Formato entrada** | PNG | JPEG | .mhd/.raw | PNG | NIfTI .nii.gz |
| **Formato salida** | .npy float32 | JPEG q95 | .npy float32 | PNG (sin cambios) | .npy float32 |
| **Dimensiones salida** | (256, 256) | (≥224, ≥224) variable AR | (64, 64, 64) | Original | (48, 48, 48) |
| **Espacio color** | Grayscale | RGB | — | Original | — |
| **Normalización** | [0, 1] | uint8 [0, 255] | [0, 1] + zero-centering | — | [0, 1] |
| **Clip de intensidad** | — | — | HU [-1000, 400] | — | HU [-150, 250] |
| **Resampleo espacial** | Resize 256² | Resize lado corto 224 | Isotrópico 1mm³ | — | Isotrópico 1mm³ + resize 64³ |
| **Interpolación** | INTER_LINEAR | LANCZOS4 | order=1 (lineal) | — | order=3 (B-spline) + order=1 (resize) |
| **Mejora contraste** | CLAHE (2.0, 8×8) | — | — | — | — |
| **Eliminación artefactos** | — | DullRazor hair removal | Máscara pulmonar | Deduplicación visual | — |
| **Región de interés** | Imagen completa | Imagen completa | Patch 64³ en candidato | Imagen completa | Crop 48³ en centroide páncreas |
| **Split unidad** | patient_id | lesion_id | seriesuid | imagen + dedup | patient_id (GroupKFold) |
| **Augmentación offline** | No | No | Sí (train_aug 2:1) | No | No |
| **# transformaciones pixel** | 3 (resize, CLAHE, float) | 2 (DullRazor, resize) | 6 (resample, mask, clip, norm, patch, zero-center) | 0 | 6 (resample, clip, norm, resize, centroid, crop) |

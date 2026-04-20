# Transformaciones Offline — Agrupación por Familia Funcional

> Proyecto MoE Médico — Fase 0 (Preprocesamiento Offline)
>
> Generado: 2026-04-19 | Scripts fuente: `src/pipeline/fase0/`

---

## Tabla de Contenidos

1. [Carga y Decodificación](#1-carga-y-decodificación)
2. [Resampleo Espacial](#2-resampleo-espacial)
3. [Enmascaramiento y Segmentación](#3-enmascaramiento-y-segmentación)
4. [Clip y Windowing de Intensidad](#4-clip-y-windowing-de-intensidad)
5. [Normalización de Intensidad](#5-normalización-de-intensidad)
6. [Mejora de Contraste](#6-mejora-de-contraste)
7. [Eliminación de Artefactos](#7-eliminación-de-artefactos)
8. [Extracción de Regiones de Interés](#8-extracción-de-regiones-de-interés)
9. [Serialización y Formato de Salida](#9-serialización-y-formato-de-salida)
10. [Splitting y Particionado](#10-splitting-y-particionado)
11. [Augmentación Offline](#11-augmentación-offline)
12. [Auditoría y Validación](#12-auditoría-y-validación)
13. [Resumen Visual por Familia](#13-resumen-visual-por-familia)

---

## 1. Carga y Decodificación

**Propósito:** Leer datos crudos desde disco al formato de array numérico en memoria.

| Dataset | Librería | Formato entrada | Función | Salida en memoria |
|---------|----------|-----------------|---------|-------------------|
| NIH | OpenCV | PNG | `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` | uint8 2D |
| ISIC | PIL/OpenCV | JPEG | `Image.open()` → `np.array()` | uint8 (H,W,3) RGB |
| LUNA16 | SimpleITK | .mhd + .raw (MetaImage) | `sitk.ReadImage()` → `sitk.GetArrayFromImage().astype(np.float32)` | float32 3D [Z,Y,X] |
| OsteoArthritis | PIL | PNG | `Image.open()` (solo para fingerprint) | uint8 2D/3D |
| Pancreas | SimpleITK | NIfTI .nii.gz | `sitk.ReadImage()` → `sitk.GetArrayFromImage().astype(np.float32)` | float32 3D [D,H,W] |

**Observación:** Los datasets 3D (LUNA, Pancreas) comparten la misma librería (SimpleITK) y el mismo patrón de carga, pero difieren en formato de archivo.

---

## 2. Resampleo Espacial

**Propósito:** Estandarizar la resolución espacial y/o dimensiones de las imágenes.

| Dataset | Tipo | Método | Interpolación | Target | Preserva AR |
|---------|------|--------|---------------|--------|-------------|
| NIH | Resize 2D | `cv2.resize()` | `INTER_LINEAR` | 256×256 px | No (fuerza cuadrado) |
| ISIC | Resize 2D | `cv2.resize()` / PIL | `INTER_LANCZOS4` / `LANCZOS` | lado corto = 224 px | **Sí** |
| LUNA16 | Resampleo 3D isotrópico | `scipy.ndimage.zoom()` | order=1 (lineal) | 1×1×1 mm³ | N/A (voxel) |
| Pancreas | Resampleo 3D isotrópico + resize | `scipy.ndimage.zoom()` | order=3 (B-spline) resampleo, order=1 (lineal) resize | 1×1×1 mm³ → 64³ voxels | N/A (voxel) |
| OsteoArthritis | — | Ninguno | — | — | — |

**Patrón 2D vs 3D:**
- **2D (NIH, ISIC):** Resize directo en píxeles. NIH fuerza cuadrado; ISIC preserva aspect ratio.
- **3D (LUNA, Pancreas):** Primero resampleo isotrópico a 1mm³ (estandarización física), luego dimensionado si aplica. Pancreas usa B-spline (order=3) por mejor fidelidad en tejido blando; LUNA usa lineal (order=1) por mayor velocidad en volúmenes grandes.

---

## 3. Enmascaramiento y Segmentación

**Propósito:** Eliminar regiones anatómicas irrelevantes para la tarea.

| Dataset | Método | Fuente de máscara | Aplicación |
|---------|--------|-------------------|------------|
| LUNA16 | Máscara de segmentación pulmonar | `seg-lungs-LUNA16/{uid}.mhd` | Voxels fuera del pulmón → -1000 HU (aire) |
| Pancreas | Centroide desde máscara de órgano | `panorama_labels/{case_id}.nii.gz`, `label==3` | No enmascara voxels; usa centroide para guiar el crop |

**Datasets sin enmascaramiento:** NIH, ISIC, OsteoArthritis — la imagen completa es relevante.

---

## 4. Clip y Windowing de Intensidad

**Propósito:** Restringir el rango dinámico a valores clínicamente relevantes (solo datasets CT).

| Dataset | Rango HU | Justificación clínica |
|---------|----------|----------------------|
| LUNA16 | [-1000, 400] | Aire (-1000) a hueso denso (400); cubre todo el parénquima pulmonar, nódulos, vasos |
| Pancreas | [-150, 250] | Tejido blando abdominal; páncreas ~30-45 HU, hígado ~40-60 HU, excluye aire y hueso |

**Implementación idéntica:**
```python
np.clip(array, hu_min, hu_max, out=array)
```

**Datasets sin clip:** NIH (radiografía, no HU), ISIC (fotografía), OsteoArthritis (radiografía, no HU).

---

## 5. Normalización de Intensidad

**Propósito:** Escalar valores al rango esperado por las redes neuronales.

| Dataset | Método | Fórmula | Rango salida |
|---------|--------|---------|-------------|
| NIH | Division por 255 | `img.astype(np.float32) / 255.0` | [0, 1] |
| LUNA16 | Min-max sobre rango HU + zero-centering | `(arr - hu_min) / (hu_max - hu_min)` luego `arr -= 0.09921630471944809` | [0,1] centrado en ~0 |
| Pancreas | Min-max sobre rango HU | `(arr - (-150)) / 400` | [0, 1] |
| ISIC | Sin normalización offline | Guardado como JPEG uint8 | [0, 255] uint8 |
| OsteoArthritis | Sin normalización offline | — | Original |

**Nota sobre zero-centering LUNA:** La media global `0.09921630471944809` se calcula sobre todos los patches de entrenamiento. Se aplica en bulk después de la extracción de patches.

---

## 6. Mejora de Contraste

**Propósito:** Aumentar la visibilidad de estructuras diagnósticas.

| Dataset | Método | Parámetros |
|---------|--------|------------|
| NIH | CLAHE (Contrast Limited Adaptive Histogram Equalization) | `clipLimit=2.0`, `tileGridSize=(8, 8)` |

**Solo NIH** usa mejora de contraste offline. Se aplica después del resize (sobre imagen 256×256).

**ISIC** tiene `shades_of_gray()` (Color Constancy) definida en el script pero se aplica **online** con p=0.5.

---

## 7. Eliminación de Artefactos

**Propósito:** Remover elementos que interfieren con el diagnóstico o el aprendizaje.

| Dataset | Artefacto | Método | Parámetros |
|---------|-----------|--------|------------|
| ISIC | Pelo (hair) | DullRazor (Lee et al., 1997) | kernel 3×3, threshold=10, inpaintRadius=3, INPAINT_TELEA, dilate 1 iter |
| OsteoArthritis | Duplicados visuales | Fingerprint hash + similitud | similarity_threshold=0.12, fingerprint_size=16 |

**DullRazor pipeline detallado:**
1. Grayscale → 2. Morphological closing (3×3) → 3. Diferencia absoluta → 4. Umbral binario (>10) → 5. Dilatación (1 iter) → 6. Inpainting Telea (radio=3)

---

## 8. Extracción de Regiones de Interés

**Propósito:** Recortar la región relevante del volumen completo.

| Dataset | Método | Parámetros | Centro |
|---------|--------|------------|--------|
| LUNA16 | Patch 3D centrado en candidato | 64×64×64 voxels a 1mm, zero-pad en bordes | Coordenadas world del candidato (candidates_V2.csv) |
| Pancreas | Crop 3D centrado en centroide | 48×48×48 voxels del volumen 64³ | Centroide de máscara label==3, fallback (32,32,32), clamp a [24,40] |

**Datasets sin extracción de ROI:** NIH, ISIC, OsteoArthritis — se usa la imagen completa.

---

## 9. Serialización y Formato de Salida

**Propósito:** Persistir el resultado preprocesado en disco en formato eficiente.

| Dataset | Formato | Tipo dato | Shape | Metadatos asociados |
|---------|---------|-----------|-------|-------------------|
| NIH | `.npy` (NumPy binary) | float32 | (256, 256) | `metadata_{split}.csv`, `stats.json` |
| ISIC | JPEG (quality=95) | uint8 | (≥224, ≥224, 3) variable | `preprocess_report.json` |
| LUNA16 | `.npy` | float32 | (64, 64, 64) | — |
| OsteoArthritis | PNG (sin cambios) | uint8 | Original | — |
| Pancreas | `.npy` | float32 | (48, 48, 48) | `pancreas_preprocess_report.json`, `centroid_strategy.txt` |

**Patrón:** Los datasets que requieren normalización numérica (NIH, LUNA, Pancreas) usan `.npy` float32. Los que se mantienen como imágenes naturales (ISIC, OA) conservan formato imagen.

---

## 10. Splitting y Particionado

**Script:** `src/pipeline/fase0/pre_modelo.py`

**Propósito:** Dividir datos en train/val/test previniendo data leakage.

### Estrategias anti-leakage por dataset

| Dataset | Unidad de agrupación | Riesgo de leakage | Mitigación |
|---------|---------------------|-------------------|------------|
| NIH | patient_id | Múltiples radiografías del mismo paciente | Split por patient_id, no por imagen |
| ISIC | lesion_id | Múltiples fotos de la misma lesión | Split por lesion_id |
| LUNA16 | seriesuid | Múltiples patches del mismo CT | Split por seriesuid (CT completo) |
| OsteoArthritis | imagen + dedup | Imágenes visualmente duplicadas | Fingerprint deduplicación (threshold=0.12) |
| Pancreas | patient_id | Múltiples volúmenes del mismo paciente | GroupKFold por patient_id |

### Proporciones y algoritmos

| Dataset | Proporción | Algoritmo | Estratificación |
|---------|-----------|-----------|-----------------|
| NIH | 80/10/10 | `StratifiedShuffleSplit` | Etiqueta más rara |
| ISIC | 80/10/10 | `StratifiedShuffleSplit` | Clase diagnóstica |
| LUNA16 | 80/10/10 | `train_test_split` | Label pos/neg |
| OsteoArthritis | 80/10/10 | `train_test_split` | KL grade |
| Pancreas | 90/— (10% test) + 5-fold CV | `GroupKFold` | — (agrupado) |

---

## 11. Augmentación Offline

**Script:** `src/pipeline/fase0/create_augmented_train.py`

**Propósito:** Balancear clases en LUNA16 generando copias augmentadas de patches positivos.

**Solo se aplica a:** LUNA16

| Familia | Augmentaciones | Parámetros |
|---------|---------------|------------|
| Geométricas | Flip (3 ejes, p=0.5), Rotación 3D (±15°, 3 planos), Zoom ([0.8, 1.2]), Traslación (±4 vox) | order=1, mode='nearest' |
| Deformación | Elástica (sigma∈[1,3], alpha∈[0,5]) | p=0.5, map_coordinates order=1 |
| Intensidad | Ruido gaussiano (σ∈[0, 25/1400]), Brillo/contraste (scale∈[0.9,1.1], offset∈[-20/1400, 20/1400]) | Ruido p=0.5, B/C siempre |
| Filtrado | Blur gaussiano (σ∈[0.1, 0.5]) | p=0.5 |
| Post-proceso | Clamp a [-GLOBAL_MEAN, 1-GLOBAL_MEAN] | Siempre |

**Ratio objetivo:** De ~10:1 (neg:pos) a **2:1**. Negativos se copian sin modificar; positivos reciben augmentaciones estocásticas.

---

## 12. Auditoría y Validación

**Scripts:** `src/pipeline/fase0/audit_dataset.py` (LUNA), validación inline en `pre_embeddings.py` (Pancreas)

**Propósito:** Verificar integridad y correctitud del preprocesamiento.

| Dataset | Checks | Script |
|---------|--------|--------|
| LUNA16 | Shape (64³), dtype (float32), zero-centering (media~0), balance pos/neg, duplicados por hash, NaN/Inf | `audit_dataset.py` |
| Pancreas | Shape (48³), dtype (float32), rango [0,1], no-trivial (no todo ceros), muestreo aleatorio de n=100 | `_validate_pancreas_sample()` en `pre_embeddings.py` |

---

## 13. Resumen Visual por Familia

```
                    NIH    ISIC   LUNA   OA    Pancreas
                    ─────  ─────  ─────  ────  ────────
Carga/Decode        ✓      ✓      ✓      ✓     ✓
Resampleo espacial  ✓      ✓      ✓      ·     ✓
Enmascaramiento     ·      ·      ✓      ·     (centroide)
Clip HU             ·      ·      ✓      ·     ✓
Normalización       ✓      ·      ✓      ·     ✓
Mejora contraste    ✓      ·      ·      ·     ·
Elim. artefactos    ·      ✓      ·      ✓     ·
Extracción ROI      ·      ·      ✓      ·     ✓
Serialización       .npy   JPEG   .npy   PNG   .npy
Splitting           ✓      ✓      ✓      ✓     ✓
Aug. offline        ·      ·      ✓      ·     ·
Auditoría           ·      ·      ✓      ·     ✓

✓ = aplica    · = no aplica
```
